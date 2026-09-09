"""LeRobot v3 export of an episode-directory dataset (10-frames §11.8; 04-runtime §10.6).

``exports/lerobot_v3/`` is rebuilt as a whole from ``episodes/<id>/`` — a remux,
never a re-encode. The algorithm mirrors lerobot's own
``convert_dataset_v21_to_v30`` (per-episode files -> v3 shards):

1. episodes with ``export_ok`` sorted by id (capture order) -> dense
   ``episode_index``; ``meta/apollo/episode_map.json``;
2. videos per camera: consecutive episodes with an identical encoder identity,
   accumulated while ``size < video_file_mb``, concatenated by the ffconcat
   demuxer + packet remux (lerobot ``concatenate_video_files`` semantics, ~50
   lines over ``av`` so the job stays torch-free); ``from_timestamp`` /
   ``to_timestamp`` = cumulative ``length / fps``; a new encoder identity always
   starts a new file;
3. data: ``frames.parquet`` files stacked in the same order into
   ``data/chunk-CCC/file-FFF.parquet`` (one row group per episode, snappy) with
   ``episode_index`` / ``index`` (global row position) / ``task_index`` added and
   the per-frame ``task`` string DROPPED (a lerobot data parquet carries exactly
   the info.json features + the five bookkeeping columns);
4. meta: ``meta/episodes/chunk-000/file-000.parquet`` (positional columns +
   flattened ``stats/*``), ``meta/stats.json`` (:func:`stats.aggregate_stats`),
   ``meta/tasks.parquet`` (row order == task_index, pandas index ``task``),
   ``meta/info.json`` (``codebase_version v3.0``, ``video.*`` probed from the
   first concatenated file);
5. sidecars into ``meta/apollo/`` (``episode_index`` rewritten);
6. validation by opening ``LeRobotDataset(repo_id, root=<export>)`` — the ONE
   place this module may import ``lerobot.datasets`` (i.e. torch); it runs on
   the export job's thread / the CLI, never on a REST handler.

Torch-free decision (phase-13): the job re-implements the ~50-line concat and
the ~60-line ``aggregate_stats`` instead of importing lerobot's, because any
``from lerobot.datasets... import`` executes ``lerobot/datasets/__init__.py``
and pulls torch into the runtime process.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apollo_mavis_v2_core.protocol import DatasetExportTelemetry

from . import stats as _stats
from .manifest import (
    EPISODE_JSON,
    EXPORTS_DIR,
    FRAMES_PARQUET,
    IMAGE_PREFIX,
    MANIFEST_LOCK,
    SCENES_DIR,
    SESSIONS_DIR,
    episode_dirs,
    read_json,
    read_manifest,
    utc_now_iso,
    write_manifest,
)

logger = logging.getLogger(__name__)

CODEBASE_VERSION = "v3.0"
EXPORT_FORMAT = "lerobot_v3"
DEFAULT_EXPORT_SUBDIR = f"{EXPORTS_DIR}/{EXPORT_FORMAT}"
CHUNK_SIZE = 1000  # lerobot DEFAULT_CHUNK_SIZE
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
# lerobot DEFAULT_FEATURES (lerobot.utils.constants), verbatim.
DEFAULT_FEATURES: dict[str, dict] = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}
# The encoder fields lerobot's get_video_info records as ``video.<field>`` next to
# the stream-derived values (VideoEncoderConfig minus vcodec).
_ENCODER_INFO_FIELDS = (
    "pix_fmt", "g", "crf", "preset", "fast_decode", "video_backend", "extra_options",
)
_PIX_FMT_CHANNELS = {
    "yuv420p": 3, "yuv444p": 3, "rgb24": 3, "gray": 1, "gray12le": 1, "yuv420p10le": 3,
}
_IDENTITY_KEYS = ("codec", "encoder", "pix_fmt", "g", "crf", "extra_options", "width", "height")


class ExportError(Exception):
    """The export cannot run (no manifest, nothing exportable, a broken episode)."""


@dataclass
class ExportProgress:
    """Thread-safe progress the telemetry builder reads (``telemetry.datasets.export``)."""

    repo_id: str
    format: str = EXPORT_FORMAT
    phase: str = "scanning"
    done: int = 0
    total: int = 0
    detail: str = ""
    # monotonic time each phase was first entered (tests assert videos+data+meta < 2 s)
    phase_started: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self.phase_started.setdefault(self.phase, time.monotonic())

    def set(self, phase: str | None = None, done: int | None = None, total: int | None = None,
            detail: str | None = None) -> None:
        with self._lock:
            if phase is not None:
                self.phase = phase
                self.phase_started.setdefault(phase, time.monotonic())
            if done is not None:
                self.done = done
            if total is not None:
                self.total = total
            if detail is not None:
                self.detail = detail

    def telemetry(self) -> DatasetExportTelemetry:
        with self._lock:
            return DatasetExportTelemetry(
                repo_id=self.repo_id, format=self.format, phase=self.phase,  # type: ignore[arg-type]
                done=self.done, total=self.total, detail=self.detail,
            )


@dataclass(frozen=True)
class ExportResult:
    path: Path
    episodes: int
    frames: int
    skipped: list[str]  # episode ids excluded (export_ok false)
    episode_ids: list[str] = field(default_factory=list)  # exported, in episode_index order
    incomplete: dict[str, str] = field(default_factory=dict)  # id -> why the directory was skipped


@dataclass
class _Episode:
    episode_id: str
    directory: Path
    meta: dict[str, Any]
    index: int = -1

    @property
    def length(self) -> int:
        return int(self.meta.get("length", 0) or 0)


def _update_indices(chunk: int, file: int) -> tuple[int, int]:
    file += 1
    if file >= CHUNK_SIZE:
        chunk, file = chunk + 1, 0
    return chunk, file


def _size_mb(path: Path) -> float:
    return path.stat().st_size / (1024**2)


def concatenate_videos(inputs: list[Path], output: Path) -> None:
    """ffconcat demuxer + packet remux, no re-encode (lerobot
    ``concatenate_video_files`` semantics); the output carries ``movflags
    faststart``. Inputs must share codec / pix_fmt / size / fps — the caller's
    encoder-identity grouping guarantees it."""
    import av

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".ffconcat", delete=False) as f:
        f.write("ffconcat version 1.0\n")
        for p in inputs:
            f.write(f"file '{Path(p).resolve()}'\n")
        concat_path = f.name
    tmp_out = output.with_name(output.name + ".tmp")
    try:
        inp = av.open(concat_path, mode="r", format="concat", options={"safe": "0"})
        out = av.open(str(tmp_out), mode="w", format="mp4", options={"movflags": "faststart"})
        stream_map = {}
        for s in inp.streams:
            if s.type in ("video", "audio", "subtitle"):
                stream_map[s.index] = out.add_stream_from_template(template=s, opaque=True)
                stream_map[s.index].time_base = s.time_base
        for packet in inp.demux():
            if packet.stream.index not in stream_map or packet.dts is None:
                continue
            packet.stream = stream_map[packet.stream.index]
            out.mux(packet)
        inp.close()
        out.close()
        os.replace(tmp_out, output)
    finally:
        Path(concat_path).unlink(missing_ok=True)
        tmp_out.unlink(missing_ok=True)


def probe_video_info(path: Path) -> dict[str, Any]:
    """The stream-derived half of lerobot's ``get_video_info`` (``video.*``)."""
    import av

    with av.open(str(path), "r") as container:
        stream = container.streams.video[0]
        info = {
            "video.height": int(stream.height),
            "video.width": int(stream.width),
            "video.codec": stream.codec.canonical_name,
            "video.pix_fmt": stream.pix_fmt,
            "video.fps": int(stream.base_rate),
            "video.channels": _PIX_FMT_CHANNELS.get(str(stream.pix_fmt), 3),
        }
    info["has_audio"] = False
    return info


def _scan(
    root: Path, progress: ExportProgress
) -> tuple[dict[str, Any], list[_Episode], list[str], dict[str, str]]:
    """Exportable episodes in capture order. ``export_ok: false`` episodes are skipped
    by design; a directory that is NOT COMPLETE (unreadable episode.json, missing
    frames.parquet, a listed video absent) is skipped too and reported by id — one
    broken episode must not fail the whole export."""
    manifest = read_manifest(root)
    if manifest is None:
        raise ExportError(f"{root} has no manifest.json (not an episode-directory dataset)")
    episodes: list[_Episode] = []
    skipped: list[str] = []
    incomplete: dict[str, str] = {}
    for d in episode_dirs(root):
        meta = read_json(d / EPISODE_JSON)
        if meta is None:
            incomplete[d.name] = f"unreadable {EPISODE_JSON}"
            continue
        if not meta.get("export_ok", True):
            skipped.append(d.name)
            continue
        if not (d / FRAMES_PARQUET).exists():
            incomplete[d.name] = f"missing {FRAMES_PARQUET}"
            continue
        missing_video = [
            cam for cam, block in (meta.get("video") or {}).items()
            if not (d / str((block or {}).get("file") or f"video/{cam}.mp4")).exists()
        ]
        if missing_video:
            incomplete[d.name] = f"missing video for {', '.join(sorted(missing_video))}"
            continue
        episodes.append(_Episode(d.name, d, meta))
    for i, ep in enumerate(episodes):
        ep.index = i
    for eid, why in incomplete.items():
        logger.warning("export: episode %s skipped (%s)", eid, why)
    detail = f"{len(episodes)} episodes"
    if skipped:
        detail += f", {len(skipped)} skipped (export_ok false)"
    if incomplete:
        detail += f", {len(incomplete)} incomplete: " + "; ".join(
            f"{k} ({v})" for k, v in incomplete.items()
        )
    progress.set(total=len(episodes), done=len(episodes), detail=detail)
    return manifest, episodes, skipped, incomplete


def exportable_ids(root: Path) -> set[str]:
    """Ids the export WOULD take right now (export_ok, complete) — the staleness reference."""
    out: set[str] = set()
    for d in episode_dirs(root):
        meta = read_json(d / EPISODE_JSON)
        if meta is None or not meta.get("export_ok", True) or not (d / FRAMES_PARQUET).exists():
            continue
        videos_ok = all(
            (d / str((block or {}).get("file") or f"video/{cam}.mp4")).exists()
            for cam, block in (meta.get("video") or {}).items()
        )
        if videos_ok:
            out.add(d.name)
    return out


def sweep_export_leftovers(out: Path) -> list[Path]:
    """Remove ``.<name>.tmp-*`` / ``.<name>.old-*`` siblings a crashed job left behind."""
    removed: list[Path] = []
    parent = out.parent
    if not parent.is_dir():
        return removed
    for p in parent.iterdir():
        leftover = p.name.startswith(f".{out.name}.tmp-") or p.name.startswith(f".{out.name}.old-")
        if p.is_dir() and leftover:
            shutil.rmtree(p, ignore_errors=True)
            removed.append(p)
            logger.warning("swept export leftover %s", p)
    return removed


def _identity(ep: _Episode, cam: str) -> tuple:
    block = (ep.meta.get("video") or {}).get(cam) or {}
    return tuple(json.dumps(block.get(k), sort_keys=True) for k in _IDENTITY_KEYS)


def _export_videos(
    staging: Path, episodes: list[_Episode], video_keys: list[str], fps: int,
    video_file_mb: float, progress: ExportProgress,
) -> dict[int, dict[str, Any]]:
    """Per-episode ``videos/<key>/...`` columns; a new file at the size cap or at a
    change of encoder identity."""
    per_ep: dict[int, dict[str, Any]] = {ep.index: {} for ep in episodes}
    total = len(episodes) * len(video_keys)
    done = 0
    progress.set(phase="videos", done=0, total=total, detail="")
    for key in video_keys:
        cam = key[len(IMAGE_PREFIX):]
        for ep in episodes:
            block = (ep.meta.get("video") or {}).get(cam)
            if not block or not (ep.directory / block["file"]).exists():
                raise ExportError(f"episode {ep.episode_id} has no video for camera {cam!r}")
        chunk, file = 0, 0
        group: list[_Episode] = []
        group_ident: tuple | None = None
        size_mb = 0.0
        for ep in episodes:
            ident = _identity(ep, cam)
            ep_mb = _size_mb(ep.directory / ep.meta["video"][cam]["file"])
            if group and (ident != group_ident or size_mb + ep_mb >= video_file_mb):
                _flush_video_group(staging, key, cam, group, chunk, file, fps, per_ep, progress)
                done += len(group)
                progress.set(done=done)
                chunk, file = _update_indices(chunk, file)
                group, size_mb = [], 0.0
            group.append(ep)
            group_ident = ident
            size_mb += ep_mb
        if group:
            _flush_video_group(staging, key, cam, group, chunk, file, fps, per_ep, progress)
            done += len(group)
            progress.set(done=done)
    return per_ep


def _flush_video_group(
    staging: Path, key: str, cam: str, group: list[_Episode], chunk: int, file: int, fps: int,
    per_ep: dict[int, dict[str, Any]], progress: ExportProgress,
) -> None:
    out = staging / VIDEO_PATH.format(video_key=key, chunk_index=chunk, file_index=file)
    progress.set(detail=str(out.relative_to(staging)))
    concatenate_videos([ep.directory / ep.meta["video"][cam]["file"] for ep in group], out)
    t = 0.0
    for ep in group:
        dur = ep.length / fps
        per_ep[ep.index].update(
            {
                f"videos/{key}/chunk_index": chunk,
                f"videos/{key}/file_index": file,
                f"videos/{key}/from_timestamp": t,
                f"videos/{key}/to_timestamp": t + dur,
            }
        )
        t += dur


def _export_data(
    staging: Path, episodes: list[_Episode], features: dict[str, dict], tasks: list[str],
    data_file_mb: float, progress: ExportProgress,
) -> dict[int, dict[str, Any]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    task_index = {t: i for i, t in enumerate(tasks)}
    data_keys = [k for k, s in features.items() if s.get("dtype") not in ("video", "image")]
    columns = data_keys + ["timestamp", "frame_index", "episode_index", "index", "task_index"]
    per_ep: dict[int, dict[str, Any]] = {}
    progress.set(phase="data", done=0, total=len(episodes), detail="")
    chunk, file = 0, 0
    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    size_mb = 0.0
    global_index = 0
    for ep in episodes:
        src = ep.directory / FRAMES_PARQUET
        ep_mb = _size_mb(src)
        if writer is not None and size_mb + ep_mb >= data_file_mb:
            writer.close()
            writer = None
            chunk, file = _update_indices(chunk, file)
            size_mb = 0.0
        table = pq.read_table(src)
        n = table.num_rows
        if n != ep.length:
            raise ExportError(
                f"episode {ep.episode_id}: {n} parquet rows but episode.json says {ep.length}"
            )
        task_col = table.column("task").to_pylist() if "task" in table.column_names else []
        if "task" in table.column_names:
            table = table.drop(["task"])
        table = table.append_column("episode_index", pa.array([ep.index] * n, type=pa.int64()))
        table = table.append_column(
            "index", pa.array(range(global_index, global_index + n), type=pa.int64())
        )
        if task_col:
            tidx = [task_index[t] for t in task_col]
        else:
            tidx = [task_index[ep.meta["tasks"][0]]] * n
        table = table.append_column("task_index", pa.array(tidx, type=pa.int64()))
        missing = [c for c in columns if c not in table.column_names]
        if missing:
            raise ExportError(f"episode {ep.episode_id}: frames.parquet lacks {missing}")
        table = table.select(columns)
        if schema is None:
            schema = table.schema
        else:
            table = table.cast(schema)
        path = staging / DATA_PATH.format(chunk_index=chunk, file_index=file)
        if writer is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(path, schema=schema, compression="snappy")
        writer.write_table(table, row_group_size=max(n, 1))  # one row group per episode
        per_ep[ep.index] = {
            "data/chunk_index": chunk,
            "data/file_index": file,
            "dataset_from_index": global_index,
            "dataset_to_index": global_index + n,
        }
        global_index += n
        size_mb += ep_mb
        progress.set(done=ep.index + 1, detail=str(path.relative_to(staging)))
    if writer is not None:
        writer.close()
    return per_ep


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=4), encoding="utf-8")


def _export_meta(
    staging: Path, manifest: dict[str, Any], episodes: list[_Episode], tasks: list[str],
    video_meta: dict[int, dict[str, Any]], data_meta: dict[int, dict[str, Any]],
    video_file_mb: int, data_file_mb: int, progress: ExportProgress,
) -> None:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    progress.set(phase="meta", done=0, total=1, detail="meta/episodes")
    fps = int(manifest["fps"])
    features: dict[str, dict] = {k: dict(v) for k, v in manifest["features"].items()}
    video_keys = [k for k, s in features.items() if s.get("dtype") == "video"]

    rows = []
    stats_list = []
    for ep in episodes:
        stats = _stats.from_jsonable(ep.meta.get("stats") or {})
        stats_list.append(stats)
        row: dict[str, Any] = {
            "episode_index": ep.index,
            "tasks": list(ep.meta.get("tasks") or []),
            "length": ep.length,
            **data_meta[ep.index],
            **video_meta.get(ep.index, {}),
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for feature, per in stats.items():
            for key, value in per.items():
                row[f"stats/{feature}/{key}"] = value.tolist()
        rows.append(row)
    episodes_path = staging / EPISODES_PATH.format(chunk_index=0, file_index=0)
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), episodes_path)

    aggregated = _stats.aggregate_stats(stats_list) if stats_list else {}
    _write_json(staging / "meta" / "stats.json", _stats.to_jsonable(aggregated))

    df = pd.DataFrame({"task_index": list(range(len(tasks)))}, index=pd.Index(tasks, name="task"))
    (staging / "meta").mkdir(parents=True, exist_ok=True)
    df.to_parquet(staging / "meta" / "tasks.parquet")

    total_frames = sum(ep.length for ep in episodes)
    for key in video_keys:
        first = staging / VIDEO_PATH.format(video_key=key, chunk_index=0, file_index=0)
        info = probe_video_info(first)
        vid = dict(manifest.get("video") or {})
        enc_fields = {
            "pix_fmt": vid.get("pix_fmt"),
            "g": vid.get("g"),
            "crf": vid.get("crf"),
            "preset": None,
            "fast_decode": 0,
            "video_backend": vid.get("backend", "pyav"),
            "extra_options": dict(vid.get("extra_options") or {}),
        }
        for name in _ENCODER_INFO_FIELDS:
            info.setdefault(f"video.{name}", enc_fields[name])
        info["is_depth_map"] = False
        features[key]["info"] = {**(features[key].get("info") or {}), **info}
    for spec in features.values():
        if isinstance(spec.get("shape"), tuple):
            spec["shape"] = list(spec["shape"])
    all_features = {**features, **json.loads(json.dumps(DEFAULT_FEATURES))}
    info_json = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": manifest.get("robot_type"),
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "chunks_size": CHUNK_SIZE,
        "data_files_size_in_mb": max(1, int(data_file_mb)),  # lerobot requires >= 1
        "video_files_size_in_mb": max(1, int(video_file_mb)),
        "fps": fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH if video_keys else None,
        "features": all_features,
    }
    _write_json(staging / "meta" / "info.json", info_json)
    progress.set(done=1, detail="meta/info.json")


def _carry_sidecars(root: Path, staging: Path, episodes: list[_Episode]) -> None:
    apollo = staging / "meta" / "apollo"
    apollo.mkdir(parents=True, exist_ok=True)
    sessions = root / SESSIONS_DIR
    if sessions.is_dir():
        for p in sorted(sessions.glob("session_*.json")):
            shutil.copy2(p, apollo / p.name)
    scenes = root / SCENES_DIR
    if scenes.is_dir():
        shutil.copytree(scenes, apollo / "scenes", dirs_exist_ok=True)
    (apollo / "episodes").mkdir(exist_ok=True)
    for ep in episodes:
        payload = dict(ep.meta)
        payload["episode_index"] = ep.index
        (apollo / "episodes" / f"episode_{ep.index:06d}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
    _write_json(
        apollo / "episode_map.json",
        {"episodes": [{"episode_index": ep.index, "episode_id": ep.episode_id} for ep in episodes]},
    )


def validate_export(repo_id: str, out: Path, expected_episodes: int) -> None:
    """Open the export with ``LeRobotDataset`` and read the first / last frame of the
    first / last episode. The ONE lerobot (torch) import of this module."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=out)
    if ds.meta.info.codebase_version != CODEBASE_VERSION:
        raise ExportError(f"export reports codebase_version {ds.meta.info.codebase_version!r}")
    if ds.num_episodes != expected_episodes:
        raise ExportError(f"export has {ds.num_episodes} episodes, expected {expected_episodes}")
    if ds.num_episodes == 0:
        return
    first = ds.meta.episodes[0]
    last = ds.meta.episodes[ds.num_episodes - 1]
    for idx in sorted({
        int(first["dataset_from_index"]), int(first["dataset_to_index"]) - 1,
        int(last["dataset_from_index"]), int(last["dataset_to_index"]) - 1,
    }):
        ds[idx]  # decodes every video key at that frame


def export_lerobot_v3(
    root: Path,
    repo_id: str,
    out: Path | None = None,
    *,
    video_file_mb: int = 200,
    data_file_mb: int = 100,
    progress: ExportProgress | None = None,
    validate: bool = True,
) -> ExportResult:
    """Rebuild the LeRobot v3 export of the dataset at ``root`` (10-frames §11.8).

    Writes into a staging directory next to ``out`` and swaps it in at the end,
    then records ``manifest.last_export``. Raises :class:`ExportError` (or the
    underlying exception) on failure — the manifest then carries the error."""
    root = Path(root)
    progress = progress or ExportProgress(repo_id)
    out = Path(out) if out is not None else root / DEFAULT_EXPORT_SUBDIR
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        sweep_export_leftovers(out)
        manifest, episodes, skipped, incomplete = _scan(root, progress)
        if not episodes:
            why = f" ({len(skipped)} excluded: export_ok false)" if skipped else ""
            if incomplete:
                why += f" ({len(incomplete)} incomplete)"
            raise ExportError("no exportable episode" + why)
        fps = int(manifest["fps"])
        features: dict[str, dict] = manifest["features"]
        video_keys = sorted(k for k, s in features.items() if s.get("dtype") == "video")
        tasks: list[str] = []
        for ep in episodes:
            for t in ep.meta.get("tasks") or []:
                if t not in tasks:
                    tasks.append(t)
        if not tasks:
            tasks = ["task"]
        staging = out.parent / f".{out.name}.tmp-{uuid.uuid4().hex[:8]}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        try:
            video_meta = _export_videos(staging, episodes, video_keys, fps, video_file_mb, progress)
            data_meta = _export_data(staging, episodes, features, tasks, data_file_mb, progress)
            _export_meta(staging, manifest, episodes, tasks, video_meta, data_meta,
                         video_file_mb, data_file_mb, progress)
            _carry_sidecars(root, staging, episodes)
            if validate:
                # Validate the STAGING tree: the previous export stays in place until the
                # new one is known good; a failure here leaves `out` untouched.
                progress.set(phase="validating", done=0, total=1, detail="LeRobotDataset(...)")
                validate_export(repo_id, staging, len(episodes))
                progress.set(done=1)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        old = None
        if out.exists():
            old = out.parent / f".{out.name}.old-{uuid.uuid4().hex[:8]}"
            os.rename(out, old)
        try:
            os.rename(staging, out)
        except BaseException:
            if old is not None:
                os.rename(old, out)
            shutil.rmtree(staging, ignore_errors=True)
            raise
        if old is not None:
            shutil.rmtree(old, ignore_errors=True)
        frames = sum(ep.length for ep in episodes)
        result = ExportResult(
            out, len(episodes), frames, skipped, [ep.episode_id for ep in episodes], incomplete
        )
        _record_export(root, out, result, error=None)
        detail = f"{len(episodes)} episodes, {frames} frames -> {out}"
        if skipped:
            detail += f"; {len(skipped)} skipped (export_ok false)"
        if incomplete:
            detail += f"; {len(incomplete)} incomplete skipped: " + "; ".join(
                f"{k} ({v})" for k, v in incomplete.items()
            )
        progress.set(phase="done", detail=detail)
        return result
    except BaseException as e:
        logger.exception("export of %s failed", repo_id)
        try:
            _record_export(root, out, None, error=f"{type(e).__name__}: {e}")
        except Exception:  # noqa: BLE001
            logger.exception("could not record the export failure in the manifest")
        progress.set(phase="failed", detail=f"{type(e).__name__}: {e}")
        raise


def _record_export(root: Path, out: Path, result: ExportResult | None, error: str | None) -> None:
    """Write ``manifest.last_export`` under the shared manifest lock. ``stale`` is DERIVED:
    the exportable episode ids on disk right now vs the ids the export took (a save or
    a delete during the job makes the fresh export stale immediately)."""
    with MANIFEST_LOCK:
        manifest = read_manifest(root)
        if manifest is None:
            return
        try:
            rel = str(out.relative_to(root))
        except ValueError:
            rel = str(out)
        last = dict(manifest.get("last_export") or {})
        if result is not None:
            last = {
                "format": EXPORT_FORMAT,
                "path": rel,
                "at": utc_now_iso(),
                "episodes": result.episodes,
                "stale": exportable_ids(root) != set(result.episode_ids),
                "error": None,
            }
        else:
            last.setdefault("format", EXPORT_FORMAT)
            last.setdefault("path", rel)
            last.setdefault("at", None)
            last.setdefault("episodes", 0)
            last.setdefault("stale", True)
            last["error"] = error
        manifest["last_export"] = last
        write_manifest(root, manifest)


__all__ = [
    "CODEBASE_VERSION",
    "DEFAULT_EXPORT_SUBDIR",
    "EXPORT_FORMAT",
    "ExportError",
    "ExportProgress",
    "ExportResult",
    "concatenate_videos",
    "export_lerobot_v3",
    "exportable_ids",
    "probe_video_info",
    "sweep_export_leftovers",
    "validate_export",
]
