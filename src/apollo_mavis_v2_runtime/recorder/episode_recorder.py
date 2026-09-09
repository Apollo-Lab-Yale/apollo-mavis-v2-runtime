"""EpisodeDirRecorder — core ``EpisodeRecorder`` over the episode-directory store
(10-frames §11; 04-runtime §10.1). LeRobot v3 is a derived export
(:mod:`.export_lerobot`), never what the recorder writes.

Ownership: constructed and driven exclusively by the :class:`RecorderThread`.
``lerobot`` is imported lazily HERE (its ``StreamingVideoEncoder`` + the encoder
config) so no other runtime module pays the import cost — the REST listing /
deletion paths (:mod:`.datasets`) and the export job stay torch-free.

Write protocol (10-frames §11.6):

1. ``start`` (control-loop thread): mint the ``episode_id``, empty the buffer.
2. ``add_frame`` (recorder thread): on the FIRST frame create
   ``episodes/.tmp-<id>/`` and ``StreamingVideoEncoder.start_episode`` with that
   directory as ``temp_dir``; then ``feed_frame`` per camera and buffer the
   non-video columns; the frames fed per camera are counted.
3. ``save``: ``finish_episode`` -> move each mp4 to ``video/<cam>.mp4`` and remove
   the encoder's ``tmp*/`` leftovers; ``fed - dropped != rows`` or ``stats is
   None`` marks ``export_ok: false`` (kept, excluded from exports);
   ``frames.parquet`` (one row group); per-feature stats (lerobot semantics,
   :mod:`.stats`); ``audio.finish`` into the directory; ``episode.json`` LAST;
   ``os.replace`` into ``episodes/<id>/``; manifest counters refreshed and the
   export marked stale. Save = one rename, nothing to finalize.
4. ``discard``: ``cancel_episode`` + rmtree the temp directory.
5. Every open sweeps stale ``.tmp-*`` directories (a crash leaves at most one).
"""

from __future__ import annotations

import functools
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from apollo_mavis_v2_core.interfaces.recorder import EpisodeRecorder

from ..config import RecorderConfig
from . import stats as _stats
from .manifest import (
    AUDIO_WAV,
    EPISODE_JSON,
    EPISODES_DIR,
    FRAMES_PARQUET,
    IMAGE_PREFIX,
    MANIFEST_LOCK,
    TMP_PREFIX,
    VIDEO_DIR,
    dataset_incompatibility,
    episode_dirs,
    is_legacy_v3,
    mark_export_stale,
    mint_episode_id,
    new_manifest,
    ns_to_iso,
    read_manifest,
    rebuild_counters,
    sweep_incomplete_episodes,
    write_json_atomic,
    write_manifest,
)

MANIFEST_REREAD_S = 1.0  # counters re-read from disk at most this often (REST deletes)

logger = logging.getLogger(__name__)

_NVENC_CODECS = frozenset({"h264_nvenc", "hevc_nvenc"})

# lerobot pins ``g=2`` (two-frame GOP so any timestamp decodes with at most
# one reference frame) for EVERY codec.  NVENC's default B-frame count (3)
# violates its own "GOP length > bf + 1" rule, so ``avcodec_open2`` fails
# with EINVAL -- regardless of driver version or frame size.  ``bf=0`` gives
# a legal GOP while keeping hardware encoding.
NVENC_EXTRA_OPTIONS: dict[str, str] = {"bf": "0"}

DEFAULT_PROBE_SIZE: tuple[int, int] = (640, 480)  # (width, height)


def encoder_extra_options(vcodec: str) -> dict[str, str]:
    """Codec-specific ``extra_options`` merged into lerobot's encoder config."""
    return dict(NVENC_EXTRA_OPTIONS) if vcodec in _NVENC_CODECS else {}


def make_rgb_encoder(vcodec: str):
    """``RGBEncoderConfig`` for an already-RESOLVED codec (never ``"auto"``),
    carrying the extras the streaming encoder needs to open at all."""
    from lerobot.configs.video import RGBEncoderConfig

    if vcodec == "auto":
        raise ValueError("resolve_vcodec() first; make_rgb_encoder needs a concrete codec")
    return RGBEncoderConfig(vcodec=vcodec, extra_options=encoder_extra_options(vcodec))


def min_video_frame_size(
    features: dict[str, dict], default: tuple[int, int] = DEFAULT_PROBE_SIZE
) -> tuple[int, int]:
    """Smallest ``(width, height)`` over the video/image features (shape is
    ``(H, W, C)``).  Hardware encoders enforce a minimum frame size, so the
    probe must use the SMALLEST stream that will actually be encoded."""
    sizes = [
        (int(spec["shape"][1]), int(spec["shape"][0]))
        for spec in features.values()
        if spec.get("dtype") in ("video", "image") and len(spec.get("shape") or ()) == 3
    ]
    if not sizes:
        return default
    return (min(w for w, _ in sizes), min(h for _, h in sizes))


@functools.cache
def _encoder_opens(codec: str, width: int, height: int, fps: int = 30) -> bool:
    """True iff PyAV can OPEN ``codec`` with the EXACT options and pixel
    format lerobot's streaming encoder will pass.

    Listing is not enough, and neither is a bare open: ``h264_nvenc`` opens
    fine without options yet fails once lerobot's ``g=2`` is applied (see
    :data:`NVENC_EXTRA_OPTIONS`), so the probe must mirror the real call.
    Cached per (codec, size): an NVENC probe costs ~0.1-0.6 s (session setup).
    """
    import io

    import av

    try:
        encoder = make_rgb_encoder(codec)
        options = encoder.get_codec_options(None, as_strings=True)
        buf = io.BytesIO()
        out = av.open(buf, "w", format="mp4")
        stream = out.add_stream(codec, rate=fps, options=options)
        stream.width, stream.height = width, height
        stream.pix_fmt = encoder.pix_fmt
        frame = av.VideoFrame.from_ndarray(np.zeros((height, width, 3), np.uint8), format="rgb24")
        for pkt in stream.encode(frame):
            out.mux(pkt)
        for pkt in stream.encode(None):
            out.mux(pkt)
        out.close()
        return True
    except Exception:
        logger.debug("encoder %s does not open at %dx%d", codec, width, height, exc_info=True)
        return False


def resolve_vcodec(vcodec: str, frame_size: tuple[int, int] = DEFAULT_PROBE_SIZE) -> str:
    """Resolve ``"auto"`` with a REAL open-probe of the hardware encoders
    (NVENC on the 4090s) at ``frame_size`` -- the smallest stream to be
    encoded -- falling back to ``libsvtav1`` (10-frames §7.5).  Anything
    other than ``"auto"`` passes through untouched."""
    if vcodec != "auto":
        return vcodec
    from lerobot.configs.video import HW_VIDEO_CODECS

    width, height = int(frame_size[0]), int(frame_size[1])
    for codec in HW_VIDEO_CODECS:
        if _encoder_opens(codec, width, height):
            logger.info("vcodec auto -> %s (probed at %dx%d)", codec, width, height)
            return codec
    logger.warning("no hardware encoder opens at %dx%d; vcodec auto -> libsvtav1", width, height)
    return "libsvtav1"


# The dataset pins the CANONICAL codec family (``manifest.video.codec``: h264 /
# hevc / av1): the export concatenates per-episode files by stream copy and
# lerobot's offline merge refuses mixed codecs, so a resumed dataset keeps its
# family whatever the current host can open.
_SOFTWARE_ENCODER_FOR_FAMILY: dict[str, str] = {"h264": "h264", "hevc": "hevc", "av1": "libsvtav1"}
_ENCODER_FAMILY_ALIASES: dict[str, str] = {
    "libx264": "h264",
    "libopenh264": "h264",
    "libx265": "hevc",
    "libsvtav1": "av1",
    "libaom-av1": "av1",
}


def codec_family(vcodec: str) -> str | None:
    """``h264`` / ``hevc`` / ``av1`` for an encoder name, ``None`` if unknown."""
    if vcodec in _ENCODER_FAMILY_ALIASES:
        return _ENCODER_FAMILY_ALIASES[vcodec]
    head = vcodec.split("_", 1)[0]  # h264_nvenc -> h264, hevc_videotoolbox -> hevc
    return head if head in _SOFTWARE_ENCODER_FOR_FAMILY else None


def stored_video_codec(root: Path) -> str | None:
    """Codec family pinned in ``<root>/manifest.json`` (``video.codec``); ``None``
    when there is no episode-directory dataset there."""
    manifest = read_manifest(root)
    if manifest is None:
        return None
    codec = (manifest.get("video") or {}).get("codec")
    return str(codec) if codec else None


def resolve_vcodec_for_family(
    vcodec: str, family: str | None, frame_size: tuple[int, int] = DEFAULT_PROBE_SIZE
) -> str:
    """:func:`resolve_vcodec`, pinned to an existing dataset's codec ``family``:
    ``"auto"`` re-probes only within it (hardware encoder first, then the
    software one), and an explicit codec of another family is rejected so the
    operator records into a fresh repo instead of producing per-episode videos
    the export cannot stream-copy together."""
    if family is None:
        return resolve_vcodec(vcodec, frame_size)
    width, height = int(frame_size[0]), int(frame_size[1])
    if vcodec != "auto":
        if codec_family(vcodec) != family:
            raise ValueError(
                f"vcodec {vcodec!r} ({codec_family(vcodec)}) would mix with the {family} "
                "videos already in the dataset; use vcodec 'auto' or a fresh repo"
            )
        return vcodec
    from lerobot.configs.video import HW_VIDEO_CODECS

    for codec in HW_VIDEO_CODECS:
        if codec_family(codec) == family and _encoder_opens(codec, width, height):
            logger.info("resumed %s dataset: vcodec auto -> %s", family, codec)
            return codec
    software = _SOFTWARE_ENCODER_FOR_FAMILY[family]
    if _encoder_opens(software, width, height):
        logger.warning(
            "resumed %s dataset: no hardware encoder opens at %dx%d; vcodec auto -> %s",
            family,
            width,
            height,
            software,
        )
        return software
    raise RuntimeError(f"no {family} encoder opens at {width}x{height}; cannot resume")


def resolve_vcodec_for_dataset(
    vcodec: str, root: Path, frame_size: tuple[int, int] = DEFAULT_PROBE_SIZE
) -> str:
    """:func:`resolve_vcodec_for_family` with the family read from ``<root>/manifest.json``."""
    return resolve_vcodec_for_family(vcodec, stored_video_codec(root), frame_size)


_PA_TYPES: dict[str, Any] = {}


def _pa_type(dtype: str):
    import pyarrow as pa

    if not _PA_TYPES:
        _PA_TYPES.update(
            {
                "float32": pa.float32(),
                "float64": pa.float64(),
                "int8": pa.int8(),
                "int16": pa.int16(),
                "int32": pa.int32(),
                "int64": pa.int64(),
                "uint8": pa.uint8(),
                "bool": pa.bool_(),
            }
        )
    try:
        return _PA_TYPES[dtype]
    except KeyError:
        raise ValueError(f"unsupported feature dtype {dtype!r}") from None


class EpisodeDirRecorder(EpisodeRecorder):
    """Buffered episode recording into ``episodes/<episode_id>/`` (04-runtime §10.1)."""

    def __init__(
        self,
        cfg: RecorderConfig,
        features: dict[str, dict],
        root: Path,
        repo_id: str,
        robot_type: str,
        default_task: str,
    ) -> None:
        self.cfg = cfg
        self.root = Path(root)
        self.repo_id = repo_id
        self.fps = int(cfg.fps)
        self.features = {k: dict(v) for k, v in features.items()}
        self.video_keys = [k for k, s in features.items() if s.get("dtype") == "video"]
        self.data_features = {
            k: dict(s) for k, s in features.items() if s.get("dtype") not in ("video", "image")
        }
        self._task = default_task or "task"
        self._encoder = None  # StreamingVideoEncoder, built on the first episode
        # serialises start / prepare / add_frame / save / discard / finalize
        self._lock = threading.RLock()
        self._recording = False
        self._episode_id: str | None = None
        self._tmp: Path | None = None
        self._encoder_open = False
        self._rows: list[dict[str, np.ndarray]] = []
        self._fed: dict[str, int] = {}
        self._first_wallclock_ns: int | None = None
        self._finished: tuple[dict, dict[str, int]] | None = None
        self._audio_block: dict[str, Any] | None = None
        self._audio_done = False
        self._published: Path | None = None  # episodes/<id> once os.replace ran
        self._finalized = False
        self._manifest_read_at = 0.0
        if is_legacy_v3(self.root):
            raise ValueError("legacy LeRobot v3 dataset (read-only; already trainable as-is)")
        manifest = read_manifest(self.root)
        if manifest is not None:
            why = dataset_incompatibility(manifest, features, self.fps, robot_type)
            if why is not None:
                raise ValueError(f"cannot be continued by this session: {why}")
            family = (manifest.get("video") or {}).get("codec")
        else:
            family = None
        self.vcodec = resolve_vcodec_for_family(cfg.vcodec, family, min_video_frame_size(features))
        self._rgb_encoder = make_rgb_encoder(self.vcodec)
        self.video_identity: dict[str, Any] = {
            "codec": codec_family(self.vcodec) or self.vcodec,
            "encoder": self.vcodec,
            "pix_fmt": self._rgb_encoder.pix_fmt,
            "g": self._rgb_encoder.g,
            "crf": self._rgb_encoder.crf,
            "extra_options": {str(k): str(v) for k, v in self._rgb_encoder.extra_options.items()},
            "backend": self._rgb_encoder.video_backend,
        }
        if manifest is None:
            manifest = new_manifest(repo_id, self.fps, robot_type, features, self.video_identity)
            (self.root / EPISODES_DIR).mkdir(parents=True, exist_ok=True)
        else:
            self._sweep_own_tmp()  # every open (10-frames §11.6 step 5)
            rebuild_counters(self.root, manifest)
        self.manifest = manifest
        write_manifest(self.root, self.manifest)

    # -- properties --------------------------------------------------------------------
    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def episode_id(self) -> str | None:
        return self._episode_id

    def _counters(self) -> tuple[int, int]:
        """Manifest counters, re-read from disk at most every ``MANIFEST_REREAD_S`` so a
        mid-session REST delete shows up in telemetry without a save."""
        now = time.monotonic()
        if now - self._manifest_read_at >= MANIFEST_REREAD_S:
            self._manifest_read_at = now
            try:
                fresh = read_manifest(self.root)
            except ValueError:
                fresh = None
            if fresh is not None:
                self.manifest = fresh
        episodes = int(self.manifest.get("episodes", 0) or 0)
        frames = int(self.manifest.get("frames", 0) or 0)
        return episodes, frames

    @property
    def episodes_saved(self) -> int:
        return self._counters()[0]

    @property
    def total_frames(self) -> int:
        return self._counters()[1]

    @property
    def frames_in_buffer(self) -> int:
        return len(self._rows)

    @property
    def open_tmp_dir(self) -> Path | None:
        """The open episode's ``.tmp-*`` directory (sweeps must skip it)."""
        return self._tmp

    # -- core EpisodeRecorder API --------------------------------------------------------
    def start(self, meta: dict[str, object]) -> None:
        """Open a new episode buffer: mint the id, no filesystem op (§11.6 step 1)."""
        with self._lock:
            if self._recording:
                raise RuntimeError("episode already recording")
            if self._finalized:
                raise RuntimeError("recorder finalized")
            task = meta.get("task")
            if task:
                self._task = str(task)
            self._episode_id = mint_episode_id()
            self._tmp = None
            self._encoder_open = False
            self._rows = []
            self._fed = dict.fromkeys(self.video_keys, 0)
            self._first_wallclock_ns = None
            self._finished = None
            self._audio_block = None
            self._audio_done = False
            self._published = None
            self._recording = True

    def prepare(self) -> None:
        """Create ``episodes/.tmp-<id>/`` and start the encoder episode NOW (recorder
        thread, right after ``episode_new``): the NVENC session start holds the GIL
        for 150-300 ms, and here that stall lands before any frame instead of under
        a moving arm (04-runtime §10.1). ``add_frame`` still opens lazily when a
        caller skipped this step."""
        with self._lock:
            if not self._recording or self._episode_id is None:
                raise RuntimeError("no episode buffer open")
            if self._tmp is None:
                tmp = self.root / EPISODES_DIR / f"{TMP_PREFIX}{self._episode_id}"
                tmp.mkdir(parents=True, exist_ok=True)
                self._tmp = tmp
            if self.video_keys and not self._encoder_open:
                enc = self._encoder_instance()
                enc.start_episode(list(self.video_keys), temp_dir=self._tmp)
                self._encoder_open = True

    def add_frame(self, frame: dict[str, object]) -> None:
        """Buffer one frame (opens the temp directory + encoder if ``prepare`` did not)."""
        with self._lock:
            if not self._recording or self._episode_id is None:
                raise RuntimeError("no episode buffer open")
            if self._tmp is None or (self.video_keys and not self._encoder_open):
                self.prepare()
            if self._first_wallclock_ns is None:
                wall = frame.get("wallclock_ns")
                self._first_wallclock_ns = (
                    int(np.asarray(wall).reshape(-1)[0]) if wall is not None else time.time_ns()
                )
            if self.video_keys:
                enc = self._encoder_instance()
                for key in self.video_keys:
                    enc.feed_frame(key, np.asarray(frame[key]))
                    self._fed[key] += 1
            row: dict[str, np.ndarray] = {}
            for key, spec in self.data_features.items():
                value = np.asarray(frame[key])
                shape = tuple(spec.get("shape") or ())
                if shape == (1,):
                    value = value.reshape(-1)[:1]
                elif len(shape) == 1:
                    value = value.reshape(-1)
                    if value.shape[0] != shape[0]:
                        raise ValueError(
                            f"feature {key!r}: got {value.shape[0]} values, schema says {shape[0]}"
                        )
                row[key] = value
            self._rows.append(row)

    def save(self, sidecar: dict[str, object], audio: Any | None = None) -> tuple[int, str]:
        """Publish ``episodes/<episode_id>/`` (§11.6 step 3). Re-entrant: a retry after a
        failure resumes after the steps that already completed (encoder finish, audio
        finish, the publication itself). Everything AFTER ``os.replace`` — manifest
        refresh, ordinal — is best-effort: the episode IS on disk by then, so a
        failure there must never leave the recorder open on a vanished temp dir.
        Returns ``(ordinal, episode_id)``."""
        with self._lock:
            if not self._recording or self._episode_id is None:
                raise RuntimeError("no episode buffer open")
            episode_id = self._episode_id
            if self._published is None:
                if self._tmp is None or not self._rows:
                    raise ValueError("empty episode")
                self._publish(sidecar, audio)
            final = self._published
            assert final is not None
            ordinal = self._after_publication(final, episode_id)
            self._reset_episode()
            return ordinal, episode_id

    def _publish(self, sidecar: dict[str, object], audio: Any | None) -> None:
        tmp = self._tmp
        assert tmp is not None
        rows = len(self._rows)
        if self._finished is None:
            results: dict = {}
            dropped: dict[str, int] = {}
            if self.video_keys:
                enc = self._encoder_instance()
                results = enc.finish_episode()
                dropped = dict(getattr(enc, "_dropped_frames", {}) or {})
            self._finished = (results, dropped)
        results, dropped = self._finished

        stats: dict[str, dict[str, np.ndarray]] = {}
        video_block: dict[str, Any] = {}
        notes: list[str] = []
        vdir = tmp / VIDEO_DIR
        vdir.mkdir(exist_ok=True)
        for key in self.video_keys:
            cam = key[len(IMAGE_PREFIX):]
            path, vstats = results[key]
            path = Path(path)
            dst = vdir / f"{cam}.mp4"
            if path.exists():
                shutil.move(str(path), str(dst))
                shutil.rmtree(path.parent, ignore_errors=True)  # the encoder's tmp*/ leftover
            fed = int(self._fed.get(key, 0))
            drop = int(dropped.get(key, 0) or 0)
            frames = fed - drop
            if frames != rows:
                notes.append(
                    f"{cam}: {frames} video frames for {rows} rows "
                    f"({drop} dropped by the encoder queue)"
                )
            if vstats is None:
                notes.append(f"{cam}: no video stats (fewer than 2 frames encoded)")
            else:
                stats[key] = _stats.video_stats_from_encoder(vstats)
            shape = self.features[key].get("shape") or (0, 0, 3)
            video_block[cam] = {
                "file": f"{VIDEO_DIR}/{cam}.mp4",
                "frames": frames,
                **self.video_identity,
                "width": int(shape[1]),
                "height": int(shape[0]),
                "encoder_drops": drop,
            }

        table, columns = self._table()
        import pyarrow.parquet as pq

        pq.write_table(table, tmp / FRAMES_PARQUET, compression="snappy", row_group_size=rows)
        for key, arr in columns.items():
            stats[key] = _stats.feature_stats(arr)

        if audio is not None and not self._audio_done:
            try:
                self._audio_block = audio.finish(tmp / AUDIO_WAV, tmp)
            except Exception:  # noqa: BLE001 - audio never blocks a save
                logger.exception("episode audio sidecar failed; episode saved without audio")
                self._audio_block = None
            self._audio_done = True  # finish() drained the sink: never call it twice

        first_ns = self._first_wallclock_ns or time.time_ns()
        export_ok = not notes
        payload: dict[str, Any] = dict(sidecar)
        payload.update(
            {
                "episode_id": self._episode_id,
                "episode_index": None,
                "recorded_at": ns_to_iso(first_ns),
                "length": rows,
                "fps": self.fps,
                "duration_s": rows / self.fps,
                "tasks": [self._task],
                "video": video_block,
                "audio": self._audio_block,
                "stats": _stats.to_jsonable(stats),
                "export_ok": export_ok,
                "export_note": None if export_ok else "; ".join(notes),
            }
        )
        payload.setdefault("frames_dropped", 0)
        payload.setdefault("success", None)
        write_json_atomic(tmp / EPISODE_JSON, payload)  # LAST file inside the temp dir
        final = self.root / EPISODES_DIR / str(self._episode_id)
        os.replace(tmp, final)
        self._published = final
        if not export_ok:
            logger.warning(
                "episode %s saved with export_ok=false: %s", self._episode_id, "; ".join(notes)
            )

    def _after_publication(self, final: Path, episode_id: str) -> int:
        """Manifest refresh (counters REBUILT from the directories, never incremented
        — a REST delete may have run mid-session) + the capture-order ordinal; both
        best-effort."""
        try:
            with MANIFEST_LOCK:
                fresh = read_manifest(self.root)
                if fresh is not None:
                    self.manifest = fresh
                rebuild_counters(self.root, self.manifest)
                mark_export_stale(self.manifest)
                write_manifest(self.root, self.manifest)
                self._manifest_read_at = time.monotonic()
        except Exception:  # noqa: BLE001 - the episode is published; counters rebuild on the next open
            logger.exception("manifest refresh after publishing %s failed", episode_id)
        try:
            return [p.name for p in episode_dirs(self.root)].index(episode_id)
        except Exception:  # noqa: BLE001
            logger.exception("ordinal lookup after publishing %s failed", episode_id)
            return max(int(self.manifest.get("episodes", 1) or 1) - 1, 0)

    def discard(self) -> None:
        """Drop the buffer: cancel the encoder, remove the temp directory (§11.6 step 4).
        A PUBLISHED episode (save failed only after its rename) is left in place."""
        with self._lock:
            if not self._recording:
                return
            if self._published is not None:
                self._reset_episode()
                return
            if self._encoder is not None:
                try:
                    self._encoder.cancel_episode()
                except Exception:  # noqa: BLE001
                    logger.exception("cancel_episode failed")
            if self._tmp is not None:
                shutil.rmtree(self._tmp, ignore_errors=True)
            self._reset_episode()

    def finalize(self) -> None:
        """Idempotent close: discard an open episode, close the encoder, sweep."""
        with self._lock:
            if self._finalized:
                return
            try:
                self.discard()
            except Exception:  # noqa: BLE001
                logger.exception("discard during finalize failed")
                self._reset_episode()
            if self._encoder is not None:
                try:
                    self._encoder.close()
                except Exception:  # noqa: BLE001
                    logger.exception("encoder close failed")
                self._encoder = None
            self._sweep_own_tmp()
            self._finalized = True

    # -- internals -----------------------------------------------------------------------
    def _encoder_instance(self):
        if self._encoder is None:
            from lerobot.datasets.video_utils import StreamingVideoEncoder

            self._encoder = StreamingVideoEncoder(self.fps, rgb_encoder=self._rgb_encoder)
        return self._encoder

    def _reset_episode(self) -> None:
        self._recording = False
        self._episode_id = None
        self._tmp = None
        self._encoder_open = False
        self._rows = []
        self._fed = {}
        self._first_wallclock_ns = None
        self._finished = None
        self._audio_block = None
        self._audio_done = False
        self._published = None

    def _sweep_own_tmp(self) -> None:
        base = self.root / EPISODES_DIR
        if not base.is_dir():
            return
        keep = {self._tmp} if self._tmp is not None else set()
        for tmp in sorted(base.glob(f"{TMP_PREFIX}*")):
            if tmp.is_dir() and tmp not in keep:
                shutil.rmtree(tmp, ignore_errors=True)
                logger.warning("swept incomplete episode directory %s", tmp)

    def _table(self):
        """``frames.parquet`` (10-frames §11.4): the §7 non-video features as their
        exact dtypes / fixed-size lists + ``timestamp`` (float32, frame_index / fps),
        ``frame_index`` (int64) and ``task`` (string). Returns the table and the
        numpy columns the stats are computed from."""
        import pyarrow as pa

        n = len(self._rows)
        cols: dict[str, Any] = {}
        arrays: dict[str, np.ndarray] = {}
        for key, spec in self.data_features.items():
            dtype = str(spec["dtype"])
            shape = tuple(spec.get("shape") or ())
            pa_type = _pa_type(dtype)
            np_dtype = np.bool_ if dtype == "bool" else np.dtype(dtype)
            if shape == (1,):
                arr = np.asarray([r[key][0] for r in self._rows], dtype=np_dtype)
                cols[key] = pa.array(arr, type=pa_type)
                arrays[key] = arr
            elif len(shape) == 1:
                arr = np.stack([np.asarray(r[key], dtype=np_dtype) for r in self._rows])
                flat = pa.array(arr.reshape(-1), type=pa_type)
                cols[key] = pa.FixedSizeListArray.from_arrays(flat, int(shape[0]))
                arrays[key] = arr
            else:
                raise ValueError(f"feature {key!r}: unsupported shape {shape}")
        cols["timestamp"] = pa.array(
            (np.arange(n, dtype=np.float64) / self.fps).astype(np.float32), type=pa.float32()
        )
        cols["frame_index"] = pa.array(np.arange(n, dtype=np.int64), type=pa.int64())
        cols["task"] = pa.array([self._task] * n, type=pa.string())
        return pa.table(cols), arrays


__all__ = [
    "DEFAULT_PROBE_SIZE",
    "EpisodeDirRecorder",
    "NVENC_EXTRA_OPTIONS",
    "codec_family",
    "dataset_incompatibility",
    "encoder_extra_options",
    "make_rgb_encoder",
    "min_video_frame_size",
    "resolve_vcodec",
    "resolve_vcodec_for_dataset",
    "resolve_vcodec_for_family",
    "stored_video_codec",
    "sweep_incomplete_episodes",
]
