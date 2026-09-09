"""LeRobot v3 export of the episode-directory store (10-frames §11.8; 04-runtime §10.6):
two episodes -> export -> ``LeRobotDataset`` reads it back (num_episodes, per-frame
timestamp, first / last frame decode, stats.json, episode_map.json); deleting one
episode and re-exporting yields byte-identical parquet rows to recording only the
other; export_ok false episodes are skipped; the manifest tracks the export."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from apollo_mavis_v2_runtime.config import RecorderConfig
from apollo_mavis_v2_runtime.recorder import stats as rstats
from apollo_mavis_v2_runtime.recorder.datasets import DatasetStore
from apollo_mavis_v2_runtime.recorder.episode_recorder import EpisodeDirRecorder
from apollo_mavis_v2_runtime.recorder.export_lerobot import (
    DEFAULT_FEATURES,
    ExportError,
    ExportProgress,
    concatenate_videos,
    export_lerobot_v3,
    probe_video_info,
)
from apollo_mavis_v2_runtime.recorder.features import ArmMeta, build_features
from apollo_mavis_v2_runtime.recorder.manifest import read_manifest
from apollo_mavis_v2_runtime.recorder.sidecars import SidecarWriter

REPO = "apollo/export_test"
CAM = "cam0"
W, H = 64, 48
FEATURES = build_features([ArmMeta("arm0", True)], {"arm0": "arm_base:arm0"}, {CAM: (W, H)})


def _frame(i: int, seed: int) -> dict:
    rng = np.random.default_rng(seed * 1000 + i)
    return {
        "action": rng.standard_normal(8).astype(np.float32),
        "observation.state": rng.standard_normal(16).astype(np.float32),
        f"observation.images.{CAM}": np.full((H, W, 3), (seed * 40 + i * 7) % 255, dtype=np.uint8),
        "intervention": np.array([False]),
        "action_source": np.array([1], dtype=np.int8),
        "wallclock_ns": np.array([1_700_000_000_000_000_000 + seed * 10**12 + i * 40_000_000],
                                 dtype=np.int64),
    }


def record(root: Path, episodes: list[tuple[int, int]], task: str = "pick") -> list[str]:
    """``episodes`` = [(length, seed), ...]; returns the ids in capture order."""
    rec = EpisodeDirRecorder(RecorderConfig(vcodec="libsvtav1"), FEATURES, root, REPO,
                             "xarm7_1arm_rail_mujoco", task)
    ids = []
    for n, seed in episodes:
        rec.start({})
        for i in range(n):
            rec.add_frame(_frame(i, seed))
        sidecar = {"session_id": "s1", "frames_dropped": 0, "frames": {"arm0": "arm_base:arm0"}}
        ids.append(rec.save(sidecar, None)[1])
        time.sleep(0.002)
    rec.finalize()
    side = SidecarWriter(root)
    side.write_session("s1", "collect", {"task": task}, {"kind": "sim", "arm_ids": ["arm0"]}, {})
    side.archive_scene_xml("<mujoco/>")
    return ids


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    base = tmp_path_factory.mktemp("ds")
    root = base / REPO
    ids = record(root, [(12, 1), (7, 2)])
    progress = ExportProgress(REPO)
    t0 = time.monotonic()
    result = export_lerobot_v3(root, REPO, progress=progress, validate=False)
    remux_s = time.monotonic() - t0
    return base, root, ids, result, progress, remux_s


def test_export_layout_and_progress(exported):
    base, root, ids, result, progress, remux_s = exported
    out = root / "exports" / "lerobot_v3"
    assert result.path == out and result.episodes == 2
    assert result.frames == 19 and result.skipped == []
    assert remux_s < 2.0, f"videos + data + meta took {remux_s:.2f}s (remux only, no decode)"
    tele = progress.telemetry()
    assert tele.phase == "done" and tele.repo_id == REPO and tele.format == "lerobot_v3"
    assert "2 episodes, 19 frames" in tele.detail
    # the phase clock: videos -> data -> meta -> done all inside 2 s (remux, no decode)
    started = progress.phase_started
    assert {"scanning", "videos", "data", "meta", "done"} <= set(started)
    assert started["done"] - started["videos"] < 2.0, started
    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    assert files == [
        "data/chunk-000/file-000.parquet",
        "meta/apollo/episode_map.json",
        "meta/apollo/episodes/episode_000000.json",
        "meta/apollo/episodes/episode_000001.json",
        "meta/apollo/scenes/" + next((root / "scenes").iterdir()).name,
        "meta/apollo/session_s1.json",
        "meta/episodes/chunk-000/file-000.parquet",
        "meta/info.json",
        "meta/stats.json",
        "meta/tasks.parquet",
        f"videos/observation.images.{CAM}/chunk-000/file-000.mp4",
    ]
    emap = json.loads((out / "meta" / "apollo" / "episode_map.json").read_text())
    assert emap == {"episodes": [{"episode_index": 0, "episode_id": ids[0]},
                                 {"episode_index": 1, "episode_id": ids[1]}]}
    ep1 = json.loads((out / "meta" / "apollo" / "episodes" / "episode_000001.json").read_text())
    assert ep1["episode_index"] == 1 and ep1["episode_id"] == ids[1]
    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0" and info["fps"] == 25
    assert (info["total_episodes"], info["total_frames"], info["total_tasks"]) == (2, 19, 1)
    assert info["splits"] == {"train": "0:2"} and info["robot_type"] == "xarm7_1arm_rail_mujoco"
    assert set(DEFAULT_FEATURES) <= set(info["features"])
    assert info["features"]["action"]["info"]["apollo_schema"] == 1
    vinfo = info["features"][f"observation.images.{CAM}"]["info"]
    assert (vinfo["video.codec"], vinfo["video.pix_fmt"]) == ("av1", "yuv420p")
    assert vinfo["video.fps"] == 25
    assert (vinfo["video.height"], vinfo["video.width"], vinfo["video.g"]) == (H, W, 2)
    assert vinfo["video.extra_options"] == {} and vinfo["is_depth_map"] is False
    # data parquet: exactly the features + the five bookkeeping columns, one row group per episode
    pf = pq.ParquetFile(out / "data" / "chunk-000" / "file-000.parquet")
    assert pf.metadata.num_rows == 19 and pf.metadata.num_row_groups == 2
    assert set(pf.schema_arrow.names) == {
        "action", "observation.state", "intervention", "action_source", "wallclock_ns",
        "timestamp", "frame_index", "episode_index", "index", "task_index",
    }
    table = pf.read()
    assert table.column("index").to_pylist() == list(range(19))
    assert table.column("episode_index").to_pylist() == [0] * 12 + [1] * 7
    assert table.column("task_index").to_pylist() == [0] * 19
    wall = np.asarray(table.column("wallclock_ns").to_pylist(), dtype=np.int64)
    assert (np.diff(wall) > 0).all()
    assert (np.asarray(table.column("action_source").to_pylist()) == 1).all()
    # meta/episodes: positional columns + flattened stats
    meta = pq.read_table(out / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pylist()
    assert [m["episode_index"] for m in meta] == [0, 1]
    assert meta[0]["dataset_from_index"] == 0 and meta[0]["dataset_to_index"] == 12
    assert meta[1]["dataset_from_index"] == 12 and meta[1]["dataset_to_index"] == 19
    assert meta[0][f"videos/observation.images.{CAM}/from_timestamp"] == 0.0
    assert meta[1][f"videos/observation.images.{CAM}/from_timestamp"] == pytest.approx(12 / 25)
    assert meta[1][f"videos/observation.images.{CAM}/to_timestamp"] == pytest.approx(19 / 25)
    assert meta[0]["tasks"] == ["pick"] and meta[0]["length"] == 12
    assert "stats/action/mean" in meta[0] and len(meta[0]["stats/action/mean"]) == 8
    # meta/stats.json = aggregate_stats over the per-episode stats
    stats = json.loads((out / "meta" / "stats.json").read_text())
    assert set(stats) == {
        "action", "observation.state", "intervention", "action_source", "wallclock_ns",
        f"observation.images.{CAM}",
    }
    assert stats["action"]["count"] == [19]
    assert np.asarray(stats[f"observation.images.{CAM}"]["mean"]).shape == (3, 1, 1)
    # the store sees a fresh export; the manifest records it
    m = read_manifest(root)
    assert m["last_export"]["episodes"] == 2 and m["last_export"]["stale"] is False
    assert m["last_export"]["path"] == "exports/lerobot_v3" and m["last_export"]["error"] is None
    assert DatasetStore(base).describe(REPO).export.state == "fresh"
    first_mp4 = out / f"videos/observation.images.{CAM}/chunk-000/file-000.mp4"
    assert probe_video_info(first_mp4)["video.codec"] == "av1"


def test_export_reads_back_with_lerobot(exported):
    """The validating step: lerobot's reader (torch) opens the export and decodes the
    first / last frame of the first / last episode."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    base, root, ids, result, progress, _ = exported
    out = root / "exports" / "lerobot_v3"
    ds = LeRobotDataset(REPO, root=out)
    assert ds.meta.info.codebase_version == "v3.0" and ds.num_episodes == 2 and ds.fps == 25
    assert ds.num_frames == 19
    frames_ep0 = pq.read_table(root / "episodes" / ids[0] / "frames.parquet")
    for idx in (0, 11, 12, 18):
        row = ds[idx]
        ep = 0 if idx < 12 else 1
        k = idx if idx < 12 else idx - 12
        assert int(row["episode_index"]) == ep and int(row["index"]) == idx
        assert float(row["timestamp"]) == pytest.approx(k / 25, abs=1e-6)
        assert row["task"] == "pick"
        img = row[f"observation.images.{CAM}"]
        assert tuple(img.shape) == (3, H, W)
        seed = 1 if ep == 0 else 2
        expected = ((seed * 40 + k * 7) % 255) / 255.0
        assert float(np.asarray(img).mean()) == pytest.approx(expected, abs=0.03)  # the right frame
        if ep == 0:
            assert np.allclose(np.asarray(row["action"]), frames_ep0.column("action")[k].as_py(),
                               atol=1e-6)
    from apollo_mavis_v2_runtime.recorder.export_lerobot import validate_export

    validate_export(REPO, out, 2)
    with pytest.raises(ExportError, match="expected 3"):
        validate_export(REPO, out, 3)


def test_delete_then_reexport_matches_recording_only_the_other(tmp_path):
    """Deleting episode 1 and re-exporting must equal an export of a dataset that
    only ever recorded episode 2 — the same parquet rows, byte for byte."""
    a = tmp_path / "a" / REPO
    ids_a = record(a, [(9, 3), (6, 4)])
    b = tmp_path / "b" / REPO
    record(b, [(6, 4)])
    store = DatasetStore(tmp_path / "a")
    export_lerobot_v3(a, REPO, validate=False)
    kept = a / "episodes" / ids_a[1]
    mtimes = {p: p.stat().st_mtime_ns for p in kept.rglob("*")}
    store.delete_episode(REPO, ids_a[0])
    assert {p: p.stat().st_mtime_ns for p in kept.rglob("*")} == mtimes  # no rewrite, no re-encode
    assert store.describe(REPO).export.state == "stale"
    ra = export_lerobot_v3(a, REPO, validate=False)
    rb = export_lerobot_v3(b, REPO, validate=False)
    assert ra.episodes == 1 and rb.episodes == 1
    ta = pq.read_table(a / "exports/lerobot_v3/data/chunk-000/file-000.parquet")
    tb = pq.read_table(b / "exports/lerobot_v3/data/chunk-000/file-000.parquet")
    assert ta.equals(tb)
    assert ta.column("index").to_pylist() == list(range(6))
    assert ta.column("episode_index").to_pylist() == [0] * 6
    assert store.describe(REPO).export.state == "fresh"
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    assert LeRobotDataset(REPO, root=a / "exports" / "lerobot_v3").num_episodes == 1


def test_export_ok_false_episodes_are_skipped_and_reported(tmp_path):
    root = tmp_path / REPO
    ids = record(root, [(5, 5), (4, 6), (3, 7)])
    ep = root / "episodes" / ids[1] / "episode.json"
    payload = json.loads(ep.read_text())
    payload["export_ok"] = False
    payload["export_note"] = "cam0: 3 video frames for 4 rows"
    ep.write_text(json.dumps(payload))
    progress = ExportProgress(REPO)
    result = export_lerobot_v3(root, REPO, progress=progress, validate=False)
    assert result.episodes == 2 and result.skipped == [ids[1]] and result.frames == 8
    assert "1 skipped" in progress.telemetry().detail
    emap = json.loads((root / "exports/lerobot_v3/meta/apollo/episode_map.json").read_text())
    assert [e["episode_id"] for e in emap["episodes"]] == [ids[0], ids[2]]
    # nothing exportable -> failed, recorded in the manifest
    for eid in (ids[0], ids[2]):
        p = root / "episodes" / eid / "episode.json"
        d = json.loads(p.read_text())
        d["export_ok"] = False
        p.write_text(json.dumps(d))
    progress = ExportProgress(REPO)
    with pytest.raises(ExportError, match="no exportable episode"):
        export_lerobot_v3(root, REPO, progress=progress, validate=False)
    assert progress.telemetry().phase == "failed"
    m = read_manifest(root)
    assert m["last_export"]["error"]
    assert DatasetStore(tmp_path).describe(REPO).export.state == "failed"


def test_video_shard_rolls_at_the_size_cap_and_on_identity_change(tmp_path):
    root = tmp_path / REPO
    ids = record(root, [(6, 8), (6, 9), (6, 10)])
    # a tiny cap: every episode in its own file, from_timestamp restarts at 0
    export_lerobot_v3(root, REPO, video_file_mb=1e-9, data_file_mb=1e-9, validate=False)
    out = root / "exports" / "lerobot_v3"
    vdir = out / f"videos/observation.images.{CAM}/chunk-000"
    assert sorted(p.name for p in vdir.iterdir()) == [
        "file-000.mp4", "file-001.mp4", "file-002.mp4",
    ]
    assert sorted(p.name for p in (out / "data/chunk-000").iterdir()) == [
        "file-000.parquet", "file-001.parquet", "file-002.parquet",
    ]
    meta = pq.read_table(out / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    assert [m[f"videos/observation.images.{CAM}/file_index"] for m in meta] == [0, 1, 2]
    assert all(m[f"videos/observation.images.{CAM}/from_timestamp"] == 0.0 for m in meta)
    assert [m["data/file_index"] for m in meta] == [0, 1, 2]
    assert [m["dataset_from_index"] for m in meta] == [0, 6, 12]
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(REPO, root=out)
    assert ds.num_episodes == 3 and int(ds[17]["episode_index"]) == 2
    # an encoder-identity change (say a resumed session on another host) splits too
    ep = root / "episodes" / ids[2] / "episode.json"
    d = json.loads(ep.read_text())
    d["video"][CAM]["encoder"] = "h264_nvenc_elsewhere"
    ep.write_text(json.dumps(d))
    export_lerobot_v3(root, REPO, validate=False)
    assert sorted(p.name for p in vdir.iterdir()) == ["file-000.mp4", "file-001.mp4"]
    meta = pq.read_table(out / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    assert [m[f"videos/observation.images.{CAM}/file_index"] for m in meta] == [0, 0, 1]
    assert meta[1][f"videos/observation.images.{CAM}/from_timestamp"] == pytest.approx(6 / 25)


def _packets(path: Path) -> list[tuple[int, int]]:
    """(size, keyframe) of every video packet, in stream order."""
    import av

    with av.open(str(path)) as c:
        return [(p.size, bool(p.is_keyframe)) for p in c.demux(video=0) if p.dts is not None]


def test_concatenate_videos_is_a_remux(tmp_path):
    root = tmp_path / REPO
    ids = record(root, [(5, 11), (4, 12)])
    paths = [root / "episodes" / i / "video" / f"{CAM}.mp4" for i in ids]
    out = tmp_path / "cat.mp4"
    concatenate_videos(paths, out)
    import av

    with av.open(str(out)) as c:
        frames = [f for f in c.decode(video=0)]
    assert len(frames) == 9
    assert [float(f.pts * f.time_base) for f in frames] == pytest.approx([k / 25 for k in range(9)])
    # a REMUX: the concatenated file's first N packets are the first episode's packets, byte
    # for byte in size (a re-encode would change them); the second episode follows
    first = _packets(paths[0])
    second = _packets(paths[1])
    cat = _packets(out)
    assert cat[: len(first)] == first and cat[len(first):] == second
    assert cat[0][1] and cat[len(first)][1]  # every episode starts on a keyframe (g=2)
    # movflags faststart: the moov atom precedes mdat in the concatenated file
    data = out.read_bytes()
    assert 0 < data.find(b"moov") < data.find(b"mdat")
    per_episode = paths[0].read_bytes()
    assert per_episode.find(b"mdat") < per_episode.find(b"moov")  # the per-episode file is not


@pytest.mark.parametrize("fps", [20, 25, 30])
def test_export_round_trips_at_every_supported_fps(tmp_path, fps):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / REPO
    rec = EpisodeDirRecorder(RecorderConfig(vcodec="libsvtav1", fps=fps), FEATURES, root, REPO,
                             "xarm7_1arm_rail_mujoco", "pick")
    for n, seed in ((fps, 21), (fps // 2, 22)):
        rec.start({})
        for i in range(n):
            rec.add_frame(_frame(i, seed))
        rec.save({"session_id": "s1", "frames_dropped": 0}, None)
        time.sleep(0.002)
    rec.finalize()
    result = export_lerobot_v3(root, REPO, validate=True)
    out = root / "exports" / "lerobot_v3"
    ds = LeRobotDataset(REPO, root=out)
    assert ds.fps == fps and ds.num_episodes == 2 and result.frames == fps + fps // 2
    row = ds[fps]  # first frame of episode 2
    assert int(row["episode_index"]) == 1 and float(row["timestamp"]) == pytest.approx(0.0)
    meta = pq.read_table(out / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    assert meta[1][f"videos/observation.images.{CAM}/from_timestamp"] == pytest.approx(1.0)
    first_mp4 = out / f"videos/observation.images.{CAM}/chunk-000/file-000.mp4"
    assert probe_video_info(first_mp4)["video.fps"] == fps


def test_aggregate_stats_parity_with_lerobot():
    """Exact parity of the torch-free aggregation with lerobot's compute_stats.aggregate_stats."""
    from lerobot.datasets.compute_stats import aggregate_stats as lerobot_aggregate

    rng = np.random.default_rng(3)
    stats_list = []
    for n in (12, 7, 30):
        vec = rng.standard_normal((n, 8))
        img = rng.uniform(0, 255, size=(n * 5, 3))
        s = {"action": rstats.feature_stats(vec),
             "observation.images.c": rstats.video_stats_from_encoder({
                 "min": img.min(0), "max": img.max(0), "mean": img.mean(0), "std": img.std(0),
                 "count": np.array([img.shape[0]]),
                 **{q: np.quantile(img, v, axis=0) for q, v in rstats.QUANTILES.items()},
             })}
        stats_list.append(s)
    ours = rstats.aggregate_stats(stats_list)
    theirs = lerobot_aggregate(stats_list)
    assert set(ours) == set(theirs)
    for key in ours:
        assert set(ours[key]) == set(theirs[key]), key
        for stat in ours[key]:
            a, b = ours[key][stat], theirs[key][stat]
            assert np.allclose(a, b, rtol=1e-12, atol=1e-12), (key, stat)
            assert ours[key][stat].shape == theirs[key][stat].shape


def test_aggregate_stats_matches_lerobot_semantics():
    a = {"x": rstats.feature_stats(np.array([[1.0, 2.0], [3.0, 4.0]]))}
    b = {"x": rstats.feature_stats(np.array([[5.0, 6.0]] * 4))}
    agg = rstats.aggregate_stats([a, b])
    assert agg["x"]["count"].tolist() == [6]
    assert np.allclose(agg["x"]["mean"], [(1 + 3 + 5 * 4) / 6, (2 + 4 + 6 * 4) / 6])
    assert np.allclose(agg["x"]["min"], [1, 2]) and np.allclose(agg["x"]["max"], [5, 6])
    assert set(agg["x"]) == set(rstats.STAT_KEYS)
    bad_image = {  # image stats must be (3,1,1) / (1,1,1): lerobot's validator refuses (3,)
        "mean": np.zeros(3), "std": np.zeros(3), "count": np.array([1]),
        "min": np.zeros(3), "max": np.zeros(3),
    }
    with pytest.raises(ValueError):
        rstats.aggregate_stats([{"observation.images.c": bad_image}])
    raw = {"mean": np.array([255.0, 0.0, 127.5]), "count": np.array([7])}
    img = rstats.video_stats_from_encoder(raw)
    assert img["mean"].shape == (3, 1, 1) and img["mean"][0, 0, 0] == 1.0
    assert img["count"].tolist() == [7]


def test_cli_exports_and_reports(tmp_path, capsys):
    from apollo_mavis_v2_runtime.tools.export_lerobot import main

    root = tmp_path / REPO
    record(root, [(4, 13)])
    assert main([REPO, "--root", str(tmp_path), "--no-validate"]) == 0
    assert "exported 1 episodes / 4 frames" in capsys.readouterr().out
    assert (root / "exports" / "lerobot_v3" / "meta" / "info.json").exists()
    assert main(["apollo/nope", "--root", str(tmp_path)]) == 2
