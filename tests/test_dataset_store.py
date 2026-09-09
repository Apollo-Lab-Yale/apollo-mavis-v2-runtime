"""DatasetStore (04-runtime §10.6; 10-frames §11.7): listing / episodes / deletion read
only manifest.json + episode.json and NEVER import lerobot (torch); the open
episode and legacy v3 trees are 409; deleting one episode touches nothing else."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from apollo_mavis_v2_core.protocol import DatasetInfo, EpisodeInfo

from apollo_mavis_v2_runtime.config import RecorderConfig
from apollo_mavis_v2_runtime.recorder.datasets import LEGACY_READ_ONLY, DatasetError, DatasetStore
from apollo_mavis_v2_runtime.recorder.episode_recorder import EpisodeDirRecorder
from apollo_mavis_v2_runtime.recorder.features import ArmMeta, build_features
from apollo_mavis_v2_runtime.recorder.manifest import read_manifest
from apollo_mavis_v2_runtime.recorder.sidecars import SidecarWriter

REPO = "apollo/store_test"
CAM = "cam0"
W, H = 64, 48


def _frame(i: int) -> dict:
    return {
        "action": np.full(8, 0.001 * i, dtype=np.float32),
        "observation.state": np.full(16, 0.01 * i, dtype=np.float32),
        f"observation.images.{CAM}": np.full((H, W, 3), i % 255, dtype=np.uint8),
        "intervention": np.array([False]),
        "action_source": np.array([1], dtype=np.int8),
        "wallclock_ns": np.array([1_700_000_000_000_000_000 + i * 40_000_000], dtype=np.int64),
    }


def record(root: Path, repo_id: str, lengths: list[int], task: str = "stack") -> list[str]:
    """Record ``lengths`` episodes with the REAL recorder (lerobot loads here, on purpose:
    the store must stay lerobot-free even when the module is already in sys.modules)."""
    feats = build_features([ArmMeta("arm0", True)], {"arm0": "arm_base:arm0"}, {CAM: (W, H)})
    rec = EpisodeDirRecorder(RecorderConfig(vcodec="libsvtav1"), feats, root, repo_id,
                             "xarm7_1arm_rail_mujoco", task)
    ids = []
    for n in lengths:
        rec.start({})
        for i in range(n):
            rec.add_frame(_frame(i))
        ids.append(rec.save({"session_id": "s1", "frames_dropped": 0}, None)[1])
        time.sleep(0.002)  # distinct millisecond ids
    rec.finalize()
    side = SidecarWriter(root)
    side.write_session("s1", "collect", {"task": task, "arms": ["arm0"]},
                       {"kind": "sim", "arm_ids": ["arm0"]}, {})
    side.archive_scene_xml("<mujoco/>")
    return ids


@pytest.fixture()
def no_lerobot(monkeypatch):
    """Any ``import lerobot...`` raises ImportError for the duration of the test,
    whatever earlier tests loaded (``sys.modules[name] = None`` is the documented
    way to make an import fail)."""
    for name in list(sys.modules):
        if name == "lerobot" or name.startswith("lerobot."):
            monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "lerobot", None)
    yield


@pytest.fixture()
def datasets(tmp_path):
    ids = record(tmp_path / REPO, REPO, [8, 5, 3])
    # a legacy phase-07 v3 tree next to it
    legacy = tmp_path / "apollo" / "old_v3"
    (legacy / "meta" / "apollo").mkdir(parents=True)
    (legacy / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v3.0", "fps": 25, "robot_type": "xarm7_1arm_rail", "total_episodes": 4,
        "total_frames": 100, "features": {f"observation.images.{CAM}": {"dtype": "video"}},
    }))
    (legacy / "meta" / "apollo" / "session_x.json").write_text(json.dumps({
        "spec": {"task": "legacy task"}, "workcell": {"kind": "hardware", "arm_ids": ["grip"]},
    }))
    return tmp_path, ids


def test_list_describe_episodes_without_lerobot(datasets, no_lerobot):
    root, ids = datasets
    store = DatasetStore(root)
    rows = store.list()
    assert [r.repo_id for r in rows] == [REPO, "apollo/old_v3"]  # newest first
    ds = rows[0]
    assert isinstance(ds, DatasetInfo) and ds.layout == "episode_dirs"
    assert (ds.total_episodes, ds.total_frames, ds.fps) == (3, 16, 25)
    assert ds.robot_type == "xarm7_1arm_rail_mujoco" and ds.kind == "sim" and ds.task == "stack"
    assert ds.cameras == [CAM] and ds.arms == ["arm0"] and ds.in_use is False
    assert ds.export is not None and ds.export.state == "none"
    legacy = rows[1]
    assert legacy.layout == "lerobot_v3" and legacy.total_episodes == 4
    assert legacy.kind == "hardware"
    assert legacy.task == "legacy task" and legacy.cameras == [CAM]
    assert store.describe("apollo/nope") is None
    eps = store.episodes(REPO)
    assert [e.episode_id for e in eps] == ids
    assert isinstance(eps[0], EpisodeInfo)
    assert [e.index for e in eps] == [0, 1, 2] and [e.frames for e in eps] == [8, 5, 3]
    assert eps[0].duration_s == pytest.approx(8 / 25) and eps[0].task == "stack"
    assert eps[0].session_id == "s1" and eps[0].recorded_at.endswith("Z")
    assert all(e.export_ok and not e.open and not e.audio for e in eps)
    assert store.episodes("apollo/old_v3") == []  # legacy: dataset level only
    with pytest.raises(DatasetError) as ei:
        store.episodes("apollo/nope")
    assert ei.value.not_found
    assert store.resolve("x") == "apollo/x" and store.resolve("ns/x") == "ns/x"
    assert store.layout_of(REPO) == "episode_dirs"
    assert store.layout_of("apollo/old_v3") == "lerobot_v3"
    assert store.layout_of("apollo/nope") is None and store.exists(REPO)
    assert "lerobot" not in sys.modules or sys.modules["lerobot"] is None


def test_delete_episode_touches_one_directory(datasets, no_lerobot):
    root, ids = datasets
    store = DatasetStore(root)
    keep = root / REPO / "episodes" / ids[1]
    mtimes = {p: p.stat().st_mtime_ns for p in keep.rglob("*")}
    store.delete_episode(REPO, ids[0])
    assert not (root / REPO / "episodes" / ids[0]).exists()
    assert {p: p.stat().st_mtime_ns for p in keep.rglob("*")} == mtimes  # untouched
    m = read_manifest(root / REPO)
    assert (m["episodes"], m["frames"]) == (2, 8)
    assert [e.episode_id for e in store.episodes(REPO)] == ids[1:]
    with pytest.raises(DatasetError) as ei:
        store.delete_episode(REPO, ids[0])
    assert ei.value.not_found
    with pytest.raises(DatasetError) as ei:
        store.delete_episode("apollo/nope", ids[1])
    assert ei.value.not_found
    with pytest.raises(DatasetError, match=LEGACY_READ_ONLY):
        store.delete_episode("apollo/old_v3", "x")
    # a path-ish id never escapes episodes/
    with pytest.raises(DatasetError):
        store.delete_episode(REPO, "../manifest.json")
    assert (root / REPO / "manifest.json").exists()


def test_delete_marks_the_export_stale(datasets, no_lerobot):
    root, ids = datasets
    m = read_manifest(root / REPO)
    m["last_export"] = {"format": "lerobot_v3", "path": "exports/lerobot_v3", "at": "x",
                        "episodes": 3, "stale": False, "error": None}
    from apollo_mavis_v2_runtime.recorder.manifest import write_manifest

    write_manifest(root / REPO, m)
    store = DatasetStore(root)
    assert store.describe(REPO).export.state == "fresh"
    store.delete_episode(REPO, ids[2])
    info = store.describe(REPO)
    assert info.export.state == "stale" and info.export.episodes == 3 and info.total_episodes == 2


def test_open_episode_and_in_use_are_protected(datasets, no_lerobot):
    root, ids = datasets
    store = DatasetStore(root)
    store.in_use_repo = lambda: REPO
    store.open_episode = lambda: "20261231T000000.000Z-abcdef"
    assert store.describe(REPO).in_use is True
    eps = store.episodes(REPO)
    assert eps[-1].open is True and eps[-1].episode_id == "20261231T000000.000Z-abcdef"
    with pytest.raises(DatasetError, match="being recorded"):
        store.delete_episode(REPO, "20261231T000000.000Z-abcdef")
    store.delete_episode(REPO, ids[0])  # other episodes CAN go during the session
    # ... unless the manager names a reason (the running Online DAgger session's rollouts:
    # its counters and the trainer's buffer are never told about a deletion)
    store.episode_delete_refusal = lambda repo_id: (
        f"dataset {repo_id!r} is in use by the running Online DAgger session" if repo_id == REPO
        else None
    )
    with pytest.raises(DatasetError, match="running Online DAgger session"):
        store.delete_episode(REPO, ids[1])
    assert (root / REPO / "episodes" / ids[1]).is_dir()
    store.episode_delete_refusal = lambda repo_id: None
    store.delete_episode(REPO, ids[1])
    with pytest.raises(DatasetError, match="in use"):
        store.delete_dataset(REPO)
    with pytest.raises(DatasetError, match="in use"):
        store.export(REPO)
    # the running session's .tmp-* survives the scan sweep; another dataset's does not
    live = root / REPO / "episodes" / ".tmp-20261231T000000.000Z-abcdef"
    live.mkdir()
    other = root / "apollo" / "other"  # a manifest-only dataset with a crashed episode
    from apollo_mavis_v2_runtime.recorder.manifest import new_manifest, write_manifest

    (other / "episodes").mkdir(parents=True)
    write_manifest(other, new_manifest("apollo/other", 25, "xarm7_1arm_rail_mujoco", {}, {}))
    crashed = other / "episodes" / ".tmp-crashed"
    crashed.mkdir()
    store.list()
    assert live.exists() and not crashed.exists()


def test_delete_dataset_and_legacy_refusals(datasets, no_lerobot):
    root, ids = datasets
    store = DatasetStore(root)
    with pytest.raises(DatasetError, match=LEGACY_READ_ONLY):
        store.delete_dataset("apollo/old_v3")
    with pytest.raises(DatasetError, match=LEGACY_READ_ONLY):
        store.export("apollo/old_v3")
    with pytest.raises(DatasetError) as ei:
        store.delete_dataset("apollo/nope")
    assert ei.value.not_found
    store.delete_dataset(REPO)
    assert not (root / REPO).exists()
    assert (root / "apollo").exists()  # namespace kept (the legacy tree is inside)
    assert [r.repo_id for r in store.list()] == ["apollo/old_v3"]


def test_unsupported_export_format_is_409(datasets, no_lerobot):
    root, _ = datasets
    with pytest.raises(DatasetError, match="unsupported"):
        DatasetStore(root).export(REPO, fmt="aloha_hdf5")


def test_store_module_imports_no_lerobot():
    """Import-time discipline: the REST path's modules never pull lerobot / torch."""
    import subprocess

    code = (
        "import sys; import apollo_mavis_v2_runtime.recorder.datasets, "
        "apollo_mavis_v2_runtime.recorder.manifest, apollo_mavis_v2_runtime.recorder.stats, "
        "apollo_mavis_v2_runtime.recorder.export_lerobot, apollo_mavis_v2_runtime.server.rest, "
        "apollo_mavis_v2_runtime.tools.export_lerobot, "
        "apollo_mavis_v2_runtime.recorder.action_filter; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] in ('lerobot', 'torch')))"
    )
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True,
                         env={**os.environ, "MUJOCO_GL": "egl"})
    assert out.stdout.strip() == "[]", out.stdout
