"""LeRobotEpisodeRecorder over the REAL lerobot writer: save/discard/finalize
round-trip, recorder_state.json lifecycle, crash repair (04-runtime §10.4)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from apollo_xarm7_runtime.config import RecorderConfig
from apollo_xarm7_runtime.recorder.episode_recorder import (
    STATE_FILENAME,
    LeRobotEpisodeRecorder,
    repair_unfinalized_datasets,
    resolve_vcodec,
)
from apollo_xarm7_runtime.recorder.features import ArmMeta, build_features

REPO_ID = "apollo/xarm7_test_1arm_dee-base"
FRAMES = {"arm0": "arm_base:arm0"}
CAM = "cam0"
W, H = 64, 48


def features():
    return build_features([ArmMeta("arm0", True)], FRAMES, {CAM: (W, H)})


def make_frame(i: int) -> dict:
    return {
        "action": np.full(8, 0.001 * i, dtype=np.float32),
        "observation.state": np.full(16, 0.01 * i, dtype=np.float32),
        f"observation.images.{CAM}": np.full((H, W, 3), i % 255, dtype=np.uint8),
        "intervention": np.array([False]),
        "action_source": np.array([1], dtype=np.int8),
        "wallclock_ns": np.array([1_000_000_000 + i * 40_000_000], dtype=np.int64),
    }


def make_recorder(root) -> LeRobotEpisodeRecorder:
    return LeRobotEpisodeRecorder(
        RecorderConfig(), features(), root, REPO_ID, "xarm7_1arm_rail_mujoco", "test task"
    )


@pytest.fixture(scope="module")
def saved_dataset(tmp_path_factory):
    """One recorder session: save ep0, discard one, save ep1, finalize."""
    root = tmp_path_factory.mktemp("ds") / REPO_ID
    rec = make_recorder(root)
    rec.start({"task": "test task"})
    for i in range(10):
        rec.add_frame(make_frame(i))
    assert rec.recording and rec.frames_in_buffer == 10
    assert rec.save() == 0
    assert not rec.recording
    rec.start({})
    for i in range(5):
        rec.add_frame(make_frame(i))
    rec.discard()  # cancels the streaming encoder; leaves no trace
    assert not rec.recording
    rec.start({})
    for i in range(6):
        rec.add_frame(make_frame(i))
    assert rec.save() == 1
    rec.finalize()
    rec.finalize()  # idempotent
    return root


def test_codebase_version_pinned():
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION

    assert CODEBASE_VERSION == "v3.0"


def test_vcodec_auto_resolves_to_usable_encoder():
    codec = resolve_vcodec("auto")
    assert codec != "auto"
    assert resolve_vcodec("libsvtav1") == "libsvtav1"  # explicit passthrough


def test_saved_dataset_reads_back(saved_dataset):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(REPO_ID, root=saved_dataset)
    assert ds.meta.info["codebase_version"] == "v3.0"
    assert ds.num_episodes == 2  # discarded episode leaves no trace
    assert ds.num_frames == 16
    assert ds.fps == 25
    feats = ds.meta.info["features"]
    assert feats["action"]["info"]["apollo_schema"] == 1
    assert feats["action"]["info"]["action_space"] == "delta_ee"
    assert feats["action"]["info"]["frames"] == FRAMES
    assert feats["action"]["info"]["rail"]["travel_m"] == 0.65
    assert feats["observation.state"]["info"]["frames"] == FRAMES
    assert feats["action_source"]["info"]["labels"] == {
        "0": "policy", "1": "teleop", "2": "joint_jog", "3": "takeover", "4": "planner",
    }
    row = ds[0]
    assert row["task"] == "test task"
    assert not bool(np.asarray(row["intervention"]).reshape(-1)[0])
    assert int(np.asarray(row["action_source"]).reshape(-1)[0]) == 1
    img = row[f"observation.images.{CAM}"]
    assert tuple(img.shape) in {(3, H, W), (H, W, 3)}  # decoded video frame


def test_recorder_state_lifecycle(saved_dataset):
    state = json.loads((saved_dataset / STATE_FILENAME).read_text())
    assert state == {"repo_id": REPO_ID, "episodes_saved": 2, "finalized": True}


def test_crash_repair_via_recorder_state(tmp_path):
    """Simulated death: no finalize -> startup repair makes it readable."""
    root = tmp_path / REPO_ID
    rec = make_recorder(root)
    rec.start({})
    for i in range(6):
        rec.add_frame(make_frame(i))
    rec.save()
    rec.start({})
    rec.add_frame(make_frame(0))
    del rec  # process dies: open episode buffer + no finalize
    state = json.loads((root / STATE_FILENAME).read_text())
    assert state["finalized"] is False
    repaired = repair_unfinalized_datasets(tmp_path)
    assert repaired == [REPO_ID]
    state = json.loads((root / STATE_FILENAME).read_text())
    assert state["finalized"] is True
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(REPO_ID, root=root)
    assert ds.num_episodes == 1  # the unsaved buffer is gone, saved data intact
    # second scan: nothing left to repair
    assert repair_unfinalized_datasets(tmp_path) == []


def test_resume_appends_to_existing_dataset(saved_dataset):
    rec = make_recorder(saved_dataset)  # resume path (meta/info.json exists)
    assert rec.episodes_saved == 2
    rec.start({})
    for i in range(4):
        rec.add_frame(make_frame(i))
    assert rec.save() == 2
    rec.finalize()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(REPO_ID, root=saved_dataset)
    assert ds.num_episodes == 3
