"""LeRobotEpisodeRecorder over the REAL lerobot writer: save/discard/finalize
round-trip, recorder_state.json lifecycle, crash repair (04-runtime §10.4)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from apollo_mavis_v2_runtime.config import RecorderConfig
from apollo_mavis_v2_runtime.recorder.episode_recorder import (
    DEFAULT_PROBE_SIZE,
    STATE_FILENAME,
    LeRobotEpisodeRecorder,
    _encoder_opens,
    codec_family,
    encoder_extra_options,
    make_rgb_encoder,
    min_video_frame_size,
    repair_unfinalized_datasets,
    resolve_vcodec,
    resolve_vcodec_for_dataset,
    stored_video_codec,
)
from apollo_mavis_v2_runtime.recorder.features import ArmMeta, build_features

REPO_ID = "apollo/xarm7_test_1arm_dee-base"
FRAMES = {"arm0": "arm_base:arm0"}
CAM = "cam0"
W, H = 64, 48


ROBOT_TYPE = "xarm7_1arm_rail_mujoco"
# lerobot's real streaming encoder must open NVENC with our options for the
# hardware-path tests; everywhere else the tiny 64x48 camera keeps CI cheap.
NVENC_OK = _encoder_opens("h264_nvenc", *DEFAULT_PROBE_SIZE)


def features(width: int = W, height: int = H):
    return build_features([ArmMeta("arm0", True)], FRAMES, {CAM: (width, height)})


def make_frame(i: int, width: int = W, height: int = H) -> dict:
    return {
        "action": np.full(8, 0.001 * i, dtype=np.float32),
        "observation.state": np.full(16, 0.01 * i, dtype=np.float32),
        f"observation.images.{CAM}": np.full((height, width, 3), i % 255, dtype=np.uint8),
        "intervention": np.array([False]),
        "action_source": np.array([1], dtype=np.int8),
        "wallclock_ns": np.array([1_000_000_000 + i * 40_000_000], dtype=np.int64),
    }


def make_recorder(root, cfg: RecorderConfig | None = None, size=(W, H)) -> LeRobotEpisodeRecorder:
    return LeRobotEpisodeRecorder(
        cfg or RecorderConfig(), features(*size), root, REPO_ID, ROBOT_TYPE, "test task"
    )


def record_episode(rec: LeRobotEpisodeRecorder, n: int, size=(W, H)) -> int:
    rec.start({})
    for i in range(n):
        rec.add_frame(make_frame(i, *size))
    return rec.save()


def video_info(root) -> dict:
    info = json.loads((Path(root) / "meta" / "info.json").read_text(encoding="utf-8"))
    return info["features"][f"observation.images.{CAM}"]["info"]


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
    # The probe must mirror lerobot's REAL open (same options + pix_fmt), so
    # whatever it picks has to open again with make_rgb_encoder's config --
    # re-probed UNCACHED (``__wrapped__``) or this would just re-read the hit.
    assert _encoder_opens.__wrapped__(codec, *DEFAULT_PROBE_SIZE)


def test_nvenc_extra_options_make_lerobot_short_gop_legal():
    # lerobot pins g=2; NVENC's default B-frames (3) violate "GOP > bf + 1"
    # and avcodec_open2 fails with EINVAL, so bf=0 must ride along.
    for codec in ("h264_nvenc", "hevc_nvenc"):
        opts = make_rgb_encoder(codec).get_codec_options(None, as_strings=True)
        assert opts["g"] == "2" and opts["bf"] == "0", opts
    assert encoder_extra_options("libsvtav1") == {}
    assert "bf" not in make_rgb_encoder("libsvtav1").get_codec_options(None, as_strings=True)
    with pytest.raises(ValueError):
        make_rgb_encoder("auto")


def test_min_video_frame_size_picks_smallest_stream():
    feats = build_features(
        [ArmMeta("arm0", True)], FRAMES, {"cam_big": (640, 480), "cam_small": (320, 240)}
    )
    assert min_video_frame_size(feats) == (320, 240)
    no_video = build_features([ArmMeta("arm0", True)], FRAMES, {})
    assert min_video_frame_size(no_video) == DEFAULT_PROBE_SIZE


def test_tiny_stream_never_gets_an_encoder_that_cannot_open():
    # Hardware encoders enforce a minimum frame size: a 16x16 stream must
    # either fall back to software or land on a codec that really opens.
    codec = resolve_vcodec("auto", (16, 16))
    assert codec == "libsvtav1" or _encoder_opens.__wrapped__(codec, 16, 16)


@pytest.mark.skipif(not NVENC_OK, reason="no usable NVENC encoder on this host")
def test_nvenc_records_through_lerobot_streaming_encoder(tmp_path):
    """The actual fix: lerobot's _CameraEncoderThread opens h264_nvenc with
    g=2 only because bf=0 rides along; frames must encode AND decode."""
    root = tmp_path / REPO_ID
    rec = make_recorder(root, RecorderConfig(vcodec="auto"), DEFAULT_PROBE_SIZE)
    assert rec.vcodec == "h264_nvenc"
    assert record_episode(rec, 6, DEFAULT_PROBE_SIZE) == 0
    rec.finalize()
    vinfo = video_info(root)
    assert vinfo["video.codec"] == "h264"
    assert vinfo["video.extra_options"] == {"bf": "0"}
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(REPO_ID, root=root)
    assert ds.num_frames == 6
    img = np.asarray(ds[3][f"observation.images.{CAM}"])
    assert img.size == DEFAULT_PROBE_SIZE[0] * DEFAULT_PROBE_SIZE[1] * 3


def test_resume_keeps_the_datasets_codec_family(tmp_path):
    """lerobot's offline merge refuses mixed-codec videos: a dataset recorded
    as av1 must stay av1 on resume even where 'auto' would now pick NVENC."""
    root = tmp_path / REPO_ID
    size = DEFAULT_PROBE_SIZE
    rec = make_recorder(root, RecorderConfig(vcodec="libsvtav1"), size)
    assert record_episode(rec, 4, size) == 0
    rec.finalize()
    assert stored_video_codec(root) == "av1"
    assert resolve_vcodec_for_dataset("auto", root, size) == "libsvtav1"
    resumed = make_recorder(root, RecorderConfig(vcodec="auto"), size)
    assert resumed.vcodec == "libsvtav1"
    assert record_episode(resumed, 3, size) == 1
    resumed.finalize()
    assert stored_video_codec(root) == "av1"
    # explicit codec of another family is rejected up front, same family passes
    with pytest.raises(ValueError, match="mix"):
        resolve_vcodec_for_dataset("h264_nvenc", root, size)
    assert resolve_vcodec_for_dataset("libsvtav1", root, size) == "libsvtav1"


def test_codec_family_and_fresh_root():
    assert codec_family("h264_nvenc") == "h264"
    assert codec_family("hevc_videotoolbox") == "hevc"
    assert codec_family("libsvtav1") == "av1"
    assert codec_family("libx264") == "h264"
    assert codec_family("bogus") is None
    assert stored_video_codec(Path("/nonexistent/apollo/xarm7")) is None
    # no dataset yet -> plain resolution
    assert resolve_vcodec_for_dataset("libsvtav1", Path("/nonexistent"), (64, 48)) == "libsvtav1"


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
