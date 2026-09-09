"""Feature schema / naming / repo-id grammar vs 10-frames §6-§8 (binding)."""

from __future__ import annotations

import pytest
from apollo_mavis_v2_core import CommandSource

from apollo_mavis_v2_runtime.recorder.features import (
    ACTION_SOURCE_LABELS,
    ArmMeta,
    arm_action_names,
    arm_state_names,
    build_features,
    build_repo_id,
    build_robot_type,
)

ARM0 = ArmMeta("arm0", True)
ARM1 = ArmMeta("arm1", True)
NORAIL = ArmMeta("fix0", False)


def test_delta_ee_block_order_and_prefix():
    names = arm_action_names("arm0", True, "delta_ee")
    assert names == [
        "arm0_ee.dx", "arm0_ee.dy", "arm0_ee.dz",
        "arm0_ee.drx", "arm0_ee.dry", "arm0_ee.drz",
        "arm0_gripper.pos", "arm0_rail.dpos",
    ]
    assert arm_action_names("fix0", False, "delta_ee")[-1] == "fix0_gripper.pos"


def test_abs_ee_and_joint_blocks():
    assert arm_action_names("a", True, "abs_ee") == [
        "a_ee.x", "a_ee.y", "a_ee.z", "a_ee.qw", "a_ee.qx", "a_ee.qy", "a_ee.qz",
        "a_gripper.pos", "a_rail.pos",
    ]
    joint = arm_action_names("a", True, "joint")
    assert joint[:7] == [f"a_joint{i}.pos" for i in range(1, 8)]
    assert joint[7:] == ["a_gripper.pos", "a_rail.pos"]
    # Dims derive from the layout listings (names are authoritative; the §6
    # table's "10/9" for abs_ee miscounts its own 9-name layout).
    assert len(arm_action_names("a", True, "abs_ee")) == 9
    assert len(arm_action_names("a", False, "abs_ee")) == 8
    assert len(arm_action_names("a", True, "joint")) == 9
    assert len(arm_action_names("a", False, "joint")) == 8


def test_state_block_16_or_15_dims():
    names = arm_state_names("arm0", True)
    assert len(names) == 16
    assert names[:7] == [f"arm0_joint{i}.pos" for i in range(1, 8)]
    assert names[7:9] == ["arm0_gripper.pos", "arm0_rail.pos"]  # rail last in block
    assert names[9:] == [
        "arm0_ee.x", "arm0_ee.y", "arm0_ee.z",
        "arm0_ee.qw", "arm0_ee.qx", "arm0_ee.qy", "arm0_ee.qz",
    ]
    assert len(arm_state_names("fix0", False)) == 15


def test_full_features_1arm(features_1arm=None):
    frames = {"arm0": "arm_base:arm0"}
    f = build_features([ARM0], frames, {"cam_front": (640, 480)})
    assert f["action"]["dtype"] == "float32" and f["action"]["shape"] == (8,)
    assert f["observation.state"]["shape"] == (16,)
    img = f["observation.images.cam_front"]
    assert img["dtype"] == "video" and img["shape"] == (480, 640, 3)
    assert img["names"] == ["height", "width", "channels"]
    # always-present columns (§7.3, verbatim shapes/dtypes)
    assert f["intervention"] == {"dtype": "bool", "shape": (1,), "names": None}
    assert f["wallclock_ns"] == {"dtype": "int64", "shape": (1,), "names": None}
    a_src = f["action_source"]
    assert a_src["dtype"] == "int8" and a_src["shape"] == (1,)
    # info dicts: authoritative convention
    info = f["action"]["info"]
    assert info["apollo_schema"] == 1
    assert info["action_space"] == "delta_ee"
    assert info["frames"] == frames
    assert info["rail"] == {"axis": "y", "travel_m": 0.65, "arms": ["arm0"]}
    assert f["observation.state"]["info"]["frames"] == frames
    # bookkeeping features NEVER included (library adds them)
    for k in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        assert k not in f
    for name in f:
        assert "/" not in name


def test_action_source_labels_mirror_command_source():
    """Six labels, verbatim core CommandSource values (10-frames §7.3); 5 = gello is
    reserved (phase-15; 16-gello §12.2: GELLO records nothing in v1)."""
    assert ACTION_SOURCE_LABELS == {
        "0": "policy", "1": "teleop", "2": "joint_jog", "3": "takeover", "4": "planner",
        "5": "gello",
    }
    assert set(ACTION_SOURCE_LABELS.values()) == {s.value for s in CommandSource}


def test_multi_arm_concatenation_order():
    frames = {"arm0": "arm_base:arm0", "arm1": "camera:cam_env"}
    f = build_features([ARM0, ARM1], frames, {}, "delta_ee")
    assert f["action"]["shape"] == (16,)
    assert f["action"]["names"][0].startswith("arm0_")
    assert f["action"]["names"][8].startswith("arm1_")
    assert f["observation.state"]["shape"] == (32,)
    # mixed rail: dims derive from names, never arm count
    f2 = build_features(
        [ARM0, NORAIL], {"arm0": "world", "fix0": "world"}, {}, "delta_ee"
    )
    assert f2["action"]["shape"] == (15,)
    assert f2["action"]["info"]["rail"]["arms"] == ["arm0"]


def test_repo_id_grammar():
    assert build_repo_id(
        "handover", [ARM0, ARM1], {"arm0": "arm_base:arm0", "arm1": "camera:cam_env"}
    ) == "apollo/xarm7_handover_2arm_dee-base+cam.cam_env"
    assert build_repo_id("Wipe Table!", [ARM0], {"arm0": "world"}, "abs_ee") == (
        "apollo/xarm7_wipe_table_1arm_aee-world"
    )
    assert build_repo_id("t", [ARM0], {"arm0": "arm_base:arm0"}, "joint").endswith(
        "_1arm_jnt-base"
    )
    with pytest.raises(ValueError):  # other arm's base frame is not a valid token
        build_repo_id("t", [ARM0], {"arm0": "arm_base:arm1"})


def test_robot_type():
    assert build_robot_type(1, sim=True) == "xarm7_1arm_rail_mujoco"
    assert build_robot_type(3, sim=False) == "xarm7_3arm_rail"


def test_build_features_requires_frames_for_every_arm():
    with pytest.raises(ValueError, match="frames map missing"):
        build_features([ARM0], {}, {})
