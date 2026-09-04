"""Feature-schema builders for LeRobot v3 datasets (10-frames §6-§8, binding).

Every recording mode shares this schema so plain-collect and DAgger datasets
stay merge-compatible. Per-dim names carry the ``<arm_id>_`` prefix; blocks
concatenate in ``WorkcellConfig.arms`` order; the authoritative convention
(action space + frames map) lives in the per-feature ``info`` dicts — the
repo name is a human-readable mirror only (§8.1).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from apollo_mavis_v2_core import CommandSource, FrameRef, parse_frame, se3

APOLLO_SCHEMA_VERSION = 1
RAIL_AXIS = "y"  # rail translates along +Y of the rail-origin frame (10-frames §2.2)

ActionSpace = Literal["delta_ee", "abs_ee", "joint"]

# int8 codes mirror core CommandSource string values (10-frames §7.3, binding).
# 2 (joint_jog) and 4 (planner) are reserved: they NEVER appear in recorded
# frames (joint_target is nacked while recording; planner motion not recorded).
ACTION_SOURCE_LABELS: dict[str, str] = {
    "0": CommandSource.POLICY.value,
    "1": CommandSource.TELEOP.value,
    "2": CommandSource.JOINT_JOG.value,
    "3": CommandSource.TAKEOVER.value,
    "4": CommandSource.PLANNER.value,
}

_DELTA_EE_DIMS = ["ee.dx", "ee.dy", "ee.dz", "ee.drx", "ee.dry", "ee.drz", "gripper.pos"]
_ABS_EE_DIMS = ["ee.x", "ee.y", "ee.z", "ee.qw", "ee.qx", "ee.qy", "ee.qz", "gripper.pos"]
_JOINT_DIMS = [f"joint{i}.pos" for i in range(1, 8)] + ["gripper.pos"]
_STATE_EE_DIMS = ["ee.x", "ee.y", "ee.z", "ee.qw", "ee.qx", "ee.qy", "ee.qz"]


@dataclass(frozen=True)
class ArmMeta:
    """The slice of arm identity the schema needs (id + rail presence)."""

    arm_id: str
    has_rail: bool


def arm_action_names(arm_id: str, has_rail: bool, action_space: ActionSpace) -> list[str]:
    """Per-arm action block dim names (10-frames §6 layouts; rail dim last)."""
    if action_space == "delta_ee":
        dims = list(_DELTA_EE_DIMS) + (["rail.dpos"] if has_rail else [])
    elif action_space == "abs_ee":
        dims = list(_ABS_EE_DIMS) + (["rail.pos"] if has_rail else [])
    elif action_space == "joint":
        dims = list(_JOINT_DIMS) + (["rail.pos"] if has_rail else [])
    else:  # pragma: no cover - Literal guards this
        raise ValueError(f"unknown action_space {action_space!r}")
    return [f"{arm_id}_{d}" for d in dims]


def arm_state_names(arm_id: str, has_rail: bool) -> list[str]:
    """Per-arm observation.state block (10-frames §6.1): joints, gripper,
    rail, then the measured TCP pose in the arm's recording frame."""
    dims = [f"joint{i}.pos" for i in range(1, 8)] + ["gripper.pos"]
    if has_rail:
        dims.append("rail.pos")
    dims += _STATE_EE_DIMS
    return [f"{arm_id}_{d}" for d in dims]


def build_features(
    arms: list[ArmMeta],
    frames: dict[str, FrameRef],
    cameras: dict[str, tuple[int, int]],  # camera_id -> (width, height)
    action_space: ActionSpace = "delta_ee",
) -> dict[str, dict]:
    """The full LeRobot features dict (10-frames §7; 04-runtime §10.2).

    The five lerobot bookkeeping features are auto-added by the library —
    never included here or in ``add_frame`` dicts.
    """
    if not arms:
        raise ValueError("need at least one arm")
    action_names: list[str] = []
    state_names: list[str] = []
    for arm in arms:
        if arm.arm_id not in frames:
            raise ValueError(f"frames map missing arm {arm.arm_id!r}")
        action_names += arm_action_names(arm.arm_id, arm.has_rail, action_space)
        state_names += arm_state_names(arm.arm_id, arm.has_rail)
    frames_map = {a.arm_id: str(frames[a.arm_id]) for a in arms}
    rail_info = {
        "axis": RAIL_AXIS,
        "travel_m": se3.RAIL_TRAVEL_M,
        "arms": [a.arm_id for a in arms if a.has_rail],
    }
    features: dict[str, dict] = {
        "action": {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": action_names,
            "info": {
                "apollo_schema": APOLLO_SCHEMA_VERSION,
                "action_space": action_space,
                "frames": frames_map,
                "rail": rail_info,
            },
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": state_names,
            "info": {
                "apollo_schema": APOLLO_SCHEMA_VERSION,
                "frames": dict(frames_map),
            },
        },
    }
    for cam_id, (width, height) in cameras.items():
        if "/" in cam_id:
            raise ValueError(f"camera id {cam_id!r} must not contain '/'")
        features[f"observation.images.{cam_id}"] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    # Always-present columns (10-frames §7.3, binding — merge compatibility).
    features["intervention"] = {"dtype": "bool", "shape": (1,), "names": None}
    features["action_source"] = {
        "dtype": "int8",
        "shape": (1,),
        "names": None,
        "info": {"labels": dict(ACTION_SOURCE_LABELS)},
    }
    features["wallclock_ns"] = {"dtype": "int64", "shape": (1,), "names": None}
    return features


_SPACE_TOKEN = {"delta_ee": "dee", "abs_ee": "aee", "joint": "jnt"}


def _frame_token(arm_id: str, ref: FrameRef) -> str:
    parsed = parse_frame(ref)
    if parsed.kind == "world":
        return "world"
    if parsed.kind == "arm_base":
        if parsed.ident != arm_id:
            raise ValueError(f"arm {arm_id!r} may not record in {ref!r} (10-frames §5.1)")
        return "base"
    if parsed.kind == "camera":
        return f"cam.{parsed.ident}"
    raise ValueError(f"{ref!r} is not a recording frame")


def slug(text: str) -> str:
    """Lowercase task token for repo ids (name is a mirror, not authority)."""
    token = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return token or "task"


def build_repo_id(
    task: str,
    arms: list[ArmMeta],
    frames: dict[str, FrameRef],
    action_space: ActionSpace = "delta_ee",
) -> str:
    """``apollo/xarm7_{task}_{n}arm_{conv}`` (10-frames §8.1 grammar)."""
    conv = _SPACE_TOKEN[action_space] + "-" + "+".join(
        _frame_token(a.arm_id, frames[a.arm_id]) for a in arms
    )
    return f"apollo/xarm7_{slug(task)}_{len(arms)}arm_{conv}"


def build_robot_type(n_arms: int, sim: bool) -> str:
    """``xarm7_{n}arm_rail`` (+ ``_mujoco`` for sim) — 10-frames §7.5."""
    return f"xarm7_{n_arms}arm_rail" + ("_mujoco" if sim else "")


__all__ = [
    "APOLLO_SCHEMA_VERSION",
    "ACTION_SOURCE_LABELS",
    "RAIL_AXIS",
    "ActionSpace",
    "ArmMeta",
    "arm_action_names",
    "arm_state_names",
    "build_features",
    "build_repo_id",
    "build_robot_type",
    "slug",
]
