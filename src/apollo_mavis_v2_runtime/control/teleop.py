"""Teleop math: held keys -> twist -> integrated target pose (04-runtime §6).

``HeldState`` is defined in ``core.interfaces.teleop``; signs come from
``core.protocol.keymap.axis_map()`` — the single source of teleop signs.

Twist frame (binding, CHANGED 2026-09-08 at the operator's request):

* **Rotations** are about the **TCP axes** — unchanged, and self-consistent
  whatever the arm is doing (roll/pitch/yaw always turn the tool the same way
  relative to itself).
* **Translations** follow ``ControlConfig.translate_frame``, default
  ``"world"`` = the **operator** frame, fixed to the table: ``W`` moves away
  from the operator (−Y), ``A`` to the operator's left (+X), ``E`` up (+Z),
  whatever the tool is doing. (Operator decision 2026-09-08 evening; the
  morning's default had been ``"camera"`` and was superseded the same day.)
  ``"camera"`` = the active arm's **wrist-camera** frame: ``W`` along the
  camera's optical axis, ``A``/``D`` image left/right, ``E``/``Q`` image
  up/down — the keys follow the tool so they keep matching the wrist stream
  the operator is watching (the complaint that started the 2026-09-08 change:
  under the old base mapping a rotated tool made every key point somewhere
  else on screen, "the whole axis set is misaligned after I rotate the EE").
  ``"base"`` is the pre-2026-09-08 behaviour.

The recording frame (``SessionSpec.frames``) still never affects control math,
and the canonical policy action frame is still ``arm_base:<id>`` — this knob
moves the KEYBOARD only (``dagger/policy_runner.py`` is untouched).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from apollo_mavis_v2_core import Pose, Twist, se3
from apollo_mavis_v2_core.protocol import HELD_CODES, HELD_MODIFIER_ACTIONS, KEYMAP, axis_map

from ..config import TeleopRates

TranslateFrame = Literal["camera", "world", "base"]

# code -> (axis, sign), resolved once from the canonical keymap. Held
# modifiers (tracker_clutch) are not axes and never reach the twist.
_CODE_AXIS: dict[str, tuple[str, float]] = {}
for _e in KEYMAP:
    if _e.kind == "held" and _e.action not in HELD_MODIFIER_ACTIONS:
        _CODE_AXIS[_e.code] = axis_map()[_e.action]

_LINEAR = {"x": 0, "y": 1, "z": 2}
_ANGULAR = {"roll": 0, "pitch": 1, "yaw": 2}

# Key axes -> CAMERA axes. Columns are the images of the three KEY basis vectors
# (x forward, y left, z up) in the wrist camera's own frame. MuJoCo cameras (and
# the twin overlay that is pinned to the real D435i streams) look along their own
# −z with +y up and +x right, so:
_CAM_FROM_KEY = np.array(
    [
        [0.0, -1.0, 0.0],  # key forward -> camera −z (the optical axis)
        [0.0, 0.0, 1.0],  # key left    -> camera −x (image left)
        [-1.0, 0.0, 0.0],  # key up      -> camera +y (image up)
    ]
)

# Fallback for an arm with NO wrist camera (test scenes such as ``guardrail_env``;
# both MAVIS arms have one): the same "follow the tool" behaviour expressed in the
# TCP frame — forward along the tool axis, which is TCP +z for a gripper arm and a
# camera-only arm alike. Only the roll about that axis is arbitrary here.
_TOOL_FROM_KEY = np.array(
    [
        [0.0, 0.0, 1.0],  # key forward -> TCP +z (the flange / tool axis)
        [0.0, -1.0, 0.0],  # key left    -> TCP −y
        [1.0, 0.0, 0.0],  # key up      -> TCP +x
    ]
)

# Key axes -> WORLD axes for the operator frame (mavis_v2 scene header: z up,
# +Y = the long table edge the operator stands at, the operator's LEFT is +X).
_WORLD_FROM_KEY = np.array(
    [
        [0.0, 1.0, 0.0],  # key forward -> world −Y (away from the operator)
        [-1.0, 0.0, 0.0],  # key left    -> world +X (the operator's left)
        [0.0, 0.0, 1.0],  # key up      -> world +Z
    ]
)


def held_to_twist(held: frozenset[str], rates: TeleopRates) -> Twist:
    """Map a held-key set to a body-rate twist (m/s, rad/s) + rail/grip rates."""
    v = np.zeros(3)
    w = np.zeros(3)
    rail_v = 0.0
    grip_v = 0.0
    for code in held:
        if code not in HELD_CODES or code not in _CODE_AXIS:
            continue  # unbound key or held modifier (13-tracker §3.3)
        axis, sign = _CODE_AXIS[code]
        if axis in _LINEAR:
            v[_LINEAR[axis]] += sign * rates.linear_mps
        elif axis in _ANGULAR:
            w[_ANGULAR[axis]] += sign * rates.angular_rps
        elif axis == "rail":
            rail_v += sign * rates.rail_mps
        elif axis == "gripper":
            grip_v += sign * rates.gripper_frac_ps
    return Twist(v=v, w=w, rail_v=rail_v, grip_v=grip_v)


def twist_to_control_frame(
    tw: Twist,
    base_quat: np.ndarray,
    tcp_quat: np.ndarray,
    frame: TranslateFrame = "world",
    cam_quat: np.ndarray | None = None,
) -> Twist:
    """Express the keyboard twist in the WORLD frame (04-runtime §6).

    Rotations are always about the current TCP axes. Translations depend on
    ``frame``:

    ``"world"`` (default since 2026-09-08 evening)
        the operator frame, fixed to the table: ``W`` away from the operator
        (−Y), ``A`` to the operator's left (+X), ``E`` up (+Z). Never follows
        the tool.
    ``"camera"`` (the 2026-09-08 morning default, superseded the same day)
        the active arm's wrist-camera frame — ``W`` along the optical axis,
        ``A``/``D`` image left/right, ``E``/``Q`` image up/down. Follows the
        tool, so the keys keep matching the wrist stream after any rotation.
        Needs ``cam_quat`` (the camera's WORLD orientation, from
        ``SceneKinematics.wrist_cam_quat_world``): the camera-to-TCP rotation is
        NOT the same on both arms — the gripper base is mounted 180° about the
        tool axis, so the gripper arm's TCP frame is that much turned against
        the camera-only arm's. Without a wrist camera (``cam_quat is None``) the
        keys fall back to the TCP frame with the same forward axis.
    ``"base"``
        pre-2026-09-08 behaviour: the arm's ``link_base`` axes. Fixed as the
        rail slides, but yawed 180° against the operator's view in this cell
        (``base_quat`` is a +90° z rotation and joint 1 = π), which is why the
        keys read reversed.
    """
    if frame == "camera":
        if cam_quat is None:
            v_world = se3.quat_rotate(tcp_quat, _TOOL_FROM_KEY @ tw.v)
        else:
            v_world = se3.quat_rotate(cam_quat, _CAM_FROM_KEY @ tw.v)
    elif frame == "world":
        v_world = _WORLD_FROM_KEY @ tw.v
    elif frame == "base":
        v_world = se3.quat_rotate(base_quat, tw.v)
    else:  # pragma: no cover - config Literal already refuses anything else
        raise ValueError(f"unknown translate frame {frame!r}")
    w_world = se3.quat_rotate(tcp_quat, tw.w)
    return Twist(v=v_world, w=w_world, rail_v=tw.rail_v, grip_v=tw.grip_v)


class TargetIntegrator:
    """Per-arm integrated TCP target pose with leash clamping.

    ``target <- target ⊕ twist*dt``, then clamp to a leash around the
    measured TCP pose (bounds IK divergence; keeps per-tick motion far
    below the firmware 10 mm/step limit). ``reanchor`` freezes the target
    back to an achieved/measured pose (glide, don't wind up).
    """

    def __init__(self, leash_pos_m: float = 0.025, leash_rot_rad: float = 0.2) -> None:
        self.leash_pos_m = float(leash_pos_m)
        self.leash_rot_rad = float(leash_rot_rad)
        self._target: dict[str, Pose] = {}

    def seed(self, arm_id: str, pose: Pose) -> None:
        self._target[arm_id] = pose

    def get(self, arm_id: str) -> Pose | None:
        return self._target.get(arm_id)

    def reanchor(self, arm_id: str, pose: Pose) -> None:
        self._target[arm_id] = pose

    def step(self, arm_id: str, twist: Twist, dt: float, measured: Pose) -> Pose:
        target = self._target.get(arm_id, measured)
        target = se3.integrate_twist(target, twist, dt)
        target = se3.clamp_pose_to_leash(target, measured, self.leash_pos_m, self.leash_rot_rad)
        self._target[arm_id] = target
        return target


__all__ = [
    "TranslateFrame",
    "held_to_twist",
    "twist_to_control_frame",
    "TargetIntegrator",
]
