"""Teleop math: held keys -> twist -> integrated target pose (04-runtime §6).

``HeldState`` is defined in ``core.interfaces.teleop``; signs come from
``core.protocol.keymap.axis_map()`` — the single source of teleop signs.
Twist frame (binding): translations along the arm's BASE axes, rotations
about the TCP axes; the recording frame never affects control math.
"""

from __future__ import annotations

import numpy as np
from apollo_mavis_v2_core import Pose, Twist, se3
from apollo_mavis_v2_core.protocol import HELD_CODES, HELD_MODIFIER_ACTIONS, KEYMAP, axis_map

from ..config import TeleopRates

# code -> (axis, sign), resolved once from the canonical keymap. Held
# modifiers (tracker_clutch) are not axes and never reach the twist.
_CODE_AXIS: dict[str, tuple[str, float]] = {}
for _e in KEYMAP:
    if _e.kind == "held" and _e.action not in HELD_MODIFIER_ACTIONS:
        _CODE_AXIS[_e.code] = axis_map()[_e.action]

_LINEAR = {"x": 0, "y": 1, "z": 2}
_ANGULAR = {"roll": 0, "pitch": 1, "yaw": 2}


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


def twist_to_control_frame(tw: Twist, base_quat: np.ndarray, tcp_quat: np.ndarray) -> Twist:
    """Express the keyboard twist in the WORLD frame per the control-frame
    convention: translations along the arm-base axes, rotations about the
    current TCP axes (04-runtime §6)."""
    v_world = se3.quat_rotate(base_quat, tw.v)
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
        target = se3.clamp_pose_to_leash(
            target, measured, self.leash_pos_m, self.leash_rot_rad
        )
        self._target[arm_id] = target
        return target


__all__ = ["held_to_twist", "twist_to_control_frame", "TargetIntegrator"]
