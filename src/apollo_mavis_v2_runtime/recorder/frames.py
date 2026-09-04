"""Record-time SE3 frame conversion (10-frames §3, binding math).

Drivers report EE poses in ``arm_base:<id>``; the recorder converts the
``ee.*`` dims of ``observation.state`` and the EE part of ``action`` into the
arm's declared recording frame exactly once, per frame, before ``add_frame``
(04-runtime §10.3). All math is ``apollo_mavis_v2_core.se3`` primitives.

Key facts encoded here:

- Poses convert with the full transform ``T_F_B = (T_W_F)^-1 ⊕ T_W_B`` —
  recomputed per frame for railed arms (T_W_B translates along the rail).
- Deltas transform as FREE VECTORS: only the rotation ``R(q_F_B)`` applies
  (10-frames §3.3) — constant even for railed arms.
- Gripper/rail/joint dims are frame-invariant scalars (§3.4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from apollo_mavis_v2_core import FrameRef, Pose, parse_frame, se3


def pose_to_frame(pose_b: Pose, t_w_b: Pose, t_w_f: Pose | None) -> Pose:
    """Re-express a base-frame pose in frame F (10-frames §3.1).

    ``t_w_f is None`` means F == world. For F == arm_base pass identity math
    by calling with ``t_w_f == t_w_b``.
    """
    pose_w = se3.pose_mul(t_w_b, pose_b)
    if t_w_f is None:
        return pose_w
    return se3.pose_mul(se3.pose_inv(t_w_f), pose_w)


def pose_from_frame(pose_f: Pose, t_w_b: Pose, t_w_f: Pose | None) -> Pose:
    """Inverse of :func:`pose_to_frame`: frame F -> arm_base."""
    pose_w = pose_f if t_w_f is None else se3.pose_mul(t_w_f, pose_f)
    return se3.pose_mul(se3.pose_inv(t_w_b), pose_w)


def delta_rotation(q_w_b: np.ndarray, q_w_f: np.ndarray | None) -> np.ndarray:
    """Quaternion ``q_F_B`` that rotates base-frame deltas into frame F."""
    if q_w_f is None:
        return se3.quat_normalize(np.asarray(q_w_b, dtype=np.float64))
    return se3.quat_normalize(se3.quat_mul(se3.quat_conj(q_w_f), q_w_b))


def delta_to_frame(
    dp_b: np.ndarray,
    dr_b: np.ndarray,
    q_w_b: np.ndarray,
    q_w_f: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a base-frame delta (δp, δr rotvec) into frame F (§3.3).

    Space-frame increments rotate and nothing else: translations of the
    frame cancel; rail travel never touches delta conversion.
    """
    q = delta_rotation(q_w_b, q_w_f)
    return se3.quat_rotate(q, dp_b), se3.quat_rotate(q, dr_b)


def delta_from_frame(
    dp_f: np.ndarray,
    dr_f: np.ndarray,
    q_w_b: np.ndarray,
    q_w_f: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`delta_to_frame`."""
    q = se3.quat_conj(delta_rotation(q_w_b, q_w_f))
    return se3.quat_rotate(q, dp_f), se3.quat_rotate(q, dr_f)


@dataclass(frozen=True)
class ArmFrameContext:
    """Static per-arm conversion context resolved at session bring-up."""

    arm_id: str
    frame: FrameRef  # "arm_base:<own id>" | "world" | "camera:<id>"
    kind: str  # parsed kind: arm_base | world | camera
    t_w_f: Pose | None  # camera frame pose (static, validated); None otherwise


class RecordingFrameConverter:
    """Per-arm converter into the session's declared recording frames.

    ``camera_poses`` maps camera_id -> static ``T_W_C`` (OpenCV convention);
    only cameras actually used as recording frames need an entry. Per-frame
    time-varying inputs (``t_w_b``) come from FK at each recorded frame.
    """

    def __init__(
        self,
        frames: dict[str, FrameRef],
        camera_poses: dict[str, Pose] | None = None,
    ) -> None:
        camera_poses = camera_poses or {}
        self._ctx: dict[str, ArmFrameContext] = {}
        for arm_id, ref in frames.items():
            parsed = parse_frame(ref)
            if parsed.kind == "arm_base":
                if parsed.ident != arm_id:
                    raise ValueError(
                        f"frames[{arm_id!r}] = {ref!r}: recording an arm in another "
                        "arm's base frame is disallowed (10-frames §5.1)"
                    )
                t_w_f = None
            elif parsed.kind == "world":
                t_w_f = None
            elif parsed.kind == "camera":
                if parsed.ident not in camera_poses:
                    raise ValueError(
                        f"frames[{arm_id!r}] = {ref!r}: no static T_W_C available "
                        "for that camera (10-frames §2.3)"
                    )
                t_w_f = camera_poses[parsed.ident]
            else:
                raise ValueError(f"frames[{arm_id!r}] = {ref!r} is not a recording frame")
            self._ctx[arm_id] = ArmFrameContext(arm_id, ref, parsed.kind, t_w_f)

    def frame_of(self, arm_id: str) -> FrameRef:
        return self._ctx[arm_id].frame

    def convert_pose(self, arm_id: str, pose_b: Pose, t_w_b: Pose) -> Pose:
        """Base-frame pose -> recording frame (canonical w >= 0 quat)."""
        ctx = self._ctx[arm_id]
        if ctx.kind == "arm_base":
            out = pose_b
        else:
            out = pose_to_frame(pose_b, t_w_b, ctx.t_w_f)
        return Pose(out.position, se3.quat_normalize(out.orientation))

    def convert_delta(
        self, arm_id: str, dp_b: np.ndarray, dr_b: np.ndarray, q_w_b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Base-frame delta -> recording frame (rotation only, §3.3)."""
        ctx = self._ctx[arm_id]
        if ctx.kind == "arm_base":
            return np.asarray(dp_b, dtype=np.float64), np.asarray(dr_b, dtype=np.float64)
        q_w_f = None if ctx.kind == "world" else ctx.t_w_f.orientation
        return delta_to_frame(dp_b, dr_b, q_w_b, q_w_f)


__all__ = [
    "pose_to_frame",
    "pose_from_frame",
    "delta_rotation",
    "delta_to_frame",
    "delta_from_frame",
    "ArmFrameContext",
    "RecordingFrameConverter",
]
