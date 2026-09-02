"""Tracker target provider for the teleop step (13-tracker §1/§4).

"Lifted mouse" model: while the clutch is held, the tracker's displacement
since engagement (scaled, yaw-aligned) is applied to the EE target that was
current at engagement. Released / stale / invalid / arm switched => the
anchors clear and the loop holds last. The provider only supplies the
*target pose*; leash, IK, residual handling, rail, ``dq_max``, the gate and
the senders stay in :class:`~apollo_xarm7_runtime.control.loop.ControlLoop`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from apollo_xarm7_core import LatestSlot, Pose, se3

from ..devices.tracker import CLUTCH_CODE

if TYPE_CHECKING:
    from ..devices.tracker import TrackerSample, TrackerSettings

TRACKER_CLUTCH_CODE: str = CLUTCH_CODE  # keymap code of ``tracker_clutch`` (KeyC)
_EPS = 1e-12


def yaw_quat(yaw_deg: float) -> np.ndarray:
    """``R_z(yaw_deg)`` as a wxyz quaternion."""
    return se3.rotvec_to_quat(np.array([0.0, 0.0, math.radians(yaw_deg)]))


def align_pose(pose: Pose, yaw_deg: float) -> Pose:
    """Lighthouse world -> MJCF world: rotate position and orientation by
    ``R_z(yaw_deg)`` (both worlds are z-up; only yaw is calibrated)."""
    q = yaw_quat(yaw_deg)
    return Pose(se3.quat_rotate(q, pose.position), se3.quat_mul(q, pose.orientation))


def interp_pose(a: Pose, b: Pose, s: float) -> Pose:
    """Pose between ``a`` (s=0) and ``b`` (s=1): lerp position, slerp orientation."""
    if s >= 1.0:
        return b
    if s <= 0.0:
        return a
    return Pose(
        a.position + (b.position - a.position) * s,
        se3.quat_slerp(a.orientation, b.orientation, s),
    )


class TrackerTeleop:
    """Per-session clutch/anchor state over the process-wide tracker slot."""

    def __init__(
        self,
        slot: LatestSlot[TrackerSample],
        settings: TrackerSettings,
        *,
        stale_s: float,
        leash_pos_m: float,
        leash_rot_rad: float,
    ) -> None:
        self.slot = slot
        self.settings = settings
        self.stale_s = float(stale_s)
        self.leash_pos_m = float(leash_pos_m)
        self.leash_rot_rad = float(leash_rot_rad)
        self.clutch = False  # tracker_clutch held this tick (telemetry)
        self.engaged_arm: str | None = None
        self._a_trk: Pose | None = None  # aligned tracker pose at engagement
        self._a_ee: Pose | None = None  # EE target at engagement (slips on truncation)
        self._target: Pose | None = None  # last target handed to IK (world)
        self._consulted = False  # target() ran this tick

    # -- samples ------------------------------------------------------------------
    def fresh_sample(self, now: float) -> TrackerSample | None:
        """Newest sample if ``now - rx_mono <= stale_s`` and valid, else None."""
        got = self.slot.get()
        if got is None:
            return None
        sample = got[0]
        if not sample.valid or now - sample.rx_mono > self.stale_s:
            return None
        return sample

    # -- anchors ----------------------------------------------------------------------
    def release(self) -> None:
        """Clutch released / stale / invalid / arm switched: hold-last, anchors cleared."""
        self.engaged_arm = None
        self._a_trk = None
        self._a_ee = None
        self._target = None

    def end_tick(self) -> None:
        """Called once per tick by the loop: a tick that never consulted the
        provider (plan/jog/fault or clutch up) clears the anchors."""
        if not self._consulted:
            self.release()
        self._consulted = False

    def target(self, arm_id: str, measured: Pose, anchor_ee: Pose, now: float) -> Pose | None:
        """Leash-clamped tracker-derived EE target for ``arm_id`` (world), or
        ``None`` (no fresh valid sample -> anchors cleared, loop holds).

        Engages (anchors ``A_trk``/``A_ee``) on the first call, after a
        stale/invalid gap and after an arm switch. When the leash truncates
        the raw target, ``A_ee`` slips by the truncated amount.
        """
        self._consulted = True
        sample = self.fresh_sample(now)
        if sample is None:
            self.release()
            return None
        s = self.settings.get()
        aligned = align_pose(sample.pose, s.yaw_deg)
        if self.engaged_arm != arm_id or self._a_trk is None or self._a_ee is None:
            self.engaged_arm = arm_id
            self._a_trk = aligned
            self._a_ee = anchor_ee
        dp = s.pos_scale * (aligned.position - self._a_trk.position)
        if s.follow_rotation:
            dq = se3.quat_mul(aligned.orientation, se3.quat_conj(self._a_trk.orientation))
        else:
            dq = np.array([1.0, 0.0, 0.0, 0.0])
        raw = Pose(self._a_ee.position + dp, se3.quat_mul(dq, self._a_ee.orientation))
        clamped = se3.clamp_pose_to_leash(raw, measured, self.leash_pos_m, self.leash_rot_rad)
        self.slip(raw, clamped)
        self._target = clamped
        return clamped

    def slip(self, intended: Pose, achieved: Pose) -> None:
        """Shift ``A_ee`` by ``achieved - intended`` (leash / IK residual
        truncation) so the hand<->arm offset stays consistent."""
        if self._a_ee is None:
            return
        dpos = achieved.position - intended.position
        dq = se3.quat_mul(achieved.orientation, se3.quat_conj(intended.orientation))
        if float(np.linalg.norm(dpos)) < _EPS and abs(dq[0] - 1.0) < _EPS:
            return
        self._a_ee = Pose(self._a_ee.position + dpos, se3.quat_mul(dq, self._a_ee.orientation))
        if self._target is not None:
            self._target = achieved

    def set_target(self, target: Pose | None) -> None:
        self._target = target

    # -- telemetry ----------------------------------------------------------------------
    def telemetry_extra(self) -> dict:
        """``session_extra["tracker"]`` payload (poses as core ``Pose``)."""
        return {
            "clutch": self.clutch,
            "engaged_arm": self.engaged_arm,
            "anchor_tcp": self._a_ee,
            "target_tcp": self._target if self.engaged_arm is not None else None,
        }


__all__ = ["TRACKER_CLUTCH_CODE", "TrackerTeleop", "align_pose", "interp_pose", "yaw_quat"]
