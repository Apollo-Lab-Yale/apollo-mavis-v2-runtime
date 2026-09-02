"""Tracker target provider for the teleop step (13-tracker §1/§4).

"Lifted mouse" model: while the clutch is held, the tracker's displacement
since engagement (filtered, scaled, yaw-aligned) is applied to the EE target
that was current at engagement. Released / stale / invalid / arm switched =>
the anchors clear and the loop holds last. The provider only supplies the
*target pose*; leash, IK, residual handling, rail, ``dq_max``, the gate and
the senders stay in :class:`~apollo_xarm7_runtime.control.loop.ControlLoop`.

Pose filter (§4 "Pose filter"): the aligned sample pose passes through a One
Euro :class:`PoseFilter` at the 100 Hz tick before the anchor/delta math; the
filter resets on engage and after stale/invalid gaps, free-runs while the
clutch is up (so telemetry ``pose_filtered`` is live), and is retuned from the
live settings (``tracker_settings``). Anchor rules (§4 "Anchor and re-seed
rules"): the settings are snapshotted at engagement and a change while engaged
re-anchors (``A_trk <- filtered(sample, new)``, ``A_ee <- current target``)
instead of re-interpreting the accumulated offset; rotation anchor slip is
applied in the BODY frame so ``D ⊗ A_ee'.q == achieved.q`` exactly.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from apollo_xarm7_core import LatestSlot, Pose, se3

from ..devices.tracker import CLUTCH_CODE
from .pose_filter import PoseFilter, PoseFilterConfig

if TYPE_CHECKING:
    from ..devices.tracker import TrackerSample, TrackerSettings, TrackerSettingsValues

TRACKER_CLUTCH_CODE: str = CLUTCH_CODE  # keymap code of ``tracker_clutch`` (KeyC)
_EPS = 1e-12
_IDENTITY_Q = np.array([1.0, 0.0, 0.0, 0.0])


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


def filter_config_from_settings(
    base: PoseFilterConfig, s: TrackerSettingsValues
) -> PoseFilterConfig:
    """Static filter config (``d_cutoff_hz``, deadbands) + the live-tunable
    fields from the settings snapshot."""
    return replace(
        base, enabled=s.filter_enabled, min_cutoff_hz=s.filter_min_cutoff_hz, beta=s.filter_beta
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
        filter_cfg: PoseFilterConfig | None = None,
    ) -> None:
        self.slot = slot
        self.settings = settings
        self.stale_s = float(stale_s)
        self.leash_pos_m = float(leash_pos_m)
        self.leash_rot_rad = float(leash_rot_rad)
        self.clutch = False  # tracker_clutch held this tick (telemetry)
        self.engaged_arm: str | None = None
        self._a_trk: Pose | None = None  # filtered aligned tracker pose at engagement
        self._a_ee: Pose | None = None  # EE target at engagement (slips on truncation)
        self._target: Pose | None = None  # last target handed to IK (world)
        self._consulted = False  # target() ran this tick
        self._applied = settings.get()  # settings the filter/anchors were built with
        self.pose_filter = PoseFilter(
            filter_config_from_settings(filter_cfg or PoseFilterConfig(), self._applied)
        )
        self._pose_filtered: Pose | None = None  # newest filtered aligned pose (telemetry)

    @property
    def applied_settings(self) -> TrackerSettingsValues:
        """Settings snapshot currently in force (engagement snapshot while engaged)."""
        return self._applied

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

    # -- settings / filter ----------------------------------------------------------
    def _sync_settings(self) -> tuple[TrackerSettingsValues, bool]:
        """Adopt the live settings: retune the filter (a yaw change resets it —
        its state lives in the aligned frame). Returns ``(settings, changed)``."""
        s = self.settings.get()
        changed = s != self._applied
        if changed:
            self.pose_filter.retune(
                min_cutoff_hz=s.filter_min_cutoff_hz, beta=s.filter_beta, enabled=s.filter_enabled
            )
            if s.yaw_deg != self._applied.yaw_deg:
                self.pose_filter.reset()
            self._applied = s
        return s, changed

    def _filtered(self, sample: TrackerSample, s: TrackerSettingsValues, now: float) -> Pose:
        """Aligned sample pose through the One Euro filter at the tick time."""
        self._pose_filtered = self.pose_filter.step(align_pose(sample.pose, s.yaw_deg), now)
        return self._pose_filtered

    def _gap(self) -> None:
        """No fresh valid sample: anchors cleared, filter restarts on the next one."""
        self.release()
        self.pose_filter.reset()
        self._pose_filtered = None

    # -- anchors ----------------------------------------------------------------------
    def release(self) -> None:
        """Clutch released / stale / invalid / arm switched: hold-last, anchors cleared."""
        self.engaged_arm = None
        self._a_trk = None
        self._a_ee = None
        self._target = None

    def end_tick(self, now: float | None = None) -> None:
        """Called once per tick by the loop: a tick that never consulted the
        provider (plan/jog/fault or clutch up) clears the anchors; the filter
        keeps running on the fresh sample so ``pose_filtered`` stays live."""
        if not self._consulted:
            self.release()
            s, _ = self._sync_settings()  # live retune even without samples
            sample = self.fresh_sample(now) if now is not None else None
            if sample is None:
                self.pose_filter.reset()
                self._pose_filtered = None
            else:
                self._filtered(sample, s, now)
        self._consulted = False

    def target(self, arm_id: str, measured: Pose, anchor_ee: Pose, now: float) -> Pose | None:
        """Leash-clamped tracker-derived EE target for ``arm_id`` (world), or
        ``None`` (no fresh valid sample -> anchors cleared, loop holds).

        Engages (filter reset, anchors ``A_trk``/``A_ee``) on the first call,
        after a stale/invalid gap and after an arm switch. A settings change
        while engaged re-anchors at the current filtered pose / current target
        (zero delta this tick). When the leash truncates the raw target,
        ``A_ee`` slips by the truncated amount.
        """
        self._consulted = True
        sample = self.fresh_sample(now)
        if sample is None:
            self._gap()
            return None
        s, changed = self._sync_settings()
        engaging = self.engaged_arm != arm_id or self._a_trk is None or self._a_ee is None
        if engaging:
            self.pose_filter.reset()
        filtered = self._filtered(sample, s, now)
        if engaging:
            self.engaged_arm = arm_id
            self._a_trk = filtered
            self._a_ee = anchor_ee
        elif changed:  # rule (c): re-anchor, never re-interpret the offset
            self._a_trk = filtered
            self._a_ee = self._target if self._target is not None else anchor_ee
        dp = s.pos_scale * (filtered.position - self._a_trk.position)
        if s.follow_rotation:
            dq = se3.quat_mul(filtered.orientation, se3.quat_conj(self._a_trk.orientation))
        else:
            dq = _IDENTITY_Q
        raw = Pose(self._a_ee.position + dp, se3.quat_mul(dq, self._a_ee.orientation))
        clamped = se3.clamp_pose_to_leash(raw, measured, self.leash_pos_m, self.leash_rot_rad)
        self.slip(raw, clamped)
        self._target = clamped
        return clamped

    def slip(self, intended: Pose, achieved: Pose) -> None:
        """Shift ``A_ee`` by the truncation (leash / IK residual) so the
        hand<->arm offset stays consistent: position by ``achieved - intended``,
        orientation in the BODY frame (``dq_b = conj(intended.q) ⊗ achieved.q``,
        ``A_ee.q <- A_ee.q ⊗ dq_b``) so that ``D ⊗ A_ee'.q == achieved.q`` for
        the hand rotation ``D`` that produced ``intended`` (rule (b))."""
        if self._a_ee is None:
            return
        dpos = achieved.position - intended.position
        dq_b = se3.quat_mul(se3.quat_conj(intended.orientation), achieved.orientation)
        if float(np.linalg.norm(dpos)) < _EPS and abs(abs(dq_b[0]) - 1.0) < _EPS:
            return
        self._a_ee = Pose(self._a_ee.position + dpos, se3.quat_mul(self._a_ee.orientation, dq_b))
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
            "pose_filtered": self._pose_filtered,
        }


__all__ = [
    "TRACKER_CLUTCH_CODE",
    "TrackerTeleop",
    "align_pose",
    "filter_config_from_settings",
    "interp_pose",
    "yaw_quat",
]
