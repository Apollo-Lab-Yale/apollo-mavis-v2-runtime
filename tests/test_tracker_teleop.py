"""Tracker teleop provider over FakeWorkcell with a 6-DoF fake kin/IK
(13-tracker §4): engage/follow, release/re-engage, stale/invalid, arm switch,
yaw alignment, pos_scale, follow_rotation, leash anchor slip, watchdog scale,
keyboard interplay, switch_arm_prev and tracker_settings."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from apollo_xarm7_core import Command, HeldState, Pose, se3
from apollo_xarm7_core.testing import FakeArm, FakeWorkcell

from apollo_xarm7_runtime.bus import RuntimeBus
from apollo_xarm7_runtime.config import ControlConfig
from apollo_xarm7_runtime.control.loop import ControlLoop
from apollo_xarm7_runtime.control.tracker_teleop import (
    TRACKER_CLUTCH_CODE,
    TrackerTeleop,
    align_pose,
    interp_pose,
    yaw_quat,
)
from apollo_xarm7_runtime.devices.tracker import TrackerSample, TrackerSettings
from apollo_xarm7_runtime.safety.gate import NullGate
from apollo_xarm7_runtime.safety.supervisor import SafetySupervisor
from apollo_xarm7_runtime.safety.watchdog import InputWatchdog

IDENT = np.array([1.0, 0.0, 0.0, 0.0])
CLUTCH = TRACKER_CLUTCH_CODE  # "KeyC"
DT = 0.01


class PoseKin:
    """q[:3] = TCP position, q[3:6] = TCP rotation vector, both in world."""

    def tcp_world(self, arm_id, q):
        return Pose(np.array(q[:3], dtype=float), se3.rotvec_to_quat(np.array(q[3:6], float)))

    def base_quat_world(self, arm_id):
        return IDENT


class PoseIK:
    def solve(self, arm_id, target, q_last):
        q = np.array(q_last, dtype=float)
        q[:3] = target.position
        q[3:6] = se3.quat_to_rotvec(target.orientation)
        return SimpleNamespace(q=q, diverged=False, pos_err_m=0.0, rot_err_rad=0.0)

    def sync_passive(self, states):
        pass

    def reset(self, arm_id, q):
        pass


class Rig:
    """Deterministic ControlLoop + provider; the tracker slot is fed by hand.

    ``device(*codes)`` scripts the device-held codes (13-tracker §1.1) that
    every subsequent sample carries; ``gripper_arms`` mirrors the loop option.
    """

    def __init__(self, tracker: bool = True, gripper_arms=None, *, filter=False, filter_cfg=None):
        self.cell = FakeWorkcell(
            {"arm0": FakeArm("arm0", has_rail=True), "arm1": FakeArm("arm1")}
        )
        self.cell.start()
        self.bus = RuntimeBus()
        # The pose filter is OFF by default here so the anchor/delta math is
        # exact; filter tests opt in (``filter=True`` / a PoseFilterConfig).
        self.settings = TrackerSettings(filter_enabled=filter)
        self.tracker = TrackerTeleop(
            self.bus.tracker, self.settings, stale_s=0.2, leash_pos_m=0.025, leash_rot_rad=0.2,
            filter_cfg=filter_cfg,
        )
        self.t = 0.0
        self.key_seq = 0
        self.sample_seq = 0
        self.held: tuple[str, ...] = ()
        self.pose: tuple[np.ndarray, np.ndarray] | None = None  # re-published each tick
        self.codes: frozenset[str] = frozenset()  # device-held codes on every sample
        self.controller = None  # ControllerState echoed on every sample
        self.click_action = None  # action bound to the newest trackpad press edge
        self.loop = ControlLoop(
            self.cell, ControlConfig(), self.bus,
            SafetySupervisor(NullGate(), InputWatchdog()), ["arm0", "arm1"],
            ik=PoseIK(), kin=PoseKin(), tracker=self.tracker if tracker else None,
            gripper_arms=gripper_arms, clock=lambda: self.t,
        )
        self.loop._seed_from_measured()

    # -- inputs -----------------------------------------------------------------
    def sample(self, pos, quat=IDENT, *, valid=True, age=0.0, sticky=True, codes=None):
        self.sample_seq += 1
        codes = self.codes if codes is None else frozenset(codes)
        s = TrackerSample(
            Pose(np.asarray(pos, float), quat), np.zeros(3), np.zeros(3),
            self.t, self.t - age, self.sample_seq, valid, self.controller, codes,
            self.click_action,
        )
        self.bus.tracker.put(s)
        self.pose = (np.asarray(pos, float), np.asarray(quat, float)) if sticky else None

    def device(self, *codes, controller=None, click_action=None):
        """Script the controller-derived codes (and the action bound to the
        newest trackpad press, as the reader derives it); re-publishes the
        current pose at once (the reader does the same on a button edge)."""
        self.codes = frozenset(codes)
        self.controller = controller
        self.click_action = click_action
        if self.pose is not None:
            self.sample(*self.pose)

    def latch_ws(self):
        """Drive the WS InputWatchdog into AWAIT_EMPTY (scale 0) without
        touching the held slot (the last KeysMsg stays in force)."""
        self.loop.supervisor.watchdog.on_disconnect()

    def hold(self, *codes):
        self.held = codes
        self._keys()

    def _keys(self):
        self.key_seq += 1
        st = HeldState(frozenset(self.held), self.key_seq, self.t)
        self.loop.supervisor.watchdog.on_keys(st)
        self.bus.held_keys.put(st)

    def tick(self, n=1, *, heartbeat=True):
        for _ in range(n):
            self.t += DT
            if heartbeat:
                self._keys()  # 25 Hz heartbeat stand-in: watchdog never trips
            if self.pose is not None:
                self.sample(*self.pose)  # a 100 Hz device re-reports a still tracker
            self.loop.run_tick(self.t)
            for a in ("arm0", "arm1"):
                self.cell.arms[a].command_joints(self.loop._last_cmd[a])
            self.cell.step(DT)

    def act(self, op, args=None):
        fut = self.bus.commands.submit(Command(op=op, args=args or {}, source="ws"))
        self.tick()
        return fut.result(timeout=1.0)

    def cmd(self, arm="arm0") -> np.ndarray:
        return np.array(self.loop._last_cmd[arm])

    def extra(self) -> dict:
        return self.bus.snapshot.get()[0].session_extra["tracker"]


# -- pure math --------------------------------------------------------------------------
def test_align_pose_rotates_by_yaw_and_interp_pose_endpoints():
    p = Pose(np.array([0.1, 0.0, 0.2]), se3.rotvec_to_quat([0.3, 0.0, 0.0]))
    a = align_pose(p, 90.0)
    assert np.allclose(a.position, [0.0, 0.1, 0.2], atol=1e-12)
    # Frame change: orientation is left-multiplied by R_z(90) (q' = q_yaw * q).
    assert np.allclose(a.orientation, se3.quat_mul(yaw_quat(90.0), p.orientation), atol=1e-12)
    # The relative rotation between two aligned poses is the lighthouse-world
    # relative rotation conjugated into the MJCF world: R (q1 q0^-1) R^T.
    q0 = se3.rotvec_to_quat([0.3, 0.0, 0.0])
    q1 = se3.quat_mul(se3.rotvec_to_quat([0.1, 0.0, 0.0]), q0)  # extra +0.1 roll about x
    a0 = align_pose(Pose(np.zeros(3), q0), 90.0)
    a1 = align_pose(Pose(np.zeros(3), q1), 90.0)
    dq = se3.quat_mul(a1.orientation, se3.quat_conj(a0.orientation))
    assert np.allclose(se3.quat_to_rotvec(dq), [0.0, 0.1, 0.0], atol=1e-12)  # x -> y
    b = Pose(np.array([1.0, 0.0, 0.0]), se3.rotvec_to_quat([0.0, 0.0, 0.4]))
    assert interp_pose(p, b, 0.0) is p and interp_pose(p, b, 1.0) is b
    mid = interp_pose(Pose.identity(), b, 0.5)
    assert np.allclose(mid.position, [0.5, 0.0, 0.0])
    assert np.allclose(se3.quat_to_rotvec(mid.orientation), [0.0, 0.0, 0.2], atol=1e-9)


# -- engage / follow / release ------------------------------------------------------------
def test_engage_then_target_follows_delta_one_to_one():
    rig = Rig()
    rig.sample([0.0, 0.0, 0.0])
    rig.hold(CLUTCH)
    rig.tick()  # engage: A_trk = sample, A_ee = measured -> no motion
    assert np.allclose(rig.cmd()[:3], 0.0)
    ex = rig.extra()
    assert ex["clutch"] is True and ex["engaged_arm"] == "arm0"
    assert ex["anchor_tcp"] is not None and ex["target_tcp"] is not None
    rig.sample([0.01, 0.0, 0.0])
    rig.tick()
    assert np.allclose(rig.cmd()[:3], [0.01, 0.0, 0.0], atol=1e-12)
    rig.sample([0.02, 0.005, -0.01])
    rig.tick()
    assert np.allclose(rig.cmd()[:3], [0.02, 0.005, -0.01], atol=1e-12)
    assert np.allclose(rig.cmd()[3:6], 0.0, atol=1e-12)  # tracker orientation unchanged
    assert np.allclose(rig.cmd("arm1"), 0.0)  # non-active arm untouched


def test_release_holds_and_reengage_never_jumps():
    rig = Rig()
    rig.sample([0.0, 0.0, 0.0])
    rig.hold(CLUTCH)
    rig.tick()
    rig.sample([0.01, 0.0, 0.0])
    rig.tick()
    q_hold = rig.cmd()
    rig.hold()  # clutch released ("lifted mouse")
    rig.sample([0.5, 0.5, 0.0])
    rig.tick(10)
    assert np.allclose(rig.cmd(), q_hold)
    ex = rig.extra()
    assert ex["clutch"] is False and ex["engaged_arm"] is None
    assert ex["anchor_tcp"] is None and ex["target_tcp"] is None
    rig.hold(CLUTCH)  # re-engage at the new tracker pose: no teleport
    rig.tick()
    assert np.allclose(rig.cmd(), q_hold)
    rig.sample([0.51, 0.5, 0.0])
    rig.tick()
    assert np.allclose(rig.cmd()[:3], [0.02, 0.0, 0.0], atol=1e-12)


@pytest.mark.parametrize("how", ["stale", "invalid"])
def test_stale_or_invalid_sample_holds_and_clears_anchors(how):
    rig = Rig()
    rig.sample([0.0, 0.0, 0.0])
    rig.hold(CLUTCH)
    rig.tick()
    rig.sample([0.01, 0.0, 0.0])
    rig.tick()
    q_hold = rig.cmd()
    if how == "stale":
        rig.sample([0.05, 0.0, 0.0], age=0.5, sticky=False)  # rx 0.5 s ago (> stale_s)
    else:
        rig.sample([0.05, 0.0, 0.0], valid=False, sticky=False)  # jump glitch
    rig.tick(3)
    assert np.allclose(rig.cmd(), q_hold)
    ex = rig.extra()
    assert ex["clutch"] is True and ex["engaged_arm"] is None and ex["anchor_tcp"] is None
    rig.sample([0.05, 0.0, 0.0])  # fresh + valid again: re-anchor, no jump
    rig.tick()
    assert np.allclose(rig.cmd(), q_hold)
    rig.sample([0.06, 0.0, 0.0])
    rig.tick()
    assert np.allclose(rig.cmd()[:3], [0.02, 0.0, 0.0], atol=1e-12)


def test_arm_switch_clears_anchors_and_reengages_on_new_arm():
    rig = Rig()
    rig.sample([0.0, 0.0, 0.0])
    rig.hold(CLUTCH)
    rig.tick()
    rig.sample([0.01, 0.0, 0.0])
    rig.tick()
    res = rig.act("switch_arm")  # tick: command drained, then arm1 engages at Δ=0
    assert res.ok and rig.loop.active_arm == "arm1"
    assert rig.extra()["engaged_arm"] == "arm1"
    assert np.allclose(rig.cmd("arm1"), 0.0)
    rig.sample([0.02, 0.0, 0.0])
    rig.tick()
    assert np.allclose(rig.cmd("arm1")[:3], [0.01, 0.0, 0.0], atol=1e-12)
    assert np.allclose(rig.cmd("arm0")[:3], [0.01, 0.0, 0.0], atol=1e-12)  # frozen


# -- settings: yaw / scale / rotation --------------------------------------------------------
def test_yaw_alignment_rotates_delta_into_mjcf_world():
    rig = Rig()
    rig.settings.update(yaw_deg=90.0)
    rig.sample([0.0, 0.0, 0.0])
    rig.hold(CLUTCH)
    rig.tick()
    rig.sample([0.01, 0.0, 0.0])
    rig.tick()
    assert np.allclose(rig.cmd()[:3], [0.0, 0.01, 0.0], atol=1e-12)


def test_pos_scale_multiplies_delta():
    rig = Rig()
    rig.settings.update(pos_scale=2.0)
    rig.sample([0.0, 0.0, 0.0])
    rig.hold(CLUTCH)
    rig.tick()
    rig.sample([0.005, 0.0, 0.0])
    rig.tick()
    assert np.allclose(rig.cmd()[:3], [0.01, 0.0, 0.0], atol=1e-12)


@pytest.mark.parametrize("follow", [True, False])
def test_follow_rotation_toggle(follow):
    rig = Rig()
    rig.settings.update(follow_rotation=follow)
    rig.sample([0.0, 0.0, 0.0])
    rig.hold(CLUTCH)
    rig.tick()
    # 0.03 rad: under the leash (0.2) and the per-tick dq_max clamp (0.04).
    rig.sample([0.01, 0.0, 0.0], se3.rotvec_to_quat([0.0, 0.0, 0.03]))
    rig.tick()
    assert np.allclose(rig.cmd()[:3], [0.01, 0.0, 0.0], atol=1e-12)
    expect = [0.0, 0.0, 0.03] if follow else [0.0, 0.0, 0.0]
    assert np.allclose(rig.cmd()[3:6], expect, atol=1e-9)


# -- leash slip / watchdog ------------------------------------------------------------------
def test_leash_truncation_slips_anchor_instead_of_accumulating():
    rig = Rig()
    rig.sample([0.0, 0.0, 0.0])
    rig.hold(CLUTCH)
    rig.tick()
    rig.sample([0.05, 0.0, 0.0])  # one 5 cm hand jump: > leash 2.5 cm
    rig.tick()
    assert rig.cmd()[0] == pytest.approx(0.025)
    rig.tick(10)  # hand still: the arm must NOT creep on to 0.05
    assert rig.cmd()[0] == pytest.approx(0.025)
    rig.sample([0.06, 0.0, 0.0])  # further +1 cm applies on top of the slipped anchor
    rig.tick()
    assert rig.cmd()[0] == pytest.approx(0.035)


def test_watchdog_scale_shrinks_step_toward_target():
    rig = Rig()
    rig.sample([0.0, 0.0, 0.0])
    rig.hold(CLUTCH)
    rig.tick(2)
    t_last_rx = rig.t  # last heartbeat
    rig.t = t_last_rx + 0.25  # deadman 0.2 s + half the 0.1 s ramp -> scale 0.5
    rig.sample([0.01, 0.0, 0.0])
    rig.loop.run_tick(rig.t)
    assert rig.loop.supervisor.watchdog.scale(rig.t) == pytest.approx(0.5)
    assert rig.cmd()[0] == pytest.approx(0.005)
    rig.t = t_last_rx + 0.35  # ramp finished: latched at zero -> hold
    rig.sample([0.02, 0.0, 0.0])
    rig.loop.run_tick(rig.t)
    assert rig.cmd()[0] == pytest.approx(0.005)


# -- keyboard interplay -----------------------------------------------------------------------
def test_keyboard_translate_ignored_but_rail_and_gripper_work_while_clutched():
    rig = Rig()
    rig.sample([0.0, 0.0, 0.0])
    rig.hold(CLUTCH, "KeyW", "KeyI")
    rig.tick(5)
    assert np.allclose(rig.cmd()[:6], 0.0)  # translate/rotate keys ignored
    rig.hold(CLUTCH, "ArrowRight")
    rig.tick(5)
    assert rig.cmd()[7] == pytest.approx(5 * 0.10 * DT)  # rail_mps * dt per tick
    assert np.allclose(rig.cmd()[:3], 0.0)
    rig.hold(CLUTCH, "KeyF")
    rig.tick(5)
    assert rig.loop._grip_frac["arm0"] == pytest.approx(1.0 - 5 * 1.2 * DT)
    # Rail keeps integrating even while the tracker sample is stale.
    rig.sample([0.0, 0.0, 0.0], age=1.0, sticky=False)
    rig.hold(CLUTCH, "ArrowRight")
    rail = rig.cmd()[7]
    rig.tick(2)
    assert rig.cmd()[7] == pytest.approx(rail + 2 * 0.10 * DT)


def test_without_provider_clutch_key_is_inert_and_keyboard_unchanged():
    rig = Rig(tracker=False)
    rig.sample([0.0, 0.0, 0.0])
    rig.hold(CLUTCH)
    rig.tick(3)
    assert np.allclose(rig.cmd(), 0.0)
    assert rig.bus.snapshot.get()[0].session_extra["tracker"] is None
    rig.hold("KeyW")
    rig.tick(1)
    assert rig.cmd()[0] == pytest.approx(0.12 * DT)


# -- commands -----------------------------------------------------------------------------------
def test_switch_arm_prev_wraps():
    rig = Rig()
    assert rig.loop.active_arm == "arm0"
    res = rig.act("switch_arm_prev")
    assert res.ok and res.detail == "arm1" and rig.loop.active_arm == "arm1"
    res = rig.act("switch_arm_prev")
    assert res.ok and res.detail == "arm0"


def test_tracker_settings_op_updates_live_settings_and_validates():
    rig = Rig()
    res = rig.act("tracker_settings", {"pos_scale": 2.0, "yaw_deg": 45.0})
    assert res.ok and "pos_scale=2" in res.detail and "yaw_deg=45" in res.detail
    v = rig.settings.get()
    assert (v.yaw_deg, v.pos_scale, v.follow_rotation) == (45.0, 2.0, True)
    res = rig.act("tracker_settings", {"follow_rotation": False})
    assert res.ok and rig.settings.get().follow_rotation is False
    assert rig.settings.get().pos_scale == 2.0  # omitted fields unchanged
    res = rig.act("tracker_settings", {"pos_scale": 9.0})
    assert not res.ok and "invalid" in res.detail
    assert rig.settings.get().pos_scale == 2.0
    res = rig.act("tracker_settings", {})
    assert res.ok
    rig2 = Rig(tracker=False)
    assert not rig2.act("tracker_settings", {"pos_scale": 1.0}).ok
