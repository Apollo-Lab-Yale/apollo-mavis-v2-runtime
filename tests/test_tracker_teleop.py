"""Tracker teleop provider over FakeWorkcell with a 6-DoF fake kin/IK
(13-tracker §4): engage/follow, release/re-engage, stale/invalid, arm switch,
yaw alignment, pos_scale, follow_rotation, leash anchor slip, watchdog scale,
keyboard interplay, switch_arm_prev and tracker_settings."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from apollo_mavis_v2_core import Command, HeldState, Pose, se3
from apollo_mavis_v2_core.testing import FakeArm, FakeWorkcell
from pydantic import ValidationError

from apollo_mavis_v2_runtime.bus import RuntimeBus
from apollo_mavis_v2_runtime.config import ControlConfig, TargetRateConfig
from apollo_mavis_v2_runtime.control.loop import ControlLoop
from apollo_mavis_v2_runtime.control.tracker_teleop import (
    TRACKER_CLUTCH_CODE,
    TrackerTeleop,
    align_pose,
    interp_pose,
    yaw_quat,
)
from apollo_mavis_v2_runtime.devices.tracker import TrackerSample, TrackerSettings
from apollo_mavis_v2_runtime.safety.gate import NullGate
from apollo_mavis_v2_runtime.safety.supervisor import SafetySupervisor
from apollo_mavis_v2_runtime.safety.watchdog import InputWatchdog

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
    every subsequent sample carries; ``gripper_arms`` mirrors the loop option;
    ``kin`` / ``ik`` swap the fake kinematics (default: rail-unaware PoseKin).
    """

    def __init__(
        self,
        tracker: bool = True,
        gripper_arms=None,
        *,
        filter=False,
        filter_cfg=None,
        kin=None,
        ik=None,
        rate_limit=False,
    ):
        # The tracker target rate limit (04-runtime §6) is OFF by default so
        # the anchor/delta tests stay exact per tick; rate-limit tests opt in.
        unlimited = TargetRateConfig(v_mps=1e9, w_radps=1e9)
        # translate_frame "base": these rigs' fake kinematics put the TCP straight in
        # q[:3] with an identity orientation, so the pre-2026-09-08 base frame is what
        # makes "one KeyW tick == +0.0012 m in x" exact. The runtime DEFAULT is
        # "world" (W = world -y, operator-fixed; 2026-09-08 evening, superseding that
        # morning's "camera") and is covered by test_teleop_math.py /
        # test_camera_frame.py; nothing here is about the frame.
        self.cfg = ControlConfig(
            target_rate=TargetRateConfig() if rate_limit else unlimited,
            translate_frame="base",
        )
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
        self.click_actions: tuple = ()  # (edge_seq, action) of the bound controller press edges
        self.loop = ControlLoop(
            self.cell, self.cfg, self.bus,
            SafetySupervisor(NullGate(), InputWatchdog()), ["arm0", "arm1"],
            ik=ik or PoseIK(), kin=kin or PoseKin(), tracker=self.tracker if tracker else None,
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
            self.click_actions, pose_rx_mono=self.t - age,
        )
        self.bus.tracker.put(s)
        self.pose = (np.asarray(pos, float), np.asarray(quat, float)) if sticky else None

    def device(self, *codes, controller=None, click_action=None, click_actions=None):
        """Script the controller-derived codes and the bound discrete actions
        as the reader derives them: ``click_action`` is the shorthand for "the
        newest press edge is bound to this action", ``click_actions`` the full
        ``(edge_seq, action)`` tuple; re-publishes the current pose at once
        (the reader does the same on a button edge)."""
        self.codes = frozenset(codes)
        self.controller = controller
        if click_actions is None:
            click_actions = () if click_action is None else ((controller.edge_seq, click_action),)
        self.click_actions = tuple(click_actions)
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
    # the browser goes silent while the loop keeps ticking (no heartbeat, 100 Hz):
    # a single 0.25 s time jump would be a PROCESS stall to the loop (2026-09-07:
    # a stall credits the deadman instead of tripping it)
    rig.tick(24, heartbeat=False)
    rig.t += DT  # deadman 0.2 s + half the 0.1 s ramp -> scale 0.5
    rig.sample([0.01, 0.0, 0.0])
    rig.loop.run_tick(rig.t)
    assert rig.loop.supervisor.watchdog.scale(rig.t) == pytest.approx(0.5)
    assert rig.cmd()[0] == pytest.approx(0.005)
    rig.tick(9, heartbeat=False)  # the ramp runs out under a still hand: the arm keeps
    reached = rig.cmd()[0]  # closing on 0.01 with a shrinking scale, never past it
    assert 0.005 < reached < 0.01
    rig.t += DT  # 0.35 s: ramp finished, latched at zero -> hold
    rig.sample([0.02, 0.0, 0.0])
    rig.loop.run_tick(rig.t)
    assert rig.cmd()[0] == pytest.approx(reached)  # a further hand move: no motion


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


# -- target rate limit + component-wise residual slip (04-runtime §6, 2026-09-02) -----------
def test_target_rate_config_rejects_zero_and_negative_rates():
    # clamp_pose_to_leash with a zero cap freezes the clutched target, a negative
    # cap steps AWAY from the hand: neither may pass config validation.
    for bad in ({"v_mps": 0.0}, {"v_mps": -1.0}, {"w_radps": 0.0}, {"w_radps": -2.0}):
        with pytest.raises(ValidationError):
            TargetRateConfig(**bad)
    with pytest.raises(ValidationError, match="target_rate"):
        ControlConfig.model_validate({"target_rate": {"v_mps": -1.0}})
    assert TargetRateConfig().v_mps == 1.0 and TargetRateConfig().w_radps == 2.0
    assert TargetRateConfig(v_mps=1e9, w_radps=1e-3).w_radps == 1e-3  # any positive value


def test_target_rate_limit_caps_the_per_tick_step_and_catches_up_without_slipping():
    rig = Rig(rate_limit=True)  # defaults: 1.0 m/s -> 10 mm/tick, 2 rad/s -> 0.02 rad/tick
    rig.sample([0.0, 0.0, 0.0])
    rig.hold(CLUTCH)
    rig.tick()
    anchor0 = rig.extra()["anchor_tcp"].position.copy()
    rig.sample([0.02, 0.0, 0.0])  # 20 mm jump, inside the 25 mm leash
    rig.tick()
    assert np.allclose(rig.cmd()[:3], [0.01, 0.0, 0.0], atol=1e-9)  # one tick = 10 mm
    assert np.allclose(rig.extra()["anchor_tcp"].position, anchor0)  # truncation NOT slipped
    rig.tick()
    assert np.allclose(rig.cmd()[:3], [0.02, 0.0, 0.0], atol=1e-9)  # caught up, hand still
    assert np.allclose(rig.extra()["anchor_tcp"].position, anchor0)
    # rotation: 0.1 rad about z in one sample -> 0.02 rad per tick, complete after 5 ticks
    rig.sample([0.02, 0.0, 0.0], se3.rotvec_to_quat([0.0, 0.0, 0.1]))
    rig.tick()
    assert np.isclose(rig.cmd()[5], 0.02, atol=1e-9)
    rig.tick(4)
    assert np.isclose(rig.cmd()[5], 0.1, atol=1e-9)
    assert np.allclose(rig.cmd()[:3], [0.02, 0.0, 0.0], atol=1e-9)  # position untouched


@pytest.mark.parametrize("channel", ["position", "rotation"])
def test_target_rate_limit_beyond_the_leash_slips_once_and_converges_on_a_slow_arm(channel):
    """Hand jump beyond the leash with an arm slower than the rate: the leash
    slips the anchor exactly once (rule: leash slip bounds the offset), the
    rate-limit truncation is never slipped, and the target converges to the
    leash-clamped pose without creeping while the arm catches up."""
    rig = Rig(rate_limit=True)  # 10 mm / 0.02 rad per tick
    rig.cell.arms["arm0"]._max_joint_speed = 0.3  # 3 mm / 0.003 rad per tick: slower than the rate
    rig.sample([0.0, 0.0, 0.0])
    rig.hold(CLUTCH)
    rig.tick()
    a0 = rig.extra()["anchor_tcp"]
    if channel == "position":
        rig.sample([0.05, 0.0, 0.0])  # 5 cm > 25 mm leash
        rig.tick()
        a1 = rig.extra()["anchor_tcp"]
        assert a1.position[0] == pytest.approx(a0.position[0] - 0.025)  # one leash slip
        assert rig.cmd()[0] == pytest.approx(0.01)  # one rate step
        rig.tick(15)
        assert rig.cmd()[0] == pytest.approx(0.025)  # converged to the leash-clamped target
        assert rig.extra()["target_tcp"].position[0] == pytest.approx(0.025)
        assert rig.extra()["anchor_tcp"].position[0] == pytest.approx(a1.position[0])  # no 2nd slip
        assert rig.cell.arms["arm0"].get_state().q[0] == pytest.approx(0.025, abs=1e-9)  # arrived
        rig.sample([0.06, 0.0, 0.0])  # +1 cm applies on top of the slipped anchor
        rig.tick(2)
        assert rig.cmd()[0] == pytest.approx(0.035)
    else:
        rig.sample([0.0, 0.0, 0.0], se3.rotvec_to_quat([0.0, 0.0, 0.5]))  # 0.5 rad > 0.2 leash
        rig.tick()
        a1 = rig.extra()["anchor_tcp"]
        assert se3.quat_geodesic(a1.orientation, a0.orientation) == pytest.approx(0.3, abs=1e-9)
        assert rig.cmd()[5] == pytest.approx(0.02, abs=1e-9)
        assert np.allclose(rig.cmd()[:3], 0.0, atol=1e-12)
        rig.tick(15)
        assert rig.cmd()[5] == pytest.approx(0.2, abs=1e-9)  # converged to the leash-clamped target
        assert se3.quat_geodesic(rig.extra()["anchor_tcp"].orientation, a1.orientation) < 1e-9
        assert np.allclose(rig.extra()["anchor_tcp"].position, a0.position, atol=1e-12)
        rig.tick(60)  # the wrist needs ~67 ticks at 0.003 rad/tick: no creep, no 2nd slip meanwhile
        assert rig.cmd()[5] == pytest.approx(0.2, abs=1e-9)
        assert se3.quat_geodesic(rig.extra()["anchor_tcp"].orientation, a1.orientation) < 1e-9
        assert rig.cell.arms["arm0"].get_state().q[5] == pytest.approx(0.2, abs=1e-9)  # arrived


def test_settings_change_mid_catch_up_keeps_the_pending_rate_limited_step():
    """Rule (c) re-anchors to the provider's leash-clamped target, not to the
    rate-limited pose handed to IK: no settings-induced motion AND the pending
    catch-up toward the hand is finished, not silently slipped into A_ee."""
    rig = Rig(rate_limit=True)
    rig.sample([0.0, 0.0, 0.0])
    rig.hold(CLUTCH)
    rig.tick()
    rig.sample([0.02, 0.0, 0.0])  # 20 mm inside the leash: two 10 mm ticks
    rig.tick()
    assert rig.cmd()[0] == pytest.approx(0.01)
    assert rig.act("tracker_settings", {"pos_scale": 1.5}).ok  # one tick: re-anchor (rule c)
    assert rig.cmd()[0] == pytest.approx(0.02)  # the second 10 mm step still happened
    assert rig.extra()["anchor_tcp"].position[0] == pytest.approx(0.02)
    rig.tick(3)
    assert rig.cmd()[0] == pytest.approx(0.02)  # hand still: no creep either way
    rig.sample([0.03, 0.0, 0.0])  # +10 mm at scale 1.5 -> +15 mm on top of 0.02 (not 0.01)
    rig.tick(2)
    assert rig.cmd()[0] == pytest.approx(0.035)
    # Rotation: 0.1 rad hand rotation = five 0.02 rad ticks; a change after the first
    # tick must not drop the remaining 0.08 rad.
    rig.sample([0.03, 0.0, 0.0], se3.rotvec_to_quat([0.0, 0.0, 0.1]))
    rig.tick()
    assert rig.cmd()[5] == pytest.approx(0.02, abs=1e-9)
    assert rig.act("tracker_settings", {"pos_scale": 1.2}).ok
    assert rig.cmd()[5] == pytest.approx(0.04, abs=1e-9)
    rig.tick(5)
    assert rig.cmd()[5] == pytest.approx(0.1, abs=1e-9)
    assert rig.cmd()[0] == pytest.approx(0.035)


class RotationLaggingIK(PoseIK):
    """Reaches the position exactly but reports an over-threshold rotation
    residual and leaves the orientation where it was (velocity-saturated wrist)."""

    def solve(self, arm_id, target, q_last):
        q = np.array(q_last, dtype=float)
        q[:3] = target.position
        return SimpleNamespace(q=q, diverged=False, pos_err_m=0.0, rot_err_rad=0.3)


def test_rotation_residual_slips_orientation_only_and_leaves_the_position_anchor():
    rig = Rig(ik=RotationLaggingIK())
    rig.sample([0.0, 0.0, 0.0])
    rig.hold(CLUTCH)
    rig.tick()
    anchor0 = rig.extra()["anchor_tcp"]
    q_hand = se3.rotvec_to_quat([0.0, 0.0, 0.3])
    rig.sample([0.01, 0.0, 0.0], q_hand)  # translate 10 mm AND rotate 0.3 rad
    rig.tick()
    ex = rig.extra()
    # position: followed 1:1, anchor position untouched by the rotation residual
    assert np.allclose(rig.cmd()[:3], [0.01, 0.0, 0.0], atol=1e-12)
    assert np.allclose(ex["target_tcp"].position, [0.01, 0.0, 0.0], atol=1e-12)
    assert np.allclose(ex["anchor_tcp"].position, anchor0.position, atol=1e-12)
    # orientation: frozen back to the achieved (unrotated) pose -> anchor slipped in rotation
    assert se3.quat_geodesic(ex["target_tcp"].orientation, IDENT) < 1e-9
    assert se3.quat_geodesic(ex["anchor_tcp"].orientation, anchor0.orientation) > 0.29
    # hand still: no further motion of either component
    rig.tick(3)
    assert np.allclose(rig.cmd()[:3], [0.01, 0.0, 0.0], atol=1e-12)
    assert np.allclose(ex["anchor_tcp"].position, rig.extra()["anchor_tcp"].position, atol=1e-12)
