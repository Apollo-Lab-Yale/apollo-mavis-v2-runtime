"""13-tracker §4 review fixes (2026-09-02) + pose-filter wiring + device discrete
actions, over the deterministic ``Rig`` of ``test_tracker_teleop``:

* re-seed after non-teleop motion (plan / jog / policy): the first clutched
  tick has zero delta;
* body-frame rotation anchor slip: ``D ⊗ A_ee'.q == achieved.q`` exactly;
* settings snapshot at engagement: a ``tracker_settings`` change while clutched
  re-anchors and never moves the target;
* One Euro filter in the provider: jitter reduced vs passthrough, reset on
  engage and after gaps, live retune / toggle via ``tracker_settings``;
* controller press edges (``edge_seq``; scripted here as trackpad clicks with an
  explicit ``click_action`` shorthand — the loop is map-agnostic) fire ``switch_arm`` /
  ``switch_arm_prev`` once per press inside the loop, latch ``device_action``
  ~1 s, and are nacked while a DAgger takeover is engaged.
"""

from __future__ import annotations

import numpy as np
import pytest
from apollo_mavis_v2_core import CommandSource, LatestSlot, Pose, se3
from apollo_mavis_v2_core.interfaces.policy import Observation, PolicySpec
from apollo_mavis_v2_core.testing import FakeArm, FakeWorkcell
from test_tracker_teleop import CLUTCH, DT, IDENT, PoseIK, PoseKin, Rig

from apollo_mavis_v2_runtime.bus import RuntimeBus
from apollo_mavis_v2_runtime.config import ControlConfig, TargetRateConfig
from apollo_mavis_v2_runtime.control.loop import DEVICE_ACTION_LINGER_S
from apollo_mavis_v2_runtime.control.pose_filter import PoseFilterConfig
from apollo_mavis_v2_runtime.control.tracker_teleop import TrackerTeleop, align_pose
from apollo_mavis_v2_runtime.dagger.gate import TakeoverGateImpl
from apollo_mavis_v2_runtime.dagger.loop import GatedPolicyExecutor
from apollo_mavis_v2_runtime.dagger.policies import ScriptedPolicy
from apollo_mavis_v2_runtime.dagger.policy_runner import ActionAnchor, PolicyRunner, SlewLimits
from apollo_mavis_v2_runtime.devices.tracker import (
    GRIPPER_OPEN_CODE,
    ControllerState,
    TrackerSettings,
    note_edges,
)
from apollo_mavis_v2_runtime.safety.gate import NullGate
from apollo_mavis_v2_runtime.safety.supervisor import SafetySupervisor
from apollo_mavis_v2_runtime.safety.watchdog import InputWatchdog


def _pad_state(x: float, y: float, prev: ControllerState | None = None, click: bool = True):
    """Classified controller state (as the reader publishes it) for a pad press."""
    raw = ControllerState(trackpad_touch=True, trackpad_click=click, trackpad_x=x, trackpad_y=y)
    return note_edges(prev, raw, 0.3)


# -- (a) re-seed after non-teleop motion ---------------------------------------------------------
def _clutch_to(rig: Rig, x: float) -> None:
    rig.sample([0.0, 0.0, 0.0])
    rig.hold(CLUTCH)
    rig.tick()
    rig.sample([x, 0.0, 0.0])
    rig.tick()
    assert rig.cmd()[0] == pytest.approx(x)
    rig.hold()
    rig.tick(2)


def _reengage_has_zero_delta(rig: Rig, expect_x: float) -> None:
    """Clutch anywhere: the first tick must not move the arm; then Δ applies."""
    rig.sample([0.3, 0.2, 0.1])
    rig.hold(CLUTCH)
    rig.tick()
    assert rig.cmd()[0] == pytest.approx(expect_x, abs=1e-9)
    ex = rig.extra()
    assert ex["engaged_arm"] == "arm0"
    assert np.allclose(ex["anchor_tcp"].position[:1], [expect_x], atol=1e-9)
    rig.tick(5)
    assert rig.cmd()[0] == pytest.approx(expect_x, abs=1e-9)  # hand still: no creep
    rig.sample([0.31, 0.2, 0.1])
    rig.tick()
    assert rig.cmd()[0] == pytest.approx(expect_x + 0.01, abs=1e-9)


def test_plan_motion_reseeds_teleop_target_zero_delta_on_first_clutched_tick():
    rig = Rig()
    _clutch_to(rig, 0.01)
    assert "arm0" in rig.loop._teleop_seeded
    goal = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05])
    rig.loop.plans.load("arm0", [goal])
    rig.loop._plan_state["arm0"] = "executing"
    rig.tick(2)
    assert rig.loop._arm_source["arm0"] is CommandSource.PLANNER
    assert "arm0" not in rig.loop._teleop_seeded  # seed invalidated by the plan
    rig.tick(38)  # slew 0.02 rad/tick, rail 0.002 m/tick -> both reached
    assert not rig.loop.plans.active("arm0")
    assert rig.cmd()[0] == pytest.approx(0.1) and rig.cmd()[7] == pytest.approx(0.05)
    assert rig.loop._arm_source["arm0"] is CommandSource.TELEOP  # idle hold ticks
    assert "arm0" not in rig.loop._teleop_seeded  # ...do not seed: only motion input does
    _reengage_has_zero_delta(rig, 0.1)
    assert rig.loop._arm_source["arm0"] is CommandSource.TELEOP


def test_jog_motion_reseeds_teleop_target():
    rig = Rig()
    _clutch_to(rig, 0.01)
    res = rig.act(
        "joint_target",
        {"arm_id": "arm0", "positions": [0.05, 0, 0, 0, 0, 0, 0, 0.0], "mode": "jog"},
    )
    assert res.ok and res.detail == "jog"
    rig.tick()
    assert rig.loop._arm_source["arm0"] is CommandSource.JOINT_JOG
    rig.tick(6)  # 0.04 rad at 0.02 rad/tick
    assert rig.cmd()[0] == pytest.approx(0.05) and not rig.loop.jog.active("arm0")
    assert "arm0" not in rig.loop._teleop_seeded
    _reengage_has_zero_delta(rig, 0.05)


def test_keyboard_teleop_after_plan_starts_from_measured_not_stale_target():
    rig = Rig()
    _clutch_to(rig, 0.01)
    rig.loop.plans.load("arm0", [np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])])
    rig.loop._plan_state["arm0"] = "executing"
    rig.tick(10)
    assert rig.cmd()[0] == pytest.approx(0.1)
    rig.hold("KeyW")  # +x at 0.12 m/s from the MEASURED pose (0.1), not the old target
    rig.tick()
    assert rig.cmd()[0] == pytest.approx(0.1 + 0.12 * DT, abs=1e-9)


# -- dagger executor rig (policy motion + takeover nack) ------------------------------------------
ARMS_META = [("arm0", True), ("arm1", False)]
DIM = 8 + 7
NAMES = [f"a{i}" for i in range(DIM)]


def _dx(v: float) -> np.ndarray:
    a = np.zeros(DIM, np.float32)
    a[0] = v  # arm0 ee.dx per policy period
    a[6] = 0.5  # arm0 absolute gripper target
    return a


class DaggerRig:
    """GatedPolicyExecutor (inference mode) with a tracker provider; the
    scripted policy pushes arm0 +x, ``_tick_once`` mirrors tests/dagger."""

    def __init__(self, mode: str = "inference"):
        self.cell = FakeWorkcell(
            {"arm0": FakeArm("arm0", has_rail=True), "arm1": FakeArm("arm1")}
        )
        self.cell.start()
        self.bus = RuntimeBus()
        self.gate = TakeoverGateImpl(["arm0", "arm1"])
        spec = PolicySpec("delta_ee", "arm_base:arm0", NAMES, [], [], 0)
        self.policy = ScriptedPolicy(spec, script=lambda k: _dx(0.01))
        self.t = 0.0
        self.runner = PolicyRunner(
            self.policy, lambda: Observation(np.zeros(4, np.float32), {}, self.t, 0),
            rate_hz=20.0, clock=lambda: self.t,
        )
        ik, kin = PoseIK(), PoseKin()
        self.settings = TrackerSettings(filter_enabled=False)
        self.tracker = TrackerTeleop(
            self.bus.tracker, self.settings, stale_s=0.2, leash_pos_m=0.025, leash_rot_rad=0.2
        )
        self.loop = GatedPolicyExecutor(
            self.cell, ControlConfig(target_rate=TargetRateConfig(v_mps=1e9, w_radps=1e9)),
            self.bus, SafetySupervisor(NullGate(), InputWatchdog()),
            ["arm0", "arm1"], gate=self.gate, runner=self.runner,
            anchor=ActionAnchor(ik, kin, SlewLimits()), arms_meta=ARMS_META,
            session_mode=mode, version_label="deploy/v001", ik=ik, kin=kin,
            tracker=self.tracker, clock=lambda: self.t,
        )
        self.loop._seed_from_measured()
        self.sample_seq = 0
        self.pose = None
        self.codes: frozenset[str] = frozenset()
        self.controller = None
        self.click_actions: tuple = ()

    def sample(self, pos, age=0.0):
        from apollo_mavis_v2_runtime.devices.tracker import TrackerSample

        self.sample_seq += 1
        self.bus.tracker.put(TrackerSample(
            Pose(np.asarray(pos, float), IDENT), np.zeros(3), np.zeros(3), self.t,
            self.t - age, self.sample_seq, True, self.controller, self.codes, self.click_actions,
            pose_rx_mono=self.t - age,
        ))
        self.pose = np.asarray(pos, float)

    def device(self, *codes, controller=None, click_action=None):
        self.codes, self.controller = frozenset(codes), controller
        self.click_actions = () if click_action is None else ((controller.edge_seq, click_action),)
        if self.pose is not None:
            self.sample(self.pose)

    def tick(self, n=1):
        for _ in range(n):
            self.t += DT
            if int(round(self.t * 100)) % 5 == 0:
                self.runner.step_once(self.t)
            if self.pose is not None:
                self.sample(self.pose)
            self.loop.run_tick(self.t)
            for a in ("arm0", "arm1"):
                self.cell.arms[a].command_joints(self.loop._last_cmd[a])
            self.cell.step(DT)

    def act(self, op, args=None):
        from apollo_mavis_v2_core import Command

        fut = self.bus.commands.submit(Command(op=op, args=args or {}, source="ws"))
        self.tick()
        return fut.result(timeout=1.0)

    def cmd(self, arm="arm0"):
        return np.array(self.loop._last_cmd[arm])

    def extra(self):
        return self.bus.snapshot.get()[0].session_extra["tracker"]


def test_policy_motion_reseeds_teleop_target_for_takeover_clutch():
    rig = DaggerRig()
    rig.tick(60)  # policy drives arm0 +x
    x_policy = rig.cmd()[0]
    assert x_policy > 0.005
    assert rig.loop._arm_source["arm0"] is CommandSource.POLICY
    assert "arm0" not in rig.loop._teleop_seeded
    assert rig.act("takeover_toggle").ok  # human engaged on arm0 (transition, then HUMAN)
    rig.tick(40)
    x_frozen = rig.cmd()[0]
    rig.sample([0.5, 0.5, 0.5])
    rig.device(CLUTCH)
    rig.tick()  # first clutched tick after policy motion: zero delta
    assert rig.cmd()[0] == pytest.approx(x_frozen, abs=1e-9)
    assert rig.extra()["engaged_arm"] == "arm0"
    assert rig.loop._arm_source["arm0"] is CommandSource.TELEOP
    rig.tick(5)
    assert rig.cmd()[0] == pytest.approx(x_frozen, abs=1e-9)
    rig.sample([0.51, 0.5, 0.5])
    rig.tick()
    assert rig.cmd()[0] == pytest.approx(x_frozen + 0.01, abs=1e-9)


def test_device_switch_arm_nacked_while_takeover_engaged():
    rig = DaggerRig()
    rig.tick(10)
    rig.sample([0.0, 0.0, 0.0])
    rig.device(controller=ControllerState())  # idle controller adopted (edge_seq 0)
    rig.tick()
    assert rig.act("takeover_toggle").ok and rig.gate.engaged_arm() == "arm0"
    st = _pad_state(0.0, 0.9)  # a press edge whose scripted action is switch_arm
    rig.device(controller=st, click_action="switch_arm")
    rig.tick()
    assert rig.loop.active_arm == "arm0"  # nacked: takeover active
    assert rig.extra()["device_action"] == "switch_arm nacked: takeover active"
    rig.tick(5)
    assert rig.loop.active_arm == "arm0"
    rig.act("takeover_toggle")  # handback
    rig.tick(50)
    st2 = _pad_state(0.0, 0.9, prev=_pad_state(0.0, 0.9, prev=st, click=False))
    assert st2.edge_seq == 2
    rig.device(controller=st2, click_action="switch_arm")
    rig.tick()
    assert rig.loop.active_arm == "arm1" and rig.extra()["device_action"] == "switch_arm"


# -- (b) body-frame rotation anchor slip -----------------------------------------------------------
def _provider() -> TrackerTeleop:
    return TrackerTeleop(
        LatestSlot(), TrackerSettings(), stale_s=0.2, leash_pos_m=0.025, leash_rot_rad=0.2
    )


def _rand_quat(rng) -> np.ndarray:
    return se3.rotvec_to_quat(rng.normal(size=3))


@pytest.mark.parametrize("seed", range(6))
def test_body_frame_slip_is_exact_for_any_hand_rotation(seed):
    rng = np.random.default_rng(seed)
    prov = _provider()
    a_ee = Pose(rng.normal(size=3), _rand_quat(rng))
    prov._a_ee = a_ee
    prov._target = a_ee
    d = _rand_quat(rng)  # hand rotation since engagement (world frame, left-multiplied)
    intended = Pose(a_ee.position + rng.normal(size=3), se3.quat_mul(d, a_ee.orientation))
    achieved = Pose(intended.position + rng.normal(size=3) * 0.01, _rand_quat(rng))  # truncated
    prov.slip(intended, achieved)
    a_new = prov._a_ee
    # The same hand rotation D applied to the slipped anchor reproduces the achieved pose.
    assert se3.quat_geodesic(se3.quat_mul(d, a_new.orientation), achieved.orientation) < 1e-9
    assert np.allclose(a_new.position + (intended.position - a_ee.position), achieved.position)
    assert prov._target is achieved
    # The old world-frame formula (dq_w ⊗ A_ee.q) does NOT have this property in general.
    dq_w = se3.quat_mul(achieved.orientation, se3.quat_conj(intended.orientation))
    a_world = se3.quat_mul(dq_w, a_ee.orientation)
    assert se3.quat_geodesic(se3.quat_mul(d, a_world), achieved.orientation) > 1e-3


def test_slip_noop_on_identity_and_without_anchor():
    prov = _provider()
    p = Pose(np.array([0.1, 0.2, 0.3]), se3.rotvec_to_quat([0.1, 0.2, 0.3]))
    prov.slip(p, p)  # no anchor: nothing to do
    prov._a_ee = p
    prov.slip(p, Pose(p.position.copy(), -p.orientation))  # same rotation (double cover)
    assert prov._a_ee is p


def test_leash_rotation_truncation_slips_anchor_in_body_frame_end_to_end():
    rig = Rig()
    rig.sample([0.0, 0.0, 0.0], se3.rotvec_to_quat([0.0, 0.0, 0.0]))
    rig.hold(CLUTCH)
    rig.tick()
    # 0.3 rad about z in one go: > leash 0.2 rad -> truncated to 0.2, anchor slips by -0.1.
    rig.sample([0.0, 0.0, 0.0], se3.rotvec_to_quat([0.0, 0.0, 0.3]))
    rig.tick(8)  # dq_max 0.04/tick -> several ticks to reach the clamped target
    assert rig.cmd()[5] == pytest.approx(0.2, abs=1e-6)
    rig.tick(10)  # hand still: no creep toward 0.3
    assert rig.cmd()[5] == pytest.approx(0.2, abs=1e-6)
    a_ee = rig.tracker._a_ee
    hand_d = se3.rotvec_to_quat([0.0, 0.0, 0.3])  # D since engagement (A_trk was identity)
    achieved = se3.rotvec_to_quat([0.0, 0.0, 0.2])
    assert se3.quat_geodesic(se3.quat_mul(hand_d, a_ee.orientation), achieved) < 1e-6
    rig.sample([0.0, 0.0, 0.0], se3.rotvec_to_quat([0.0, 0.0, 0.35]))  # +0.05 on top
    rig.tick(5)
    assert rig.cmd()[5] == pytest.approx(0.25, abs=1e-6)


# -- (c) settings snapshot at engagement -----------------------------------------------------------
def test_settings_change_while_clutched_reanchors_and_never_moves_the_target():
    rig = Rig()
    rig.sample([0.0, 0.0, 0.0])
    rig.hold(CLUTCH)
    rig.tick()
    rig.sample([0.01, 0.0, 0.0], se3.rotvec_to_quat([0.0, 0.0, 0.03]))
    rig.tick()
    q_before = rig.cmd()
    assert q_before[0] == pytest.approx(0.01) and q_before[5] == pytest.approx(0.03, abs=1e-9)
    res = rig.act(
        "tracker_settings", {"pos_scale": 2.0, "yaw_deg": 90.0, "follow_rotation": False}
    )
    assert res.ok
    assert np.allclose(rig.cmd(), q_before, atol=1e-12)  # the arm did not move
    rig.tick(5)
    assert np.allclose(rig.cmd(), q_before, atol=1e-12)
    ex = rig.extra()
    assert ex["engaged_arm"] == "arm0"
    aligned = align_pose(Pose(np.array([0.01, 0.0, 0.0]), se3.rotvec_to_quat([0, 0, 0.03])), 90.0)
    assert np.allclose(rig.tracker._a_trk.position, aligned.position, atol=1e-12)
    assert np.allclose(ex["anchor_tcp"].position, q_before[:3], atol=1e-12)  # A_ee <- target
    assert rig.tracker.applied_settings.pos_scale == 2.0
    # Motion resumes under the NEW settings from the re-anchored offset: +5 mm hand x
    # -> yaw 90 turns it into +y, scale 2 doubles it; rotation no longer followed.
    rig.sample([0.015, 0.0, 0.0], se3.rotvec_to_quat([0.0, 0.0, 0.3]))
    rig.tick()
    assert np.allclose(rig.cmd()[:3], q_before[:3] + [0.0, 0.01, 0.0], atol=1e-9)
    assert rig.cmd()[5] == pytest.approx(0.03, abs=1e-9)


def test_settings_change_while_released_applies_on_next_engage():
    rig = Rig()
    rig.sample([0.0, 0.0, 0.0])
    rig.tick()
    assert rig.act("tracker_settings", {"pos_scale": 3.0}).ok
    rig.hold(CLUTCH)
    rig.tick()
    rig.sample([0.005, 0.0, 0.0])
    rig.tick()
    assert rig.cmd()[0] == pytest.approx(0.015)


# -- pose filter wiring ---------------------------------------------------------------------
def _jitter_run(rig: Rig, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centre = np.array([0.0, 0.0, 0.0])
    rig.sample(centre)
    rig.hold(CLUTCH)
    rig.tick()
    xs = []
    for _ in range(n):
        rig.sample(centre + rng.normal(0.0, 0.003, 3))  # 3 mm lighthouse jitter
        rig.tick()
        xs.append(rig.cmd()[:3])
    return np.array(xs[n // 3:])


def test_filter_reduces_target_jitter_versus_passthrough():
    raw = _jitter_run(Rig(filter=False), 300)
    filt = _jitter_run(Rig(filter=True), 300)
    raw_std = raw.std(axis=0).mean()
    filt_std = filt.std(axis=0).mean()
    assert raw_std > 0.002  # passthrough: the target jitters like the hand
    assert filt_std < raw_std / 5, (raw_std, filt_std)
    assert np.linalg.norm(filt.mean(axis=0)) < 2e-3  # centred, not biased


def test_filter_passthrough_when_disabled_and_lag_when_enabled():
    rig = Rig(filter=True)
    rig.sample([0.0, 0.0, 0.0])
    rig.hold(CLUTCH)
    rig.tick()
    rig.sample([0.01, 0.0, 0.0])  # 1 cm step
    rig.tick()
    first = rig.cmd()[0]
    assert 0.0 <= first < 0.005  # smoothed: far from the raw 1 cm on the first tick
    rig.tick(300)
    assert rig.cmd()[0] == pytest.approx(0.01, abs=2.5e-3)  # converges (within the deadband)
    ex = rig.extra()
    assert ex["pose_filtered"] is not None
    assert np.allclose(ex["pose_filtered"].position, [0.01, 0.0, 0.0], atol=2.5e-3)
    # Live toggle off: raw passthrough from the next tick, no re-anchoring jump.
    q = rig.cmd()
    assert rig.act("tracker_settings", {"filter_enabled": False}).ok
    assert np.allclose(rig.cmd(), q, atol=1e-12)
    assert rig.tracker.pose_filter.enabled is False
    rig.sample([0.02, 0.0, 0.0])
    rig.tick()
    assert rig.cmd()[0] == pytest.approx(q[0] + 0.01, abs=1e-9)  # exact 1:1 again
    assert np.allclose(rig.extra()["pose_filtered"].position, [0.02, 0.0, 0.0])


def test_filter_resets_on_engage_and_after_stale_gap():
    rig = Rig(filter=True)
    rig.sample([0.0, 0.0, 0.0])
    rig.hold(CLUTCH)
    rig.tick()
    rig.sample([0.01, 0.0, 0.0])
    rig.tick(20)
    q_hold = rig.cmd()
    rig.hold()  # release: filter keeps running (telemetry) but anchors clear
    rig.sample([0.5, 0.5, 0.0])
    rig.tick(3)
    assert rig.extra()["pose_filtered"] is not None and rig.extra()["engaged_arm"] is None
    rig.hold(CLUTCH)  # re-engage: filter reset -> filtered == raw, zero delta
    rig.tick()
    assert np.allclose(rig.cmd(), q_hold, atol=1e-12)
    assert np.allclose(rig.extra()["pose_filtered"].position, [0.5, 0.5, 0.0], atol=1e-12)
    rig.tick(5)
    assert np.allclose(rig.cmd(), q_hold, atol=1e-12)  # a still hand: exactly no motion
    # Stale gap: pose_filtered drops, anchors clear; fresh again -> reset, no jump.
    rig.sample([0.5, 0.5, 0.0], age=0.5, sticky=False)
    rig.tick(2)
    assert rig.extra()["pose_filtered"] is None and rig.extra()["engaged_arm"] is None
    rig.sample([0.7, 0.5, 0.0])
    rig.tick()
    assert np.allclose(rig.cmd(), q_hold, atol=1e-12)
    assert np.allclose(rig.extra()["pose_filtered"].position, [0.7, 0.5, 0.0], atol=1e-12)


def test_filter_retune_via_tracker_settings_and_validation():
    rig = Rig(filter=True, filter_cfg=PoseFilterConfig(d_cutoff_hz=2.0, deadband_m=0.0))
    f = rig.tracker.pose_filter
    assert (f.enabled, f.min_cutoff_hz, f.beta, f.cfg.d_cutoff_hz) == (True, 1.0, 0.05, 2.0)
    res = rig.act("tracker_settings", {"filter_min_cutoff_hz": 10.0, "filter_beta": 1.5})
    assert res.ok and "filter_min_cutoff_hz=10" in res.detail and "filter_beta=1.5" in res.detail
    assert (f.min_cutoff_hz, f.beta, f.enabled) == (10.0, 1.5, True)
    v = rig.settings.get()
    assert (v.filter_min_cutoff_hz, v.filter_beta, v.filter_enabled) == (10.0, 1.5, True)
    assert v.pos_scale == 1.0  # omitted fields unchanged
    for bad in ({"filter_min_cutoff_hz": 0.0}, {"filter_beta": -1.0}, {"filter_enabled": "x"}):
        res = rig.act("tracker_settings", bad)
        assert not res.ok and "invalid" in res.detail, bad
    assert (f.min_cutoff_hz, f.beta) == (10.0, 1.5)
    res = rig.act("tracker_settings", {"filter_enabled": False})
    assert res.ok and "filter_enabled=false" in res.detail and f.enabled is False
    # A higher cutoff tracks a step faster than the default 1 Hz.
    rig_fast = Rig(filter=True)
    rig_fast.settings.update(filter_min_cutoff_hz=10.0)
    rig_fast.sample([0.0, 0.0, 0.0])
    rig_fast.hold(CLUTCH)
    rig_fast.tick()
    rig_fast.sample([0.01, 0.0, 0.0])
    rig_fast.tick()
    rig_slow = Rig(filter=True)
    rig_slow.sample([0.0, 0.0, 0.0])
    rig_slow.hold(CLUTCH)
    rig_slow.tick()
    rig_slow.sample([0.01, 0.0, 0.0])
    rig_slow.tick()
    assert rig_fast.cmd()[0] > rig_slow.cmd()[0] + 1e-3


# -- device discrete actions (controller press edges, scripted click_action) ------------------
def test_device_click_fires_switch_arm_once_per_press_and_latches_device_action():
    rig = Rig()
    rig.sample([0.0, 0.0, 0.0])
    rig.device(controller=ControllerState())  # idle controller adopted (edge_seq 0)
    rig.tick()
    assert rig.extra()["device_action"] is None and rig.loop.active_arm == "arm0"
    up = _pad_state(0.0, 0.9)
    rig.device(controller=up, click_action="switch_arm")
    rig.tick()
    assert rig.loop.active_arm == "arm1" and rig.extra()["device_action"] == "switch_arm"
    rig.tick(10)  # held: fires once
    assert rig.loop.active_arm == "arm1" and rig.extra()["device_action"] == "switch_arm"
    released = _pad_state(0.0, 0.9, prev=up, click=False)
    rig.device(controller=released, click_action="switch_arm")
    rig.tick(3)
    assert rig.loop.active_arm == "arm1"  # release edge fires nothing
    rig.tick(int(DEVICE_ACTION_LINGER_S / DT))
    assert rig.extra()["device_action"] is None  # ~1 s latch expired
    down = _pad_state(0.0, -0.9, prev=released)
    assert down.edge_seq == 2
    rig.device(controller=down, click_action="switch_arm_prev")
    rig.tick()
    assert rig.loop.active_arm == "arm0" and rig.extra()["device_action"] == "switch_arm_prev"
    # A held-only click (scripted as gripper_open) advances the counter but maps to no action.
    right = _pad_state(0.9, 0.0, prev=_pad_state(0.0, -0.9, prev=down, click=False))
    rig.device(GRIPPER_OPEN_CODE, controller=right, click_action=None)
    rig.tick(3)
    assert rig.loop.active_arm == "arm0" and rig.loop.sources.device == {GRIPPER_OPEN_CODE}


def test_device_click_edges_first_observation_and_stale_samples_do_not_fire():
    rig = Rig()
    rig.sample([0.0, 0.0, 0.0])
    already = _pad_state(0.0, 0.9)  # controller first seen with edge_seq 1: adopt silently
    rig.device(controller=already, click_action="switch_arm")
    rig.tick(3)
    assert rig.loop.active_arm == "arm0"
    nxt = _pad_state(0.0, 0.9, prev=_pad_state(0.0, 0.9, prev=already, click=False))
    rig.controller, rig.click_actions = nxt, ((nxt.edge_seq, "switch_arm"),)
    rig.sample([0.0, 0.0, 0.0], age=0.5, sticky=False)  # edge arrives on a stale sample
    rig.tick(2)
    assert rig.loop.active_arm == "arm0"  # dropped, counter adopted
    rig.sample([0.0, 0.0, 0.0])
    rig.tick(2)
    assert rig.loop.active_arm == "arm0"  # same counter fresh again: nothing
    rig2 = Rig(tracker=False)  # no provider: device edges are inert
    rig2.sample([0.0, 0.0, 0.0])
    rig2.device(controller=ControllerState())
    rig2.tick()
    rig2.device(controller=_pad_state(0.0, 0.9), click_action="switch_arm")
    rig2.tick(2)
    assert rig2.loop.active_arm == "arm0"


# -- entry point: Python logging configured once ------------------------------------------------
def test_entry_point_configures_logging_only_when_unconfigured(monkeypatch):
    import logging
    import sys

    from apollo_mavis_v2_runtime.__main__ import LOG_FORMAT, configure_logging

    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])  # pretend nothing configured logging yet
    monkeypatch.setattr(root, "level", root.level)
    assert configure_logging() is True
    assert len(root.handlers) == 1 and root.level == logging.INFO
    handler = root.handlers[0]
    assert isinstance(handler, logging.StreamHandler) and handler.stream is sys.stderr
    assert handler.formatter is not None and handler.formatter._fmt == LOG_FORMAT
    assert configure_logging() is False and len(root.handlers) == 1  # idempotent
    monkeypatch.setattr(root, "handlers", [object()])  # an embedding app's handler
    assert configure_logging() is False
