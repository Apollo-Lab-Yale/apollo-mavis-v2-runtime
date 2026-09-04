"""GatedPolicyExecutor units over FakeWorkcell: mux, freeze, jump-free,
NaN 3-strike, telemetry deposits, episode-op semantics (12-dagger §2/§3/§12)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from apollo_mavis_v2_core import Command, Pose
from apollo_mavis_v2_core.dagger import ControlMode, TrainerStatus
from apollo_mavis_v2_core.interfaces.policy import Observation, PolicySpec
from apollo_mavis_v2_core.protocol import EpisodeStatus
from apollo_mavis_v2_core.testing import FakeArm, FakeWorkcell

from apollo_mavis_v2_runtime.bus import RuntimeBus
from apollo_mavis_v2_runtime.config import ControlConfig
from apollo_mavis_v2_runtime.dagger.gate import TakeoverGateImpl
from apollo_mavis_v2_runtime.dagger.loop import GatedPolicyExecutor
from apollo_mavis_v2_runtime.dagger.policies import ScriptedPolicy
from apollo_mavis_v2_runtime.dagger.policy_runner import ActionAnchor, PolicyRunner, SlewLimits
from apollo_mavis_v2_runtime.safety.gate import NullGate
from apollo_mavis_v2_runtime.safety.supervisor import SafetySupervisor
from apollo_mavis_v2_runtime.safety.watchdog import InputWatchdog

ARMS_META = [("arm0", True), ("arm1", False)]
DIM = 8 + 7
NAMES = [f"a{i}" for i in range(DIM)]


class ExecKin:
    def tcp_world(self, arm_id, q):
        return Pose(np.array(q[:3], dtype=float), np.array([1.0, 0.0, 0.0, 0.0]))

    def base_quat_world(self, arm_id):
        return np.array([1.0, 0.0, 0.0, 0.0])


class ExecIK:
    def solve(self, arm_id, target, q_last):
        q = np.array(q_last, dtype=float)
        q[:3] = target.position
        return SimpleNamespace(q=q, diverged=False, pos_err_m=0.0, rot_err_rad=0.0)

    def sync_passive(self, states):
        pass

    def reset(self, arm_id, q):
        pass


class StubRecorder:
    """Just enough episode state machine for the executor's boundary logic."""

    def __init__(self):
        self.state = "idle"
        self.fps = 25

    def request(self, op):
        if op == "new" and self.state == "idle":
            self.state = "recording"
            return True, "episode 0"
        if op in ("save", "discard") and self.state == "recording":
            self.state = "idle"
            return True, op
        return False, self.state

    def status(self):
        return EpisodeStatus(state=self.state, index=0, frames=0, duration_s=0.0)


class FakeReloader:
    def __init__(self):
        self.current_version = 0
        self.rollbacks = 0
        self.goods = 0
        self.swap_calls: list[bool] = []

    def staged_version(self):
        return None

    def maybe_swap(self, at_boundary, mode):
        self.swap_calls.append(at_boundary)
        return None

    def rollback(self):
        self.rollbacks += 1
        return 0

    def mark_good(self):
        self.goods += 1


class FakeTrainerClient:
    def __init__(self):
        self.submitted = []

    def status(self):
        return TrainerStatus(state="idle", new_label_frames=42)

    def submit_episode(self, path, summary):
        self.submitted.append((path, summary))


def spec():
    return PolicySpec("delta_ee", "arm_base:arm0", NAMES, [], [], 0)


def build(mode="inference", policy=None, recorder=None, reloader=None, client=None):
    cell = FakeWorkcell({"arm0": FakeArm("arm0", has_rail=True), "arm1": FakeArm("arm1")})
    cell.start()
    bus = RuntimeBus()
    supervisor = SafetySupervisor(NullGate(), InputWatchdog())
    gate = TakeoverGateImpl(["arm0", "arm1"])
    policy = policy or ScriptedPolicy(spec(), script=lambda k: _dx(0.01))
    clock = {"t": 0.0}
    runner = PolicyRunner(
        policy,
        lambda: Observation(np.zeros(4, np.float32), {}, clock["t"], 0),
        rate_hz=20.0, clock=lambda: clock["t"],
    )
    ik, kin = ExecIK(), ExecKin()
    loop = GatedPolicyExecutor(
        cell, ControlConfig(), bus, supervisor, ["arm0", "arm1"],
        gate=gate, runner=runner, anchor=ActionAnchor(ik, kin, SlewLimits()),
        arms_meta=ARMS_META, session_mode=mode,
        run_id="run1" if mode == "dagger" else "",
        version_label="deploy/v001" if mode == "inference" else None,
        reloader=reloader, trainer_client=client,
        recorder=recorder, ik=ik, kin=kin,
        clock=lambda: clock["t"],
    )
    loop._seed_from_measured()
    return SimpleNamespace(cell=cell, bus=bus, gate=gate, runner=runner,
                           loop=loop, clock=clock, policy=policy)


def _dx(v):
    a = np.zeros(DIM, np.float32)
    a[0] = v  # arm0 ee.dx per policy period
    a[6] = 0.5  # arm0 absolute gripper target
    return a


def _tick_once(rig):
    """One deterministic 100 Hz tick; policy queried at 20 Hz; commands are
    forwarded to the FakeArms by hand (no ArmSender threads in unit tests)."""
    rig.clock["t"] += 0.01
    t = rig.clock["t"]
    if int(round(t * 100)) % 5 == 0:
        rig.runner.step_once(t)
    snap = rig.loop.run_tick(t)
    for arm_id in ("arm0", "arm1"):
        rig.cell.arms[arm_id].command_joints(rig.loop._last_cmd[arm_id])
    rig.cell.step(0.01)
    return snap


def run(rig, n, held=frozenset()):
    return [_tick_once(rig) for _ in range(n)]


def act(rig, op, args=None):
    fut = rig.bus.commands.submit(Command(op=op, args=args or {}, source="ws"))
    rig.clock["t"] += 0.01
    rig.loop.run_tick(rig.clock["t"])
    return fut.result(timeout=1.0)


def test_inference_policy_drives_and_escape_toggles():
    rig = build("inference")
    run(rig, 50)
    q = rig.cell.arms["arm0"].get_state().q
    assert q[0] > 0.005  # dx flowed to the arm
    assert rig.loop._session_extra(0.0)["inference"].control_mode is ControlMode.POLICY
    res = act(rig, "takeover_toggle")
    assert res.ok and res.detail == "takeover_transition"
    run(rig, 40)  # T_blend 0.3 s -> HUMAN
    st = rig.loop._session_extra(0.0)["inference"]
    assert st.control_mode is ControlMode.HUMAN and st.engaged_arm == "arm0"
    assert st.policy_version == "deploy/v001"
    x_frozen = rig.cell.arms["arm0"].get_state().q[0]
    run(rig, 30)  # no held keys: human holds; policy must NOT drive
    assert rig.cell.arms["arm0"].get_state().q[0] == pytest.approx(x_frozen, abs=1e-9)
    res = act(rig, "takeover_toggle")  # handback
    assert res.ok and rig.gate.engaged_arm() is None
    run(rig, 10)  # next policy tick performs reset + fresh query
    assert rig.policy.resets >= 1  # chunk drop + re-query on handback


def test_jump_free_switching_max_dq_bounded():
    rig = build("inference")
    dq_max = rig.loop.cfg.dq_max_rad
    last = [np.array(rig.loop._last_cmd["arm0"])]

    def max_step(n):
        worst = 0.0
        for _ in range(n):
            _tick_once(rig)
            q = np.array(rig.loop._last_cmd["arm0"])
            worst = max(worst, float(np.max(np.abs(q[:7] - last[0][:7]))))
            last[0] = q
        return worst

    assert max_step(30) <= dq_max + 1e-9
    act(rig, "takeover_toggle")  # policy -> human
    assert max_step(40) <= dq_max + 1e-9
    act(rig, "takeover_toggle")  # human -> policy (slew window)
    assert max_step(60) <= dq_max + 1e-9


def test_freeze_other_arms_and_switch_arm_rejected():
    rig = build("inference")
    run(rig, 10)
    act(rig, "takeover_toggle")  # engage arm0 (active)
    q1 = rig.cell.arms["arm1"].get_state().q.copy()
    run(rig, 30)
    assert np.allclose(rig.cell.arms["arm1"].get_state().q, q1)  # frozen hold
    res = act(rig, "switch_arm")
    assert not res.ok and res.detail == "takeover active"
    st = rig.loop._session_extra(0.0)["inference"]
    assert st.engaged_arm == "arm0"


def test_episode_ops_nack_in_inference():
    rig = build("inference")
    for op in ("episode_new", "episode_save", "episode_discard"):
        res = act(rig, op)
        assert not res.ok and res.detail == "no recorder in this mode"


def test_dagger_policy_gated_on_recording_and_boundary_hooks():
    rec, rel, cli = StubRecorder(), FakeReloader(), FakeTrainerClient()
    rig = build("dagger", recorder=rec, reloader=rel, client=cli)
    run(rig, 30)
    assert rig.cell.arms["arm0"].get_state().q[0] == pytest.approx(0.0)  # idle: no policy
    assert act(rig, "episode_new").ok
    run(rig, 50)
    assert rig.cell.arms["arm0"].get_state().q[0] > 0.005  # recording: policy drives
    dg = rig.loop._session_extra(0.0)["dagger"]
    assert dg.policy_version == "run1/v000000"
    assert dg.trainer.state == "idle" and dg.new_label_frames == 42
    act(rig, "takeover_toggle")
    run(rig, 35)
    dg = rig.loop._session_extra(0.0)["dagger"]
    assert dg.control_mode is ControlMode.HUMAN and dg.frozen_arms == ["arm1"]
    assert act(rig, "episode_save").ok  # boundary: gate reset + swap point
    run(rig, 2)
    assert rig.gate.engaged_arm() is None  # reset at episode boundary
    assert rel.swap_calls and all(rel.swap_calls)  # ONLY boundary swaps
    assert rel.goods >= 1  # clean episode advanced LAST_KNOWN_GOOD


def test_nan_three_strikes_pause_hold_rollback():
    policy = ScriptedPolicy(spec(), script=lambda k: _dx(0.01), nan_at={5, 7, 9})
    rec, rel = StubRecorder(), FakeReloader()
    rig = build("dagger", policy=policy, recorder=rec, reloader=rel)
    act(rig, "episode_new")
    run(rig, 80)
    assert rig.runner.paused  # 3 NaN outputs in one episode -> all arms hold
    assert rel.rollbacks == 1
    assert len(rig.loop.anomalies) == 1 and rig.loop.anomalies[0].kind == "nan"
    x = rig.cell.arms["arm0"].get_state().q[0]
    run(rig, 20)
    assert rig.cell.arms["arm0"].get_state().q[0] == pytest.approx(x, abs=1e-9)
    ann = rig.loop._session_extra(0.0)["dagger_frame"]
    assert ann["policy_action"] is None  # unqueried -> recorder writes NaN row
    assert act(rig, "episode_save").ok  # boundary clears the strike state
    run(rig, 2)
    assert not rig.runner.paused
    assert rel.goods == 0  # dirty episode never advances LAST_KNOWN_GOOD


def test_counterfactual_deposit_present_during_policy_and_human():
    rec = StubRecorder()
    rig = build("dagger", recorder=rec, reloader=FakeReloader())
    act(rig, "episode_new")
    run(rig, 10)
    ann = rig.loop._session_extra(0.0)["dagger_frame"]
    assert ann["control_mode"] == 0 and ann["action_source"] == 0
    assert ann["policy_action"] is not None and ann["policy_action"].shape == (DIM,)
    act(rig, "takeover_toggle")
    run(rig, 40)
    ann = rig.loop._session_extra(0.0)["dagger_frame"]
    assert ann["control_mode"] == 1 and ann["action_source"] == 3
    assert ann["policy_action"] is not None  # queried even during HUMAN (§4)


def test_switch_arm_prev_wraps_and_is_nacked_during_takeover():
    rig = build("inference")
    assert rig.loop.active_arm == "arm0"
    res = act(rig, "switch_arm_prev")
    assert res.ok and res.detail == "arm1"  # (0 - 1) mod 2 wraps
    res = act(rig, "switch_arm_prev")
    assert res.ok and res.detail == "arm0"
    assert act(rig, "takeover_toggle").ok  # human engaged on arm0
    res = act(rig, "switch_arm_prev")
    assert not res.ok and res.detail == "takeover active"
    res = act(rig, "switch_arm")
    assert not res.ok and res.detail == "takeover active"
    assert rig.loop.active_arm == "arm0"
