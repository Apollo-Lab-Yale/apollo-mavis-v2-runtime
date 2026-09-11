"""GatedPolicyExecutor units over FakeWorkcell: mux, freeze, jump-free,
NaN 3-strike, telemetry deposits, episode-op semantics (12-dagger §2/§3/§12)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from apollo_mavis_v2_core import Command, Pose
from apollo_mavis_v2_core.dagger import ControlMode, TrainerStatus
from apollo_mavis_v2_core.interfaces.policy import Observation, PolicySpec
from apollo_mavis_v2_core.protocol import EpisodeStatus, OnlineDaggerStatus
from apollo_mavis_v2_core.testing import FakeArm, FakeWorkcell

from apollo_mavis_v2_runtime.bus import RuntimeBus
from apollo_mavis_v2_runtime.config import ControlConfig
from apollo_mavis_v2_runtime.dagger.gate import TakeoverGateImpl
from apollo_mavis_v2_runtime.dagger.loop import (
    ALREADY_TAKEN_OVER,
    NOT_ONLINE_DAGGER,
    POLICY_ALREADY_DRIVING,
    POLICY_DRIVING,
    GatedPolicyExecutor,
)
from apollo_mavis_v2_runtime.dagger.policies import ScriptedPolicy
from apollo_mavis_v2_runtime.dagger.policy_runner import (
    ActionAnchor,
    PolicyRunner,
    SlewLimits,
    anchor_leash_kwargs,
    split_action,
)
from apollo_mavis_v2_runtime.dagger.step import policy_step
from apollo_mavis_v2_runtime.recorder.features import arm_action_names
from apollo_mavis_v2_runtime.safety.gate import NullGate
from apollo_mavis_v2_runtime.safety.supervisor import SafetySupervisor
from apollo_mavis_v2_runtime.safety.watchdog import InputWatchdog

ARMS_META = [("arm0", True), ("arm1", False)]
DIM = 8 + 7
NAMES = [f"a{i}" for i in range(DIM)]
# abs_ee: [x, y, z, r6(6), gripper, rail?] -> 11 + 10 (widths from arm_action_names, never
# hard-coded; the r6 of the identity rotation is the first two columns of I)
ABS_NAMES = [n for a, r in ARMS_META for n in arm_action_names(a, r, "abs_ee")]
ABS_DIM = len(ABS_NAMES)
R6_IDENTITY = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
IDENT = np.array([1.0, 0.0, 0.0, 0.0])


class ExecKin:
    """TCP = q[:3] in a world whose arm bases sit at the origin with identity rotation."""

    def tcp_world(self, arm_id, q):
        return Pose(np.array(q[:3], dtype=float), IDENT.copy())

    def base_quat_world(self, arm_id):
        return IDENT.copy()

    def base_world(self, arm_id, q):
        return Pose(np.zeros(3), IDENT.copy())


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


class ReturningStubRecorder(StubRecorder):
    """A save / discard walks ``saving -> returning -> idle`` (one step per ``status()``
    call) like the real recorder under return-to-start (D6: the dagger default), instead
    of snapping to ``idle``."""

    def __init__(self, returning_steps: int = 20):
        super().__init__()
        self.returning_steps = returning_steps
        self.walk: list[str] = []

    def request(self, op):
        if op in ("save", "discard") and self.state == "recording":
            self.state = "saving"
            self.walk = ["saving"] + ["returning"] * self.returning_steps + ["idle"]
            return True, op
        return super().request(op)

    def status(self):
        if self.walk:
            self.state = self.walk.pop(0)
        return super().status()


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


def spec(space="delta_ee"):
    if space == "abs_ee":
        return PolicySpec("abs_ee", "arm_base:arm0", ABS_NAMES, [], [], 0)
    return PolicySpec("delta_ee", "arm_base:arm0", NAMES, [], [], 0)


def build(
    mode="inference",
    policy=None,
    recorder=None,
    reloader=None,
    client=None,
    arm_speed=1.0,  # FakeArm joint speed rad/s: 1.0 settles within the tick, 0.05 LAGS
):
    cell = FakeWorkcell(
        {
            "arm0": FakeArm("arm0", has_rail=True, max_joint_speed_rad_s=arm_speed),
            "arm1": FakeArm("arm1", max_joint_speed_rad_s=arm_speed),
        }
    )
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
    anchor = ActionAnchor(ik, kin, SlewLimits(), action_space=policy.spec.action_space)
    loop = GatedPolicyExecutor(
        cell, ControlConfig(), bus, supervisor, ["arm0", "arm1"],
        gate=gate, runner=runner, anchor=anchor,
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


def _abs_row(x=0.05, grip=0.3, rail=0.2, x1=0.0):
    """One abs_ee whole-cell row: arm0 at (x, 0, 0) identity rotation, gripper ``grip``,
    rail ``rail``; arm1 at (x1, 0, 0) identity, gripper 0.7."""
    a = np.zeros(ABS_DIM, np.float32)
    a[0] = x
    a[3:9] = R6_IDENTITY
    a[9] = grip
    a[10] = rail
    a[11] = x1
    a[14:20] = R6_IDENTITY
    a[20] = 0.7
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


def test_reset_and_goto_are_refused_while_the_policy_drives_and_allowed_after_takeover():
    """Decided 2026-09-08 (04-runtime §10.5): `R` / Go to profile never pre-empt a live
    rollout - the operator takes over first (Space), then the motion is allowed like in
    teleop (the base validation runs: this rig has no profile store, so ITS nack shows).
    Between DAgger episodes the policy is not driving and nothing is refused here."""
    rig = build("inference")
    run(rig, 5)
    for op, args in (("reset_to_initial", {}), ("goto_profile", {"profile_id": "p"})):
        res = act(rig, op, args)
        assert (res.ok, res.detail) == (False, POLICY_DRIVING)
    assert not rig.loop.plans.active_arms
    assert act(rig, "takeover_toggle").ok
    run(rig, 40)  # T_blend -> HUMAN: the human drives, the policy's arms hold
    assert rig.gate.engaged_arm() == "arm0"
    assert act(rig, "reset_to_initial").detail == "no profile store"
    assert act(rig, "goto_profile", {"profile_id": "p"}).detail == "no profile store"
    dagger = build("dagger", recorder=StubRecorder(), client=FakeTrainerClient())
    run(dagger, 3)  # idle recorder: scene staging, the policy is off
    assert act(dagger, "reset_to_initial").detail == "no profile store"
    assert act(dagger, "episode_new").ok
    run(dagger, 3)
    assert act(dagger, "reset_to_initial").detail == "recording - save or discard first"


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


# -- phase-14 (15-online-dagger §3): the coordinator hooks ---------------------------------------
class StubCoordinator:
    """The coordinator surface the executor touches."""

    def __init__(self):
        self.refuse = None
        self.train_calls: list[bool] = []
        self.train_result = (True, "asked the trainer to train (1 rollout saved)")
        self.status_calls = 0
        self.gate_payloads: list[dict] = []

    def refuse_episode_new(self):
        return self.refuse

    def request_train_now(self, episode_open=False):
        self.train_calls.append(episode_open)
        return self.train_result

    def on_gate_events(self, payloads):
        self.gate_payloads.extend(payloads)
        return [("gate", p) for p in payloads]

    def status(self, now=None):
        self.status_calls += 1
        return OnlineDaggerStatus(
            session_name="s1", phase="rollout", rollouts_saved=2, trainer_alive=True,
        )


def test_online_dagger_coordinator_gates_episode_new_and_serves_train_now():
    rec, coord = StubRecorder(), StubCoordinator()
    rig = build("dagger", recorder=rec, reloader=FakeReloader())
    rig.loop.coordinator = coord
    coord.refuse = "waiting for the trainer to report ready (trainer preparing)"
    res = act(rig, "episode_new")
    assert not res.ok and res.detail == coord.refuse and rec.state == "idle"
    coord.refuse = None
    assert act(rig, "episode_new").ok and rec.state == "recording"
    run(rig, 2)
    res = act(rig, "train_now")  # an episode is open: the coordinator is told
    assert res.ok and coord.train_calls == [True]
    assert act(rig, "episode_save").ok
    run(rig, 2)
    coord.train_result = (False, "no Online DAgger trainer attached")
    res = act(rig, "train_now")
    assert (res.ok, res.detail) == coord.train_result and coord.train_calls == [True, False]
    dg = rig.loop._session_extra(0.0)["dagger"]
    assert dg.online_dagger is not None and dg.online_dagger.session_name == "s1"
    assert (dg.online_dagger.phase, dg.online_dagger.rollouts_saved) == ("rollout", 2)
    assert dg.online_dagger.trainer_alive is True
    n = coord.status_calls
    run(rig, 8)  # rebuilt at the telemetry cadence (every 4th tick), not per tick
    assert 1 <= coord.status_calls - n <= 3
    # the recorder's own refusal still follows a coordinator that allows
    rec.state = "recording"
    res = act(rig, "episode_new")
    assert not res.ok and res.detail == "recording"


def test_train_now_is_nacked_outside_an_online_dagger_session(fake_loop):
    rig = build("dagger", recorder=StubRecorder(), reloader=FakeReloader())
    res = act(rig, "train_now")
    assert not res.ok and res.detail == NOT_ONLINE_DAGGER == "not an Online DAgger session"
    assert rig.loop._session_extra(0.0)["dagger"].online_dagger is None
    inf = build("inference")
    assert act(inf, "train_now").detail == NOT_ONLINE_DAGGER
    # a plain teleop ControlLoop nacks the same way
    cell, bus, loop = fake_loop
    fut = bus.commands.submit(Command(op="train_now", args={}, source="ws"))
    loop.run_tick(0.01)
    res = fut.result(timeout=1.0)
    assert not res.ok and res.detail == "not an Online DAgger session"


# -- phase-14 (15-online-dagger D3): the explicit gate API + events.gate --------------------------
def test_takeover_and_handback_are_idempotent_siblings_of_space():
    rec = StubRecorder()
    rig = build("dagger", recorder=rec, reloader=FakeReloader())
    payloads: list[dict] = []
    rig.loop.on_gate_events = payloads.extend
    assert act(rig, "episode_new").ok
    run(rig, 5)
    # handback with nothing engaged: an idempotent ack, no gate event
    res = act(rig, "handback")
    assert res.ok and res.detail == POLICY_ALREADY_DRIVING == "policy already driving"
    assert payloads == [] and rig.gate.engaged_arm() is None
    # takeover: the same transition Space makes, source "action"
    res = act(rig, "takeover")
    assert res.ok and res.detail == "takeover_transition"
    assert rig.gate.engaged_arm() == "arm0"
    assert payloads == [
        {"arm_id": "arm0", "mode": "takeover_transition", "seq": 1, "source": "action",
         "episode_id": None},  # the stub recorder has no open_episode_id
    ]
    # takeover again in TRANSITION: idempotent, nothing moves, no event
    res = act(rig, "takeover")
    assert res.ok and res.detail == ALREADY_TAKEN_OVER == "already taken over"
    assert len(payloads) == 1 and rig.gate.mode("arm0") is ControlMode.TAKEOVER_TRANSITION
    run(rig, 35)  # T_blend -> HUMAN: the auto_advance event is published too
    assert rig.gate.mode("arm0") is ControlMode.HUMAN
    assert payloads[-1] == {"arm_id": "arm0", "mode": "human", "seq": 2, "source": "auto_advance",
                            "episode_id": None}
    res = act(rig, "takeover")  # in HUMAN: idempotent
    assert res.ok and res.detail == ALREADY_TAKEN_OVER and len(payloads) == 2
    # switch_arm still refused while engaged (unchanged)
    assert act(rig, "switch_arm").detail == "takeover active"
    # handback: back to the policy with a fresh query
    resets = rig.policy.resets
    res = act(rig, "handback")
    assert res.ok and res.detail == "policy" and rig.gate.engaged_arm() is None
    run(rig, 10)
    assert rig.policy.resets > resets  # chunk drop + re-query on handback
    assert payloads[-1] == {"arm_id": "arm0", "mode": "policy", "seq": 3, "source": "action",
                            "episode_id": None}
    # handback again: idempotent
    assert act(rig, "handback").detail == POLICY_ALREADY_DRIVING and len(payloads) == 3
    # Space keeps its own source and is published as well
    assert act(rig, "takeover_toggle").detail == "takeover_transition"
    assert payloads[-1]["source"] == "keyboard" and payloads[-1]["seq"] == 4
    assert act(rig, "takeover_toggle").detail == "policy"  # abort inside the blend
    assert payloads[-1] == {"arm_id": "arm0", "mode": "policy", "seq": 5, "source": "keyboard",
                            "episode_id": None}
    # takeover while ANOTHER arm is engaged: nack like Space
    assert act(rig, "switch_arm").ok and rig.loop.active_arm == "arm1"
    assert act(rig, "takeover").detail == "takeover_transition"
    assert act(rig, "switch_arm_prev").detail == "takeover active"
    rig.loop.active_arm = "arm0"  # force the mux onto the other arm
    assert act(rig, "takeover").detail == "takeover active"
    assert act(rig, "takeover_toggle").detail == "takeover active"
    # handback hands back the ENGAGED arm whatever the active one is
    res = act(rig, "handback")
    assert res.ok and res.detail == "policy" and rig.gate.engaged_arm() is None
    assert payloads[-1]["arm_id"] == "arm1" and payloads[-1]["source"] == "action"


def test_episode_boundary_reset_is_published_and_nothing_publishes_without_a_hook():
    """The boundary's ``episode_reset`` events go out too; with ``on_gate_events`` None the
    executor is silent (plain sessions without the bridge)."""
    rec = StubRecorder()
    rig = build("dagger", recorder=rec, reloader=FakeReloader())
    assert act(rig, "episode_new").ok
    run(rig, 3)
    assert act(rig, "takeover").ok  # no hook yet: nothing to publish, nothing raised
    payloads: list[dict] = []
    rig.loop.on_gate_events = payloads.extend
    run(rig, 35)
    assert [p["source"] for p in payloads] == ["auto_advance"]
    assert act(rig, "episode_save").ok  # boundary: the gate resets the held takeover
    run(rig, 2)
    assert rig.gate.engaged_arm() is None
    assert payloads[-1] == {"arm_id": "arm0", "mode": "policy", "seq": 3, "source": "episode_reset",
                            "episode_id": None}
    # a hook that raises never breaks the tick
    rig.loop.on_gate_events = lambda p: 1 / 0
    assert act(rig, "takeover").ok and act(rig, "handback").ok


def test_gate_payload_carries_the_open_episode_id():
    """Gate events name the open episode; the boundary's ``episode_reset`` names the
    episode that just closed (cached at open: the recorder has cleared its id by then,
    whether the boundary runs while the state still reads ``saving`` or after the flip)."""
    class IdRecorder(StubRecorder):
        @property
        def open_episode_id(self):
            return "20260908T100000.000Z-abcdef" if self.state != "idle" else None

    rec = IdRecorder()
    rig = build("dagger", recorder=rec, reloader=FakeReloader())
    payloads: list[dict] = []
    rig.loop.on_gate_events = payloads.extend
    assert act(rig, "episode_new").ok
    run(rig, 2)
    assert act(rig, "takeover").ok
    assert payloads[-1]["episode_id"] == "20260908T100000.000Z-abcdef"
    assert act(rig, "episode_discard").ok
    run(rig, 2)  # the boundary reset fires after the episode closed: it names THAT episode
    assert payloads[-1]["source"] == "episode_reset"
    assert payloads[-1]["episode_id"] == "20260908T100000.000Z-abcdef"


def test_gate_ops_nack_in_teleop_like_space(fake_loop):
    cell, bus, loop = fake_loop
    for op in ("takeover", "handback", "takeover_toggle"):
        fut = bus.commands.submit(Command(op=op, args={}, source="ws"))
        loop.run_tick(0.01)
        res = fut.result(timeout=1.0)
        assert not res.ok and res.detail == "takeover not available in teleop", op
    # inference sessions serve them like dagger ones (Space is the safety escape there too)
    inf = build("inference")
    run(inf, 5)
    assert act(inf, "takeover").detail == "takeover_transition"
    assert act(inf, "handback").detail == "policy"
    assert act(inf, "handback").detail == POLICY_ALREADY_DRIVING


def test_boundary_resets_spell_episode_boundary_and_handback_keeps_its_name():
    rec = StubRecorder()
    rig = build("dagger", recorder=rec, reloader=FakeReloader())
    reasons: list[str] = []
    rig.runner.drop_and_requery = lambda reason="handback": reasons.append(reason)
    assert act(rig, "episode_new").ok
    run(rig, 5)
    assert act(rig, "takeover_toggle").detail == "takeover_transition"
    run(rig, 35)  # T_blend 0.3 s -> HUMAN
    assert act(rig, "takeover_toggle").detail == "policy"  # handback INSIDE the episode
    assert reasons == ["handback"]
    assert act(rig, "episode_save").ok
    run(rig, 2)  # the boundary
    assert reasons == ["handback", "episode_boundary"]
    assert act(rig, "episode_new").ok
    run(rig, 2)
    assert act(rig, "episode_discard").ok
    run(rig, 2)  # a discard is a boundary too
    assert reasons == ["handback", "episode_boundary", "episode_boundary"]


def test_discard_under_return_to_start_is_a_boundary():
    """D6 made return-to-start the dagger default: after a discard the recorder walks
    ``recording -> saving -> returning -> idle`` (the manager flags ``returning`` BEFORE
    the flip to idle), never ``saving -> idle``. The boundary — gate reset, runner
    resume, ``policy_reset(episode_boundary)``, per-episode counters — must fire all the
    same, else a takeover held at discard time stays engaged into the next rollout and
    the policy never drives that arm again."""
    rec = ReturningStubRecorder()
    rig = build("dagger", recorder=rec, reloader=FakeReloader())
    reasons: list[str] = []
    rig.runner.drop_and_requery = lambda reason="handback": reasons.append(reason)
    assert act(rig, "episode_new").ok
    run(rig, 5)
    assert act(rig, "takeover_toggle").detail == "takeover_transition"
    run(rig, 35)  # T_blend 0.3 s -> HUMAN, held through the discard
    assert rig.gate.engaged_arm() == "arm0" and rig.loop._ep_ticks > 0
    rig.runner.pause()  # a NaN-strike pause persists until the boundary
    assert act(rig, "episode_discard").ok
    run(rig, 3)  # saving -> returning: the episode is gone, the arms drive back
    assert rec.state == "returning"
    assert rig.gate.engaged_arm() is None and not rig.runner.paused
    assert rig.loop._ep_ticks == 0 and reasons == ["episode_boundary"]
    run(rig, 20)  # ... -> idle: no second boundary for the same episode
    assert rec.state == "idle" and reasons == ["episode_boundary"]
    # the next rollout starts with the policy driving, no takeover carried over
    assert act(rig, "episode_new").ok
    run(rig, 2)
    assert rig.gate.engaged_arm() is None and rig.loop._policy_active()
    assert act(rig, "takeover_toggle").detail == "takeover_transition"  # Space works again


def test_save_boundary_fires_once_even_when_the_callback_lands_before_the_flip():
    """The recorder thread arms the save boundary BEFORE the state leaves ``saving``; the
    tick may consume it while the state still reads ``saving`` and must not treat the
    following ``saving -> returning -> idle`` walk as a second (discard) boundary."""
    rec = ReturningStubRecorder()
    rig = build("dagger", recorder=rec, reloader=FakeReloader())
    reasons: list[str] = []
    rig.runner.drop_and_requery = lambda reason="handback": reasons.append(reason)
    assert act(rig, "episode_new").ok
    run(rig, 3)
    # the save op and the recorder thread's callback both land before the next tick: the
    # tick consumes the flag while the state still reads `saving`
    fut = rig.bus.commands.submit(Command(op="episode_save", args={}, source="ws"))
    rig.loop.on_episode_saved(0, SimpleNamespace(n_label_frames=0), "/spool/x")
    rig.clock["t"] += 0.01
    rig.loop.run_tick(rig.clock["t"])
    assert fut.result(timeout=1.0).ok
    assert reasons == ["episode_boundary"] and rec.state != "idle"
    run(rig, 20)  # saving -> returning -> idle: not a second (discard) boundary
    assert rec.state == "idle" and reasons == ["episode_boundary"]
    assert act(rig, "episode_new").ok
    run(rig, 2)
    assert act(rig, "episode_discard").ok
    run(rig, 20)
    assert reasons == ["episode_boundary", "episode_boundary"]


def test_train_now_sees_an_episode_opened_in_the_same_tick():
    """``episode_state`` is refreshed at the END of the tick: an ``episode_new`` and a
    ``train_now`` handled in the same tick must still tell the coordinator the episode is
    open (else the trainer is asked to train with a rollout just opened)."""
    rec, coord = StubRecorder(), StubCoordinator()
    rig = build("dagger", recorder=rec, reloader=FakeReloader())
    rig.loop.coordinator = coord
    f_new = rig.bus.commands.submit(Command(op="episode_new", args={}, source="ws"))
    f_train = rig.bus.commands.submit(Command(op="train_now", args={}, source="ws"))
    rig.clock["t"] += 0.01
    rig.loop.run_tick(rig.clock["t"])
    assert f_new.result(timeout=1.0).ok and f_train.result(timeout=1.0).ok
    assert coord.train_calls == [True]


def test_discard_boundary_runs_when_episode_new_lands_right_after_the_flip_to_idle():
    """Without return-to-start (off, or skipped: watchdog latched / no profile / session not
    RUNNING) the recorder flips ``saving -> idle`` on its own thread and the operator's
    ``N`` can land in the very next tick. Commands drain at step 1, BEFORE the boundary
    check reads the recorder, so that read already says ``recording`` (the NEW episode):
    the discard boundary must not be lost - the new rollout must start with the gate in
    POLICY, the runner resumed, ``policy_reset(episode_boundary)`` published and the
    per-episode counters reset. The gate reset carries the CLOSED episode's id."""
    class IdRecorder(StubRecorder):
        ids = iter(["ep-A", "ep-B"])
        current = None

        def request(self, op):
            ok, detail = super().request(op)
            if ok and op == "new":
                self.current = next(self.ids)
            return ok, detail

        @property
        def open_episode_id(self):
            return self.current if self.state != "idle" else None

    rec = IdRecorder()
    rig = build("dagger", recorder=rec, reloader=FakeReloader())
    reasons: list[str] = []
    rig.runner.drop_and_requery = lambda reason="handback": reasons.append(reason)
    payloads: list[dict] = []
    rig.loop.on_gate_events = payloads.extend
    assert act(rig, "episode_new").ok
    run(rig, 5)
    assert act(rig, "takeover_toggle").detail == "takeover_transition"
    run(rig, 35)  # T_blend 0.3 s -> HUMAN, held through the discard
    assert rig.gate.engaged_arm() == "arm0" and rig.loop._ep_ticks > 0
    rig.runner.pause()  # a NaN-strike pause persists until the boundary
    rec.state = "saving"  # the discard op was accepted (recorder thread still clearing)
    run(rig, 1)
    assert rig.gate.engaged_arm() == "arm0" and reasons == []  # not closed yet
    # the recorder thread flips to idle and the operator's N lands in the same tick
    rec.state = "idle"
    assert act(rig, "episode_new").ok
    assert rec.state == "recording" and rec.current == "ep-B"
    assert rig.gate.engaged_arm() is None and not rig.runner.paused
    assert rig.loop._ep_ticks == 0 and reasons == ["episode_boundary"]
    assert payloads[-1]["source"] == "episode_reset" and payloads[-1]["episode_id"] == "ep-A"
    run(rig, 3)
    assert rig.loop._policy_active() and reasons == ["episode_boundary"]
    # the new episode's own boundary is intact: a discard of it fires exactly once
    assert act(rig, "takeover_toggle").detail == "takeover_transition"
    run(rig, 35)
    assert rig.gate.engaged_arm() == "arm0"
    assert act(rig, "episode_discard").ok
    run(rig, 2)
    assert rig.gate.engaged_arm() is None
    assert reasons == ["episode_boundary", "episode_boundary"]
    assert payloads[-1]["source"] == "episode_reset" and payloads[-1]["episode_id"] == "ep-B"


def test_save_boundary_consumed_in_the_reopen_tick_leaves_the_new_episode_its_own_boundary():
    """The save twin of the case above: the save callback armed the flag, the recorder
    flipped to idle and ``N`` landed in the same tick. The boundary fires BEFORE the new
    episode opens (its gate reset is not attributed to the new one) and the new episode
    keeps its own boundary (``_boundary_taken`` was left True here before the fix, so the
    NEXT discard was silently skipped)."""
    rec = StubRecorder()
    rig = build("dagger", recorder=rec, reloader=FakeReloader())
    reasons: list[str] = []
    rig.runner.drop_and_requery = lambda reason="handback": reasons.append(reason)
    assert act(rig, "episode_new").ok
    run(rig, 5)
    assert act(rig, "takeover_toggle").detail == "takeover_transition"
    run(rig, 35)
    assert rig.gate.engaged_arm() == "arm0"
    rec.state = "saving"
    run(rig, 1)
    rig.loop.on_episode_saved(0, SimpleNamespace(n_label_frames=0), "/spool/x")
    rec.state = "idle"
    assert act(rig, "episode_new").ok  # the same tick as the flip
    assert rig.gate.engaged_arm() is None and reasons == ["episode_boundary"]
    run(rig, 3)
    assert act(rig, "takeover_toggle").detail == "takeover_transition"
    run(rig, 35)
    assert act(rig, "episode_discard").ok
    run(rig, 2)
    assert rig.gate.engaged_arm() is None
    assert reasons == ["episode_boundary", "episode_boundary"]


# -- v1.3 (2026-09-11): a policy that drives a SUBSET of the arms (14-dora §6.1) ------------------
class DrivenRunner:
    """A ``PolicySource`` proxy that also declares which arms it drives - the shape of the
    dora ``ExternalPolicySource`` for a Manipulation-Arm-only policy: the undriven arm's
    block is NaN in the whole-cell vector and must be a HOLD, never a NaN strike."""

    def __init__(self, inner, driven):
        self._inner = inner
        self._driven = frozenset(driven)

    def driven_arms(self):
        return self._driven

    def staleness_scale(self, now, arm_id=None):
        return self._inner.staleness_scale(now)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _subset_policy():
    def script(k):
        a = _dx(0.01)  # arm0 moves +x, gripper 0.5
        a[8:] = np.nan  # arm1: "no opinion"
        return a

    return ScriptedPolicy(spec(), script=script)


def test_undriven_arm_holds_and_its_nan_block_is_never_a_strike():
    rig = build("inference", policy=_subset_policy())
    rig.loop.runner = DrivenRunner(rig.runner, {"arm0"})
    q1_before = np.array(rig.cell.arms["arm1"].get_state().q)
    x0 = float(rig.cell.arms["arm0"].get_state().q[0])
    run(rig, 60)
    assert rig.loop._nan_strikes == 0 and not rig.runner.paused
    assert float(rig.cell.arms["arm0"].get_state().q[0]) > x0 + 0.02  # the driven arm moves
    np.testing.assert_allclose(rig.cell.arms["arm1"].get_state().q, q1_before, atol=1e-9)
    cf = rig.loop._last_cf
    assert cf is not None and np.isfinite(cf[:8]).all() and np.isnan(cf[8:]).all()
    # telemetry stays "policy driving", not stale
    extra = rig.loop._session_extra(rig.clock["t"])
    assert extra["inference"].policy_stale is False


def test_without_a_driven_set_a_nan_block_still_trips_the_three_strike_guard():
    """The contract the per-arm source relies on: a source that does NOT declare its driven
    arms is the whole-cell one, so a NaN anywhere is a strike (12-dagger §12 unchanged)."""
    rig = build("inference", policy=_subset_policy())
    run(rig, 60)
    assert rig.loop._nan_strikes >= 3 and rig.runner.paused


# -- 2026-09-11: integrate-on-command + leash (delta_ee), abs_ee waypoints ------------------------
def test_split_action_widths_follow_the_space_names():
    """Block widths come from ``arm_action_names`` for BOTH spaces (8/7 delta, 11/10 abs)."""
    d = split_action(np.arange(DIM, dtype=np.float32), ARMS_META)
    assert d["arm0"].shape == (8,) and d["arm1"].shape == (7,) and d["arm1"][0] == 8
    a = split_action(np.arange(ABS_DIM, dtype=np.float32), ARMS_META, "abs_ee")
    assert a["arm0"].shape == (len(arm_action_names("arm0", True, "abs_ee")),)
    assert a["arm1"].shape == (len(arm_action_names("arm1", False, "abs_ee")),)
    assert a["arm1"][0] == a["arm0"].shape[0]
    assert (ABS_DIM, a["arm0"].shape[0]) == (21, 11)  # the r6 layout: 11 + 10


def test_lagging_arm_command_runs_ahead_to_the_leash_and_intent_accumulates():
    """The anchor is ``FK(q_last)``, not the measured pose: a FakeArm at 0.05 rad/s
    (0.5 mm / tick) cannot follow the 2 mm / tick command, so the command runs AHEAD of the
    measured pose up to the 25 mm leash and stays there while the arm keeps integrating the
    policy's intent. Under measured re-anchoring the gap could never exceed one tick's 2 mm
    and the arm would crawl at 0.5 mm / tick."""
    rig = build("inference", arm_speed=0.05)
    gaps = []
    for _ in range(60):
        _tick_once(rig)
        cmd = float(rig.loop._last_cmd["arm0"][0])
        meas = float(rig.cell.arms["arm0"].get_state().q[0])
        gaps.append(cmd - meas)
    leash = rig.loop.anchor.leash_pos_m
    assert max(gaps) > 0.01  # far beyond one tick's 2 mm: the intent accumulated
    assert max(gaps) <= leash + 1e-9  # ... and never past the leash to the measured pose
    # steady state: leashed to the measured pose at the START of the tick (the FakeArm then
    # moves its 0.5 mm within the same tick, so the gap read after it is leash - 0.5 mm)
    assert gaps[-1] == pytest.approx(leash - 0.0005, abs=1e-6)
    # and the measured arm moved at its own full speed on every tick that had a command
    # (the first policy output lands at t = 0.05, the 5th tick): 56 x 0.5 mm
    assert rig.cell.arms["arm0"].get_state().q[0] == pytest.approx(56 * 0.0005, abs=1e-9)


def test_leash_defaults_follow_the_control_config_and_the_dagger_override():
    from apollo_mavis_v2_runtime.config import DaggerConfig, LeashConfig

    ccfg = ControlConfig()
    assert anchor_leash_kwargs(DaggerConfig(), ccfg) == {"leash_pos_m": 0.025, "leash_rot_rad": 0.2}
    dcfg = DaggerConfig(anchor_leash=LeashConfig(pos_m=0.01, rot_rad=0.1))
    kw = anchor_leash_kwargs(dcfg, ccfg)
    assert kw == {"leash_pos_m": 0.01, "leash_rot_rad": 0.1}
    anchor = ActionAnchor(ExecIK(), ExecKin(), SlewLimits(), **kw)
    assert (anchor.leash_pos_m, anchor.leash_rot_rad) == (0.01, 0.1)
    assert anchor.action_space == "delta_ee"


def _abs_rig(mode="inference", **kw):
    policy = ScriptedPolicy(spec("abs_ee"), script=lambda k: _abs_row())
    return build(mode, policy=policy, **kw)


def test_abs_ee_row_is_reached_at_its_deadline_with_gripper_and_absolute_rail():
    """Deadline interpolation: the row becomes current at t_row and the command REACHES it
    one period (50 ms = 5 ticks) later - a linear ramp, never a jump; the rail slot walks to
    its absolute target with the same fraction; the gripper comes from index 9."""
    rig = _abs_rig()
    xs, rails = [], []
    for _ in range(5):  # t = 0.01 .. 0.05: the first query lands at t = 0.05
        _tick_once(rig)
        xs.append(float(rig.loop._last_cmd["arm0"][0]))
        rails.append(float(rig.loop._last_cmd["arm0"][7]))
    assert xs[:4] == [0.0] * 4  # no output yet: hold
    assert xs[4] == pytest.approx(0.05 * 0.2)  # tick of the row: 1/5 of the way
    for _ in range(4):
        _tick_once(rig)
        xs.append(float(rig.loop._last_cmd["arm0"][0]))
        rails.append(float(rig.loop._last_cmd["arm0"][7]))
    np.testing.assert_allclose(xs[4:], 0.05 * np.array([0.2, 0.4, 0.6, 0.8, 1.0]), atol=1e-9)
    np.testing.assert_allclose(rails[4:], 0.2 * np.array([0.2, 0.4, 0.6, 0.8, 1.0]), atol=1e-9)
    assert rig.loop._grip_frac["arm0"] == pytest.approx(0.3)  # gripper.pos at index 9
    run(rig, 20)  # steady state: at the waypoint, no overshoot, arm1 untouched at x1 = 0
    assert rig.loop._last_cmd["arm0"][0] == pytest.approx(0.05, rel=1e-5)
    assert rig.loop._last_cmd["arm0"][7] == pytest.approx(0.2, rel=1e-5)
    assert rig.loop._last_cmd["arm1"][0] == pytest.approx(0.0, abs=1e-9)
    # the counterfactual keeps the delta_ee width: the remaining delta per FRAME (zero at the
    # waypoint), gripper absolute
    cf = rig.loop._last_cf
    assert cf is not None and cf.shape == (DIM,)
    assert cf[0] == pytest.approx(0.0, abs=1e-6) and cf[6] == pytest.approx(0.3)
    assert cf[8 + 6] == pytest.approx(0.7)


def test_abs_ee_counterfactual_is_the_per_frame_delta_toward_the_waypoint():
    """Before the arm gets there the deposit is (waypoint - FK(q_last)) * dt/period *
    frame_scale, NOT the absolute pose scaled (which would drift toward the origin)."""
    rig = _abs_rig()
    run(rig, 5)  # the row lands at t = 0.05; the command is at 1/5 of the way (0.01)
    cf = rig.loop._last_cf
    s = (rig.loop.dt / rig.runner.period) * rig.loop._frame_scale  # 0.2 * 4 = 0.8 per frame
    # _update_counterfactual runs BEFORE this tick's step: the delta is from x = 0 (t=0.05)
    assert cf[0] == pytest.approx(0.05 * s) and cf[7] == pytest.approx(0.2 * s)
    assert np.all(cf[1:6] == 0.0)


class _AbsStubRunner:
    """A ``PolicySource`` shape for direct ``policy_step`` calls."""

    def __init__(self, row, t_row, stale=1.0, period=0.05):
        self.spec = spec("abs_ee")
        self.period = period
        self._row = row
        self._t = t_row
        self.stale = stale

    def latest(self):
        if self._row is None:
            return None, -1e9
        return SimpleNamespace(actions=self._row), self._t

    def staleness_scale(self, now, arm_id=None):
        return self.stale

    def driven_arms(self):
        return None


def _abs_step(runner, now=0.06, q_last=None, dt=0.01):
    ik, kin = ExecIK(), ExecKin()
    anchor = ActionAnchor(ik, kin, SlewLimits(), action_space="abs_ee")
    q_last = np.zeros(8) if q_last is None else q_last
    grips = []
    q = policy_step(
        runner=runner, anchor=anchor, arms_meta=ARMS_META, arm_id="arm0",
        state=SimpleNamespace(q=q_last.copy()), q_last=q_last, now=now, dt=dt,
        on_gripper=lambda a, g: grips.append((a, g)),
    )
    return q, grips


def test_policy_step_abs_ee_holds_when_stale_or_non_finite_and_never_scales_by_staleness():
    fresh = _AbsStubRunner(_abs_row(), t_row=0.05)
    q, grips = _abs_step(fresh)
    assert q is not None and q[0] == pytest.approx(0.05 * 0.25)  # 0.04 s left -> dt/0.04
    assert q[7] == pytest.approx(0.2 * 0.25, rel=1e-5)
    assert len(grips) == 1 and grips[0][0] == "arm0" and grips[0][1] == pytest.approx(0.3)
    # a partially stale row is NOT scaled toward the origin: same interpolation
    half = _AbsStubRunner(_abs_row(), t_row=0.05, stale=0.5)
    q_half, _ = _abs_step(half)
    np.testing.assert_allclose(q_half, q)
    # staleness_scale <= 0 -> hold (None); a NaN block for this arm -> hold; no output -> hold
    assert _abs_step(_AbsStubRunner(_abs_row(), t_row=0.05, stale=0.0)) == (None, [])
    row = _abs_row()
    row[3] = np.nan
    assert _abs_step(_AbsStubRunner(row, t_row=0.05)) == (None, [])
    assert _abs_step(_AbsStubRunner(None, t_row=0.05)) == (None, [])
    # past the deadline the remaining time floors at dt: the row is reached in one tick
    late = _AbsStubRunner(_abs_row(), t_row=0.0)
    q_late, _ = _abs_step(late, now=1.0)
    assert q_late[0] == pytest.approx(0.05, rel=1e-5) and q_late[7] == pytest.approx(0.2, rel=1e-5)
    # a degenerate rotation (parallel r6 columns) is a hold, not a crash - and the gripper
    # of a rejected row is not applied either
    bad = _abs_row()
    bad[3:9] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    assert _abs_step(_AbsStubRunner(bad, t_row=0.05)) == (None, [])


def test_abs_ee_handback_window_caps_the_per_tick_step():
    """Inside the handback slew window an abs waypoint approaches at most lin_mps*dt per
    tick (1.5 mm) even when the deadline interpolation asks for more."""
    ik, kin = ExecIK(), ExecKin()
    anchor = ActionAnchor(ik, kin, SlewLimits(), action_space="abs_ee")
    from apollo_mavis_v2_core.dagger import GateEvent

    anchor.on_gate_event(
        GateEvent(arm_id="arm0", mode=ControlMode.POLICY, seq=1, t_mono=0.0, source="keyboard")
    )
    row = _abs_row(x=0.5)
    q = anchor.apply_absolute("arm0", row, np.zeros(8), np.zeros(8), 0.01, 0.01, 0.05, 0.0)
    assert q[0] == pytest.approx(SlewLimits().lin_mps * 0.01)  # 1.5 mm, not 0.5 * 0.2
    q = anchor.apply_absolute("arm0", row, np.zeros(8), np.zeros(8), 0.01, 1.0, 0.05, 1.0)
    assert q[0] == pytest.approx(0.5 * 0.2)  # window closed: the deadline fraction dt/period
