"""ControlLoop unit behaviour over FakeWorkcell: jog/goto, switch_arm, acks."""

from __future__ import annotations

import time

import numpy as np
import pytest
from apollo_mavis_v2_core import Command, CommandSource, HeldState, PlanResult
from conftest import run_ticks


def submit(bus, op, **args):
    return bus.commands.submit(Command(op=op, args=args))


def test_jog_slew_limited(fake_loop):
    cell, bus, loop = fake_loop
    fut = submit(bus, "joint_target", arm_id="arm0", positions=[0.1] * 7 + [0.05], mode="jog")
    t = run_ticks(loop, cell, 1)
    assert fut.result(0).ok
    # After one tick: exactly one slew step (0.02 rad joints, 0.002 m rail).
    q = loop._last_cmd["arm0"]
    assert np.allclose(q[:7], 0.02)
    assert abs(q[7] - 0.002) < 1e-12
    run_ticks(loop, cell, 30, t)
    assert np.allclose(loop._last_cmd["arm0"][:7], 0.1)
    assert abs(loop._last_cmd["arm0"][7] - 0.05) < 1e-12


def test_large_jog_is_accepted_and_walked_at_the_slew_rate(fake_loop):
    """No `goto_threshold_rad` any more (2026-09-07): a jog is a DESTINATION, so a
    delta of any size is accepted and approached at `slew_rad_per_tick` — the UI
    joint panel relies on this for slider drags and typed values alike."""
    cell, bus, loop = fake_loop
    target = 1.5  # 10x the old 0.15 rad threshold
    fut = submit(bus, "joint_target", arm_id="arm0", positions=[target] * 7 + [0.0], mode="jog")
    run_ticks(loop, cell, 1)
    res = fut.result(0)
    assert res.ok and res.detail == "jog"
    slew = loop.cfg.jog.slew_rad_per_tick
    # One tick of motion, not a jump: still slew-many radians from the start.
    assert loop._last_cmd["arm0"][0] == pytest.approx(slew, abs=1e-9)
    run_ticks(loop, cell, 9, 0.01)
    assert loop._last_cmd["arm0"][0] == pytest.approx(10 * slew, abs=1e-9)
    assert loop._last_cmd["arm0"][0] < target  # still on its way


def test_jog_nacked_while_recording(fake_loop):
    cell, bus, loop = fake_loop
    loop.episode_state = "recording"
    fut = submit(bus, "joint_target", arm_id="arm0", positions=[0.05] * 7 + [0.0], mode="jog")
    run_ticks(loop, cell, 1)
    res = fut.result(0)
    assert not res.ok and res.detail == "recording"


def test_wrong_dof_nacked(fake_loop):
    cell, bus, loop = fake_loop
    fut = submit(bus, "joint_target", arm_id="arm0", positions=[0.0] * 7, mode="jog")
    run_ticks(loop, cell, 1)
    assert not fut.result(0).ok  # arm0 has a rail: needs 8 slots


def test_switch_arm_cycles_server_side(fake_loop):
    cell, bus, loop = fake_loop
    assert loop.active_arm == "arm0"
    fut = submit(bus, "switch_arm")
    run_ticks(loop, cell, 1)
    assert fut.result(0).ok and loop.active_arm == "arm1"
    submit(bus, "switch_arm")
    run_ticks(loop, cell, 1)
    assert loop.active_arm == "arm0"  # wraps around


def test_switch_arm_with_an_arm_id_selects_explicitly(fake_loop):
    """The Cockpit's clickable arm rows (05-ui §8.2): pick, do not cycle."""
    cell, bus, loop = fake_loop
    f1 = submit(bus, "switch_arm", arm_id="arm1")
    run_ticks(loop, cell, 1)
    assert f1.result(0).ok and loop.active_arm == "arm1"
    # Re-selecting the active arm is an accepted no-op (a click on the active
    # row must not release a live tracker clutch).
    f2 = submit(bus, "switch_arm", arm_id="arm1")
    run_ticks(loop, cell, 1)
    assert f2.result(0).ok and loop.active_arm == "arm1"
    f3 = submit(bus, "switch_arm", arm_id="arm0")
    run_ticks(loop, cell, 1)
    assert f3.result(0).ok and loop.active_arm == "arm0"
    # An arm outside the session is refused; the active arm does not change.
    f4 = submit(bus, "switch_arm", arm_id="nope")
    run_ticks(loop, cell, 1)
    assert not f4.result(0).ok and loop.active_arm == "arm0"


def test_unknown_op_and_episode_ops_nacked(fake_loop):
    cell, bus, loop = fake_loop
    f1 = submit(bus, "bogus_op")
    f2 = submit(bus, "episode_new")
    f3 = submit(bus, "takeover_toggle")
    run_ticks(loop, cell, 1)
    assert not f1.result(0).ok
    assert not f2.result(0).ok and "recorder" in f2.result(0).detail
    assert not f3.result(0).ok


class FakePlanner:
    """Scriptable twin.plan for goto tests."""

    def __init__(self, result: PlanResult) -> None:
        self.result = result
        self.requests = []

    def plan(self, req):
        self.requests.append(req)
        return self.result


def test_goto_routes_via_planner_and_completes(fake_loop):
    cell, bus, loop = fake_loop
    goal = [0.3] * 7 + [0.1]
    loop.planner = FakePlanner(PlanResult(ok=True, waypoints={"arm0": [goal]}))
    fut = submit(bus, "joint_target", arm_id="arm0", positions=goal, mode="goto")
    t = run_ticks(loop, cell, 2)
    res = fut.result(1.0)  # ack immediate
    assert res.ok and res.detail == "accepted"
    # Wait for the plan worker round-trip, then stream waypoints.
    for _ in range(100):
        t = run_ticks(loop, cell, 1, t)
        if not loop.plans.active("arm0") and loop.tick_count > 5:
            if np.allclose(loop._last_cmd["arm0"], goal):
                break
            time.sleep(0.002)  # yield to the plan-worker thread
    assert np.allclose(loop._last_cmd["arm0"], goal)
    assert loop._plan_status in ("done", None)


def test_held_movement_key_cancels_plan(fake_loop):
    cell, bus, loop = fake_loop
    goal = [0.5] * 7 + [0.2]
    loop.planner = FakePlanner(PlanResult(ok=True, waypoints={"arm0": [goal]}))
    submit(bus, "joint_target", arm_id="arm0", positions=goal, mode="goto")
    t = run_ticks(loop, cell, 2)
    for _ in range(50):  # wait until executing
        t = run_ticks(loop, cell, 1, t)
        if loop.plans.active("arm0"):
            break
        time.sleep(0.002)  # yield to the plan-worker thread
    assert loop.plans.active("arm0")
    bus.held_keys.put(HeldState(held=frozenset({"KeyW"}), seq=1, rx_mono=t))
    loop.supervisor.watchdog.on_keys(HeldState(frozenset({"KeyW"}), 1, t))
    t = run_ticks(loop, cell, 1, t)
    assert not loop.plans.active("arm0")
    assert loop._plan_status == "cancelled"
    assert not np.allclose(loop._last_cmd["arm0"], goal)  # stopped short


class _Sender:
    """Stands in for the ArmSender thread: records gripper puts."""

    def __init__(self):
        self.puts: list[float] = []

    def put_gripper(self, open_frac: float) -> None:
        self.puts.append(open_frac)


def test_gripper_keys_and_targets_skip_gripperless_arm():
    """arm0 is camera-only (no gripper): F/H, execute_plan gripper targets
    and the snapshot never touch it; the same inputs drive arm1 normally."""
    from apollo_mavis_v2_core.testing import FakeArm, FakeWorkcell

    from apollo_mavis_v2_runtime.bus import RuntimeBus
    from apollo_mavis_v2_runtime.config import ControlConfig
    from apollo_mavis_v2_runtime.control.loop import ControlLoop
    from apollo_mavis_v2_runtime.safety.gate import NullGate
    from apollo_mavis_v2_runtime.safety.supervisor import SafetySupervisor
    from apollo_mavis_v2_runtime.safety.watchdog import InputWatchdog

    cell = FakeWorkcell({"arm0": FakeArm("arm0"), "arm1": FakeArm("arm1")})
    cell.start()
    bus = RuntimeBus()
    loop = ControlLoop(
        cell,
        ControlConfig(),
        bus,
        SafetySupervisor(NullGate(), InputWatchdog()),
        ["arm0", "arm1"],
        gripper_arms=["arm1"],
    )
    senders = {a: _Sender() for a in ("arm0", "arm1")}
    loop._senders.update(senders)  # what start() wires, minus the threads

    def hold(key: str, n: int, t: float) -> float:
        for _ in range(n):  # fresh KeysMsg every tick: deadman scale stays 1.0
            t += loop.dt
            hs = HeldState(held=frozenset({key}), seq=loop.tick_count + 1, rx_mono=t)
            loop.supervisor.watchdog.on_keys(hs)
            bus.held_keys.put(hs)
            loop.run_tick(t)
            cell.step(loop.dt)
        return t

    assert loop.active_arm == "arm0" and loop.gripper_arms == {"arm1"}
    t = hold("KeyF", 20, 0.0)  # close on the camera-only arm: ignored
    snap = loop.run_tick(t + loop.dt)
    assert "arm0" not in loop._grip_frac and "arm0" not in snap.gripper_frac
    assert senders["arm0"].puts == [] and senders["arm1"].puts == []

    # start_from gripper targets: only the gripper arm receives one.
    submit(bus, "execute_plan", waypoints={}, gripper={"arm0": 0.3, "arm1": 0.3})
    t = run_ticks(loop, cell, 1, t)
    assert loop._grip_frac == {"arm1": 0.3}
    assert senders["arm0"].puts == [] and senders["arm1"].puts == [0.3]

    # Tab to arm1 (has a gripper): F/H integrate and reach the sender.
    submit(bus, "switch_arm")
    t = hold("KeyH", 20, t)
    assert loop.active_arm == "arm1"
    assert loop._grip_frac["arm1"] > 0.3 and len(senders["arm1"].puts) > 1
    assert senders["arm0"].puts == []


# -- default active arm: the Manipulation Arm (grip) whatever its position ------------
def _loop_over(tmp_path, arm_ids: list[str]):
    """ControlLoop over FakeWorkcell for ``arm_ids`` (same wiring as ``fake_loop``)."""
    from apollo_mavis_v2_core import ProfileStore
    from apollo_mavis_v2_core.testing import FakeArm, FakeWorkcell

    from apollo_mavis_v2_runtime.bus import RuntimeBus
    from apollo_mavis_v2_runtime.config import ControlConfig
    from apollo_mavis_v2_runtime.control.loop import ControlLoop
    from apollo_mavis_v2_runtime.safety.gate import NullGate
    from apollo_mavis_v2_runtime.safety.supervisor import SafetySupervisor
    from apollo_mavis_v2_runtime.safety.watchdog import InputWatchdog

    cell = FakeWorkcell({a: FakeArm(a, has_rail=True) for a in arm_ids})
    cell.start()
    bus = RuntimeBus()
    loop = ControlLoop(
        cell,
        ControlConfig(),
        bus,
        SafetySupervisor(NullGate(), InputWatchdog()),
        list(arm_ids),
        profile_store=ProfileStore(tmp_path / "profiles"),
        workcell_kind="sim",
    )
    return cell, bus, loop


def test_default_active_arm_helper():
    from apollo_mavis_v2_runtime.control.loop import DEFAULT_ACTIVE_ARM, default_active_arm

    assert DEFAULT_ACTIVE_ARM == "grip"
    assert default_active_arm(["view", "grip"]) == "grip"  # not spec.arms[0]
    assert default_active_arm(["grip", "view"]) == "grip"
    assert default_active_arm(["grip"]) == "grip"
    assert default_active_arm(["view"]) == "view"  # no grip: first arm
    assert default_active_arm(["arm0", "arm1"]) == "arm0"
    assert default_active_arm([]) is None


def test_session_with_view_then_grip_starts_on_grip(tmp_path):
    """A session created with arms ['view', 'grip'] reports active arm grip from the
    first tick (telemetry snapshot included); Tab / switch_arm_prev cycle from there."""
    cell, bus, loop = _loop_over(tmp_path, ["view", "grip"])
    assert loop.active_arm == "grip"
    run_ticks(loop, cell, 2)
    assert bus.snapshot.get()[0].active_arm == "grip"
    fut = submit(bus, "switch_arm")
    run_ticks(loop, cell, 1)
    assert fut.result(0).ok and fut.result(0).detail == "view" and loop.active_arm == "view"
    fut = submit(bus, "switch_arm")
    run_ticks(loop, cell, 1)
    assert fut.result(0).ok and loop.active_arm == "grip"  # wraps back
    fut = submit(bus, "switch_arm_prev")
    run_ticks(loop, cell, 1)
    assert fut.result(0).ok and loop.active_arm == "view"  # (i - 1) mod n
    assert bus.snapshot.get()[0].active_arm == "view"


def test_session_without_grip_starts_on_first_arm(tmp_path):
    _, _, loop = _loop_over(tmp_path, ["view", "aux"])
    assert loop.active_arm == "view"


def test_health_line_once_per_period_with_window_deltas(fake_loop, caplog):
    """2026-09-07 (04-runtime §14 "Logging"): one INFO `loop:` line per
    control.health_log_every_s, carrying window deltas (ticks, overruns, IK
    slips) plus the gate verdict and the command-vs-measured lag; nothing at all
    with the period at 0."""
    import logging

    cell, bus, loop = fake_loop
    loop.cfg = loop.cfg.model_copy(update={"health_log_every_s": 0.5})
    caplog.set_level(logging.INFO, logger="apollo_mavis_v2_runtime.control.loop")
    t = run_ticks(loop, cell, 49)  # first tick at 0.01 s arms the period: due at 0.51 s
    assert not [r for r in caplog.records if r.getMessage().startswith("loop:")]
    t = run_ticks(loop, cell, 2, t)  # 0.51 s
    lines = [r for r in caplog.records if r.getMessage().startswith("loop:")]
    assert len(lines) == 1 and lines[0].levelno == logging.INFO
    msg = lines[0].getMessage()
    assert msg.startswith("loop: 50 ticks/0.5s (100 Hz") and "+0 overruns" in msg
    assert "active=arm0" in msg and "src=teleop" in msg and "held=[]" in msg
    assert "gate=ok" in msg and "ik_slips=+0" in msg and "ik_diverged=+0" in msg
    assert "cmd-meas={arm0:" in msg and "arm1:" in msg
    assert "servo=" not in msg and "tracker=" not in msg  # sim, no tracker: omitted
    # Steady state: exactly one more line per period, counters as window deltas.
    loop.ik_slips += 3
    t = run_ticks(loop, cell, 50, t)  # 1.01 s
    lines = [r for r in caplog.records if r.getMessage().startswith("loop:")]
    assert len(lines) == 2 and "ik_slips=+3" in lines[1].getMessage()
    t = run_ticks(loop, cell, 50, t)  # 1.51 s: the delta resets
    lines = [r for r in caplog.records if r.getMessage().startswith("loop:")]
    assert len(lines) == 3 and "ik_slips=+0" in lines[2].getMessage()
    # Period 0 disables it.
    caplog.clear()
    loop.cfg = loop.cfg.model_copy(update={"health_log_every_s": 0.0})
    run_ticks(loop, cell, 200, t + 0.5)
    assert not [r for r in caplog.records if r.getMessage().startswith("loop:")]


def test_watchdog_latch_and_clear_are_logged_once_per_edge(fake_loop, caplog):
    import logging

    cell, bus, loop = fake_loop
    caplog.set_level(logging.INFO, logger="apollo_mavis_v2_runtime.control.loop")
    t = run_ticks(loop, cell, 2)
    bus.held_keys.put(HeldState(held=frozenset({"KeyW"}), seq=1, rx_mono=t))
    loop.supervisor.watchdog.on_keys(HeldState(frozenset({"KeyW"}), 1, t))
    t = run_ticks(loop, cell, 2, t)
    # Silence from the browser: the WS watchdog latches after stale_s (0.2 s).
    t = run_ticks(loop, cell, 40, t)
    latched = [r for r in caplog.records if "watchdog LATCHED" in r.getMessage()]
    assert len(latched) == 1 and latched[0].levelno == logging.WARNING
    assert "KeyW" in latched[0].getMessage()
    t = run_ticks(loop, cell, 40, t)  # stays latched: no repeat
    assert len([r for r in caplog.records if "watchdog LATCHED" in r.getMessage()]) == 1
    # Every key released -> cleared, once.
    bus.held_keys.put(HeldState(held=frozenset(), seq=2, rx_mono=t))
    loop.supervisor.watchdog.on_keys(HeldState(frozenset(), 2, t))
    run_ticks(loop, cell, 5, t)
    cleared = [r for r in caplog.records if "watchdog cleared" in r.getMessage()]
    assert len(cleared) == 1


def test_a_latched_ws_deadman_drops_the_jog_instead_of_wedging_the_arm(fake_loop):
    """Regression, 2026-09-07 first live hardware session (11-safety §10.1).

    The Joint panel's jog is scaled by the WS input watchdog. When the browser's
    KeysMsg heartbeat stopped (it used to run only while keyboard capture was
    armed, so clicking the panel itself killed it), the deadman latched and
    ``_jog_step`` returned None BEFORE ``JogState.step`` could ever retire the
    target: the arm stayed in ``JOINT_JOG`` for good, ignoring the panel AND —
    since jog outranks teleop in ``_resolve_arms`` — the tracker. The latch edge
    now drops pending targets, so the arm falls back to teleop and no stale
    destination resumes when the browser returns.
    """
    cell, bus, loop = fake_loop
    t = run_ticks(loop, cell, 2)
    bus.held_keys.put(HeldState(held=frozenset({"KeyW"}), seq=1, rx_mono=t))
    loop.supervisor.watchdog.on_keys(HeldState(frozenset({"KeyW"}), 1, t))
    fut = submit(bus, "joint_target", arm_id="arm0", positions=[1.0] * 7 + [0.0], mode="jog")
    t = run_ticks(loop, cell, 2, t)
    assert fut.result(0).ok
    assert loop.jog.active("arm0")
    assert loop._last_cmd["arm0"][0] > 0.0  # walking toward the destination

    # Browser goes silent: 0.2 s of timeout at full scale, then the 0.1 s ramp,
    # then AWAIT_EMPTY.
    t = run_ticks(loop, cell, 40, t)
    assert loop.supervisor.watchdog.tripped
    assert not loop.jog.active("arm0")  # target dropped, arm not wedged
    assert loop._arm_source["arm0"] is not CommandSource.JOINT_JOG  # fell back to teleop
    q_latched = loop._last_cmd["arm0"].copy()
    assert q_latched[0] < 1.0  # stopped short of the destination

    # The browser comes back with everything released: no stale jog resumes.
    bus.held_keys.put(HeldState(held=frozenset(), seq=2, rx_mono=t))
    loop.supervisor.watchdog.on_keys(HeldState(frozenset(), 2, t))
    run_ticks(loop, cell, 5, t)  # < timeout_s, or it simply latches again
    assert not loop.supervisor.watchdog.tripped
    assert np.allclose(loop._last_cmd["arm0"], q_latched)


# -- a plan goal the command path cannot reach exactly (2026-09-09) -------------------
def _railed_loop(tmp_path, rail0: float, gate=None, hold_s: float = 3.0):
    """ControlLoop over a railed ``arm0`` PARKED at ``rail0`` (+ a rail-less ``arm1``),
    NullGate unless ``gate`` is given - the ``fake_loop`` wiring with a chosen start."""
    from apollo_mavis_v2_core import ProfileStore
    from apollo_mavis_v2_core.testing import FakeArm, FakeWorkcell

    from apollo_mavis_v2_runtime.bus import RuntimeBus
    from apollo_mavis_v2_runtime.config import ControlConfig
    from apollo_mavis_v2_runtime.control.loop import ControlLoop
    from apollo_mavis_v2_runtime.safety.gate import NullGate
    from apollo_mavis_v2_runtime.safety.supervisor import SafetySupervisor
    from apollo_mavis_v2_runtime.safety.watchdog import InputWatchdog

    q0 = np.array([0.0] * 7 + [rail0])
    cell = FakeWorkcell({"arm0": FakeArm("arm0", has_rail=True, q0=q0), "arm1": FakeArm("arm1")})
    cell.start()
    bus = RuntimeBus()
    loop = ControlLoop(
        cell,
        ControlConfig(),
        bus,
        SafetySupervisor(gate or NullGate(), InputWatchdog()),
        ["arm0", "arm1"],
        profile_store=ProfileStore(tmp_path / "profiles"),
        workcell_kind="sim",
        plan_gate_hold_s=hold_s,
    )
    return cell, bus, loop


class _UnrelatedPairGate:
    """Scripted gate: BLOCKED on a pair the planned arm's rail cannot change, holding
    every PLANNER command at the last safe posture; everything else passes (duck-types
    ``SafetyGate.filter`` for the supervisor)."""

    PAIR = ("arm0_link4", "arm1_link5")

    def __init__(self) -> None:
        self._last_safe: dict[str, np.ndarray] = {}
        self._block_pairs: set = set()

    def filter(self, q_cmd, q_meas, source=CommandSource.TELEOP, ts=None, stale=False):
        from apollo_mavis_v2_core import CollisionReport

        from apollo_mavis_v2_runtime.safety.gate import GateDecision

        if not self._last_safe:
            self._last_safe = {a: np.array(q) for a, q in q_meas.items()}
        if source is CommandSource.PLANNER:
            self._block_pairs = {self.PAIR}
            report = CollisionReport(
                blocked=True, severity="blocked", pairs=[self.PAIR], min_clearance_m=0.0061
            )
            return GateDecision({a: self._last_safe[a] for a in q_cmd}, True, report, [])
        self._block_pairs = set()
        self._last_safe = {a: np.array(q) for a, q in q_cmd.items()}
        return GateDecision(dict(q_cmd), False, CollisionReport.ok(), [])

    def reseed(self, arm_id, q_meas) -> None:
        self._last_safe[arm_id] = np.array(q_meas)


def _rail_limit_cases():
    from apollo_mavis_v2_runtime.control.loop import RAIL_TRAVEL_M

    return [(0.0, -0.0004, 0.0), (RAIL_TRAVEL_M, RAIL_TRAVEL_M + 0.0004, RAIL_TRAVEL_M)]


@pytest.mark.parametrize(("rail0", "goal_rail", "limit"), _rail_limit_cases())
def test_a_rail_goal_a_hair_outside_the_travel_finishes_at_the_reachable_limit(
    tmp_path, rail0, goal_rail, limit
):
    """``_confirm_plan_arrivals`` (2026-09-09): a 1-waypoint plan whose rail goal lies a
    few tenths of a mm past an end stop (a goal copied verbatim from a carriage that
    settled at -0.4 mm / 650.4 mm) is clamped to the travel by ``_cap_joint_step`` every
    tick, so ``q_out`` never equals the goal. With the gate clear and no progress left
    the plan finishes ``done`` within a few ticks - ``plan_cancel_reason None``, the
    command at the travel end, the gate-hold clock untouched - and the manager's arrival
    rule (2 mm on the carriage) accepts the measured posture. Before the rule the plan
    hung ``executing`` for good and blocked every later reset / Go-to-profile / R."""
    from conftest import q_of

    from apollo_mavis_v2_runtime.session.manager import SessionManager

    cell, bus, loop = _railed_loop(tmp_path, rail0)
    goal = [0.0] * 7 + [goal_rail]
    fut = submit(bus, "execute_plan", waypoints={"arm0": [goal]}, gripper={}, interruptible=True)
    t = run_ticks(loop, cell, 1)
    res = fut.result(0)
    assert res.ok and res.detail == "executing", res
    ticks = 1
    while loop.plans.active_arms and ticks < 5:
        t = run_ticks(loop, cell, 1, t)
        ticks += 1
    assert not loop.plans.active_arms, f"still executing after {ticks} ticks"
    assert ticks <= 3, ticks  # "within a few ticks": the goal step + the clamp verdict
    assert loop._plan_status == "done" and loop.plan_cancel_reason is None
    assert loop._plan_state.get("arm0") is None
    assert loop._plan_gate_hold_since is None
    assert np.allclose(loop._last_cmd["arm0"][:7], 0.0)
    assert loop._last_cmd["arm0"][7] == pytest.approx(limit, abs=1e-12)
    t = run_ticks(loop, cell, 5, t)
    assert SessionManager._arrived(q_of(cell, "arm0"), goal)


def test_a_clamped_rail_goal_the_gate_holds_stays_executing_until_the_watch_cancels(tmp_path):
    """Control case: the same clamped rail goal while the gate is BLOCKED on a pair the
    rail cannot change. The reachable-limit rule must NOT finish it (``dec.blocked``):
    the plan stays ``executing`` and ``_plan_gate_watch`` cancels it after
    ``plan_gate_hold_s`` naming the blocking pair (the current, documented behaviour of
    this corner - the reason is the gate's pair, not the clamp; 2026-09-09 review)."""
    from apollo_mavis_v2_runtime.control.loop import GATE_HOLD_PREFIX

    cell, bus, loop = _railed_loop(tmp_path, 0.0, gate=_UnrelatedPairGate(), hold_s=0.1)
    goal = [0.0] * 7 + [-0.0004]
    fut = submit(bus, "execute_plan", waypoints={"arm0": [goal]}, gripper={}, interruptible=True)
    t = run_ticks(loop, cell, 1)
    assert fut.result(0).ok
    assert loop.plans.active_arms == ["arm0"]  # the goal went back into the executor
    assert loop._plan_status == "executing" and loop._plan_gate_hold_since is not None
    t = run_ticks(loop, cell, 5, t)
    assert loop.plans.active_arms == ["arm0"] and loop._plan_status == "executing"
    t = run_ticks(loop, cell, 10, t)  # past the 0.1 s hold limit
    assert not loop.plans.active_arms
    assert loop._plan_status == "cancelled"
    assert loop.plan_cancel_reason is not None
    assert loop.plan_cancel_reason.startswith(GATE_HOLD_PREFIX)
    assert "arm0_link4 / arm1_link5" in loop.plan_cancel_reason
    assert loop._last_cmd["arm0"][7] == 0.0  # never moved


def test_a_joint_goal_the_cap_holds_is_not_reported_as_an_arrival(tmp_path):
    """The reachable-limit rule is confined to the rail slot at a travel end: a
    non-positive ``dq_max`` (the fail-safe hold of ``_cap_joint_step``) leaves the
    joints at ``q_last`` with the gate clear and no progress - the plan stays
    ``executing`` (the manager's budget reports it) instead of finishing ``done`` on
    an arm that never moved (2026-09-09 review)."""
    cell, bus, loop = _railed_loop(tmp_path, 0.0)
    loop.cfg = loop.cfg.model_copy(update={"dq_max_rad": 0.0})  # bypasses the gt=0 validator
    goal = [0.01] * 7 + [0.0]
    fut = submit(bus, "execute_plan", waypoints={"arm0": [goal]}, gripper={}, interruptible=True)
    t = run_ticks(loop, cell, 1)
    assert fut.result(0).ok
    run_ticks(loop, cell, 20, t)
    assert loop.plans.active_arms == ["arm0"]
    assert loop._plan_status == "executing" and loop.plan_cancel_reason is None
    assert np.allclose(loop._last_cmd["arm0"], 0.0)  # held, not arrived
