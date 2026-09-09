"""Sequential execution of twin-planned multi-arm motions + the gate-held abort
(2026-09-08 evening; 04-runtime §10.5 / §5.2, 11-safety §9).

Measured at 23:16:52 that day in ``var/logs/runtime.log``: a ``reset_to_initial`` on the
real cell planned both arms SEQUENTIALLY (``ResetPlanner``: arm k with the arms before it
frozen at their goals, the arms after it at their starts) but the manager submitted ONE
``execute_plan`` carrying both arms' waypoints, so the executor moved them simultaneously
through combinations the planner never validated; the twin gate blocked at 5.2 mm
(``grip_right_finger`` / ``view_link3``) and the return sat gate-blocked for the whole
30 s budget. Operator decisions: execute one arm at a time in ``PlanResult.arm_order``,
and tell the operator WHICH pair holds a plan instead of waiting for the budget
(``hardware_session.plan_gate_hold_s``). Default speed 100 %.

Everything here runs on the hand-built two-arm fake session of
``test_return_manager_units`` (``arm0`` railed, ``arm1`` not; NullGate) plus a scripted
holding gate for the abort."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from apollo_mavis_v2_core import (
    ArmPosture,
    CollisionReport,
    Command,
    CommandSource,
    HeldState,
    PlanResult,
    ProfileStore,
    StateProfile,
)
from apollo_mavis_v2_core.testing import FakeArm, FakeWorkcell
from conftest import run_ticks
from pydantic import ValidationError
from test_return_manager_units import (
    FakeRT,
    FakeTwin,
    _spy_execute_plan,
    _start_from_session,
    make,
    tick_until,
)

import apollo_mavis_v2_runtime.session.manager as manager_mod
from apollo_mavis_v2_runtime.bus import RuntimeBus
from apollo_mavis_v2_runtime.config import ControlConfig, HardwareSessionConfig
from apollo_mavis_v2_runtime.control.loop import GATE_HOLD_PREFIX, ControlLoop
from apollo_mavis_v2_runtime.safety.gate import GateDecision, NullGate
from apollo_mavis_v2_runtime.safety.supervisor import SafetySupervisor
from apollo_mavis_v2_runtime.safety.watchdog import InputWatchdog
from apollo_mavis_v2_runtime.session.manager import SessionManager
from apollo_mavis_v2_runtime.session.types import SessionState
from apollo_mavis_v2_runtime.streams.hub import VideoHub

ARMS = ("arm0", "arm1")


class OrderedTwin(FakeTwin):
    """A planner that moves every joint by ``step`` rad (the rail slot kept) and reports
    ``arm_order`` - inserting the waypoints in REQUEST order so a consumer that honours
    the dict order instead of ``arm_order`` is caught."""

    def __init__(self, order: list[str], step: float = 0.5) -> None:
        super().__init__()
        self.order, self.step = list(order), step

    def plan(self, req):
        self.plans.append(req)
        wps = {}
        for arm, goal in req.q_goal.items():  # request order, NOT arm_order
            start = list(req.q_start[arm])
            wps[arm] = [start, [g + self.step for g in goal[:7]] + list(goal[7:])]
        return PlanResult(ok=True, waypoints=wps, arm_order=list(self.order))


class Recorder:
    """Per-tick commanded q of both arms + the executor's active set."""

    def __init__(self, loop, cell) -> None:
        self.loop, self.cell = loop, cell
        self.rows: list[tuple[dict[str, np.ndarray], list[str], float | None]] = []
        self.t = 0.0

    def tick(self, session=None) -> None:
        self.t = run_ticks(self.loop, self.cell, 1, self.t)
        self.rows.append((
            {a: self.loop._last_cmd[a].copy() for a in ARMS},
            list(self.loop.plans.active_arms),
            getattr(session, "start_from_progress", None) if session is not None else None,
        ))

    def first_move_tick(self, arm: str) -> int | None:
        q0 = self.rows[0][0][arm]
        for i, (q, _, _) in enumerate(self.rows):
            if not np.allclose(q[arm], q0):
                return i
        return None

    def last_active_tick(self, arm: str) -> int:
        return max(i for i, (_, active, _) in enumerate(self.rows) if arm in active)


def _drive(rec: Recorder, done: threading.Event, seconds: float = 20.0, session=None) -> None:
    t0 = time.monotonic()
    while not done.is_set():
        assert time.monotonic() - t0 < seconds, "worker never finished"
        rec.tick(session)
        time.sleep(0.001)
    rec.tick(session)  # one more: the executor's last tick is visible


def _worker(fn) -> threading.Event:
    done = threading.Event()

    def run():
        try:
            fn()
        finally:
            done.set()

    threading.Thread(target=run, daemon=True).start()
    return done


def _assert_sequential(rec: Recorder, first: str, second: str) -> None:
    """``first`` arrives before ``second`` moves at all; never two arms in the executor."""
    assert all(len(active) <= 1 for _, active, _ in rec.rows)
    arrived = rec.last_active_tick(first)
    started = rec.first_move_tick(second)
    assert started is not None and started > arrived, (started, arrived)
    q2_0 = rec.rows[0][0][second]
    assert all(np.allclose(rec.rows[i][0][second], q2_0) for i in range(arrived + 1))
    assert rec.first_move_tick(first) is not None and rec.first_move_tick(first) < started


# -- the per-episode return, two arms -------------------------------------------------------
@pytest.mark.parametrize("order", [["arm0", "arm1"], ["arm1", "arm0"]])
def test_two_arm_return_executes_one_arm_at_a_time_in_arm_order(
    tmp_path, fake_loop, monkeypatch, order
):
    """``arm_order[0]`` reaches its last waypoint BEFORE ``arm_order[1]`` moves at all -
    whichever order the planner reports (the waypoints dict is in request order, so the
    reversed case proves ``arm_order`` wins over insertion order)."""
    cell, bus, loop = fake_loop
    manager, session, rt, prof = make(tmp_path, fake_loop, twin=OrderedTwin(order))
    calls = _spy_execute_plan(monkeypatch)
    rec = Recorder(loop, cell)
    rec.tick()
    done = _worker(lambda: manager._return_home_worker(session, prof, "saved"))
    _drive(rec, done, session=session)
    assert rt.calls[-1] == (False, "")
    assert [list(c["waypoints"]) for c in calls] == [[order[0]], [order[1]]]
    _assert_sequential(rec, order[0], order[1])
    for a in ARMS:  # both arrived at the planner's goals
        assert loop._last_cmd[a][:7] == pytest.approx(np.asarray(prof.arms[a].q) + 0.5, abs=1e-6)


def test_gripper_targets_ride_the_last_arm_and_apply_on_its_arrival(
    tmp_path, fake_loop, monkeypatch
):
    """The profile closes arm0's gripper to 0.3; arm0 moves FIRST. The target is not sent
    with arm0's plan (it would apply when arm0 arrives, halfway through the motion) but
    with the LAST arm's, and the loop applies it when THAT plan arrives - the
    interruptible on-arrival semantics kept across the sequence."""
    from test_return_plan_units import GripSpy

    cell, bus, loop = fake_loop
    manager, session, rt, _ = make(tmp_path, fake_loop, twin=OrderedTwin(["arm0", "arm1"]))
    spies = {a: GripSpy() for a in ARMS}
    loop._senders.update(spies)
    q = {a: list(cell.arms[a].get_state().q[:7]) for a in ARMS}
    prof = manager.profile_store.save(StateProfile(
        name="grip", workcell_kind="sim",
        arms={a: ArmPosture(q=[x + 0.05 for x in q[a]], gripper_open_frac=0.3) for a in ARMS},
    ))
    calls = _spy_execute_plan(monkeypatch)
    rec = Recorder(loop, cell)
    rec.tick()
    grip_before = {a: loop._grip_frac[a] for a in ARMS}
    assert grip_before["arm0"] != 0.3
    fracs: list[float] = []

    def tick_and_watch():
        rec.tick()
        fracs.append(loop._grip_frac["arm0"])

    done = _worker(lambda: manager._return_home_worker(session, prof, "saved"))
    t0 = time.monotonic()
    while not done.is_set():
        assert time.monotonic() - t0 < 20.0
        tick_and_watch()
        time.sleep(0.001)
    tick_and_watch()
    assert rt.calls[-1] == (False, "")
    assert [c["gripper"] for c in calls] == [{}, {"arm0": 0.3, "arm1": 0.3}]  # last arm only
    # ``rec.rows[0]`` predates the worker, so ``fracs[k]`` is the frac after row k + 1; the
    # tick AFTER the last row with arm1 active is its arrival tick (the executor pops the
    # arm and ``_finish_plan`` applies the deferred targets within that tick)
    arrival = rec.last_active_tick("arm1")  # fracs index of the arrival tick
    assert arrival > 3 and all(f == grip_before["arm0"] for f in fracs[:arrival])  # untouched
    assert fracs[arrival] == 0.3  # ... applied on the LAST arm's arrival, not arm0's
    assert loop._grip_frac["arm0"] == 0.3 and spies["arm0"].puts == [0.3]
    assert loop._grip_frac["arm1"] == 0.3 and spies["arm1"].puts == [0.3]


def test_cancel_during_the_first_arm_leaves_the_second_unmoved_and_names_the_arm(
    tmp_path, fake_loop, monkeypatch
):
    cell, bus, loop = fake_loop
    manager, session, rt, prof = make(tmp_path, fake_loop, twin=OrderedTwin(["arm0", "arm1"]))
    calls = _spy_execute_plan(monkeypatch)
    q1_0 = loop._last_cmd["arm1"].copy()
    done = _worker(lambda: manager._return_home_worker(session, prof, "saved"))
    tick_until(loop, cell, lambda: loop.plans.active_arms == ["arm0"], 10.0)
    t = run_ticks(loop, cell, 5)
    held = HeldState(held=frozenset({"KeyW"}), seq=1, rx_mono=t)
    loop.supervisor.watchdog.on_keys(held)
    bus.held_keys.put(held)
    t = run_ticks(loop, cell, 1, t)
    assert not loop.plans.active_arms and loop.plan_cancel_reason == "movement key"
    empty = HeldState(held=frozenset(), seq=2, rx_mono=t)
    loop.supervisor.watchdog.on_keys(empty)
    bus.held_keys.put(empty)
    tick_until(loop, cell, done.is_set, 10.0)
    run_ticks(loop, cell, 10, t)
    assert rt.calls[-1] == (False, "return cancelled: movement key - arm0; arm1 not moved")
    assert len(calls) == 1 and list(calls[0]["waypoints"]) == ["arm0"]  # arm1 never submitted
    assert np.allclose(loop._last_cmd["arm1"], q1_0)
    assert np.allclose(cell.arms["arm1"].get_state().q, q1_0)
    assert not np.allclose(loop._last_cmd["arm0"][:7], np.asarray(prof.arms["arm0"].q) + 0.5)


def test_a_single_moving_arm_behaves_exactly_as_before(tmp_path, fake_loop, monkeypatch):
    """A profile that leaves arm1 where it is: one ``execute_plan`` (arm0 only - an arm the
    planner did not move is not submitted at all), the cancel text without any arm suffix."""
    cell, bus, loop = fake_loop
    manager, session, rt, _ = make(tmp_path, fake_loop, twin=OrderedTwin(["arm0", "arm1"]))
    q = {a: list(cell.arms[a].get_state().q[:7]) for a in ARMS}

    class Arm0OnlyTwin(OrderedTwin):
        def plan(self, req):
            res = super().plan(req)
            wps = dict(res.waypoints)
            wps["arm1"] = [list(req.q_start["arm1"])] * 2  # planner: arm1 stays put
            return PlanResult(ok=True, waypoints=wps, arm_order=res.arm_order)

    session.twin = Arm0OnlyTwin(["arm1", "arm0"])
    prof = manager.profile_store.save(StateProfile(
        name="arm0-only", workcell_kind="sim",
        arms={"arm0": ArmPosture(q=[x + 0.05 for x in q["arm0"]]), "arm1": ArmPosture(q=q["arm1"])},
    ))
    calls = _spy_execute_plan(monkeypatch)
    done = _worker(lambda: manager._return_home_worker(session, prof, "saved"))
    tick_until(loop, cell, lambda: loop.plans.active_arms == ["arm0"], 10.0)
    t = run_ticks(loop, cell, 3)
    held = HeldState(held=frozenset({"KeyW"}), seq=1, rx_mono=t)
    loop.supervisor.watchdog.on_keys(held)
    bus.held_keys.put(held)
    run_ticks(loop, cell, 1, t)
    tick_until(loop, cell, done.is_set, 10.0)
    assert rt.calls[-1] == (False, "return cancelled: movement key")  # no suffix
    assert len(calls) == 1 and calls[0]["gripper"] == {"arm0": 1.0, "arm1": 1.0}


def test_ordered_waypoints_and_moving_arms_helpers():
    res = PlanResult(ok=True, waypoints={"a": [[0.0], [1.0]], "b": [[0.0], [0.0]], "c": [[1.0]]},
                     arm_order=["b", "a"])
    ordered = SessionManager._ordered_waypoints(res)
    assert list(ordered) == ["b", "a", "c"]  # arm_order first, unnamed arms last
    assert SessionManager._moving_arms(ordered) == ["a"]  # b and c go nowhere
    # the parked threshold is the arrival tolerance (1e-3), not 1e-6: a hardware start that
    # differs from the goal by the SDK's ~1e-4 rad read-back noise is parked, not a micro-plan
    noise = {"n": [[0.0] * 8, [5e-4] * 8], "m": [[0.0] * 8, [2e-3] + [0.0] * 7]}
    assert SessionManager._moving_arms(noise) == ["m"]
    assert manager_mod.PLAN_ARRIVAL_TOL_RAD == 1e-3 and manager_mod.PLAN_ARRIVAL_TOL_RAIL_M == 2e-3
    assert SessionManager._arrived([0.0] * 7 + [0.1], [5e-4] * 7 + [0.1015])
    assert not SessionManager._arrived([0.0] * 7 + [0.1], [0.0] * 7 + [0.103])  # rail 3 mm off
    assert not SessionManager._arrived([0.0] * 7, [0.0] * 6 + [2e-3])
    assert SessionManager._arrival_error([0.0] * 7, [0.0] * 6 + [2e-3]) == (2e-3, None)
    legacy = PlanResult(ok=True, waypoints={"y": [[0.0], [1.0]], "x": [[0.0], [1.0]]})
    assert list(SessionManager._ordered_waypoints(legacy)) == ["y", "x"]  # dict order fallback
    assert SessionManager._sequence_detail("movement key", ["grip"], 0) == "movement key"
    assert SessionManager._sequence_detail("movement key", ["view", "grip"], 0) == (
        "movement key - Perception Arm; Manipulation Arm not moved"
    )
    assert SessionManager._sequence_detail("budget 30.0 s", ["view", "grip"], 1) == (
        "budget 30.0 s - Manipulation Arm"
    )


# -- start_from, two arms -------------------------------------------------------------------
def test_start_from_two_arms_is_sequential_and_progress_counts_every_waypoint(
    tmp_path, fake_loop, monkeypatch
):
    cell, bus, loop = fake_loop
    manager, session, prof = _start_from_session(
        tmp_path, fake_loop, 0.0, twin=OrderedTwin(["arm1", "arm0"])
    )
    calls = _spy_execute_plan(monkeypatch)
    rec = Recorder(loop, cell)
    rec.tick(session)
    done = _worker(lambda: manager._start_from_worker(session))
    _drive(rec, done, session=session)
    assert [list(c["waypoints"]) for c in calls] == [["arm1"], ["arm0"]]
    assert [c["gripper"] for c in calls] == [{}, {"arm0": 1.0, "arm1": 1.0}]
    _assert_sequential(rec, "arm1", "arm0")
    progress = [p for _, _, p in rec.rows if p is not None]
    assert progress and all(b >= a for a, b in zip(progress, progress[1:], strict=False))
    # counted over ALL arms' waypoints: partial values while arm1 runs AND while arm0 runs
    arm1_arrived = rec.last_active_tick("arm1")
    during_1 = [p for i, (_, _, p) in enumerate(rec.rows) if p is not None and i <= arm1_arrived]
    during_0 = [p for i, (_, _, p) in enumerate(rec.rows) if p is not None and i > arm1_arrived]
    assert during_1 and max(during_1) < 1.0
    assert during_0 and 0.0 < min(during_0) and max(during_0) <= 1.0
    assert session.start_from_progress is None and session.state is SessionState.RUNNING
    assert session.motion_detail == "" and manager._bringup is None


def test_start_from_cancelled_on_the_first_arm_reports_it_and_moves_nothing_else(
    tmp_path, fake_loop, monkeypatch
):
    cell, bus, loop = fake_loop
    manager, session, prof = _start_from_session(
        tmp_path, fake_loop, 0.0, twin=OrderedTwin(["arm0", "arm1"])
    )
    calls = _spy_execute_plan(monkeypatch)
    q1_0 = loop._last_cmd["arm1"].copy()
    done = _worker(lambda: manager._start_from_worker(session))
    tick_until(loop, cell, lambda: loop.plans.active_arms == ["arm0"], 10.0)
    t = run_ticks(loop, cell, 3)
    held = HeldState(held=frozenset({"KeyD"}), seq=1, rx_mono=t)
    loop.supervisor.watchdog.on_keys(held)
    bus.held_keys.put(held)
    run_ticks(loop, cell, 1, t)
    tick_until(loop, cell, done.is_set, 10.0)
    assert len(calls) == 1
    assert session.motion_detail == (
        "start_from cancelled: movement key - arm0; arm1 not moved (Go to profile retries it)"
    )
    assert np.allclose(loop._last_cmd["arm1"], q1_0)
    assert session.state is SessionState.RUNNING and session.start_from_progress is None


# -- gate-held abort -------------------------------------------------------------------------
PAIR = ("arm0_link6", "arm1_link3")


class HoldingGate:
    """Scripted gate: holds every PLANNER-sourced command (hold-last-safe, blocked report
    naming ``PAIR`` at 5.2 mm) for the first ``hold_ticks`` planner ticks, passes
    everything else. Duck-types ``SafetyGate.filter`` for the supervisor."""

    def __init__(self, hold_ticks: float = float("inf")) -> None:
        self.hold_ticks = hold_ticks
        self.planner_ticks = 0
        self._last_safe: dict[str, np.ndarray] = {}
        self._block_pairs: set = set()

    def filter(self, q_cmd, q_meas, source=CommandSource.TELEOP, ts=None, stale=False):
        if not self._last_safe:
            self._last_safe = {a: np.array(q) for a, q in q_meas.items()}
        if source is CommandSource.PLANNER and self.planner_ticks < self.hold_ticks:
            self.planner_ticks += 1
            self._block_pairs = {PAIR}
            report = CollisionReport(blocked=True, severity="blocked", pairs=[PAIR],
                                     min_clearance_m=0.0052)
            return GateDecision({a: self._last_safe[a] for a in q_cmd}, True, report, [])
        self._block_pairs = set()
        self._last_safe = {a: np.array(q) for a, q in q_cmd.items()}
        return GateDecision(dict(q_cmd), False, CollisionReport.ok(), [])

    def reseed(self, arm_id, q_meas) -> None:
        self._last_safe[arm_id] = np.array(q_meas)


def _gated_loop(tmp_path, gate, hold_s: float):
    cell = FakeWorkcell({"arm0": FakeArm("arm0", has_rail=True), "arm1": FakeArm("arm1")})
    cell.start()
    bus = RuntimeBus()
    loop = ControlLoop(
        cell, ControlConfig(), bus, SafetySupervisor(gate, InputWatchdog()), list(ARMS),
        profile_store=ProfileStore(tmp_path / "profiles"), workcell_kind="sim",
        plan_gate_hold_s=hold_s,
    )
    return cell, bus, loop


def _execute(bus, loop, cell, goal, *, interruptible=True):
    fut = bus.commands.submit(Command(
        op="execute_plan",
        args={"waypoints": {"arm0": [goal]}, "gripper": {}, "interruptible": interruptible},
        source="internal",
    ))
    t = run_ticks(loop, cell, 1)
    assert fut.result(0).ok and loop.plans.active_arms == ["arm0"]
    return t


def test_gate_held_plan_is_cancelled_after_plan_gate_hold_s_with_the_pair(tmp_path):
    """The loop cancels a plan the gate has held for ``plan_gate_hold_s`` (0.2 s = 20 ticks
    of the 100 Hz fake clock) with the blocking pair and distance in the reason; telemetry
    ``plan_status`` reads ``cancelled``; the arm holds where it was."""
    cell, bus, loop = _gated_loop(tmp_path, HoldingGate(), 0.2)
    run_ticks(loop, cell, 1)
    goal = [float(x) for x in loop._last_cmd["arm0"]]
    goal[0] += 1.0
    held_at = loop._last_cmd["arm0"].copy()
    t = _execute(bus, loop, cell, goal)
    ticks = 0
    while loop.plans.active_arms:
        t += loop.dt
        snap = loop.run_tick(t)
        cell.step(loop.dt)
        ticks += 1
        assert ticks < 40, "the gate-held abort never fired"
    assert 19 <= ticks <= 22, ticks  # ~plan_gate_hold_s, not the 30 s budget
    assert loop.plan_cancel_reason == f"{GATE_HOLD_PREFIX}: arm0_link6 / arm1_link3 at 5.2 mm"
    assert loop._plan_status == "cancelled" and snap.session_extra["plan_status"] == "cancelled"
    assert np.allclose(loop._last_cmd["arm0"], held_at)  # never left the last safe posture
    run_ticks(loop, cell, 5, t)
    assert np.allclose(loop._last_cmd["arm0"], held_at) and not loop.plans.active_arms


def test_a_transient_gate_hold_shorter_than_the_limit_does_not_cancel(tmp_path):
    cell, bus, loop = _gated_loop(tmp_path, HoldingGate(hold_ticks=10), 0.2)
    run_ticks(loop, cell, 1)
    goal = [float(x) for x in loop._last_cmd["arm0"]]
    goal[0] += 0.1  # 5 slew ticks once released
    t = _execute(bus, loop, cell, goal)
    run_ticks(loop, cell, 30, t)
    assert not loop.plans.active_arms and loop.plan_cancel_reason is None
    assert loop._plan_status in ("done", None)
    assert loop._last_cmd["arm0"] == pytest.approx(goal)
    assert loop._plan_gate_hold_since is None


def test_hold_limit_zero_cancels_on_the_first_held_tick_and_the_default_is_three_seconds(
    tmp_path,
):
    cell, bus, loop = _gated_loop(tmp_path, HoldingGate(), 0.0)
    run_ticks(loop, cell, 1)
    goal = [float(x) for x in loop._last_cmd["arm0"]]
    goal[0] += 1.0
    fut = bus.commands.submit(Command(
        op="execute_plan",
        args={"waypoints": {"arm0": [goal]}, "gripper": {}, "interruptible": True},
        source="internal",
    ))
    run_ticks(loop, cell, 1)  # loaded, stepped once, held, cancelled - all in this tick
    assert fut.result(0).ok
    assert not loop.plans.active_arms and loop.plan_cancel_reason.startswith(GATE_HOLD_PREFIX)
    assert HardwareSessionConfig().plan_gate_hold_s == 3.0
    assert HardwareSessionConfig().default_speed_scale == 1.0  # operator decision 2026-09-08
    with pytest.raises(ValidationError):
        HardwareSessionConfig(plan_gate_hold_s=-0.1)
    _cell, _bus, default_loop = _gated_loop(tmp_path, HoldingGate(), 3.0)
    assert default_loop.plan_gate_hold_s == 3.0


def test_the_manager_reports_a_gate_held_return_immediately_with_the_pair(tmp_path):
    """Through the manager: the per-episode return says which pair held it (not "timed
    out" 30 s later), and the reset / exit motion's dialog text names the pair; the wire
    status is ``timeout`` (the ``ReturnHomeResult`` literal for "held by the gate, stopped
    where they are") - reported after ``plan_gate_hold_s``, not after the budget."""
    gated = _gated_loop(tmp_path, HoldingGate(), 0.2)
    cell, bus, loop = gated
    manager, session, rt, prof = make(tmp_path, gated, twin=OrderedTwin(["arm0", "arm1"]))
    t0 = time.monotonic()
    done = _worker(lambda: manager._return_home_worker(session, prof, "saved"))
    tick_until(loop, cell, done.is_set, 10.0)
    assert time.monotonic() - t0 < 5.0  # NOT the 30 s budget
    assert rt.calls[-1] == (
        False,
        f"return stopped - {GATE_HOLD_PREFIX}: arm0_link6 / arm1_link3 at 5.2 mm - arm0; "
        "arm1 not moved",
    )
    assert not loop.plans.active_arms
    # the reset-to-initial / exit path (two-phase machinery, one phase here: no rail stored)
    session.supervisor.gate.planner_ticks = 0
    done = threading.Event()
    out: dict = {}

    def run():
        out["res"] = manager._return_to_initial_motion(session, prof, label="return_home")
        done.set()

    threading.Thread(target=run, daemon=True).start()
    tick_until(loop, cell, done.is_set, 10.0)
    res = out["res"]
    assert (res.ok, res.status) == (False, "timeout")
    assert res.detail == (
        "the motion was held by the safety gate and stopped (arm0_link6 / arm1_link3 at "
        "5.2 mm - arm0; arm1 not moved). The arms hold where they are."
    )
    assert SessionManager._return_phase_text("joints", "held", "a / b at 5.2 mm", True) == (
        "the motion was held by the safety gate while moving the joints and stopped "
        "(a / b at 5.2 mm). The arms hold where they are."
    )
    # the real-budget timeout branch is untouched
    assert SessionManager._return_phase_text("joints", "timeout", "budget 30.0 s", False) == (
        "the motion was held by the safety gate and stopped part-way (budget 30.0 s)."
    )


def test_fake_session_helpers_are_the_ones_the_runtime_uses(tmp_path, fake_loop):
    """Guard for this file's plumbing: ``make`` builds the manager the runtime builds and
    the hand-built session carries every field ``_execute_arms`` reads."""
    cell, bus, loop = fake_loop
    manager, session, rt, _ = make(tmp_path, fake_loop)
    assert isinstance(manager, SessionManager) and isinstance(rt, FakeRT)
    assert isinstance(session, SimpleNamespace) and isinstance(manager.hub, VideoHub)
    assert session.loop is loop and session.state is SessionState.RUNNING


# -- the loop is the last line: one plan at a time, one arm per plan (2026-09-08 review) -----
class _StubPlanner:
    def __init__(self, result) -> None:
        self.result, self.requests = result, []

    def plan(self, req):
        self.requests.append(req)
        return self.result


def _long_plan(bus, loop, cell, arm="arm0", *, interruptible=False, gripper=None):
    run_ticks(loop, cell, 1)  # seed from measured
    goal = [float(x) for x in loop._last_cmd[arm]]
    goal[0] += 1.0  # 50 slew ticks
    fut = bus.commands.submit(Command(
        op="execute_plan",
        args={"waypoints": {arm: [goal]}, "gripper": gripper or {},
              "interruptible": interruptible},
        source="internal",
    ))
    t = run_ticks(loop, cell, 1)
    assert fut.result(0).ok and loop.plans.active_arms == [arm]
    return goal, t


def test_goto_for_another_arm_is_refused_while_a_plan_runs(fake_loop):
    """Reproduces the review's throwaway probe: a non-interruptible plan (start_from) walks
    arm0; the joint panel's ``goto`` for arm1 used to be ACCEPTED, planned with arm0 frozen
    at its mid-path posture, and loaded beside it - two arms in the executor at once
    (``active_arms == ['arm0', 'arm1']``). Now it is refused with ``plan executing``; a
    jog for the other arm is still allowed (not a plan; the gate checks it)."""
    cell, bus, loop = fake_loop
    _goal, t = _long_plan(bus, loop, cell)
    goal1 = [0.3] * 7
    loop.planner = _StubPlanner(PlanResult(ok=True, waypoints={"arm1": [goal1]}))
    fut = bus.commands.submit(Command(
        op="joint_target", args={"arm_id": "arm1", "positions": goal1, "mode": "goto"},
        source="ws",
    ))
    t = run_ticks(loop, cell, 1, t)
    ack = fut.result(0)
    assert (ack.ok, ack.detail) == (False, "plan executing")
    assert loop.plans.active_arms == ["arm0"] and loop.planner.requests == []
    assert loop._plan_state == {"arm0": "executing"}
    t = run_ticks(loop, cell, 5, t)
    assert loop.plans.active_arms == ["arm0"]  # the running plan is untouched
    # a plan RESULT that lands while a plan runs (a worker that started before it) is
    # dropped, never loaded on top; the running plan's status is untouched
    fut = bus.commands.submit(Command(
        op="_plan_ready",
        args={"arms": ["arm1"], "result": PlanResult(ok=True, waypoints={"arm1": [goal1]}),
              "detail": ""},
        source="internal",
    ))
    t = run_ticks(loop, cell, 1, t)
    ack = fut.result(0)
    assert (ack.ok, ack.detail) == (False, "plan executing")
    assert loop.plans.active_arms == ["arm0"] and loop._plan_status == "executing"
    assert loop._plan_state == {"arm0": "executing", "arm1": "failed"}
    # a jog of the other arm is not a plan and keeps working
    fut = bus.commands.submit(Command(
        op="joint_target", args={"arm_id": "arm1", "positions": goal1, "mode": "jog"},
        source="ws",
    ))
    run_ticks(loop, cell, 1, t)
    assert fut.result(0).ok and loop.jog.active("arm1")
    # once the plan is done the goto is accepted again
    t = run_ticks(loop, cell, 120, t)
    assert not loop.plans.active_arms
    fut = bus.commands.submit(Command(
        op="joint_target", args={"arm_id": "arm1", "positions": goal1, "mode": "goto"},
        source="ws",
    ))
    run_ticks(loop, cell, 1, t)
    assert fut.result(0).ok and fut.result(0).detail == "accepted"


# -- a plan is finished AFTER the gate (2026-09-08 review) -------------------------------------
class GoalHoldingGate(HoldingGate):
    """Holds a PLANNER command only when it IS the plan's goal (the final step), for
    ``hold_ticks`` such ticks; everything before passes."""

    def __init__(self, goal, hold_ticks: float = float("inf")) -> None:
        super().__init__(hold_ticks)
        self.goal = np.asarray(goal, dtype=np.float64)

    def filter(self, q_cmd, q_meas, source=CommandSource.TELEOP, ts=None, stale=False):
        at_goal = "arm0" in q_cmd and np.allclose(q_cmd["arm0"], self.goal, atol=1e-9)
        if not (source is CommandSource.PLANNER and at_goal):
            self._last_safe = {a: np.array(q) for a, q in q_cmd.items()}
            self._block_pairs = set()
            return GateDecision(dict(q_cmd), False, CollisionReport.ok(), [])
        return super().filter(q_cmd, q_meas, source, ts, stale)


def test_a_goal_step_the_gate_holds_keeps_the_plan_executing(tmp_path):
    """Before: ``_finish_plan`` ran inside ``_resolve_arms``, BEFORE the gate, so a final
    step the gate held was reported ``done`` with ``plan_cancel_reason None``, ``_last_cmd``
    one slew step (0.02 rad) short of the goal, the deferred gripper applied on an arm that
    had not arrived and the gate-held watch never running. Now the goal goes back into the
    executor until the gated output IS the goal - here the hold lasts, so the gate-held
    abort names the pair after ``plan_gate_hold_s`` and the gripper stays untouched."""
    from test_return_plan_units import GripSpy

    goal = [0.0] * 8
    goal[0] = 0.1  # 5 slew ticks
    cell, bus, loop = _gated_loop(tmp_path, GoalHoldingGate(goal), 0.2)
    spy = GripSpy()
    loop._senders["arm0"] = spy
    run_ticks(loop, cell, 1)
    fut = bus.commands.submit(Command(
        op="execute_plan",
        args={"waypoints": {"arm0": [goal]}, "gripper": {"arm0": 0.3}, "interruptible": True},
        source="internal",
    ))
    t = run_ticks(loop, cell, 1)
    assert fut.result(0).ok
    t = run_ticks(loop, cell, 6, t)  # the goal step happened and was held
    assert loop.plans.active_arms == ["arm0"]  # NOT done: back in the executor
    assert loop._plan_state == {"arm0": "executing"} and loop._plan_status == "executing"
    assert loop.plan_cancel_reason is None
    assert loop._last_cmd["arm0"][0] == pytest.approx(0.08)  # one slew step short
    assert spy.puts == [] and loop._grip_frac["arm0"] == 1.0  # deferred gripper untouched
    t = run_ticks(loop, cell, 25, t)  # > plan_gate_hold_s (0.2 s = 20 ticks)
    assert not loop.plans.active_arms
    assert loop.plan_cancel_reason == f"{GATE_HOLD_PREFIX}: arm0_link6 / arm1_link3 at 5.2 mm"
    assert loop._plan_status == "cancelled" and spy.puts == []
    assert loop._last_cmd["arm0"][0] == pytest.approx(0.08)
    # ... and a hold that lifts in time finishes the plan at the goal, gripper applied
    cell, bus, loop = _gated_loop(tmp_path, GoalHoldingGate(goal, hold_ticks=8), 0.2)
    spy = GripSpy()
    loop._senders["arm0"] = spy
    run_ticks(loop, cell, 1)
    fut = bus.commands.submit(Command(
        op="execute_plan",
        args={"waypoints": {"arm0": [goal]}, "gripper": {"arm0": 0.3}, "interruptible": True},
        source="internal",
    ))
    t = run_ticks(loop, cell, 8, t)
    assert fut.result(0).ok and loop.plans.active_arms == ["arm0"] and spy.puts == []
    t = run_ticks(loop, cell, 10, t)
    assert not loop.plans.active_arms and loop.plan_cancel_reason is None
    assert loop._plan_status == "done" and loop._last_cmd["arm0"] == pytest.approx(goal)
    assert spy.puts == [0.3] and loop._grip_frac["arm0"] == 0.3


class BandHoldingGate(HoldingGate):
    """A hold INSIDE the gate's hysteresis band: the report carries no pair (and its
    ``min_clearance_m`` is the 1.0 m default) while ``_block_pairs`` still names it."""

    def __init__(self, twin=None) -> None:
        super().__init__()
        self.twin = twin

    def filter(self, q_cmd, q_meas, source=CommandSource.TELEOP, ts=None, stale=False):
        dec = super().filter(q_cmd, q_meas, source, ts, stale)
        if dec.blocked:
            dec.report = CollisionReport(blocked=True, severity="blocked", pairs=[])
        return dec


def test_gate_hold_reason_inside_the_hysteresis_band_never_prints_the_default_distance(
    tmp_path,
):
    """2026-09-08 review: the fallback to ``gate._block_pairs`` formatted the distance from
    the pair-less report, so the reason read "... at 1000.0 mm". With a twin the real
    distance at the measured posture is asked for; without one the pair stands alone."""
    twin = SimpleNamespace(pair_distance=lambda pair, q, distmax: 0.0061)
    for gate, expect in (
        (BandHoldingGate(twin), f"{GATE_HOLD_PREFIX}: arm0_link6 / arm1_link3 at 6.1 mm"),
        (BandHoldingGate(None), f"{GATE_HOLD_PREFIX}: arm0_link6 / arm1_link3"),
    ):
        cell, bus, loop = _gated_loop(tmp_path, gate, 0.1)
        run_ticks(loop, cell, 1)
        goal = [float(x) for x in loop._last_cmd["arm0"]]
        goal[0] += 1.0
        t = _execute(bus, loop, cell, goal)
        run_ticks(loop, cell, 15, t)
        assert not loop.plans.active_arms
        assert loop.plan_cancel_reason == expect
        assert "1000" not in loop.plan_cancel_reason


# -- hand-over on MEASURED arrival (2026-09-08 review) -----------------------------------------
def _lagging_rail_session(tmp_path, *, rail_speed_m_s: float, order=("arm0", "arm1")):
    """The hand-built two-arm fake session with arm0's TRACK following its targets at
    ``rail_speed_m_s`` (the loop commands the rail slot at ``jog.rail_m_per_tick`` x 100 Hz
    = 0.2 m/s) and a return profile that moves arm0's carriage to 0.1 m."""
    cell = FakeWorkcell({
        "arm0": FakeArm("arm0", has_rail=True, max_rail_speed_m_s=rail_speed_m_s),
        "arm1": FakeArm("arm1"),
    })
    cell.start()
    bus = RuntimeBus()
    loop = ControlLoop(
        cell, ControlConfig(), bus, SafetySupervisor(NullGate(), InputWatchdog()), list(ARMS),
        profile_store=ProfileStore(tmp_path / "profiles"), workcell_kind="sim",
    )
    fl = (cell, bus, loop)
    manager, session, rt, _ = make(tmp_path, fl, twin=OrderedTwin(list(order), step=0.05))
    q = {a: list(cell.arms[a].get_state().q[:7]) for a in ARMS}
    prof = manager.profile_store.save(StateProfile(
        name="rail", workcell_kind="sim",
        arms={"arm0": ArmPosture(q=q["arm0"], rail_pos_m=0.1), "arm1": ArmPosture(q=q["arm1"])},
    ))
    return fl, manager, session, rt, prof


def test_the_next_arm_waits_for_the_previous_arms_measured_arrival(tmp_path, monkeypatch):
    """arm0's carriage is commanded at 0.2 m/s but its track follows at 0.05 m/s, so the
    executor retires arm0's waypoints ~1.5 s before the carriage is there. Before: arm1
    started moving at once against a carriage the gate believed parked (it checks the
    COMMANDED q). Now arm1's plan is submitted only once arm0's MEASURED rail is within
    2 mm of the goal, and the return reports done only when both measured arms are."""
    (cell, bus, loop), manager, session, rt, prof = _lagging_rail_session(
        tmp_path, rail_speed_m_s=0.05
    )
    rail_at_submit: list[float] = []

    def before(loop_, n):
        if n == 2:
            rail_at_submit.append(float(cell.arms["arm0"].get_state().q[7]))

    calls = _spy_execute_plan(monkeypatch, before=before)
    rec = Recorder(loop, cell)
    rec.tick()
    done = _worker(lambda: manager._return_home_worker(session, prof, "saved"))
    _drive(rec, done, seconds=30.0, session=session)
    assert rt.calls[-1] == (False, "")
    assert [list(c["waypoints"]) for c in calls] == [["arm0"], ["arm1"]]
    assert rail_at_submit and abs(rail_at_submit[0] - 0.1) <= manager_mod.PLAN_ARRIVAL_TOL_RAIL_M
    # the command reached 0.1 m (0.5 s = 50 ticks) long before the track did (2 s = 200 ticks)
    cmd_arrived = next(i for i, (q, _, _) in enumerate(rec.rows) if q["arm0"][7] >= 0.1 - 1e-9)
    started = rec.first_move_tick("arm1")
    assert started is not None and started - cmd_arrived > 100, (started, cmd_arrived)
    _assert_sequential(rec, "arm0", "arm1")
    st = cell.states()
    assert abs(st["arm0"].q[7] - 0.1) <= manager_mod.PLAN_ARRIVAL_TOL_RAIL_M
    assert np.allclose(st["arm1"].q[:7], np.asarray(prof.arms["arm1"].q) + 0.05, atol=1e-3)


def test_an_arm_that_never_settles_is_reported_stalled_and_stops_the_sequence(
    tmp_path, monkeypatch
):
    """arm0's track does not move at all (a stuck carriage): the command arrives, the
    measured rail stays at 0.0, and instead of declaring arm0 done and moving arm1 the
    manager reports ``stalled`` with how far off it is and stops the sequence."""
    (cell, bus, loop), manager, session, rt, prof = _lagging_rail_session(
        tmp_path, rail_speed_m_s=0.0
    )
    monkeypatch.setattr(manager_mod, "PLAN_ARRIVAL_GRACE_S", 0.3)
    monkeypatch.setattr(SessionManager, "_return_budget_s", lambda self, s, w: 2.0)
    calls = _spy_execute_plan(monkeypatch)
    q1_0 = loop._last_cmd["arm1"].copy()
    rec = Recorder(loop, cell)
    rec.tick()
    done = _worker(lambda: manager._return_home_worker(session, prof, "saved"))
    _drive(rec, done, seconds=30.0, session=session)
    assert len(calls) == 1 and list(calls[0]["waypoints"]) == ["arm0"]
    detail = rt.calls[-1][1]
    assert rt.calls[-1][0] is False
    assert detail.startswith("return stopped - did not arrive: joints off by 0.0 mrad, "
                             "carriage off by 100 mm after ")
    assert detail.endswith(" s - arm0; arm1 not moved")
    assert np.allclose(loop._last_cmd["arm1"], q1_0)  # arm1 never started
    assert loop._last_cmd["arm0"][7] == pytest.approx(0.1)  # the COMMAND did arrive
    assert cell.arms["arm0"].get_state().q[7] == 0.0  # the carriage did not
    # the reset / exit path maps it to the wire's ``timeout`` with the operator sentence
    session.workcell = cell
    out: dict = {}
    done = _worker(lambda: out.setdefault(
        "res", manager._return_to_initial_motion(session, prof, label="return_home")
    ))
    _drive(rec, done, seconds=30.0, session=session)
    res = out["res"]
    assert (res.ok, res.status) == (False, "timeout")
    # two phases here (the first return left arm0's joints at the twin's +0.05 goal, so
    # the joints phase runs first and arrives); in the carriage phase arm1 is parked
    # (single moving arm: no arm suffix) and arm0's carriage stalls again
    assert res.detail.startswith(
        "an arm did not settle at its goal while moving the carriage (did not arrive: "
        "joints off by 0.0 mrad, carriage off by 100 mm after "
    )
    assert res.detail.endswith(" s). The arms hold where they are.")


def test_start_from_gripper_only_profile_still_applies_the_gripper(
    tmp_path, fake_loop, monkeypatch
):
    """Regression of the 2026-09-08 sequential change: a profile that leaves every arm in
    place (``_moving_arms`` empty) returned "done" without submitting anything, so its
    gripper targets were dropped (the single pre-change ``execute_plan`` applied them
    even with parked waypoints). Now ONE gripper-only plan is submitted and applied."""
    from test_return_plan_units import GripSpy

    cell, bus, loop = fake_loop
    manager, session, prof = _start_from_session(tmp_path, fake_loop, 0.0)

    class ParkedTwin(FakeTwin):
        def plan(self, req):
            self.plans.append(req)
            return PlanResult(
                ok=True,
                waypoints={a: [list(req.q_start[a])] * 2 for a in req.q_goal},
                arm_order=list(req.q_goal),
            )

    session.twin = ParkedTwin()
    q = {a: list(cell.arms[a].get_state().q[:7]) for a in ARMS}
    grip_prof = manager.profile_store.save(StateProfile(
        name="grip-only", workcell_kind="sim",
        arms={a: ArmPosture(q=q[a], gripper_open_frac=0.3) for a in ARMS},
    ))
    session.spec = session.spec.model_copy(update={"start_from": f"profile:{grip_prof.profile_id}"})
    spies = {a: GripSpy() for a in ARMS}
    loop._senders.update(spies)
    calls = _spy_execute_plan(monkeypatch)
    assert loop._grip_frac["arm0"] == 1.0
    done = _worker(lambda: manager._start_from_worker(session))
    tick_until(loop, cell, done.is_set, 10.0)
    assert [(c["waypoints"], c["gripper"]) for c in calls] == [({}, {"arm0": 0.3, "arm1": 0.3})]
    assert loop._grip_frac == {"arm0": 0.3, "arm1": 0.3}
    assert spies["arm0"].puts == [0.3] and spies["arm1"].puts == [0.3]
    assert session.state is SessionState.RUNNING and session.motion_detail == ""
    assert session.start_from_progress is None and not loop.plans.active_arms
    assert loop._plan_status in ("done", None)


def test_a_refused_later_arm_names_the_arms_that_moved(tmp_path, fake_loop, monkeypatch):
    """arm1 faults while arm0 walks: its plan is refused. Before, the notice was the loop's
    words verbatim ("arm 'arm1' is faulted") as if nothing had moved; now it names the
    refused arm like the cancelled / held / timeout branches do."""
    cell, bus, loop = fake_loop
    manager, session, rt, prof = make(tmp_path, fake_loop, twin=OrderedTwin(["arm0", "arm1"]))

    def before(loop_, n):
        if n == 2:
            loop_._faulted.add("arm1")  # what a FaultEvent does to the loop's sets

    calls = _spy_execute_plan(monkeypatch, before=before)
    done = _worker(lambda: manager._return_home_worker(session, prof, "saved"))
    tick_until(loop, cell, done.is_set, 20.0)
    assert len(calls) == 2
    assert rt.calls[-1] == (False, "return refused: arm 'arm1' is faulted - arm1")
    assert loop._last_cmd["arm0"][:7] == pytest.approx(np.asarray(prof.arms["arm0"].q) + 0.5)
    # a refusal of the FIRST arm stays verbatim (nothing moved; ``_start_from_refusal`` and
    # the existing tests parse it)
    assert SessionManager._sequence_detail("x", ["arm0", "arm1"], 1) == "x - arm1"
