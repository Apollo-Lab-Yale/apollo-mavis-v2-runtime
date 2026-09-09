"""SessionManager's return-to-start plumbing on a hand-built session (04-runtime §10.5;
2026-09-07 review items 8 / 9 / 12): the hook skips a return while the browser's
deadman is latched or the session is not RUNNING, a planning failure produces no
motion, a return past its budget is cancelled THROUGH the loop (``cancel_plan``),
the budget derives from the plan, and a refused ``execute_plan`` surfaces as a
``start_from refused`` notice (``ActiveSession.motion_detail`` -> telemetry
``session.fault_detail``).

start_from robustness (2026-09-08; ``hardware_session.start_from_fault_grace_s``): the
worker waits for a TRANSIENT post-bring-up fault (an arm RECOVERING for a tick) to clear
before it hands the pre-planned motion to the loop, retries ONCE when the arms clear
right after a refusal, and on a persistent fault names the arm, its controller state and
error code - and moves nothing. 2026-09-08 review: the notice survives the fault cycle
(``fault_detail`` is wiped when the session runs again, the hint must not be), a RECOVERING
arm is told to release its inputs (not to clear a C0), the worker's bookkeeping follows
the executor rather than ``session.state`` (a transient fault mid-plan used to leave the
bring-up rows on screen for good), and the twin-planned profile motions are serialized
behind one manager claim."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from apollo_mavis_v2_core import ArmPosture, HeldState, PlanResult, ProfileStore, StateProfile
from apollo_mavis_v2_core.protocol import SessionSpec
from conftest import make_runtime_config, run_ticks

from apollo_mavis_v2_runtime.config import HardwareSessionConfig
from apollo_mavis_v2_runtime.control.loop import ControlLoop
from apollo_mavis_v2_runtime.session.manager import (
    MOTION_BUSY,
    ActiveSession,
    SessionManager,
    _BringupProgress,
)
from apollo_mavis_v2_runtime.session.types import SessionState
from apollo_mavis_v2_runtime.streams.hub import VideoHub


class FakeRT:
    """The RecorderThread surface the manager touches."""

    def __init__(self) -> None:
        self.calls: list[tuple[bool, str]] = []

    def set_returning(self, active: bool, detail: str = "") -> None:
        self.calls.append((active, detail))

    @property
    def detail(self) -> str:
        return self.calls[-1][1] if self.calls else ""


class FakeTwin:
    def __init__(self, ok: bool = True, far: bool = False) -> None:
        self.ok, self.far = ok, far
        self.plans: list = []
        self.synced = 0

    def sync(self, states) -> None:
        self.synced += 1

    def plan(self, req):
        self.plans.append(req)
        if not self.ok:
            return PlanResult(ok=False, waypoints={}, failure="goal_in_collision",
                              failing_pair=("arm0_link6", "table"))
        wps = {}
        for arm, goal in req.q_goal.items():
            start = list(req.q_start[arm])
            if self.far:  # a long straight segment: many slew ticks
                goal = [g + 5.0 for g in goal]
            wps[arm] = [start, list(goal)]
        return PlanResult(ok=True, waypoints=wps)


class LongTwin(FakeTwin):
    """A plan long enough to inject driver events mid-motion (0.5 rad on every joint,
    the rail slot kept) that still ARRIVES - ``FakeTwin(far=True)`` never does (its +5 m
    rail target is clamped to the travel)."""

    def plan(self, req):
        self.plans.append(req)
        wps = {}
        for arm, goal in req.q_goal.items():
            start = list(req.q_start[arm])
            wps[arm] = [start, [g + 0.5 for g in goal[:7]] + list(goal[7:])]
        return PlanResult(ok=True, waypoints=wps)


def make(tmp_path, fake_loop, *, twin=None, return_to_start=True, start_from="keep_current"):
    cell, bus, loop = fake_loop
    cfg = make_runtime_config(tmp_path)
    store = ProfileStore(tmp_path / "profiles")
    manager = SessionManager(cfg, bus, VideoHub(bus), store, "epoch")
    # the designated initial condition = the arms' current posture + a small offset
    run_ticks(loop, cell, 1)
    q = {a: list(cell.arms[a].get_state().q[:7]) for a in ("arm0", "arm1")}
    prof = store.save(StateProfile(
        name="initial", workcell_kind="sim", is_initial_condition=True,
        arms={a: ArmPosture(q=[x + 0.05 for x in q[a]], rail_pos_m=None) for a in q},
    ))
    store.set_initial(prof.profile_id)
    spec = SessionSpec(mode="collect", kind="sim", arms=["arm0", "arm1"], frames={},
                       sim_scene="single_rail", task="t", return_to_start=return_to_start,
                       start_from=start_from)
    rt = FakeRT()
    session = SimpleNamespace(
        spec=spec, state=SessionState.RUNNING, recorder_thread=rt, loop=loop,
        supervisor=loop.supervisor, workcell=cell, twin=twin or FakeTwin(),
        session_id="s", fault_detail="", motion_detail="", start_from_progress=None,
        planned_start=None,
    )
    manager.session = session
    return manager, session, rt, prof


def tick_until(loop, cell, pred, seconds: float = 5.0) -> None:
    t0 = time.monotonic()
    t = loop._clock() if hasattr(loop, "_clock") else 0.0
    while time.monotonic() - t0 < seconds:
        t = run_ticks(loop, cell, 1, t)
        if pred():
            return
        time.sleep(0.002)
    raise AssertionError("condition not met")


def test_hook_skips_the_return_while_the_deadman_is_latched_or_not_running(tmp_path, fake_loop):
    manager, session, rt, _ = make(tmp_path, fake_loop)
    session.supervisor.watchdog.on_disconnect()  # AWAIT_EMPTY
    manager._on_episode_done("saved", 0)
    assert rt.calls == [(False, "return skipped: browser input latched - release every key")]
    rt.calls.clear()
    empty = HeldState(held=frozenset(), seq=1, rx_mono=time.monotonic())
    session.supervisor.watchdog.on_keys(empty)
    session.state = SessionState.FAULT
    manager._on_episode_done("discarded", None)
    assert rt.calls == [(False, "return skipped: session fault")]
    rt.calls.clear()
    session.state = SessionState.RUNNING
    session.spec = session.spec.model_copy(update={"return_to_start": False})
    manager._on_episode_done("saved", 0)
    assert rt.calls == []  # unticked: nothing at all


def test_planning_failure_reports_and_moves_nothing(tmp_path, fake_loop):
    cell, bus, loop = fake_loop
    manager, session, rt, prof = make(tmp_path, fake_loop, twin=FakeTwin(ok=False))
    before = {a: loop._last_cmd[a].copy() for a in ("arm0", "arm1")}
    manager._return_home_worker(session, prof, "saved")
    assert rt.calls[-1][0] is False
    assert rt.calls[-1][1].startswith("return failed: goal_in_collision (arm0_link6 / table)")
    run_ticks(loop, cell, 3)
    assert not loop.plans.active_arms
    assert all(np.allclose(loop._last_cmd[a], before[a]) for a in before)


def test_return_executes_through_the_loop_and_completes(tmp_path, fake_loop):
    cell, bus, loop = fake_loop
    manager, session, rt, prof = make(tmp_path, fake_loop)
    done = threading.Event()

    def worker():
        manager._return_home_worker(session, prof, "saved")
        done.set()

    threading.Thread(target=worker, daemon=True).start()
    tick_until(loop, cell, done.is_set, 10.0)
    assert rt.calls[-1] == (False, "")
    for a in ("arm0", "arm1"):
        assert loop._last_cmd[a][:7] == pytest.approx(prof.arms[a].q, abs=1e-6)


def test_return_past_its_budget_is_cancelled_through_the_loop(tmp_path, fake_loop, monkeypatch):
    cell, bus, loop = fake_loop
    manager, session, rt, prof = make(tmp_path, fake_loop, twin=FakeTwin(far=True))
    monkeypatch.setattr(SessionManager, "_return_budget_s", lambda self, s, w: 0.3)
    done = threading.Event()

    def worker():
        manager._return_home_worker(session, prof, "saved")
        done.set()

    threading.Thread(target=worker, daemon=True).start()
    tick_until(loop, cell, done.is_set, 10.0)
    assert rt.calls[-1] == (False, "return timed out - held by the gate; arm stopped")
    assert not loop.plans.active_arms
    assert loop.plan_cancel_reason == "return-to-start: return timed out"


def test_return_budget_derives_from_the_plan(tmp_path, fake_loop):
    cell, bus, loop = fake_loop
    manager, session, rt, prof = make(tmp_path, fake_loop)
    jog, rate = loop.cfg.jog, 1.0 / loop.dt
    assert manager._return_budget_s(session, {"arm0": [[0.0] * 8, [0.01] * 8]}) == 30.0  # min
    far = {"arm0": [[0.0] * 8, [20.0] * 7 + [0.0]], "arm1": [[0.0] * 8, [0.1] * 8]}
    expect = 3.0 * (20.0 / jog.slew_rad_per_tick) / rate + 10.0  # the LONGEST arm x3 + 10 s
    assert expect > 30.0 and manager._return_budget_s(session, far) == pytest.approx(expect)
    rail = {"arm0": [[0.0] * 8, [0.0] * 7 + [6.0]]}  # the rail slot at its own per-tick slew
    assert manager._return_budget_s(session, rail) == pytest.approx(
        max(30.0, 3.0 * (6.0 / jog.rail_m_per_tick) / rate + 10.0)
    )


# -- start_from robustness (2026-09-08) ------------------------------------------------------
REFUSAL_HINT = "use Clear errors & resume, then Go to profile"


def _start_from_session(tmp_path, fake_loop, grace_s: float, *, twin=None):
    """A hand-built session in BRINGUP whose ``start_from`` is the initial profile, with
    the fault grace set to ``grace_s`` (the runtime default is 3 s)."""
    cell, bus, loop = fake_loop
    manager, session, rt, prof = make(tmp_path, fake_loop, twin=twin)
    manager.cfg = manager.cfg.model_copy(
        update={
            "hardware_session": HardwareSessionConfig(armed=True, start_from_fault_grace_s=grace_s)
        }
    )
    session.spec = session.spec.model_copy(update={"start_from": f"profile:{prof.profile_id}"})
    session.state = SessionState.BRINGUP
    run_ticks(loop, cell, 1)
    return manager, session, prof


def _spy_execute_plan(monkeypatch, before=None, after=None) -> list:
    """Count ``ControlLoop._op_execute_plan`` calls; ``before`` / ``after`` (loop, n) run
    around the n-th call so a test can make the loop's fault sets change exactly as the
    plan is drained."""
    calls: list = []
    real = ControlLoop._op_execute_plan

    def spy(self, cmd):
        calls.append(cmd.args)
        n = len(calls)
        if before is not None:
            before(self, n)
        try:
            return real(self, cmd)
        finally:
            if after is not None:
                after(self, n)

    monkeypatch.setattr(ControlLoop, "_op_execute_plan", spy)
    return calls


def _run_start_from(manager, session, loop, cell, seconds: float = 10.0) -> None:
    done = threading.Event()

    def worker():
        manager._start_from_worker(session)
        done.set()

    threading.Thread(target=worker, daemon=True).start()
    tick_until(loop, cell, done.is_set, seconds)


def test_start_from_waits_out_a_transient_recovery_then_executes(tmp_path, fake_loop, monkeypatch):
    """2026-09-08 18:44:52 on the real cell: the Perception Arm was RECOVERING for ONE
    tick (controller state 4 right after enabling) as the pre-planned motion reached the
    loop; the loop refused it and the plan was dropped for good. The worker now waits
    for the arm to clear (the loop lifts RECOVERING once no input is held) and the
    motion executes."""
    cell, bus, loop = fake_loop
    manager, session, prof = _start_from_session(tmp_path, fake_loop, 2.0)
    calls = _spy_execute_plan(monkeypatch)
    loop._recovering.add("arm0")  # what a ReseedEvent / RecoveredEvent leaves behind
    done = threading.Event()

    def worker():
        manager._start_from_worker(session)
        done.set()

    threading.Thread(target=worker, daemon=True).start()
    time.sleep(0.15)  # the worker polls while the arm is still RECOVERING: nothing sent
    assert calls == [] and session.state is SessionState.BRINGUP
    assert session.start_from_progress == 0.0
    tick_until(loop, cell, done.is_set, 10.0)  # the first tick clears RECOVERING
    # one execute_plan PER ARM (2026-09-08 evening: sequential execution)
    assert len(calls) == 2 and session.motion_detail == "" and session.fault_detail == ""
    assert [list(c["waypoints"]) for c in calls] == [["arm0"], ["arm1"]]
    assert session.state is SessionState.RUNNING and session.start_from_progress is None
    for a in ("arm0", "arm1"):
        assert loop._last_cmd[a][:7] == pytest.approx(prof.arms[a].q, abs=1e-6)


def test_start_from_retries_once_when_the_arms_clear_right_after_a_refusal(
    tmp_path, fake_loop, monkeypatch
):
    """The arms were clear when the worker submitted, a fault landed exactly as the loop
    drained the plan (refused) and cleared again: ONE retry within the grace, then the
    motion runs; the refusal never reaches the notice."""
    cell, bus, loop = fake_loop
    manager, session, prof = _start_from_session(tmp_path, fake_loop, 2.0)

    def fault_on_first(loop_, n):
        if n == 1:
            loop_._faulted.add("arm0")

    def clear_after_first(loop_, n):
        if n == 1:
            loop_._faulted.discard("arm0")

    calls = _spy_execute_plan(monkeypatch, before=fault_on_first, after=clear_after_first)
    _run_start_from(manager, session, loop, cell)
    # the first arm's plan: refused, then retried with the SAME pre-planned waypoints;
    # the second arm follows once the first arrived (sequential execution)
    assert len(calls) == 3 and calls[0] == calls[1]
    assert [list(c["waypoints"]) for c in calls] == [["arm0"], ["arm0"], ["arm1"]]
    assert session.motion_detail == "" and session.state is SessionState.RUNNING
    for a in ("arm0", "arm1"):
        assert loop._last_cmd[a][:7] == pytest.approx(prof.arms[a].q, abs=1e-6)


def test_start_from_refused_by_a_persistent_fault_names_the_arm_and_moves_nothing(
    tmp_path, fake_loop, monkeypatch
):
    """A fault that outlives the grace: the plan is handed over once (the loop refuses
    it), NOT retried, and the notice (``motion_detail`` - NOT ``fault_detail``, which the
    fault callback owns and wipes on recovery) says which arm, its controller state and
    error code and what to do; the session is RUNNING (arms held) and nothing moved."""
    cell, bus, loop = fake_loop
    manager, session, prof = _start_from_session(tmp_path, fake_loop, 0.3)
    calls = _spy_execute_plan(monkeypatch)
    cell.arms["arm0"].inject_error(31)  # the driver's view: C31 latched
    loop._faulted.add("arm0")  # what _on_fault_event leaves behind for it
    before = {a: loop._last_cmd[a].copy() for a in ("arm0", "arm1")}
    t0 = time.monotonic()
    _run_start_from(manager, session, loop, cell)
    assert time.monotonic() - t0 >= 0.3  # the whole grace was given
    assert session.motion_detail == (
        f"start_from refused: arm0 faulted (controller state 0, code C31) - {REFUSAL_HINT}"
    )
    assert session.fault_detail == ""  # the arm rows carry the fault; the notice is separate
    assert len(calls) == 1  # no retry while the fault persists
    assert session.state is SessionState.RUNNING and session.start_from_progress is None
    assert manager._bringup is None
    assert not loop.plans.active_arms
    run_ticks(loop, cell, 3)
    assert all(np.allclose(loop._last_cmd[a], before[a]) for a in before)  # no motion


def test_start_from_refusal_text_names_every_stuck_arm(tmp_path, fake_loop):
    """A FAULTED arm is told to clear its error; a RECOVERING arm is lifted by releasing
    every live input (``_update_recovering``), so it is told THAT - never to clear a C0
    with 'controller state 0' (a clutch held through bring-up keeps the arm RECOVERING
    for the whole grace)."""
    cell, bus, loop = fake_loop
    manager, session, _ = _start_from_session(tmp_path, fake_loop, 0.0)
    loop._faulted.add("arm0")
    loop._recovering.add("arm1")
    cell.arms["arm1"].inject_error(22)
    assert manager._start_from_refusal(session, "arm 'arm0' is faulted") == (
        "start_from refused: arm0 faulted (controller state 0, code C0) - use Clear errors & "
        "resume; arm1 is recovering - release every input (clutch / keys), then Go to profile"
    )
    loop._faulted.clear()
    assert manager._start_from_refusal(session, "arm 'arm1' is faulted") == (
        "start_from refused: arm1 is recovering - release every input (clutch / keys), "
        "then Go to profile"
    )
    loop._recovering.clear()
    # cleared between the refusal and the report: the loop's ack still names the arm
    assert manager._start_from_refusal(session, "arm 'arm1' is faulted").startswith(
        "start_from refused: arm1 faulted (controller state 0, code C22) - "
    )
    assert manager._start_from_refusal(session, "unknown arm 'x'") == (
        "start_from refused: unknown arm 'x'"
    )


def test_start_from_grace_zero_submits_at_once(tmp_path, fake_loop, monkeypatch):
    """``start_from_fault_grace_s: 0`` is the pre-2026-09-08 behaviour: the plan is
    handed over immediately, a faulted arm refuses it (the operator then uses Go to
    profile) and no grace is waited."""
    cell, bus, loop = fake_loop
    manager, session, prof = _start_from_session(tmp_path, fake_loop, 0.0)
    calls = _spy_execute_plan(monkeypatch)
    loop._faulted.add("arm0")
    t0 = time.monotonic()
    _run_start_from(manager, session, loop, cell)
    assert time.monotonic() - t0 < 1.0
    assert len(calls) == 1
    assert session.motion_detail.startswith("start_from refused: arm0 faulted (")
    assert session.state is SessionState.RUNNING and not loop.plans.active_arms


def test_start_from_bookkeeping_survives_a_transient_fault_mid_plan(
    tmp_path, fake_loop, monkeypatch
):
    """The fault callback is LIVE (``attach_fault_state``, as in the runtime) and a fault
    lands AFTER the plan was accepted: the session walks START_FROM -> FAULT -> RUNNING
    while the plan keeps executing. The worker used to follow ``session.state`` and exit
    at once, leaving ``start_from_progress`` None and the bring-up rows on screen for the
    rest of the session; it now follows the executor and clears both when the motion is
    over."""
    cell, bus, loop = fake_loop
    manager, session, prof = _start_from_session(tmp_path, fake_loop, 0.5, twin=LongTwin())
    manager.attach_fault_state(session)
    manager._bringup = _BringupProgress(session_id="s", spec=session.spec)  # rows on screen
    _spy_execute_plan(monkeypatch)
    done = threading.Event()

    def worker():
        manager._start_from_worker(session)
        done.set()

    threading.Thread(target=worker, daemon=True).start()
    tick_until(
        loop,
        cell,
        lambda: bool(loop.plans.active_arms) and session.state is SessionState.START_FROM,
    )
    assert session.start_from_progress is not None and manager._bringup is not None
    loop._faulted.add("arm0")  # a transient driver fault on one arm ...
    run_ticks(loop, cell, 2)  # ... published: START_FROM -> FAULT
    assert session.state is SessionState.FAULT and not done.is_set()
    loop._faulted.discard("arm0")  # ... and gone again (re-seeded, inputs released)
    run_ticks(loop, cell, 2)  # published: FAULT -> RUNNING with the plan still executing
    assert session.state is SessionState.RUNNING and loop.plans.active_arms
    assert not done.is_set()  # the worker keeps following the executor
    tick_until(loop, cell, done.is_set, 30.0)
    assert session.state is SessionState.RUNNING
    assert session.start_from_progress is None
    assert manager._bringup is None  # the bring-up rows are gone
    assert not loop.plans.active_arms and session.motion_detail == ""


def test_the_notice_and_the_fault_text_are_kept_apart_for_the_wire(tmp_path, fake_loop):
    """``ActiveSession.notice()`` is what ``SessionTelemetry.fault_detail`` /
    ``SessionInfo.fault_detail`` carry: the motion notice first (it must survive the
    fault cycle that wipes ``fault_detail``), else the fault text - but only while no
    arm row already shows a fault (the FaultBanner lists those per arm)."""
    cell, bus, loop = fake_loop
    manager, session, rt, prof = make(tmp_path, fake_loop)
    real = ActiveSession(
        session_id="s", spec=session.spec, state=SessionState.RUNNING, workcell=cell, loop=loop,
        supervisor=loop.supervisor, twin=None, render_service=None,
    )
    assert real.notice() == "" and real.notice(arms_carry_faults=True) == ""
    real.fault_detail = "arm0: controller error 24: Speed Exceeds Limit"
    assert real.notice() == real.fault_detail  # a fault before the first snapshot
    assert real.notice(arms_carry_faults=True) == ""  # the arm row says it already
    real.motion_detail = (
        f"start_from refused: arm0 faulted (controller state 4, code C24) - {REFUSAL_HINT}"
    )
    assert real.notice() == real.motion_detail
    assert real.notice(arms_carry_faults=True) == real.motion_detail
    # the fault callback's own bookkeeping never touches the notice
    manager.session = real
    manager.attach_fault_state(real)
    manager._on_arm_fault_state(real, "fault")
    manager._on_arm_fault_state(real, None)
    assert real.state is SessionState.RUNNING and real.fault_detail == ""
    assert real.motion_detail.startswith("start_from refused")
    assert manager.info().fault_detail == real.motion_detail


def test_profile_motions_are_serialized_behind_one_claim(tmp_path, fake_loop):
    """The manager-side "plan executing" blockers run BEFORE a worker plans, so the
    per-episode return, `R`, Go to profile and the exit return could all pass them and
    hand the loop a second ``execute_plan`` mid-motion. One claim, token-released."""
    cell, bus, loop = fake_loop
    manager, session, rt, prof = make(tmp_path, fake_loop)
    manager.attach_fault_state(session)
    token = manager._claim_profile_motion("goto_profile")
    assert token is not None and manager.profile_motion_in_flight == "goto_profile"
    assert manager._claim_profile_motion("reset_to_initial") is None
    assert manager.request_goto_profile(prof) == (False, MOTION_BUSY)
    assert manager.request_reset_to_initial(prof) == (False, MOTION_BUSY)
    res = manager.return_to_initial()
    assert (res.ok, res.status, res.detail) == (False, "refused", MOTION_BUSY)
    manager._on_episode_done("saved", 0)
    assert rt.calls == [(False, f"return skipped: {MOTION_BUSY}")]
    manager._release_profile_motion(object())  # a foreign token releases nothing
    assert manager.profile_motion_in_flight == "goto_profile"
    manager._release_profile_motion(token)
    assert manager.profile_motion_in_flight is None
    assert not loop.plans.active_arms  # nothing was ever handed to the loop
    # the per-episode return holds the claim while it runs and releases it when done
    rt.calls.clear()
    manager._on_episode_done("saved", 0)
    assert rt.calls[0] == (True, "returning to profile 'initial'")
    tick_until(loop, cell, lambda: rt.calls[-1] == (False, ""), 10.0)
    assert _wait_release(manager)


def _wait_release(manager, seconds: float = 2.0) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        if manager.profile_motion_in_flight is None:
            return True
        time.sleep(0.002)
    return False


def test_dagger_sessions_return_to_start_too(tmp_path, fake_loop):
    """D6 (15-online-dagger, 2026-09-08): the hook fires for ``mode: dagger`` exactly like
    collect (default ON), the create-time check refuses a dagger POST without any
    return profile, and teleop / inference stay out of it."""
    from apollo_mavis_v2_runtime.errors import SessionError

    cell, bus, loop = fake_loop
    manager, session, rt, prof = make(tmp_path, fake_loop)
    dagger = SessionSpec(mode="dagger", kind="sim", arms=["arm0", "arm1"], frames={},
                         sim_scene="single_rail", task="t", policy_source="external")
    assert dagger.return_to_start is True  # default ON for dagger too
    manager._check_return_to_start(dagger)  # the initial-condition profile exists
    with pytest.raises(SessionError, match="return_to_start needs a start_from profile"):
        manager._check_return_to_start(dagger.model_copy(update={"start_from": "profile:nope"}))
    manager._check_return_to_start(dagger.model_copy(update={"return_to_start": False}))
    session.spec = dagger
    manager._on_episode_done("saved", 0)  # starts the return worker
    tick_until(loop, cell, lambda: rt.calls and rt.calls[-1] == (False, ""), 10.0)
    assert rt.calls[0] == (True, "returning to profile 'initial'")
    for a in ("arm0", "arm1"):
        assert loop._last_cmd[a][:7] == pytest.approx(prof.arms[a].q, abs=1e-6)
    rt.calls.clear()
    session.spec = SessionSpec(mode="inference", kind="sim", arms=["arm0", "arm1"], frames={},
                               sim_scene="single_rail", policy_source="external")
    manager._on_episode_done("saved", 0)
    assert rt.calls == []  # never for inference
