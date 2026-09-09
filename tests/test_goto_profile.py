"""``goto_profile {profile_id}`` — "Go to profile" (2026-09-08 operator request; mirrors
``tests/test_reset_to_initial.py``).

Unit half: ``ControlLoop._op_goto_profile`` validates inline (unknown id, a profile of
another workcell kind, one covering no session arm, recording, a running plan, no
manager) and hands the CHOSEN profile to the SessionManager hook without blocking the
loop; ``SessionManager.request_goto_profile`` applies the reset-to-initial blockers
and drives the two-phase (joints, then carriage) motion through the gated
``execute_plan`` path on a fake workcell.

E2E half over a real sim server: the ``goto_profile`` action over /ws/control walks the
arm (joints AND rail) to a saved profile that is NOT the initial condition, refuses an
unknown id / the wrong workcell kind, and core's ``validate_action_args`` rejects a
missing ``profile_id`` before anything reaches the loop.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from apollo_mavis_v2_core import ArmPosture, Command, StateProfile
from conftest import LiveServer, make_runtime_config, run_ticks
from test_e2e_teleop import PulsingCtl, Tele
from test_return_manager_units import make, tick_until

from apollo_mavis_v2_runtime.control.loop import ControlLoop
from apollo_mavis_v2_runtime.session.manager import MOTION_BUSY
from apollo_mavis_v2_runtime.session.types import SessionState

# -- unit: the loop's inline validation ------------------------------------------------


def _ack(loop, bus, **args):
    fut = bus.commands.submit(Command(op="goto_profile", args=args, source="ws"))
    loop.run_tick()
    return fut.result(timeout=1.0)


def _save(loop, name: str, *, kind: str = "sim", arms=("arm0", "arm1"), rail=None):
    return loop.profile_store.save(
        StateProfile(
            name=name,
            workcell_kind=kind,  # type: ignore[arg-type]
            arms={a: ArmPosture(q=[0.1] * 7, rail_pos_m=rail) for a in arms},
        )
    )


def test_unknown_id_other_kind_uncovered_and_missing_id_are_refused(fake_loop):
    cell, bus, loop = fake_loop
    loop.run_tick()
    called: list[object] = []
    loop.on_goto_profile = lambda profile: (called.append(profile), (True, "moving"))[1]

    ack = _ack(loop, bus, profile_id="nope")
    assert (ack.ok, ack.detail) == (False, "unknown profile 'nope'")
    lab = _save(loop, "lab", kind="hardware")
    ack = _ack(loop, bus, profile_id=lab.profile_id)
    assert (ack.ok, ack.detail) == (False, "profile 'lab' is for the hardware workcell")
    other = _save(loop, "other cell", arms=("nosucharm",))
    ack = _ack(loop, bus, profile_id=other.profile_id)
    assert (ack.ok, ack.detail) == (False, "profile 'other cell' covers no arm of this session")
    ack = _ack(loop, bus)
    assert (ack.ok, ack.detail) == (False, "profile_id required")
    assert called == []  # nothing was asked to move
    assert not loop.plans.active_arms


def test_hands_the_chosen_profile_to_the_manager_hook(fake_loop):
    cell, bus, loop = fake_loop
    loop.run_tick()
    shelf = _save(loop, "shelf", rail=0.2)  # a plain saved profile, NOT the initial condition
    assert loop.profile_store.initial_for("sim") is None
    calls: list[object] = []

    def hook(target):
        calls.append(target)
        return True, f"going to profile '{target.name}'"

    loop.on_goto_profile = hook
    ack = _ack(loop, bus, profile_id=shelf.profile_id)
    assert (ack.ok, ack.detail) == (True, "going to profile 'shelf'")
    # Resolved ONCE on the loop thread and handed over: the manager never re-reads the
    # store from the 100 Hz thread.
    assert [p.profile_id for p in calls] == [shelf.profile_id]


def test_refused_while_recording_during_a_plan_and_without_a_manager(fake_loop):
    cell, bus, loop = fake_loop
    loop.run_tick()
    shelf = _save(loop, "shelf")
    loop.on_goto_profile = lambda profile: (True, "moving")
    loop.episode_state = "recording"
    ack = _ack(loop, bus, profile_id=shelf.profile_id)
    assert (ack.ok, ack.detail) == (False, "recording - save or discard first")
    loop.episode_state = "idle"
    loop._plan_state["arm1"] = "planning"  # a joint-panel goto is being planned
    ack = _ack(loop, bus, profile_id=shelf.profile_id)
    assert (ack.ok, ack.detail) == (False, "plan executing")
    loop._plan_state.clear()
    loop.on_goto_profile = None  # a bare unit loop: refused, never silently ignored
    ack = _ack(loop, bus, profile_id=shelf.profile_id)
    assert (ack.ok, ack.detail) == (False, "no session manager attached")


# -- unit: the manager's blockers + the motion on a fake workcell -------------------------


def _chosen(manager, cell, *, rail_delta: float = 0.2):
    """A saved profile (not the initial condition) a bit away from the current posture,
    with a rail slot for the railed arm so the carriage phase has to run."""
    q = {a: list(cell.arms[a].get_state().q[:7]) for a in ("arm0", "arm1")}
    rail_now = float(cell.arms["arm0"].get_state().q[7])
    rail_goal = rail_now + rail_delta if rail_now + rail_delta <= 0.65 else rail_now - rail_delta
    return manager.profile_store.save(
        StateProfile(
            name="shelf",
            workcell_kind="sim",
            arms={
                "arm0": ArmPosture(q=[x - 0.08 for x in q["arm0"]], rail_pos_m=rail_goal),
                "arm1": ArmPosture(q=[x + 0.1 for x in q["arm1"]]),
            },
        )
    ), rail_now, rail_goal


def test_manager_refusals_mirror_reset_to_initial(tmp_path, fake_loop):
    cell, bus, loop = fake_loop
    manager, session, rt, initial = make(tmp_path, fake_loop)
    manager.attach_fault_state(session)
    assert loop.on_goto_profile == manager.request_goto_profile
    lab = manager.profile_store.save(
        StateProfile(name="lab", workcell_kind="hardware", arms={"arm0": ArmPosture(q=[0.0] * 7)})
    )
    assert manager.request_goto_profile(lab) == (
        False,
        "profile 'lab' is for the hardware workcell",
    )
    loop._faulted.add("arm0")
    assert manager.request_goto_profile(initial) == (
        False,
        "arm0 is faulted - clear the error and resume first",
    )
    loop._faulted.clear()
    loop._recovering.add("arm0")  # lifted by releasing the inputs, not by clearing errors
    assert manager.request_goto_profile(initial) == (
        False,
        "arm0 is recovering - release every input (clutch / keys) first",
    )
    loop._recovering.clear()
    rt.open_episode_id = "ep-1"
    assert manager.request_goto_profile(initial) == (
        False,
        "an episode is still recording - save or discard it first",
    )
    # ``open_episode_id`` stays set while the episode is SAVING: say that, not "save it"
    rt.status = lambda: SimpleNamespace(state="saving")
    assert manager.request_goto_profile(initial) == (
        False,
        "an episode is still saving - wait for it to finish",
    )
    del rt.status
    del rt.open_episode_id
    loop.plans.load("arm1", [[0.0] * 7, [0.1] * 7])
    assert manager.request_goto_profile(initial) == (False, MOTION_BUSY)
    loop.plans.cancel()
    session.state = SessionState.FAULT
    assert manager.request_goto_profile(initial) == (False, "the session is fault, not running")
    session.state = SessionState.RUNNING
    manager.session = None
    assert manager.request_goto_profile(initial) == (False, "no active session")
    assert not loop.plans.active_arms


def test_goto_profile_walks_the_arms_to_the_chosen_profile_joints_then_carriage(
    tmp_path, fake_loop, monkeypatch
):
    """The whole op on a fake workcell: ``goto_profile`` over the bus -> the loop resolves
    the profile -> ``request_goto_profile`` -> ``_return_to_initial_motion`` runs TWO
    interruptible ``execute_plan`` phases (joints with the carriage held, then the
    carriage) and both arms end at the CHOSEN profile - not at the initial condition."""
    cell, bus, loop = fake_loop
    manager, session, rt, initial = make(tmp_path, fake_loop)
    manager.attach_fault_state(session)
    chosen, rail_now, rail_goal = _chosen(manager, cell)
    executed: list = []
    real = ControlLoop._op_execute_plan

    def spy(self, cmd):
        executed.append(cmd.args)
        return real(self, cmd)

    monkeypatch.setattr(ControlLoop, "_op_execute_plan", spy)

    fut = bus.commands.submit(
        Command(op="goto_profile", args={"profile_id": chosen.profile_id}, source="ws")
    )
    run_ticks(loop, cell, 1)
    ack = fut.result(timeout=1.0)
    assert (ack.ok, ack.detail) == (True, "going to profile 'shelf'")

    def arrived() -> bool:
        return (
            len(executed) == 3
            and not loop.plans.active_arms
            and np.allclose(loop._last_cmd["arm0"][:7], chosen.arms["arm0"].q, atol=1e-6)
            and abs(loop._last_cmd["arm0"][7] - rail_goal) < 1e-6
            and np.allclose(loop._last_cmd["arm1"][:7], chosen.arms["arm1"].q, atol=1e-6)
        )

    tick_until(loop, cell, arrived, 20.0)
    # 2026-09-08 evening: ONE arm per execute_plan, in the planner's order - phase 1 is
    # two submissions (arm0, then arm1); phase 2 moves arm0's carriage only: arm1's
    # MEASURED posture followed phase 1 (``run_ticks`` dispatches the loop's output to the
    # fake arms like the sender threads do, and the manager waits for measured arrival
    # since 2026-09-08), so the planner leaves it in place and it is not submitted at all
    joints_0, joints_1, carriage = executed
    assert all(c["interruptible"] is True for c in executed)
    assert [list(c["waypoints"]) for c in executed] == [["arm0"], ["arm1"], ["arm0"]]
    # phase 1 moved the joints with the carriage HELD ...
    assert all(abs(w[7] - rail_now) < 1e-9 for w in joints_0["waypoints"]["arm0"])
    assert joints_0["waypoints"]["arm0"][-1][:7] == pytest.approx(chosen.arms["arm0"].q)
    assert joints_1["waypoints"]["arm1"][-1][:7] == pytest.approx(chosen.arms["arm1"].q)
    # ... phase 2 brought the carriage to the profile's rail slot with the joints AT the
    # profile, planned from the MEASURED posture phase 1 left (= the profile's joints)
    assert carriage["waypoints"]["arm0"][0][:7] == pytest.approx(chosen.arms["arm0"].q, abs=1e-3)
    assert carriage["waypoints"]["arm0"][-1][:7] == pytest.approx(chosen.arms["arm0"].q)
    assert carriage["waypoints"]["arm0"][-1][7] == pytest.approx(rail_goal)
    # the gripper targets ride the LAST arm of the LAST phase (applied on its arrival)
    assert joints_0["gripper"] == {} and joints_1["gripper"] == {}
    assert carriage["gripper"] == {"arm0": 1.0, "arm1": 1.0}
    # and the fake arms MEASURABLY arrived (the manager's hand-over criterion)
    st = cell.states()
    assert np.allclose(st["arm0"].q[:7], chosen.arms["arm0"].q, atol=1e-3)
    assert abs(st["arm0"].q[7] - rail_goal) <= 2e-3
    assert np.allclose(st["arm1"].q[:7], chosen.arms["arm1"].q, atol=1e-3)
    # not the initial condition: that profile is 0.05 rad the OTHER way
    assert not np.allclose(loop._last_cmd["arm0"][:7], initial.arms["arm0"].q, atol=0.01)
    assert session.state is SessionState.RUNNING


def test_goto_profile_reports_a_planning_failure_it_could_not_ack(tmp_path, fake_loop):
    """Finding (2026-09-08 review): goto_profile / reset_to_initial ack "started" and used
    to stay silent when phase-1 planning failed BEFORE any ``execute_plan`` (the loop's
    ``plan_status`` never changes). The outcome now rides ``session.motion_detail`` ->
    telemetry ``session.fault_detail`` with the operator's wording."""
    from test_return_manager_units import FakeTwin

    cell, bus, loop = fake_loop
    manager, session, rt, _initial = make(tmp_path, fake_loop, twin=FakeTwin(ok=False))
    manager.attach_fault_state(session)
    chosen, _rail_now, _rail_goal = _chosen(manager, cell)
    before = {a: loop._last_cmd[a].copy() for a in ("arm0", "arm1")}
    assert manager.request_goto_profile(chosen) == (True, "going to profile 'shelf'")
    tick_until(loop, cell, lambda: manager.profile_motion_in_flight is None, 10.0)
    assert session.motion_detail == (
        "Go to profile 'shelf': the digital twin could not plan a collision-free path while "
        "moving the joints: goal_in_collision (arm0_link6 / table). The arms have not moved."
    )
    run_ticks(loop, cell, 3)
    assert not loop.plans.active_arms
    assert all(np.allclose(loop._last_cmd[a], before[a]) for a in before)


def test_goto_profile_phase_2_is_refused_when_an_arm_faults_between_the_phases(
    tmp_path, fake_loop, monkeypatch
):
    """Phase 1 (joints) arrived; a driver fault lands before phase 2 (carriage) is handed
    over: the loop refuses phase 2, the arm holds at the phase-1 posture with the
    carriage where it was, and the notice names the phase and the arm."""
    from test_return_manager_units import _spy_execute_plan

    cell, bus, loop = fake_loop
    manager, session, rt, _initial = make(tmp_path, fake_loop)
    manager.attach_fault_state(session)
    chosen, rail_now, rail_goal = _chosen(manager, cell)

    def fault_before_phase_2(loop_, n):
        if n == 3:  # phase 1 = arm0 then arm1 (sequential); the 3rd plan is the carriage
            loop_._faulted.add("arm0")  # what _on_fault_event leaves behind

    executed = _spy_execute_plan(monkeypatch, before=fault_before_phase_2)
    assert manager.request_goto_profile(chosen) == (True, "going to profile 'shelf'")
    tick_until(loop, cell, lambda: manager.profile_motion_in_flight is None, 20.0)
    assert len(executed) == 3
    assert session.motion_detail == (
        "Go to profile 'shelf': the motion was refused while moving the carriage: "
        "arm 'arm0' is faulted."
    )
    assert not loop.plans.active_arms
    assert loop._last_cmd["arm0"][:7] == pytest.approx(chosen.arms["arm0"].q, abs=1e-6)
    assert abs(loop._last_cmd["arm0"][7] - rail_now) < 1e-9  # the carriage never moved
    assert abs(rail_goal - rail_now) > 0.1


def test_skip_text_names_the_profile_under_the_goto_label(tmp_path, fake_loop):
    cell, bus, loop = fake_loop
    manager, session, rt, _initial = make(tmp_path, fake_loop)
    manager.attach_fault_state(session)
    q = {a: list(cell.arms[a].get_state().q[:7]) for a in ("arm0", "arm1")}
    here = manager.profile_store.save(
        StateProfile(name="here", workcell_kind="sim",
                     arms={a: ArmPosture(q=q[a], rail_pos_m=None) for a in q})
    )
    res = manager._return_to_initial_motion(session, here, label="goto_profile")
    assert (res.ok, res.status, res.detail) == (True, "skipped", "already at profile 'here'")
    res = manager._return_to_initial_motion(session, here)  # `R` / the exit return
    assert (res.ok, res.status, res.detail) == (True, "skipped", "already at the initial condition")
    # ... and the notice says so too (a skipped motion is an outcome the operator asked about)
    assert manager.request_goto_profile(here) == (True, "going to profile 'here'")
    tick_until(loop, cell, lambda: manager.profile_motion_in_flight is None, 5.0)
    assert session.motion_detail == "Go to profile 'here': already at profile 'here'"
    # an arrival clears it
    chosen, _rail_now, _rail_goal = _chosen(manager, cell)
    assert manager.request_goto_profile(chosen)[0] is True
    tick_until(loop, cell, lambda: manager.profile_motion_in_flight is None, 20.0)
    assert session.motion_detail == ""


def test_goto_profile_is_cancelled_by_a_movement_key_and_the_arms_hold(
    tmp_path, fake_loop
):
    from apollo_mavis_v2_core import HeldState

    cell, bus, loop = fake_loop
    manager, session, rt, _initial = make(tmp_path, fake_loop)
    manager.attach_fault_state(session)
    chosen, _rail_now, _rail_goal = _chosen(manager, cell)
    fut = bus.commands.submit(
        Command(op="goto_profile", args={"profile_id": chosen.profile_id}, source="ws")
    )
    run_ticks(loop, cell, 1)
    assert fut.result(timeout=1.0).ok
    tick_until(loop, cell, lambda: bool(loop.plans.active_arms), 10.0)
    run_ticks(loop, cell, 3)
    # one browser movement key while the plan runs: cancelled, the arm holds
    held = HeldState(held=frozenset({"KeyQ"}), seq=1, rx_mono=time.monotonic())
    session.supervisor.watchdog.on_keys(held)
    bus.held_keys.put(held)
    run_ticks(loop, cell, 1)
    assert not loop.plans.active_arms and loop.plan_cancel_reason == "movement key"
    empty = HeldState(held=frozenset(), seq=2, rx_mono=time.monotonic())
    session.supervisor.watchdog.on_keys(empty)
    bus.held_keys.put(empty)
    run_ticks(loop, cell, 2)
    held_at = loop._last_cmd["arm0"].copy()
    run_ticks(loop, cell, 20)
    assert np.allclose(loop._last_cmd["arm0"], held_at)  # ... and keeps holding
    assert not np.allclose(loop._last_cmd["arm0"][:7], chosen.arms["arm0"].q, atol=1e-3)


# -- e2e over a real sim server -------------------------------------------------------

BASE = {
    "kind": "sim",
    "arms": ["arm0"],
    "frames": {"arm0": "arm_base:arm0"},
    "sim_scene": "guardrail_env",
}


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    srv = LiveServer(make_runtime_config(tmp_path_factory.mktemp("rt"), scene="guardrail_env"))
    yield srv
    srv.stop()


@pytest.fixture(scope="module")
def api(server):
    with httpx.Client(base_url=server.http, timeout=200.0) as client:
        yield client


def wait_running(api) -> None:
    for _ in range(400):
        if api.get("/api/session").json()["state"] == "running":
            return
        time.sleep(0.05)
    raise AssertionError("session never reached running")


def max_joint_err(a, b) -> float:
    return max(abs(x - y) for x, y in zip(a[:7], b[:7], strict=False))


def test_goto_profile_over_ws_walks_to_a_saved_profile_that_is_not_the_initial_condition(
    server, api
):
    assert api.post("/api/session", json={**BASE, "mode": "teleop"}).status_code == 200
    wait_running(api)
    ctl, tele = PulsingCtl(server), Tele(server)
    try:
        ack = ctl.action("save_profile", {"name": "shelf"})  # NOT the initial condition
        assert ack["ok"], ack
        pid = ack["detail"]
        assert api.get("/api/profiles").json()[0]["is_initial_condition"] is False
        q_shelf = tele.q_full()

        # Move joints AND carriage away, then ask for the profile by id.
        ctl.hold(["KeyE", "ArrowRight"], 1.5)
        time.sleep(0.5)
        q_away = tele.q_full()
        assert max_joint_err(q_away, q_shelf) > 0.05
        assert abs(q_away[7] - q_shelf[7]) > 0.02
        ack = ctl.action("goto_profile", {"profile_id": pid})
        assert ack["ok"] is True and ack["detail"] == "going to profile 'shelf'", ack
        deadline = time.monotonic() + 60.0
        while True:
            q_now = tele.q_full()
            if max_joint_err(q_now, q_shelf) <= 0.02 and abs(q_now[7] - q_shelf[7]) <= 2e-3:
                break
            assert time.monotonic() < deadline, "goto_profile never brought the arm back"
            time.sleep(0.05)
        time.sleep(0.5)
        assert not server.runtime.manager.session.loop.plans.active_arms

        # Refusals: an unknown id, a profile of the other workcell kind, no id at all
        # (core's validate_action_args refuses it before the loop sees it).
        ack = ctl.action("goto_profile", {"profile_id": "deadbeef"})
        assert ack["ok"] is False and ack["detail"] == "unknown profile 'deadbeef'", ack
        lab = server.runtime.manager.profile_store.save(
            StateProfile(
                name="lab", workcell_kind="hardware", arms={"arm0": ArmPosture(q=[0.0] * 7)}
            )
        )
        ack = ctl.action("goto_profile", {"profile_id": lab.profile_id})
        assert ack["ok"] is False and ack["detail"] == "profile 'lab' is for the hardware workcell"
        ack = ctl.action("goto_profile", {})
        assert ack["ok"] is False and ack["detail"].startswith("invalid args"), ack
        q_still = tele.q_full()
        time.sleep(0.4)
        assert max_joint_err(tele.q_full(), q_still) < 5e-3  # nothing moved on a refusal
    finally:
        ctl.close()
        tele.close()
        api.delete("/api/session")


def test_goto_profile_is_refused_while_an_episode_records(server, api):
    spec = {**BASE, "mode": "collect", "task": "goto e2e", "dataset": "goto_key",
            "return_to_start": False}
    assert api.post("/api/session", json=spec).status_code == 200
    wait_running(api)
    ctl, tele = PulsingCtl(server), Tele(server)
    try:
        pid = ctl.action("save_profile", {"name": "shelf2"})["detail"]
        assert ctl.action("episode_new")["ok"]
        time.sleep(0.8)  # let the encoder open before anything else
        q_rec = tele.q_full()
        ack = ctl.action("goto_profile", {"profile_id": pid})
        assert ack["ok"] is False and "recording" in ack["detail"], ack
        assert max_joint_err(tele.q_full(), q_rec) < 5e-3
        assert ctl.action("episode_discard")["ok"]
    finally:
        ctl.close()
        tele.close()
        api.delete("/api/session")
