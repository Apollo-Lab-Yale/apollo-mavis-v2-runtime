"""ControlLoop unit behaviour over FakeWorkcell: jog/goto, switch_arm, acks."""

from __future__ import annotations

import numpy as np
from apollo_xarm7_core import Command, HeldState, PlanResult
from conftest import run_ticks


def submit(bus, op, **args):
    return bus.commands.submit(Command(op=op, args=args))


def test_jog_slew_limited(fake_loop):
    cell, bus, loop = fake_loop
    fut = submit(bus, "joint_target", arm_id="arm0",
                 positions=[0.1] * 7 + [0.05], mode="jog")
    t = run_ticks(loop, cell, 1)
    assert fut.result(0).ok
    # After one tick: exactly one slew step (0.02 rad joints, 0.002 m rail).
    q = loop._last_cmd["arm0"]
    assert np.allclose(q[:7], 0.02)
    assert abs(q[7] - 0.002) < 1e-12
    run_ticks(loop, cell, 30, t)
    assert np.allclose(loop._last_cmd["arm0"][:7], 0.1)
    assert abs(loop._last_cmd["arm0"][7] - 0.05) < 1e-12


def test_jog_above_goto_threshold_nacked(fake_loop):
    cell, bus, loop = fake_loop
    fut = submit(bus, "joint_target", arm_id="arm0",
                 positions=[0.2] * 7 + [0.0], mode="jog")
    run_ticks(loop, cell, 1)
    res = fut.result(0)
    assert not res.ok and "goto" in res.detail


def test_jog_nacked_while_recording(fake_loop):
    cell, bus, loop = fake_loop
    loop.episode_state = "recording"
    fut = submit(bus, "joint_target", arm_id="arm0",
                 positions=[0.05] * 7 + [0.0], mode="jog")
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
    assert loop.plans.active("arm0")
    bus.held_keys.put(HeldState(held=frozenset({"KeyW"}), seq=1, rx_mono=t))
    loop.supervisor.watchdog.on_keys(HeldState(frozenset({"KeyW"}), 1, t))
    t = run_ticks(loop, cell, 1, t)
    assert not loop.plans.active("arm0")
    assert loop._plan_status == "cancelled"
    assert not np.allclose(loop._last_cmd["arm0"], goal)  # stopped short
