"""Interruptible-plan rules on the ControlLoop (04-runtime §10.5; 2026-09-07 review):
cancel on a movement code from ANY source regardless of the deadman scale, on a
jog, on an arm switch, on a device-held code / clutch, on the browser's disconnect;
a start_from plan is NOT cancelled by jog / arm switch; the deferred gripper target;
the ``cancel_plan`` op; the process-stall hold + deadman credit; a faulted arm
refuses ``execute_plan``."""

from __future__ import annotations

import numpy as np
import pytest
from apollo_mavis_v2_core import Command, HeldState, ProfileStore
from apollo_mavis_v2_core.testing import FakeArm
from conftest import run_ticks
from fakes import EventFakeWorkcell

from apollo_mavis_v2_runtime.bus import RuntimeBus
from apollo_mavis_v2_runtime.config import ControlConfig
from apollo_mavis_v2_runtime.control.loop import ControlLoop
from apollo_mavis_v2_runtime.safety.gate import NullGate
from apollo_mavis_v2_runtime.safety.supervisor import SafetySupervisor
from apollo_mavis_v2_runtime.safety.watchdog import InputWatchdog, WatchdogState

GOAL = [0.3] * 7 + [0.2]


class GripSpy:
    def __init__(self) -> None:
        self.puts: list[float] = []

    def put_gripper(self, frac: float) -> None:
        self.puts.append(frac)

    def pause(self) -> None:
        pass

    def resume(self) -> None:
        pass


def make_loop(tmp_path):
    cell = EventFakeWorkcell({"arm0": FakeArm("arm0", has_rail=True), "arm1": FakeArm("arm1")},
                             kind="sim")
    cell.start()
    bus = RuntimeBus()
    loop = ControlLoop(cell, ControlConfig(), bus, SafetySupervisor(NullGate(), InputWatchdog()),
                       ["arm0", "arm1"], profile_store=ProfileStore(tmp_path / "profiles"),
                       workcell_kind="sim", gripper_arms=["arm0", "arm1"])
    return cell, bus, loop


def execute(bus, loop, cell, *, interruptible: bool, gripper=None, arms=("arm0",), waypoints=None):
    wps = waypoints or {a: [list(GOAL)] for a in arms}
    fut = bus.commands.submit(Command(
        op="execute_plan",
        args={"waypoints": wps, "gripper": gripper or {}, "interruptible": interruptible},
        source="internal",
    ))
    t = run_ticks(loop, cell, 1)
    assert fut.result(0).ok
    assert loop.plans.active_arms
    return t


def hold_ws(loop, bus, cell, t, codes: set[str], ticks: int = 3, fresh: bool = True):
    for _ in range(ticks):
        t += loop.dt
        rx = t if fresh else t - 10.0
        hs = HeldState(held=frozenset(codes), seq=loop.tick_count + 1, rx_mono=rx)
        loop.supervisor.watchdog.on_keys(hs)
        bus.held_keys.put(hs)
        loop.run_tick(t)
        cell.step(loop.dt)
    return t


def test_movement_key_cancels_any_plan_and_records_the_reason(tmp_path):
    cell, bus, loop = make_loop(tmp_path)
    t = execute(bus, loop, cell, interruptible=True)
    hold_ws(loop, bus, cell, t, {"KeyW"})
    assert not loop.plans.active_arms and loop.plan_cancel_reason == "movement key"


def test_key_held_under_a_latched_deadman_still_cancels_an_interruptible_plan(tmp_path):
    """The reviewers' probe: the save's GIL stall trips the deadman, the operator's
    held key then had scale 0 and did NOT cancel the return. Presence must win."""
    cell, bus, loop = make_loop(tmp_path)
    t = execute(bus, loop, cell, interruptible=True)
    wd = loop.supervisor.watchdog
    wd.on_disconnect()  # AWAIT_EMPTY: every WS key now has scale 0
    assert wd.state is WatchdogState.AWAIT_EMPTY
    t += loop.dt
    hs = HeldState(held=frozenset({"KeyE"}), seq=1, rx_mono=t)
    wd.on_keys(hs)
    bus.held_keys.put(hs)
    loop.run_tick(t)
    assert wd.scale(t) == 0.0  # the key drives nothing ...
    assert not loop.plans.active_arms and loop.plan_cancel_reason == "movement key"  # ... but stops
    # a NON-interruptible plan (start_from) keeps the old rule: scale 0 -> no cancel
    cell, bus, loop = make_loop(tmp_path)
    t = execute(bus, loop, cell, interruptible=False)
    wd = loop.supervisor.watchdog
    wd.on_disconnect()
    t += loop.dt
    hs = HeldState(held=frozenset({"KeyE"}), seq=1, rx_mono=t)
    wd.on_keys(hs)
    bus.held_keys.put(hs)
    loop.run_tick(t)
    assert loop.plans.active_arms


def test_jog_and_arm_switch_cancel_only_interruptible_plans(tmp_path):
    cell, bus, loop = make_loop(tmp_path)
    t = execute(bus, loop, cell, interruptible=True)
    fut = bus.commands.submit(Command(op="joint_target", args={
        "arm_id": "arm0", "positions": [0.0] * 8, "mode": "jog"}))
    t = run_ticks(loop, cell, 1, t)
    assert fut.result(0).ok and fut.result(0).detail == "jog"  # the panel wins
    assert not loop.plans.active_arms and loop.plan_cancel_reason == "jog"

    cell, bus, loop = make_loop(tmp_path)
    t = execute(bus, loop, cell, interruptible=True)
    bus.commands.submit(Command(op="switch_arm", args={}))
    t = run_ticks(loop, cell, 1, t)
    assert not loop.plans.active_arms and loop.plan_cancel_reason == "arm switch"

    # start_from (not interruptible): jog is refused, the switch does not cancel
    cell, bus, loop = make_loop(tmp_path)
    t = execute(bus, loop, cell, interruptible=False)
    fut = bus.commands.submit(Command(op="joint_target", args={
        "arm_id": "arm0", "positions": [0.0] * 8, "mode": "jog"}))
    t = run_ticks(loop, cell, 1, t)
    assert not fut.result(0).ok and fut.result(0).detail == "plan executing"
    bus.commands.submit(Command(op="switch_arm", args={}))
    t = run_ticks(loop, cell, 1, t)
    assert loop.plans.active_arms and loop.plan_cancel_reason is None


def test_device_held_clutch_and_browser_disconnect_cancel(tmp_path):
    from test_tracker_teleop import CLUTCH, Rig

    rig = Rig()
    loop, bus = rig.loop, rig.bus
    rig.tick()
    fut = bus.commands.submit(Command(
        op="execute_plan", args={"waypoints": {"arm0": [list(GOAL)]}, "interruptible": True},
        source="internal",
    ))
    rig.tick()
    assert fut.result(0).ok and loop.plans.active_arms
    rig.sample(np.array([0.1, 0.2, 1.1]))
    rig.device(CLUTCH)  # the controller's trigger: a device-held movement code (KeyC)
    rig.tick()
    assert CLUTCH in loop.sources.held
    assert not loop.plans.active_arms and loop.plan_cancel_reason == "movement key"

    cell, bus, loop = make_loop(tmp_path)
    t = execute(bus, loop, cell, interruptible=True)
    loop.note_controller_disconnect()  # Runtime.on_controller_disconnect -> loop
    run_ticks(loop, cell, 1, t)
    assert not loop.plans.active_arms and loop.plan_cancel_reason == "browser disconnected"
    # not interruptible: a disconnect does not cancel start_from
    cell, bus, loop = make_loop(tmp_path)
    t = execute(bus, loop, cell, interruptible=False)
    loop.note_controller_disconnect()
    run_ticks(loop, cell, 1, t)
    assert loop.plans.active_arms


def test_cancel_plan_op_and_deferred_gripper(tmp_path):
    cell, bus, loop = make_loop(tmp_path)
    spy = GripSpy()
    loop._senders["arm0"] = spy
    run_ticks(loop, cell, 1)  # the first tick re-seeds _grip_frac from the measured gripper
    loop._grip_frac["arm0"] = 0.2
    t = execute(bus, loop, cell, interruptible=True, gripper={"arm0": 0.9})
    assert spy.puts == [] and loop._grip_frac["arm0"] == 0.2  # NOT applied at load
    fut = bus.commands.submit(Command(op="cancel_plan", args={"reason": "return-to-start: test"},
                                      source="internal"))
    t = run_ticks(loop, cell, 1, t)
    assert fut.result(0).ok and fut.result(0).detail == "cancelled"
    assert not loop.plans.active_arms and loop.plan_cancel_reason == "return-to-start: test"
    assert spy.puts == [] and loop._grip_frac["arm0"] == 0.2  # a cancelled return: untouched
    fut = bus.commands.submit(Command(op="cancel_plan", args={}, source="internal"))
    run_ticks(loop, cell, 1, t)
    assert fut.result(0).ok and fut.result(0).detail == "no plan"
    # arrival applies it
    cell, bus, loop = make_loop(tmp_path)
    spy = GripSpy()
    loop._senders["arm0"] = spy
    run_ticks(loop, cell, 1)
    near = [float(x) for x in loop._last_cmd["arm0"]]
    near[0] += 0.1  # 5 slew ticks away
    t = execute(bus, loop, cell, interruptible=True, gripper={"arm0": 0.9},
                waypoints={"arm0": [near]})
    run_ticks(loop, cell, 10, t)
    assert not loop.plans.active_arms and spy.puts == [0.9] and loop._grip_frac["arm0"] == 0.9
    # a start_from plan (not interruptible) applies it at load like before
    cell, bus, loop = make_loop(tmp_path)
    spy = GripSpy()
    loop._senders["arm0"] = spy
    execute(bus, loop, cell, interruptible=False, gripper={"arm0": 0.4})
    assert spy.puts == [0.4]


def test_a_second_execute_plan_during_an_active_one_is_refused(tmp_path):
    """2026-09-08 review: ``PlanExecutor.load`` silently replaces the waypoints, and the
    manager-side blockers run before a worker plans, so two profile-motion workers could
    both pass them; the loop is the last line and nacks "plan executing"."""
    cell, bus, loop = make_loop(tmp_path)
    t = execute(bus, loop, cell, interruptible=True)
    running = [w.copy() for w in loop.plans._waypoints["arm0"]]
    fut = bus.commands.submit(Command(
        op="execute_plan",
        args={"waypoints": {"arm1": [[0.0] * 7]}, "gripper": {"arm0": 0.1}},
        source="internal",
    ))
    t = run_ticks(loop, cell, 1, t)
    ack = fut.result(0)
    assert (ack.ok, ack.detail) == (False, "plan executing")
    assert loop.plans.active_arms == ["arm0"]  # the running plan is untouched ...
    assert all(
        np.array_equal(a, b) for a, b in zip(loop.plans._waypoints["arm0"], running, strict=True)
    )
    assert loop._plan_interruptible is True  # ... and still interruptible
    assert "arm0" not in loop._grip_frac or loop._grip_frac.get("arm0") != 0.1


def test_execute_plan_is_refused_for_a_faulted_arm(tmp_path):
    cell, bus, loop = make_loop(tmp_path)
    t = run_ticks(loop, cell, 1)
    cell.fault("arm0", 31, source="monitor")
    t = run_ticks(loop, cell, 1, t)
    fut = bus.commands.submit(Command(op="execute_plan", args={"waypoints": {"arm0": [GOAL]}},
                                      source="internal"))
    run_ticks(loop, cell, 1, t)
    assert not fut.result(0).ok and "faulted" in fut.result(0).detail
    assert not loop.plans.active_arms


def test_process_stall_holds_the_arms_and_credits_the_deadman(tmp_path):
    """A tick gap longer than the deadman (the encoder's GIL stall) must not step the
    arms (no catch-up jump) and must not latch a browser that was fresh when the
    stall began."""
    cell, bus, loop = make_loop(tmp_path)
    t = 0.0
    for _ in range(3):
        t += loop.dt
        hs = HeldState(held=frozenset({"KeyW"}), seq=loop.tick_count + 1, rx_mono=t)
        loop.supervisor.watchdog.on_keys(hs)
        bus.held_keys.put(hs)
        loop.run_tick(t)
        cell.step(loop.dt)
    before = loop._last_cmd["arm0"].copy()
    stalls = loop.process_stalls
    t += 0.35  # the process froze for 350 ms; the browser kept sending, we could not read
    loop.run_tick(t)
    assert loop.process_stalls == stalls + 1
    assert np.allclose(loop._last_cmd["arm0"], before)  # held: no jump
    assert loop.supervisor.watchdog.state is WatchdogState.OK  # not latched
    assert loop.supervisor.watchdog.scale(t) == 1.0
    # ... but a browser that was ALREADY stale at stall start is not forgiven: the loop
    # kept ticking (10 ms gaps) while no key arrived = a genuine deadman trip, then a stall
    cell, bus, loop = make_loop(tmp_path)
    hs = HeldState(held=frozenset({"KeyW"}), seq=1, rx_mono=0.01)
    loop.supervisor.watchdog.on_keys(hs)
    bus.held_keys.put(hs)
    t = 0.01
    for _ in range(35):  # 0.35 s of silence at 100 Hz -> TRIPPED -> AWAIT_EMPTY
        t += 0.01
        loop.run_tick(t)
        cell.step(loop.dt)
    assert loop.supervisor.watchdog.state is WatchdogState.AWAIT_EMPTY
    loop.run_tick(t + 0.5)  # then a stall
    assert loop.supervisor.watchdog.state is WatchdogState.AWAIT_EMPTY  # not forgiven


def test_on_process_stall_watchdog_rule():
    wd = InputWatchdog(0.2, 0.1)
    wd.on_keys(HeldState(held=frozenset({"KeyW"}), seq=1, rx_mono=1.0))
    wd.on_process_stall(now=1.5, stall_started=1.1)  # fresh at stall start -> credited
    assert wd.scale(1.5) == 1.0
    wd2 = InputWatchdog(0.2, 0.1)
    wd2.on_keys(HeldState(held=frozenset({"KeyW"}), seq=1, rx_mono=1.0))
    wd2.on_process_stall(now=2.0, stall_started=1.5)  # already 0.5 s stale at stall start
    assert wd2.scale(2.0) < 1.0
    with pytest.raises(AssertionError):
        assert wd2.state is WatchdogState.OK


def test_driver_fault_on_any_arm_cancels_an_interruptible_plan(tmp_path):
    """A return-to-start (interruptible) stops on EVERY arm's fault, also a fault of the
    arm the plan is not moving; a start_from plan (not interruptible) keeps its arm
    moving through a sibling's fault, as before. Plans carry ONE arm since 2026-09-08
    (``test_execute_plan_carries_one_arm``), so the sibling is the un-planned arm."""
    cell, bus, loop = make_loop(tmp_path)
    t = execute(bus, loop, cell, interruptible=True)
    assert loop.plans.active_arms == ["arm0"]
    cell.fault("arm1", 31, source="monitor")
    run_ticks(loop, cell, 1, t)
    assert loop.plans.active_arms == [] and loop.plan_cancel_reason == "driver fault"
    # a start_from plan (not interruptible) keeps the sibling moving, as before
    cell, bus, loop = make_loop(tmp_path)
    t = execute(bus, loop, cell, interruptible=False)
    cell.fault("arm1", 31, source="monitor")
    run_ticks(loop, cell, 1, t)
    assert loop.plans.active_arms == ["arm0"]
    # ... and a fault of the PLANNED arm stops its own plan either way
    cell.fault("arm0", 31, source="monitor")
    run_ticks(loop, cell, 1, t)
    assert loop.plans.active_arms == [] and loop.plan_cancel_reason == "driver fault"


def test_execute_plan_carries_one_arm(tmp_path):
    """Defence in depth (2026-09-08 review): the manager submits one arm per plan since
    the 23:16:52 incident; the loop refuses a command carrying two, so the incident
    class can not recur through this op even if a caller regresses."""
    cell, bus, loop = make_loop(tmp_path)
    fut = bus.commands.submit(Command(
        op="execute_plan",
        args={"waypoints": {"arm0": [list(GOAL)], "arm1": [[0.3] * 7]}, "gripper": {},
              "interruptible": True},
        source="internal",
    ))
    run_ticks(loop, cell, 1)
    ack = fut.result(0)
    assert not ack.ok and ack.detail.startswith("one arm per plan")
    assert not loop.plans.active_arms and loop._plan_state == {}
