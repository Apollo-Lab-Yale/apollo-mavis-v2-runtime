"""Control-loop consumption of driver events (phase-09b; 04-runtime §15): a
FaultEvent stops ONLY that arm (sender paused, held, plan/jog dropped, siblings
keep moving, ``fault_detail`` set, state ``fault``); ReseedEvent / RecoveredEvent
re-seed the targets from the MEASURED position and hold the arm in
``recovering`` until every live input is released (device clutch re-grip; WS
codes stay behind the watchdog's AWAIT_EMPTY latch); StudioConflictWarning is a
lingering telemetry warning that stops nothing. Scriptable
``tests/fakes.EventFakeWorkcell``, no hardware."""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import pytest
from apollo_mavis_v2_core import Command, GripperCommand, HeldState, LatestSlot, ProfileStore
from apollo_mavis_v2_core.testing import FakeArm
from conftest import run_ticks
from fakes import (
    EventFakeWorkcell,
    FaultEvent,
    RecoveredEvent,
    ReseedEvent,
    StudioConflictWarning,
)
from test_tracker_teleop import CLUTCH, Rig

from apollo_mavis_v2_runtime.bus import RuntimeBus
from apollo_mavis_v2_runtime.config import ControlConfig
from apollo_mavis_v2_runtime.control.arm_sender import ArmSender
from apollo_mavis_v2_runtime.control.loop import (
    FAULT_EVENT_NAMES,
    STUDIO_WARNING_LINGER_S,
    ControlLoop,
    controller_error_title,
)
from apollo_mavis_v2_runtime.safety.gate import NullGate
from apollo_mavis_v2_runtime.safety.supervisor import SafetySupervisor
from apollo_mavis_v2_runtime.safety.watchdog import InputWatchdog, WatchdogState

pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")


def make_loop(tmp_path, arms=("arm0", "arm1")):
    cell = EventFakeWorkcell(
        {"arm0": FakeArm("arm0", has_rail=True), "arm1": FakeArm("arm1")}, kind="sim"
    )
    cell.start()
    bus = RuntimeBus()
    loop = ControlLoop(
        cell,
        ControlConfig(),
        bus,
        SafetySupervisor(NullGate(), InputWatchdog()),
        list(arms),
        profile_store=ProfileStore(tmp_path / "profiles"),
        workcell_kind="sim",
    )
    states: list[str | None] = []
    loop.on_fault_state = states.append
    return cell, bus, loop, states


def jog(bus, arm_id: str, value: float, dof: int):
    return bus.commands.submit(
        Command(
            op="joint_target", args={"arm_id": arm_id, "positions": [value] * dof, "mode": "jog"}
        )
    )


def extra(bus) -> dict:
    return bus.snapshot.get()[0].session_extra


# -- FaultEvent ------------------------------------------------------------------------------
def test_fault_event_stops_only_that_arm_and_reports_fault(tmp_path):
    cell, bus, loop, states = make_loop(tmp_path)
    jog(bus, "arm0", 0.1, 8)
    jog(bus, "arm1", 0.1, 7)
    t = run_ticks(loop, cell, 3)
    assert loop.jog.active("arm0") and loop.jog.active("arm1")
    q0_before = loop._last_cmd["arm0"].copy()
    slot0_put = bus.arm_slot("arm0").get()[1]

    ev = cell.fault("arm0", 24)  # controller error 24 on arm0 (arm1 untouched)
    t = run_ticks(loop, cell, 1, t)
    assert loop.fault_state == "fault" and loop.faulted_arms == {"arm0"}
    assert states == ["fault"]
    faults = extra(bus)["arm_faults"]
    assert faults["arm0"].startswith("controller error 24")
    assert faults["arm0"] == controller_error_title(ev.error_code)
    assert extra(bus)["arm_recovering"] == []
    assert not loop.jog.active("arm0") and loop.jog.active("arm1")  # the jog on arm0 is dropped
    t = run_ticks(loop, cell, 5, t)
    assert np.allclose(loop._last_cmd["arm0"], q0_before)  # held
    assert bus.arm_slot("arm0").get()[1] == slot0_put  # nothing published while faulted
    assert np.allclose(loop._last_cmd["arm1"][:7], 0.1)  # sibling finished its jog
    assert bus.arm_slot("arm1").get()[1] > slot0_put
    assert loop.fault_events == 1 and states == ["fault"]  # edge-triggered callback


def test_fault_drops_a_running_plan_and_cancels_its_status(tmp_path):
    cell, bus, loop, _ = make_loop(tmp_path)
    fut = bus.commands.submit(
        Command(
            op="execute_plan",
            args={"waypoints": {"arm0": [[0.05] * 8, [0.1] * 8]}},
            source="internal",
        )
    )
    t = run_ticks(loop, cell, 1)
    assert fut.result(0).ok and loop.plans.active("arm0")
    cell.fault("arm0", 31, source="monitor")
    run_ticks(loop, cell, 1, t)
    assert not loop.plans.active("arm0") and "arm0" not in loop._plan_state
    assert extra(bus)["plan_status"] == "cancelled"


def test_latch_and_user_fault_texts(tmp_path):
    cell, bus, loop, _ = make_loop(tmp_path)
    cell.queue(
        FaultEvent(
            "arm0",
            "latch",
            0,
            error_code=1,
            detail="motion_enable failed (release the physical e-stop?)",
        )
    )
    run_ticks(loop, cell, 1)
    text = extra(bus)["arm_faults"]["arm0"]
    assert text.startswith("controller error 1") and text.endswith(
        "- motion_enable failed (release the physical e-stop?)"
    )
    cell.queue(FaultEvent("arm1", "user", 0, error_code=0))  # recovery from STREAMING
    run_ticks(loop, cell, 1)
    assert extra(bus)["arm_faults"]["arm1"] == (
        "operator-requested recovery (re-seed from the measured position)"
    )
    cell.queue(FaultEvent("arm1", "report", -1))  # link loss: no code, no detail
    run_ticks(loop, cell, 1)
    assert extra(bus)["arm_faults"]["arm1"] == "driver fault (source report, code -1)"


def test_events_for_arms_outside_the_session_are_ignored(tmp_path):
    cell, bus, loop, states = make_loop(tmp_path, arms=("arm0",))
    cell.queue(FaultEvent("arm1", "monitor", 24, error_code=24), ReseedEvent("nope", (0.0,) * 7))
    run_ticks(loop, cell, 1)
    assert loop.fault_state is None and states == [] and extra(bus)["arm_faults"] == {}


# -- Reseed / Recovered -> RECOVERING -> RUNNING -------------------------------------------------
def test_recovery_reseeds_from_measured_then_runs_once_inputs_are_up(tmp_path):
    cell, bus, loop, states = make_loop(tmp_path)
    jog(bus, "arm0", 0.1, 8)
    t = run_ticks(loop, cell, 3)
    cell.fault("arm0", 24)
    t = run_ticks(loop, cell, 1, t)
    assert states == ["fault"]
    # The arm moved while faulted (the driver's own hold / a physical nudge).
    arm0 = cell.arms["arm0"]
    arm0.command_joints(np.array([0.3] * 7 + [0.1]))
    for _ in range(120):
        arm0.step(0.01)
    measured = arm0.get_state().q.copy()
    assert not np.allclose(loop._last_cmd["arm0"], measured)

    cell.request_recovery("arm0")  # FaultEvent(user) -> ReseedEvent -> RecoveredEvent
    t = run_ticks(loop, cell, 1, t)
    assert cell.recovery_result("arm0").ok and arm0.get_state().error_code == 0
    assert np.allclose(loop._last_cmd["arm0"], measured)  # re-seeded from MEASURED
    assert loop.fault_state == "recovering" and loop.recovering_arms == {"arm0"}
    assert loop.recoveries == 1 and states == ["fault", "recovering"]
    assert loop.supervisor.watchdog.state is WatchdogState.AWAIT_EMPTY  # §8 re-seed rule
    assert extra(bus)["arm_recovering"] == ["arm0"]
    assert extra(bus)["arm_faults"]["arm0"].startswith("controller error 24")  # kept

    t = run_ticks(loop, cell, 1, t)  # nothing held by any live source -> RUNNING
    assert loop.fault_state is None and states == ["fault", "recovering", None]
    assert extra(bus)["arm_faults"] == {} and extra(bus)["arm_recovering"] == []
    assert np.allclose(loop._last_cmd["arm0"], measured)  # still holding, no jump


def test_recovering_waits_for_the_device_clutch_to_be_released_and_regripped():
    rig = Rig()
    rig.cell = EventFakeWorkcell(
        {"arm0": FakeArm("arm0", has_rail=True), "arm1": FakeArm("arm1")},
        kind="sim",
        clock=lambda: rig.t,
    )
    rig.cell.start()
    rig.loop.workcell = rig.cell
    rig.loop._seed_from_measured()
    states: list[str | None] = []
    rig.loop.on_fault_state = states.append
    rig.device(CLUTCH)
    rig.sample([0.0, 0.0, 0.0])
    rig.tick()
    rig.sample([0.01, 0.0, 0.0])  # hand moves +x -> arm follows
    rig.tick()
    assert rig.cmd()[0] > 0.0

    rig.cell.fault("arm0", 31)  # collision while clutched
    rig.tick()
    assert states == ["fault"] and rig.loop._clutch_arm is None
    rig.cell.request_recovery("arm0")
    rig.tick()
    assert rig.loop.fault_state == "recovering" and states[-1] == "recovering"
    held_cmd = rig.cmd()
    rig.sample([0.05, 0.0, 0.0])  # the hand keeps moving with the clutch still held ...
    rig.tick(5)
    assert rig.loop.fault_state == "recovering"  # ... the arm does not follow
    assert np.allclose(rig.cmd(), held_cmd)

    rig.device()  # clutch released
    rig.tick()
    assert rig.loop.fault_state is None and states[-1] is None
    rig.device(CLUTCH)  # re-grip: true rising edge -> zero delta on engage
    rig.tick()
    assert np.allclose(rig.cmd()[:3], rig.cell.arms["arm0"].get_state().q[:3], atol=1e-9)
    rig.sample([0.06, 0.0, 0.0])
    rig.tick()
    assert rig.cmd()[0] > held_cmd[0]  # following again


def test_ws_codes_stay_behind_the_watchdog_latch_after_recovery(tmp_path):
    cell, bus, loop, states = make_loop(tmp_path)
    wd = loop.supervisor.watchdog
    t = 0.0

    def keys(seq: int, held: list[str]) -> None:
        st = HeldState(frozenset(held), seq, t)
        wd.on_keys(st)
        bus.held_keys.put(st)

    keys(1, ["KeyW"])
    t = run_ticks(loop, cell, 1, t)
    cell.fault("arm0", 24)
    t = run_ticks(loop, cell, 1, t)
    cell.request_recovery("arm0")
    keys(2, ["KeyW"])  # the operator still holds W when the arm recovers
    t = run_ticks(loop, cell, 1, t)
    assert loop.fault_state == "recovering" and wd.state is WatchdogState.AWAIT_EMPTY
    keys(3, ["KeyW"])
    t = run_ticks(loop, cell, 1, t)
    # The held W is latched by the watchdog (scale 0 -> not a live input), so the
    # arm leaves RECOVERING, but no WS code moves anything until an EMPTY KeysMsg.
    assert loop.fault_state is None and wd.scale(t) == 0.0 and wd.needs_all_up
    keys(4, [])
    assert wd.state is WatchdogState.OK
    assert states == ["fault", "recovering", None]


# -- StudioConflictWarning ---------------------------------------------------------------
def test_studio_conflict_warning_lingers_without_stopping_the_arm(tmp_path):
    cell, bus, loop, states = make_loop(tmp_path)
    jog(bus, "arm0", 0.1, 8)
    t = run_ticks(loop, cell, 2)
    cell.queue(StudioConflictWarning("arm0", mode=0, state=0))
    t = run_ticks(loop, cell, 1, t)
    assert extra(bus)["arm_faults"] == {"arm0": "warning: close UFACTORY Studio live control"}
    assert loop.fault_state is None and states == [] and loop.jog.active("arm0")
    run_ticks(loop, cell, 1, t + STUDIO_WARNING_LINGER_S)
    assert extra(bus)["arm_faults"] == {}
    # A real fault replaces a pending warning.
    cell.queue(
        StudioConflictWarning("arm1", mode=0, state=0),
        FaultEvent("arm1", "monitor", 24, error_code=24),
    )
    run_ticks(loop, cell, 1, t + STUDIO_WARNING_LINGER_S + 0.02)
    assert extra(bus)["arm_faults"]["arm1"].startswith("controller error 24")


# -- gripper keys on a faulted / recovering arm ---------------------------------------------------
class _RecordingSender:
    """Stand-in for the ArmSender the loop would own after start(): records the
    gripper targets it is handed plus pause/resume edges."""

    def __init__(self) -> None:
        self.grips: list[float] = []
        self.calls: list[str] = []

    def put_gripper(self, open_frac: float) -> None:
        self.grips.append(float(open_frac))

    def pause(self) -> None:
        self.calls.append("pause")

    def resume(self) -> None:
        self.calls.append("resume")


def test_gripper_keys_do_nothing_while_the_arm_is_faulted_or_recovering():
    """A gripper key held through FAULT / RECOVERING integrates nothing and hands
    the sender nothing (no pre-fault target is replayed once the driver streams
    again); the re-seed re-syncs the gripper target from the MEASURED opening."""
    rig = Rig()
    rig.cell = EventFakeWorkcell(
        {"arm0": FakeArm("arm0", has_rail=True), "arm1": FakeArm("arm1")},
        kind="sim",
        clock=lambda: rig.t,
    )
    rig.cell.start()
    rig.loop.workcell = rig.cell
    rig.loop._seed_from_measured()
    sender = _RecordingSender()
    rig.loop._senders["arm0"] = sender  # type: ignore[assignment]
    assert rig.loop.active_arm == "arm0" and rig.loop._grip_frac["arm0"] == 1.0

    rig.device("KeyF")  # device-held gripper close (KEYMAP: KeyF -> gripper_close)
    rig.sample([0.0, 0.0, 0.0])
    rig.tick(10)
    frac_running = rig.loop._grip_frac["arm0"]
    assert frac_running < 1.0 and sender.grips  # integrating + handed to the sender
    n_grips = len(sender.grips)

    rig.cell.fault("arm0", 31)  # collision while the key is still held
    rig.tick()
    assert rig.loop.fault_state == "fault" and sender.calls == ["pause"]
    rig.tick(60)  # the operator keeps the key pressed through the whole fault
    assert rig.loop._grip_frac["arm0"] == frac_running  # nothing integrated
    assert len(sender.grips) == n_grips  # nothing queued for the sender

    # The driver's hold / a manual nudge moved the gripper meanwhile.
    rig.cell.arms["arm0"].command_gripper(GripperCommand(open_frac=0.5))
    rig.cell.request_recovery("arm0")  # FaultEvent(user) -> ReseedEvent -> RecoveredEvent
    rig.tick()
    # FaultEvent(user) pauses once more, ReseedEvent + RecoveredEvent both resume
    assert rig.loop.fault_state == "recovering" and sender.calls[-1] == "resume"
    assert rig.loop._grip_frac["arm0"] == 0.5  # re-synced from MEASURED, not the stale integral
    rig.tick(30)  # key still held -> stays RECOVERING, gripper still untouched
    assert rig.loop.fault_state == "recovering"
    assert rig.loop._grip_frac["arm0"] == 0.5 and len(sender.grips) == n_grips

    rig.device()  # release everything -> RUNNING
    rig.tick()
    assert rig.loop.fault_state is None
    rig.device("KeyF")  # a fresh press integrates again, from the measured opening
    rig.tick(10)
    assert rig.loop._grip_frac["arm0"] < 0.5
    # the sender only ever saw values integrated from the measured 0.5 (sent every 10 ticks)
    assert len(sender.grips) > n_grips and rig.loop._grip_frac["arm0"] <= sender.grips[-1] < 0.5


# -- ArmSender gating --------------------------------------------------------------------------
def _publish_until(slot: LatestSlot, value: float, sender: ArmSender, sent: int) -> None:
    """Re-publish ``value`` every 5 ms (the loop re-publishes its held target every
    tick) until the sender dispatched ``sent`` commands. A single put can fall
    into the gap between two ``wait_fresh`` iterations and is then never
    delivered - by design (LatestSlot delivers puts AFTER the wait starts)."""
    deadline = time.monotonic() + 2.0
    while sender.sent_count < sent and time.monotonic() < deadline:
        slot.put(np.full(7, value))
        time.sleep(0.005)


def test_arm_sender_pause_drains_the_slot_without_dispatching():
    arm = FakeArm("arm0")
    slot: LatestSlot = LatestSlot()
    sender = ArmSender("arm0", arm, slot)
    sender.start()
    try:
        _publish_until(slot, 0.1, sender, 1)
        assert sender.sent_count >= 1 and np.allclose(arm._target, 0.1)
        sender.pause()
        assert sender.paused
        sent_at_pause = sender.sent_count
        for _ in range(20):
            slot.put(np.full(7, 0.2))
            time.sleep(0.01)
        assert sender.sent_count == sent_at_pause  # drained, not sent
        assert np.allclose(arm._target, 0.1)
        sender.resume()
        time.sleep(0.15)
        assert sender.sent_count == sent_at_pause  # the stale pre-resume target is never replayed
        _publish_until(slot, 0.3, sender, sent_at_pause + 1)
        assert sender.sent_count >= sent_at_pause + 1 and np.allclose(arm._target, 0.3)
    finally:
        sender.stop()


def test_arm_sender_pause_drops_the_pending_gripper_target():
    """A gripper target put before the fault is never dispatched after resume():
    only a FRESH put_gripper() reaches ``command_gripper`` (04-runtime §15)."""
    arm = FakeArm("arm0")
    slot: LatestSlot = LatestSlot()
    sender = ArmSender("arm0", arm, slot)
    sender.put_gripper(0.28)  # queued right before the fault, never dispatched
    sender.pause()
    sender.start()
    try:
        for _ in range(10):
            slot.put(np.full(7, 0.1))
            time.sleep(0.01)
        assert arm.gripper_commands == [] and sender.sent_count == 0
        sender.resume()
        _publish_until(slot, 0.1, sender, 1)
        time.sleep(0.05)
        assert arm.gripper_commands == []  # the pre-fault 0.28 was dropped, not replayed
        sender.put_gripper(0.9)  # fresh operator input after the re-seed
        _publish_until(slot, 0.1, sender, sender.sent_count + 1)
        deadline = time.monotonic() + 2.0
        while not arm.gripper_commands and time.monotonic() < deadline:
            time.sleep(0.005)
        assert [c.open_frac for c in arm.gripper_commands] == [0.9]
    finally:
        sender.stop()


# -- coupling to the hardware package (by name) ------------------------------------------------
def test_fake_events_mirror_the_hardware_event_types():
    hw = pytest.importorskip("apollo_mavis_v2_hardware.events")
    names = {name for name in dir(hw) if not name.startswith("_")}
    assert FAULT_EVENT_NAMES <= names
    for fake in (FaultEvent, RecoveredEvent, ReseedEvent, StudioConflictWarning):
        real = getattr(hw, fake.__name__)
        assert [f.name for f in dataclasses.fields(fake)] == [
            f.name for f in dataclasses.fields(real)
        ], fake.__name__
    from apollo_mavis_v2_hardware import RecoveryResult as HwRecoveryResult
    from fakes import RecoveryResult

    assert [f.name for f in dataclasses.fields(RecoveryResult)] == [
        f.name for f in dataclasses.fields(HwRecoveryResult)
    ]
    # The SDK title table is what fault_detail shows on the wire.
    assert controller_error_title(24) == "controller error 24: Speed Exceeds Limit"
    assert controller_error_title(0) == ""
