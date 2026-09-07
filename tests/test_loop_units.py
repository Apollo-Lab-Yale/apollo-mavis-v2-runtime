"""ControlLoop unit behaviour over FakeWorkcell: jog/goto, switch_arm, acks."""

from __future__ import annotations

import time

import numpy as np
from apollo_mavis_v2_core import Command, HeldState, PlanResult
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


def test_jog_above_goto_threshold_nacked(fake_loop):
    cell, bus, loop = fake_loop
    fut = submit(bus, "joint_target", arm_id="arm0", positions=[0.2] * 7 + [0.0], mode="jog")
    run_ticks(loop, cell, 1)
    res = fut.result(0)
    assert not res.ok and "goto" in res.detail


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
