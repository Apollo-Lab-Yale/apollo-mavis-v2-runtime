"""Device-held codes in the ControlLoop (13-tracker §1.1): ``held ∪
device_codes`` with per-source scales — device trigger engages the clutch and
drives the EE while the WS watchdog is latched, a dead controller stream holds
and stops the gripper within ``stale_s``, trackpad up/down drive the gripper
of the grip arm only, both sources holding the clutch take the larger scale,
device movement codes cancel plans, no provider => device codes inert."""

from __future__ import annotations

import numpy as np
import pytest
from test_tracker_teleop import CLUTCH, DT, Rig

from apollo_xarm7_runtime.control.loop import HeldSources
from apollo_xarm7_runtime.devices.tracker import GRIPPER_CLOSE_CODE, GRIPPER_OPEN_CODE
from apollo_xarm7_runtime.safety.watchdog import WatchdogState

OPEN, CLOSE = GRIPPER_OPEN_CODE, GRIPPER_CLOSE_CODE  # KeyH / KeyF
GRIP_STEP = 1.2 * DT  # gripper_frac_ps * dt


def test_held_sources_merge_and_per_code_scale():
    src = HeldSources(frozenset({"KeyW", CLUTCH}), 0.5, frozenset({CLUTCH, CLOSE}), 1.0)
    assert src.held == {"KeyW", CLUTCH, CLOSE}
    assert src.scale_for("KeyW") == 0.5 and src.scale_for(CLOSE) == 1.0
    assert src.scale_for(CLUTCH) == 1.0  # both hold it: the larger scale
    assert src.scale_for("KeyX") == 0.0
    latched = HeldSources(frozenset({"KeyW"}), 0.0, frozenset({CLUTCH}), 1.0)
    assert latched.moving(frozenset({"KeyW"})) is False and latched.moving(frozenset({CLUTCH}))
    assert HeldSources().held == frozenset() and HeldSources().scale_for(CLUTCH) == 0.0


def test_device_trigger_drives_ee_and_gripper_with_ws_deadman_latched():
    rig = Rig()
    rig.hold("KeyW")
    rig.tick(2)
    rig.latch_ws()  # WS deadman: scale 0 until an EMPTY KeysMsg arrives
    rig.tick(3, heartbeat=False)
    q0 = rig.cmd()
    assert rig.loop.supervisor.watchdog.state is WatchdogState.AWAIT_EMPTY
    assert rig.loop.supervisor.watchdog.scale(rig.t) == 0.0
    rig.sample([0.0, 0.0, 0.0])
    rig.device(CLUTCH, CLOSE)  # trigger click + trackpad-down, no KeysMsg at all
    rig.tick(heartbeat=False)  # engage at Δ=0
    assert np.allclose(rig.cmd()[:3], q0[:3])
    ex = rig.extra()
    assert ex["clutch"] is True and ex["engaged_arm"] == "arm0"
    assert rig.loop.sources.device == {CLUTCH, CLOSE} and rig.loop.sources.device_scale == 1.0
    assert rig.loop.sources.ws == {"KeyW"} and rig.loop.sources.ws_scale == 0.0
    rig.sample([0.01, 0.0, 0.0])
    rig.tick(heartbeat=False)
    assert np.allclose(rig.cmd()[:3], q0[:3] + [0.01, 0.0, 0.0], atol=1e-12)  # full step
    assert rig.loop._grip_frac["arm0"] == pytest.approx(1.0 - 2 * GRIP_STEP)
    rig.sample([0.01, 0.0, 0.0], age=0.5, sticky=False)  # sample stale: hold, codes gone
    rig.tick(heartbeat=False)
    assert np.allclose(rig.cmd()[:3], q0[:3] + [0.01, 0.0, 0.0], atol=1e-12)
    assert rig.loop.sources.device == frozenset() and rig.loop.sources.device_scale == 0.0
    assert rig.loop._grip_frac["arm0"] == pytest.approx(1.0 - 2 * GRIP_STEP)
    assert rig.extra()["engaged_arm"] is None


def test_controller_stream_dies_holds_and_gripper_stops_within_stale_s():
    rig = Rig()
    rig.sample([0.0, 0.0, 0.0])
    rig.device(CLUTCH, CLOSE)
    rig.tick()
    rig.sample([0.01, 0.0, 0.0])
    rig.tick(4)
    q_hold = rig.cmd()
    assert q_hold[0] == pytest.approx(0.01)
    frac = rig.loop._grip_frac["arm0"]
    assert frac == pytest.approx(1.0 - 5 * GRIP_STEP)
    rig.pose = None  # the device stops reporting (no more samples)
    rig.tick(15)  # 0.15 s < stale_s: the last sample is still fresh, codes still count
    assert rig.loop._grip_frac["arm0"] == pytest.approx(frac - 15 * GRIP_STEP)
    rig.tick(35)  # crosses stale_s (0.2 s): the gripper stops within 20 +- 1 ticks
    frac_stale = rig.loop._grip_frac["arm0"]
    assert frac - 21 * GRIP_STEP - 1e-9 <= frac_stale <= frac - 19 * GRIP_STEP + 1e-9
    rig.tick(10)  # stale: gripper stays, clutch anchors cleared, arm holds
    assert rig.loop._grip_frac["arm0"] == pytest.approx(frac_stale)
    assert np.allclose(rig.cmd(), q_hold)
    ex = rig.extra()
    assert ex["clutch"] is False and ex["engaged_arm"] is None and ex["anchor_tcp"] is None


def test_trackpad_up_down_drive_gripper_of_grip_arm_only():
    rig = Rig(gripper_arms={"arm1"})  # arm0 is camera-only (mavis_v2 "view")
    rig.sample([0.0, 0.0, 0.0])
    rig.device(CLOSE)
    rig.tick(5)
    assert "arm0" not in rig.loop._grip_frac and rig.loop._grip_frac["arm1"] == 1.0
    assert rig.act("switch_arm").ok and rig.loop.active_arm == "arm1"  # one tick of CLOSE
    rig.tick(4)
    assert rig.loop._grip_frac["arm1"] == pytest.approx(1.0 - 5 * GRIP_STEP)
    rig.device(OPEN)
    rig.tick(3)
    assert rig.loop._grip_frac["arm1"] == pytest.approx(1.0 - 2 * GRIP_STEP)
    rig.device()  # trackpad released (|y| within the deadzone): no change
    rig.tick(3)
    assert rig.loop._grip_frac["arm1"] == pytest.approx(1.0 - 2 * GRIP_STEP)
    rig.device(OPEN)
    rig.hold(CLOSE)  # WS closes while the device opens: rates cancel out
    rig.tick(3)
    assert rig.loop._grip_frac["arm1"] == pytest.approx(1.0 - 2 * GRIP_STEP)
    assert np.allclose(rig.cmd("arm1")[:3], 0.0)  # gripper codes never move the EE


def test_clutch_held_by_both_sources_takes_the_larger_scale():
    rig = Rig()
    rig.sample([0.0, 0.0, 0.0])
    rig.hold(CLUTCH)
    rig.device(CLUTCH)
    rig.tick()  # engage (both sources fresh)
    rig.sample([0.01, 0.0, 0.0])
    rig.tick()
    assert rig.cmd()[0] == pytest.approx(0.01)
    assert rig.loop.sources.scale_for(CLUTCH) == 1.0
    # WS mid-ramp (scale 0.5) but the device still fresh: full step, not half.
    t_last_rx = rig.t
    rig.t = t_last_rx + 0.25
    rig.sample([0.02, 0.0, 0.0])
    rig.loop.run_tick(rig.t)
    assert rig.loop.supervisor.watchdog.scale(rig.t) == pytest.approx(0.5)
    assert rig.loop.sources.scale_for(CLUTCH) == 1.0
    assert rig.cmd()[0] == pytest.approx(0.02)
    # WS heartbeat back (empty set clears the latch), device stale: WS scale rules.
    rig.hold()
    rig.hold(CLUTCH)
    rig.device()
    rig.sample([0.02, 0.0, 0.0], age=0.5, sticky=False)  # device gone stale
    rig.loop.run_tick(rig.t)
    assert rig.loop.sources.device == frozenset() and rig.loop.sources.ws == {CLUTCH}
    assert rig.loop.sources.scale_for(CLUTCH) == 1.0  # WS alone, fresh
    rig.sample([0.02, 0.0, 0.0])
    rig.tick()  # re-anchors after the stale gap: no jump
    assert rig.cmd()[0] == pytest.approx(0.02)
    rig.sample([0.03, 0.0, 0.0])
    rig.tick()
    assert rig.cmd()[0] == pytest.approx(0.03)


def test_ws_clutch_latched_but_device_clutch_fresh_still_moves():
    rig = Rig()
    rig.sample([0.0, 0.0, 0.0])
    rig.hold(CLUTCH)
    rig.tick()
    rig.latch_ws()
    rig.device(CLUTCH)
    rig.tick(heartbeat=False)
    rig.sample([0.01, 0.0, 0.0])
    rig.tick(heartbeat=False)
    assert rig.loop.sources.ws == {CLUTCH} and rig.loop.sources.ws_scale == 0.0
    assert rig.cmd()[0] == pytest.approx(0.01)
    rig.device()  # device releases: only the latched WS clutch remains -> hold
    rig.sample([0.02, 0.0, 0.0])
    rig.tick(3, heartbeat=False)
    assert rig.cmd()[0] == pytest.approx(0.01)
    assert rig.extra()["clutch"] is True and rig.extra()["engaged_arm"] is None


def test_device_movement_code_cancels_plan_even_with_ws_latched():
    rig = Rig()
    rig.latch_ws()
    goal = np.array([0.5] * 7 + [0.2])
    rig.loop.plans.load("arm0", [goal])
    rig.loop._plan_state["arm0"] = "executing"
    rig.tick(2, heartbeat=False)
    assert rig.loop.plans.active("arm0")
    rig.sample([0.0, 0.0, 0.0])
    rig.device(CLUTCH)
    rig.tick(heartbeat=False)
    assert not rig.loop.plans.active("arm0") and rig.loop._plan_status == "cancelled"
    assert not np.allclose(rig.cmd(), goal)


def test_without_provider_device_codes_are_inert():
    rig = Rig(tracker=False)
    rig.sample([0.0, 0.0, 0.0])
    rig.device(CLUTCH, CLOSE)
    rig.tick()
    rig.sample([0.05, 0.0, 0.0])
    rig.tick(3)
    assert np.allclose(rig.cmd(), 0.0)
    assert rig.loop._grip_frac["arm0"] == 1.0
    assert rig.loop.sources.device == frozenset() and rig.loop.sources.device_scale == 0.0
