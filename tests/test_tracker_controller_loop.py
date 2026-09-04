"""Device-held codes in the ControlLoop (13-tracker §1.1): ``held ∪
device_codes`` with per-source scales — device trigger engages the clutch and
drives the EE while the WS watchdog is latched, a dead controller stream holds
and stops the gripper within ``stale_s``, trackpad up/down drive the gripper
of the grip arm only, both sources holding the clutch take the larger scale,
device movement codes cancel plans, no provider => device codes inert; device
rail codes (trackpad left/right) integrate the rail at the device scale with no
browser / a latched WS deadman and never exceed ``rail_mps``; a menu press edge
fires ``switch_arm`` through the device path once per press and two press edges
inside one tick both fire (lossless ``click_actions``); button/touch edges on a
dead pose stream never keep the clutch engaged (the re-publish does not refresh
the pose's age); a rail-only tick holds the joint posture (the TCP rides the
rail) and invalidates the teleop seed so the next translate / clutch tick has no
leash-sized jump (13-tracker §4 (d))."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from apollo_xarm7_core import Pose, se3
from test_tracker_teleop import CLUTCH, DT, IDENT, Rig

from apollo_xarm7_runtime.config import ControllerMapConfig, TrackerConfig
from apollo_xarm7_runtime.control.loop import HeldSources
from apollo_xarm7_runtime.devices.tracker import (
    GRIPPER_CLOSE_CODE,
    GRIPPER_OPEN_CODE,
    RAIL_NEG_CODE,
    RAIL_POS_CODE,
    ControllerState,
    TrackerReader,
    derive_click_actions,
    derive_held_codes,
    note_edges,
)
from apollo_xarm7_runtime.safety.watchdog import WatchdogState

OPEN, CLOSE = GRIPPER_OPEN_CODE, GRIPPER_CLOSE_CODE  # KeyH / KeyF
RAIL_NEG, RAIL_POS = RAIL_NEG_CODE, RAIL_POS_CODE  # ArrowLeft / ArrowRight
GRIP_STEP = 1.2 * DT  # gripper_frac_ps * dt
RAIL_STEP = 0.10 * DT  # rail_mps * dt
LIN_STEP = 0.12 * DT  # linear_mps * dt (one KeyW tick)
DZ = 0.3


class RailPoseKin:
    """Rail-aware fake: the rail (q[7], along world x) carries the arm base, so
    TCP.x = q[0] + q[7] like ``SceneKinematics.tcp_world`` on a rail scene."""

    def tcp_world(self, arm_id, q):
        pos = np.array(q[:3], dtype=float)
        if len(q) > 7:
            pos[0] += q[7]
        return Pose(pos, se3.rotvec_to_quat(np.array(q[3:6], float)))

    def base_quat_world(self, arm_id):
        return IDENT


class RailPoseIK:
    """Exact IK for RailPoseKin at the seed's rail position."""

    def solve(self, arm_id, target, q_last):
        q = np.array(q_last, dtype=float)
        q[:3] = target.position
        if len(q) > 7:
            q[0] -= q_last[7]
        q[3:6] = se3.quat_to_rotvec(target.orientation)
        return SimpleNamespace(q=q, diverged=False, pos_err_m=0.0, rot_err_rad=0.0)

    def sync_passive(self, states):
        pass

    def reset(self, arm_id, q):
        pass


def _rail_rig(**kw) -> Rig:
    return Rig(kin=RailPoseKin(), ik=RailPoseIK(), **kw)


def _teleop_then_rail(rig: Rig, rail_ticks: int = 100) -> np.ndarray:
    """KeyW x5 (seeds the integrator), release, rail_pos held ``rail_ticks``
    (base slides under the frozen target), release; returns the held q."""
    rig.hold("KeyW")
    rig.tick(5)
    rig.hold()
    rig.tick()
    rig.hold(RAIL_POS)
    rig.tick(rail_ticks)
    rig.hold()
    rig.tick()
    return rig.cmd()


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


# -- device rail codes (13-tracker §1.1: rail at the device scale) ------------------------------
def test_device_rail_code_moves_rail_with_ws_scale_zero_and_stops_when_stale():
    rig = Rig()
    rig.latch_ws()  # no browser holds the control link: WS scale 0.0, no KeysMsg at all
    rig.sample([0.0, 0.0, 0.0])
    rig.device(RAIL_POS)  # trackpad right
    rig.tick(5, heartbeat=False)
    assert rig.loop.sources.ws_scale == 0.0 and rig.loop.sources.device == {RAIL_POS}
    assert rig.cmd()[7] == pytest.approx(5 * RAIL_STEP)
    assert np.allclose(rig.cmd()[:7], 0.0)  # joints hold: a rail-only tick moves the rail only
    assert "arm0" not in rig.loop._teleop_seeded  # ...and never seeds the teleop integrator
    assert rig.extra()["engaged_arm"] is None
    rig.device(RAIL_NEG)  # trackpad left
    rig.tick(2, heartbeat=False)
    assert rig.cmd()[7] == pytest.approx(3 * RAIL_STEP)
    rig.sample([0.0, 0.0, 0.0], age=0.5, sticky=False)  # controller stream stale: released
    rig.tick(3, heartbeat=False)
    assert rig.loop.sources.device == frozenset() and rig.loop.sources.device_scale == 0.0
    assert rig.cmd()[7] == pytest.approx(3 * RAIL_STEP)  # rail stopped
    rig.sample([0.0, 0.0, 0.0])  # fresh again: resumes at the device scale
    rig.tick(1, heartbeat=False)
    assert rig.cmd()[7] == pytest.approx(2 * RAIL_STEP)


def test_device_rail_code_survives_ws_deadman_latch_keyboard_rail_does_not():
    rig = Rig()
    rig.sample([0.0, 0.0, 0.0])
    rig.hold(RAIL_POS)  # keyboard ArrowRight alone
    rig.tick(3)
    assert rig.cmd()[7] == pytest.approx(3 * RAIL_STEP)
    assert np.allclose(rig.cmd()[:7], 0.0)
    rig.latch_ws()
    rig.tick(2, heartbeat=False)
    assert rig.loop.supervisor.watchdog.state is WatchdogState.AWAIT_EMPTY
    assert rig.cmd()[7] == pytest.approx(3 * RAIL_STEP)  # WS rail latched: holds
    rig.device(RAIL_POS)
    rig.tick(2, heartbeat=False)
    assert rig.loop.sources.ws == {RAIL_POS} and rig.loop.sources.ws_scale == 0.0
    assert rig.cmd()[7] == pytest.approx(5 * RAIL_STEP)  # device rail at 1.0 while WS is latched
    rig.device()
    rig.tick(2, heartbeat=False)
    assert rig.cmd()[7] == pytest.approx(5 * RAIL_STEP)


def test_same_direction_ws_and_device_rail_codes_never_exceed_rail_mps():
    rig = Rig()
    rig.sample([0.0, 0.0, 0.0])
    rig.hold(RAIL_POS)
    rig.device(RAIL_POS)
    rig.tick(4)
    assert rig.loop.sources.scale_for(RAIL_POS) == 1.0
    assert rig.cmd()[7] == pytest.approx(4 * RAIL_STEP)  # not 2x
    rig.hold(RAIL_NEG)  # WS left vs device right: rates cancel
    rig.tick(3)
    assert rig.cmd()[7] == pytest.approx(4 * RAIL_STEP)
    rig.hold()
    rig.device()
    rig.tick()
    # The per-source sum is clamped to the configured rail speed even when the
    # sources' scales would add up (defensive: WS mid-ramp + device fresh).
    src = HeldSources(frozenset({RAIL_POS}), 0.5, frozenset({RAIL_POS}), 1.0)
    assert rig.loop._rail_rate(src) == pytest.approx(0.10)
    assert rig.loop._rail_rate(HeldSources(frozenset({RAIL_NEG}), 0.5)) == pytest.approx(-0.05)
    assert rig.loop._rail_rate(HeldSources(frozenset({RAIL_NEG, RAIL_POS}), 1.0)) == 0.0
    assert rig.loop._rail_rate(HeldSources(frozenset({"KeyW", CLUTCH}), 1.0)) == 0.0


def test_rail_and_gripper_device_codes_together_while_clutched_and_latched():
    rig = Rig()
    rig.sample([0.0, 0.0, 0.0])
    rig.latch_ws()
    rig.device(CLUTCH, RAIL_POS, CLOSE)
    rig.tick(heartbeat=False)  # engage at Δ=0, rail + gripper start
    rig.sample([0.01, 0.0, 0.0])
    rig.tick(heartbeat=False)
    q = rig.cmd()
    assert q[0] == pytest.approx(0.01) and q[7] == pytest.approx(2 * RAIL_STEP)
    assert rig.loop._grip_frac["arm0"] == pytest.approx(1.0 - 2 * GRIP_STEP)


# -- rail-only semantics + seed invalidation (13-tracker §4 re-seed rule (d)) --------------------
def test_rail_only_hold_keeps_posture_and_tcp_rides_the_rail():
    rig = _rail_rig()
    rig.hold("KeyW")
    rig.tick(5)
    q_before = rig.cmd()
    assert q_before[0] == pytest.approx(5 * LIN_STEP) and "arm0" in rig.loop._teleop_seeded
    rig.hold(RAIL_POS)
    rig.tick(100)
    q = rig.cmd()
    assert np.allclose(q[:7], q_before[:7])  # joints hold: no IK compensation
    assert q[7] == pytest.approx(100 * RAIL_STEP)  # 0.10 m of rail travel
    tcp = rig.loop.kin.tcp_world("arm0", q)
    assert tcp.position[0] == pytest.approx(q_before[0] + 100 * RAIL_STEP)  # TCP rode the rail
    assert "arm0" not in rig.loop._teleop_seeded  # seed invalidated: base moved under the target
    assert rig.loop.integrator.get("arm0").position[0] == pytest.approx(5 * LIN_STEP)  # stale
    # Never-seeded arm: a rail-only tick is a no-op for the seed set too.
    rig2 = _rail_rig()
    rig2.hold(RAIL_POS)
    rig2.tick(3)
    assert np.allclose(rig2.cmd()[:7], 0.0) and rig2.cmd()[7] == pytest.approx(3 * RAIL_STEP)
    assert "arm0" not in rig2.loop._teleop_seeded


def test_translate_after_rail_move_reseeds_from_measured_tcp_no_jump():
    rig = _rail_rig()
    q0 = _teleop_then_rail(rig)
    rig.hold("KeyW")
    rig.tick()  # first driven tick after the rail move: one twist step, not the 25 mm leash
    dq = rig.cmd()[:7] - q0[:7]
    assert dq[0] == pytest.approx(LIN_STEP, abs=1e-9)
    assert np.allclose(dq[1:], 0.0, atol=1e-12)
    assert rig.cmd()[7] == pytest.approx(q0[7])
    tgt = rig.loop.integrator.get("arm0")
    assert tgt.position[0] == pytest.approx(q0[0] + q0[7] + LIN_STEP, abs=1e-9)  # re-seeded
    rig.tick(3)
    assert rig.cmd()[0] == pytest.approx(q0[0] + 4 * LIN_STEP, abs=1e-9)


def test_clutch_after_rail_move_engages_with_zero_delta():
    rig = _rail_rig()
    rig.sample([0.0, 0.0, 0.0])
    q0 = _teleop_then_rail(rig)
    rig.hold(CLUTCH)
    rig.tick()  # engage with a still tracker: A_ee = measured TCP, no motion
    assert np.allclose(rig.cmd()[:7], q0[:7], atol=1e-12)
    assert rig.extra()["engaged_arm"] == "arm0"
    rig.sample([0.01, 0.0, 0.0])
    rig.tick()
    assert rig.cmd()[0] == pytest.approx(q0[0] + 0.01, abs=1e-9)  # hand delta 1:1 from there
    assert rig.cmd()[7] == pytest.approx(q0[7])


def test_device_rail_with_ws_latched_then_device_clutch_engages_with_zero_delta():
    rig = _rail_rig()
    rig.sample([0.0, 0.0, 0.0])
    rig.hold("KeyW")
    rig.tick(5)
    rig.hold()
    rig.tick()
    rig.latch_ws()  # browser gone: WS scale 0, device codes still drive
    rig.device(RAIL_POS)
    rig.tick(100, heartbeat=False)
    rig.device()
    rig.tick(heartbeat=False)
    q0 = rig.cmd()
    assert q0[7] == pytest.approx(100 * RAIL_STEP) and q0[0] == pytest.approx(5 * LIN_STEP)
    assert "arm0" not in rig.loop._teleop_seeded
    rig.device(CLUTCH)
    rig.tick(heartbeat=False)
    assert np.allclose(rig.cmd()[:7], q0[:7], atol=1e-12)
    rig.sample([0.0, 0.0, 0.005])
    rig.tick(heartbeat=False)
    assert np.allclose(rig.cmd()[:3], q0[:3] + [0.0, 0.0, 0.005], atol=1e-9)


def test_rail_while_tracker_stale_invalidates_seed_and_reengage_has_no_jump():
    rig = _rail_rig()
    rig.sample([0.0, 0.0, 0.0])
    rig.hold(CLUTCH)
    rig.tick()
    rig.sample([0.02, 0.0, 0.0])
    rig.tick()
    assert rig.cmd()[0] == pytest.approx(0.02)
    rig.sample([0.02, 0.0, 0.0], age=1.0, sticky=False)  # sample stale: clutch holds...
    rig.hold(CLUTCH, RAIL_POS)  # ...but the rail keeps integrating under the frozen target
    rig.tick(50)
    q0 = rig.cmd()
    assert q0[0] == pytest.approx(0.02) and q0[7] == pytest.approx(50 * RAIL_STEP)
    assert "arm0" not in rig.loop._teleop_seeded
    rig.hold(CLUTCH)
    rig.sample([0.05, 0.0, 0.0])  # fresh again, hand moved meanwhile: re-anchor, no jump
    rig.tick()
    assert np.allclose(rig.cmd()[:7], q0[:7], atol=1e-12)


# -- device discrete actions through the real derivation (menu -> switch_arm) --------------------
def _script(rig: Rig, cfg: TrackerConfig, prev: ControllerState | None, **fields):
    """Feed the loop what the reader would publish for a new raw state."""
    st = note_edges(prev, ControllerState(**fields), DZ)
    codes = derive_held_codes(st, cfg)
    rig.device(*codes, controller=st, click_actions=derive_click_actions(st, cfg))
    return st


def test_menu_press_edge_fires_switch_arm_once_via_device_path():
    cfg = TrackerConfig()
    rig = Rig()
    rig.sample([0.0, 0.0, 0.0])
    idle = _script(rig, cfg, None)  # idle controller adopted (edge_seq 0)
    rig.tick()
    assert rig.loop.active_arm == "arm0" and rig.extra()["device_action"] is None
    menu = _script(rig, cfg, idle, menu=True)
    assert menu.edge_seq == 1 and menu.edge_input == "menu_click"
    assert rig.codes == frozenset() and rig.click_actions == ((1, "switch_arm"),)
    rig.tick()
    assert rig.loop.active_arm == "arm1" and rig.extra()["device_action"] == "switch_arm"
    rig.tick(25)  # menu held: fires once
    assert rig.loop.active_arm == "arm1"
    released = _script(rig, cfg, menu)
    rig.tick(3)
    assert rig.loop.active_arm == "arm1"  # release edge fires nothing
    again = _script(rig, cfg, released, menu=True)
    assert again.edge_seq == 2
    rig.tick()
    assert rig.loop.active_arm == "arm0"  # second press wraps
    # Trackpad clicks are held bindings by default: they advance the counter, fire nothing.
    up = _script(rig, cfg, again, trackpad_touch=True, trackpad_click=True, trackpad_y=0.9)
    assert up.edge_seq == 3 and rig.codes == {OPEN}
    assert rig.click_actions == ((1, "switch_arm"), (2, "switch_arm"))  # remembered, both seen
    rig.tick(2)
    assert rig.loop.active_arm == "arm0" and rig.loop.sources.device == {OPEN}
    # Grip is unbound by default: edge counted, nothing fired, no code.
    grip = _script(rig, cfg, up, grip=True)
    assert grip.edge_seq == 4 and rig.codes == frozenset()
    assert rig.click_actions == ((1, "switch_arm"), (2, "switch_arm"))
    rig.tick(2)
    assert rig.loop.active_arm == "arm0"


def test_grip_bound_to_arm_prev_fires_switch_arm_prev_via_device_path():
    cfg = TrackerConfig(controller_map=ControllerMapConfig(arm_prev="grip_click"))
    rig = Rig()
    rig.sample([0.0, 0.0, 0.0])
    idle = _script(rig, cfg, None)
    rig.tick()
    grip = _script(rig, cfg, idle, grip=True)
    assert rig.click_actions == ((1, "switch_arm_prev"),)
    rig.tick()
    assert rig.loop.active_arm == "arm1" and rig.extra()["device_action"] == "switch_arm_prev"
    rig.tick(5)
    assert rig.loop.active_arm == "arm1"
    _script(rig, cfg, grip)  # release
    rig.tick()
    assert rig.loop.active_arm == "arm1"


# -- two press edges inside one tick: lossless (13-tracker §1.1) ---------------------------------
def test_two_press_edges_within_one_tick_both_reach_the_loop():
    cfg = TrackerConfig()  # menu -> switch_arm, grip unbound, trackpad up -> gripper_open (held)
    rig = Rig()
    rig.sample([0.0, 0.0, 0.0])
    idle = _script(rig, cfg, None)
    rig.tick()
    # menu DOWN then grip DOWN before the next tick: the slot only holds the grip
    # sample (edge_seq 2) -- its click_actions still carry the bound menu edge.
    menu = _script(rig, cfg, idle, menu=True)
    grip = _script(rig, cfg, menu, menu=True, grip=True)
    assert grip.edge_seq == 2 and grip.edge_input == "grip_click"
    assert rig.click_actions == ((1, "switch_arm"),)
    rig.tick()
    assert rig.loop.active_arm == "arm1" and rig.extra()["device_action"] == "switch_arm"
    rig.tick(3)
    assert rig.loop.active_arm == "arm1"  # fired exactly once
    released = _script(rig, cfg, grip)
    rig.tick()
    # menu DOWN then a trackpad-up click (held gripper binding) inside one tick.
    menu2 = _script(rig, cfg, released, menu=True)
    both = _script(
        rig, cfg, menu2, menu=True, trackpad_touch=True, trackpad_click=True, trackpad_y=0.9
    )
    assert both.edge_seq == 4 and rig.codes == {OPEN}
    rig.tick()
    assert rig.loop.active_arm == "arm0" and rig.loop.sources.device == {OPEN}
    # Two BOUND edges inside one tick fire in order: arm0 -> arm1 (next) -> arm0 (prev).
    cfg2 = TrackerConfig(controller_map=ControllerMapConfig(arm_prev="grip_click"))
    rig2 = Rig()
    rig2.sample([0.0, 0.0, 0.0])
    idle2 = _script(rig2, cfg2, None)
    rig2.tick()
    m = _script(rig2, cfg2, idle2, menu=True)
    g = _script(rig2, cfg2, m, menu=True, grip=True)
    assert rig2.click_actions == ((1, "switch_arm"), (2, "switch_arm_prev"))
    rig2.tick()
    assert rig2.loop.active_arm == "arm0"
    assert rig2.extra()["device_action"] == "switch_arm_prev"  # the last one fired
    # ... and grip then menu in the other order: prev then next, same end state.
    rel = _script(rig2, cfg2, g)
    rig2.tick()
    g2 = _script(rig2, cfg2, rel, grip=True)
    _script(rig2, cfg2, g2, grip=True, menu=True)
    assert rig2.click_actions[-2:] == ((3, "switch_arm_prev"), (4, "switch_arm"))
    rig2.tick()
    assert rig2.loop.active_arm == "arm0" and rig2.extra()["device_action"] == "switch_arm"


# -- button edges on a dead pose stream (13-tracker §4: stale => hold, anchors cleared) ----------
def _reader_rig() -> tuple[Rig, TrackerReader]:
    """The real reader publishes into the Rig's slot (the Rig itself feeds no
    samples); poses and controller states are driven by hand."""
    rig = Rig()
    reader = TrackerReader(TrackerConfig(backend="fake"), rig.bus.tracker, clock=lambda: rig.t)
    return rig, reader


def _pose(reader: TrackerReader, rig: Rig, x: float) -> None:
    reader._publish(Pose(np.array([x, 0.0, 0.0]), IDENT), np.zeros(3), np.zeros(3), rig.t)


@pytest.mark.parametrize("taps", [True, False])
def test_touch_edges_on_a_dead_pose_stream_release_the_clutch_and_never_jump(taps):
    rig, reader = _reader_rig()
    ctl = ControllerState(trigger=1.0, trigger_pressed=True)  # trigger held: device clutch
    reader._on_controller(ctl)
    for _ in range(3):
        _pose(reader, rig, 0.0)
        rig.tick()
    assert rig.extra()["engaged_arm"] == "arm0" and rig.loop.sources.device == {CLUTCH}
    q_hold = rig.cmd()
    t_last_pose = rig.t - DT
    # The pose stream dies for 1 s; with ``taps`` the thumb rests on / lifts off
    # the pad every 0.15 s (TOUCH_DOWN/UP edges re-publish the last pose).
    released_at = None
    for i in range(100):
        if taps and i % 15 == 14:
            ctl = replace(ctl, trackpad_touch=not ctl.trackpad_touch)
            assert reader._on_controller(ctl) is True
            assert rig.bus.tracker.get()[0].held_codes == {CLUTCH}  # codes keep riding along
        rig.tick()
        if released_at is None and rig.extra()["engaged_arm"] is None:
            released_at = rig.t
    assert released_at is not None and released_at - t_last_pose <= 0.2 + 2 * DT
    assert rig.extra()["engaged_arm"] is None and rig.extra()["anchor_tcp"] is None
    assert np.allclose(rig.cmd(), q_hold)
    st = reader.status()
    assert st.status == "stale" and st.age_s == pytest.approx(rig.t - t_last_pose, abs=1e-9)
    assert st.rate_hz == 0.0  # edges are not poses
    if taps:
        assert rig.bus.tracker.get()[0].valid is False and "older than" in st.detail
        assert rig.loop.sources.device == {CLUTCH}  # the controller stream itself is fresh
    # The pose resumes 8 cm away: re-engage at zero delta, never a leash-sized jump.
    _pose(reader, rig, 0.08)
    rig.tick()
    assert np.allclose(rig.cmd(), q_hold)
    assert rig.extra()["engaged_arm"] == "arm0"
    assert reader.status().status == "tracking" and reader.status().detail == ""
    _pose(reader, rig, 0.09)
    rig.tick()
    assert rig.cmd()[0] == pytest.approx(q_hold[0] + 0.01)


# -- rail inputs slide the WHOLE arm while driven (04-runtime §6 "Rail", 2026-09-03) ------------
def test_keyboard_translate_plus_rail_code_rides_the_rail_no_ik_fold_back():
    rig = _rail_rig()
    rig.hold("KeyW")
    rig.tick(5)
    q0 = rig.cmd()
    rig.hold("KeyW", RAIL_POS)
    rig.tick(10)
    q = rig.cmd()
    assert q[7] == pytest.approx(q0[7] + 10 * RAIL_STEP)
    # joints carry the translate steps only: no compensation of the rail travel
    assert q[0] == pytest.approx(q0[0] + 10 * LIN_STEP, abs=1e-9)
    tcp = rig.loop.kin.tcp_world("arm0", q)
    assert tcp.position[0] == pytest.approx(q0[0] + q0[7] + 10 * (LIN_STEP + RAIL_STEP), abs=1e-9)
    tgt = rig.loop.integrator.get("arm0")
    assert tgt.position[0] == pytest.approx(tcp.position[0], abs=1e-9)  # target rode along
    assert "arm0" in rig.loop._teleop_seeded  # a driven tick keeps its seed
    rig.hold("KeyW")
    rig.tick()
    assert rig.cmd()[0] == pytest.approx(q[0] + LIN_STEP, abs=1e-9)  # continues seamlessly
    assert rig.cmd()[7] == pytest.approx(q[7])


def test_clutched_rail_code_slides_arm_and_keeps_hand_offset():
    rig = _rail_rig()
    rig.sample([0.0, 0.0, 0.0])
    rig.hold(CLUTCH)
    rig.tick()  # engage with a still hand
    q0 = rig.cmd()
    rig.hold(CLUTCH, RAIL_POS)
    rig.tick(50)
    q = rig.cmd()
    assert q[7] == pytest.approx(q0[7] + 50 * RAIL_STEP)
    assert np.allclose(q[:7], q0[:7], atol=1e-9)  # posture held: the anchor rode the rail
    assert rig.extra()["engaged_arm"] == "arm0"  # the clutch session survived the rail move
    anchor = rig.extra()["anchor_tcp"]
    assert anchor.position[0] == pytest.approx(q0[0] + q0[7] + 50 * RAIL_STEP, abs=1e-9)
    rig.hold(CLUTCH)
    rig.sample([0.02, 0.0, 0.0])
    rig.tick()
    q2 = rig.cmd()
    assert q2[0] == pytest.approx(q0[0] + 0.02, abs=1e-9)  # hand delta 1:1 from the ridden anchor
    assert q2[7] == pytest.approx(q[7])


def test_rail_bound_caps_the_ride_along_shift():
    rig = _rail_rig()
    rig.hold("KeyW")
    rig.tick()
    rig.hold("KeyW", RAIL_NEG)  # rail already at 0: the step is clamped, so no shift
    rig.tick(5)
    q = rig.cmd()
    assert q[7] == 0.0
    assert q[0] == pytest.approx(6 * LIN_STEP, abs=1e-9)
    assert rig.loop.integrator.get("arm0").position[0] == pytest.approx(6 * LIN_STEP, abs=1e-9)
