"""EngageMachine (phase-15; 16-gello §6.1 / §6.2): the ±2π unwrap on joints 1/3/5/7 incl.
the +π vs −π branch, the joint-limit helpers (the local table equals the hardware
package's), and the state machine — engage tolerance, leash, pause / resume (sticky),
fault -> paused, planned motion -> re-engage at the end, pause-before-motion."""

from __future__ import annotations

import math

import numpy as np
import pytest

from apollo_mavis_v2_runtime.config import GelloConfig
from apollo_mavis_v2_runtime.devices.gello import GelloSample
from apollo_mavis_v2_runtime.gello.engage import (
    JOINT_LIMIT_MARGIN_RAD,
    TWO_PI,
    WRAP_JOINTS,
    XARM7_JOINT_LIMITS_RAD,
    EngageMachine,
    apply_unwrap,
    clip_to_joint_limits,
    joint_limit_bounds,
    joint_limit_violation,
    unwrap_to_reference,
)

PI = math.pi
CFG = GelloConfig(backend="fake")  # engage_tol 0.10, leash 0.80, stale_s 0.2
Z7 = np.zeros(7)


def S(q, rx: float, valid: bool = True, seq: int = 1) -> GelloSample:
    q = np.asarray(q, dtype=np.float64)
    return GelloSample(q_raw=q.copy(), q=q, gripper_frac=1.0, rx_mono=rx, seq=seq, valid=valid)


# -- unwrap -------------------------------------------------------------------------------
def test_unwrap_picks_the_branch_nearest_the_reference_on_joints_1_3_5_7_only():
    leader = np.array([3.0, 3.0, -3.0, 0.0, 0.1, 0.0, 6.0])
    ref = np.array([-3.0, -3.0, 3.0, 0.0, 0.1, 0.0, -0.2])
    q, k = unwrap_to_reference(leader, ref)
    assert k.tolist() == [-1, 0, 1, 0, 0, 0, -1]
    assert q[0] == pytest.approx(3.0 - TWO_PI) and q[2] == pytest.approx(-3.0 + TWO_PI)
    assert q[1] == 3.0  # joint 2 is never unwrapped
    assert q[6] == pytest.approx(6.0 - TWO_PI)
    assert np.allclose(apply_unwrap(leader, k), q)
    assert WRAP_JOINTS == (0, 2, 4, 6)


def test_unwrap_plus_pi_versus_minus_pi_branch_and_the_exact_tie():
    # leader just below +π, arm just above −π: the −π branch is 0.2 away, +π is 6.08 away
    q, k = unwrap_to_reference([PI - 0.1, 0, 0, 0, 0, 0, 0], [-PI + 0.1, 0, 0, 0, 0, 0, 0])
    assert k[0] == -1 and q[0] == pytest.approx(-PI - 0.1)
    # mirrored
    q, k = unwrap_to_reference([-PI + 0.1, 0, 0, 0, 0, 0, 0], [PI - 0.1, 0, 0, 0, 0, 0, 0])
    assert k[0] == 1 and q[0] == pytest.approx(PI + 0.1)
    # an exact tie (π away either way) keeps the leader's own reading
    q, k = unwrap_to_reference([0.0] * 7, [PI, 0, 0, 0, 0, 0, 0])
    assert k[0] == 0 and q[0] == 0.0
    # k never leaves {-1, 0, 1}; the pure nearest rule (limits=None) picks +2π for a
    # reference three turns away, the limit-aware default keeps the in-limit 0.0
    q, k = unwrap_to_reference([0.0] * 7, [3 * TWO_PI] * 7, limits=None)
    assert set(k.tolist()) <= {-1, 0, 1} and q[0] == pytest.approx(TWO_PI)
    q, k = unwrap_to_reference([0.0] * 7, [3 * TWO_PI] * 7)
    assert k[0] == 0 and q[0] == 0.0
    with pytest.raises(ValueError):
        unwrap_to_reference([0.0] * 7, [0.0] * 8)


def test_unwrap_is_limit_aware_and_never_refuses_an_in_limit_leader_posture():
    """2026-09-09 review (16-gello §15.2 item 11): with the arm at J1 = 3.5 rad and GELLO
    reading 0.3 rad the nearest branch is 6.58 rad - beyond ±2π less the margin - although
    the leader's own reading is a reachable target. Candidates outside the limits are
    dropped when another lies inside: -5.98 (in), 0.3 (in), 6.58 (out) -> 0.3, k = 0."""
    lo, hi = joint_limit_bounds()
    leader = np.array([0.3, 0, 0, 0, 0, 0, 0])
    meas = np.array([3.5, 0, 0, 0, 0, 0, 0])
    q, k = unwrap_to_reference(leader, meas)
    assert k[0] == 0 and q[0] == pytest.approx(0.3)
    assert joint_limit_violation(q) is None
    q_raw, k_raw = unwrap_to_reference(leader, meas, limits=None)  # the old rule, for contrast
    assert k_raw[0] == 1 and q_raw[0] == pytest.approx(0.3 + TWO_PI) and q_raw[0] > hi[0]
    # mirrored: arm at -3.5, leader -0.3 -> -0.3 (k = 0), not -6.58
    q, k = unwrap_to_reference([-0.3, 0, 0, 0, 0, 0, 0], [-3.5, 0, 0, 0, 0, 0, 0])
    assert k[0] == 0 and q[0] == pytest.approx(-0.3)
    # when the in-limit survivors are the -2π and 0 branches the NEAREST of them wins
    q, k = unwrap_to_reference([3.0, 0, 0, 0, 0, 0, 0], [-3.0, 0, 0, 0, 0, 0, 0])
    assert k[0] == -1 and q[0] == pytest.approx(3.0 - TWO_PI)
    # a leader reading itself beyond the limit: the in-limit branch is taken (never a
    # refusal for a branch GELLO did not read); joints 2/4/6 are untouched
    q, k = unwrap_to_reference([6.5, 2.5, 0, 0, 0, 0, 0], [6.4, 0, 0, 0, 0, 0, 0])
    assert k[0] == -1 and q[0] == pytest.approx(6.5 - TWO_PI) and q[1] == 2.5 and k[1] == 0
    # the engage rule uses the same helper: out_of_sync by 3.2 rad, not a joint-limit clip
    m = EngageMachine(CFG)
    assert m.step(1.0, S(leader, 1.0), meas, meas) == "out_of_sync"
    assert m.lag_rad()[0] == pytest.approx(0.3 - 3.5) and "joint 1" in m.detail


# -- joint limits -------------------------------------------------------------------------
def test_local_joint_limit_table_equals_the_hardware_package():
    hw = pytest.importorskip("apollo_mavis_v2_hardware.config")
    assert XARM7_JOINT_LIMITS_RAD == hw.XARM7_JOINT_LIMITS_RAD
    assert JOINT_LIMIT_MARGIN_RAD == hw.ServoLimits().joint_limit_margin_rad


def test_joint_limit_clip_and_violation_text():
    lo, hi = joint_limit_bounds()
    assert lo[0] == pytest.approx(-TWO_PI + JOINT_LIMIT_MARGIN_RAD)
    assert hi[1] == pytest.approx(math.radians(120.0) - JOINT_LIMIT_MARGIN_RAD)
    q = np.array([0.0, 2.5, 0.0, -0.5, 0.0, 0.0, 7.0])
    c = clip_to_joint_limits(q)
    assert c[1] == pytest.approx(hi[1]) and c[3] == pytest.approx(lo[3])
    assert c[6] == pytest.approx(hi[6])
    assert joint_limit_violation(Z7) is None
    v = joint_limit_violation(q)
    assert v is not None and v.joint == 2 and v.value_rad == 2.5
    assert v.describe().startswith("joint 2 = 2.50 rad, limit [")
    v7 = joint_limit_violation([0, 0, 0, 0, 0, 0, 7.0])
    assert v7.joint == 7 and "±6.27" in v7.describe()
    assert joint_limit_violation([float("nan"), 0, 0, 0, 0, 0, 0]).joint == 1


# -- the state machine --------------------------------------------------------------------
def test_no_leader_for_missing_stale_or_invalid_samples():
    m = EngageMachine(CFG)
    assert m.state == "no_leader"
    assert m.step(1.0, None, Z7, Z7) == "no_leader" and "no leader" in m.detail
    assert m.step(1.0, S(Z7, rx=0.5), Z7, Z7) == "no_leader" and "stale" in m.detail  # 0.5 s old
    assert m.step(1.0, S(Z7, rx=1.0, valid=False), Z7, Z7) == "no_leader"
    assert "invalid" in m.detail and m.lag_rad() is None and m.target() is None


def test_engage_rule_tracking_within_tolerance_out_of_sync_beyond_and_automatic_reengage():
    m = EngageMachine(CFG)
    meas = np.array([PI, 0.0, 0.0, 0.5, 0.0, 0.3, 0.0, 0.65])  # 8 values: rail ignored
    near = meas[:7] + np.array([0.05, -0.05, 0.0, 0.02, 0.0, 0.0, 0.09])
    assert m.step(1.0, S(near, 1.0), meas, meas) == "tracking"
    assert m.detail == "tracking the leader" and m.transitions == 1
    assert np.allclose(m.lag_rad(), near - meas[:7]) and m.max_lag_rad() == pytest.approx(0.09)
    assert np.allclose(m.target(), near)
    far = meas[:7] + np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    # a leader 0.5 rad off while tracking is within the 0.8 leash of the command -> keep tracking
    assert m.step(1.01, S(far, 1.01), meas, meas) == "tracking"
    # ... but from out_of_sync / no_leader the engage rule applies: 0.5 > 0.10
    m2 = EngageMachine(CFG)
    assert m2.step(1.0, S(far, 1.0), meas, meas) == "out_of_sync"
    assert "0.50 rad from the arm (joint 3)" in m2.detail and "within 0.10 rad" in m2.detail
    assert m2.target() is None and m2.max_lag_rad() == pytest.approx(0.5)
    assert m2.step(1.02, S(near, 1.02), meas, meas) == "tracking"  # automatic re-engagement


def test_unwrap_at_engagement_then_continuity_while_tracking():
    m = EngageMachine(CFG)
    meas = np.array([-PI + 0.02, 0, 0, 0, 0, 0, 0])
    leader = np.array([PI - 0.02, 0, 0, 0, 0, 0, 0])  # the other side of the ±π seam
    assert m.step(1.0, S(leader, 1.0), meas, meas) == "tracking"
    assert m.target()[0] == pytest.approx(-PI - 0.02)  # unwrapped to the measured branch
    assert m.lag_rad()[0] == pytest.approx(-0.04)
    # the leader moves on: k stays -1 (no re-pick), the target follows continuously
    leader2 = np.array([PI - 0.3, 0, 0, 0, 0, 0, 0])
    q_cmd = np.array([-PI - 0.02, 0, 0, 0, 0, 0, 0])
    assert m.step(1.01, S(leader2, 1.01), q_cmd, q_cmd) == "tracking"
    assert m.target()[0] == pytest.approx(PI - 0.3 - TWO_PI)


def test_leash_while_tracking_drops_to_out_of_sync_then_reengages():
    m = EngageMachine(CFG)
    meas = Z7.copy()
    assert m.step(1.0, S(Z7, 1.0), meas, meas) == "tracking"
    ahead = np.array([0.0, 0.9, 0, 0, 0, 0, 0])  # 0.9 > leash 0.8 from the command
    assert m.step(1.01, S(ahead, 1.01), meas, meas) == "out_of_sync"
    assert "leash" in m.detail and "(joint 2" in m.detail and m.target() is None
    # the follower holds (command unchanged); the leader must come back within tolerance
    assert m.step(1.02, S(ahead, 1.02), meas, meas) == "out_of_sync"
    assert m.step(1.03, S(np.array([0, 0.05, 0, 0, 0, 0, 0]), 1.03), meas, meas) == "tracking"
    # the leash is measured against the COMMAND, not the measured arm
    m3 = EngageMachine(CFG)
    assert m3.step(1.0, S(Z7, 1.0), Z7, Z7) == "tracking"
    lagging_arm = np.array([0, 0.7, 0, 0, 0, 0, 0])  # measured arm 0.7 behind, command at leader
    assert m3.step(1.01, S(Z7, 1.01), lagging_arm, Z7) == "tracking"


def test_pause_is_sticky_and_resume_reengages():
    m = EngageMachine(CFG)
    assert m.step(1.0, S(Z7, 1.0), Z7, Z7) == "tracking"
    m.pause()
    assert m.state == "paused" and m.detail == "paused by operator" and m.target() is None
    assert m.step(1.01, S(Z7, 1.01), Z7, Z7) == "paused"  # in tolerance, still paused
    m.pause()  # idempotent
    assert m.step(1.02, None, Z7, Z7) == "paused"  # no leader does not change the display
    assert m.paused
    m.resume()
    assert not m.paused
    assert m.step(1.03, S(Z7, 1.03), Z7, Z7) == "tracking"
    m.resume()  # idempotent: no effect while not paused
    assert m.state == "tracking"
    # the same through step()'s paused_request
    assert m.step(1.04, S(Z7, 1.04), Z7, Z7, paused_request=True) == "paused"
    assert m.step(1.05, S(Z7, 1.05), Z7, Z7) == "paused"
    assert m.step(1.06, S(Z7, 1.06), Z7, Z7, paused_request=False) == "tracking"
    # resume with the leader far away -> out_of_sync (the tolerance rule), not tracking
    m.pause()
    far = np.array([0.5, 0, 0, 0, 0, 0, 0])
    assert m.step(1.07, S(far, 1.07), Z7, Z7, paused_request=False) == "out_of_sync"
    # lag keeps being published while paused (the panel's bars)
    m.pause()
    m.step(1.08, S(far, 1.08), Z7, Z7)
    assert m.max_lag_rad() == pytest.approx(0.5)


def test_fault_forces_paused_until_resume():
    m = EngageMachine(CFG)
    assert m.step(1.0, S(Z7, 1.0), Z7, Z7) == "tracking"
    m.on_fault("controller error C24")
    assert m.state == "paused" and m.detail == "paused: controller error C24"
    assert m.step(1.01, S(Z7, 1.01), Z7, Z7) == "paused"
    m.resume()
    assert m.step(1.02, S(Z7, 1.02), Z7, Z7) == "tracking"


def test_planned_motion_owns_the_arm_and_the_engage_rule_runs_at_its_end():
    m = EngageMachine(CFG)
    assert m.step(1.0, S(Z7, 1.0), Z7, Z7) == "tracking"
    # via step(plan_active=...): the launch motion
    assert m.step(1.01, S(Z7, 1.01), Z7, Z7, plan_active=True) == "motion"
    assert m.target() is None and "planned motion" in m.detail
    moved = np.array([0.4, 0, 0, 0, 0, 0, 0])  # the arm went somewhere else meanwhile
    assert m.step(1.02, S(Z7, 1.02), moved, moved, plan_active=True) == "motion"
    assert m.max_lag_rad() == pytest.approx(0.4)  # deltas published during the motion
    assert m.step(1.03, S(Z7, 1.03), moved, moved) == "out_of_sync"  # arrived far from the leader
    assert m.step(1.04, S(moved, 1.04), moved, moved) == "tracking"  # operator brings GELLO back
    # via the explicit hooks
    m.on_motion_start()
    assert m.state == "motion"
    assert m.on_motion_end(moved) == "tracking"  # the last sample was within tolerance
    m.on_motion_start()
    assert m.on_motion_end(Z7) == "out_of_sync"  # arrived 0.4 rad from the leader
    # no sample seen -> no_leader at the end
    fresh = EngageMachine(CFG)
    fresh.on_motion_start()
    assert fresh.on_motion_end(Z7) == "no_leader"
    # a stale last sample -> no_leader too
    m4 = EngageMachine(CFG)
    m4.step(1.0, S(Z7, 1.0), Z7, Z7)
    m4.on_motion_start()
    m4.step(5.0, None, Z7, Z7, plan_active=True)  # time passes, no new sample
    assert m4.step(5.1, None, Z7, Z7) == "no_leader"


def test_pause_before_an_in_session_motion_leaves_it_paused_afterwards():
    """16-gello §5.3: R / Go to profile / the exit return force paused first; the state is
    paused after the plan retires, the operator presses Resume to re-engage."""
    m = EngageMachine(CFG)
    assert m.step(1.0, S(Z7, 1.0), Z7, Z7) == "tracking"
    m.pause("paused: planned motion (R)")
    assert m.state == "paused"
    m.on_motion_start()
    assert m.state == "motion"  # the display says motion while the plan runs
    assert m.step(1.01, S(Z7, 1.01), Z7, Z7, plan_active=True) == "motion"
    assert m.on_motion_end(Z7) == "paused" and m.paused  # sticky
    assert m.step(1.02, S(Z7, 1.02), Z7, Z7) == "paused"
    m.resume()
    assert m.step(1.03, S(Z7, 1.03), Z7, Z7) == "tracking"
    # pause() while a motion runs keeps the display at motion but latches
    m.on_motion_start()
    m.pause()
    assert m.state == "motion" and m.paused
    assert m.step(1.04, S(Z7, 1.04), Z7, Z7) == "paused"  # plan retired -> paused, not tracking
