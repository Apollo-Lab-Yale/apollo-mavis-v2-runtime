"""Unit tests of the control loop's per-tick step cap and the keyboard target pull-back
(04-runtime §6 "Per-tick joint step cap", 2026-09-09; operator report: "when I press
forward/back with the keyboard the arm also drifts up/down").

Pure behaviours over FakeWorkcell / fake kinematics, no sim, milliseconds:
``ControlLoop._cap_joint_step`` scales the WHOLE joint step by one factor (the larger
of the per-joint ratio against ``dq_max`` and the streamer's lever-weighted Cartesian
ratio against ``jog.plan_cart_step_m``), so the joint-space direction survives and every
bound holds; the rail slot keeps its own bound; a non-positive ``dq_max`` holds instead of
passing steps unbounded; ``clamp_ticks`` counts only the ticks the cap bound; and
``_hold_key_target`` moves the integrated keyboard target only along the commanded
translation direction / about the commanded rotation axis. The live-server measurement
of the same behaviour is ``tests/test_teleop_axis_purity.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from apollo_mavis_v2_core import Command, Pose, ProfileStore, Twist, se3
from apollo_mavis_v2_core.testing import FakeArm, FakeWorkcell
from conftest import run_ticks
from pydantic import ValidationError

from apollo_mavis_v2_runtime.bus import RuntimeBus
from apollo_mavis_v2_runtime.config import ControlConfig
from apollo_mavis_v2_runtime.control.joint_panel import RATIO_EPS
from apollo_mavis_v2_runtime.control.loop import RAIL_TRAVEL_M, ControlLoop
from apollo_mavis_v2_runtime.safety.gate import NullGate
from apollo_mavis_v2_runtime.safety.supervisor import SafetySupervisor
from apollo_mavis_v2_runtime.safety.watchdog import InputWatchdog
from apollo_mavis_v2_runtime.session.hardware import (
    ExecutorCaps,
    apply_executor_caps,
    apply_teleop_caps,
)

# The driver's ServoLimits at speed scale 1.0, as ExecutorCaps per 100 Hz loop tick
# (explicit numbers, not the [hardware] defaults: the arithmetic below is spelled out).
DQ_MAX = 0.006  # 0.6 rad/s / 100 Hz
CART = 0.004  # m per tick, lever-weighted
LEVER = (1.20, 1.20, 1.00, 0.75, 0.44, 0.30, 0.10)
IDENT = np.array([1.0, 0.0, 0.0, 0.0])


def hardware_cfg() -> ControlConfig:
    """What a hardware bring-up hands the loop at 100 %: dq_max lowered to the servo
    joint step, ``jog.plan_cart_step_m`` / ``plan_lever_arm_m`` set."""
    caps = ExecutorCaps(
        slew_rad_per_tick=DQ_MAX,
        cart_step_m=CART,
        lever_arm_m=LEVER,
        source="servo",
        joint_step_rad=DQ_MAX,
    )
    cfg = apply_teleop_caps(apply_executor_caps(ControlConfig(), caps), caps)
    assert cfg.dq_max_rad == pytest.approx(DQ_MAX) and cfg.jog.plan_cart_step_m == CART
    return cfg


class PoseKin:
    """q[:3] = TCP position, q[3:6] = TCP rotation vector, both in world."""

    def tcp_world(self, arm_id, q):
        return Pose(np.array(q[:3], dtype=float), se3.rotvec_to_quat(np.array(q[3:6], float)))

    def base_quat_world(self, arm_id):
        return IDENT


def make_loop(cfg: ControlConfig, tmp_path, *, kin=None):
    cell = FakeWorkcell({"arm0": FakeArm("arm0", has_rail=True), "arm1": FakeArm("arm1")})
    cell.start()
    bus = RuntimeBus()
    loop = ControlLoop(
        cell,
        cfg,
        bus,
        SafetySupervisor(NullGate(), InputWatchdog()),
        ["arm0", "arm1"],
        kin=kin,
        profile_store=ProfileStore(tmp_path / "profiles"),
        workcell_kind="sim",
    )
    return cell, bus, loop


def old_per_joint_clip(q, q_last, dq_max):
    """The pre-2026-09-09 rule: clip every joint independently."""
    q = np.array(q, dtype=np.float64)
    q[:7] = np.clip(q[:7], q_last[:7] - dq_max, q_last[:7] + dq_max)
    return q


def cosine(a, b) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def lever_sum(dq) -> float:
    return float(np.sum(np.abs(dq[:7]) * np.asarray(LEVER)))


# -- _cap_joint_step: per-joint bound ---------------------------------------------------------
def test_uniform_scaling_preserves_direction_and_bounds_every_joint(fake_loop):
    """Two joints over the cap by DIFFERENT ratios (2.0x and 1.5x): the whole step is
    divided by the largest ratio, exactly; the old per-joint clip bends the direction."""
    _, _, loop = fake_loop
    dq_max = loop.cfg.dq_max_rad  # 0.04 in the repo config
    q_last = np.array([0.3, -0.2, 0.1, 0.5, 0.0, 0.4, -0.1, 0.2])
    dq = dq_max * np.array([2.0, -1.5, 0.25, 0.0, 0.1, 0.0, 0.5])
    q = q_last.copy()
    q[:7] += dq
    out, capped = loop._cap_joint_step(q, q_last)
    assert capped
    dq_out = out[:7] - q_last[:7]
    np.testing.assert_allclose(dq_out, dq * (dq_max / np.max(np.abs(dq))), rtol=0, atol=1e-15)
    assert np.max(np.abs(dq_out)) == pytest.approx(dq_max, rel=1e-12)
    assert np.all(np.abs(dq_out) <= dq_max * (1.0 + 1e-12))
    assert cosine(dq_out, dq) == pytest.approx(1.0, abs=1e-12)
    assert out[7] == q_last[7]  # rail slot untouched
    # The old rule saturates joints 1 and 2 at +-dq_max and keeps the others whole.
    old = old_per_joint_clip(q, q_last, dq_max)[:7] - q_last[:7]
    assert cosine(old, dq) < 1.0 - 1e-3, "np.clip would have preserved the direction"
    assert not np.allclose(old, dq_out)


def test_step_within_the_cap_passes_untouched(fake_loop):
    _, _, loop = fake_loop
    dq_max = loop.cfg.dq_max_rad
    q_last = np.zeros(8)
    q = np.array([0.5 * dq_max, -0.9 * dq_max, 0.0, 0.1 * dq_max, 0.0, 0.0, 0.99 * dq_max, 0.0])
    out, capped = loop._cap_joint_step(q, q_last)
    assert not capped
    np.testing.assert_array_equal(out, q)


def test_a_step_exactly_at_the_cap_is_not_counted_as_capped(fake_loop):
    """A planned step the executor already bounded may land at the cap up to float
    drift (ratio 1 + 1e-16): it passes as is and does not count as a saturated tick."""
    _, _, loop = fake_loop
    dq_max = loop.cfg.dq_max_rad
    q_last = np.zeros(8)
    q = np.zeros(8)
    q[0] = dq_max * (1.0 + RATIO_EPS / 10.0)
    out, capped = loop._cap_joint_step(q, q_last)
    assert not capped
    np.testing.assert_array_equal(out, q)


def test_rail_slot_keeps_an_independent_bound(fake_loop):
    _, _, loop = fake_loop
    dq_max = loop.cfg.dq_max_rad
    q_last = np.array([0.0] * 7 + [0.30])
    # a big carriage step never slows the joints ...
    q = np.array([0.5 * dq_max] * 7 + [0.30 + 10.0 * dq_max])
    out, capped = loop._cap_joint_step(q, q_last)
    assert not capped  # capped is about the joints
    np.testing.assert_array_equal(out[:7], q[:7])
    assert out[7] == pytest.approx(0.30 + dq_max)
    # ... and a capped joint step never scales the carriage.
    q = np.array([3.0 * dq_max] * 7 + [0.30 + 0.5 * dq_max])
    out, capped = loop._cap_joint_step(q, q_last)
    assert capped
    assert out[7] == pytest.approx(0.30 + 0.5 * dq_max)
    # travel limits
    q = np.array([0.0] * 7 + [-1.0])
    out, _ = loop._cap_joint_step(q, np.array([0.0] * 7 + [0.01]))
    assert out[7] == 0.0
    q = np.array([0.0] * 7 + [RAIL_TRAVEL_M + 1.0])
    out, _ = loop._cap_joint_step(q, np.array([0.0] * 7 + [RAIL_TRAVEL_M - 0.001]))
    assert out[7] == RAIL_TRAVEL_M


def test_seven_slot_arm_has_no_rail_slot_to_bound(fake_loop):
    _, _, loop = fake_loop
    dq_max = loop.cfg.dq_max_rad
    q_last = np.zeros(7)
    q = np.full(7, 2.0 * dq_max)
    out, capped = loop._cap_joint_step(q, q_last)
    assert capped and out.shape == (7,)
    np.testing.assert_allclose(out, dq_max, rtol=1e-12)


# -- _cap_joint_step: the streamer's lever-weighted Cartesian bound (hardware loops) ----------
def test_lever_weighted_cartesian_bound_scales_uniformly(tmp_path):
    """Every joint under dq_max, yet the streamer's estimate sum|dq_j| * lever_j is
    ~25 mm (0.005 rad x 4.99 m) against the 4 mm cap: the whole step is divided by
    that ratio (6.24) so the streamer's own scaling - and its per-joint clip after
    it - never has to act."""
    _, _, loop = make_loop(hardware_cfg(), tmp_path)
    q_last = np.array([0.3, -0.2, 0.1, 0.5, 0.0, 0.4, -0.1, 0.2])
    dq = np.array([0.005, -0.005, 0.005, 0.005, -0.005, 0.005, 0.005])
    assert np.all(np.abs(dq) < DQ_MAX)
    ratio = lever_sum(dq) / CART
    assert ratio == pytest.approx(6.2375)
    q = q_last.copy()
    q[:7] += dq
    out, capped = loop._cap_joint_step(q, q_last)
    assert capped
    dq_out = out[:7] - q_last[:7]
    np.testing.assert_allclose(dq_out, dq / ratio, rtol=1e-12)
    assert lever_sum(dq_out) == pytest.approx(CART, rel=1e-12)
    assert cosine(dq_out, dq) == pytest.approx(1.0, abs=1e-12)
    assert out[7] == q_last[7]


def test_the_larger_of_the_joint_and_cartesian_ratios_wins(tmp_path):
    _, _, loop = make_loop(hardware_cfg(), tmp_path)
    q_last = np.zeros(8)
    # joint ratio dominates: joint 7 (lever 0.10 m) asks 2x dq_max, lever sum 1.2 mm
    q = np.zeros(8)
    q[6] = 2.0 * DQ_MAX
    out, capped = loop._cap_joint_step(q, q_last)
    assert capped and out[6] == pytest.approx(DQ_MAX, rel=1e-12)
    # Cartesian ratio dominates: joints 1+2 at 5/6 of dq_max, lever sum 12 mm -> /3
    q = np.zeros(8)
    q[0] = q[1] = 0.005
    out, capped = loop._cap_joint_step(q, q_last)
    assert capped
    np.testing.assert_allclose(out[:2], 0.005 / 3.0, rtol=1e-12)
    assert lever_sum(out[:7]) == pytest.approx(CART, rel=1e-12)
    # neither binds
    q = np.zeros(8)
    q[6] = 0.9 * DQ_MAX  # lever sum 0.54 mm
    out, capped = loop._cap_joint_step(q, q_last)
    assert not capped and out[6] == q[6]


def test_sim_config_has_no_cartesian_term(fake_loop):
    """No driver published a Cartesian bound (``jog.plan_cart_step_m`` None): only
    ``dq_max`` applies, even for a step whose lever-weighted sum would be 25 mm."""
    _, _, loop = fake_loop
    assert loop.cfg.jog.plan_cart_step_m is None
    q_last = np.zeros(8)
    q = np.array([0.005] * 7 + [0.0])
    out, capped = loop._cap_joint_step(q, q_last)
    assert not capped
    np.testing.assert_array_equal(out, q)


def test_cartesian_bound_follows_a_swapped_config(fake_loop):
    """The bound is read from ``loop.cfg.jog`` (cached per config object), so a loop
    handed a hardware config after construction applies it."""
    _, _, loop = fake_loop
    q_last = np.zeros(8)
    q = np.array([0.005] * 7 + [0.0])
    assert not loop._cap_joint_step(q, q_last)[1]
    loop.cfg = hardware_cfg()
    out, capped = loop._cap_joint_step(q, q_last)
    assert capped and lever_sum(out[:7]) == pytest.approx(CART, rel=1e-12)


# -- fail-safe direction -----------------------------------------------------------------------
def test_non_positive_dq_max_holds_instead_of_passing_unbounded(fake_loop):
    """``dq_max <= 0`` is a mis-derived cap; the old np.clip froze the joints, and so
    does the uniform rule (the rail too) - never a step that passes unbounded."""
    _, _, loop = fake_loop
    for bad in (0.0, -0.01):
        loop.cfg = ControlConfig().model_copy(update={"dq_max_rad": bad})  # bypasses gt=0
        q_last = np.array([0.1] * 7 + [0.2])
        q = np.array([0.5] * 7 + [0.3])
        out, capped = loop._cap_joint_step(q, q_last)
        assert capped
        np.testing.assert_array_equal(out, q_last)
        out, capped = loop._cap_joint_step(q_last.copy(), q_last)
        assert not capped and np.array_equal(out, q_last)


def test_control_config_rejects_a_non_positive_cap():
    with pytest.raises(ValidationError):
        ControlConfig(dq_max_rad=0.0)
    with pytest.raises(ValidationError):
        ControlConfig(dq_max_rad=-0.004)


# -- clamp_ticks (the health line's dq_capped) --------------------------------------------------
def test_clamp_ticks_counts_only_the_ticks_the_cap_bound(tmp_path):
    """A jog slewing 0.02 rad/tick against dq_max 0.01: every tick on the way is
    capped (and walked at 0.01, direction intact), the ticks after arrival are not."""
    cfg = ControlConfig(dq_max_rad=0.01)
    cell, bus, loop = make_loop(cfg, tmp_path)
    goal = [0.1, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1] + [0.05]
    fut = bus.commands.submit(
        Command(op="joint_target", args={"arm_id": "arm0", "positions": goal, "mode": "jog"})
    )
    t = run_ticks(loop, cell, 1)
    assert fut.result(0).ok
    q1 = loop._last_cmd["arm0"]
    # the jog asked 0.02 on joint 1 (its slew) and 0.01 on joint 2 (its whole delta);
    # the cap divides the WHOLE step by 2: 0.01 and 0.005, not 0.01 and 0.01
    assert q1[0] == pytest.approx(0.01, abs=1e-12)
    assert q1[1] == pytest.approx(0.005, abs=1e-12)
    assert loop.clamp_ticks == 1
    t = run_ticks(loop, cell, 40, t)  # 0.1 rad at 0.01/tick = 10 ticks, then idle
    assert np.allclose(loop._last_cmd["arm0"][:7], goal[:7])
    capped_total = loop.clamp_ticks
    # 10 steps of 0.01 on joint 1; the 10th asks exactly the cap and is NOT counted
    assert capped_total == 9
    run_ticks(loop, cell, 20, t)
    assert loop.clamp_ticks == capped_total  # idle ticks never count


# -- _hold_key_target: keyboard target pull-back -----------------------------------------------
def test_hold_key_target_pulls_back_only_along_the_commanded_translation(tmp_path):
    _, _, loop = make_loop(ControlConfig(), tmp_path, kin=PoseKin())
    target = Pose(np.array([0.10, 0.20, 0.30]), IDENT)
    loop.integrator.seed("arm0", target)
    d = np.array([0.0, -1.0, 0.0])  # W in the world frame: -y
    loop._key_twist["arm0"] = Twist(v=0.12 * d, w=np.zeros(3))
    # the capped step reached 0.03 m short along -y AND sits 5 mm off in x, 7 mm in z
    achieved = np.array([0.105, 0.23, 0.293, 0.0, 0.0, 0.0, 0.0, 0.0])
    loop._hold_key_target("arm0", achieved)
    new = loop.integrator.get("arm0")
    np.testing.assert_allclose(new.position, [0.10, 0.23, 0.30], atol=1e-15)  # y only
    np.testing.assert_array_equal(new.orientation, IDENT)  # no rotation commanded


def test_hold_key_target_pulls_back_only_about_the_commanded_rotation_axis(tmp_path):
    _, _, loop = make_loop(ControlConfig(), tmp_path, kin=PoseKin())
    target = Pose(np.array([0.10, 0.20, 0.30]), IDENT)
    loop.integrator.seed("arm0", target)
    u = np.array([0.0, 0.0, 1.0])  # yaw about world z
    loop._key_twist["arm0"] = Twist(v=np.zeros(3), w=0.6 * u)
    achieved = np.zeros(8)
    achieved[:3] = [0.12, 0.19, 0.31]  # position off: NOT driven, must stay pinned
    achieved[3:6] = [0.02, -0.01, 0.30]  # 0.30 rad about z reached, plus off-axis tilt
    loop._hold_key_target("arm0", achieved)
    new = loop.integrator.get("arm0")
    np.testing.assert_array_equal(new.position, target.position)
    np.testing.assert_allclose(
        se3.quat_to_rotvec(new.orientation), [0.0, 0.0, 0.30], atol=1e-12
    )  # the z component of the achieved rotation only


def test_hold_key_target_is_a_no_op_without_a_keyboard_twist(tmp_path):
    _, _, loop = make_loop(ControlConfig(), tmp_path, kin=PoseKin())
    target = Pose(np.array([0.10, 0.20, 0.30]), IDENT)
    loop.integrator.seed("arm0", target)
    achieved = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.0, 0.0])
    loop._hold_key_target("arm0", achieved)  # tracker / jog / plan tick: nothing recorded
    assert loop.integrator.get("arm0") is target
    loop._key_twist["arm1"] = Twist(v=np.array([1.0, 0.0, 0.0]), w=np.zeros(3))
    loop._hold_key_target("arm1", achieved)  # never seeded: nothing to pull back
    assert loop.integrator.get("arm1") is None
    loop.kin = None
    loop._key_twist["arm0"] = Twist(v=np.array([1.0, 0.0, 0.0]), w=np.zeros(3))
    loop._hold_key_target("arm0", achieved)
    assert loop.integrator.get("arm0") is target


def test_key_twist_is_cleared_every_tick(fake_loop):
    cell, _, loop = fake_loop
    loop._key_twist["arm0"] = Twist(v=np.array([1.0, 0.0, 0.0]), w=np.zeros(3))
    run_ticks(loop, cell, 1)
    assert loop._key_twist == {}
