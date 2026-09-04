"""Teleop math: keymap signs, twist frames, leash clamps (04-runtime §6)."""

from __future__ import annotations

import numpy as np
from apollo_mavis_v2_core import Pose, se3

from apollo_mavis_v2_runtime.config import TeleopRates
from apollo_mavis_v2_runtime.control.teleop import (
    TargetIntegrator,
    held_to_twist,
    twist_to_control_frame,
)

R = TeleopRates()


def test_translate_signs_match_keymap():
    for code, axis, sign in [
        ("KeyW", 0, +1), ("KeyS", 0, -1), ("KeyA", 1, +1),
        ("KeyD", 1, -1), ("KeyE", 2, +1), ("KeyQ", 2, -1),
    ]:
        tw = held_to_twist(frozenset({code}), R)
        expect = np.zeros(3)
        expect[axis] = sign * R.linear_mps
        assert np.allclose(tw.v, expect), code
        assert np.allclose(tw.w, 0)


def test_rotate_gripper_rail_channels():
    tw = held_to_twist(frozenset({"KeyI", "KeyJ", "KeyU"}), R)
    assert np.allclose(tw.w, [R.angular_rps] * 3)  # roll+, pitch+, yaw+
    assert held_to_twist(frozenset({"KeyK"}), R).w[0] == -R.angular_rps
    assert held_to_twist(frozenset({"KeyH"}), R).grip_v == R.gripper_frac_ps
    assert held_to_twist(frozenset({"KeyF"}), R).grip_v == -R.gripper_frac_ps
    assert held_to_twist(frozenset({"ArrowRight"}), R).rail_v == R.rail_mps
    assert held_to_twist(frozenset({"ArrowLeft"}), R).rail_v == -R.rail_mps


def test_opposing_keys_cancel_and_unknown_ignored():
    tw = held_to_twist(frozenset({"KeyW", "KeyS", "F5", "Space"}), R)
    assert np.allclose(tw.v, 0) and np.allclose(tw.w, 0)


def test_control_frame_translation_along_base_axes():
    tw = held_to_twist(frozenset({"KeyW"}), R)
    # Base rotated 180 deg about Z: +x key must move along world -x.
    base_quat = np.array([0.0, 0.0, 0.0, 1.0])
    tcp_quat = np.array([1.0, 0.0, 0.0, 0.0])
    out = twist_to_control_frame(tw, base_quat, tcp_quat)
    assert np.allclose(out.v, [-R.linear_mps, 0, 0], atol=1e-12)


def test_rotation_about_tcp_axes():
    tw = held_to_twist(frozenset({"KeyI"}), R)  # roll+ about TCP x
    base_quat = np.array([1.0, 0.0, 0.0, 0.0])
    tcp_quat = se3.rpy_to_quat(np.array([0.0, 0.0, np.pi / 2]))  # TCP yawed 90
    out = twist_to_control_frame(tw, base_quat, tcp_quat)
    assert np.allclose(out.w, [0, R.angular_rps, 0], atol=1e-9)


def test_leash_clamps_position_and_rotation():
    integ = TargetIntegrator(leash_pos_m=0.025, leash_rot_rad=0.2)
    anchor = Pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    tw = held_to_twist(frozenset({"KeyW", "KeyU"}), R)
    target = anchor
    for _ in range(200):  # 2 s of holding: target may not run away
        integ.reanchor("a", target)
        target = integ.step("a", tw, 0.01, anchor)
    assert np.linalg.norm(target.position - anchor.position) <= 0.025 + 1e-9
    assert se3.quat_geodesic(anchor.orientation, target.orientation) <= 0.2 + 1e-9


def test_reanchor_freezes_target():
    integ = TargetIntegrator()
    pose = Pose(np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 0.0, 0.0]))
    integ.reanchor("a", pose)
    got = integ.get("a")
    assert np.allclose(got.position, pose.position)
