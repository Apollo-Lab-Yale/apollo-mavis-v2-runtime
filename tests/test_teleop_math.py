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


def test_base_frame_translation_along_base_axes():
    """``frame="base"`` = the pre-2026-09-08 behaviour, kept as a config option."""
    tw = held_to_twist(frozenset({"KeyW"}), R)
    # Base rotated 180 deg about Z: +x key must move along world -x.
    base_quat = np.array([0.0, 0.0, 0.0, 1.0])
    tcp_quat = np.array([1.0, 0.0, 0.0, 0.0])
    out = twist_to_control_frame(tw, base_quat, tcp_quat, "base")
    assert np.allclose(out.v, [-R.linear_mps, 0, 0], atol=1e-12)


def test_camera_frame_translation_maps_keys_onto_the_camera_axes():
    """``frame="camera"`` (the 2026-09-08 morning default; ``world`` took over the same
    evening): with the camera's WORLD orientation supplied, W runs
    along the optical axis (camera −z), A/D are image left/right (camera ∓x) and E/Q
    image up/down (camera ±y). The arm base is irrelevant, so a deliberately absurd
    ``base_quat`` must not change anything, and neither must the TCP orientation —
    the camera quaternion alone decides (the gripper arm's TCP frame is turned 180°
    about the tool axis against the camera-only arm's, verified against the model)."""
    base_quat = se3.rpy_to_quat(np.array([0.3, -0.7, 1.1]))
    tcp_quat = se3.rpy_to_quat(np.array([1.0, 2.0, 3.0]))
    cam_quat = np.array([1.0, 0.0, 0.0, 0.0])  # camera axes == world axes
    v = R.linear_mps

    def key(code):
        return twist_to_control_frame(
            held_to_twist(frozenset({code}), R), base_quat, tcp_quat, "camera", cam_quat
        ).v

    assert np.allclose(key("KeyW"), [0, 0, -v], atol=1e-12)  # forward = camera −z
    assert np.allclose(key("KeyS"), [0, 0, +v], atol=1e-12)
    assert np.allclose(key("KeyA"), [-v, 0, 0], atol=1e-12)  # image left = camera −x
    assert np.allclose(key("KeyD"), [+v, 0, 0], atol=1e-12)
    assert np.allclose(key("KeyE"), [0, +v, 0], atol=1e-12)  # image up = camera +y
    assert np.allclose(key("KeyQ"), [0, -v, 0], atol=1e-12)


def test_camera_frame_translation_rotates_with_the_camera():
    """The bug this frame fixes: with the tool rotated, the keys must still line up
    with the wrist image, i.e. the world direction of W turns WITH the camera."""
    tw = held_to_twist(frozenset({"KeyW"}), R)
    base_quat = np.array([1.0, 0.0, 0.0, 0.0])
    tcp_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def w_dir(cam_quat):
        return twist_to_control_frame(tw, base_quat, tcp_quat, "camera", cam_quat).v

    upright = w_dir(np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(upright, [0, 0, -R.linear_mps], atol=1e-12)
    # Pitch the camera 90 deg about world x: its −z axis swings to +y.
    assert np.allclose(
        w_dir(se3.rpy_to_quat(np.array([np.pi / 2, 0.0, 0.0]))),
        [0, R.linear_mps, 0],
        atol=1e-9,
    )
    # A roll ABOUT the optical axis only rolls the image: W must not change.
    assert np.allclose(w_dir(se3.rpy_to_quat(np.array([0.0, 0.0, np.pi / 2]))), upright, atol=1e-9)


def test_camera_frame_falls_back_to_the_tool_axis_without_a_wrist_camera():
    """``cam_quat is None`` (an arm with no wrist camera — some test scenes): same
    "follow the tool" behaviour, expressed in the TCP frame with forward = TCP +z."""
    base_quat = se3.rpy_to_quat(np.array([0.3, -0.7, 1.1]))
    tcp_quat = np.array([1.0, 0.0, 0.0, 0.0])
    v = R.linear_mps

    def key(code):
        return twist_to_control_frame(
            held_to_twist(frozenset({code}), R), base_quat, tcp_quat, "camera", None
        ).v

    assert np.allclose(key("KeyW"), [0, 0, +v], atol=1e-12)  # forward = the tool axis
    assert np.allclose(key("KeyA"), [0, -v, 0], atol=1e-12)
    assert np.allclose(key("KeyE"), [+v, 0, 0], atol=1e-12)


def test_world_frame_translation_is_fixed_to_the_operator():
    """``frame="world"`` (the DEFAULT since 2026-09-08 evening): W away from the
    operator (-Y), A to the operator's left (+X), E up (+Z) — whatever the tool and
    the base are doing (mavis_v2 header)."""
    base_quat = se3.rpy_to_quat(np.array([0.3, -0.7, 1.1]))
    v = R.linear_mps
    for tcp_quat in (np.array([1.0, 0.0, 0.0, 0.0]), se3.rpy_to_quat(np.array([1.0, 2.0, 3.0]))):

        def key(code, tcp_quat=tcp_quat):
            return twist_to_control_frame(
                held_to_twist(frozenset({code}), R), base_quat, tcp_quat, "world"
            ).v

        assert np.allclose(key("KeyW"), [0, -v, 0], atol=1e-12)
        assert np.allclose(key("KeyA"), [+v, 0, 0], atol=1e-12)
        assert np.allclose(key("KeyD"), [-v, 0, 0], atol=1e-12)
        assert np.allclose(key("KeyE"), [0, 0, +v], atol=1e-12)


def test_every_frame_keeps_rotations_about_the_tcp_axes_and_passes_rail_grip():
    tw = held_to_twist(frozenset({"KeyI", "ArrowRight", "KeyH"}), R)
    base_quat = se3.rpy_to_quat(np.array([0.0, 0.0, 0.9]))
    tcp_quat = se3.rpy_to_quat(np.array([0.0, 0.0, np.pi / 2]))  # TCP yawed 90
    for frame in ("camera", "world", "base"):
        out = twist_to_control_frame(tw, base_quat, tcp_quat, frame)
        assert np.allclose(out.w, [0, R.angular_rps, 0], atol=1e-9), frame  # roll+ about TCP x
        assert out.rail_v == R.rail_mps and out.grip_v == R.gripper_frac_ps


def test_rotation_about_tcp_axes():
    tw = held_to_twist(frozenset({"KeyI"}), R)  # roll+ about TCP x
    base_quat = np.array([1.0, 0.0, 0.0, 0.0])
    tcp_quat = se3.rpy_to_quat(np.array([0.0, 0.0, np.pi / 2]))  # TCP yawed 90
    out = twist_to_control_frame(tw, base_quat, tcp_quat)
    assert np.allclose(out.w, [0, R.angular_rps, 0], atol=1e-9)
    # A rotation-only twist never leaks into translation, in any frame.
    assert np.allclose(out.v, 0.0, atol=1e-12)


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
