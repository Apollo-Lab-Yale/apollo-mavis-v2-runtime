"""Record-frame conversion math vs 10-frames-and-data.md §3 + §10 (binding)."""

from __future__ import annotations

import numpy as np
import pytest
from apollo_xarm7_core import Pose, se3

from apollo_xarm7_runtime.recorder.frames import (
    RecordingFrameConverter,
    delta_from_frame,
    delta_to_frame,
    pose_from_frame,
    pose_to_frame,
)

S2 = 1.0 / np.sqrt(2.0)

# Worked numeric example (10-frames §10): 2-arm mixed frames.
T_W_RO1 = Pose(np.array([1.20, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))  # Rz(pi)
T_W_C = Pose(np.array([0.60, -1.00, 0.80]), np.array([S2, -S2, 0.0, 0.0]))  # Rx(-90)


def rail_base(t_w_ro: Pose, d: float) -> Pose:
    """T_W_B(t) = T_W_RO ⊕ Trans(0, d, 0) (10-frames §2.2)."""
    return se3.pose_mul(t_w_ro, Pose(np.array([0.0, d, 0.0]), np.array([1.0, 0, 0, 0])))


@pytest.fixture
def arm1_ctx():
    t_w_b1 = rail_base(T_W_RO1, 0.10)
    ee_b1 = Pose(np.array([0.40, 0.10, 0.30]), np.array([0.0, 1.0, 0.0, 0.0]))
    return t_w_b1, ee_b1


def test_worked_example_base_pose_from_rail(arm1_ctx):
    t_w_b1, _ = arm1_ctx
    assert np.allclose(t_w_b1.position, [1.20, -0.10, 0.0], atol=1e-12)
    assert np.allclose(t_w_b1.orientation, [0.0, 0.0, 0.0, 1.0], atol=1e-12)


def test_worked_example_pose_base_to_world_to_camera(arm1_ctx):
    t_w_b1, ee_b1 = arm1_ctx
    ee_w = pose_to_frame(ee_b1, t_w_b1, None)
    assert np.allclose(ee_w.position, [0.80, -0.20, 0.30], atol=1e-12)
    assert np.allclose(ee_w.orientation, [0.0, 0.0, 1.0, 0.0], atol=1e-12)
    ee_c = pose_to_frame(ee_b1, t_w_b1, T_W_C)
    assert np.allclose(ee_c.position, [0.20, 0.50, 0.80], atol=1e-9)
    assert np.allclose(ee_c.orientation, [0.0, 0.0, S2, S2], atol=1e-9)


def test_worked_example_delta_rotation_only(arm1_ctx):
    t_w_b1, _ = arm1_ctx
    dp_b = np.array([0.002, 0.0, 0.0])
    dr_b = np.array([0.0, 0.0, 0.010])
    dp_c, dr_c = delta_to_frame(dp_b, dr_b, t_w_b1.orientation, T_W_C.orientation)
    assert np.allclose(dp_c, [-0.002, 0.0, 0.0], atol=1e-12)
    assert np.allclose(dr_c, [0.0, -0.010, 0.0], atol=1e-12)


def test_worked_example_arm0_identity_passthrough():
    conv = RecordingFrameConverter({"arm0": "arm_base:arm0"})
    ee_b0 = Pose(np.array([0.50, 0.0, 0.35]), np.array([0.0, 1.0, 0.0, 0.0]))
    t_w_b0 = rail_base(Pose.identity(), 0.30)
    out = conv.convert_pose("arm0", ee_b0, t_w_b0)
    assert np.allclose(out.position, ee_b0.position)
    assert np.allclose(out.orientation, ee_b0.orientation)
    dp, dr = conv.convert_delta(
        "arm0", np.array([0.002, 0, 0]), np.zeros(3), t_w_b0.orientation
    )
    assert np.allclose(dp, [0.002, 0, 0]) and np.allclose(dr, 0.0)


def _random_pose(rng) -> Pose:
    q = se3.quat_normalize(rng.normal(size=4))
    return Pose(rng.normal(size=3), q)


def test_roundtrip_identity_pose_and_delta():
    """§3.5: to/from roundtrip < 1e-9 (pos m, rot rad)."""
    rng = np.random.default_rng(7)
    for _ in range(25):
        t_w_b, t_w_f, x_b = _random_pose(rng), _random_pose(rng), _random_pose(rng)
        for f in (t_w_f, None):
            x_f = pose_to_frame(x_b, t_w_b, f)
            back = pose_from_frame(x_f, t_w_b, f)
            pos_err, rot_err = se3.pose_error(back, x_b)
            assert pos_err < 1e-9 and rot_err < 1e-9
        dp, dr = rng.normal(size=3) * 0.01, rng.normal(size=3) * 0.01
        q_f = t_w_f.orientation
        dp_f, dr_f = delta_to_frame(dp, dr, t_w_b.orientation, q_f)
        dp_b, dr_b = delta_from_frame(dp_f, dr_f, t_w_b.orientation, q_f)
        assert np.allclose(dp_b, dp, atol=1e-12) and np.allclose(dr_b, dr, atol=1e-12)


def _apply_delta(pose: Pose, dp: np.ndarray, dr: np.ndarray) -> Pose:
    """§3.2 left/space composition: p' = p + δp; q' = exp(δr) ⊗ q."""
    return Pose(
        pose.position + dp,
        se3.quat_normalize(se3.quat_mul(se3.rotvec_to_quat(dr), pose.orientation)),
    )


def test_delta_consistency_across_frames():
    """§3.5: apply(Δ_A, pose_A) converted to B == apply(Δ_B, pose_B)."""
    rng = np.random.default_rng(11)
    for _ in range(25):
        t_w_b, t_w_f, x_b = _random_pose(rng), _random_pose(rng), _random_pose(rng)
        dp_b, dr_b = rng.normal(size=3) * 0.02, rng.normal(size=3) * 0.02
        applied_b = _apply_delta(x_b, dp_b, dr_b)
        lhs = pose_to_frame(applied_b, t_w_b, t_w_f)
        x_f = pose_to_frame(x_b, t_w_b, t_w_f)
        dp_f, dr_f = delta_to_frame(dp_b, dr_b, t_w_b.orientation, t_w_f.orientation)
        rhs = _apply_delta(x_f, dp_f, dr_f)
        pos_err, rot_err = se3.pose_error(lhs, rhs)
        assert pos_err < 1e-9 and rot_err < 1e-9


def test_rail_invariance_of_world_z():
    """§3.5: world ee.z of a railed arm invariant under rail motion (rail ⊥ z)."""
    ee_b = Pose(np.array([0.4, 0.1, 0.3]), np.array([0.0, 1.0, 0.0, 0.0]))
    z = [
        pose_to_frame(ee_b, rail_base(T_W_RO1, d), None).position[2]
        for d in (0.0, 0.2, 0.65)
    ]
    assert np.allclose(z, z[0], atol=1e-12)


def test_converter_rejects_bad_frames():
    with pytest.raises(ValueError, match="another"):
        RecordingFrameConverter({"arm0": "arm_base:arm1"})
    with pytest.raises(ValueError, match="T_W_C"):
        RecordingFrameConverter({"arm0": "camera:cam_env"})  # no pose given
    conv = RecordingFrameConverter({"arm0": "camera:cam_env"}, {"cam_env": T_W_C})
    assert conv.frame_of("arm0") == "camera:cam_env"


def test_converted_quat_is_canonical_w_nonneg():
    rng = np.random.default_rng(3)
    conv = RecordingFrameConverter({"a": "world"})
    for _ in range(20):
        out = conv.convert_pose("a", _random_pose(rng), _random_pose(rng))
        assert out.orientation[0] >= 0.0
