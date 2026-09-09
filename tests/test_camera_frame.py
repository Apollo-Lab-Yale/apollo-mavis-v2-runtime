"""Keyboard camera translate frame against the REAL scene model (04-runtime §6;
2026-09-08 operator decision).

``tests/test_teleop_math.py`` pins the algebra with hand-built quaternions; this
file pins the one fact the algebra depends on and that no unit test can see: the
wrist camera's orientation as MuJoCo computes it. The trap it guards is that the
camera-to-TCP rotation is NOT the same on the two MAVIS arms — the gripper base
is mounted ``quat="0 0 0 1"`` (180° about the tool axis) under link7, so the
Manipulation Arm's TCP frame is that much turned against the Perception Arm's. A
single constant key->TCP matrix therefore sends ``A``/``D`` and ``E``/``Q`` the
WRONG WAY on the Manipulation Arm — the default teleop arm.
"""

from __future__ import annotations

import numpy as np
import pytest
from apollo_mavis_v2_core import Twist, se3

from apollo_mavis_v2_runtime.config import TeleopRates
from apollo_mavis_v2_runtime.control.teleop import held_to_twist, twist_to_control_frame

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("apollo_mavis_v2_sim")

R = TeleopRates()


@pytest.fixture(scope="module")
def cell():
    from apollo_mavis_v2_sim import REGISTRY

    from apollo_mavis_v2_runtime.control.fk import SceneKinematics

    scene = REGISTRY.build("mavis_v2", None)
    return scene, SceneKinematics(scene), mujoco.MjData(scene.model)


def camera_axes(scene, data, arm_id: str, q):
    """(forward, up, right) of the arm's wrist camera in WORLD, from the model."""
    a = scene.addressing[arm_id]
    data.qpos[a.qpos_adr] = np.asarray(q, dtype=np.float64)
    mujoco.mj_kinematics(scene.model, data)
    mujoco.mj_camlight(scene.model, data)  # cam_xmat
    cam = np.array(data.cam_xmat[a.wrist_cam_id]).reshape(3, 3)
    return -cam[:, 2], cam[:, 1], cam[:, 0]  # MuJoCo cameras look along −z, +y up


def angle_deg(u, v) -> float:
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    cos = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def key_direction(kin, arm_id: str, q, code: str) -> np.ndarray:
    """The WORLD direction one held translate key produces, exactly as the loop
    computes it (``ControlLoop._teleop_step`` keyboard branch)."""
    tcp = kin.tcp_world(arm_id, q)
    tw = held_to_twist(frozenset({code}), R)
    out = twist_to_control_frame(
        Twist(v=tw.v, w=tw.w),
        kin.base_quat_world(arm_id),
        tcp.orientation,
        "camera",
        kin.wrist_cam_quat_world(arm_id, tcp.orientation),
    )
    return np.asarray(out.v) / float(np.linalg.norm(out.v))


def postures(scene, arm_id: str, n: int = 25):
    """The keyframe plus pseudo-random joint perturbations of it (fixed seed)."""
    a = scene.addressing[arm_id]
    q0 = np.array(scene.model.key_qpos[0][a.qpos_adr], dtype=np.float64)
    rng = np.random.default_rng(20260908)
    yield q0
    for _ in range(n - 1):
        q = np.array(q0)
        q[:7] += rng.uniform(-1.0, 1.0, 7) * 0.8
        yield q


@pytest.mark.parametrize("arm_id", ["grip", "view"])
def test_translate_keys_follow_the_wrist_camera_axes(cell, arm_id):
    scene, kin, data = cell
    for q in postures(scene, arm_id):
        forward, up, right = camera_axes(scene, data, arm_id, q)
        for code, want in (
            ("KeyW", forward),
            ("KeyS", -forward),
            ("KeyA", -right),
            ("KeyD", right),
            ("KeyE", up),
            ("KeyQ", -up),
        ):
            got = key_direction(kin, arm_id, q, code)
            assert angle_deg(got, want) < 1e-3, (arm_id, code, got, want)


def test_the_two_arms_need_different_tcp_to_camera_rotations(cell):
    """Why the camera orientation is read from the model per arm instead of being a
    constant key->TCP matrix: the two arms' TCP frames differ by 180° about the tool
    axis, so the same constant is right for one arm and reversed for the other."""
    _scene, kin, _data = cell
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    rel = {arm: kin.wrist_cam_quat_world(arm, identity) for arm in ("grip", "view")}
    assert all(v is not None for v in rel.values())
    # Same optical axis relative to the TCP (the flange/tool axis on both arms) ...
    axis = {arm: se3.quat_rotate(q, [0.0, 0.0, -1.0]) for arm, q in rel.items()}
    assert angle_deg(axis["grip"], axis["view"]) < 1e-6
    # ... but image up/right are opposite: a 180° roll about that axis.
    up = {arm: se3.quat_rotate(q, [0.0, 1.0, 0.0]) for arm, q in rel.items()}
    assert abs(angle_deg(up["grip"], up["view"]) - 180.0) < 1e-3
    assert se3.quat_geodesic(rel["grip"], rel["view"]) == pytest.approx(np.pi, abs=1e-6)


def test_arm_without_a_wrist_camera_has_no_camera_quat():
    """``guardrail_env``'s arm carries no wrist camera: the getter says so and the
    twist falls back to the tool frame instead of raising (04-runtime §6)."""
    from apollo_mavis_v2_sim import REGISTRY

    from apollo_mavis_v2_runtime.control.fk import SceneKinematics

    scene = REGISTRY.build("guardrail_env", None)
    kin = SceneKinematics(scene)
    tcp_quat = np.array([1.0, 0.0, 0.0, 0.0])
    assert kin.wrist_cam_quat_world("arm0", tcp_quat) is None
    tw = held_to_twist(frozenset({"KeyW"}), R)
    v = twist_to_control_frame(
        Twist(v=tw.v, w=tw.w), kin.base_quat_world("arm0"), tcp_quat, "camera", None
    ).v
    assert np.allclose(v, [0.0, 0.0, R.linear_mps], atol=1e-12)  # forward = the tool axis
