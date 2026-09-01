"""RecorderKinematics over a real scene: FK consistency, OpenCV camera
convention (10-frames §2.3), static-camera validation for recording frames."""

from __future__ import annotations

import numpy as np
import pytest
from apollo_xarm7_core import se3

from apollo_xarm7_runtime.recorder.kinematics import RecorderKinematics


@pytest.fixture(scope="module")
def scene():
    from apollo_xarm7_sim import REGISTRY

    return REGISTRY.build("single_rail")


@pytest.fixture(scope="module")
def kin(scene):
    return RecorderKinematics(scene)


Q = np.array([0.2, -0.5, 0.1, 0.9, 0.0, 1.4, -0.2, 0.25])


def test_base_world_translates_with_rail_only(kin):
    b0 = kin.base_world("arm0", np.array([*Q[:7], 0.0]))
    b1 = kin.base_world("arm0", np.array([*Q[:7], 0.4]))
    assert np.allclose(b0.orientation, b1.orientation, atol=1e-12)  # rotation constant
    assert np.linalg.norm(b1.position - b0.position) == pytest.approx(0.4, abs=1e-9)


def test_tcp_base_composes_to_world_site(kin, scene):
    from apollo_xarm7_runtime.control.fk import SceneKinematics

    world = SceneKinematics(scene).tcp_world("arm0", Q)
    composed = se3.pose_mul(kin.base_world("arm0", Q), kin.tcp_base("arm0", Q))
    pos_err, rot_err = se3.pose_error(composed, world)
    assert pos_err < 1e-9 and rot_err < 1e-9


def test_camera_staticness(kin):
    assert kin.camera_static("cam_front") is True  # worldbody camera
    assert kin.camera_static("arm0_wrist_cam") is False  # rides the arm
    with pytest.raises(KeyError):
        kin.camera_static("nope")


def test_camera_world_is_opencv_convention(kin, scene):
    """R_cv = R_mj @ R_x(pi): +Z looks INTO the scene, +Y down in the image."""
    import mujoco

    pose = kin.camera_world("cam_front", {"arm0": Q})
    data = mujoco.MjData(scene.model)
    mujoco.mj_kinematics(scene.model, data)
    mujoco.mj_camlight(scene.model, data)
    cid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_front")
    r_mj = np.array(data.cam_xmat[cid]).reshape(3, 3)
    r_cv = se3.quat_to_mat(pose.orientation)
    assert np.allclose(pose.position, data.cam_xpos[cid], atol=1e-12)
    assert np.allclose(r_cv[:, 2], -r_mj[:, 2], atol=1e-9)  # optical axis flips
    assert np.allclose(r_cv[:, 0], r_mj[:, 0], atol=1e-9)  # +X (right) unchanged


def test_collect_session_rejects_moving_camera_frame(tmp_path):
    """camera:<wrist> as a recording frame -> SessionError before any lerobot
    import (10-frames §5.1: T_W_C must be static over the session)."""
    from apollo_xarm7_core import ProfileStore
    from apollo_xarm7_core.protocol import SessionSpec
    from conftest import make_runtime_config

    from apollo_xarm7_runtime.bus import RuntimeBus
    from apollo_xarm7_runtime.errors import SessionError
    from apollo_xarm7_runtime.session.manager import SessionManager
    from apollo_xarm7_runtime.streams.hub import VideoHub

    cfg = make_runtime_config(tmp_path)
    bus = RuntimeBus()
    manager = SessionManager(
        cfg, bus, VideoHub(bus), ProfileStore(tmp_path / "profiles"), "epoch"
    )
    spec = SessionSpec(
        mode="collect", kind="sim", arms=["arm0"],
        frames={"arm0": "camera:arm0_wrist_cam"},
        sim_scene="single_rail", task="t",
    )
    try:
        with pytest.raises(SessionError, match="not static"):
            manager.create(spec)
        assert manager.session is None  # bringup cleaned up after itself
    finally:
        manager.stop_previews()  # bringup failure re-armed previews (EGL thread)
