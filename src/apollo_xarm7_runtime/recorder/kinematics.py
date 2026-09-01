"""RecorderKinematics — FK the recorder thread owns (sim extra only).

A private ``MjData`` (one thread per MjData; the model is shared read-only)
gives the recorder base-in-world / TCP-in-base poses at arbitrary q and
static camera world poses in the **OpenCV convention**: MuJoCo cameras look
along −Z with +Y up, so the MJCF orientation is post-multiplied by
``R_x(π)`` (quat ``(0,1,0,0)``) — 10-frames §2.3.
"""

from __future__ import annotations

import mujoco
import numpy as np
from apollo_xarm7_core import Pose, se3

_R_X_PI = np.array([0.0, 1.0, 0.0, 0.0])  # OpenCV fix-up quat


class RecorderKinematics:
    """FK helper for record-time frame conversion + extrinsics snapshots."""

    def __init__(self, scene) -> None:  # scene: apollo_xarm7_sim BuiltScene
        self.model = scene.model
        self.data = mujoco.MjData(scene.model)
        self.addr = scene.addressing
        mujoco.mj_kinematics(self.model, self.data)

    def _fk(self, arm_id: str, q: np.ndarray) -> None:
        a = self.addr[arm_id]
        self.data.qpos[a.qpos_adr] = np.asarray(q, dtype=np.float64)
        mujoco.mj_kinematics(self.model, self.data)

    def base_world(self, arm_id: str, q: np.ndarray) -> Pose:
        """``T_W_B`` at config q (railed arms: translates with q[7])."""
        self._fk(arm_id, q)
        a = self.addr[arm_id]
        pos = np.array(self.data.xpos[a.base_body_id])
        quat = se3.mat_to_quat(np.array(self.data.xmat[a.base_body_id]).reshape(3, 3))
        return Pose(pos, quat)

    def tcp_base(self, arm_id: str, q: np.ndarray) -> Pose:
        """TCP pose in the arm_base frame at config q (matches ArmState.ee_pose)."""
        self._fk(arm_id, q)
        a = self.addr[arm_id]
        p_site = self.data.site_xpos[a.tcp_site_id]
        r_site = self.data.site_xmat[a.tcp_site_id].reshape(3, 3)
        p_base = self.data.xpos[a.base_body_id]
        r_base = self.data.xmat[a.base_body_id].reshape(3, 3)
        return Pose(r_base.T @ (p_site - p_base), se3.mat_to_quat(r_base.T @ r_site))

    # -- cameras ------------------------------------------------------------------
    def _cam_id(self, camera_id: str) -> int:
        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id)
        if cid < 0:
            raise KeyError(f"no MJCF camera {camera_id!r}")
        return cid

    def camera_static(self, camera_id: str) -> bool:
        """True iff the camera is worldbody-attached (constant ``T_W_C``)."""
        return int(self.model.cam_bodyid[self._cam_id(camera_id)]) == 0

    def camera_world(
        self, camera_id: str, q_by_arm: dict[str, np.ndarray] | None = None
    ) -> Pose:
        """``T_W_C`` (OpenCV convention) at the given arm configs."""
        for arm_id, q in (q_by_arm or {}).items():
            a = self.addr[arm_id]
            self.data.qpos[a.qpos_adr] = np.asarray(q, dtype=np.float64)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)  # populates cam_xpos/cam_xmat
        cid = self._cam_id(camera_id)
        pos = np.array(self.data.cam_xpos[cid])
        q_mj = se3.mat_to_quat(np.array(self.data.cam_xmat[cid]).reshape(3, 3))
        return Pose(pos, se3.quat_normalize(se3.quat_mul(q_mj, _R_X_PI)))


__all__ = ["RecorderKinematics"]
