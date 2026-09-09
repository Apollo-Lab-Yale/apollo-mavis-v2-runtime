"""SceneKinematics — control-loop FK over a built scene (sim extra only).

Gives the teleop pipeline TCP/base poses in the WORLD frame, plus the wrist
camera's world orientation (the keyboard translate frame, 2026-09-08). Owns a
private ``MjData`` used exclusively from the control-loop thread (one thread per
MjData; the model is shared read-only). Unit tests substitute a fake
implementing ``tcp_world`` / ``base_quat_world``, so this module is imported
lazily and callers treat ``wrist_cam_quat_world`` as optional.
"""

from __future__ import annotations

import mujoco
import numpy as np
from apollo_mavis_v2_core import Pose, se3


class SceneKinematics:
    """FK helper: measured/commanded q -> world-frame TCP pose."""

    def __init__(self, scene) -> None:  # scene: apollo_mavis_v2_sim BuiltScene
        self.model = scene.model
        self.data = mujoco.MjData(scene.model)
        self.addr = scene.addressing
        mujoco.mj_kinematics(self.model, self.data)
        # TCP -> wrist-camera rotation per arm: both are welded to the same link7,
        # so it is a CONSTANT and worth computing once (the 100 Hz path then costs
        # one quaternion product). It is NOT the same on both arms — the gripper
        # base carries quat (0,0,0,1), a 180° turn about the tool axis, so the
        # gripper arm's TCP frame is that much rotated against a camera-only arm's.
        mujoco.mj_camlight(self.model, self.data)  # populates cam_xmat
        self._cam_from_tcp: dict[str, np.ndarray] = {}
        for arm_id, a in self.addr.arms.items():
            if a.wrist_cam_id is None:
                continue  # no wrist camera on this arm (some test scenes)
            r_tcp = np.array(self.data.site_xmat[a.tcp_site_id]).reshape(3, 3)
            r_cam = np.array(self.data.cam_xmat[a.wrist_cam_id]).reshape(3, 3)
            self._cam_from_tcp[arm_id] = se3.mat_to_quat(r_tcp.T @ r_cam)

    def tcp_world(self, arm_id: str, q: np.ndarray) -> Pose:
        """TCP site world pose at the given full arm config (7|8 incl. rail)."""
        a = self.addr[arm_id]
        self.data.qpos[a.qpos_adr] = np.asarray(q, dtype=np.float64)
        mujoco.mj_kinematics(self.model, self.data)
        pos = np.array(self.data.site_xpos[a.tcp_site_id])
        quat = se3.mat_to_quat(np.array(self.data.site_xmat[a.tcp_site_id]).reshape(3, 3))
        return Pose(pos, quat)

    def wrist_cam_quat_world(self, arm_id: str, tcp_quat: np.ndarray) -> np.ndarray | None:
        """The arm's wrist-camera orientation in world, given its TCP orientation;
        ``None`` when the arm carries no wrist camera. Used by the keyboard's
        camera translate frame (04-runtime §6)."""
        rel = self._cam_from_tcp.get(arm_id)
        if rel is None:
            return None
        return se3.quat_mul(np.asarray(tcp_quat, dtype=np.float64), rel)

    def base_quat_world(self, arm_id: str) -> np.ndarray:
        """Arm base (link_base) orientation in world — constant (rail slides)."""
        a = self.addr[arm_id]
        mat = np.array(self.data.xmat[a.base_body_id]).reshape(3, 3)
        return se3.mat_to_quat(mat)


__all__ = ["SceneKinematics"]
