"""SceneKinematics — control-loop FK over a built scene (sim extra only).

Gives the teleop pipeline TCP/base poses in the WORLD frame. Owns a private
``MjData`` used exclusively from the control-loop thread (one thread per
MjData; the model is shared read-only). Unit tests substitute a fake
implementing the same two methods, so this module is imported lazily.
"""

from __future__ import annotations

import mujoco
import numpy as np
from apollo_xarm7_core import Pose, se3


class SceneKinematics:
    """FK helper: measured/commanded q -> world-frame TCP pose."""

    def __init__(self, scene) -> None:  # scene: apollo_xarm7_sim BuiltScene
        self.model = scene.model
        self.data = mujoco.MjData(scene.model)
        self.addr = scene.addressing
        mujoco.mj_kinematics(self.model, self.data)

    def tcp_world(self, arm_id: str, q: np.ndarray) -> Pose:
        """TCP site world pose at the given full arm config (7|8 incl. rail)."""
        a = self.addr[arm_id]
        self.data.qpos[a.qpos_adr] = np.asarray(q, dtype=np.float64)
        mujoco.mj_kinematics(self.model, self.data)
        pos = np.array(self.data.site_xpos[a.tcp_site_id])
        quat = se3.mat_to_quat(np.array(self.data.site_xmat[a.tcp_site_id]).reshape(3, 3))
        return Pose(pos, quat)

    def base_quat_world(self, arm_id: str) -> np.ndarray:
        """Arm base (link_base) orientation in world — constant (rail slides)."""
        a = self.addr[arm_id]
        mat = np.array(self.data.xmat[a.base_body_id]).reshape(3, 3)
        return se3.mat_to_quat(mat)


__all__ = ["SceneKinematics"]
