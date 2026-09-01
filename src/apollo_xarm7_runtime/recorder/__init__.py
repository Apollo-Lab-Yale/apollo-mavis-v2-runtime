"""Episode recording over LeRobot dataset v3 (04-runtime §10; 10-frames §6-§9).

Import discipline: ``lerobot`` (and through it torch) is imported lazily
inside :mod:`.episode_recorder` only — importing this package must stay
cheap so non-collect code paths and tests never pay the cost.
"""

from .features import (
    ACTION_SOURCE_LABELS,
    APOLLO_SCHEMA_VERSION,
    ArmMeta,
    arm_action_names,
    arm_state_names,
    build_features,
    build_repo_id,
    build_robot_type,
)
from .frames import RecordingFrameConverter, delta_to_frame, pose_from_frame, pose_to_frame

__all__ = [
    "ACTION_SOURCE_LABELS",
    "APOLLO_SCHEMA_VERSION",
    "ArmMeta",
    "arm_action_names",
    "arm_state_names",
    "build_features",
    "build_repo_id",
    "build_robot_type",
    "RecordingFrameConverter",
    "delta_to_frame",
    "pose_from_frame",
    "pose_to_frame",
]
