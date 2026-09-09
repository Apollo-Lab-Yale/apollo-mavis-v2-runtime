"""GELLO Manipulation (phase-15; 16-gello): the passive leader arm's engagement logic,
calibration store and — built on top by the session layer — the ``GelloLoop`` /
viewpoint source / preview.

The leader itself is a runtime DEVICE (``devices/gello.py`` ``GelloReader``, the tracker
pattern); this package holds the pure, thread-free logic around its samples:

* :mod:`.calibration` — ``var/gello_calibration.json`` (joint offsets, gripper endpoints)
  and the calibration math (``match_arm_offsets``, ``gripper_frac``);
* :mod:`.engage` — the engagement state machine (``EngageMachine``), the ±2π unwrap and
  the xArm7 joint-limit helpers.

Arm roles are fixed by the cell (00-overview §0): GELLO drives the Manipulation Arm
(:data:`FOLLOWER_ARM_ID`), the Perception Arm (:data:`VIEW_ARM_ID`) is the viewpoint.
"""

from __future__ import annotations

FOLLOWER_ARM_ID = "grip"  # the Manipulation Arm: the only arm GELLO drives (16-gello §0 item 2)
VIEW_ARM_ID = "view"  # the Perception Arm: external viewpoint node or the GELLO hold posture

__all__ = ["FOLLOWER_ARM_ID", "VIEW_ARM_ID"]
