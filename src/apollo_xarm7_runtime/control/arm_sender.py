"""ArmSender — one thread per arm consuming a depth-1 q_cmd slot (04-runtime §3).

The xArm SDK is sync TCP; a slow arm must never stall the 100 Hz tick. Each
sender blocks on ``wait_fresh`` and forwards the newest target to
``ArmInterface.command_joints`` — the ONLY call site outside the interface
definitions and the fakes (the chokepoint scan whitelists this module and
``control/loop.py``, 11-safety §4).
"""

from __future__ import annotations

import logging
import threading

import numpy as np
from apollo_xarm7_core import ArmInterface, CommandError, GripperCommand, LatestSlot

logger = logging.getLogger(__name__)

_WAIT_S = 0.1


class ArmSender:
    """Consumes ``q_cmd[arm_id]``; optionally forwards gripper targets."""

    def __init__(self, arm_id: str, arm: ArmInterface, slot: LatestSlot) -> None:
        self.arm_id = arm_id
        self._arm = arm
        self._slot = slot
        self._grip_slot: LatestSlot[float] = LatestSlot()
        self._last_grip: float | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self.error_count = 0

    def put_gripper(self, open_frac: float) -> None:
        self._grip_slot.put(float(open_frac))

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name=f"arm-sender-{self.arm_id}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        while self._running:
            got = self._slot.wait_fresh(_WAIT_S)
            if got is not None:
                q = np.asarray(got[0], dtype=np.float64)
                try:
                    self._arm.command_joints(q)
                except CommandError:
                    self.error_count += 1
                    logger.exception("%s: command_joints rejected", self.arm_id)
                except Exception:
                    self.error_count += 1
                    logger.exception("%s: command_joints failed", self.arm_id)
            grip = self._grip_slot.get()
            if grip is not None and grip[0] != self._last_grip:
                self._last_grip = grip[0]
                try:
                    self._arm.command_gripper(GripperCommand(open_frac=grip[0]))
                except Exception:
                    logger.exception("%s: command_gripper failed", self.arm_id)


__all__ = ["ArmSender"]
