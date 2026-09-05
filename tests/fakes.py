"""Scriptable driver-event fakes for the FAULT -> RECOVERING -> RUNNING plumbing
(phase-09b; 04-runtime §15 / §16 "FakeWorkcell ... scriptable error codes").

The event classes mirror ``apollo_mavis_v2_hardware.events`` field-for-field and
BY NAME: the control loop dispatches on ``type(ev).__name__`` so the runtime never
imports the optional hardware package, and ``test_fault_recovery_loop.py`` pins
these names / fields against the hardware package whenever it is importable.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from apollo_mavis_v2_core import CommandError
from apollo_mavis_v2_core.testing import FakeArm, FakeCamera, FakeWorkcell


@dataclass(frozen=True)
class FaultEvent:
    arm_id: str
    source: str  # "servo" | "monitor" | "report" | "external" | "user_stop" | "user" | "latch"
    code: int
    error_code: int = 0
    warn_code: int = 0
    detail: str = ""
    t_mono: float = 0.0


@dataclass(frozen=True)
class RecoveredEvent:
    arm_id: str
    error_code: int
    t_mono: float = 0.0


@dataclass(frozen=True)
class ReseedEvent:
    arm_id: str
    q: tuple[float, ...]
    t_mono: float = 0.0


@dataclass(frozen=True)
class StudioConflictWarning:
    arm_id: str
    mode: int
    state: int
    detail: str = "close UFACTORY Studio live control"
    t_mono: float = 0.0


@dataclass(frozen=True)
class RecoveryResult:
    """Mirror of ``apollo_mavis_v2_hardware.RecoveryResult``."""

    seq: int
    ok: bool
    error_code: int
    detail: str = ""
    user_initiated: bool = False
    t_mono: float = 0.0


class EventFakeWorkcell(FakeWorkcell):
    """``FakeWorkcell`` + the hardware workcell's event / recovery surface.

    ``queue(*events)`` scripts events for the next ``drain_events()``;
    ``fault(arm_id, code)`` injects a controller error the way the driver would
    (``ArmState.error_code`` + a FaultEvent); ``request_recovery(arm_id)``
    emulates ``HardwareWorkcell.request_recovery``: after ``recovery_latency_s``
    (0 = synchronously, on the caller's thread) it clears the arm's error and
    queues ``FaultEvent(source="user") -> ReseedEvent -> RecoveredEvent`` plus a
    ``RecoveryResult(ok=True)`` - or, when ``latch_next[arm_id]`` holds a reason,
    a latch ``FaultEvent`` plus ``RecoveryResult(ok=False, detail=reason)`` and
    the error stays latched.
    """

    def __init__(
        self,
        arms: dict[str, FakeArm] | None = None,
        cameras: dict[str, FakeCamera] | None = None,
        kind: Literal["hardware", "sim"] = "hardware",
        *,
        recovery_latency_s: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(arms, cameras if cameras is not None else {}, kind)
        self._events: list[object] = []
        self._lock = threading.Lock()
        self.recovery_latency_s = float(recovery_latency_s)
        self._clock = clock
        self.latch_next: dict[str, str] = {}  # arm -> latch reason for the next recovery
        self._results: dict[str, RecoveryResult] = {}
        self._seq: dict[str, int] = {}
        self.recovery_requests: list[str] = []
        self.drains = 0

    # -- events -------------------------------------------------------------------------
    def queue(self, *events: object) -> None:
        with self._lock:
            self._events.extend(events)

    def drain_events(self) -> list[object]:
        with self._lock:
            out, self._events = self._events, []
        self.drains += 1
        return out

    def fault(
        self,
        arm_id: str,
        error_code: int = 24,
        *,
        source: str = "monitor",
        detail: str = "",
        code: int | None = None,
    ) -> FaultEvent:
        """Latch ``error_code`` on the arm and queue the matching FaultEvent."""
        arm = self.arms[arm_id]
        arm.inject_error(error_code)  # type: ignore[attr-defined]
        ev = FaultEvent(
            arm_id,
            source=source,
            code=error_code if code is None else code,
            error_code=error_code,
            detail=detail,
            t_mono=self._clock(),
        )
        self.queue(ev)
        return ev

    # -- recovery channel (HardwareWorkcell surface) -------------------------------------
    def request_recovery(self, arm_id: str) -> None:
        if arm_id not in self.arms:
            raise KeyError(arm_id)
        if not self.started:
            raise CommandError(f"{arm_id}: driver not connected")
        self.recovery_requests.append(arm_id)
        if self.recovery_latency_s > 0.0:
            timer = threading.Timer(self.recovery_latency_s, self._complete_recovery, (arm_id,))
            timer.daemon = True
            timer.start()
        else:
            self._complete_recovery(arm_id)

    def _complete_recovery(self, arm_id: str) -> None:
        arm = self.arms[arm_id]
        err = int(arm.get_state().error_code)
        now = self._clock()
        seq = self._seq.get(arm_id, 0) + 1
        self._seq[arm_id] = seq
        reason = self.latch_next.pop(arm_id, None)
        events: list[object] = [FaultEvent(arm_id, "user", code=0, error_code=err, t_mono=now)]
        if reason is None:
            arm.clear_errors()  # type: ignore[attr-defined]
            q = tuple(float(v) for v in arm.get_state().q)
            events += [ReseedEvent(arm_id, q, now), RecoveredEvent(arm_id, err, now)]
            result = RecoveryResult(seq, True, err, "", True, now)
        else:
            events.append(FaultEvent(arm_id, "latch", 0, error_code=err, detail=reason, t_mono=now))
            result = RecoveryResult(seq, False, err, reason, True, now)
        self.queue(*events)
        self._results[arm_id] = result

    def recovery_result(self, arm_id: str) -> RecoveryResult | None:
        return self._results.get(arm_id)


__all__ = [
    "EventFakeWorkcell",
    "FaultEvent",
    "RecoveredEvent",
    "RecoveryResult",
    "ReseedEvent",
    "StudioConflictWarning",
]
