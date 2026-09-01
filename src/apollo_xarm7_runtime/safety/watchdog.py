"""InputWatchdog + ArmReportWatchdog (11-safety §10; 04-runtime §8).

``InputWatchdog`` states: OK -> TRIPPED -> AWAIT_EMPTY -> OK. Teleop twist,
joint-jog slew and gripper/rail held-keys are multiplied by ``scale()`` each
tick. Leaving TRIPPED/AWAIT_EMPTY requires a FRESH KeysMsg with an empty
held set — never resume from a replayed held set (T4/T5). Heartbeats
resuming with keys still held do NOT restore motion.
"""

from __future__ import annotations

from enum import Enum

from apollo_xarm7_core import HeldState


class WatchdogState(str, Enum):
    OK = "ok"
    TRIPPED = "tripped"  # ramping twist to zero
    AWAIT_EMPTY = "await_empty"  # latched at zero until an empty held set


class InputWatchdog:
    """Stale-input deadman with linear ramp and all-keys-up resume latch."""

    def __init__(self, timeout_s: float = 0.2, ramp_s: float = 0.1) -> None:
        self.timeout_s = float(timeout_s)
        self.ramp_s = float(ramp_s)
        self._state = WatchdogState.OK
        self._last_rx: float | None = None  # rx_mono of newest accepted KeysMsg
        self._last_seq: int | None = None
        self._trip_t: float | None = None  # moment the deadman expired

    # -- feeds ---------------------------------------------------------------
    def on_keys(self, held: HeldState) -> bool:
        """Feed one accepted KeysMsg; returns False on non-monotonic seq."""
        if self._last_seq is not None and held.seq <= self._last_seq:
            return False
        self._last_seq = held.seq
        self._last_rx = held.rx_mono
        if self._state is not WatchdogState.OK and not held.held:
            # A fresh EMPTY held set is the only way back to OK.
            self._state = WatchdogState.OK
            self._trip_t = None
        return True

    def on_disconnect(self) -> None:
        """Controller WS drop: immediate deadman path + drop held state."""
        self._state = WatchdogState.AWAIT_EMPTY
        self._trip_t = None
        self._last_rx = None
        self._last_seq = None  # a fresh controller restarts its own seq space

    def on_recovery(self) -> None:
        """Driver error recovery: re-seeded upstream; force all-keys-up."""
        self._state = WatchdogState.AWAIT_EMPTY
        self._trip_t = None

    # -- per-tick ----------------------------------------------------------------
    def scale(self, now: float) -> float:
        """1.0 fresh; ramps linearly to 0.0 over ``ramp_s`` after ``timeout_s``
        of silence; latches 0.0 (AWAIT_EMPTY) until an empty held set. A
        heartbeat resuming mid-ramp neither restores 1.0 nor resets the ramp."""
        if self._state is WatchdogState.AWAIT_EMPTY:
            return 0.0
        if self._state is WatchdogState.OK:
            if self._last_rx is None:
                return 1.0  # nothing ever held; twist is zero anyway
            if now - self._last_rx <= self.timeout_s:
                return 1.0
            self._state = WatchdogState.TRIPPED
            self._trip_t = self._last_rx + self.timeout_s
        # TRIPPED: ramp from the trip instant, regardless of later heartbeats.
        assert self._trip_t is not None
        frac = (now - self._trip_t) / self.ramp_s if self.ramp_s > 0 else 1.0
        if frac >= 1.0:
            self._state = WatchdogState.AWAIT_EMPTY
            return 0.0
        return max(0.0, 1.0 - frac)

    @property
    def state(self) -> WatchdogState:
        return self._state

    @property
    def needs_all_up(self) -> bool:
        return self._state is not WatchdogState.OK

    @property
    def tripped(self) -> bool:
        return self._state is not WatchdogState.OK


class ArmReportWatchdog:
    """Twin-staleness check: any stale arm fails the gate closed (11 §6.1)."""

    def __init__(self, twin_staleness_s: float = 0.15) -> None:
        self.twin_staleness_s = float(twin_staleness_s)

    def stale(self, states, now: float) -> bool:
        for st in states.values():
            if st.stale or (now - st.t_mono) > self.twin_staleness_s:
                return True
        return False


__all__ = ["WatchdogState", "InputWatchdog", "ArmReportWatchdog"]
