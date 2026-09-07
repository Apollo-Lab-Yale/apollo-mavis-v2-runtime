"""SafetySupervisor — twin sync + gate + events + telemetry (11-safety §1)."""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Mapping

import numpy as np
from apollo_mavis_v2_core import (
    ArmState,
    CollisionEvent,
    CollisionReport,
    CommandSource,
)

from .gate import GateDecision, NullGate, SafetyGate
from .watchdog import ArmReportWatchdog, InputWatchdog

logger = logging.getLogger(__name__)

CLEARANCE_EVERY_N_TICKS = 4  # 25 Hz sweep on the measured config (§7.2)
CLEARANCE_SWEEP_M = 0.10  # default sweep range, m (SafetyConfig.clearance_sweep_m)


class SafetySupervisor:
    """Composes the twin gate + watchdogs; called only from the control loop."""

    def __init__(
        self,
        gate: SafetyGate | NullGate,
        watchdog: InputWatchdog,
        twin=None,
        report_watchdog: ArmReportWatchdog | None = None,
        warn_clearance_m: float = 0.025,
        clearance_sweep_m: float = CLEARANCE_SWEEP_M,
    ) -> None:
        self.gate = gate
        self.watchdog = watchdog
        self.twin = twin  # None in plain sim mode (NullGate)
        self.report_watchdog = report_watchdog or ArmReportWatchdog()
        self.warn_clearance_m = float(warn_clearance_m)
        # Sweep range (SafetyConfig.clearance_sweep_m): pairs farther apart than this
        # are not in ``clearances``. The Cockpit's proximity frame (05-ui §8.2) fades
        # in from this distance, so it must exceed the UI's 0.05 m "close" grade.
        self.clearance_sweep_m = float(clearance_sweep_m)
        self._tick = 0
        self._stale = False
        self._clearances: list[tuple[tuple[str, str], float]] = []
        self.events: deque[CollisionEvent] = deque(maxlen=50)  # session log pane
        self._last_report: CollisionReport = CollisionReport.ok()

    # -- top of every tick -----------------------------------------------------
    def sync(self, states: Mapping[str, ArmState], now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self._tick += 1
        if self.twin is None:
            return
        self._stale = self.report_watchdog.stale(states, now)
        try:
            self.twin.sync(states)
            if self._tick % CLEARANCE_EVERY_N_TICKS == 0:
                sweep = self.twin.clearance(distmax=self.clearance_sweep_m)
                self._clearances = [(pc.body_pair, pc.dist_m) for pc in sweep[:5]]
        except Exception:
            logger.exception("twin sync/clearance failed; failing closed")
            self._stale = True

    def filter(
        self,
        q_cmd: dict[str, np.ndarray],
        q_meas: dict[str, np.ndarray],
        source: CommandSource,
    ) -> GateDecision:
        if isinstance(self.gate, NullGate):
            dec = self.gate.filter(q_cmd, q_meas, source)
        else:
            try:
                dec = self.gate.filter(q_cmd, q_meas, source, stale=self._stale)
            except Exception:
                logger.exception("gate filter failed; holding all arms")
                dec = self.gate.filter(q_cmd, q_meas, source, stale=True)
        self.publish(dec.report, dec.events)
        return dec

    def publish(self, report: CollisionReport, events: list[CollisionEvent]) -> None:
        self._last_report = report
        for ev in events:
            self.events.append(ev)
            logger.info("collision event: %s pairs=%s", ev.kind, ev.pairs)

    def reseed(self, arm_id: str, q_meas: np.ndarray) -> None:
        self.gate.reseed(arm_id, q_meas)
        self.watchdog.on_recovery()

    # -- telemetry views ---------------------------------------------------------
    @property
    def clearances(self) -> list[tuple[tuple[str, str], float]]:
        return self._clearances

    def merged_report(self) -> CollisionReport:
        """Latest gate verdict merged with the 25 Hz warn sweep (§7.2)."""
        report = self._last_report
        if report.blocked:
            return report
        if self._clearances and self._clearances[0][1] < self.warn_clearance_m:
            return CollisionReport(
                blocked=False,
                severity="warn",
                pairs=[self._clearances[0][0]],
                min_clearance_m=self._clearances[0][1],
                ts=report.ts,
            )
        return report


__all__ = ["CLEARANCE_EVERY_N_TICKS", "CLEARANCE_SWEEP_M", "SafetySupervisor"]
