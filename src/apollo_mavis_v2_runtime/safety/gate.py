"""SafetyGate / NullGate — hold-last-safe command gate (11-safety §7, BINDING).

Behaviorally consistent with the reference implementation in
``apollo_mavis_v2_sim.tools.guardrail_check.SafetyGate``, plus §7.1 step 1
(stale-twin fail-closed) which the virtual-tick reference omits.

The twin dependency is duck-typed (``check`` / ``pair_distance`` /
``_arms_of_pair`` / ``inflation_m``) so unit tests can script a fake twin
without importing mujoco.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from apollo_mavis_v2_core import (
    CollisionEvent,
    CollisionReport,
    CommandSource,
    SafetyConfig,
)

ESCAPE_EPS_M = 1e-5  # strict-opening margin, §7 step 6


class GateTwin(Protocol):
    """The slice of DigitalTwin the gate needs (duck-typed for tests)."""

    inflation_m: float

    def check(self, q_by_arm) -> CollisionReport: ...
    def pair_distance(self, pair, q_by_arm, distmax) -> float: ...
    def _arms_of_pair(self, pair) -> list[str]: ...


@dataclass
class GateDecision:
    q_out: dict[str, np.ndarray]  # what may be dispatched (cmd, held, or escape)
    blocked: bool
    report: CollisionReport  # severity ok|warn|blocked, pairs, min_clearance_m
    events: list[CollisionEvent]  # transitions only (edge-triggered)


class NullGate:
    """Pass-through gate (sim default, 11-safety §5) — identical tick path."""

    def filter(
        self,
        q_cmd: dict[str, np.ndarray],
        q_meas: dict[str, np.ndarray],
        source: CommandSource = CommandSource.TELEOP,
        ts: float | None = None,
    ) -> GateDecision:
        return GateDecision(dict(q_cmd), False, CollisionReport.ok(), [])

    def reseed(self, arm_id: str, q_meas: np.ndarray) -> None:
        pass


class SafetyGate:
    """Hold-last-safe twin gate; state: ``_last_safe`` per arm, ``_blocked``,
    ``_block_pairs`` (11-safety §7.1)."""

    def __init__(self, twin: GateTwin, cfg: SafetyConfig) -> None:
        self.twin = twin
        self.cfg = cfg
        self._last_safe: dict[str, np.ndarray] | None = None
        self._blocked = False
        self._block_pairs: set[tuple[str, str]] = set()

    def _unblock_threshold(self) -> float:
        return self.twin.inflation_m + self.cfg.min_clearance_m + self.cfg.hysteresis_m

    def reseed(self, arm_id: str, q_meas: np.ndarray) -> None:
        """Reset ``_last_safe`` to measured after error recovery (§7.1)."""
        if self._last_safe is None:
            self._last_safe = {}
        self._last_safe[arm_id] = np.array(q_meas, dtype=np.float64)

    def filter(
        self,
        q_cmd: dict[str, np.ndarray],
        q_meas: dict[str, np.ndarray],
        source: CommandSource = CommandSource.TELEOP,
        ts: float | None = None,
        stale: bool = False,
    ) -> GateDecision:
        ts = time.monotonic() if ts is None else ts
        if self._last_safe is None:  # session start: seed from measured
            self._last_safe = {a: np.array(q) for a, q in q_meas.items()}
        for arm_id in q_cmd:  # late-joining arm (defensive)
            if arm_id not in self._last_safe:
                self._last_safe[arm_id] = np.array(q_meas[arm_id])

        if stale:  # §7.1 step 1: fail closed for ALL arms
            q_out = {a: self._last_safe.get(a, q_meas[a]) for a in q_cmd}
            event = CollisionEvent(
                ts=ts,
                kind="stale_twin",
                pairs=[],
                dists_m=[],
                min_clearance_m=0.0,
                source=source,
                arm_ids=sorted(q_cmd),
            )
            report = CollisionReport(
                blocked=True,
                severity="blocked",
                ts=ts,
                violations=[event],
            )
            events = [event] if not self._blocked else []
            self._blocked = True
            self._block_pairs = set()
            return GateDecision(q_out, True, report, events)

        report = self.twin.check(q_cmd)  # §7.1 steps 2-3 (COMMANDED config)
        viols: dict[tuple[str, str], float] = {}
        for ev in report.violations:
            viols[ev.pairs[0]] = ev.min_clearance_m
        if self._blocked:  # hysteresis band (δ, δ+hyst]: invisible to contacts
            for pair in self._block_pairs - set(viols):
                d = self.twin.pair_distance(pair, q_cmd, self._unblock_threshold() + 0.01)
                if d < self._unblock_threshold():
                    viols[pair] = d
        if not viols:  # §7.1 step 4
            events: list[CollisionEvent] = []
            if self._blocked:
                self._blocked = False
                self._block_pairs = set()
                events.append(
                    CollisionEvent(
                        ts=ts,
                        kind="cleared",
                        pairs=[],
                        dists_m=[],
                        min_clearance_m=self._unblock_threshold(),
                        source=source,
                    )
                )
            self._last_safe = {a: np.array(q) for a, q in q_cmd.items()}
            return GateDecision(dict(q_cmd), False, report, events)

        # §7.1 steps 5-7: per-arm escape test, else hold-last-safe.
        offending = {a for pair in viols for a in self.twin._arms_of_pair(pair)}
        d_cmd = {p: self.twin.pair_distance(p, q_cmd, 0.5) for p in viols}
        d_meas = {p: self.twin.pair_distance(p, None, 0.5) for p in viols}
        # "Violating at q_meas" uses the same step-3 window as the command: while blocked
        # the hysteresis band counts, else a pair the escape has already opened past δ
        # (measured 8.03 mm, commanded 8.16 mm) would read as NEW and step 7 would hold
        # the arm inside the band for good - a plan executed at 10 / 50 % speed cannot
        # jump the 2 mm band in one tick (2026-09-09, tests/test_plan_passes_gate.py).
        meas_thr = (
            self._unblock_threshold()
            if self._blocked
            else self.twin.inflation_m + self.cfg.min_clearance_m
        )
        meas_viols = {p for p, d in d_meas.items() if d < meas_thr}
        q_out: dict[str, np.ndarray] = {}
        for arm_id, q in q_cmd.items():
            if arm_id not in offending:
                q_out[arm_id] = q  # step 2 checked all arms jointly
                continue
            arm_pairs = [p for p in viols if arm_id in self.twin._arms_of_pair(p)]
            opens = all(d_cmd[p] >= d_meas[p] + ESCAPE_EPS_M for p in arm_pairs)
            no_new = all(p in meas_viols for p in arm_pairs)  # step 6, clause 2
            if opens and no_new:  # step 6 (T8 escape)
                q_out[arm_id] = q
                self._last_safe[arm_id] = np.array(q)
            else:  # step 7: hold
                q_out[arm_id] = self._last_safe[arm_id]
        events = []
        new_pairs = set(viols) - self._block_pairs
        if not self._blocked or new_pairs:  # rising edge / pair-set change
            pairs = sorted(viols)
            events.append(
                CollisionEvent(
                    ts=ts,
                    kind="penetration" if min(d_cmd.values()) <= 0.0 else "blocked",
                    pairs=pairs,
                    dists_m=[d_cmd[p] for p in pairs],
                    min_clearance_m=min(d_cmd.values()),
                    source=source,
                    arm_ids=sorted(offending),
                )
            )
        self._blocked = True
        self._block_pairs = set(viols)
        return GateDecision(q_out, True, report, events)


__all__ = ["ESCAPE_EPS_M", "GateDecision", "GateTwin", "NullGate", "SafetyGate"]
