"""SafetyGate vs a scripted fake twin (11-safety §14.1)."""

from __future__ import annotations

import numpy as np
import pytest
from apollo_mavis_v2_core import CollisionEvent, CollisionReport, SafetyConfig

from apollo_mavis_v2_runtime.safety.gate import NullGate, SafetyGate

PAIR = ("arm0_link5", "table")


class FakeTwin:
    """Scriptable GateTwin: per-call violation list + pair distances."""

    inflation_m = 0.008

    def __init__(self) -> None:
        self.violations: list[tuple[tuple[str, str], float]] = []
        self.dist_cmd: dict[tuple[str, str], float] = {}
        self.dist_meas: dict[tuple[str, str], float] = {}

    def check(self, q_by_arm) -> CollisionReport:
        if not self.violations:
            return CollisionReport(blocked=False, severity="ok")
        events = [
            CollisionEvent(ts=0.0, kind="blocked", pairs=[p], dists_m=[d],
                           min_clearance_m=d, arm_ids=self._arms_of_pair(p))
            for p, d in self.violations
        ]
        return CollisionReport(
            blocked=True, severity="blocked", pairs=[p for p, _ in self.violations],
            min_clearance_m=min(d for _, d in self.violations), violations=events,
        )

    def pair_distance(self, pair, q_by_arm, distmax) -> float:
        table = self.dist_meas if q_by_arm is None else self.dist_cmd
        return table.get(tuple(pair), distmax)

    def _arms_of_pair(self, pair) -> list[str]:
        return sorted({label.split("_")[0] for label in pair if label.startswith("arm")})


@pytest.fixture
def gate():
    return SafetyGate(FakeTwin(), SafetyConfig()), FakeTwin.__new__


def mk(q0=0.0):
    return {"arm0": np.full(7, q0), "arm1": np.full(7, q0)}


def test_pass_through_and_last_safe_update():
    twin = FakeTwin()
    g = SafetyGate(twin, SafetyConfig())
    dec = g.filter(mk(0.1), mk(0.0))
    assert not dec.blocked and not dec.events
    assert np.allclose(dec.q_out["arm0"], 0.1)


def test_block_holds_last_safe_and_emits_edge_event():
    twin = FakeTwin()
    g = SafetyGate(twin, SafetyConfig())
    g.filter(mk(0.1), mk(0.0))  # last_safe = 0.1
    twin.violations = [(PAIR, 0.004)]
    twin.dist_cmd[PAIR] = 0.004
    twin.dist_meas[PAIR] = 0.006  # command closes the gap -> no escape
    dec = g.filter(mk(0.2), mk(0.1))
    assert dec.blocked
    assert np.allclose(dec.q_out["arm0"], 0.1)  # hold-last-safe
    assert np.allclose(dec.q_out["arm1"], 0.2)  # non-offending arm keeps q_cmd
    assert [e.kind for e in dec.events] == ["blocked"]
    # Second blocked tick with the same pair set: edge-triggered, no new event.
    dec2 = g.filter(mk(0.2), mk(0.1))
    assert dec2.blocked and not dec2.events


def test_hysteresis_no_chatter():
    twin = FakeTwin()
    cfg = SafetyConfig()  # unblock needs inflation + hysteresis = 0.010
    g = SafetyGate(twin, cfg)
    g.filter(mk(0.0), mk(0.0))
    twin.violations = [(PAIR, 0.002)]
    twin.dist_cmd[PAIR] = 0.002
    twin.dist_meas[PAIR] = 0.002
    assert g.filter(mk(0.1), mk(0.0)).blocked
    # Contact leaves the detection window but sits inside the hysteresis band.
    twin.violations = []
    twin.dist_cmd[PAIR] = 0.0095  # < 0.008 + 0.002
    dec = g.filter(mk(0.1), mk(0.0))
    assert dec.blocked and not any(e.kind == "cleared" for e in dec.events)
    # Now clear of inflation + hysteresis: unblocks with a "cleared" event.
    twin.dist_cmd[PAIR] = 0.011
    dec = g.filter(mk(0.1), mk(0.0))
    assert not dec.blocked
    assert [e.kind for e in dec.events] == ["cleared"]


def test_escape_rule_accepts_strictly_opening_command():
    twin = FakeTwin()
    g = SafetyGate(twin, SafetyConfig())
    g.filter(mk(0.0), mk(0.0))
    twin.violations = [(PAIR, 0.004)]
    twin.dist_cmd[PAIR] = 0.005  # command opens by 1 mm > 1e-5
    twin.dist_meas[PAIR] = 0.004  # pair already violating at measured
    dec = g.filter(mk(0.3), mk(0.0))
    assert dec.blocked  # still inside the band...
    assert np.allclose(dec.q_out["arm0"], 0.3)  # ...but the escape passes


def test_escape_rule_rejects_new_violation():
    twin = FakeTwin()
    g = SafetyGate(twin, SafetyConfig())
    g.filter(mk(0.0), mk(0.0))
    twin.violations = [(PAIR, 0.004)]
    twin.dist_cmd[PAIR] = 0.005
    twin.dist_meas[PAIR] = 0.02  # NOT violating at measured -> a NEW violation
    dec = g.filter(mk(0.3), mk(0.0))
    assert dec.blocked
    assert np.allclose(dec.q_out["arm0"], 0.0)  # held


def test_stale_twin_fails_closed_all_arms():
    twin = FakeTwin()
    g = SafetyGate(twin, SafetyConfig())
    g.filter(mk(0.1), mk(0.0))
    dec = g.filter(mk(0.5), mk(0.2), stale=True)
    assert dec.blocked
    assert np.allclose(dec.q_out["arm0"], 0.1)  # last safe, both arms
    assert np.allclose(dec.q_out["arm1"], 0.1)
    assert [e.kind for e in dec.events] == ["stale_twin"]


def test_reseed_after_recovery():
    twin = FakeTwin()
    g = SafetyGate(twin, SafetyConfig())
    g.filter(mk(0.1), mk(0.0))
    g.reseed("arm0", np.full(7, 0.7))
    dec = g.filter(mk(0.9), mk(0.7), stale=True)
    assert np.allclose(dec.q_out["arm0"], 0.7)  # never replays pre-e-stop 0.1


def test_null_gate_passthrough():
    dec = NullGate().filter(mk(0.4), mk(0.0))
    assert not dec.blocked and dec.report.severity == "ok"
    assert np.allclose(dec.q_out["arm0"], 0.4)


def test_escape_continues_through_the_hysteresis_band():
    """2026-09-09: step 6's "no NEW violating pair" uses the step-3 window. While blocked,
    a pair the escape has opened into the band [δ, δ + hysteresis) is still the SAME
    violation at q_meas, so an opening command keeps passing until the pair clears the
    band; before the fix it read as new and the arm was held inside the band for good
    (a twin plan at 10 / 50 % speed cannot jump the 2 mm band in one tick)."""
    twin = FakeTwin()
    g = SafetyGate(twin, SafetyConfig())  # unblock at 0.008 + 0.002
    g.filter(mk(0.0), mk(0.0))
    twin.violations = [(PAIR, 0.002)]
    twin.dist_cmd[PAIR] = 0.0025
    twin.dist_meas[PAIR] = 0.002
    dec = g.filter(mk(0.1), mk(0.0))
    assert dec.blocked and np.allclose(dec.q_out["arm0"], 0.1)  # T8 escape inside the shell
    # measured now inside the band, the command opens further but stays inside it
    twin.violations = []
    twin.dist_meas[PAIR] = 0.0085
    twin.dist_cmd[PAIR] = 0.0090
    dec = g.filter(mk(0.2), mk(0.1))
    assert dec.blocked and not dec.events
    assert np.allclose(dec.q_out["arm0"], 0.2)  # passes: same pair, still opening
    # ... a CLOSING command inside the band is still held ...
    twin.dist_meas[PAIR] = 0.0090
    twin.dist_cmd[PAIR] = 0.0088
    dec = g.filter(mk(0.3), mk(0.2))
    assert dec.blocked and np.allclose(dec.q_out["arm0"], 0.2)
    # ... and clearing the band unblocks
    twin.dist_cmd[PAIR] = 0.0101
    dec = g.filter(mk(0.3), mk(0.2))
    assert not dec.blocked and [e.kind for e in dec.events] == ["cleared"]
    # a pair NOT violating at q_meas by the window (unblocked: δ only) stays "new"
    twin.violations = [(PAIR, 0.005)]
    twin.dist_cmd[PAIR] = 0.006
    twin.dist_meas[PAIR] = 0.0095  # inside the band but the gate is not blocked
    dec = g.filter(mk(0.4), mk(0.3))
    assert dec.blocked and np.allclose(dec.q_out["arm0"], 0.3)  # held
