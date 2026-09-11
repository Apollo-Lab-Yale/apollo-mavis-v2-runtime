"""``ActionAnchor.row_step``: a ``delta_ee`` row is applied exactly once in total whatever the
source's real cadence (2026-09-11 dry run: rows at ~21 Hz against a 30 Hz period were
re-applied on every tick they stayed the newest -> 1.3-1.4x overshoot)."""

from __future__ import annotations

import numpy as np

from apollo_mavis_v2_runtime.dagger.policy_runner import ActionAnchor, SlewLimits

DT_OVER_PERIOD = 0.3  # 100 Hz loop, 30 Hz rows


def _anchor() -> ActionAnchor:
    return ActionAnchor(ik=None, kin=None, slew=SlewLimits())


def _run(anchor: ActionAnchor, schedule: list[tuple[int, int]]) -> np.ndarray:
    """schedule = [(row_index, ticks the row stays newest), ...]; every row = unit delta."""
    total = np.zeros(7)
    for row, ticks in schedule:
        for _ in range(ticks):
            total += anchor.row_step("arm0", (0, float(row), 0), np.ones(7), DT_OVER_PERIOD)
    return total


def test_slow_source_does_not_over_apply():
    # each row stays newest for 5 ticks (a 20 Hz source): the old rule applied 5 x 0.3 = 1.5
    total = _run(_anchor(), [(k, 5) for k in range(10)])
    np.testing.assert_allclose(total, 10.0 * np.ones(7), atol=1e-9)


def test_nominal_cadence_is_exact_and_smooth():
    a = _anchor()
    shares = [a.row_step("arm0", (0, 0.0, 0), np.ones(7), DT_OVER_PERIOD)[0] for _ in range(4)]
    np.testing.assert_allclose(shares, [0.3, 0.3, 0.3, 0.1], atol=1e-12)  # 3.33 ticks per period
    assert a.row_step("arm0", (0, 0.0, 0), np.ones(7), DT_OVER_PERIOD)[0] == 0.0  # then hold


def test_fast_source_carries_the_remainder_forward():
    # rows replaced every 2 ticks (a 50 Hz source): 0.6 applied per row, 0.4 carried into the next
    total = _run(_anchor(), [(k, 2) for k in range(10)])
    assert 9.0 <= total[0] <= 10.0  # nothing lost beyond one in-flight row
    # let the last row drain
    a = _anchor()
    tot = _run(a, [(k, 2) for k in range(10)]) + sum(
        a.row_step("arm0", (0, 9.0, 0), np.ones(7), DT_OVER_PERIOD) for _ in range(10)
    )
    np.testing.assert_allclose(tot, 10.0 * np.ones(7), atol=1e-9)


def test_gate_event_drops_the_half_applied_row():
    from apollo_mavis_v2_core.dagger import ControlMode, GateEvent

    a = _anchor()
    a.row_step("arm0", (0, 0.0, 0), np.ones(7), DT_OVER_PERIOD)
    a.on_gate_event(
        GateEvent(arm_id="arm0", mode=ControlMode.HUMAN, t_mono=1.0, seq=1, source="keyboard")
    )
    # a new row after the switch starts from a clean budget: no carry from the interrupted row
    assert a.row_step("arm0", (0, 1.0, 0), np.ones(7), DT_OVER_PERIOD)[0] == 0.3
