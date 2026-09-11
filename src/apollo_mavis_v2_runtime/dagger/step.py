"""``policy_step`` — the per-tick application of ONE policy row to ONE arm (12-dagger §6).

Shared by ``GatedPolicyExecutor._policy_step`` (DAgger / inference) and the base
``ControlLoop``'s action-column playback (``POST /api/session/playback`` with
``source: delta_ee | abs_ee``): both hand a ``PolicySource``-shaped ``runner`` and an
``ActionAnchor`` in and get the arm's next joint command out. torch- and dora-free.

Per tick, per arm:

1. ``runner.latest()`` -> no output: hold (``None``);
2. split the whole-cell row into per-arm blocks by the NAME-derived widths of the
   announced space (``arm_action_names``; 11/10 for ``abs_ee``, 8/7 for ``delta_ee``) -
   a missing, short or non-finite block for THIS arm: hold (an undriven arm's block is
   NaN by contract and never a strike here - the strike guard lives in the executor);
3. ``runner.staleness_scale(now, arm)`` <= 0: hold (both spaces);
4. ``delta_ee``: the row is ONE per-period increment applied exactly once in total -
   ``anchor.row_step`` spreads it by budget (``dt / period`` per tick until spent, then
   hold; a replaced row carries its remainder), staleness only decays the share - then
   ``anchor.apply_delta`` integrates it on the last command;
   ``abs_ee``: hand the row VERBATIM to ``anchor.apply_absolute`` (deadline
   interpolation toward the waypoint; an absolute value is never multiplied by a tick
   or chunk factor);
5. the gripper dim (index 6 / 9) goes to ``on_gripper(arm_id, frac)`` when finite - only
   for a row that produced a command (a rejected row or an IK failure holds the whole arm).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import numpy as np

from ..recorder.features import arm_action_names
from .policy_runner import ActionAnchor, split_action

logger = logging.getLogger(__name__)

GRIPPER_INDEX = {"delta_ee": 6, "abs_ee": 9}


def action_space_of(runner, anchor: ActionAnchor) -> str:
    """The layout the rows follow: the source's announced spec first, the anchor's
    construction-time value as the fallback (test stubs without a spec)."""
    spec = getattr(runner, "spec", None)
    space = getattr(spec, "action_space", None)
    return str(space) if space else str(anchor.action_space)


def staleness_of(runner, now: float, arm_id: str) -> float:
    """This arm's staleness (a source without the per-arm form answers for the cell)."""
    try:
        return float(runner.staleness_scale(now, arm_id))
    except TypeError:
        return float(runner.staleness_scale(now))


def policy_step(
    *,
    runner,
    anchor: ActionAnchor,
    arms_meta: Sequence[tuple[str, bool]],
    arm_id: str,
    state,  # ArmState (measured q)
    q_last: np.ndarray,
    now: float,
    dt: float,
    on_gripper: Callable[[str, float], None] | None = None,
) -> np.ndarray | None:
    """One arm's next joint command from the source's newest row, or ``None`` = hold."""
    out, t_row = runner.latest()
    if out is None:
        return None  # hold; the counterfactual row records NaN (12-dagger §12)
    space = action_space_of(runner, anchor)
    has_rail = next((bool(r) for a, r in arms_meta if a == arm_id), state.q.shape[0] > 7)
    width = len(arm_action_names(arm_id, has_rail, space))
    block = split_action(out.actions, list(arms_meta), space).get(arm_id)
    if block is None or block.shape[0] < width or not np.all(np.isfinite(block)):
        return None  # hold: no / short / NaN block for THIS arm (v1.3: per-arm finiteness)
    stale = staleness_of(runner, now, arm_id)
    if stale <= 0.0:
        return None  # > 5 stale periods (of this arm's stream): hold
    grip_index = GRIPPER_INDEX.get(space)
    if space == "delta_ee":
        # one row = one per-period increment, applied exactly once in total: the anchor
        # spreads it over the ticks by budget (cadence-robust), staleness only decays it
        row_key = (
            int(getattr(out, "version", 0)),
            float(t_row),
            int(getattr(out, "chunk_remaining", 0)),
        )
        rail_row = float(block[7]) if has_rail and block.shape[0] > 7 else 0.0
        delta_row = np.concatenate([np.asarray(block[:6], dtype=np.float64), [rail_row]])
        dt_over_period = float(dt) / float(runner.period)
        share = anchor.row_step(arm_id, row_key, delta_row, dt_over_period) * stale
        rail_d = float(share[6]) if has_rail and block.shape[0] > 7 else None
        q = anchor.apply_delta(
            arm_id,
            share[:3],
            share[3:6],
            rail_d,
            q_last,
            state.q,
            dt,
            now,
        )
    elif space == "abs_ee":
        q = anchor.apply_absolute(
            arm_id, block, q_last, state.q, dt, now, float(runner.period), float(t_row)
        )
    else:  # "joint" has no executor path yet (12-dagger §6 rule 3): hold
        return None
    if q is None:
        return None  # the row was rejected (bad rotation) or IK failed: hold, gripper included
    if on_gripper is not None and grip_index is not None:
        grip = float(block[grip_index])
        if np.isfinite(grip):
            on_gripper(arm_id, float(np.clip(grip, 0.0, 1.0)))
    return q


__all__ = ["policy_step", "action_space_of", "staleness_of", "GRIPPER_INDEX"]
