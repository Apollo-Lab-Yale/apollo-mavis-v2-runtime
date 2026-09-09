"""Direct joint-control path: jog / goto (04-runtime §7).

``jog`` = per-tick slew toward a latest-wins target, joint-space (no IK), at ANY
delta: the target is a destination, never a step, so the arm always approaches at
``slew_rad_per_tick`` and a fast slider drag just trails and catches up (the UI
joint panel is exactly this and has no other mode). ``goto`` = twin plan ->
waypoint stream through the same slew-limited gated path, kept for the profile /
rail-homing paths that need obstacle routing. Both pass the gate; both are nacked
while recording (phase-07 wires the recorder flag).
"""

from __future__ import annotations

import math

import numpy as np

from ..config import JogConfig

N_JOINTS = 7  # arm joints; rail is slot 7 when present
# ``PlanExecutor`` walks a segment in ``ceil(ratio)`` EQUAL ticks (``ratio`` = the segment
# measured in caps), so no tick is shorter than half a full tick and, for a segment of many
# ticks, every tick is practically a full one (2026-09-09; 11-safety §9). The twin
# planner's escape phase (sim ``planner._tick_count``, pinned by
# tests/test_plan_passes_gate.py) requires the gate's strict opening per tick of the
# slowest session; the previous "full ticks, then the remainder" rule emitted an arbitrarily
# small last tick that opened an arbitrarily small amount.
RATIO_EPS = 1e-9  # a ratio this close to an integer counts as that integer (float drift)


class JogState:
    """Per-arm latest-wins jog target consumed by the control loop."""

    def __init__(self, cfg: JogConfig) -> None:
        self.cfg = cfg
        self._target: dict[str, np.ndarray] = {}

    def set_target(self, arm_id: str, positions: np.ndarray) -> None:
        self._target[arm_id] = np.asarray(positions, dtype=np.float64).copy()

    def clear(self, arm_id: str) -> None:
        self._target.pop(arm_id, None)

    def clear_all(self) -> list[str]:
        """Drop every pending target; returns the arms that had one.

        Used on the WS input watchdog's latch edge (11-safety §10.1): a jog is a
        DESTINATION, so leaving it pending would (a) wedge the arm in
        ``CommandSource.JOINT_JOG`` forever, because the loop's deadman check
        returns before ``step()`` can ever retire the target - which also blocks
        that arm's tracker teleop - and (b) resume motion toward a stale
        destination the moment the browser reconnects, which is exactly what the
        all-keys-up latch exists to prevent (T4/T5).
        """
        had = sorted(self._target)
        self._target.clear()
        return had

    def active(self, arm_id: str) -> bool:
        return arm_id in self._target

    def step(self, arm_id: str, q_last: np.ndarray) -> np.ndarray | None:
        """One slew-limited step from ``q_last`` toward the jog target.

        Returns None when no jog is active; clears itself on arrival.
        """
        target = self._target.get(arm_id)
        if target is None:
            return None
        limit = np.full(q_last.shape, self.cfg.slew_rad_per_tick)
        if q_last.shape[0] > N_JOINTS:  # rail slot slews in meters
            limit[N_JOINTS:] = self.cfg.rail_m_per_tick
        delta = np.clip(target - q_last, -limit, limit)
        q_next = q_last + delta
        if np.allclose(q_next, target, atol=1e-9):
            self._target.pop(arm_id, None)
            return target.copy()
        return q_next


class PlanExecutor:
    """Streams planner waypoints through the slew-limited gated path (§7/§9).

    Waypoints are per-arm joint configs (post-shortcut). Each tick the
    executor moves toward the current waypoint ALONG THE STRAIGHT JOINT-SPACE
    SEGMENT - every joint advances by the same fraction of its remaining delta,
    the fastest joint at ``slew_rad_per_tick`` (rail slot ``rail_m_per_tick``)
    - and advances when reached. The planner validated exactly these straight
    segments (edge checks; phase-09d's position-agnostic ``check_path`` densifies
    them the same way), so the executed path never leaves the validated one: a
    per-joint clip would let the small-delta joints arrive first and bend the
    path off the checked line (the hardware gate then holds it for good).
    On a hardware loop ``JogConfig.plan_cart_step_m`` / ``plan_lever_arm_m``
    (from the driver's ``ServoLimits``) additionally bound the lever-weighted
    Cartesian step ``sum|dq_j| * lever_j`` per tick - the same uniform scaling
    the driver's servo streamer applies - so the COMMANDED q never runs ahead of
    what the streamer can follow: otherwise its per-joint velocity clip would
    bend the physical path off the validated segment while the gate only sees
    the commanded (on-line) posture. ``cancel()`` freezes at the current
    commanded q (decelerating stop is a single slew-limited hold at 100 Hz).

    Equal ticks (2026-09-09): a segment of ``ratio`` caps is walked in
    ``ceil(ratio)`` equal steps - the same number of ticks as "full ticks, then
    the remainder", still on the straight segment and under the caps, but no
    step is ever shorter than half a tick (a segment of many ticks: practically
    full ticks). The gate's T8 escape rule demands a strict opening on EVERY
    tick of an arm inside the shell, so a tiny remainder tick (measured 3-13 %
    of a tick at 10 % speed, opening 4-8 um < the gate's 10 um) was held;
    because the executor had already advanced its index, the next tick then
    headed for the following waypoint from the held point - an unvalidated
    bend. The twin planner's escape phase judges its segments with exactly this
    tick model (sim ``planner._tick_count``).
    """

    def __init__(self, cfg: JogConfig) -> None:
        self.cfg = cfg
        self._waypoints: dict[str, list[np.ndarray]] = {}
        self._index: dict[str, int] = {}
        cart = getattr(cfg, "plan_cart_step_m", None)
        lever = getattr(cfg, "plan_lever_arm_m", None)
        self._cart_step_m: float | None = (
            float(cart) if cart is not None and lever is not None and float(cart) > 0.0 else None
        )
        self._lever = (
            np.asarray(list(lever), dtype=np.float64)[:N_JOINTS]
            if self._cart_step_m is not None
            else None
        )

    def load(self, arm_id: str, waypoints: list[list[float]]) -> None:
        self._waypoints[arm_id] = [np.asarray(w, dtype=np.float64) for w in waypoints]
        self._index[arm_id] = 0

    def cancel(self, arm_id: str | None = None) -> None:
        if arm_id is None:
            self._waypoints.clear()
            self._index.clear()
        else:
            self._waypoints.pop(arm_id, None)
            self._index.pop(arm_id, None)

    def active(self, arm_id: str) -> bool:
        return arm_id in self._waypoints

    @property
    def active_arms(self) -> list[str]:
        return list(self._waypoints)

    def step(self, arm_id: str, q_last: np.ndarray) -> np.ndarray | None:
        """One slew-limited step along the waypoint stream; None when idle.

        Pops the arm from the active set after the final waypoint is reached
        (the returned value is the goal itself on that tick).
        """
        wps = self._waypoints.get(arm_id)
        if not wps:
            return None
        i = self._index[arm_id]
        target = wps[i]
        limit = np.full(q_last.shape, self.cfg.slew_rad_per_tick)
        if q_last.shape[0] > N_JOINTS:
            limit[N_JOINTS:] = self.cfg.rail_m_per_tick
        delta = target - q_last
        ratio = float(np.max(np.abs(delta) / limit)) if delta.size else 0.0
        if self._cart_step_m is not None and self._lever is not None:
            # hardware: the lever-weighted Cartesian bound of the servo streamer (uniform
            # scaling - the step stays on the straight segment)
            n = min(N_JOINTS, delta.shape[0], self._lever.shape[0])
            cart = float(np.sum(np.abs(delta[:n]) * self._lever[:n]))
            ratio = max(ratio, cart / self._cart_step_m)
        if ratio > 1.0 + RATIO_EPS:  # straight segment, ceil(ratio) equal steps (docstring)
            delta = delta / math.ceil(ratio - RATIO_EPS)
        q_next = q_last + delta
        if np.max(np.abs(q_next - target)) < 1e-9:
            if i + 1 < len(wps):
                self._index[arm_id] = i + 1
            else:
                self.cancel(arm_id)  # done
                return target.copy()
        return q_next


__all__ = ["N_JOINTS", "RATIO_EPS", "JogState", "PlanExecutor"]
