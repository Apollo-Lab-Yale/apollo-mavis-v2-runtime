"""Direct joint-control path: jog / goto (04-runtime §7).

``jog`` = per-tick slew toward a latest-wins target, joint-space (no IK);
``jog`` with ``max|Δq| > goto_threshold`` is nacked. ``goto`` = twin plan ->
waypoint stream through the same slew-limited gated path. Both pass the
gate; both are nacked while recording (phase-07 wires the recorder flag).
"""

from __future__ import annotations

import numpy as np

from ..config import JogConfig

N_JOINTS = 7  # arm joints; rail is slot 7 when present


class JogState:
    """Per-arm latest-wins jog target consumed by the control loop."""

    def __init__(self, cfg: JogConfig) -> None:
        self.cfg = cfg
        self._target: dict[str, np.ndarray] = {}

    def set_target(self, arm_id: str, positions: np.ndarray) -> None:
        self._target[arm_id] = np.asarray(positions, dtype=np.float64).copy()

    def clear(self, arm_id: str) -> None:
        self._target.pop(arm_id, None)

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
        if ratio > 1.0:  # straight segment: the fastest slot at its cap, the rest in proportion
            delta = delta / ratio
        q_next = q_last + delta
        if np.max(np.abs(q_next - target)) < 1e-9:
            if i + 1 < len(wps):
                self._index[arm_id] = i + 1
            else:
                self.cancel(arm_id)  # done
                return target.copy()
        return q_next


__all__ = ["N_JOINTS", "JogState", "PlanExecutor"]
