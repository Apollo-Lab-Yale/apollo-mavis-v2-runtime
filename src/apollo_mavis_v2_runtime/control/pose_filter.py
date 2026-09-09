"""One Euro pose filter for tracked-input teleop (13-tracker-teleop §4).

Casiez, Roussel & Vogel, "1€ Filter" (CHI 2012): an adaptive first-order
low-pass whose cutoff rises with the estimated speed — strong smoothing while
the hand rests (kills lighthouse jitter), little lag while it moves. Position is
filtered as a 3-vector with a shared, speed-dependent cutoff; orientation is
filtered on the rotation-vector increment relative to the previous filtered
orientation (a slerp toward the measurement by the same adaptive factor). A rest
deadband then drops sub-millimetre / sub-milliradian creep so a resting
controller commands exactly zero motion.

BETA IS IN Hz PER (m/s) AND THAT IS THE WHOLE TUNING TRAP (2026-09-07). The
paper's beta ~0.007 is per PIXEL/s, where hand speeds are hundreds of units/s;
in METRES they are 0.1-1, so a beta carried over from those figures makes
``beta * speed`` negligible against ``min_cutoff_hz`` and the "adaptive" filter
degenerates into a FIXED first-order low-pass. At the shipped-until-now
``min_cutoff_hz 1.0, beta 0.05`` the cutoff never left 1.0 Hz (tau = 159 ms):
measured against a constant-velocity ramp the output trailed the hand by 148 ms
/ 15 mm at 0.1 m/s and 131 ms / 39 mm at 0.3 m/s — more than the 25 mm teleop
leash, so the leash truncated every tick and ``TrackerTeleop.slip()`` folded the
truncation into the anchor. That was the operator's "trigger has a delay /
doesn't follow the hand". ``beta 5`` brings that to 7.6 mm /
~25 ms at 0.3 m/s (4.3 mm / 43 ms at 0.1, 11.7 mm / 15 ms at 0.8) and costs
nothing measurable at rest.

``d_cutoff_hz`` STAYS AT 1.0 and that is a deliberate refusal. It is the cutoff
of the speed estimate, so raising it lets the filter react to acceleration
sooner (beta 10 + d_cutoff 10 measured 5.2 mm / 17 ms) but also couples the
INPUT NOISE into the speed estimate, which raises the cutoff while the hand
rests: against 3 mm-std white noise the rest suppression falls 4.9x -> 2.8x
(``test_resting_jitter_is_suppressed_by_an_order_of_magnitude`` requires 4x).
Buying 8 ms with a third of the jitter rejection is the wrong trade here
because libsurvive keeps corrupting the lighthouse calibration
(``libsurvive rewrites the lighthouse config`` in CLAUDE.md) and the degraded
regime is exactly the noisy one — the filter has to stay usable there. The
healthy cell measures 0.1 mm p-p at rest, an order below ``deadband_m``, so
nothing is lost when the calibration is good.

Re-measure with the same two experiments (ramp lag against a constant-velocity
ramp, output std against white noise) before moving any of these again.

Pure NumPy, no threads. The provider calls ``reset()`` on every clutch engage
and after stale/invalid gaps (measured harmless either way: a reset and a
warm filter give identical engage transients, because a resting hand leaves the
speed estimate at ~0 anyway); ``retune()`` serves live ``tracker_settings``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from apollo_mavis_v2_core import Pose, se3


@dataclass(frozen=True)
class PoseFilterConfig:
    enabled: bool = True
    min_cutoff_hz: float = 1.0  # cutoff at rest (lower = smoother, laggier)
    beta: float = 5.0  # Hz per (m/s): cutoff = min_cutoff + beta * |velocity|
    d_cutoff_hz: float = 1.0  # cutoff of the velocity estimate = adaptation rate
    deadband_m: float = 0.001  # rest deadband on position (10x the measured rest noise)
    deadband_rad: float = 0.005  # rest deadband on orientation


def smoothing_factor(cutoff_hz: float, dt: float) -> float:
    """First-order low-pass gain for a cutoff (Hz) at sample period ``dt``."""
    tau = 1.0 / (2.0 * math.pi * max(cutoff_hz, 1e-6))
    return 1.0 / (1.0 + tau / dt)


class OneEuroVector:
    """One Euro filter over an N-vector with a shared speed-dependent cutoff."""

    def __init__(self, min_cutoff_hz: float, beta: float, d_cutoff_hz: float) -> None:
        self.min_cutoff_hz = float(min_cutoff_hz)
        self.beta = float(beta)
        self.d_cutoff_hz = float(d_cutoff_hz)
        self._x_hat: np.ndarray | None = None
        self._dx_hat: np.ndarray | None = None
        self._t: float | None = None

    def reset(self) -> None:
        self._x_hat = None
        self._dx_hat = None
        self._t = None

    def step(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x_hat is None or self._dx_hat is None or self._t is None:
            self._x_hat, self._dx_hat, self._t = x.copy(), np.zeros_like(x), float(t)
            return x.copy()
        dt = float(t) - self._t
        if dt <= 0.0:  # duplicate / non-monotonic stamp: hold the estimate
            return self._x_hat.copy()
        self._t = float(t)
        a_d = smoothing_factor(self.d_cutoff_hz, dt)
        self._dx_hat = a_d * ((x - self._x_hat) / dt) + (1.0 - a_d) * self._dx_hat
        cutoff = self.min_cutoff_hz + self.beta * float(np.linalg.norm(self._dx_hat))
        a = smoothing_factor(cutoff, dt)
        self._x_hat = a * x + (1.0 - a) * self._x_hat
        return self._x_hat.copy()


class PoseFilter:
    """One Euro on position and on orientation increments, then a rest deadband."""

    def __init__(self, cfg: PoseFilterConfig | None = None) -> None:
        self.cfg = cfg or PoseFilterConfig()
        c = self.cfg
        self._pos = OneEuroVector(c.min_cutoff_hz, c.beta, c.d_cutoff_hz)
        self.enabled = bool(c.enabled)  # live-toggled via retune(); False = passthrough
        self.min_cutoff_hz = c.min_cutoff_hz
        self.beta = c.beta
        self._q_hat: np.ndarray | None = None
        self._w_hat = np.zeros(3)  # filtered angular velocity estimate (rad/s)
        self._t: float | None = None
        self._emitted: Pose | None = None

    def reset(self) -> None:
        self._pos.reset()
        self._q_hat = None
        self._w_hat = np.zeros(3)
        self._t = None
        self._emitted = None

    def retune(
        self,
        min_cutoff_hz: float | None = None,
        beta: float | None = None,
        enabled: bool | None = None,
    ) -> None:
        """Live tuning from ``tracker_settings``; keeps the current state.
        Re-enabling after a passthrough stretch restarts from the next pose."""
        if min_cutoff_hz is not None:
            self.min_cutoff_hz = self._pos.min_cutoff_hz = max(float(min_cutoff_hz), 1e-3)
        if beta is not None:
            self.beta = self._pos.beta = max(float(beta), 0.0)
        if enabled is not None and bool(enabled) != self.enabled:
            self.enabled = bool(enabled)
            self.reset()

    def _step_orientation(self, q: np.ndarray, t: float) -> np.ndarray:
        if self._q_hat is None or self._t is None:
            self._q_hat = np.array(q, dtype=np.float64)
            return self._q_hat
        dt = float(t) - self._t
        if dt <= 0.0:
            return self._q_hat
        # rotation-vector increment from the filtered orientation to the measurement
        r = se3.quat_to_rotvec(se3.quat_mul(q, se3.quat_conj(self._q_hat)))
        a_d = smoothing_factor(self.cfg.d_cutoff_hz, dt)
        self._w_hat = a_d * (r / dt) + (1.0 - a_d) * self._w_hat
        cutoff = self.min_cutoff_hz + self.beta * float(np.linalg.norm(self._w_hat))
        a = smoothing_factor(cutoff, dt)
        self._q_hat = se3.quat_mul(se3.rotvec_to_quat(a * r), self._q_hat)  # slerp by a
        return self._q_hat

    def step(self, pose: Pose, t: float) -> Pose:
        if not self.enabled:
            return pose
        p_hat = self._pos.step(pose.position, t)
        q_hat = self._step_orientation(np.asarray(pose.orientation), t)
        self._t = float(t) if self._t is None else max(self._t, float(t))
        filtered = Pose(p_hat, q_hat)
        if self._emitted is None:
            self._emitted = filtered
            return filtered
        dp = float(np.linalg.norm(filtered.position - self._emitted.position))
        dr = float(se3.quat_geodesic(filtered.orientation, self._emitted.orientation))
        if dp < self.cfg.deadband_m and dr < self.cfg.deadband_rad:
            return self._emitted  # resting: exactly no motion
        self._emitted = filtered
        return filtered


__all__ = ["OneEuroVector", "PoseFilter", "PoseFilterConfig", "smoothing_factor"]
