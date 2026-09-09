"""PolicyRunner + ActionAnchor — policy-rate inference, 100 Hz application.

PolicyRunner queries the policy at 10-30 Hz on its own thread (latest-wins
output; the servo loop never blocks on inference). ActionAnchor implements
the jump-free action-space contract (12-dagger §6): delta-EE actions apply
to the current MEASURED pose (hil-serl mechanism); on gate events targets
re-anchor, pending chunks drop, and human->policy handback opens a
slew-limited window (chunked/abs policies re-query after ``reset()``).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from apollo_mavis_v2_core import Pose, se3
from apollo_mavis_v2_core.dagger import ControlMode, GateEvent
from apollo_mavis_v2_core.interfaces.policy import Observation, Policy, PolicyOutput

logger = logging.getLogger(__name__)

POLICY_TIMEOUT_MARGIN_S = 0.05  # stale after period + 50 ms (phase brief)
STALE_DECAY_PERIODS = 5  # interpolate toward hold over <=5 periods, then hold
NAN_STRIKES_PER_EPISODE = 3
BLOCK_STREAK_TICKS = 30  # gate-blocked AUTONOMOUS ticks -> anomaly (12-dagger §10)


@dataclass(frozen=True)
class SlewLimits:
    """Blend-window caps for human->policy re-entry (12-dagger §6, binding)."""

    lin_mps: float = 0.15
    ang_radps: float = 1.5
    rail_mps: float = 0.15
    window_s: float = 0.4  # valid 0.3-0.5
    joint_rad_per_tick: float = 0.05  # `joint` policies only


@dataclass(frozen=True)
class PolicyAnomalyEvent:
    """NaN 3-strike / gate block-streak alarm (12-dagger §12)."""

    kind: str  # "nan" | "block_streak"
    arm_id: str | None
    t_mono: float
    detail: str


class PolicyRunner:
    """Owns the policy thread; publishes the newest ``PolicyOutput``.

    The control loop supplies ``obs_fn`` (built from the latest snapshot on
    read) and consumes :meth:`latest`. ``policy_lock`` serializes ``act`` vs
    reloader ``load_weights`` (12-dagger §8).
    """

    def __init__(
        self,
        policy: Policy,
        obs_fn: Callable[[], Observation | None],
        rate_hz: float = 15.0,
        policy_lock: threading.Lock | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.policy = policy
        self.obs_fn = obs_fn
        self.period = 1.0 / float(rate_hz)
        self.lock = policy_lock or threading.Lock()
        self._clock = clock
        self._out: PolicyOutput | None = None
        self._out_t: float = -1e9
        self._out_lock = threading.Lock()
        self._requery = False
        self._paused = False
        self._thread: threading.Thread | None = None
        self._running = False

    # -- lifecycle -------------------------------------------------------------
    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, name="policy-runner", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        next_t = self._clock()
        while self._running:
            try:
                self.step_once(self._clock())
            except Exception:
                logger.exception("policy step failed")
            next_t += self.period
            lag = self._clock() - next_t
            if lag > self.period:
                next_t = self._clock()
            elif lag < 0.0:
                time.sleep(-lag)

    # -- one inference (public for deterministic tests) --------------------------
    def step_once(self, now: float) -> PolicyOutput | None:
        if self._paused:
            return None
        obs = self.obs_fn()
        if obs is None:
            return None
        with self.lock:
            if self._requery:
                self.policy.reset()  # drop chunks; fresh query from current obs
                self._requery = False
            out = self.policy.act(obs)
        with self._out_lock:
            self._out = out
            self._out_t = now
        return out

    # -- consumers ----------------------------------------------------------------
    def latest(self) -> tuple[PolicyOutput | None, float]:
        with self._out_lock:
            return self._out, self._out_t

    def drop_and_requery(self, reason: str = "handback") -> None:
        """Handback / boundary: clear stale output, ``policy.reset()`` before next act
        (``reason`` is the wire spelling an external source publishes; unused here)."""
        with self._out_lock:
            self._out = None
            self._out_t = -1e9
        self._requery = True

    def pause(self) -> None:
        """NaN 3-strike: stop policy output until episode boundary."""
        self._paused = True
        with self._out_lock:
            self._out = None

    def resume(self) -> None:
        self._paused = False

    @property
    def paused(self) -> bool:
        return self._paused

    # -- PolicySource surface (14-dora §11.1; the executor reads these) -----------------
    @property
    def spec(self):
        return self.policy.spec

    def version_label(self) -> str:
        return str(getattr(self.policy.spec, "version", 0))

    def current_version(self) -> int:
        return int(getattr(self.policy.spec, "version", 0))

    def staleness_scale(self, now: float) -> float:
        """1.0 fresh; linear decay to 0 over 5 periods past the timeout."""
        _, t = self.latest()
        age = now - t
        timeout = self.period + POLICY_TIMEOUT_MARGIN_S
        if age <= timeout:
            return 1.0
        return float(np.clip(1.0 - (age - timeout) / (STALE_DECAY_PERIODS * self.period), 0.0, 1.0))


class ActionAnchor:
    """Delta-EE anchoring to the measured pose + handback slew (12-dagger §6)."""

    def __init__(self, ik, kin, slew: SlewLimits, action_space: str = "delta_ee") -> None:
        self.ik = ik
        self.kin = kin
        self.slew = slew
        self.action_space = action_space
        self._window_until: dict[str, float] = {}  # arm_id -> slew window end

    def on_gate_event(self, ev: GateEvent) -> None:
        """Re-anchor on every switch; open the slew window on handback."""
        if ev.mode is ControlMode.POLICY and ev.source != "episode_reset":
            self._window_until[ev.arm_id] = ev.t_mono + self.slew.window_s

    def in_window(self, arm_id: str, now: float) -> bool:
        return now < self._window_until.get(arm_id, -1e9)

    def apply_delta(
        self,
        arm_id: str,
        delta_pos_b: np.ndarray,  # (3,) m, canonical (arm_base) frame, per tick
        delta_rot_b: np.ndarray,  # (3,) rotvec rad, per tick
        rail_delta: float | None,
        q_last: np.ndarray,
        q_meas: np.ndarray,
        dt: float,
        now: float,
    ) -> np.ndarray | None:
        """``target = measured ⊕ clamp(Δ)`` -> IK -> q (returns None on IK fail)."""
        dp = np.asarray(delta_pos_b, dtype=np.float64)
        dr = np.asarray(delta_rot_b, dtype=np.float64)
        slewing = self.in_window(arm_id, now) or self.action_space != "delta_ee"
        if slewing:  # window caps apply always for abs/chunked; delta is jump-free
            dp = _clamp_norm(dp, self.slew.lin_mps * dt)
            dr = _clamp_norm(dr, self.slew.ang_radps * dt)
        measured = self.kin.tcp_world(arm_id, q_meas)
        base_quat = self.kin.base_quat_world(arm_id)
        dp_w = se3.quat_rotate(base_quat, dp)
        dr_w = se3.quat_rotate(base_quat, dr)
        target = Pose(
            measured.position + dp_w,
            se3.quat_mul(se3.rotvec_to_quat(dr_w), measured.orientation),
        )
        result = self.ik.solve(arm_id, target, q_last)
        if result.diverged or not np.isfinite(result.pos_err_m):
            return None
        q = np.array(result.q)
        if q.shape[0] > 7:
            base = q_last[7]
            d = 0.0 if rail_delta is None else float(rail_delta)
            if slewing:
                d = float(np.clip(d, -self.slew.rail_mps * dt, self.slew.rail_mps * dt))
            q[7] = float(np.clip(base + d, 0.0, se3.RAIL_TRAVEL_M))
        return q


def _clamp_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= max_norm or n == 0.0:
        return v
    return v * (max_norm / n)


def split_action(vec: np.ndarray, arms: list[tuple[str, bool]]) -> dict[str, np.ndarray]:
    """Concatenated per-arm blocks -> per-arm slices (delta_ee layout:
    6 delta dims + gripper [+ rail]); arms = [(arm_id, has_rail), ...]."""
    out: dict[str, np.ndarray] = {}
    i = 0
    for arm_id, has_rail in arms:
        n = 8 if has_rail else 7
        out[arm_id] = np.asarray(vec[i : i + n])
        i += n
    return out


__all__ = [
    "PolicyRunner",
    "ActionAnchor",
    "SlewLimits",
    "PolicyAnomalyEvent",
    "split_action",
    "POLICY_TIMEOUT_MARGIN_S",
    "STALE_DECAY_PERIODS",
    "NAN_STRIKES_PER_EPISODE",
    "BLOCK_STREAK_TICKS",
]
