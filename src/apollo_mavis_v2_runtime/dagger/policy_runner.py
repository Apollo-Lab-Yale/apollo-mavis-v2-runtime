"""PolicyRunner + ActionAnchor — policy-rate inference, 100 Hz application.

PolicyRunner queries the policy at 10-30 Hz on its own thread (latest-wins
output; the servo loop never blocks on inference). ActionAnchor implements
the jump-free action-space contract (12-dagger §6, amended 2026-09-11):

- ``delta_ee`` rows INTEGRATE ON THE LAST GATED COMMAND (``FK(q_last)``) and
  are LEASHED to the measured pose (``ControlConfig.leash``: 25 mm / 0.2 rad,
  the keyboard / tracker teleop rule) - never re-anchored to the measured pose,
  so a lagging arm accumulates the policy's intent up to the leash instead of
  capping the executed velocity at its own lag;
- ``abs_ee`` rows are waypoints: the commanded TCP interpolates from
  ``FK(q_last)`` toward the row so it arrives at the row's deadline (one
  policy period after the row became current);
- on gate events pending chunks drop and human->policy handback opens a
  slew-limited window whose per-tick caps apply to BOTH spaces.
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

from ..recorder.features import arm_action_names

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

    def staleness_scale(self, now: float, arm_id: str | None = None) -> float:
        """1.0 fresh; linear decay to 0 over 5 periods past the timeout (one output feeds
        every arm, so ``arm_id`` changes nothing here)."""
        _, t = self.latest()
        age = now - t
        timeout = self.period + POLICY_TIMEOUT_MARGIN_S
        if age <= timeout:
            return 1.0
        return float(np.clip(1.0 - (age - timeout) / (STALE_DECAY_PERIODS * self.period), 0.0, 1.0))

    def driven_arms(self) -> frozenset[str] | None:
        """An in-process checkpoint drives every session arm (its layout IS the session's)."""
        return None


class ActionAnchor:
    """Apply a policy row to one arm: integrate-on-command + leash (``delta_ee``) or
    deadline interpolation (``abs_ee``), plus the handback slew window (12-dagger §6).

    Stateless between ticks apart from the window bookkeeping: the anchor of every
    tick is ``FK(q_last)``, the GATED + CAPPED command of the previous tick, so intent
    the gate or the step cap refused is dropped automatically and no stored pose can
    survive a motion made by another source (plan, jog, takeover, fault reseed)."""

    def __init__(
        self,
        ik,
        kin,
        slew: SlewLimits,
        action_space: str = "delta_ee",
        *,
        leash_pos_m: float = 0.025,  # = ControlConfig.leash defaults (config.py LeashConfig)
        leash_rot_rad: float = 0.2,
    ) -> None:
        self.ik = ik
        self.kin = kin
        self.slew = slew
        self.action_space = action_space
        self.leash_pos_m = float(leash_pos_m)
        self.leash_rot_rad = float(leash_rot_rad)
        self._window_until: dict[str, float] = {}  # arm_id -> slew window end
        # arm_id -> (row key, budget left in [0, 1], effective per-period delta [dp3, dr3, rail])
        self._rows: dict[str, tuple[tuple, float, np.ndarray]] = {}

    def on_gate_event(self, ev: GateEvent) -> None:
        """Open the slew window on every handback (the anchor itself is ``FK(q_last)`` on
        every tick, so there is no stored pose to re-anchor)."""
        if ev.mode is ControlMode.POLICY and ev.source != "episode_reset":
            self._window_until[ev.arm_id] = ev.t_mono + self.slew.window_s
        self._rows.pop(ev.arm_id, None)  # never carry a half-applied row across a switch

    def row_step(
        self, arm_id: str, row_key: tuple, delta_row: np.ndarray, dt_over_period: float
    ) -> np.ndarray:
        """Cadence-robust spreading of ONE per-period ``delta_ee`` row over the control ticks.

        Every row is applied EXACTLY ONCE in total: ``min(dt / period, budget)`` of it per
        tick until its budget (1.0) is spent, then hold until the next row. A row that is
        replaced before it was fully applied carries its remainder into the next row
        (spread over that row's period). So a source that publishes slower than its
        announced rate no longer over-applies (each row used to be re-applied on every
        tick it stayed the newest - 1.3-1.4x on the 2026-09-11 dry run at ~21 Hz against a
        30 Hz period) and a faster one does not lose motion. ``delta_row`` = ``[dp3, dr3,
        rail]`` in per-period units; the returned vector is this tick's share."""
        key, budget, eff = self._rows.get(arm_id, (None, 0.0, None))
        if key != row_key:
            carry = eff * budget if (eff is not None and budget > 0.0) else 0.0
            eff = np.asarray(delta_row, dtype=np.float64) + carry
            budget = 1.0
        step = min(max(float(dt_over_period), 0.0), budget)
        budget -= step
        self._rows[arm_id] = (row_key, budget, eff)
        return eff * step

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
        """``target = leash(FK(q_last) ⊕ Δ, measured)`` -> IK -> q (None on IK fail).

        The command integrates on itself and the leash to the MEASURED pose bounds the
        wind-up (04-runtime §6 teleop rule); inside the handback window the per-tick
        step is additionally capped by ``SlewLimits``."""
        dp = np.asarray(delta_pos_b, dtype=np.float64)
        dr = np.asarray(delta_rot_b, dtype=np.float64)
        if self.in_window(arm_id, now):
            dp = _clamp_norm(dp, self.slew.lin_mps * dt)
            dr = _clamp_norm(dr, self.slew.ang_radps * dt)
        anchor = self.kin.tcp_world(arm_id, q_last)
        measured = self.kin.tcp_world(arm_id, q_meas)
        base_quat = self.kin.base_quat_world(arm_id)
        dp_w = se3.quat_rotate(base_quat, dp)
        dr_w = se3.quat_rotate(base_quat, dr)
        target = Pose(
            anchor.position + dp_w,
            se3.quat_mul(se3.rotvec_to_quat(dr_w), anchor.orientation),
        )
        target = se3.clamp_pose_to_leash(target, measured, self.leash_pos_m, self.leash_rot_rad)
        result = self.ik.solve(arm_id, target, q_last)
        if result.diverged or not np.isfinite(result.pos_err_m):
            return None
        q = np.array(result.q)
        if q.shape[0] > 7:
            base = q_last[7]
            d = 0.0 if rail_delta is None else float(rail_delta)
            if self.in_window(arm_id, now):
                d = float(np.clip(d, -self.slew.rail_mps * dt, self.slew.rail_mps * dt))
            q[7] = float(np.clip(base + d, 0.0, se3.RAIL_TRAVEL_M))
        return q

    def apply_absolute(
        self,
        arm_id: str,
        block: np.ndarray,  # [x, y, z, r6(6), gripper, rail?] in the arm's recording frame
        q_last: np.ndarray,
        q_meas: np.ndarray,  # reserved (the abs path leashes nothing to the measured pose)
        dt: float,
        now: float,
        period: float,
        t_row: float,
    ) -> np.ndarray | None:
        """One ``abs_ee`` waypoint: interpolate ``FK(q_last)`` toward the row so the
        command REACHES it at the row's deadline (``t_row + period``); IK from ``q_last``;
        the rail slot approaches its absolute target with the same fraction. Inside the
        handback window the per-tick step is capped by ``SlewLimits``. None on a bad
        rotation (Gram-Schmidt) or an IK failure. The gripper dim is NOT applied here
        (the caller owns the gripper channel)."""
        del q_meas
        row = np.asarray(block, dtype=np.float64)
        try:
            quat_b = se3.rot6d_to_quat(row[3:9])
        except ValueError:
            return None
        base = self.kin.base_world(arm_id, q_last)
        target_w = se3.pose_mul(base, Pose(row[:3], quat_b))
        prev = self.kin.tcp_world(arm_id, q_last)
        remaining = max(float(period) - (float(now) - float(t_row)), float(dt))
        frac = min(1.0, float(dt) / remaining)
        pose = se3.pose_interp(prev, target_w, frac)
        if self.in_window(arm_id, now):
            pose = se3.clamp_pose_to_leash(
                pose, prev, self.slew.lin_mps * dt, self.slew.ang_radps * dt
            )
        result = self.ik.solve(arm_id, pose, q_last)
        if result.diverged or not np.isfinite(result.pos_err_m):
            return None
        q = np.array(result.q)
        if q.shape[0] > 7:
            cur = float(q_last[7])
            if row.shape[0] > 10:
                d = frac * (float(row[10]) - cur)
                if self.in_window(arm_id, now):
                    d = float(np.clip(d, -self.slew.rail_mps * dt, self.slew.rail_mps * dt))
                q[7] = float(np.clip(cur + d, 0.0, se3.RAIL_TRAVEL_M))
            else:
                q[7] = cur
        return q


def _clamp_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= max_norm or n == 0.0:
        return v
    return v * (max_norm / n)


def split_action(
    vec: np.ndarray, arms: list[tuple[str, bool]], action_space: str = "delta_ee"
) -> dict[str, np.ndarray]:
    """Concatenated per-arm blocks -> per-arm slices; block widths come from
    :func:`~apollo_mavis_v2_runtime.recorder.features.arm_action_names` for the given
    space (never hard-coded); arms = [(arm_id, has_rail), ...] in session order."""
    out: dict[str, np.ndarray] = {}
    i = 0
    for arm_id, has_rail in arms:
        n = len(arm_action_names(arm_id, bool(has_rail), action_space))
        out[arm_id] = np.asarray(vec[i : i + n])
        i += n
    return out


def anchor_leash_kwargs(dagger_cfg, control_cfg) -> dict[str, float]:
    """The ``ActionAnchor`` leash kwargs for a session: ``DaggerConfig.anchor_leash`` when
    set, else ``ControlConfig.leash`` (the teleop leash). One-liner for the manager:
    ``ActionAnchor(ik, kin, slew, action_space=..., **anchor_leash_kwargs(dcfg, ccfg))``."""
    leash = getattr(dagger_cfg, "anchor_leash", None) or control_cfg.leash
    return {"leash_pos_m": float(leash.pos_m), "leash_rot_rad": float(leash.rot_rad)}


__all__ = [
    "PolicyRunner",
    "ActionAnchor",
    "SlewLimits",
    "PolicyAnomalyEvent",
    "split_action",
    "anchor_leash_kwargs",
    "POLICY_TIMEOUT_MARGIN_S",
    "STALE_DECAY_PERIODS",
    "NAN_STRIKES_PER_EPISODE",
    "BLOCK_STREAK_TICKS",
]
