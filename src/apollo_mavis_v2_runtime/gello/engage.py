"""GELLO engagement (16-gello D3 / §6.1 / §6.2): pure, thread-free.

State ``GelloState = no_leader | out_of_sync | tracking | paused | motion``; the follower
streams the leader ONLY in ``tracking``, every other state HOLDS the last command (the
loop's ``None`` resolution re-servos ``_last_cmd``) — never a move. Evaluated every tick by
:meth:`EngageMachine.step`:

* a plan owns the arm (``plan_active`` / :meth:`EngageMachine.on_motion_start`) -> ``motion``;
  when it retires the engage rule runs once (unless a pause is latched);
* ``paused`` is STICKY: entered by :meth:`EngageMachine.pause` (``gello_pause``), by
  :meth:`EngageMachine.on_fault` (a ``FaultEvent`` / RECOVERING of the Manipulation Arm) and
  by the loop before every planned in-session motion (§5.3); left only by
  :meth:`EngageMachine.resume` (``gello_resume``), which re-runs the engage rule;
* sample missing / stale (``age > stale_s``) / ``valid == False`` -> ``no_leader``;
* engage rule: ``max_j |unwrap(q_leader)_j - q_meas_j| <= engage_tol_rad`` -> ``tracking``
  (the ±2π branch ``k`` of joints 1/3/5/7 is fixed here), else ``out_of_sync`` with the
  per-joint deltas published;
* while ``tracking``: ``max_j |q_leader_j - q_cmd_j| > leash_rad`` -> ``out_of_sync`` (the
  gate held the follower or the operator outran the 0.6 rad/s cap; the follower holds
  instead of chasing); re-engagement from ``out_of_sync`` is automatic once the rule passes.

Unwrap (§6.2): for the joints with a ±2π range (indices :data:`WRAP_JOINTS` = J1/J3/J5/J7)
pick ``k in {-1, 0, 1}`` minimising ``|q_leader + 2*pi*k - q_ref|`` with ``q_ref`` = the
measured joint at engagement, then keep ``k`` while tracking (continuity); joints 2/4/6 are
clipped to their limits. The branch choice is LIMIT-AWARE (2026-09-09 review, 16-gello
§15.2 item 11): a candidate whose unwrapped value lies outside the margined joint limits is
dropped whenever another candidate lies inside them, so a leader reading that is itself a
reachable target (J1 = 0.3 rad against an arm at 3.5 rad) is never refused as "joint 1 =
6.58 rad" - the nearest IN-LIMIT branch wins (0.3 rad, k = 0), ties keep k = 0. The xArm7
limit table is a local copy of the hardware package's ``XARM7_JOINT_LIMITS_RAD`` (that
package pulls in the xArm SDK at import; a test pins the two tables equal when it is
importable).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from apollo_mavis_v2_core.protocol.gello import GelloState

if TYPE_CHECKING:
    from ..config import GelloConfig
    from ..devices.gello import GelloSample

TWO_PI = 2.0 * math.pi
WRAP_JOINTS: tuple[int, ...] = (0, 2, 4, 6)  # J1, J3, J5, J7: the ±2π joints
# xArm7 factory joint limits, rad (UFACTORY user manual) — the same table as
# apollo_mavis_v2_hardware.config.XARM7_JOINT_LIMITS_RAD (tests/test_gello_engage.py pins it).
XARM7_JOINT_LIMITS_RAD: tuple[tuple[float, float], ...] = (
    (-TWO_PI, TWO_PI),
    (math.radians(-118.0), math.radians(120.0)),
    (-TWO_PI, TWO_PI),
    (math.radians(-11.0), math.radians(225.0)),
    (-TWO_PI, TWO_PI),
    (math.radians(-97.0), math.radians(180.0)),
    (-TWO_PI, TWO_PI),
)
JOINT_LIMIT_MARGIN_RAD = 0.0087  # 0.5 deg inside the limits (the driver's own margin; §5.1 item 3)


def unwrap_to_reference(
    q_leader,
    q_ref,
    wrap_joints: tuple[int, ...] = WRAP_JOINTS,
    *,
    limits: tuple[tuple[float, float], ...] | None = XARM7_JOINT_LIMITS_RAD,
    margin: float = JOINT_LIMIT_MARGIN_RAD,
) -> tuple[np.ndarray, np.ndarray]:
    """``(q_unwrapped, k)``: for each index in ``wrap_joints`` add ``2*pi*k`` with
    ``k in {-1, 0, 1}`` minimising the distance to ``q_ref``; ``k`` is 0 elsewhere. A tie
    (exactly ±π away) keeps the branch closest to the leader's own reading (k = 0).

    Limit-aware (2026-09-09 review): with ``limits`` (the margined xArm7 table by default;
    ``None`` = the pure nearest-branch rule) a candidate outside ``[lo, hi]`` is dropped
    whenever at least one candidate lies inside, so the check never refuses a leader
    posture that IS inside the limits because its nearest branch is not (16-gello §15.2
    item 11). The same helper serves the launch check / preview and the engage rule."""
    q = np.array(q_leader, dtype=np.float64).reshape(-1)
    ref = np.asarray(q_ref, dtype=np.float64).reshape(-1)
    if q.shape != ref.shape:
        raise ValueError(f"q_leader {q.shape} and q_ref {ref.shape} differ in length")
    lo = hi = None
    if limits is not None:
        lo, hi = joint_limit_bounds(limits, margin)
    k = np.zeros(q.shape[0], dtype=np.int64)
    for j in wrap_joints:
        if j >= q.shape[0]:
            continue
        cands = [0, -1, 1]  # k = 0 first: it wins every tie
        if lo is not None and j < lo.shape[0]:
            inside = [c for c in cands if lo[j] <= q[j] + TWO_PI * c <= hi[j]]
            if inside:
                cands = inside
        best_k, best_d = cands[0], abs(q[j] + TWO_PI * cands[0] - ref[j])
        for cand in cands[1:]:
            d = abs(q[j] + TWO_PI * cand - ref[j])
            if d < best_d - 1e-12:
                best_k, best_d = cand, d
        k[j] = best_k
        q[j] += TWO_PI * best_k
    return q, k


def apply_unwrap(q_leader, k) -> np.ndarray:
    """``q_leader + 2*pi*k`` (continuity while tracking: the branch fixed at engagement)."""
    return np.asarray(q_leader, dtype=np.float64) + TWO_PI * np.asarray(k, dtype=np.float64)


def joint_limit_bounds(
    limits: tuple[tuple[float, float], ...] = XARM7_JOINT_LIMITS_RAD,
    margin: float = JOINT_LIMIT_MARGIN_RAD,
) -> tuple[np.ndarray, np.ndarray]:
    """``(lo, hi)`` arrays with the margin applied."""
    lo = np.array([a for a, _ in limits], dtype=np.float64) + margin
    hi = np.array([b for _, b in limits], dtype=np.float64) - margin
    return lo, hi


def clip_to_joint_limits(
    q,
    limits: tuple[tuple[float, float], ...] = XARM7_JOINT_LIMITS_RAD,
    margin: float = JOINT_LIMIT_MARGIN_RAD,
) -> np.ndarray:
    """Clip the 7 joints into ``[lo + margin, hi - margin]`` (§6.2: joints 2/4/6 are
    clipped; the ±2π joints only ever hit their limits after an unwrap)."""
    lo, hi = joint_limit_bounds(limits, margin)
    return np.clip(np.asarray(q, dtype=np.float64)[:7], lo, hi)


@dataclass(frozen=True)
class JointLimitViolation:
    """First joint outside the (margined) limits: ``joint`` is 1-based for people."""

    joint: int
    value_rad: float
    lo_rad: float
    hi_rad: float

    def describe(self) -> str:
        sym = abs(self.lo_rad + self.hi_rad) < 1e-9
        limit = f"±{self.hi_rad:.2f}" if sym else f"[{self.lo_rad:.2f}, {self.hi_rad:.2f}]"
        return f"joint {self.joint} = {self.value_rad:.2f} rad, limit {limit}"


def joint_limit_violation(
    q,
    limits: tuple[tuple[float, float], ...] = XARM7_JOINT_LIMITS_RAD,
    margin: float = JOINT_LIMIT_MARGIN_RAD,
) -> JointLimitViolation | None:
    """The first joint outside the margined limits (for the 409 / preview text
    ``GELLO posture outside the Manipulation Arm's joint limits (joint N = x.xx rad,
    limit ±y.yy)``), or None."""
    lo, hi = joint_limit_bounds(limits, margin)
    qq = np.asarray(q, dtype=np.float64)[:7]
    for j in range(7):
        if not (lo[j] <= qq[j] <= hi[j]) or not math.isfinite(qq[j]):
            return JointLimitViolation(j + 1, float(qq[j]), float(lo[j]), float(hi[j]))
    return None


class EngageMachine:
    """The engagement state machine of one gello session (16-gello §6.1).

    Feed it every tick: ``step(now, sample, q_meas, q_cmd, plan_active=...)`` with the
    reader's newest :class:`GelloSample` (or None), the Manipulation Arm's measured joints
    (7 or 8 values; the first 7 are used) and the last commanded joints. Read ``state``,
    ``detail`` (operator-facing reason), :meth:`lag_rad` / :meth:`max_lag_rad` (unwrap(leader)
    - measured, per joint) and, while ``tracking``, :meth:`target` (the unwrapped, clipped
    leader joints the loop streams). Events: :meth:`pause` / :meth:`resume`
    (``gello_pause`` / ``gello_resume``, idempotent), :meth:`on_fault`, :meth:`on_motion_start`
    / :meth:`on_motion_end`. Thread-free: the control loop owns it.
    """

    def __init__(self, cfg: GelloConfig) -> None:
        self.cfg = cfg
        self.state: GelloState = "no_leader"
        self.detail: str = "no leader sample yet"
        self._paused = False
        self._pause_reason = "paused"  # the operator-facing reason, kept across a motion window
        self._motion = False
        self._k: np.ndarray | None = None  # ±2π branch fixed at engagement (tracking only)
        self._lag: np.ndarray | None = None
        self._last_sample: GelloSample | None = None
        self._last_now: float | None = None
        self.transitions: int = 0

    # -- readouts ---------------------------------------------------------------------------
    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def reason(self) -> str:
        return self.detail

    def lag_rad(self) -> np.ndarray | None:
        """``unwrap(leader) - measured`` per joint from the last step, None without a leader."""
        return None if self._lag is None else self._lag.copy()

    def max_lag_rad(self) -> float | None:
        return None if self._lag is None else float(np.max(np.abs(self._lag)))

    def target(self, sample: GelloSample | None = None) -> np.ndarray | None:
        """The 7 joints to stream while ``tracking`` (unwrapped with the engagement branch,
        clipped to the limits); None in every other state."""
        sample = self._last_sample if sample is None else sample
        if self.state != "tracking" or sample is None or self._k is None:
            return None
        return clip_to_joint_limits(apply_unwrap(sample.q, self._k))

    # -- events -----------------------------------------------------------------------------
    def pause(self, reason: str = "paused by operator") -> None:
        """``gello_pause`` (idempotent) — sticky until :meth:`resume`."""
        self._paused = True
        self._pause_reason = reason
        self._k = None
        if not self._motion:
            self._set("paused", reason)

    def resume(self) -> None:
        """``gello_resume`` (idempotent): drop the latch; the next :meth:`step` (or
        :meth:`on_motion_end`) runs the engage rule."""
        if not self._paused:
            return
        self._paused = False
        if not self._motion:
            self._set("out_of_sync", "resumed - waiting for the engage rule")

    def on_fault(self, detail: str = "arm fault") -> None:
        """A ``FaultEvent`` / RECOVERING of the Manipulation Arm forces ``paused``."""
        self.pause(f"paused: {detail}")

    def on_motion_start(self) -> None:
        """A twin-planned motion took the arm (launch, R, Go to profile, return)."""
        self._motion = True
        self._k = None
        self._set("motion", "planned motion owns the Manipulation Arm")

    def on_motion_end(self, q_meas) -> GelloState:
        """The plan retired: paused if a pause is latched, else the engage rule against the
        last sample seen (fresh) — the launch motion's "engage on arrival"."""
        self._motion = False
        if self._paused:
            self._set("paused", self._pause_reason)
            return self.state
        if self._last_sample is None or self._last_now is None:
            self._set("no_leader", "no leader sample yet")
            return self.state
        if not self._leader_ok(self._last_sample, self._last_now):
            self._set("no_leader", self._leader_reason(self._last_sample, self._last_now))
            return self.state
        self._engage_rule(self._last_sample, q_meas)
        return self.state

    # -- the tick ---------------------------------------------------------------------------
    def step(
        self,
        now: float,
        sample: GelloSample | None,
        q_meas,
        q_cmd,
        plan_active: bool = False,
        paused_request: bool | None = None,
    ) -> GelloState:
        """Evaluate one tick. ``paused_request`` True/False = a pause / resume arrived
        this tick (None = neither). Returns the new state."""
        if paused_request is True:
            self.pause()
        elif paused_request is False:
            self.resume()
        if sample is not None:
            self._last_sample, self._last_now = sample, now
        # motion (a plan owns the arm) wins over everything: nothing follows meanwhile
        if plan_active:
            if not self._motion:
                self.on_motion_start()
            self._update_lag(sample, now, q_meas)
            return self.state
        if self._motion:  # the plan retired since the last tick
            self._motion = False
            if not self._paused:
                self._set("out_of_sync", "planned motion ended - waiting for the engage rule")
        if self._paused:
            if self.state != "paused":
                self._set("paused", self._pause_reason)
            self._update_lag(sample, now, q_meas)
            return self.state
        if sample is None or not self._leader_ok(sample, now):
            self._k = None
            self._lag = None
            self._set("no_leader", self._leader_reason(sample, now))
            return self.state
        if self.state == "tracking" and self._k is not None:
            q_leader = apply_unwrap(sample.q, self._k)
            self._lag = q_leader - np.asarray(q_meas, dtype=np.float64)[:7]
            dev = np.abs(q_leader - np.asarray(q_cmd, dtype=np.float64)[:7])
            j = int(np.argmax(dev))
            if dev[j] > self.cfg.leash_rad:
                self._k = None
                self._set(
                    "out_of_sync",
                    f"leash: leader {dev[j]:.2f} rad from the command (joint {j + 1}, leash "
                    f"{self.cfg.leash_rad:.2f}) - the follower holds; bring GELLO back within "
                    f"{self.cfg.engage_tol_rad:.2f} rad of the arm",
                )
            return self.state
        self._engage_rule(sample, q_meas)
        return self.state

    # -- internals --------------------------------------------------------------------------
    def _leader_ok(self, sample: GelloSample, now: float) -> bool:
        return sample.valid and (now - sample.rx_mono) <= self.cfg.stale_s

    def _leader_reason(self, sample: GelloSample | None, now: float) -> str:
        if sample is None:
            return "no leader sample yet"
        age = now - sample.rx_mono
        if age > self.cfg.stale_s:
            return f"leader sample stale ({age:.2f} s > {self.cfg.stale_s:.2f} s)"
        return "leader sample invalid (jump / uncalibrated)"

    def _engage_rule(self, sample: GelloSample, q_meas) -> None:
        meas = np.asarray(q_meas, dtype=np.float64)[:7]
        q_unwrapped, k = unwrap_to_reference(sample.q, meas)
        lag = q_unwrapped - meas
        self._lag = lag
        j = int(np.argmax(np.abs(lag)))
        if abs(lag[j]) <= self.cfg.engage_tol_rad:
            self._k = k
            self._set("tracking", "tracking the leader")
        else:
            self._k = None
            self._set(
                "out_of_sync",
                f"leader {abs(lag[j]):.2f} rad from the arm (joint {j + 1}) - move GELLO within "
                f"{self.cfg.engage_tol_rad:.2f} rad",
            )

    def _update_lag(self, sample: GelloSample | None, now: float, q_meas) -> None:
        """Keep the per-joint delta published while paused / in motion (the panel's bars)."""
        if sample is None or not self._leader_ok(sample, now):
            self._lag = None
            return
        meas = np.asarray(q_meas, dtype=np.float64)[:7]
        q_unwrapped, _ = unwrap_to_reference(sample.q, meas)
        self._lag = q_unwrapped - meas

    def _set(self, state: GelloState, detail: str) -> None:
        if state != self.state:
            self.transitions += 1
        self.state = state
        self.detail = detail


__all__ = [
    "GelloState",
    "TWO_PI",
    "WRAP_JOINTS",
    "XARM7_JOINT_LIMITS_RAD",
    "JOINT_LIMIT_MARGIN_RAD",
    "JointLimitViolation",
    "EngageMachine",
    "unwrap_to_reference",
    "apply_unwrap",
    "joint_limit_bounds",
    "clip_to_joint_limits",
    "joint_limit_violation",
]
