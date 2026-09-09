"""Full-travel rail sweep that gates the ``home_rail`` maintenance op (phase-09c).

``home_rail`` (``set_linear_track_back_origin``) drives the carriage to the
track's zero end from an UNKNOWN position - the register is meaningless while
the track is unhomed - so before the runtime writes anything it checks, in a
dedicated :class:`apollo_mavis_v2_sim.DigitalTwin`, that the whole travel
``0 .. travel_m`` is collision-free at the arm's CURRENT joint posture
(``sample.q``) against the other arm(s) at THEIR last monitor sample.

Recipe (03-sim ``check_config_violations`` / ``pair_distance``; D4 margins):

1. one private twin per checker, built lazily from
   ``REGISTRY.build(twin_scene, SceneOverrides(microphones, base_pose))`` at
   ``inflation_m`` (default 0.025 m: a blind sweep from an unknown start gets
   the guardrail's debug margin, while the gate twin keeps
   ``safety.geom_inflation_m``) plus ``safety.allowed_pairs_extra``; guarded by
   a lock (a twin is single-threaded) and NEVER shared with the overlay's or a
   session's twin;
2. every OTHER arm is posed from its sample: ``q[:7]`` and, when railed, its
   rail position (``rail_pos_m``; ``None`` -> ``rail_fallback_m[arm]`` and an
   ``assumptions`` entry) mapped through ``rail_flip`` (``q_sim = 0.65 -
   q_track``); an arm without any sample keeps the scene keyframe (assumption
   recorded);
3. the target arm gets ``q[:7]`` from its sample and its rail slot steps
   through ``linspace(0, travel_m, round(travel_m / step_m) + 1)`` (5 mm ->
   131 checks); at every step ``check_config_violations`` decides
   blocked / clear and ``mj_geomDistance`` over the twin's monitored pairs
   records the tightest pair (``min_clearance_*``).

The verdict is core's :class:`RailSweepVerdict`; the runtime hands
``q_checked`` to the hardware monitor as ``expected_q`` so the poll thread
refuses (zero writes) if the arm moved between the sweep and the homing.

**Phase-09d - planning on the same twin.** When the current posture is NOT
sweep-clear the rail-homing job first moves the arm to a rail-safe posture
(``devices/rail_homing.py``); the two twin services it needs live here because
they share this checker's private twin and lock:

* :meth:`RailSweepChecker.plan_path` - joint-space RRT-Connect
  (``apollo_mavis_v2_sim.planner.ResetPlanner`` over THIS twin, so the plan is
  validated at the sweep's 0.025 m margin) from the current 7 joints to a
  candidate posture with the arm's rail slot LOCKED at ``rail_fallback_m``
  (the carriage is unknown, the job never moves it: every waypoint carries the
  fallback) and the other arm posed from its sample exactly as in ``check``;
* :meth:`RailSweepChecker.check_path` - the position-agnostic validation
  (:class:`PathVerdict`): the path is densified to ``max_step_rad`` (a
  shortcut plan is often just start -> goal) and EVERY configuration is checked
  at EVERY rail position of the sweep (131 at 5 mm) under the planner's
  start-posture hysteresis (11-safety §9): a pair the CURRENT posture already
  violates at a given rail position may stay violated only while it does not
  get closer than ``PATH_HYSTERESIS_TOL_M`` below its reference distance (the
  arm is physically there - it moves away, never in); the reference is the
  start distance and only ever OPENS UP as the pair gains clearance (never
  lowered, so a slow creep of less than the tolerance per configuration cannot
  accumulate), a whitelisted pair in contact (``dist <= 0``) blocks outright,
  it is re-armed once clear, and any NEW violation at any position blocks. The goal
  posture itself must be sweep-clear (checked by the caller with ``check``), so
  every whitelist is empty by the last waypoint. This check is the ONLY safety
  basis of the pre-positioning motion (the gate twin guesses the carriage).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from apollo_mavis_v2_core import PlanRequest, PlanResult, WorkcellConfig
from apollo_mavis_v2_core.protocol import RailSweepVerdict

logger = logging.getLogger(__name__)

RAIL_TRAVEL_M = 0.65  # linear track travel (core se3.RAIL_TRAVEL_M)
DEFAULT_SWEEP_INFLATION_M = 0.025  # D4: the guardrail's debug margin
DEFAULT_SWEEP_STEP_M = 0.005  # D4: 5 mm -> 131 checks over 0.65 m
CLEARANCE_DISTMAX_M = 0.10  # pairs farther apart than this are not reported
PATH_MAX_STEP_RAD = 0.05  # phase-09d: path densification for check_path (planner edge resolution)
PATH_HYSTERESIS_TOL_M = 0.002  # a whitelisted pair may not get closer by more than this
PLAN_TIMEOUT_S = 5.0  # RRT-Connect budget per candidate posture
N_JOINTS = 7


@dataclass
class PathVerdict:
    """Position-agnostic verdict of a planned joint path (phase-09d; module docstring).

    ``clear`` iff every densified configuration is collision-free at every rail
    position under the start-posture hysteresis. On a block ``bad_waypoint`` is
    the index of the planner waypoint the blocking configuration belongs to
    (the segment start; 0 = the start posture itself), ``bad_rail_m`` the rail
    position and ``bad_pair`` the blocking geometry pair. ``checked_configs`` /
    ``checked_rail_positions`` size the check (131 positions at 5 mm);
    ``assumptions`` mirror the sweep's (other arm's rail fallback ...).
    """

    clear: bool
    waypoints: int
    checked_configs: int
    checked_rail_positions: int
    bad_waypoint: int | None = None
    bad_rail_m: float | None = None
    bad_pair: list[str] = field(default_factory=list)
    detail: str = ""
    assumptions: list[str] = field(default_factory=list)


class _QOnly:
    """Minimal ``ArmState``-like carrier (``.q``) for ``DigitalTwin.sync``."""

    __slots__ = ("q",)

    def __init__(self, q: np.ndarray) -> None:
        self.q = np.asarray(q, dtype=np.float64)


def densify_path(
    waypoints: Sequence[Sequence[float]], max_step_rad: float = PATH_MAX_STEP_RAD
) -> list[tuple[int, np.ndarray]]:
    """``(waypoint index, q)`` for every configuration along the piecewise-linear
    path at joint steps of at most ``max_step_rad`` (rail slot excluded from the
    step count); the waypoints themselves are included, the last one last."""
    qs = [np.asarray(list(w), dtype=np.float64) for w in waypoints]
    if not qs:
        return []
    out: list[tuple[int, np.ndarray]] = []
    for i, (a, b) in enumerate(zip(qs[:-1], qs[1:], strict=True)):
        n = max(1, int(np.ceil(float(np.max(np.abs(b[:N_JOINTS] - a[:N_JOINTS]))) / max_step_rad)))
        for k in range(n):
            out.append((i, a + (b - a) * (k / n)))
    out.append((len(qs) - 1, qs[-1]))
    return out


def sweep_positions(travel_m: float = RAIL_TRAVEL_M, step_m: float = DEFAULT_SWEEP_STEP_M):
    """Rail positions checked: ``0, step, ..., travel`` (both ends included)."""
    n = int(round(float(travel_m) / float(step_m))) + 1
    return np.linspace(0.0, float(travel_m), max(2, n))


class RailSweepChecker:
    """Lazily-built private twin + the sweep (module docstring).

    ``check`` raises ``KeyError`` for an arm the twin scene lacks and
    ``ValueError`` for an arm without a rail; the sim extra missing raises
    ``ImportError`` at first use (the caller maps it to a 409).
    """

    def __init__(
        self,
        workcell_cfg: WorkcellConfig,
        twin_scene: str,
        inflation_m: float = DEFAULT_SWEEP_INFLATION_M,
        step_m: float = DEFAULT_SWEEP_STEP_M,
        *,
        rail_flip: bool = False,
        travel_m: float = RAIL_TRAVEL_M,
    ) -> None:
        self.workcell = workcell_cfg
        self.scene_id = str(twin_scene)
        self.inflation_m = float(inflation_m)
        self.step_m = float(step_m)
        self.rail_flip = bool(rail_flip)
        self.travel_m = float(travel_m)
        self._lock = threading.Lock()
        self._twin: Any = None
        self._mujoco: Any = None
        self._planner: Any = None  # phase-09d: lazy ResetPlanner over the private twin
        self.sweeps = 0  # verdicts produced (tests / diagnostics)

    # -- twin ------------------------------------------------------------------------------
    def _build(self) -> Any:
        import mujoco
        from apollo_mavis_v2_sim import REGISTRY, DigitalTwin, SceneOverrides

        from ..streams.twin_overlay import base_pose_overrides

        known = {a.id for a in REGISTRY.descriptor(self.scene_id).arms}
        overrides = SceneOverrides(
            microphones={a.id: bool(a.microphone) for a in self.workcell.arms if a.id in known},
            base_pose={k: v for k, v in base_pose_overrides(self.workcell).items() if k in known},
        )
        twin = DigitalTwin(
            REGISTRY.build(self.scene_id, overrides),
            inflation_m=self.inflation_m,
            allowed_pairs_extra=self.workcell.safety.allowed_pairs_extra,
            hysteresis_m=self.workcell.safety.hysteresis_m,
        )
        self._mujoco = mujoco
        logger.info(
            "rail sweep twin built: scene %r, inflation %.3f m, step %.3f m, %d monitored pairs",
            self.scene_id,
            self.inflation_m,
            self.step_m,
            len(twin.monitored_pairs),
        )
        return twin

    @property
    def twin(self) -> Any:
        """The private twin (built on first access; hold ``_lock`` to use it)."""
        if self._twin is None:
            self._twin = self._build()
        return self._twin

    @property
    def built(self) -> bool:
        return self._twin is not None

    # -- the sweep -------------------------------------------------------------------------
    def check(
        self,
        arm_id: str,
        samples: Mapping[str, Any],
        rail_fallback_m: Mapping[str, float] | None = None,
    ) -> RailSweepVerdict:
        """Sweep ``arm_id``'s rail over the full travel at its sampled posture.

        ``samples`` = ``{arm_id: ArmMonitorSample-like}`` (``q``, ``rail_pos_m``,
        ``rail_homed``, ``seq``); the target arm's sample is REQUIRED. Other arms
        are posed from their sample (rail -> fallback / keyframe as documented).
        """
        fallback = dict(rail_fallback_m or {})
        target = samples.get(arm_id)
        if target is None:
            raise ValueError(f"{arm_id}: no monitor sample - the sweep needs the current posture")
        q_target = np.asarray(list(target.q)[:7], dtype=np.float64)
        if q_target.shape != (7,):
            raise ValueError(f"{arm_id}: sample has {q_target.shape[0]} joints, expected 7")
        with self._lock:
            twin = self.twin
            mujoco = self._mujoco
            addr = self._rail_addr(arm_id)
            q_full = np.array(twin.data.qpos, dtype=np.float64)  # keyframe for unsampled arms
            assumptions, other_arms = self._pose_others(arm_id, samples, fallback, q_full)
            q_full[addr.qpos_adr[:7]] = q_target
            rail_adr = int(addr.qpos_adr[7])

            first_blocked_m: float | None = None
            first_blocked_pair: list[str] = []
            min_d = np.inf
            min_at: float | None = None
            min_pair: list[str] = []
            positions = sweep_positions(self.travel_m, self.step_m)
            model, data = twin.model, twin.data
            saved = np.array(data.qpos)
            try:
                for s in positions:
                    q_full[rail_adr] = float(s)
                    violations = twin.check_config_violations(q_full)
                    if violations and first_blocked_m is None:
                        pair, _dist = min(violations, key=lambda v: v[1])
                        first_blocked_m = float(s)
                        first_blocked_pair = sorted(pair)
                    # tightest monitored pair at this step (mj_geomDistance early-outs on
                    # bounding spheres, so the whole sweep stays in the tens of ms)
                    data.qpos[:] = q_full
                    mujoco.mj_kinematics(model, data)
                    for g1, g2 in twin.monitored_pairs:
                        d = mujoco.mj_geomDistance(
                            model, data, int(g1), int(g2), CLEARANCE_DISTMAX_M, None
                        )
                        if d < min_d:
                            min_d = float(d)
                            min_at = float(s)
                            min_pair = sorted(
                                (
                                    twin.allowed.label_of_geom(int(g1)),
                                    twin.allowed.label_of_geom(int(g2)),
                                )
                            )
            finally:
                data.qpos[:] = saved
                mujoco.mj_kinematics(model, data)
            self.sweeps += 1
        return RailSweepVerdict(
            scene_id=self.scene_id,
            inflation_m=self.inflation_m,
            step_m=self.step_m,
            travel_m=self.travel_m,
            clear=first_blocked_m is None,
            first_blocked_m=first_blocked_m,
            first_blocked_pair=first_blocked_pair,
            min_clearance_m=None if not np.isfinite(min_d) else min_d,
            min_clearance_at_m=min_at,
            min_clearance_pair=min_pair,
            q_checked=[float(v) for v in q_target],
            other_arms=other_arms,
            assumptions=assumptions,
            sample_seq=int(getattr(target, "seq", 0) or 0),
        )

    # -- shared posing ---------------------------------------------------------------------
    def _rail_addr(self, arm_id: str) -> Any:
        addr = self.twin.addr[arm_id]  # KeyError: arm not in the twin scene
        if not addr.has_rail:
            raise ValueError(f"{arm_id}: the twin scene {self.scene_id!r} has no rail for it")
        return addr

    def _pose_others(
        self,
        arm_id: str,
        samples: Mapping[str, Any],
        fallback: Mapping[str, float],
        q_full: np.ndarray,
    ) -> tuple[list[str], dict[str, list[float]]]:
        """Write every OTHER arm's sampled posture into ``q_full`` (module docstring
        step 2; ``_lock`` held) -> ``(assumptions, other_arms)``."""
        twin = self.twin
        assumptions: list[str] = []
        other_arms: dict[str, list[float]] = {}
        for other_id, other in twin.addr.arms.items():
            if other_id == arm_id:
                continue
            sample = samples.get(other_id)
            if sample is None or len(sample.q) < 7:
                assumptions.append(f"{other_id}: no monitor sample - keyframe posture assumed")
                other_arms[other_id] = [float(v) for v in q_full[other.qpos_adr]]
                continue
            q_other = np.asarray(list(sample.q)[:7], dtype=np.float64)
            q_full[other.qpos_adr[:7]] = q_other
            if other.has_rail:
                pos = getattr(sample, "rail_pos_m", None)
                if pos is None:
                    fb = float(fallback.get(other_id, 0.0))
                    homed = getattr(sample, "rail_homed", False)
                    why = "rail not enabled" if homed else "rail unknown"
                    assumptions.append(f"{other_id} {why} - used fallback {fb:.2f} m")
                    pos = fb
                elif self.rail_flip:
                    pos = self.travel_m - float(pos)
                q_full[other.qpos_adr[7]] = min(self.travel_m, max(0.0, float(pos)))
            other_arms[other_id] = [float(v) for v in q_full[other.qpos_adr]]
        if self.rail_flip:
            assumptions.append("rail_flip: other arms' rail positions mirrored (0.65 - q)")
        return assumptions, other_arms

    # -- phase-09d: candidate postures, planning, position-agnostic path check --------------
    def candidate_postures(self, arm_id: str) -> list[tuple[str, list[float]]]:
        """``[(source, q7), ...]`` in trial order: the scene keyframe's 7 joints for
        this arm (``"keyframe"``, key 0 - the factory-zero folded posture) and the
        ``<arm>_home`` keyframe (``"home"``) when the model has one."""
        with self._lock:
            twin = self.twin
            mujoco = self._mujoco
            addr = self._rail_addr(arm_id)
            model = twin.model
            out: list[tuple[str, list[float]]] = []
            if model.nkey > 0:
                out.append(("keyframe", [float(v) for v in model.key_qpos[0][addr.qpos_adr[:7]]]))
            key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, f"{arm_id}_home")
            if key_id >= 0:
                q_home = [float(v) for v in model.key_qpos[key_id][addr.qpos_adr[:7]]]
                if not out or not np.allclose(q_home, out[0][1], atol=1e-6):
                    out.append(("home", q_home))
        return out

    def plan_path(
        self,
        arm_id: str,
        q_start7: Sequence[float],
        q_goal7: Sequence[float],
        samples: Mapping[str, Any],
        rail_fallback_m: Mapping[str, float] | None = None,
        *,
        timeout_s: float = PLAN_TIMEOUT_S,
        speed_scale: float = 0.1,
    ) -> PlanResult:
        """RRT-Connect on this twin from ``q_start7`` to ``q_goal7`` with the arm's
        rail slot LOCKED at its fallback (module docstring). Other arms are posed
        from ``samples`` like ``check``. Every returned waypoint is 8-dof with the
        fallback in the rail slot; the twin's own qpos is restored afterwards.
        ``speed_scale`` is the speed the plan will run at (the rail-homing job's 10 %):
        a pinched start's escape is judged tick by tick at it (``PlanRequest``)."""
        from apollo_mavis_v2_sim.planner import ResetPlanner

        fallback = dict(rail_fallback_m or {})
        q_start = np.asarray(list(q_start7)[:7], dtype=np.float64)
        q_goal = np.asarray(list(q_goal7)[:7], dtype=np.float64)
        if q_start.shape != (7,) or q_goal.shape != (7,):
            raise ValueError(f"{arm_id}: plan_path needs 7-joint start and goal postures")
        with self._lock:
            twin = self.twin
            mujoco = self._mujoco
            addr = self._rail_addr(arm_id)
            rail_m = min(self.travel_m, max(0.0, float(fallback.get(arm_id, 0.0))))
            saved = np.array(twin.data.qpos, dtype=np.float64)
            q_full = saved.copy()
            self._pose_others(arm_id, samples, fallback, q_full)
            q_full[addr.qpos_adr[:7]] = q_start
            q_full[addr.qpos_adr[7]] = rail_m
            if self._planner is None:
                self._planner = ResetPlanner(twin)
            planner = self._planner
            lo, hi = planner._jnt_range[arm_id]
            locked_lo, locked_hi = lo.copy(), hi.copy()
            locked_lo[7] = locked_hi[7] = rail_m  # the RRT never samples a rail move
            try:
                # the planner reads the other arms from the twin's measured context
                twin.sync({a: _QOnly(q_full[ad.qpos_adr]) for a, ad in twin.addr.arms.items()})
                planner._jnt_range[arm_id] = (locked_lo, locked_hi)
                result = planner.plan(
                    PlanRequest(
                        q_start={arm_id: [*map(float, q_start), rail_m]},
                        q_goal={arm_id: [*map(float, q_goal), rail_m]},
                        timeout_s=float(timeout_s),
                        speed_scale=float(speed_scale),
                    )
                )
            finally:
                planner._jnt_range[arm_id] = (lo, hi)
                twin.sync({a: _QOnly(saved[ad.qpos_adr]) for a, ad in twin.addr.arms.items()})
                twin.data.qpos[:] = saved
                mujoco.mj_kinematics(twin.model, twin.data)
        if result.ok and arm_id in result.waypoints:
            pinned = []
            for w in result.waypoints[arm_id]:
                q = [float(v) for v in w][:7]
                pinned.append([*q, rail_m])
            result = result.model_copy(update={"waypoints": {arm_id: pinned}})
        return result

    def check_path(
        self,
        arm_id: str,
        waypoints: Sequence[Sequence[float]],
        samples: Mapping[str, Any],
        rail_fallback_m: Mapping[str, float] | None = None,
        *,
        max_step_rad: float = PATH_MAX_STEP_RAD,
    ) -> PathVerdict:
        """Position-agnostic validation of a planned joint path (module docstring):
        every densified configuration x every rail position of the sweep x the
        other arm at its sample / fallback, under the start-posture hysteresis."""
        fallback = dict(rail_fallback_m or {})
        wps = [np.asarray(list(w), dtype=np.float64) for w in waypoints]
        if not wps:
            raise ValueError(f"{arm_id}: check_path needs at least one waypoint")
        if any(w.shape[0] < 7 for w in wps):
            raise ValueError(f"{arm_id}: every waypoint needs at least 7 joints")
        positions = sweep_positions(self.travel_m, self.step_m)
        configs = densify_path(wps, max_step_rad)
        with self._lock:
            twin = self.twin
            mujoco = self._mujoco
            addr = self._rail_addr(arm_id)
            q_full = np.array(twin.data.qpos, dtype=np.float64)
            assumptions, _others = self._pose_others(arm_id, samples, fallback, q_full)
            joint_adr = addr.qpos_adr[:7]
            rail_adr = int(addr.qpos_adr[7])
            saved = np.array(twin.data.qpos)
            bad: tuple[int, float, list[str], str] | None = None
            checked = 0
            try:
                # start-posture hysteresis PER rail position: pairs the current posture
                # already violates there (the arm is physically at one of these positions)
                whitelist: list[dict[tuple[str, str], float]] = []
                q_full[joint_adr] = wps[0][:7]
                for s in positions:
                    q_full[rail_adr] = float(s)
                    viol = twin.check_config_violations(q_full)
                    whitelist.append({tuple(sorted(p)): float(d) for p, d in viol})
                for wp_index, q in configs:
                    q_full[joint_adr] = q[:7]
                    for si, s in enumerate(positions):
                        q_full[rail_adr] = float(s)
                        checked += 1
                        viol = twin.check_config_violations(q_full)
                        wl = whitelist[si]
                        present: set[tuple[str, str]] = set()
                        for pair, dist in viol:
                            key = tuple(sorted(pair))
                            present.add(key)
                            d0 = wl.get(key)
                            if d0 is None:
                                bad = (
                                    wp_index,
                                    float(s),
                                    list(key),
                                    f"new violation {key[0]} / {key[1]} ({dist * 100:.1f} cm)",
                                )
                                break
                            if dist <= 0.0:
                                # a whitelisted pair may be tight, never in contact: the
                                # inflated geometries touching / overlapping is a hard block
                                bad = (
                                    wp_index,
                                    float(s),
                                    list(key),
                                    f"{key[0]} / {key[1]} in contact ({dist * 100:.1f} cm)",
                                )
                                break
                            if dist < d0 - PATH_HYSTERESIS_TOL_M:
                                bad = (
                                    wp_index,
                                    float(s),
                                    list(key),
                                    f"{key[0]} / {key[1]} gets closer ({d0 * 100:.1f} -> "
                                    f"{dist * 100:.1f} cm)",
                                )
                                break
                            # the reference only ever OPENS UP: once the pair gained
                            # clearance it may not fall back below that (minus the
                            # tolerance), so a slow creep of < tol per configuration
                            # can never accumulate (the reference is never lowered)
                            wl[key] = max(d0, float(dist))
                        if bad is not None:
                            break
                        for key in list(wl):
                            if key not in present:
                                del wl[key]  # re-armed once clear: a later dip blocks
                    if bad is not None:
                        break
            finally:
                twin.data.qpos[:] = saved
                mujoco.mj_kinematics(twin.model, twin.data)
        if bad is None:
            return PathVerdict(
                clear=True,
                waypoints=len(wps),
                checked_configs=len(configs),
                checked_rail_positions=len(positions),
                detail=(
                    f"path clear: {len(configs)} configurations x {len(positions)} rail positions "
                    f"at {self.inflation_m * 1000:.0f} mm inflation"
                ),
                assumptions=assumptions,
            )
        wp_index, s, pair, why = bad
        return PathVerdict(
            clear=False,
            waypoints=len(wps),
            checked_configs=len(configs),
            checked_rail_positions=len(positions),
            bad_waypoint=wp_index,
            bad_rail_m=s,
            bad_pair=pair,
            detail=(
                f"path blocked after waypoint {wp_index} at rail {s:.3f} m: {why} "
                f"({checked} of {len(configs) * len(positions)} checks done)"
            ),
            assumptions=assumptions,
        )


def describe_verdict(verdict: RailSweepVerdict, *, dry_run: bool) -> str:
    """Operator-facing one-liner for ``ArmMaintenanceResult.detail``."""
    if verdict.clear:
        tight = (
            f"; tightest pair {verdict.min_clearance_pair[0]} / {verdict.min_clearance_pair[1]} "
            f"at {verdict.min_clearance_m * 100:.1f} cm (rail {verdict.min_clearance_at_m:.3f} m)"
            if verdict.min_clearance_m is not None and len(verdict.min_clearance_pair) == 2
            else ""
        )
        text = (
            f"rail sweep clear: the full 0-{verdict.travel_m:g} m travel at the current posture "
            f"is collision-free at {verdict.inflation_m * 1000:.0f} mm inflation{tight}"
        )
        return text + (" (dry run, nothing written)" if dry_run else "")
    pair = " / ".join(verdict.first_blocked_pair) or "unknown pair"
    at = f"{verdict.first_blocked_m:.3f} m" if verdict.first_blocked_m is not None else "?"
    return (
        f"home_rail refused: rail sweep blocked at {at} ({pair}) - fold the arm into a tighter "
        "posture (xArm Studio) and retry; nothing was written"
    )


__all__ = [
    "CLEARANCE_DISTMAX_M",
    "DEFAULT_SWEEP_INFLATION_M",
    "DEFAULT_SWEEP_STEP_M",
    "PATH_HYSTERESIS_TOL_M",
    "PATH_MAX_STEP_RAD",
    "PLAN_TIMEOUT_S",
    "RAIL_TRAVEL_M",
    "PathVerdict",
    "RailSweepChecker",
    "densify_path",
    "describe_verdict",
    "sweep_positions",
]
