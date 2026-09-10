"""Fuzz: the whole reset / return flow proven HEADLESS on the mavis_v2 twin (04-runtime
§10.5; 11-safety §7.1 / §9; 03-sim §10). Every real-cell attempt of the flow so far failed
(2026-09-08 23:16 both arms executed at once; 2026-09-09 01:14 a whitelisted pinched pair),
so the operator asked for the flow to be exercised in simulation FIRST, from many random
"pinched" two-arm starts, before the next live try.

What runs per case (the manager's rule, headless):

* ``SessionManager._return_goals`` (imported - it needs only ``workcell.states()`` and
  ``loop.gripper_arms`` of a session, faked here) splits a profile into the JOINTS goal
  (carriages held) and the FULL goal; the phase list of ``_return_to_initial_motion`` and
  the per-phase skip / plan of ``_return_phase`` / ``_plan_return`` are replicated line for
  line (they need a live session and are documented as such);
* each phase is planned with ``twin.plan`` at the session's ``speed_scale`` and executed
  ONE ARM AT A TIME in ``PlanResult.arm_order`` (``SessionManager._ordered_waypoints`` /
  ``_moving_arms``, imported) with the real ``PlanExecutor`` at the hardware caps
  (0.006 rad/tick, rail 0.0005 m/tick, 4 mm Cartesian, x speed), the loop's per-tick
  joint step cap, EVERY tick through the real ``SafetyGate.filter`` with the servo
  assumed to follow (q_meas trails the gated command by one tick), the loop's
  ``plan_gate_hold_s`` abort (3 s of consecutive held ticks, ``_plan_gate_watch``'s rule),
  the loop's arrival rule (``_confirm_plan_arrivals``: a held final step re-enters the
  executor; a goal the rail clamp cannot reach finishes at the travel end -
  ``ControlLoop._clamped_rail_goal`` is imported so both proofs judge arrival alike) and
  the manager's measured arrival check (``_arrival_error``: 1e-3 rad / 2 mm).

The twin is the one the real cell gates with: ``REGISTRY.build("mavis_v2",
SceneOverrides(microphones={"view": True}))`` - the hardware session's
``_microphone_overrides`` of ``configs/mavis_v2.yaml`` (``view.microphone: true``; the
scene default is ``false``) - and, parametrized, the mic-less plain-sim twin. On the
mic-equipped twin the microphone body is the closest cross-arm pair of ~40 % of random
starts, so the first sweeps (mic-less) never modelled those pinch geometries. The
``SafetyConfig`` is core's DEFAULTS: ``workcells.hardware.safety`` of the cell config only
pins ``enabled`` (δ = 8 mm, hysteresis 2 mm, escape epsilon 10 um).

The gate is seeded the way a live session finds it: the first tick commands the measured
posture (the rising-edge ``blocked`` event), and every pair inside the gate's hysteresis
band at the start is put into ``_block_pairs`` as well - the state a partial teleop
retreat leaves behind (``tests/test_plan_passes_gate.py::test_band_pair_gate_history_...``).

Starts, in FOUR classes judged by the START GEOMETRY the gate sees (not by the sampling
range alone; a draw whose geometry lands in another class is redrawn):

* ``sub_delta`` - some monitored pair inside the shell (< δ): the gate BLOCKS the start
  and demands an escape. The operator's failure class; the majority of the run
  (``MAVIS_FUZZ_N``, default 24 starts);
* ``band`` - no pair inside δ, some pair inside the gate's hysteresis band [δ, δ + 2 mm):
  the gate holds such a pair only through a block HISTORY (seeded here);
* ``near`` - nothing within δ + hysteresis, closest cross-arm pair in [10, 20) mm: an
  ordinary start the gate never blocks, kept to show the sweep is not tuned to pinches;
* ``normal`` - closest cross-arm pair in [20, 60] mm.

Each start is found by walking a joint-space line from a random clear configuration toward
a random far one (both arms move) and bisecting the first crossing of a random target
CROSS-ARM distance, so the pinch geometry is not hand-picked; "cross-arm" means both labels
are moved by DIFFERENT arms' joints (``twin._arms_of_label``: the static ``*_rail_base``
bodies belong to no arm and are arm-vs-fixture, excluded from the target metric though
still monitored). Goals: the seeded default posture (``profiles/seed_initial.py``,
carriages kept -> the joints phase only) and the mavis_v2 keyframe posture with its
carriages at the rail ends (-> joints phase, then carriage phase). Seeds are deterministic
(``MAVIS_FUZZ_SEED``); failing seeds are printed.

Pass criterion (2026-09-09 review: the previous "clean or honest" alone was vacuous - a
planner refusing every pinched start with ``timeout`` passed it):

* every run is CLEAN (plan ok, no gate hold, both arms arrive) or an HONEST planner
  failure (``goal_in_collision`` / ``no_escape`` / ``timeout`` naming a pair; a ``timeout``
  counts only if the planner really spent ≥ 90 % of ``PlanRequest.timeout_s`` - the
  spurious 0.7 s-of-5 s timeout of the first sweeps cannot re-enter as honest);
* ``timeout`` failures: at most ``MAX_HONEST_TIMEOUTS`` (2) over the whole sweep;
* the operator's class (``sub_delta``): at least ``MIN_CLEAN_FRACTION`` (80 %) of its runs
  are clean, at most ``MAX_NO_ESCAPE`` (2) are ``no_escape``, and EVERY clean run of it
  executed an escape (``escape_len > 0``: the first segments open the pinched pairs past
  δ + ``REARM_MARGIN_M``) - so a planner that whitelists or refuses the pinched class is
  caught;
* the sweep exercised a carriage phase.

The summary table is printed past pytest's capture so the operator can read it in any run.

Knobs: ``MAVIS_FUZZ_N`` (``sub_delta`` starts; ``band`` / ``near`` follow at a third each),
``MAVIS_FUZZ_SPEED`` (session speed scale the plan is judged and replayed at; default 1.0
= the Hardware tab's 100 %), ``MAVIS_FUZZ_SEED``. Perf-free: no timing assertion, so a
loaded machine only costs time (a genuine planner ``timeout`` is still an honest failure,
counted separately and capped).

What the first sweeps found (2026-09-09; fixed in sim ``planner.py``, 03-sim §10 items 2 /
5): the RRT planned at the bare δ and skimmed a pair at 8-10 mm for whole segments after a
correct escape, the repair pass ran out of nudges and the planner reported ``timeout``
after 0.7 s of a 5 s budget (now the RRT carries the gate's 2 mm margin and re-seeds until
the deadline; two-arm orderings are probed first; samples are start/goal-local); the
"joints" phase, documented as "carriages held", slid a carriage by up to 125.8 mm because
the RRT sampled the rail like any other dof (now a rail slot whose start and goal coincide
is pinned). The 200-start sweep then produced the one real GATE HOLD (seeds 20261013 /
20261014): the Perception Arm's camera mount 3.7 / 6.7 mm from ``grip_rail_base`` - the
Manipulation Arm's STATIC rail body - which the twin attributed to both arms by name prefix,
so the gate demanded that the Manipulation Arm open a distance none of its joints can change
and held its (correctly planned) return from the first moving tick; teleop would have frozen
the same way. Pair ownership is kinematic now (sim ``twin._arms_of_pair``; 11-safety §7.1
step 5; ``tests/test_plan_passes_gate.py`` replays the case).
"""

from __future__ import annotations

import copy
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from apollo_mavis_v2_core import ArmPosture, CommandSource, PlanRequest, SafetyConfig, StateProfile

from apollo_mavis_v2_runtime.config import ControlConfig, JogConfig
from apollo_mavis_v2_runtime.control.joint_panel import PlanExecutor
from apollo_mavis_v2_runtime.control.loop import ControlLoop
from apollo_mavis_v2_runtime.profiles.seed_initial import default_profile
from apollo_mavis_v2_runtime.safety.gate import SafetyGate
from apollo_mavis_v2_runtime.session.manager import (
    PLAN_ARRIVAL_TOL_RAD,
    PLAN_ARRIVAL_TOL_RAIL_M,
    SessionManager,
)

pytest.importorskip("apollo_mavis_v2_sim")
import mujoco  # noqa: E402
from apollo_mavis_v2_sim import REGISTRY, DigitalTwin, SceneOverrides  # noqa: E402
from apollo_mavis_v2_sim.planner import REARM_MARGIN_M  # noqa: E402
from apollo_mavis_v2_sim.twin import apply_inflation  # noqa: E402

CELL_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "mavis_v2.yaml"
ARMS = ["grip", "view"]  # SessionSpec.arms order of a hardware session (both arms, always)
MIC_LABEL = "view_microphone"

# Hardware executor caps at speed scale 1.0 (the bring-up log line; pinned against the
# hardware package by tests/test_plan_passes_gate.py::test_planner_constants_mirror_the_cell_gate).
HW_SLEW_RAD = 0.006
HW_RAIL_M = 0.0005
HW_CART_STEP_M = 0.004
HW_LEVER_ARM_M = (1.20, 1.20, 1.00, 0.75, 0.44, 0.30, 0.10)
RAIL_TRAVEL_M = 0.65
PLAN_GATE_HOLD_TICKS = 300  # HardwareSessionConfig.plan_gate_hold_s (3 s) at 100 Hz
TICK_CAP = 200_000  # runaway guard per arm (a 10 % carriage sweep is ~13 000 ticks)

N_SUB_DELTA = int(os.environ.get("MAVIS_FUZZ_N", "24"))
N_CASES = {
    "sub_delta": N_SUB_DELTA,
    "band": max(2, N_SUB_DELTA // 3),
    "near": max(2, N_SUB_DELTA // 3),
    "normal": 10,
}
SEED_OFFSET = {"sub_delta": 0, "band": 10_000, "near": 20_000, "normal": 100_000}
PINCH_CLASSES = ("sub_delta", "band", "near")  # the classes sampled against a pinch target
SPEED = float(os.environ.get("MAVIS_FUZZ_SPEED", "1.0"))
BASE_SEED = int(os.environ.get("MAVIS_FUZZ_SEED", "20260909"))
HONEST_FAILURES = ("goal_in_collision", "no_escape", "timeout")
PLAN_TIMEOUT_S = PlanRequest(q_start={}, q_goal={}).timeout_s  # per arm (core default)
TIMEOUT_HONEST_FRACTION = 0.9  # a ``timeout`` is honest only past this share of the budget
MAX_HONEST_TIMEOUTS = 2
MAX_NO_ESCAPE = 2  # boxed-in sub-δ starts the planner may refuse
MIN_CLEAN_FRACTION = 0.8  # of the sub_delta runs

# The mavis_v2 keyframe posture (j1 = π, the rest 0, carriages at the rail ends). Joints
# 1 / 3 / 5 / 7 have ±2π limits, so the same physical posture has two joint readings; the
# goal is written on the branch nearest the case's start (``keyframe_profile``) - a stored
# profile on the far branch would ask for a spurious full turn, which is a property of
# joint-space profiles, not of the return flow under test.
KEYFRAME_Q = {"view": [math.pi, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "grip": [math.pi] + [0.0] * 6}
KEYFRAME_RAIL = {"view": 0.0, "grip": 0.65}
PERIODIC_JOINTS = (0, 2, 4, 6)

# ``SessionManager._return_goals`` never touches ``self``; an uninitialised instance
# lets the test call the manager's rule verbatim without a runtime.
MANAGER_RULES = SessionManager.__new__(SessionManager)


def cell_safety_config() -> SafetyConfig:
    """The real cell's gate values: ``workcells.hardware.safety`` of configs/mavis_v2.yaml
    on top of core's defaults - the file only pins ``enabled``, so these ARE the defaults
    (δ 8 mm, hysteresis 2 mm)."""
    doc = yaml.safe_load(CELL_CONFIG.read_text())
    return SafetyConfig(**doc["workcells"]["hardware"].get("safety", {}))


# The shell this fuzz corpus was written for. The cell's own geom_inflation_m was raised
# 0.008 -> 0.025 on 2026-09-09 evening (03-sim §4.5: the twin disagrees with the real cameras
# by 15-30 mm, so an 8 mm shell was smaller than the geometry error). This file KEEPS 0.008
# because its sampler's premise stops existing at 0.025: the "near" and "normal" classes
# require a start with NO pair inside [0, delta + hysteresis), and with a 27 mm band almost
# every random far configuration in this tight cell has one, so the rejection search exhausts
# ("no near start found"). The planner / gate LOGIC it fuzzes is delta-independent; the LIVE
# shell is exercised by tests/test_plan_passes_gate.py, which reads the config.
FUZZ_INFLATION_M = 0.008


def class_ranges(cfg: SafetyConfig) -> dict[str, tuple[float, float]]:
    """Target CROSS-ARM distance range per class (m), from the gate's own numbers."""
    delta = FUZZ_INFLATION_M
    band = delta + cfg.hysteresis_m
    # Expressed relative to the band so the corpus survives a change of FUZZ_INFLATION_M;
    # at 0.008 these are the original literals exactly (band 0.010 -> near (0.010, 0.020),
    # normal (0.020, 0.060)).
    return {
        "sub_delta": (0.001, delta),
        "band": (delta, band),
        "near": (band, 2.0 * band),
        "normal": (2.0 * band, 2.0 * band + 0.040),
    }


def hw_jog(speed: float) -> JogConfig:
    return JogConfig(
        slew_rad_per_tick=HW_SLEW_RAD * speed,
        rail_m_per_tick=HW_RAIL_M * speed,
        plan_cart_step_m=HW_CART_STEP_M * speed,
        plan_lever_arm_m=HW_LEVER_ARM_M,
    )


def nearest_branch(goal_q: list[float], start_q, lo, hi) -> list[float]:
    """``goal_q`` with the 2π-periodic joints moved by whole turns onto the branch nearest
    ``start_q`` (within the limits)."""
    out = list(goal_q)
    for j in PERIODIC_JOINTS:
        best = out[j]
        for k in (-2, -1, 0, 1, 2):
            v = out[j] + k * 2.0 * math.pi
            if lo[j] <= v <= hi[j] and abs(v - start_q[j]) < abs(best - start_q[j]):
                best = v
        out[j] = best
    return out


def keyframe_profile(start: dict[str, np.ndarray], lo, hi) -> StateProfile:
    return StateProfile(
        name="mavis_v2 keyframe (rails at the ends)",
        workcell_kind="sim",
        arms={
            a: ArmPosture(
                q=nearest_branch(KEYFRAME_Q[a], start[a], lo[a], hi[a]),
                rail_pos_m=KEYFRAME_RAIL[a],
                gripper_open_frac=1.0,
            )
            for a in ARMS
        },
    )


DEFAULT_PROFILE = default_profile("sim")
GOAL_NAMES = ("default", "keyframe")


def goal_profile(name: str, case: Case, sampler: Sampler) -> StateProfile:
    """``default``: the seeded default posture, carriages kept (the joints phase only);
    ``keyframe``: the folded keyframe with the carriages at the rail ends (both phases)."""
    if name == "default":
        return DEFAULT_PROFILE
    return keyframe_profile(case.q, sampler.lo, sampler.hi)


def build_twin(mic: bool) -> DigitalTwin:
    """The cell's gate twin: the hardware session's ``_microphone_overrides`` for
    ``configs/mavis_v2.yaml`` (``mic=True``) or the mic-less plain-sim scene."""
    cfg = cell_safety_config()
    scene = REGISTRY.build("mavis_v2", SceneOverrides(microphones={"view": bool(mic)}))
    twin = DigitalTwin(scene, inflation_m=FUZZ_INFLATION_M, hysteresis_m=cfg.hysteresis_m)
    assert (MIC_LABEL in twin._geoms_of_label) == bool(mic)
    return twin


@pytest.fixture(scope="module", params=[True, False], ids=["mic", "nomic"])
def twin(request) -> DigitalTwin:
    return build_twin(request.param)


def twin_has_mic(twin: DigitalTwin) -> bool:
    return MIC_LABEL in twin._geoms_of_label


# -- start sampling -------------------------------------------------------------------------
@dataclass
class Case:
    idx: int
    cls: str  # sub_delta | band | near | normal (the START GEOMETRY, verified)
    seed: int
    q: dict[str, np.ndarray]
    cross_d: float  # min monitored cross-arm distance at the start
    cross_pair: tuple[str, str]
    inside: dict[tuple[str, str], float]  # every pair within δ + hysteresis at the start
    tight_d: float  # the tightest of ``inside`` (any pair, incl. intra-arm / fixtures) or inf


class Sampler:
    """Random two-arm starts on the twin with a controlled cross-arm pinch."""

    def __init__(self, twin: DigitalTwin, cfg: SafetyConfig) -> None:
        self.twin = twin
        self.cfg = cfg
        self.model = twin.model
        self.data = mujoco.MjData(twin.model)
        self.band_m = twin.inflation_m + cfg.hysteresis_m
        self.band_model = copy.copy(twin.model)
        apply_inflation(self.band_model, self.band_m)
        self.band_data = mujoco.MjData(self.band_model)
        # Cross-arm = the two labels are moved by DIFFERENT arms' joints (kinematic
        # ownership, twin._build_arms_of_label). A static ``*_rail_base`` belongs to no arm:
        # arm-vs-fixture, not inter-arm - excluded from the pinch target (still monitored).
        label = twin.allowed.label_of_geom
        owners = twin._arms_of_label
        self.cross = [
            (g1, g2)
            for g1, g2 in twin.monitored_pairs
            if owners.get(label(g1)) and owners.get(label(g2))
            and owners[label(g1)] != owners[label(g2)]
        ]
        assert self.cross, "no cross-arm monitored pairs"
        self.lo: dict[str, np.ndarray] = {}
        self.hi: dict[str, np.ndarray] = {}
        for a in ARMS:
            lo, hi = [], []
            for i in range(1, 8):
                r = self.model.joint(f"{a}_joint{i}").range
                lo.append(float(r[0]))
                hi.append(float(r[1]))
            r = self.model.joint(f"{a}_rail_joint").range
            lo.append(float(r[0]))
            hi.append(float(r[1]))
            self.lo[a], self.hi[a] = np.array(lo), np.array(hi)
        self.default = {a: np.array([*DEFAULT_PROFILE.arms[a].q, 0.3]) for a in ARMS}

    def full(self, q: dict[str, np.ndarray]) -> np.ndarray:
        ctx = np.array(self.twin._q_meas_full)
        for a, v in q.items():
            ctx[self.twin.addr[a].qpos_adr] = v
        return ctx

    def cross_min(self, q: dict[str, np.ndarray], distmax: float = 0.1):
        self.data.qpos[:] = self.full(q)
        mujoco.mj_kinematics(self.model, self.data)
        best, pair = distmax, (-1, -1)
        for g1, g2 in self.cross:
            d = mujoco.mj_geomDistance(self.model, self.data, g1, g2, best, None)
            if d < best:
                best, pair = d, (g1, g2)
        label = self.twin.allowed.label_of_geom
        names = tuple(sorted((label(pair[0]), label(pair[1])))) if pair[0] >= 0 else ("", "")
        return float(best), names

    def violations(self, q: dict[str, np.ndarray]) -> dict[tuple[str, str], float]:
        """Every non-allowed pair closer than δ (what BLOCKS the gate)."""
        out: dict[tuple[str, str], float] = {}
        for pair, d in self.twin.check_config_violations(self.full(q), data=self.data):
            key = tuple(sorted(pair))
            out[key] = min(d, out.get(key, np.inf))
        return out

    def within_band(self, q: dict[str, np.ndarray]) -> dict[tuple[str, str], float]:
        """Every non-allowed pair closer than δ + hysteresis (the planner's V0)."""
        self.band_data.qpos[:] = self.full(q)
        mujoco.mj_kinematics(self.band_model, self.band_data)
        mujoco.mj_collision(self.band_model, self.band_data)
        out: dict[tuple[str, str], float] = {}
        for pair, d in self.twin._violations(self.band_data):
            key = tuple(sorted(pair))
            out[key] = min(d, out.get(key, np.inf))
        return out

    def geometry_class(self, inside: dict[tuple[str, str], float]) -> str | None:
        """``sub_delta`` / ``band`` from the START GEOMETRY; None = nothing within the band
        (the caller's ``near`` / ``normal`` by its cross-arm range)."""
        if not inside:
            return None
        return "sub_delta" if min(inside.values()) < self.twin.inflation_m else "band"

    def random_posture(self, rng: np.random.Generator, arm: str) -> np.ndarray:
        """Half the draws around the default posture (where teleop leaves the arm), half
        uniform within the joint limits; the carriage uniform over its travel. Joints 1 / 3
        / 5 / 7 have ±2π software limits (two physical turns): the uniform half draws them
        within ONE turn around the default posture's branch, so every physical posture is
        covered without asking the return for a spurious full-turn rotation (j1 = 4.28 →
        -π is the same posture as 4.28 → π one turn away; the controller reads the arm on
        the default's branch)."""
        lo, hi = self.lo[arm], self.hi[arm]
        if rng.uniform() < 0.5:
            q = self.default[arm].copy()
            q[0] += rng.normal(scale=1.5)
            q[1:7] += rng.normal(scale=0.8, size=6)
        else:
            q = rng.uniform(lo, hi)
            for j in (0, 2, 4, 6):
                q[j] = self.default[arm][j] + rng.uniform(-math.pi, math.pi)
        q[7] = rng.uniform(lo[7], hi[7])
        return np.clip(q, lo, hi)

    def sample(self, idx: int, cls: str, seed: int, d_range: tuple[float, float]) -> Case:
        """A start of class ``cls`` whose min cross-arm distance is a random target in
        ``d_range`` and no pair penetrates: a clear start, a random far configuration, the
        first crossing of the target along the joint-space line between them (bisected).
        The class is VERIFIED on the start geometry (any pair, not just cross-arm ones): a
        ``band`` draw with some other pair inside δ, or a ``near`` / ``normal`` draw with
        anything inside the band, is redrawn."""
        rng = np.random.default_rng(seed)
        for _ in range(4000):
            q_free = {a: self.random_posture(rng, a) for a in ARMS}
            if self.violations(q_free) or self.cross_min(q_free)[0] < 0.08:
                continue
            q_far = {a: self.random_posture(rng, a) for a in ARMS}
            target = float(rng.uniform(*d_range))

            def at(t: float) -> dict[str, np.ndarray]:
                return {a: q_free[a] + t * (q_far[a] - q_free[a]) for a in ARMS}  # noqa: B023

            ts = np.linspace(0.0, 1.0, 241)
            t_prev, t_hit = 0.0, None
            for t in ts[1:]:
                if self.cross_min(at(float(t)))[0] < target:
                    t_hit = float(t)
                    break
                t_prev = float(t)
            if t_hit is None:
                continue
            a, b = t_prev, t_hit
            for _ in range(50):
                m = 0.5 * (a + b)
                if self.cross_min(at(m))[0] < target:
                    b = m
                else:
                    a = m
            q = at(b)
            d, pair = self.cross_min(q)
            viol = self.violations(q)
            if any(v <= 0.0 for v in viol.values()):
                continue  # penetration somewhere (self / table / obstacle): not a start
            if not (d_range[0] - 1e-4 <= d <= d_range[1] + 1e-4):
                continue
            inside = self.within_band(q)
            geo = self.geometry_class(inside)
            if geo != (cls if cls in ("sub_delta", "band") else None):
                continue  # the geometry belongs to another class: redraw
            tight = min(inside.values()) if inside else math.inf
            return Case(idx, cls, seed, q, d, pair, inside, tight)
        raise RuntimeError(f"no {cls} start found for seed {seed}")


# -- the headless cell: measured q + the real gate ----------------------------------------
@dataclass
class Hold:
    phase: str
    arm: str
    tick: int
    pairs: list
    dist_m: float


@dataclass
class ArmRun:
    arm: str
    ticks: int = 0
    holds: list[Hold] = field(default_factory=list)
    status: str = "done"  # done | held (plan_gate_hold_s abort) | runaway
    arrival_rad: float = 0.0
    arrival_rail_m: float | None = None
    rail_excursion_m: float = 0.0  # how far the carriage strayed from its start (joints phase)
    joint_excursion_rad: float = 0.0  # how far a joint strayed from its start (carriage phase)
    at_rail_limit: bool = False  # finished by the loop's reachable-limit rule (clamped goal)


class HeadlessCell:
    """The control loop's plan execution without the loop: ``PlanExecutor`` at the
    hardware caps, the loop's joint step cap, ``SafetyGate.filter`` on every tick, the
    servo following the gated command by the next tick (11-safety §7 / 04-runtime §7)."""

    def __init__(
        self, twin: DigitalTwin, cfg: SafetyConfig, q_meas: dict[str, np.ndarray], speed: float
    ) -> None:
        self.twin, self.cfg, self.speed = twin, cfg, speed
        self.q_meas = {a: np.array(q, dtype=float) for a, q in q_meas.items()}
        self.gate = SafetyGate(twin, cfg)
        self.jog = hw_jog(speed)
        self.dq_max = ControlConfig().dq_max_rad * speed  # scale_control_config
        self.events: list = []

    def states(self) -> dict:
        return {a: SimpleNamespace(q=q.copy()) for a, q in self.q_meas.items()}

    def sync(self) -> None:
        self.twin.sync(self.states())  # SafetySupervisor.sync, step 4

    def session_start(self, band_pairs: dict[tuple[str, str], float]) -> list:
        """Tick 1 commands the measured posture (the gate's rising edge); then the gate is
        given the block history a partial teleop retreat leaves: every pair inside the
        band is in ``_block_pairs`` and demanded to OPEN on each tick while blocked."""
        self.sync()
        dec = self.gate.filter(dict(self.q_meas), dict(self.q_meas), CommandSource.TELEOP)
        self.events.extend(dec.events)
        if band_pairs:
            self.gate._blocked = True
            self.gate._block_pairs |= set(band_pairs)
        return dec.events

    def cap_joint_step(self, q: np.ndarray, q_last: np.ndarray) -> np.ndarray:
        """``ControlLoop._cap_joint_step`` (uniform scaling; rail bound + travel clamp)."""
        q = np.array(q, dtype=float)
        dq = q[:7] - q_last[:7]
        ratio = float(np.max(np.abs(dq))) / self.dq_max
        if ratio > 1.0:
            q[:7] = q_last[:7] + dq / ratio
        q[7] = min(max(q[7], q_last[7] - self.dq_max), q_last[7] + self.dq_max)
        q[7] = min(max(q[7], 0.0), RAIL_TRAVEL_M)
        return q

    def execute_arm(self, phase: str, arm: str, waypoints: list) -> ArmRun:
        """One arm's plan through the gated tick path, the loop's rules verbatim:

        * ``_confirm_plan_arrivals`` - the tick the executor hands back the goal ends the
          plan only if the gated output IS the goal, or (gate clear, no progress) the only
          difference is the rail slot pinned at a travel end the goal lies beyond
          (``ControlLoop._clamped_rail_goal``); otherwise the goal re-enters the executor;
        * ``_plan_gate_watch`` - a tick is HELD when the gate is blocked and the output
          equals the previous command (the first tick re-commands waypoint 0, so a pinched
          start counts one held tick, reset by the first moving one); ``plan_gate_hold_s``
          of consecutive held ticks aborts the plan.

        ``holds`` records the ticks on which the gate CHANGED the command (hold-last-safe),
        the events the operator would see; the watch's count includes the clamp case too."""
        run = ArmRun(arm)
        goal = np.asarray(waypoints[-1], dtype=float)
        rail0 = float(self.q_meas[arm][7])
        joints0 = self.q_meas[arm][:7].copy()
        ex = PlanExecutor(self.jog)
        ex.load(arm, waypoints)
        consecutive = 0
        while True:
            if not ex.active(arm):
                ex.load(arm, [goal.tolist()])  # the goal re-entered the executor
            q_next = ex.step(arm, self.q_meas[arm])
            arriving = not ex.active(arm)  # the executor handed back the goal this tick
            run.ticks += 1
            q_prev = self.q_meas[arm]  # the loop's _last_cmd (the servo follows by one tick)
            q_cmd = dict(self.q_meas)  # the other arm: its last command = measured
            q_cmd[arm] = self.cap_joint_step(q_next, q_prev)
            self.sync()
            dec = self.gate.filter(q_cmd, dict(self.q_meas), CommandSource.PLANNER)
            self.events.extend(dec.events)
            q_out = dec.q_out[arm]
            if np.array_equal(q_out, q_prev) and not np.array_equal(q_cmd[arm], q_prev):
                pairs = list(dec.report.pairs) or sorted(self.gate._block_pairs)
                run.holds.append(Hold(phase, arm, run.ticks, pairs, dec.report.min_clearance_m))
            if dec.blocked and np.array_equal(q_out, q_prev):
                consecutive += 1  # _plan_gate_watch's rule
                if consecutive >= PLAN_GATE_HOLD_TICKS:
                    run.status = "held"
                    break
            else:
                consecutive = 0
            self.q_meas = {a: np.array(q, dtype=float) for a, q in dec.q_out.items()}
            run.rail_excursion_m = max(
                run.rail_excursion_m, abs(float(self.q_meas[arm][7]) - rail0)
            )
            run.joint_excursion_rad = max(
                run.joint_excursion_rad, float(np.max(np.abs(self.q_meas[arm][:7] - joints0)))
            )
            if arriving:
                if np.allclose(q_out, goal, atol=1e-9):
                    break  # the gated output IS the goal
                progressed = not np.allclose(q_out, q_prev, atol=1e-9)
                if (
                    not dec.blocked
                    and not progressed
                    and ControlLoop._clamped_rail_goal(q_out, goal)
                ):
                    run.at_rail_limit = True
                    break  # the loop finishes a goal the rail clamp cannot reach
            if run.ticks > TICK_CAP:
                run.status = "runaway"
                break
        joints, rail = SessionManager._arrival_error(self.q_meas[arm], goal)
        run.arrival_rad, run.arrival_rail_m = joints, rail
        return run


# -- the manager's two-phase return, headless --------------------------------------------
@dataclass
class PhaseRun:
    label: str  # joints | carriage
    status: str  # skipped | failed | done | held | stalled | runaway
    failure: str | None = None
    failing_pair: tuple[str, str] | None = None
    arm_order: list[str] = field(default_factory=list)
    escape_len: dict[str, int] = field(default_factory=dict)
    arms: list[ArmRun] = field(default_factory=list)
    plan_s: float = 0.0

    @property
    def ticks(self) -> int:
        return sum(a.ticks for a in self.arms)

    @property
    def holds(self) -> list[Hold]:
        return [h for a in self.arms for h in a.holds]


@dataclass
class RunResult:
    case: Case
    goal: str
    phases: list[PhaseRun] = field(default_factory=list)
    start_events: list = field(default_factory=list)

    @property
    def status(self) -> str:
        for p in self.phases:
            if p.status not in ("done", "skipped"):
                return p.status
        return "done"

    @property
    def failure(self) -> PhaseRun | None:
        return next((p for p in self.phases if p.status == "failed"), None)

    @property
    def holds(self) -> list[Hold]:
        return [h for p in self.phases for h in p.holds]

    @property
    def ticks(self) -> int:
        return sum(p.ticks for p in self.phases)

    @property
    def escape_len(self) -> int:
        return sum(sum(p.escape_len.values()) for p in self.phases)

    @property
    def worst_arrival(self) -> tuple[float, float]:
        rad = max((a.arrival_rad for p in self.phases for a in p.arms), default=0.0)
        rail = max(
            (a.arrival_rail_m for p in self.phases for a in p.arms if a.arrival_rail_m is not None),
            default=0.0,
        )
        return rad, rail

    @property
    def honest(self) -> bool:
        """An honest planner failure: a known kind naming a pair, every other phase done /
        skipped, and a ``timeout`` only when the budget was really spent."""
        f = self.failure
        if f is None or f.failure not in HONEST_FAILURES or f.failing_pair is None:
            return False
        if f.failure == "timeout" and f.plan_s < TIMEOUT_HONEST_FRACTION * PLAN_TIMEOUT_S:
            return False
        return all(p.status in ("done", "skipped") for p in self.phases if p is not f)

    @property
    def failure_kind(self) -> str | None:
        return self.failure.failure if self.honest else None

    @property
    def clean(self) -> bool:
        return (
            self.status == "done"
            and not self.holds
            and all(
                a.arrival_rad <= PLAN_ARRIVAL_TOL_RAD
                and (a.arrival_rail_m is None or a.arrival_rail_m <= PLAN_ARRIVAL_TOL_RAIL_M)
                for p in self.phases
                for a in p.arms
            )
        )


def escape_length(twin: DigitalTwin, arm: str, wps: list, ctx: dict[str, np.ndarray]) -> int:
    """Leading segments of ``wps`` during which some pair of ``arm`` that started inside the
    shell or the gate's band is not yet re-armed (> δ + REARM_MARGIN_M) - the planner's
    escape phase as the executed path shows it (0 for an unpinched start)."""
    start = {**ctx, arm: np.asarray(wps[0], dtype=float)}
    active = set(_band_pairs_of(twin, start, arm))
    if not active:
        return 0
    for i, w in enumerate(wps[1:], start=1):
        q = {**ctx, arm: np.asarray(w, dtype=float)}
        for p in list(active):
            if twin.pair_distance(p, q) > twin.inflation_m + REARM_MARGIN_M:
                active.discard(p)
        if not active:
            return i
    return len(wps) - 1


def _full(twin: DigitalTwin, q: dict[str, np.ndarray]) -> np.ndarray:
    ctx = np.array(twin._q_meas_full)
    for a, v in q.items():
        ctx[twin.addr[a].qpos_adr] = v
    return ctx


_BAND_CACHE: dict[int, tuple[mujoco.MjModel, mujoco.MjData]] = {}


def _band_pairs_of(twin: DigitalTwin, q: dict[str, np.ndarray], arm: str):
    """This arm's non-allowed pairs closer than δ + hysteresis at ``q`` (sorted labels)."""
    key = id(twin)
    if key not in _BAND_CACHE:
        model = copy.copy(twin.model)
        apply_inflation(model, twin.inflation_m + twin.hysteresis_m)
        _BAND_CACHE[key] = (model, mujoco.MjData(model))
    model, data = _BAND_CACHE[key]
    data.qpos[:] = _full(twin, q)
    mujoco.mj_kinematics(model, data)
    mujoco.mj_collision(model, data)
    for pair, _d in twin._violations(data):
        key_pair = tuple(sorted(pair))
        if arm in twin._arms_of_pair(key_pair):
            yield key_pair


def run_return(
    twin: DigitalTwin, cfg: SafetyConfig, case: Case, goal: str, profile: StateProfile, speed: float
):
    """``SessionManager._return_to_initial_motion`` headless: phases, per-phase plan from
    the MEASURED posture, one arm at a time through the gate, measured arrival."""
    cell = HeadlessCell(twin, cfg, case.q, speed)
    result = RunResult(case, goal)
    result.start_events = cell.session_start(case.inside)
    fake_session = SimpleNamespace(
        workcell=SimpleNamespace(states=cell.states), loop=SimpleNamespace(gripper_arms=["grip"])
    )
    _states, q_start, q_joint, q_full, _grippers = MANAGER_RULES._return_goals(
        fake_session, profile, ARMS
    )
    joints_move = any(not np.allclose(q_start[a][:7], q_joint[a][:7], atol=1e-3) for a in q_joint)
    rail_move = any(not np.allclose(q_joint[a], q_full[a], atol=1e-3) for a in q_full)
    phases = []
    if joints_move:
        phases.append(("joints", q_joint))
    if rail_move:
        phases.append(("carriage", q_full))
    for label, goal_q in phases:
        phase = PhaseRun(label, "done")
        result.phases.append(phase)
        # _return_phase: plan from the arms' MEASURED posture, skip when already there
        states = cell.states()
        q_start = {a: [float(x) for x in states[a].q] for a in goal_q}
        if all(np.allclose(q_start[a], goal_q[a], atol=1e-3) for a in goal_q):
            phase.status = "skipped"
            continue
        twin.sync(states)  # _plan_return keeps the plan twin fresh
        t0 = time.monotonic()
        res = twin.plan(PlanRequest(q_start=q_start, q_goal=goal_q, speed_scale=speed))
        phase.plan_s = time.monotonic() - t0
        if not res.ok:
            phase.status, phase.failure, phase.failing_pair = (
                "failed",
                res.failure,
                res.failing_pair,
            )
            break
        waypoints = SessionManager._ordered_waypoints(res)
        phase.arm_order = list(waypoints)
        order = SessionManager._moving_arms(waypoints)
        ctx = {a: np.asarray(q_start[a], dtype=float) for a in ARMS}
        for arm in order:
            phase.escape_len[arm] = escape_length(twin, arm, waypoints[arm], ctx)
            run = cell.execute_arm(label, arm, waypoints[arm])
            phase.arms.append(run)
            ctx[arm] = cell.q_meas[arm]  # arm k+1 was validated against arm k AT its goal
            if run.status != "done":
                phase.status = run.status
                break
            if not SessionManager._arrived(cell.q_meas[arm], waypoints[arm][-1]):
                phase.status = "stalled"
                break
        if phase.status != "done":
            break
    return result


# -- the table -----------------------------------------------------------------------------
def _pair_text(pair) -> str:
    return " / ".join(pair) if pair else "-"


def by_class(results: list[RunResult]) -> dict[str, list[RunResult]]:
    out: dict[str, list[RunResult]] = {c: [] for c in N_CASES}
    for r in results:
        out[r.case.cls].append(r)
    return out


def format_table(results: list[RunResult], elapsed_s: float, speed: float, mic: bool) -> str:
    lines = [
        "",
        f"return fuzz on mavis_v2 ({'with' if mic else 'WITHOUT'} the Perception Arm "
        f"microphone) - speed {speed:.2f}, seed base {BASE_SEED}, {len(results)} runs "
        f"in {elapsed_s:.0f} s",
        f"{'case':>4} {'class':9} {'seed':>9} {'cross mm':>8} {'tight mm':>8} {'goal':8} "
        f"{'status':8} {'failure':17} {'escape':>6} {'ticks':>6} {'holds':>5} "
        f"{'arrive mrad/mm':>14} {'plan s':>6}  pair",
    ]
    for r in results:
        f = r.failure
        rad, rail = r.worst_arrival
        pair = r.case.cross_pair
        if f:
            pair = f.failing_pair
        elif r.holds and r.holds[0].pairs:
            pair = r.holds[0].pairs[0]
        tight = f"{r.case.tight_d * 1e3:>8.2f}" if math.isfinite(r.case.tight_d) else f"{'-':>8}"
        lines.append(
            f"{r.case.idx:>4} {r.case.cls:9} {r.case.seed:>9} {r.case.cross_d * 1e3:>8.2f} "
            f"{tight} {r.goal:8} {r.status:8} {(f'{f.failure}@{f.label}' if f else '-'):17} "
            f"{r.escape_len:>6} "
            f"{r.ticks:>6} {len(r.holds):>5} {rad * 1e3:>7.3f}/{rail * 1e3:<6.2f} "
            f"{sum(p.plan_s for p in r.phases):>6.2f}  {_pair_text(pair)}"
        )
    held = [r for r in results if r.holds]
    other = [r for r in results if not r.clean and not r.honest and not r.holds]
    lines += [
        "",
        f"cases {len(results) // len(GOAL_NAMES)} x goals {len(GOAL_NAMES)} = {len(results)} runs",
        f"  {'class':9} {'cases':>5} {'runs':>4} {'clean':>5} {'escaped':>7} "
        f"{'goal_in_collision':>17} {'no_escape':>9} {'timeout':>7} {'holds':>5} {'other':>5}",
    ]
    for cls, rs in by_class(results).items():
        kinds = [r.failure_kind for r in rs]
        lines.append(
            f"  {cls:9} {len(rs) // len(GOAL_NAMES):>5} {len(rs):>4} "
            f"{sum(r.clean for r in rs):>5} {sum(r.clean and r.escape_len > 0 for r in rs):>7} "
            f"{kinds.count('goal_in_collision'):>17} {kinds.count('no_escape'):>9} "
            f"{kinds.count('timeout'):>7} {sum(bool(r.holds) for r in rs):>5} "
            f"{sum(1 for r in rs if not r.clean and not r.honest and not r.holds):>5}"
        )
    clean = sum(r.clean for r in results)
    honest = [r for r in results if r.honest]
    kinds = [r.failure_kind for r in honest]
    lines += [
        f"  clean (plan ok, no gate hold, arrived): {clean}",
        "  honest planner failures: "
        + ", ".join(f"{k} {kinds.count(k)}" for k in HONEST_FAILURES)
        + f" (total {len(honest)})",
        f"  gate holds: {len(held)} runs, {sum(len(r.holds) for r in held)} held ticks",
        f"  other (not clean, not honest, no hold): {len(other)}",
    ]
    rail_stray = max(
        (
            a.rail_excursion_m
            for r in results
            for p in r.phases
            if p.label == "joints"
            for a in p.arms
        ),
        default=0.0,
    )
    joint_stray = max(
        (
            a.joint_excursion_rad
            for r in results
            for p in r.phases
            if p.label == "carriage"
            for a in p.arms
        ),
        default=0.0,
    )
    plan_s = sum(p.plan_s for r in results for p in r.phases)
    limit_runs = sum(1 for r in results for p in r.phases for a in p.arms if a.at_rail_limit)
    lines.append(f"  carriage excursion during the joints phase (max): {rail_stray * 1e3:.1f} mm")
    lines.append(f"  joint excursion during the carriage phase (max): {joint_stray * 1e3:.1f} mrad")
    lines.append(
        f"  arms finished by the loop's reachable-limit rule (clamped rail goal): {limit_runs}"
    )
    lines.append(f"  planning time: {plan_s:.0f} s of {elapsed_s:.0f} s")
    if held or other:
        lines.append("  FINDINGS (seed, goal, first hold / status):")
        for r in held + other:
            first = r.holds[0] if r.holds else None
            where = (
                f"{first.phase} {first.arm} tick {first.tick} {_pair_text(first.pairs[0])} "
                f"at {first.dist_m * 1e3:.2f} mm"
                if first
                else f"status {r.status}"
            )
            lines.append(f"    seed {r.case.seed} goal {r.goal}: {where}")
    for r in results:
        f = r.failure
        if f is not None and not r.honest:
            lines.append(
                f"  DISHONEST failure: seed {r.case.seed} goal {r.goal} phase {f.label}: "
                f"{f.failure} pair {f.failing_pair} after {f.plan_s:.2f} s"
            )
    return "\n".join(lines)


# -- the test ------------------------------------------------------------------------------
def sample_cases(sampler: Sampler) -> list[Case]:
    ranges = class_ranges(sampler.cfg)
    cases = []
    idx = 0
    for cls, n in N_CASES.items():
        for i in range(n):
            cases.append(sampler.sample(idx, cls, BASE_SEED + SEED_OFFSET[cls] + i, ranges[cls]))
            idx += 1
    return cases


def test_the_sub_delta_class_is_the_majority_of_the_pinch_classes():
    """The operator's class dominates the sweep by construction (module docstring)."""
    assert N_CASES["sub_delta"] > N_CASES["band"] + N_CASES["near"]
    assert PLAN_TIMEOUT_S == 5.0  # the per-arm budget an honest ``timeout`` must have spent


def test_sampler_produces_the_requested_classes(twin):
    """The generator's contract, per class: the min cross-arm distance is in the class
    range, the class is the START GEOMETRY (some pair < δ / only the band / nothing within
    the band), nothing penetrates, the pinched pair is moved by BOTH arms (never a static
    rail base), and the draw is deterministic. With the microphone on, the mic body is
    among the cross-arm pairs the pinch target can pick."""
    cfg = cell_safety_config()
    sampler = Sampler(twin, cfg)
    ranges = class_ranges(cfg)
    label = twin.allowed.label_of_geom
    cross_labels = {label(g) for pair in sampler.cross for g in pair}
    assert not any(lbl.endswith("_rail_base") for lbl in cross_labels)
    assert (MIC_LABEL in cross_labels) == twin_has_mic(twin)
    for cls, rng_m in ranges.items():
        a = sampler.sample(0, cls, BASE_SEED + SEED_OFFSET[cls], rng_m)
        b = sampler.sample(0, cls, BASE_SEED + SEED_OFFSET[cls], rng_m)
        assert all(np.array_equal(a.q[arm], b.q[arm]) for arm in ARMS)
        assert a.cls == cls
        assert rng_m[0] - 1e-4 <= a.cross_d <= rng_m[1] + 1e-4, (cls, a.cross_d)
        assert set(twin._arms_of_pair(a.cross_pair)) == {"grip", "view"}, a.cross_pair
        assert all(d > 0.0 for d in sampler.violations(a.q).values())
        if cls == "sub_delta":
            assert a.tight_d < twin.inflation_m and sampler.violations(a.q)
        elif cls == "band":
            assert twin.inflation_m <= a.tight_d < sampler.band_m
            assert not sampler.violations(a.q)
        else:
            assert not a.inside and not math.isfinite(a.tight_d)


def test_return_fuzz_mavis_v2(twin, capsys):
    cfg = cell_safety_config()
    mic = twin_has_mic(twin)
    t0 = time.monotonic()
    sampler = Sampler(twin, cfg)
    cases = sample_cases(sampler)
    results: list[RunResult] = []
    for case in cases:
        for goal in GOAL_NAMES:
            profile = goal_profile(goal, case, sampler)
            results.append(run_return(twin, cfg, case, goal, profile, SPEED))
    table = format_table(results, time.monotonic() - t0, SPEED, mic)
    with capsys.disabled():
        print(table)

    def describe(r: RunResult) -> str:
        rad, rail = r.worst_arrival
        hs = r.holds[:3]
        f = r.failure
        return (
            f"seed {r.case.seed} ({r.case.cls}, {r.case.cross_d * 1e3:.2f} mm "
            f"{_pair_text(r.case.cross_pair)}) goal {r.goal}: status {r.status}, "
            f"failure {(f.failure, f.failing_pair, round(f.plan_s, 2)) if f else None}, "
            f"holds {[(h.phase, h.arm, h.tick, h.pairs[:1], round(h.dist_m * 1e3, 2)) for h in hs]}"
            f", arrival {rad * 1e3:.3f} mrad / {rail * 1e3:.2f} mm"
        )

    # 1. every run: clean, or an honest planner failure
    problems = [describe(r) for r in results if not (r.clean or r.honest)]
    assert not problems, "\n".join(problems)
    # 2. genuine timeouts are rare
    timeouts = [r for r in results if r.failure_kind == "timeout"]
    assert len(timeouts) <= MAX_HONEST_TIMEOUTS, "\n".join(describe(r) for r in timeouts)
    # 3. the operator's class: mostly clean, every clean run escaped, few boxed-in refusals
    sub = by_class(results)["sub_delta"]
    assert sub
    sub_clean = [r for r in sub if r.clean]
    assert len(sub_clean) >= MIN_CLEAN_FRACTION * len(sub), (
        f"{len(sub_clean)} of {len(sub)} sub-delta runs clean;\n"
        + "\n".join(describe(r) for r in sub if not r.clean)
    )
    no_escape = [r for r in sub if r.failure_kind == "no_escape"]
    assert len(no_escape) <= MAX_NO_ESCAPE, "\n".join(describe(r) for r in no_escape)
    unescaped = [r for r in sub_clean if r.escape_len == 0]
    assert not unescaped, "\n".join(describe(r) for r in unescaped)
    # 4. the sweep exercised a carriage phase
    assert any(p.label == "carriage" and p.status == "done" for r in results for p in r.phases)
