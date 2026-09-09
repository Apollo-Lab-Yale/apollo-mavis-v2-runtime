"""A twin plan from INSIDE the inflation shell passes the real SafetyGate tick by tick
(11-safety §7.1 / §9; 03-sim §10 item 3; 2026-09-09 01:14 incident).

``var/logs/runtime.log`` lines ~24972-25051 that night: the Manipulation Arm's
``grip_right_finger`` sat 2.1 mm from the Perception Arm's ``view_link3`` (health line
``gate=blocked min=0.0021m``). ``goto_profile`` planned ``grip`` as ONE straight segment
(2 waypoints, max |dq| 1.65 rad) because the planner WHITELISTED pairs already violating
at the start, so it never noticed the segment first CLOSED the pair to 1.1 mm; the gate's
T8 escape rule (a blocked arm's command passes only if EVERY blocked pair opens and no new
one appears) held the first step and ``plan_gate_hold_s`` aborted the plan after 3 s.

This file replays plans the way the control loop executes them - ``PlanExecutor`` at the
hardware caps, one arm at a time in ``arm_order``, every tick through ``SafetyGate.filter``
with the servo assumed to follow (q_meas = the previous tick's q_out) - and asserts the
gate never holds and never raises a new ``blocked`` event once the escape begins. The
old planner's plan shape fails this (``test_the_incident_shape_is_held_and_now_refused``);
the escape phase + gate-resolution verification pass it at 100 / 50 / 10 % speed (the
10 % case walks THROUGH the gate's 2 mm hysteresis band instead of jumping it, which
needs the gate's step-6 "new violation" window to match step 3 - fixed the same day).

The same-day review added (all replayed here): the escape is judged per EXECUTOR TICK at
the speed the plan runs at (``PlanRequest.speed_scale``), not per planner sample, and the
executor walks a segment in equal ticks (a tiny remainder tick opened 4-8 um < the gate's
10 um and was held); the gate's hysteresis band counts as pinched (a pair the gate holds in
its band must open too, else a 10 % plan is held from its first tick); only pairs the
planned arm's joints can move count (the other arm's pinch is a constant, not a
``no_escape``); an unpinched plan replays clean too.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from apollo_mavis_v2_core import CommandSource, PlanRequest, SafetyConfig

from apollo_mavis_v2_runtime.config import JogConfig
from apollo_mavis_v2_runtime.control.joint_panel import PlanExecutor
from apollo_mavis_v2_runtime.devices.rail_homing import JOB_SPEED_SCALE
from apollo_mavis_v2_runtime.safety.gate import ESCAPE_EPS_M, SafetyGate

pytest.importorskip("apollo_mavis_v2_sim")
from apollo_mavis_v2_sim import REGISTRY, DigitalTwin  # noqa: E402
from apollo_mavis_v2_sim.planner import (  # noqa: E402
    ESCAPE_RATE_MARGIN,
    FINE_STEP_M,
    HW_CART_STEP_M,
    HW_RAIL_M_PER_TICK,
    HW_SLEW_RAD_PER_TICK,
    MIN_SPEED_SCALE,
    PLANNER_ESCAPE_EPS_M,
    REARM_MARGIN_M,
    ResetPlanner,
    _tick_count,
)

CELL_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "mavis_v2.yaml"
# Hardware executor caps at speed scale 1.0 (the bring-up log line of the incident session:
# "plan executor capped by the servo stream - slew 0.00600 rad/tick, cart 0.00400 m/tick,
# rail 0.00050 m/tick"); the lever arms are the driver's ServoLimits default.
HW_SLEW_RAD = 0.006
HW_RAIL_M = 0.0005
HW_CART_STEP_M_ = 0.004
HW_LEVER_ARM_M = (1.20, 1.20, 1.00, 0.75, 0.44, 0.30, 0.10)

DEG = np.pi / 180.0
VIEW_INIT = [x * DEG for x in (0.0, 0.8, 0.0, 28.9, 0.0, 28.2, 0.0)] + [0.30]  # seeded initial
GRIP_INIT = [x * DEG for x in (-180.0, -12.0, -20.0, 30.0, -5.0, 35.0, -8.9)] + [0.30]
GRIP_FREE = [0.7, -1.1, 0.0, 0.3, 0.1, -0.4, 0.0, 0.30]  # reaching toward the Perception Arm
VIEW_GOAL = [0.4, 0.8 * DEG, 0.0, 28.9 * DEG + 0.3, 0.0, 28.2 * DEG, 0.0, 0.35]
PINCH_PAIR = ("grip_right_finger", "view_link3")
PINCH_DEPTH_M = 0.002  # the incident's 2.1 mm
# Found by random search on the twin (2026-09-09 review). ``VIEW_SELF_PINCH``: the Perception
# Arm pinched against ITSELF (``view_link1`` / ``view_link5`` at 7.96 mm) - a constant for the
# Manipulation Arm. ``GRIP_BAND_START``: ``grip_link4`` 6.00 mm from ``view_link2`` (inside the
# shell) AND 9.73 mm from ``view_link3`` (inside the gate's [δ, δ + 2 mm) band) with the
# Perception Arm at VIEW_INIT - the state a partial teleop retreat from a two-pair block leaves.
VIEW_SELF_PINCH = [-0.405, -0.8213, -0.1354, -0.0209, 0.6008, 0.5786, 0.4693, 0.3]
GRIP_BAND_START = [-2.9805, -0.91488, -1.46179, 1.67498, 0.58877, 0.1508, -0.36369, 0.15581]
BAND_PAIR_P = ("grip_link4", "view_link2")
BAND_PAIR_B = ("grip_link4", "view_link3")


def cell_safety_config() -> SafetyConfig:
    """The real cell's gate values: ``workcells.hardware.safety`` of configs/mavis_v2.yaml
    on top of core's defaults (the file only pins ``enabled``)."""
    doc = yaml.safe_load(CELL_CONFIG.read_text())
    return SafetyConfig(**doc["workcells"]["hardware"].get("safety", {}))


def hw_jog(speed_scale: float) -> JogConfig:
    return JogConfig(
        slew_rad_per_tick=HW_SLEW_RAD * speed_scale,
        rail_m_per_tick=HW_RAIL_M * speed_scale,
        plan_cart_step_m=HW_CART_STEP_M_ * speed_scale,
        plan_lever_arm_m=HW_LEVER_ARM_M,
    )


@pytest.fixture(scope="module")
def twin() -> DigitalTwin:
    cfg = cell_safety_config()
    return DigitalTwin(
        REGISTRY.build("mavis_v2"), inflation_m=cfg.geom_inflation_m, hysteresis_m=cfg.hysteresis_m
    )


def violating_pairs(twin, q_by_arm) -> dict[tuple[str, str], float]:
    ctx = np.array(twin._q_meas_full)
    for arm_id, q in q_by_arm.items():
        ctx[twin.addr[arm_id].qpos_adr] = q
    out: dict[tuple[str, str], float] = {}
    for pair, d in twin.check_config_violations(ctx):
        key = tuple(sorted(pair))
        out[key] = min(d, out.get(key, np.inf))
    return out


def pinch_on_line(twin, q_free, q_far, depth_m: float) -> np.ndarray:
    """grip config on the joint-space line ``q_free -> q_far`` where PINCH_PAIR reads
    ``depth_m`` (bisection), the Perception Arm at VIEW_INIT."""
    q_free, q_far = np.asarray(q_free, dtype=float), np.asarray(q_far, dtype=float)

    def dist(t: float) -> float:
        return twin.pair_distance(
            PINCH_PAIR, {"grip": q_free + t * (q_far - q_free), "view": np.asarray(VIEW_INIT)}
        )

    assert dist(0.0) > twin.inflation_m
    inside = next(t for t in np.linspace(0.0, 1.0, 401) if dist(t) < depth_m)
    lo, hi = 0.0, float(inside)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if dist(mid) > depth_m:
            lo = mid
        else:
            hi = mid
    return q_free + hi * (q_far - q_free)


def random_pinch(twin, rng: np.random.Generator, lo, hi, depth_m: float = PINCH_DEPTH_M):
    """A random Manipulation Arm posture whose closest pair reads ``depth_m`` (every pair
    > 0, at least one against the Perception Arm at VIEW_INIT): a random clear posture near
    GRIP_INIT, a random far posture, the first crossing of ``depth_m`` on the line between
    them by bisection."""
    grip_init = np.asarray(GRIP_INIT)
    for _ in range(2000):
        q_free = grip_init.copy()
        q_free[:7] = np.clip(q_free[:7] + rng.normal(scale=0.5, size=7), lo[:7], hi[:7])
        q_free[7] = rng.uniform(0.15, 0.45)
        if violating_pairs(twin, {"grip": q_free, "view": VIEW_INIT}):
            continue
        q_far = q_free.copy()
        q_far[:7] = np.clip(q_far[:7] + rng.normal(scale=0.8, size=7), lo[:7], hi[:7])
        t_prev, t_hit = 0.0, None
        for t in np.linspace(0.0, 1.0, 801)[1:]:
            v = violating_pairs(twin, {"grip": q_free + t * (q_far - q_free), "view": VIEW_INIT})
            if v and min(v.values()) <= depth_m:
                t_hit = t
                break
            t_prev = t
        if t_hit is None:
            continue
        a, b = t_prev, t_hit
        for _ in range(60):
            m = 0.5 * (a + b)
            v = violating_pairs(twin, {"grip": q_free + m * (q_far - q_free), "view": VIEW_INIT})
            if v and min(v.values()) <= depth_m:
                b = m
            else:
                a = m
        q = q_free + b * (q_far - q_free)
        v = violating_pairs(twin, {"grip": q, "view": VIEW_INIT})
        if (
            v
            and all(d > 0.0 for d in v.values())
            and abs(min(v.values()) - depth_m) < 2e-4
            and any("view" in p[0] or "view" in p[1] for p in v)
        ):
            return q, v
    raise RuntimeError("no pinch found")


def start_and_goals(twin):
    q_start = pinch_on_line(twin, GRIP_FREE, GRIP_INIT, PINCH_DEPTH_M)
    free, init = np.asarray(GRIP_FREE), np.asarray(GRIP_INIT)
    q_goal = free + 0.25 * (init - free)  # the far side of view_link3, 13.6 cm clear
    return q_start, q_goal


class Replay:
    """The control loop's execution of a plan, gate included (04-runtime §7/§10.5)."""

    def __init__(self, twin, gate: SafetyGate, q_meas: dict[str, np.ndarray]) -> None:
        self.twin, self.gate = twin, gate
        self.q_meas = {a: np.array(q, dtype=float) for a, q in q_meas.items()}
        self.holds: list[tuple[str, int, list, float]] = []
        self.events: list = []
        self.ticks: dict[str, int] = {}
        self.pair_dists: list[float] = []  # measured PINCH_PAIR distance per tick

    def sync(self) -> None:  # what SafetySupervisor.sync does with the driver states
        self.twin.sync({a: SimpleNamespace(q=q) for a, q in self.q_meas.items()})

    def session_start(self) -> list:
        """The first tick commands the measured posture: the gate's rising-edge event."""
        self.sync()
        dec = self.gate.filter(dict(self.q_meas), dict(self.q_meas), CommandSource.TELEOP)
        return dec.events

    def execute(self, res, jog: JogConfig, tick_cap: int = 40_000, stop_after: int | None = None):
        """Returns False when ``stop_after`` ticks passed without the plan finishing."""
        for arm in res.arm_order:  # one arm at a time, in the planner's order
            ex = PlanExecutor(jog)
            ex.load(arm, res.waypoints[arm])
            n = 0
            while ex.active(arm):
                if stop_after is not None and n >= stop_after:
                    return False
                q_next = ex.step(arm, self.q_meas[arm])
                n += 1
                q_cmd = dict(self.q_meas)
                q_cmd[arm] = q_next
                self.sync()
                dec = self.gate.filter(q_cmd, dict(self.q_meas), CommandSource.PLANNER)
                self.events.extend(dec.events)
                if not np.array_equal(dec.q_out[arm], q_next):
                    self.holds.append((arm, n, list(dec.report.pairs), dec.report.min_clearance_m))
                self.q_meas = {a: np.array(q) for a, q in dec.q_out.items()}
                self.pair_dists.append(
                    self.twin.pair_distance(PINCH_PAIR, {a: q for a, q in self.q_meas.items()})
                )
                assert n <= tick_cap, f"{arm}: executor never finished (held: {self.holds[:3]})"
            self.ticks[arm] = n
        return True


def assert_clean_replay(replay: Replay, res, *, cleared: int | None = None) -> None:
    """No hold on any tick, no new ``blocked`` / ``penetration`` event, every arm arrived."""
    assert replay.holds == [], replay.holds[:5]
    kinds = [e.kind for e in replay.events]
    assert "blocked" not in kinds and "penetration" not in kinds, replay.events
    if cleared is not None:
        assert kinds.count("cleared") == cleared, kinds
    for arm in res.arm_order:
        assert np.allclose(replay.q_meas[arm], res.waypoints[arm][-1], atol=1e-9)


def test_planner_constants_mirror_the_cell_gate():
    """The planner duplicates the gate's and the executor's numbers (sim must not import the
    runtime or the hardware package): the strict-opening margin, the re-arm margin above the
    gate's hysteresis band, the executor's caps the escape is judged tick by tick against
    (the driver's ServoLimits at 100 Hz), and the slowest speed the runtime offers."""
    cfg = cell_safety_config()
    assert PLANNER_ESCAPE_EPS_M == ESCAPE_EPS_M
    assert ESCAPE_RATE_MARGIN > 1.0
    assert REARM_MARGIN_M > cfg.hysteresis_m + cfg.min_clearance_m
    assert FINE_STEP_M == HW_CART_STEP_M_ == HW_CART_STEP_M
    assert (HW_SLEW_RAD_PER_TICK, HW_RAIL_M_PER_TICK) == (HW_SLEW_RAD, HW_RAIL_M)
    # the default judgement speed = the core default = the rail-homing job's scale
    assert PlanRequest(q_start={}, q_goal={}).speed_scale == MIN_SPEED_SCALE == JOB_SPEED_SCALE
    hw = pytest.importorskip("apollo_mavis_v2_hardware.config")
    limits = hw.ServoLimits()
    assert tuple(limits.lever_arm_m) == HW_LEVER_ARM_M
    assert limits.max_cart_step_m == HW_CART_STEP_M
    assert min(limits.max_joint_vel) / limits.rate_hz == pytest.approx(HW_SLEW_RAD_PER_TICK)
    rail_mm_s = hw.XArmDriverConfig.model_fields["rail_speed_mm_s"].default
    assert rail_mm_s / 1000.0 / limits.rate_hz == pytest.approx(HW_RAIL_M_PER_TICK)


@pytest.mark.parametrize("speed_scale", [1.0, 0.5, 0.1])
def test_executor_walks_a_segment_in_equal_ticks_the_planner_can_count(speed_scale):
    """``PlanExecutor.step`` splits a segment into ``ceil(ratio)`` EQUAL ticks (no tiny
    remainder tick onto the waypoint), and the planner's ``_tick_count`` predicts exactly
    that number at the same speed - the escape's per-tick opening is judged on the very
    ticks the executor will emit."""
    jog = hw_jog(speed_scale)
    rng = np.random.default_rng(0)
    for _ in range(40):
        q0 = np.zeros(8)
        dq = np.concatenate([rng.normal(scale=0.02, size=7), [rng.normal(scale=0.002)]])
        ex = PlanExecutor(jog)
        ex.load("grip", [q0.tolist(), (q0 + dq).tolist()])
        q = q0.copy()
        steps: list[np.ndarray] = []
        while ex.active("grip"):
            q_next = ex.step("grip", q)
            if steps or np.any(q_next != q):  # tick 1 re-commands waypoint 0
                steps.append(q_next - q)
            q = q_next
        assert np.allclose(q, q0 + dq, atol=1e-12)
        assert len(steps) == _tick_count(dq, speed_scale), (len(steps), dq)
        first = steps[0]
        for s in steps[1:]:  # equal ticks along the straight segment
            assert np.allclose(s, first, atol=1e-9), (s, first)
        # every tick under the caps
        lever = np.asarray(HW_LEVER_ARM_M)
        assert np.max(np.abs(first[:7])) <= jog.slew_rad_per_tick * (1 + 1e-9)
        assert abs(first[7]) <= jog.rail_m_per_tick * (1 + 1e-9)
        assert float(np.sum(np.abs(first[:7]) * lever)) <= jog.plan_cart_step_m * (1 + 1e-9)
    # the pathological remainders of the old rule: 1.3 ticks -> two of 0.65, 4.2 -> five of 0.84
    # (joint 7 alone: its 0.10 m lever keeps the Cartesian cap out of the way)
    for ticks in (1.3, 4.2, 5.0):
        ex = PlanExecutor(jog)
        goal = np.zeros(8)
        goal[6] = ticks * jog.slew_rad_per_tick
        ex.load("grip", [goal.tolist()])
        q = np.zeros(8)
        sizes = []
        while ex.active("grip"):
            q_next = ex.step("grip", q)
            sizes.append(float(q_next[6] - q[6]))
            q = q_next
        assert len(sizes) == int(np.ceil(ticks - 1e-9)), (ticks, sizes)
        assert all(abs(s - sizes[0]) < 1e-12 for s in sizes)
        assert sizes[0] == pytest.approx(goal[6] / len(sizes))


@pytest.mark.parametrize("speed_scale", [1.0, 0.5, 0.1])
def test_pinched_start_plan_passes_the_gate_for_both_arms(twin, speed_scale):
    """The property that failed on the real cell: replayed at the hardware caps, the gate
    never holds a tick of either arm's plan and raises no ``blocked`` event after the
    session-start one; the pinched pair is ``cleared`` on the way; both arms arrive."""
    cfg = cell_safety_config()
    q_start, q_goal = start_and_goals(twin)
    req = PlanRequest(
        q_start={"grip": q_start.tolist(), "view": VIEW_INIT},
        q_goal={"grip": q_goal.tolist(), "view": VIEW_GOAL},
        speed_scale=speed_scale,
    )
    res = twin.plan(req)
    assert res.ok, (res.failure, res.failing_pair)
    assert res.arm_order == ["grip", "view"]

    replay = Replay(twin, SafetyGate(twin, cfg), {"grip": q_start, "view": np.asarray(VIEW_INIT)})
    first = replay.session_start()
    assert [e.kind for e in first] == ["blocked"]  # the cell sat blocked, as in the log
    assert first[0].pairs == [tuple(sorted(PINCH_PAIR))]
    replay.execute(res, hw_jog(speed_scale))
    assert_clean_replay(replay, res, cleared=1)
    for arm in res.arm_order:
        assert replay.ticks[arm] > 1
    if speed_scale < 1.0:
        # slow enough to sit inside the hysteresis band [δ, δ + hysteresis) for a tick:
        # the gate must let the opening continue there instead of holding
        band = [
            d
            for d in replay.pair_dists
            if twin.inflation_m <= d < twin.inflation_m + cfg.hysteresis_m
        ]
        assert band, "the replay never sampled the hysteresis band"


def test_the_incident_shape_is_held_and_now_refused(twin):
    """What the pre-escape planner returned for this start (verified 2026-09-09 against the
    pre-change ``planner.py``): the whitelist excused the pinched pair at the GOAL too, so a
    goal 1.0 mm from ``view_link3`` planned as ONE straight closing segment. The gate holds
    its first tick with exactly the incident's pair; the shipped planner refuses the goal."""
    q_start = pinch_on_line(twin, GRIP_FREE, GRIP_INIT, PINCH_DEPTH_M)
    q_goal_in_shell = pinch_on_line(twin, GRIP_FREE, GRIP_INIT, 0.001)
    replay = Replay(twin, SafetyGate(twin, cell_safety_config()),
                    {"grip": q_start, "view": np.asarray(VIEW_INIT)})
    replay.session_start()
    old_shape = SimpleNamespace(
        arm_order=["grip"], waypoints={"grip": [q_start.tolist(), q_goal_in_shell.tolist()]}
    )
    finished = replay.execute(old_shape, hw_jog(1.0), stop_after=300)  # 3 s = plan_gate_hold_s
    assert not finished and replay.holds, "the closing segment must be held"
    # tick 1 re-commands the start itself (waypoint 0); the first MOVING tick is held with
    # the incident's pair. The 1 mm closing segment is 1.x ticks at 100 %, so the equal-tick
    # executor walks half of it per tick and the held command reads 1.5 mm (the incident's
    # full-tick executor read 1.1 mm in the log: "grip_right_finger / view_link3 at 1.1 mm").
    arm, tick, pairs, dist = replay.holds[0]
    assert (arm, tick) == ("grip", 2) and pairs == [tuple(sorted(PINCH_PAIR))]
    assert 0.0 < dist < PINCH_DEPTH_M and dist == pytest.approx(0.0015, abs=0.0002)
    assert len(replay.holds) == 299  # held on every following tick: hold-last-safe forever
    assert np.allclose(replay.q_meas["grip"], q_start)  # never moved

    res = ResetPlanner(twin).plan(
        PlanRequest(
            q_start={"grip": q_start.tolist(), "view": VIEW_INIT},
            q_goal={"grip": q_goal_in_shell.tolist(), "view": VIEW_INIT},
            arm_order=["grip", "view"],
        )
    )
    assert not res.ok and res.failure == "goal_in_collision"
    assert res.failing_pair == tuple(sorted(PINCH_PAIR)) and res.arm_order == []


def test_random_pinches_return_to_the_initial_condition_without_a_hold(twin):
    """Sweep (review item 6): random 2 mm pinches of the Manipulation Arm against the
    Perception Arm, planned for a 10 % session and replayed at 10 % AND 100 % - no tick is
    ever held, the block is ``cleared`` exactly once, the arm arrives at the seeded initial
    condition. Before the tick-rate criterion + equal-tick executor, 5 of 31 such replays
    at 10 % were held on a remainder tick (openings 4.5-8 um < 10 um). A ``no_escape`` is
    tolerated for at most one pinch (a start the gate itself would not let out at 10 %:
    every candidate direction opens some pinched pair by less than 10 um per 0.4 mm tick)."""
    cfg = cell_safety_config()
    planner = ResetPlanner(twin)
    lo, hi = planner._jnt_range["grip"]
    rng = np.random.default_rng(7)
    refused = []
    for i in range(6):
        q_start, pinch = random_pinch(twin, rng, lo, hi)
        req = PlanRequest(
            q_start={"grip": q_start.tolist(), "view": VIEW_INIT},
            q_goal={"grip": GRIP_INIT, "view": VIEW_INIT},
            speed_scale=0.1,
        )
        res = planner.plan(req)
        if not res.ok:
            assert res.failure == "no_escape", (i, res.failure, res.failing_pair, pinch)
            refused.append((i, res.failing_pair, pinch))
            continue
        for speed in (0.1, 1.0):
            replay = Replay(
                twin, SafetyGate(twin, cfg), {"grip": q_start, "view": np.asarray(VIEW_INIT)}
            )
            first = replay.session_start()
            assert [e.kind for e in first] == ["blocked"], (i, pinch)
            replay.execute(res, hw_jog(speed))
            assert_clean_replay(replay, res, cleared=1)
    assert len(refused) <= 1, refused


def test_band_pair_gate_history_is_escaped_at_ten_percent(twin):
    """Review item 5 (the author's open item): the gate keeps pairs inside its hysteresis
    band [δ, δ + 2 mm) in ``_block_pairs`` after a block and demands that they OPEN on every
    tick too. A two-pair teleop block followed by a partial retreat leaves P inside the shell
    and B inside the band; a return planned as if only P were pinched is held from its first
    moving tick at 10 % (B opens 4 um/tick) and cancelled after ``plan_gate_hold_s``. The
    planner now escapes the band too: replayed with the gate seeded the way the retreat
    leaves it, no tick is held at 10 % or 100 %, and B opens along the first segment."""
    cfg = cell_safety_config()
    q_start = np.asarray(GRIP_BAND_START)
    viol = violating_pairs(twin, {"grip": q_start, "view": VIEW_INIT})
    assert set(viol) == {BAND_PAIR_P} and 0.0 < viol[BAND_PAIR_P] < twin.inflation_m
    d_b = twin.pair_distance(BAND_PAIR_B, {"grip": q_start, "view": np.asarray(VIEW_INIT)})
    assert twin.inflation_m <= d_b < twin.inflation_m + cfg.hysteresis_m
    res = twin.plan(
        PlanRequest(
            q_start={"grip": q_start.tolist(), "view": VIEW_INIT},
            q_goal={"grip": GRIP_INIT, "view": VIEW_INIT},
            speed_scale=0.1,
        )
    )
    assert res.ok, (res.failure, res.failing_pair)
    wps = [np.asarray(w) for w in res.waypoints["grip"]]
    d_b1 = twin.pair_distance(BAND_PAIR_B, {"grip": wps[1], "view": np.asarray(VIEW_INIT)})
    assert d_b1 > d_b + PLANNER_ESCAPE_EPS_M  # the band pair is escaped, not tolerated
    for speed in (0.1, 1.0):
        gate = SafetyGate(twin, cfg)
        replay = Replay(twin, gate, {"grip": q_start, "view": np.asarray(VIEW_INIT)})
        first = replay.session_start()
        assert [e.kind for e in first] == ["blocked"]
        gate._blocked = True  # what the operator's retreat left behind
        gate._block_pairs = {BAND_PAIR_P, BAND_PAIR_B}
        replay.execute(res, hw_jog(speed))
        assert_clean_replay(replay, res, cleared=1)


def test_the_free_arm_plans_while_the_other_arm_is_pinched(twin):
    """Review blocker: pairs the planned arm cannot move are constants. The Perception Arm
    sits pinched against ITSELF (``view_link1`` / ``view_link5`` 7.96 mm - an intra-arm pair
    the monitored set does not even contain); the operator wants to move the Manipulation
    Arm. Single-arm ``goto``: plans, and the gate - which never holds an arm outside the
    offending set - passes every tick while the session stays ``blocked`` on the view pair.
    Two-arm request with the free arm first: plans too (the first escape implementation
    failed ``no_escape`` naming the OTHER arm's pair); the heuristic orders the pinched arm
    first."""
    cfg = cell_safety_config()
    q_view = np.asarray(VIEW_SELF_PINCH)
    grip_goal = np.asarray(GRIP_INIT)
    grip_goal[3] += 0.3
    grip_goal[7] = 0.45
    viol = violating_pairs(twin, {"grip": GRIP_INIT, "view": q_view})
    assert viol and all(a.startswith("view_") for p in viol for a in p), viol
    assert all(0.0 < d < twin.inflation_m for d in viol.values())
    planner = ResetPlanner(twin)
    res = planner.plan(
        PlanRequest(
            q_start={"grip": GRIP_INIT, "view": q_view.tolist()},
            q_goal={"grip": grip_goal.tolist()},
            speed_scale=0.1,
        )
    )
    assert res.ok, (res.failure, res.failing_pair)
    assert res.arm_order == ["grip"]
    for speed in (0.1, 1.0):
        replay = Replay(
            twin, SafetyGate(twin, cfg), {"grip": np.asarray(GRIP_INIT), "view": q_view}
        )
        first = replay.session_start()
        assert [e.kind for e in first] == ["blocked"] and first[0].arm_ids == ["view"]
        replay.execute(res, hw_jog(speed))
        assert replay.holds == [], replay.holds[:3]
        assert [e.kind for e in replay.events] == []  # still blocked on the view pair, no edge
        assert np.allclose(replay.q_meas["grip"], grip_goal, atol=1e-9)
    two = PlanRequest(
        q_start={"grip": GRIP_INIT, "view": q_view.tolist()},
        q_goal={"grip": grip_goal.tolist(), "view": VIEW_INIT},
    )
    assert planner._heuristic_order(two, ["grip", "view"]) == ["view", "grip"]
    res2 = planner.plan(two.model_copy(update={"arm_order": ["grip", "view"]}))
    assert res2.ok and res2.arm_order == ["grip", "view"], (res2.failure, res2.failing_pair)
    res3 = planner.plan(two)
    assert res3.ok and res3.arm_order == ["view", "grip"], (res3.failure, res3.failing_pair)


@pytest.mark.parametrize("speed_scale", [1.0, 0.1])
def test_an_unpinched_plan_replays_without_a_hold(twin, speed_scale):
    """Regression for the fine verification (03-sim §10 item 5): the unpinched Manipulation
    Arm from its reaching posture to the seeded initial condition. The pre-review planner's
    plan for this request was held for good by the real gate (``grip_right_finger`` /
    ``view_link3`` at 7.99 mm from tick 326, between two 0.05 rad samples); the shipped plan
    replays clean at 100 % and 10 %."""
    cfg = cell_safety_config()
    assert not violating_pairs(twin, {"grip": GRIP_FREE, "view": VIEW_INIT})
    res = twin.plan(
        PlanRequest(
            q_start={"grip": GRIP_FREE, "view": VIEW_INIT},
            q_goal={"grip": GRIP_INIT, "view": VIEW_INIT},
            speed_scale=speed_scale,
        )
    )
    assert res.ok, (res.failure, res.failing_pair)
    replay = Replay(
        twin, SafetyGate(twin, cfg), {"grip": np.asarray(GRIP_FREE), "view": np.asarray(VIEW_INIT)}
    )
    assert replay.session_start() == []
    replay.execute(res, hw_jog(speed_scale))
    assert_clean_replay(replay, res, cleared=0)


# The 200-start fuzz sweep (tests/test_return_fuzz_mavis_v2.py, seeds 20261013 / 20261014):
# the Perception Arm's camera mount 6.7 mm from ``grip_rail_base`` - the Manipulation Arm's
# STATIC rail body (a child of ``world`` with no joint). By name prefix the gate attributed
# the pair to BOTH arms and demanded that the Manipulation Arm open a distance none of its
# joints can change: its planned return was held from the first moving tick for good, and
# teleop would have frozen the same way. Pair ownership is kinematic now (sim twin).
RAIL_BASE_PINCH_GRIP = [-2.148, -0.2721, 0.1854, 1.0892, 1.7294, 0.2123, 0.8133, 0.503]
RAIL_BASE_PINCH_VIEW = [0.9105, -1.2391, 2.4242, 0.6749, 0.8878, -0.1517, 2.2932, 0.0952]
RAIL_BASE_PAIR = ("grip_rail_base", "view_d435_mount")


@pytest.mark.parametrize("speed_scale", [1.0, 0.1])
def test_the_other_arm_pinched_against_this_arms_rail_base_does_not_freeze_it(
    twin, speed_scale
):
    """Replayed through the real gate: the session-start block names the Perception Arm
    ONLY; the Manipulation Arm's return (planned as a constant-pair plan, in either
    order) passes every tick while the block persists; the Perception Arm escapes it;
    both arrive; ``cleared`` exactly once."""
    cfg = cell_safety_config()
    q_grip, q_view = np.asarray(RAIL_BASE_PINCH_GRIP), np.asarray(RAIL_BASE_PINCH_VIEW)
    viol = violating_pairs(twin, {"grip": q_grip, "view": q_view})
    assert RAIL_BASE_PAIR in viol and 0.0 < viol[RAIL_BASE_PAIR] < twin.inflation_m
    assert twin._arms_of_pair(RAIL_BASE_PAIR) == ["view"]
    goal = {
        "grip": GRIP_INIT[:7] + [RAIL_BASE_PINCH_GRIP[7]],
        "view": VIEW_INIT[:7] + [RAIL_BASE_PINCH_VIEW[7]],
    }
    for order in (None, ["grip", "view"]):  # the heuristic's (view first) and grip first
        res = twin.plan(
            PlanRequest(
                q_start={"grip": q_grip.tolist(), "view": q_view.tolist()},
                q_goal=goal,
                speed_scale=speed_scale,
                arm_order=order,
            )
        )
        assert res.ok, (order, res.failure, res.failing_pair)
        replay = Replay(twin, SafetyGate(twin, cfg), {"grip": q_grip, "view": q_view})
        first = replay.session_start()
        assert [e.kind for e in first] == ["blocked"]
        assert first[0].arm_ids == ["view"], first[0]
        assert RAIL_BASE_PAIR in first[0].pairs
        replay.execute(res, hw_jog(speed_scale))
        assert_clean_replay(replay, res, cleared=1)
        for arm in res.arm_order:
            assert replay.ticks[arm] > 1
