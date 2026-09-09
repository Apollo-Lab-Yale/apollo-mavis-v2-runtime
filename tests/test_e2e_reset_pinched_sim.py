"""Full-stack sim e2e of the reset / return-to-initial flow from a PINCHED two-arm
start, over a REAL uvicorn server with the FULL safety stack (``safety_debug`` gate +
twin) on the ``mavis_v2`` cell (2026-09-09; the operator asked for the whole reset /
return flow to be proven in simulation FIRST — every real-cell attempt so far failed:
the 2026-09-08 23:16 simultaneous execution and the 2026-09-09 01:14 whitelisted pinched
pair).

Why this exists on top of ``tests/test_plan_passes_gate.py`` (which replays a plan tick
by tick through the real ``SafetyGate``) and ``tests/test_reset_to_initial.py`` /
``tests/test_return_to_start.py`` (which drive the manager over guardrail_env, ONE arm,
NullGate): none of those exercise the WHOLE stack — the WS ``reset_to_initial`` action /
``POST /api/session/return_home`` -> ``SessionManager`` two-phase, one-arm-at-a-time
``_execute_arms`` with measured-arrival hand-over -> ``ControlLoop`` -> the real
``SafetyGate`` over the ``mavis_v2`` twin -> telemetry — from a real sub-δ pinch of the
two real arms. This file does.

What this file PROVES (2026-09-09 review), and where:

* sequential execution + measured arrival: both return tests sample telemetry at 25 Hz
  and assert that no sample shows both arms moving, that the sequence of movers equals
  the planner's ``PlanResult.arm_order`` of each phase (captured from the manager's
  ``_plan_return`` - not merely printed), and that both arms end at the initial posture;
* the ESCAPE from the pinch: ``test_reset_to_initial_key_returns_both_arms_sequentially``
  (and the ``return_home`` twin of it) records ``ResetPlanner._escape`` and checks the
  executed plan on a private twin: the pinched start's STRAIGHT joint-space segment to the
  initial posture first CLOSES the pinch pair (the premise - the pre-2026-09-09 planner,
  which whitelisted pairs already violating at the start, would have planned exactly that
  segment and the gate would have held it, the 01:14 incident), while the plan's first
  segment OPENS it. The earlier constants of this file had a start whose straight segment
  happened to open the pair, so the tests passed with the escape removed.

The Perception Arm carries the MICROPHONE body here (``microphone: true``), as in the
hardware twin the real cell gates with (``configs/mavis_v2.yaml``); the sim scene's
default is mic-less. ``SafetyConfig`` is core's defaults plus ``safety_debug`` (the
cell config pins nothing else).

Reproducing the pinched START faithfully (04-runtime §7). Under the real gate an arm
CANNOT be driven into a sub-δ pinch through any gated path: teleop / jog / plan commands
that would cross δ are hold-last-safe'd at the boundary (~δ), never inside it, and
``start_from`` to a pinched profile is refused ``goal_in_collision`` before it moves.
The real cell got pinched because the arms were LEFT there (Studio / a previous session);
the twin sat blocked at 2.1 mm. We reproduce exactly that resting state: after the
session is running we write the pinched joint targets straight into the sim workcell,
pin the loop's held command and re-seed the gate's last-safe to the same posture, and
let the servo settle. The gate then sees the pinch as the resting COMMANDED config,
reports ``blocked`` and (rising edge) raises one ``blocked`` event — the "cell sat
blocked" of the incident. From there the ``R`` key / exit return must escape it.

The pinched posture is the ``(grip_right_finger, view_link3)`` pair at ~2.5 mm — the
incident's geometry — found by the bisection method of ``tests/test_plan_passes_gate``
(``pinch_on_line``) on this twin: the Manipulation Arm walks a joint-space line from a
random clear posture TOWARD its initial posture and the first crossing of 2.5 mm is the
start, so continuing straight toward the goal closes the pair (to -7.9 mm penetration at
2 % of the segment); the Perception Arm's wrist joints are displaced so its ``link3`` is
what the finger closes on. ``test_pinched_start_reports_blocked_...`` verifies the
resting geometry landed inside δ, and the return tests re-verify the closing premise on
the MEASURED start, so the hard-coded numbers stay honest.
"""

from __future__ import annotations

import math
import threading
import time

import httpx
import numpy as np
import pytest
from apollo_mavis_v2_core import ArmPosture, StateProfile
from conftest import ControlConfig, DatasetsConfig, LiveServer, RuntimeConfig, VideoConfig
from test_e2e_teleop import PulsingCtl, Tele

from apollo_mavis_v2_runtime.session.manager import SessionManager

pytest.importorskip("apollo_mavis_v2_sim")
from apollo_mavis_v2_sim import REGISTRY, DigitalTwin, SceneOverrides  # noqa: E402
from apollo_mavis_v2_sim.planner import REARM_MARGIN_M, ResetPlanner  # noqa: E402

pytestmark = pytest.mark.egl

DEG = math.pi / 180.0
ARMS = ("grip", "view")
PINCH_PAIR = ("grip_right_finger", "view_link3")
DELTA_M = 0.008  # SafetyConfig.geom_inflation_m default = the gate's δ
SPEC = {
    "mode": "teleop",
    "kind": "sim",
    "arms": ["grip", "view"],
    "frames": {"grip": "arm_base:grip", "view": "arm_base:view"},
    "sim_scene": "mavis_v2",
}

# The operator's seeded default posture (``profiles.seed_initial.DEFAULT_POSTURE_DEG``) as
# the initial condition. Joint 1 is spelled +pi instead of -180 deg: the same posture, but
# the sim keyframe base sits at +pi, and a return plan would otherwise turn the base a full
# 2 pi (same convention as tests/test_teleop_axis_purity.initial_profile). Carriages are set
# here (grip 0.65 = operator's right end, view 0.0 = left end, the cell's opposite-ends
# start) so the return runs BOTH phases — the joints, then the carriages — and the carriage
# assertion has teeth; ``seed_initial`` itself leaves ``rail_pos_m`` unset ("keep").
GRIP_INIT = [x * DEG for x in (-180.0, -12.0, -20.0, 30.0, -5.0, 35.0, -8.9)]
GRIP_INIT[0] += 2.0 * math.pi
VIEW_INIT = [x * DEG for x in (0.0, 0.8, 0.0, 28.9, 0.0, 28.2, 0.0)]
GRIP_INIT_RAIL = 0.65
VIEW_INIT_RAIL = 0.0

# The pinched posture: ``(grip_right_finger, view_link3)`` at ~2.5 mm with both carriages at
# mid-travel (0.3 m), replicating the 2026-09-09 01:14 geometry. Found by bisection on the
# mic-equipped twin (tests/test_plan_passes_gate.pinch_on_line, seed 1 of the search): the
# Manipulation Arm on the joint-space line from a random clear posture TOWARD ``GRIP_INIT``
# (rail held), at the first crossing of 2.5 mm, so the straight segment to the goal CLOSES
# the pair (1.74 mm at 0.1 % of it, -1.26 mm at 0.5 %, -7.86 mm at 2 %); the Perception
# Arm's wrist (joints 4/5/7) is displaced from its initial posture so its link3 is the body
# the finger closes on. Only this pair is inside δ (the next, link3 again at 17.8 mm).
# Verified inside δ at rest by ``test_pinched_start_reports_blocked_with_the_pair_inside_delta``
# and closing on the measured start by the return tests (``_assert_closing_premise``).
GRIP_PINCH = [0.30057, -1.13817, -0.01429, 0.20784, 0.37752, -0.42138, -0.07023]
VIEW_PINCH = [0.0, 0.014, 0.0, 0.5044, 0.4, 0.1922, 0.5]
PINCH_RAIL = 0.3
CLOSING_FRACTIONS = (0.005, 0.01)  # of the straight segment: the pair must be closer there


def _make_config(tmp_path) -> RuntimeConfig:
    """A ``mavis_v2`` sim workcell with BOTH arms (the Perception Arm with its microphone
    body) and the full hardware safety stack (``safety_debug`` -> real ``SafetyGate`` +
    twin + clearance sweep). Every root is pinned under ``tmp_path`` so the test never
    reads or writes the operator's ``var/``.
    """
    wc = {
        "kind": "sim",
        "sim_scene": "mavis_v2",
        "arms": [
            # the microphone body as on the real cell's twin (configs/mavis_v2.yaml)
            {"id": "view", "base_in_world": {}, "gripper": "none", "microphone": True},
            {"id": "grip", "base_in_world": {}},
        ],
        "cameras": [],
        "safety": {"safety_debug": True, "geom_inflation_m": DELTA_M},
    }
    return RuntimeConfig(
        workcells={"sim": wc},
        profiles_dir=tmp_path / "profiles",
        datasets_root=tmp_path / "datasets",
        datasets=DatasetsConfig(default_namespace="apollo", namespaces={}),
        checkpoints_root=tmp_path / "ckpts",
        calibration_dir=tmp_path / "calibration",
        video=VideoConfig(preview_fps=15, session_fps=30),
        # base frame: this file asserts nothing about the keyboard frame; it never
        # teleops, it presses R / calls return_home. Pin the deterministic pre-2026-09-08
        # frame like the other e2e suites (conftest.make_runtime_config note).
        control=ControlConfig(translate_frame="base"),
    )


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    srv = LiveServer(_make_config(tmp_path_factory.mktemp("rt")))
    store = srv.runtime.profile_store
    initial = store.save(
        StateProfile(
            name="initial",
            workcell_kind="sim",
            arms={
                "grip": ArmPosture(q=GRIP_INIT, rail_pos_m=GRIP_INIT_RAIL, gripper_open_frac=1.0),
                "view": ArmPosture(q=VIEW_INIT, rail_pos_m=VIEW_INIT_RAIL),
            },
        )
    )
    store.set_initial(initial.profile_id)
    pinched = store.save(
        StateProfile(
            name="pinched",
            workcell_kind="sim",
            arms={
                "grip": ArmPosture(q=GRIP_PINCH, rail_pos_m=PINCH_RAIL),
                "view": ArmPosture(q=VIEW_PINCH, rail_pos_m=PINCH_RAIL),
            },
        )
    )
    srv.initial_id = initial.profile_id  # type: ignore[attr-defined]
    srv.pinched_id = pinched.profile_id  # type: ignore[attr-defined]
    yield srv
    srv.stop()


@pytest.fixture()
def api(server):
    with httpx.Client(base_url=server.http, timeout=120.0) as client:
        yield client


@pytest.fixture(scope="module")
def check_twin() -> DigitalTwin:
    """A PRIVATE twin (same scene + microphone + δ as the session's) for the geometric
    checks of the recorded plans: the session twin is the gate's, written by the loop
    thread every tick, so the tests never touch it from their own thread."""
    return DigitalTwin(
        REGISTRY.build("mavis_v2", SceneOverrides(microphones={"view": True})),
        inflation_m=DELTA_M,
    )


class PlanRecorder:
    """Records what the manager PLANNED - ``(q_start, q_goal, PlanResult)`` per phase from
    ``SessionManager._plan_return`` - and every ``ResetPlanner._escape`` call
    ``(arm_id, pinched pairs, path length | failure)``, so the tests can hold the executed
    motion against the planner's own ``arm_order`` and prove the escape happened."""

    def __init__(self, server, monkeypatch) -> None:
        self.plans: list[tuple[dict, dict, object]] = []
        self.escapes: list[tuple[str, list, object]] = []
        manager = server.runtime.manager
        orig_plan = manager._plan_return

        def plan_return(session, states, q_start, q_goal):
            result = orig_plan(session, states, q_start, q_goal)
            starts = {a: list(q) for a, q in q_start.items()}
            goals = {a: list(q) for a, q in q_goal.items()}
            self.plans.append((starts, goals, result))
            return result

        orig_escape = ResetPlanner._escape

        def escape(planner, arm_id, q_start, pinched, *args, **kwargs):
            out = orig_escape(planner, arm_id, q_start, pinched, *args, **kwargs)
            self.escapes.append(
                (arm_id, sorted(pinched), len(out) if isinstance(out, list) else out.failure)
            )
            return out

        monkeypatch.setattr(manager, "_plan_return", plan_return)
        monkeypatch.setattr(ResetPlanner, "_escape", escape)

    def expected_movers(self) -> list[str]:
        """The arms the manager submits, phase after phase, in ``arm_order`` (parked arms
        - those whose waypoints go nowhere - are never submitted)."""
        out: list[str] = []
        for _q_start, _q_goal, res in self.plans:
            if res.ok:
                out += SessionManager._moving_arms(SessionManager._ordered_waypoints(res))
        return out


@pytest.fixture()
def recorder(server, monkeypatch) -> PlanRecorder:
    return PlanRecorder(server, monkeypatch)


# -- helpers --------------------------------------------------------------------------


def _wait_running(api) -> None:
    for _ in range(400):
        if api.get("/api/session").json()["state"] == "running":
            return
        time.sleep(0.05)
    raise AssertionError("session never reached running")


def _full_q(state) -> np.ndarray:
    q = list(state.q[:7])
    q.append(float(state.q[7]) if state.q.shape[0] > 7 else 0.0)
    return np.asarray(q, dtype=np.float64)


def _joint_err(state, goal) -> float:
    return float(np.max(np.abs(np.asarray(state.q[:7], dtype=np.float64) - np.asarray(goal))))


def _force_pinch(server) -> float:
    """Place both live arms at the pinched posture (module docstring) and wait for the
    servo to SETTLE there. Returns the settled ``PINCH_PAIR`` clearance (m).

    Waiting on the pair distance alone is wrong: the two arms swing in from the keyframe
    (grip from the far rail end, the Perception Arm's wrist across), the pair momentarily
    grazes ~0 mm and then the still-moving wrist bounces it back through ~15 mm before it
    settles to ~2.5 mm at ~1.3 s. So wait until BOTH arms have actually ARRIVED at their
    pinch targets (joints < 1e-3 rad, carriage < 2 mm), then read the resting clearance."""
    session = server.runtime.manager.session
    loop, workcell, supervisor = session.loop, session.workcell, session.supervisor
    addr = workcell.scene.addressing
    qg = np.array([*GRIP_PINCH, PINCH_RAIL], dtype=np.float64)
    qv = np.array([*VIEW_PINCH, PINCH_RAIL], dtype=np.float64)
    workcell._write_targets(addr["grip"].ctrl_adr, qg)
    workcell._write_targets(addr["view"].ctrl_adr, qv)
    loop._last_cmd["grip"] = qg.copy()  # hold the loop THERE (else it servos back to keyframe)
    loop._last_cmd["view"] = qv.copy()
    supervisor.reseed("grip", qg)  # gate last-safe = the pinch, so it holds it, blocked
    supervisor.reseed("view", qv)

    def _at_pinch(state, target) -> bool:
        return _joint_err(state, target[:7]) < 1e-3 and abs(float(state.q[7]) - target[7]) < 2e-3

    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        st = workcell.states()
        if _at_pinch(st["grip"], qg) and _at_pinch(st["view"], qv):
            time.sleep(0.3)  # let the resting state show up on telemetry / in the gate
            return session.twin.pair_distance(PINCH_PAIR, None, 0.5)
        time.sleep(0.05)
    raise AssertionError(
        f"arms never settled at the pinch: {PINCH_PAIR} at "
        f"{session.twin.pair_distance(PINCH_PAIR, None, 0.5) * 1e3:.2f} mm"
    )


def _arrived_initial(session) -> bool:
    st = session.workcell.states()
    return (
        _joint_err(st["grip"], GRIP_INIT) < 1e-3
        and abs(float(st["grip"].q[7]) - GRIP_INIT_RAIL) < 2e-3
        and _joint_err(st["view"], VIEW_INIT) < 1e-3
        and abs(float(st["view"].q[7]) - VIEW_INIT_RAIL) < 2e-3
    )


def _wait_arrival(session, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _arrived_initial(session):
            return
        time.sleep(0.05)
    raise AssertionError("arms never reached the initial condition")


def _wait_motion_settled(server, timeout: float = 20.0) -> None:
    """Wait until the profile-motion worker has fully unwound (no claim in flight).

    ``_wait_arrival`` returns the instant the MEASURED arms reach the goal, which is
    inside the last phase's arrival wait — the worker still has to end the phase loop,
    write its outcome and release the motion claim. Anything that then inspects the loop
    (``plans.active_arms``) or the session notice must wait for this, else it races the
    worker's tail (a phase's last arm can still be settling in the executor)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.runtime.manager.profile_motion_in_flight is None:
            return
        time.sleep(0.05)
    raise AssertionError("profile motion never finished unwinding")


def _assert_reached_initial(session) -> None:
    st = session.workcell.states()
    assert _joint_err(st["grip"], GRIP_INIT) < 1e-3, ("grip joints", st["grip"].q[:7])
    assert abs(float(st["grip"].q[7]) - GRIP_INIT_RAIL) < 2e-3, ("grip rail", st["grip"].q[7])
    assert _joint_err(st["view"], VIEW_INIT) < 1e-3, ("view joints", st["view"].q[:7])
    assert abs(float(st["view"].q[7]) - VIEW_INIT_RAIL) < 2e-3, ("view rail", st["view"].q[7])


class _Sampler(threading.Thread):
    """Samples telemetry (its own /ws/telemetry socket) at ~25 Hz into ``samples`` —
    ``(plan_status, {arm: full-q})`` per frame — so the reset can be judged one arm at a
    time. ``Tele.latest`` paces at the telemetry rate."""

    def __init__(self, server) -> None:
        super().__init__(name="reset-sampler", daemon=True)
        self.tele = Tele(server)
        self.samples: list[tuple[str | None, dict[str, np.ndarray]]] = []
        self._halt = threading.Event()  # NOT ``_stop``: that is a Thread internal
        self._closed = False

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                msg = self.tele.latest()
            except Exception:  # noqa: BLE001 - socket closed at stop()
                return
            qs = {
                a["arm_id"]: np.asarray(
                    [*a["q"], a["rail_pos_m"] if a["rail_pos_m"] is not None else 0.0],
                    dtype=np.float64,
                )
                for a in msg["arms"]
            }
            self.samples.append(((msg.get("session") or {}).get("plan_status"), qs))

    def stop(self) -> None:
        """Idempotent and never raises: a failing stop must not mask the session delete."""
        self._halt.set()
        try:
            self.join(timeout=3.0)
        finally:
            if not self._closed:
                self._closed = True
                try:
                    self.tele.close()
                except Exception:  # noqa: BLE001
                    pass


def _moving_between(q0: dict, q1: dict, thresh: float = 1e-3) -> list[str]:
    """Arms whose full-q changed by more than ``thresh`` between two samples. The moving
    arm advances ~0.08 rad / 8 mm per 40 ms sample; a held arm holds within servo noise
    (< 1e-4), so ``thresh`` cleanly separates them."""
    return [
        a
        for a in ARMS
        if a in q0 and a in q1 and float(np.max(np.abs(q1[a] - q0[a]))) > thresh
    ]


def _assert_one_arm_at_a_time(samples) -> None:
    for (_, q0), (_, q1) in zip(samples, samples[1:], strict=False):
        moving = _moving_between(q0, q1)
        assert len(moving) < 2, f"both arms moved between two 25 Hz samples: {moving}"


def _plan_status_walk(samples) -> list[str]:
    walk: list[str] = []
    for status, _ in samples:
        if status and (not walk or walk[-1] != status):
            walk.append(status)
    return walk


def _first_mover(samples) -> str | None:
    if not samples:
        return None
    base = samples[0][1]
    for _, qs in samples[1:]:
        moving = _moving_between(base, qs)
        if moving:
            return moving[0]
    return None


def _mover_sequence(samples) -> list[str]:
    """The arms that moved, in order, consecutive repeats collapsed (``[grip, view]`` for a
    phase that moved grip then view; a phase boundary where the same arm moves twice in a
    row collapses too, so both sides are compared collapsed)."""
    seq: list[str] = []
    for (_, q0), (_, q1) in zip(samples, samples[1:], strict=False):
        for arm in _moving_between(q0, q1):
            if not seq or seq[-1] != arm:
                seq.append(arm)
    return seq


def _collapse(seq: list[str]) -> list[str]:
    out: list[str] = []
    for a in seq:
        if not out or out[-1] != a:
            out.append(a)
    return out


def _assert_executed_in_arm_order(samples, rec: PlanRecorder) -> None:
    """The executed order IS the planner's: the first mover is ``arm_order[0]`` of the
    joints phase and the whole mover sequence equals the phases' ``arm_order`` (moving
    arms only), collapsed."""
    assert len(rec.plans) == 2, [p[2].failure for p in rec.plans]  # joints, then carriage
    assert all(p[2].ok for p in rec.plans), [(p[2].failure, p[2].failing_pair) for p in rec.plans]
    expected = rec.expected_movers()
    assert len(expected) >= 3, expected  # both arms in the joints phase, at least one carriage
    first = _first_mover(samples)
    assert first == rec.plans[0][2].arm_order[0] == expected[0], (first, rec.plans[0][2].arm_order)
    assert _mover_sequence(samples) == _collapse(expected), (_mover_sequence(samples), expected)


def _assert_closing_premise_and_escape(check_twin, rec: PlanRecorder) -> str:
    """On the recorded joints-phase plan: (premise) the first-moving arm's STRAIGHT segment
    from its MEASURED start to its goal closes ``PINCH_PAIR`` - what the pre-escape planner
    would have executed and the gate would have held; (proof) the planner's escape ran for
    that arm from a start with the pair inside δ, and the executed plan's first segment
    OPENS the pair by at least 1 mm. Returns a one-line report."""
    q_start, q_goal, res = rec.plans[0]
    first = res.arm_order[0]
    ctx = {a: np.asarray(q_start[a], dtype=np.float64) for a in ARMS}
    start, goal = ctx[first], np.asarray(q_goal[first], dtype=np.float64)
    d0 = check_twin.pair_distance(PINCH_PAIR, ctx)
    assert 0.0 < d0 < DELTA_M, d0
    straight = [
        check_twin.pair_distance(PINCH_PAIR, {**ctx, first: start + f * (goal - start)})
        for f in CLOSING_FRACTIONS
    ]
    assert all(d < d0 for d in straight), (d0, straight)  # the premise: it CLOSES
    escapes = [e for e in rec.escapes if e[0] == first]
    assert escapes, rec.escapes
    arm, pinched, length = escapes[0]
    assert tuple(sorted(PINCH_PAIR)) in pinched, pinched
    assert isinstance(length, int) and length >= 2, length  # >= 1 escape segment
    wps = [np.asarray(w, dtype=np.float64) for w in res.waypoints[first]]
    assert len(wps) >= 3, len(wps)  # escape + at least one RRT segment
    assert np.allclose(wps[0], start, atol=1e-9)
    d1 = check_twin.pair_distance(PINCH_PAIR, {**ctx, first: wps[1]})
    assert d1 >= d0 + 1e-3, (d0, d1)  # the first executed segment OPENS the pair
    # the pinched pair is re-armed (past δ + REARM_MARGIN_M) somewhere before the goal
    along = [check_twin.pair_distance(PINCH_PAIR, {**ctx, first: w}) for w in wps[1:-1]]
    assert any(d > DELTA_M + REARM_MARGIN_M for d in along), along
    return (
        f"first mover {first}: pair {d0 * 1e3:.2f} mm at the start, straight segment "
        f"{[round(d * 1e3, 2) for d in straight]} mm at {CLOSING_FRACTIONS} (closes), "
        f"escape {length - 1} segment(s), plan's first segment -> {d1 * 1e3:.2f} mm, "
        f"{len(wps)} waypoints"
    )


def _new_event_kinds(supervisor, since: int) -> list[str]:
    return [e.kind for e in list(supervisor.events)[since:]]


# -- step 2: the pinched start is blocked, the pair is inside δ -----------------------


def test_pinched_start_reports_blocked_with_the_pair_inside_delta(server, api):
    """After the forced pinch the gate reports ``blocked`` on ``PINCH_PAIR`` and the 25 Hz
    clearance sweep's tightest pair is that pair inside δ (2-5 mm, > 0 = no penetration) —
    the resting state the ``R`` key / exit return have to escape."""
    assert api.post("/api/session", json=SPEC).status_code == 200
    _wait_running(api)
    tele = Tele(server)
    try:
        settled = _force_pinch(server)
        assert 0.0 < settled < DELTA_M, f"{settled * 1e3:.2f} mm"
        msg = tele.latest()
        col = msg["collision"]
        assert col["severity"] == "blocked", col
        assert tuple(sorted(col["pairs"][0])) == tuple(sorted(PINCH_PAIR)), col["pairs"]
        assert 0.0 < col["min_clearance_m"] < DELTA_M, col["min_clearance_m"]
        top = msg["clearances"][0]
        assert tuple(sorted(top["pair"])) == tuple(sorted(PINCH_PAIR)), top
        assert 0.0 < top["dist_m"] < DELTA_M, top
        print(
            f"REPORT pinched-start: {PINCH_PAIR} at {top['dist_m'] * 1e3:.2f} mm at rest, "
            f"gate severity={col['severity']} min={col['min_clearance_m'] * 1e3:.2f} mm"
        )
    finally:
        tele.close()
        api.delete("/api/session")


# -- step 3: the R key returns both arms, ONE AT A TIME, no gate hold ----------------


def test_reset_to_initial_key_returns_both_arms_sequentially(server, api, recorder, check_twin):
    """``R`` (WS ``reset_to_initial``) from the pinch: ack ok; telemetry ``plan_status``
    walks executing -> done; both arms reach the initial posture (joint < 1e-3 rad,
    carriage < 2 mm); the motion runs ONE ARM AT A TIME (no 25 Hz sample shows both arms
    moving) IN THE PLANNER'S ``arm_order`` (first mover and the whole mover sequence, per
    phase); the planner ESCAPED the pinch (``_assert_closing_premise_and_escape``: the
    straight segment would have closed the pair, the executed first segment opens it);
    NO ``blocked`` CollisionEvent after the plan began (the pinch is ``cleared``); the
    session-level notice is empty."""
    assert api.post("/api/session", json=SPEC).status_code == 200
    _wait_running(api)
    ctl = PulsingCtl(server)
    sampler = _Sampler(server)
    try:
        _force_pinch(server)
        session = server.runtime.manager.session
        supervisor = session.supervisor
        events_before = len(supervisor.events)
        sampler.start()
        t0 = time.monotonic()
        ack = ctl.action("reset_to_initial")
        assert ack["ok"] is True and ack["detail"] == "returning to 'initial'", ack
        _wait_arrival(session)
        _wait_motion_settled(server)  # the worker fully unwound before we inspect the loop
        wall = time.monotonic() - t0
        time.sleep(0.4)  # let the last telemetry frames land
        sampler.stop()

        _assert_reached_initial(session)
        _assert_one_arm_at_a_time(sampler.samples)
        _assert_executed_in_arm_order(sampler.samples, recorder)
        escape_report = _assert_closing_premise_and_escape(check_twin, recorder)
        walk = _plan_status_walk(sampler.samples)
        assert any(
            walk[i] == "executing" and "done" in walk[i + 1 :] for i in range(len(walk))
        ), walk
        kinds = _new_event_kinds(supervisor, events_before)
        assert "blocked" not in kinds, kinds  # no NEW block once the escape began
        assert "cleared" in kinds, kinds  # the pinched pair is resolved on the way out
        assert (session.motion_detail or "") == "", session.motion_detail
        assert not session.loop.plans.active_arms
        print(
            f"REPORT R-key reset: wall {wall:.1f}s, movers {_mover_sequence(sampler.samples)} "
            f"= arm_order {[p[2].arm_order for p in recorder.plans]}, plan_status walk {walk}, "
            f"gate events {kinds}; {escape_report}"
        )
    finally:
        sampler.stop()
        ctl.close()
        api.delete("/api/session")


# -- step 4a: goto_profile back to the pinched profile = an honest goal_in_collision ---


def test_goto_profile_back_to_the_pinched_profile_is_a_goal_in_collision_refusal(server, api):
    """After returning to the initial condition, ``goto_profile`` back to the pinched
    profile: the fire-and-forget ack only says it STARTED; the outcome lands on the
    session notice (``telemetry.session.fault_detail``) as ``goal_in_collision`` naming
    the pinch pair — a goal inside δ is refused by design. The joint phase is reachable at
    the initial carriages (the arms far apart), so the honest refusal falls on the
    CARRIAGE phase, which slides the pinch closed."""
    assert api.post("/api/session", json=SPEC).status_code == 200
    _wait_running(api)
    ctl = PulsingCtl(server)
    tele = Tele(server)
    try:
        _force_pinch(server)
        session = server.runtime.manager.session
        assert ctl.action("reset_to_initial")["ok"] is True
        _wait_arrival(session)

        ack = ctl.action("goto_profile", {"profile_id": server.pinched_id})
        assert ack["ok"] is True and ack["detail"] == "going to profile 'pinched'", ack
        # The fire-and-forget motion runs its (safe) joint phase to the pinch joints at
        # the initial carriages, then refuses the carriage phase; wait for it to fully
        # unwind so the notice and the loop are read race-free.
        _wait_motion_settled(server, timeout=40.0)
        detail = (tele.latest().get("session") or {}).get("fault_detail") or ""
        assert "goal_in_collision" in detail, detail
        assert "grip_right_finger" in detail and "view_link3" in detail, detail
        assert not session.loop.plans.active_arms
        print(f"REPORT goto_profile back to pinched: honest refusal -> {detail!r}")
    finally:
        ctl.close()
        tele.close()
        api.delete("/api/session")


# -- step 4b: the exit path (POST return_home) from a FRESH pinch ---------------------


def test_return_home_exit_path_returns_sequentially_from_a_fresh_pinch(
    server, api, recorder, check_twin
):
    """``POST /api/session/return_home`` (the Cockpit's "End session" path) from a fresh
    pinch: synchronous ``status: done`` with the initial profile id and both arms; both
    arms reach the initial posture; the motion ran ONE ARM AT A TIME in the planner's
    ``arm_order``; the pinch was escaped (same checks as the ``R``-key test); NO
    ``blocked`` CollisionEvent after the plan began."""
    assert api.post("/api/session", json=SPEC).status_code == 200
    _wait_running(api)
    sampler = _Sampler(server)
    try:
        _force_pinch(server)
        session = server.runtime.manager.session
        supervisor = session.supervisor
        events_before = len(supervisor.events)
        sampler.start()
        t0 = time.monotonic()
        res = api.post("/api/session/return_home")
        wall = time.monotonic() - t0
        time.sleep(0.4)
        sampler.stop()

        assert res.status_code == 200, res.text
        body = res.json()
        assert body["ok"] is True and body["status"] == "done" and body["detail"] == "", body
        assert body["profile_id"] == server.initial_id, body
        assert sorted(body["arms"]) == ["grip", "view"], body
        _assert_reached_initial(session)
        _assert_one_arm_at_a_time(sampler.samples)
        _assert_executed_in_arm_order(sampler.samples, recorder)
        escape_report = _assert_closing_premise_and_escape(check_twin, recorder)
        walk = _plan_status_walk(sampler.samples)
        kinds = _new_event_kinds(supervisor, events_before)
        assert "blocked" not in kinds, kinds
        assert not session.loop.plans.active_arms
        print(
            f"REPORT return_home exit: wall {wall:.1f}s, movers "
            f"{_mover_sequence(sampler.samples)} = arm_order "
            f"{[p[2].arm_order for p in recorder.plans]}, plan_status walk {walk}, "
            f"gate events {kinds}; {escape_report}"
        )
    finally:
        sampler.stop()
        api.delete("/api/session")
