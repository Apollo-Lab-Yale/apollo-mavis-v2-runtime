"""``devices/rail_sweep.py`` (phase-09c): the full-travel twin sweep that gates the
``home_rail`` maintenance op, over the real ``mavis_v2`` twin (kinematics only:
no GL, no renderer). One private twin per module (built lazily on the first
check)."""

from __future__ import annotations

import math
import time

import pytest
from apollo_mavis_v2_core import WorkcellConfig
from apollo_mavis_v2_core.protocol import RailSweepVerdict
from conftest import FakeMonitorSample

from apollo_mavis_v2_runtime.devices.rail_sweep import (
    DEFAULT_SWEEP_INFLATION_M,
    DEFAULT_SWEEP_STEP_M,
    RailSweepChecker,
    describe_verdict,
    sweep_positions,
)

PI = math.pi
KEYFRAME_Q = (PI, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # xArm7 factory zero, joint 1 = pi (base flip)
# Manipulation Arm bent so the gripper dives below the table top (found on the twin):
# the sweep must block on a table pair at the very first step.
TABLE_DIVE_Q = (PI, 0.8, 0.0, 0.5, 0.0, 0.3, 0.0)
FALLBACK = {"grip": 0.65, "view": 0.0}

HW = WorkcellConfig.model_validate(
    {
        "kind": "hardware",
        "digital_twin_scene": "mavis_v2",
        "arms": [
            {"id": "grip", "ip": "192.168.1.201", "base_in_world": {}, "gripper": "xarm_g2"},
            {
                "id": "view",
                "ip": "192.168.2.219",
                "base_in_world": {},
                "gripper": "none",
                "microphone": True,
            },
        ],
        "cameras": [],
        "safety": {"enabled": True},
    }
)


def _sample(arm_id: str, q=KEYFRAME_Q, *, seq: int = 1, rail_pos_m=None, homed=False, **kw):
    return FakeMonitorSample(
        arm_id,
        seq=seq,
        t_mono=time.monotonic(),
        q=tuple(q),
        rail_present=True,
        rail_homed=homed,
        rail_enabled=homed,
        rail_pos_m=rail_pos_m,
        **kw,
    )


@pytest.fixture(scope="module")
def checker():
    chk = RailSweepChecker(HW, "mavis_v2")
    assert not chk.built  # lazy: nothing is built until the first check
    return chk


def test_defaults_and_step_count():
    assert (DEFAULT_SWEEP_INFLATION_M, DEFAULT_SWEEP_STEP_M) == (0.025, 0.005)  # D4
    positions = sweep_positions()
    assert len(positions) == 131 and positions[0] == 0.0 and positions[-1] == pytest.approx(0.65)
    assert positions[1] == pytest.approx(0.005)
    chk = RailSweepChecker(HW, "mavis_v2", inflation_m=0.01, step_m=0.01, rail_flip=True)
    assert (chk.inflation_m, chk.step_m, chk.rail_flip, chk.travel_m) == (0.01, 0.01, True, 0.65)


def test_keyframe_posture_is_clear_and_assumptions_name_the_unknown_rail(checker):
    samples = {"grip": _sample("grip", seq=7), "view": _sample("view", seq=9)}
    t0 = time.perf_counter()
    v = checker.check("grip", samples, FALLBACK)
    elapsed = time.perf_counter() - t0  # includes the one-off twin build
    assert isinstance(v, RailSweepVerdict) and checker.built
    assert (v.scene_id, v.inflation_m, v.step_m, v.travel_m) == ("mavis_v2", 0.025, 0.005, 0.65)
    assert v.clear and v.first_blocked_m is None and v.first_blocked_pair == []
    assert v.q_checked == pytest.approx(list(KEYFRAME_Q)) and v.sample_seq == 7
    # The Perception Arm's rail is unknown (unhomed) -> the fallback 0.00 m is assumed,
    # recorded, and echoed in other_arms (q7 + rail, twin coordinates).
    assert v.assumptions == ["view rail unknown - used fallback 0.00 m"]
    assert list(v.other_arms) == ["view"]
    assert v.other_arms["view"] == pytest.approx(list(KEYFRAME_Q) + [0.0])
    # The tightest pair over the whole travel is reported even when clear (the mic
    # tip hangs 3.8 cm over the table at the keyframe; 03-sim §4.3).
    assert v.min_clearance_m is not None and 0.03 < v.min_clearance_m < 0.05
    assert v.min_clearance_pair == ["table", "view_microphone"]
    assert v.min_clearance_at_m is not None
    assert elapsed < 30.0  # build-dominated; the sweep itself is tens of ms (below)
    t0 = time.perf_counter()
    checker.check("grip", samples, FALLBACK)
    assert time.perf_counter() - t0 < 2.0  # 131 checks + 310 pair distances each
    assert checker.sweeps == 2
    text = describe_verdict(v, dry_run=True)
    assert text.startswith("rail sweep clear") and text.endswith("(dry run, nothing written)")
    assert "table / view_microphone" in text
    assert not describe_verdict(v, dry_run=False).endswith("(dry run, nothing written)")


def test_tool_below_the_table_blocks_on_a_table_pair(checker):
    samples = {
        "grip": _sample("grip", TABLE_DIVE_Q, seq=3),
        "view": _sample("view", rail_pos_m=0.0, homed=True),
    }
    v = checker.check("grip", samples, FALLBACK)
    assert not v.clear
    assert v.first_blocked_m == 0.0  # blocked from the very first step
    assert "table" in v.first_blocked_pair and len(v.first_blocked_pair) == 2
    assert any(p.startswith("grip_") for p in v.first_blocked_pair)
    assert v.min_clearance_m is not None and v.min_clearance_m < 0.025
    assert v.assumptions == []  # the other arm's rail is known (homed + enabled)
    assert v.other_arms["view"][7] == 0.0
    text = describe_verdict(v, dry_run=False)
    assert text.startswith("home_rail refused: rail sweep blocked at 0.000 m (")
    assert "table" in text and text.endswith("nothing was written")


def test_other_arm_in_the_channel_blocks_the_sweep_mid_travel(checker):
    """The Perception Arm parked in the channel at rail 0.30 m with its camera
    lowered: the Manipulation Arm's carriage sweep must hit it somewhere along
    the travel (a cross-arm pair, not the table)."""
    view = _sample("view", (PI, 1.0, 0.0, 1.2, 0.0, 0.0, 0.0), rail_pos_m=0.3, homed=True)
    v = checker.check("grip", {"grip": _sample("grip"), "view": view}, FALLBACK)
    assert not v.clear and v.first_blocked_m is not None
    assert any(p.startswith(("grip_", "view_")) for p in v.first_blocked_pair)
    assert v.other_arms["view"][7] == pytest.approx(0.3)


def test_rail_flip_mirrors_the_other_arm_and_records_it():
    chk = RailSweepChecker(HW, "mavis_v2", rail_flip=True)
    view = _sample("view", rail_pos_m=0.1, homed=True)
    v = chk.check("grip", {"grip": _sample("grip"), "view": view}, FALLBACK)
    assert v.other_arms["view"][7] == pytest.approx(0.55)  # 0.65 - 0.1
    assert any("rail_flip" in a for a in v.assumptions)


def test_missing_sample_and_unknown_arm_are_refused(checker):
    with pytest.raises(ValueError, match="no monitor sample"):
        checker.check("grip", {"view": _sample("view")}, FALLBACK)
    with pytest.raises(KeyError):
        checker.check("arm9", {"arm9": _sample("arm9")}, FALLBACK)
    # An unsampled OTHER arm keeps the keyframe posture and says so.
    v = checker.check("grip", {"grip": _sample("grip")}, FALLBACK)
    assert v.clear and v.assumptions == ["view: no monitor sample - keyframe posture assumed"]
    assert v.other_arms["view"] == pytest.approx(list(KEYFRAME_Q) + [0.0])


# -- phase-09d: candidates, rail-locked planning and the position-agnostic path check ----------
REACH_SIDE_Q = (PI / 2, 0.9, 0.0, 0.9, 0.0, 0.0, 0.0)  # gripper meets the rail base near 0 m


def test_candidate_postures_are_the_keyframe_then_the_home_key(checker):
    cands = checker.candidate_postures("grip")
    assert [src for src, _ in cands] == ["keyframe", "home"]
    assert cands[0][1] == pytest.approx(list(KEYFRAME_Q))
    assert cands[1][1] == pytest.approx([0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0], abs=1e-4)
    assert [src for src, _ in checker.candidate_postures("view")] == ["keyframe", "home"]
    with pytest.raises(KeyError):
        checker.candidate_postures("arm9")


def test_densify_path_steps_and_indices():
    from apollo_mavis_v2_runtime.devices.rail_sweep import densify_path

    a = [0.0] * 7 + [0.65]
    b = [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.65]
    c = [0.0, 0.2, 0.0, 0.11, 0.0, 0.0, 0.0, 0.65]
    cfgs = densify_path([a, b, c], 0.05)
    assert [i for i, _ in cfgs] == [0, 0, 0, 0, 1, 1, 1, 2]  # 4 steps, 3 steps, the goal
    assert cfgs[0][1] == pytest.approx(a) and cfgs[-1][1] == pytest.approx(c)
    assert cfgs[1][1][1] == pytest.approx(0.05)
    assert densify_path([a], 0.05) == [(0, pytest.approx(a))] or len(densify_path([a])) == 1
    assert densify_path([]) == []


def test_plan_path_locks_the_rail_slot_and_restores_the_twin(checker):
    samples = {"grip": _sample("grip", TABLE_DIVE_Q, seq=3), "view": _sample("view")}
    before = checker.twin.data.qpos.copy()
    res = checker.plan_path("grip", TABLE_DIVE_Q, KEYFRAME_Q, samples, FALLBACK)
    assert res.ok and res.failure is None
    wps = res.waypoints["grip"]
    assert len(wps) >= 2 and all(len(w) == 8 for w in wps)
    assert wps[0][:7] == pytest.approx(list(TABLE_DIVE_Q))
    assert wps[-1][:7] == pytest.approx(list(KEYFRAME_Q))
    assert {round(w[7], 6) for w in wps} == {0.65}  # every waypoint at the fallback: no rail move
    assert checker.twin.data.qpos == pytest.approx(before)  # twin restored (keyframe)
    # the other arm's fallback / rail_flip are honoured like in check(); a deep collision
    # at the start refuses (start_in_collision) instead of planning through it
    deep = checker.plan_path(
        "grip", (PI + PI / 2, 1.0, 0.0, 1.2, 0.0, 0.2, 0.0), KEYFRAME_Q, samples, FALLBACK
    )
    assert not deep.ok and deep.failure == "start_in_collision"
    with pytest.raises(ValueError):
        checker.plan_path("grip", (0.0,) * 6, KEYFRAME_Q, samples, FALLBACK)


def test_check_path_is_position_agnostic_with_start_hysteresis(checker):
    from apollo_mavis_v2_runtime.devices.rail_sweep import PathVerdict

    samples = {"grip": _sample("grip", TABLE_DIVE_Q, seq=3), "view": _sample("view")}
    # a single sweep-clear configuration: 1 config x 131 positions
    v = checker.check_path("grip", [list(KEYFRAME_Q) + [0.65]], samples, FALLBACK)
    assert isinstance(v, PathVerdict) and v.clear
    assert (v.waypoints, v.checked_configs, v.checked_rail_positions) == (1, 1, 131)
    assert v.bad_waypoint is None and v.bad_rail_m is None and v.bad_pair == []
    assert v.detail.startswith("path clear: 1 configurations x 131 rail positions")
    assert v.assumptions == ["view rail unknown - used fallback 0.00 m"]
    # table-dive -> keyframe: the table pair the start posture violates at rail 0 m only
    # opens up along the straight joint path -> clear for EVERY rail position
    wps = checker.plan_path("grip", TABLE_DIVE_Q, KEYFRAME_Q, samples, FALLBACK).waypoints["grip"]
    v = checker.check_path("grip", wps, samples, FALLBACK)
    assert v.clear and v.checked_configs > 2 and v.checked_rail_positions == 131
    # reach-side -> keyframe: at rail 0 m the gripper base is already INSIDE the table's
    # inflated shell (2.7 cm of overlap - the planner never saw it, its rail slot is
    # locked at the 0.65 m fallback) -> a whitelisted pair in contact is refused at
    # waypoint 0 itself, rail 0.000 m, that pair
    side = {"grip": _sample("grip", REACH_SIDE_Q, seq=4), "view": _sample("view")}
    wps = checker.plan_path("grip", REACH_SIDE_Q, KEYFRAME_Q, side, FALLBACK).waypoints["grip"]
    v = checker.check_path("grip", wps, side, FALLBACK)
    assert not v.clear and v.bad_waypoint == 0 and v.bad_rail_m == 0.0
    assert v.bad_pair == ["grip_xarm_gripper_base_link", "table"]
    assert "in contact" in v.detail and v.detail.startswith("path blocked after waypoint 0")
    # a NEW violation (a configuration the start never had) blocks too
    v = checker.check_path(
        "grip", [list(KEYFRAME_Q) + [0.65], list(TABLE_DIVE_Q) + [0.65]], samples, FALLBACK
    )
    assert not v.clear and "new violation" in v.detail and "table" in v.bad_pair
    with pytest.raises(ValueError):
        checker.check_path("grip", [], samples, FALLBACK)


# -- the start-posture hysteresis must never ratchet DOWN (review fix) -------------------------
class _CreepTwin:
    """Scripted stand-in for the sweep twin: one arm, 8 qpos slots, a single
    monitored pair whose distance is ``dist_fn(q)`` (only at rail 0 m; every other
    rail position is clear). Lets a test drive ``check_path`` through an exact
    distance profile no real posture could reproduce deterministically."""

    class _Addr:
        qpos_adr = list(range(8))
        has_rail = True

    class _Addressing:
        def __init__(self, addr):
            self.arms = {"grip": addr}

        def __getitem__(self, key):
            return self.arms[key]

    def __init__(self, dist_fn, inflation_m: float = DEFAULT_SWEEP_INFLATION_M) -> None:
        from types import SimpleNamespace

        import numpy as np

        self.addr = self._Addressing(self._Addr())
        self.data = SimpleNamespace(qpos=np.zeros(8))
        self.model = object()
        self.monitored_pairs = []
        self.inflation_m = inflation_m
        self._dist_fn = dist_fn
        self.calls = 0

    def check_config_violations(self, q):
        self.calls += 1
        if q[7] > 1e-9:
            return []
        d = float(self._dist_fn(q))
        return [(("grip_link6", "table"), d)] if d < self.inflation_m else []


def _creep_checker(dist_fn) -> RailSweepChecker:
    from types import SimpleNamespace

    chk = RailSweepChecker(HW, "mavis_v2", step_m=0.65)  # 2 rail positions: 0 and 0.65
    chk._twin = _CreepTwin(dist_fn)
    chk._mujoco = SimpleNamespace(mj_kinematics=lambda m, d: None)
    return chk


_CREEP_SAMPLES = {"grip": _sample("grip", (0.0,) * 7)}
_START = [0.0] * 8
_GOAL = [1.0] + [0.0] * 7  # one 1 rad segment on joint 1 -> 20 densified steps of 0.05 rad


def test_check_path_blocks_a_slow_creep_of_a_whitelisted_pair():
    """A pair the start posture already violates (20 mm at 25 mm inflation) that
    gets 1 mm closer per densified configuration - always less than the 2 mm
    tolerance per step - must block: the reference is the START distance (only
    ever opening up), never the previous step's, so the creep cannot accumulate
    its way to contact."""
    chk = _creep_checker(lambda q: 0.020 - 0.020 * float(q[0]))  # 20 mm -> 0 mm (contact)
    v = chk.check_path("grip", [_START, _GOAL], _CREEP_SAMPLES, {"grip": 0.0}, max_step_rad=0.05)
    assert not v.clear and "gets closer" in v.detail and v.bad_rail_m == 0.0
    assert v.bad_waypoint == 0 and v.bad_pair == ["grip_link6", "table"]
    # blocked as soon as the pair is > 2 mm below the start reference (3rd step, 17 mm)
    assert "2.0 -> 1.7 cm" in v.detail
    # the same profile with a big enough single step is the pre-fix behaviour's blind
    # spot no more: any step crossing the tolerance blocks too
    chk2 = _creep_checker(lambda q: 0.020 - 0.030 * float(q[0]))  # 1.5 mm/step into penetration
    v2 = chk2.check_path("grip", [_START, _GOAL], _CREEP_SAMPLES, {"grip": 0.0}, max_step_rad=0.05)
    assert not v2.clear and "gets closer" in v2.detail


def test_check_path_reference_only_opens_up_and_contact_blocks():
    """A whitelisted pair that first opens up (10 -> 20 mm) raises its reference;
    falling back below it minus the tolerance blocks even though it never gets
    closer than at the start. A whitelisted pair that reaches contact (dist <= 0)
    blocks outright, even inside the tolerance."""
    import math

    # opens 10 -> 20 mm over the first half, then drops to 17 mm: 17 < 20 - 2 -> blocked
    def open_then_dip(q):
        x = float(q[0])
        return 0.010 + 0.020 * x if x <= 0.5 else 0.020 - 0.006 * (x - 0.5) / 0.5

    v = _creep_checker(open_then_dip).check_path(
        "grip", [_START, _GOAL], _CREEP_SAMPLES, {"grip": 0.0}, max_step_rad=0.05
    )
    assert not v.clear and "gets closer" in v.detail and "2.0 ->" in v.detail
    # the same opening WITHOUT the dip (10 -> 20 mm, still violating at 25 mm) is clear
    v = _creep_checker(lambda q: 0.010 + 0.010 * float(q[0])).check_path(
        "grip", [_START, _GOAL], _CREEP_SAMPLES, {"grip": 0.0}, max_step_rad=0.05
    )
    assert v.clear and v.checked_configs == 21 and v.checked_rail_positions == 2
    # contact: 1 mm at the start, 0 mm from the first step (within the 2 mm tolerance)
    v = _creep_checker(lambda q: 0.001 if float(q[0]) < 1e-9 else 0.0).check_path(
        "grip", [_START, _GOAL], _CREEP_SAMPLES, {"grip": 0.0}, max_step_rad=0.05
    )
    assert not v.clear and "in contact" in v.detail and v.bad_waypoint == 0
    # a start posture already in contact is refused at waypoint 0 itself
    v = _creep_checker(lambda q: -0.001).check_path(
        "grip", [_START, _GOAL], _CREEP_SAMPLES, {"grip": 0.0}, max_step_rad=0.05
    )
    assert not v.clear and "in contact" in v.detail and math.isclose(v.bad_rail_m, 0.0)
