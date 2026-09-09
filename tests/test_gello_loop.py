"""``GelloLoop`` tick tests over ``FakeWorkcell`` (16-gello §6; phase-15): the Manipulation Arm
streams the leader only while ``tracking`` and at the cap, ←/→ move ONLY its rail slot, the
translate / rotate / gripper keys do nothing to it, the trigger gripper is quantised and sent
on the policy path's cadence, ``gello_pause`` / ``gello_resume`` are idempotent and sticky, a
driver fault forces ``paused``, ``switch_arm`` / takeover / the episode ops / ``joint_target``
are nacked with the mode reasons, the manager's ``gello_motion`` window reads ``motion`` and
engages on release, the Perception Arm follows a fake viewpoint source (view block only) and
holds otherwise, and ``session_extra["gello"]`` carries the session half of the telemetry."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from apollo_mavis_v2_core import Command, CommandSource, ProfileStore
from apollo_mavis_v2_core.interfaces.policy import PolicyOutput
from apollo_mavis_v2_core.testing import FakeArm, FakeWorkcell
from conftest import run_ticks
from fakes import EventFakeWorkcell

from apollo_mavis_v2_runtime.bus import RuntimeBus
from apollo_mavis_v2_runtime.config import ControlConfig, GelloConfig
from apollo_mavis_v2_runtime.control.loop import GRIPPER_SEND_EVERY_N_TICKS
from apollo_mavis_v2_runtime.devices.gello import GelloSample
from apollo_mavis_v2_runtime.gello.loop import (
    GELLO_NO_ARM_SWITCH,
    GELLO_NO_JOINT_PANEL,
    GELLO_NO_RECORDING,
    GELLO_NO_TAKEOVER,
    RESUME_DURING_MOTION,
    GelloLoop,
)
from apollo_mavis_v2_runtime.gello.viewpoint import PAUSED_DETAIL, ViewpointSource
from apollo_mavis_v2_runtime.safety.gate import NullGate
from apollo_mavis_v2_runtime.safety.supervisor import SafetySupervisor
from apollo_mavis_v2_runtime.safety.watchdog import InputWatchdog

PI = math.pi
GRIP_Q0 = np.array([PI, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.65])
VIEW_Q0 = np.array([2.646, -1.598, 0.018, 1.637, 0.25, 2.007, 0.029, 0.0])


class FakeReader:
    """The ``GelloReader`` surface the loop reads: ``latest()``."""

    def __init__(self) -> None:
        self.sample: GelloSample | None = None
        self.seq = 0

    def set(self, q, gripper_frac=1.0, *, t: float, valid: bool = True) -> GelloSample:
        self.seq += 1
        self.sample = GelloSample(
            q_raw=np.asarray(q, dtype=np.float64),
            q=np.asarray(q, dtype=np.float64),
            gripper_frac=gripper_frac,
            rx_mono=t,
            seq=self.seq,
            valid=valid,
        )
        return self.sample

    def latest(self) -> GelloSample | None:
        return self.sample


class FakeKin:
    """``kin`` seam for the ActionAnchor: identity-ish FK over the 7 joints."""

    def tcp_world(self, arm_id, q):
        from apollo_mavis_v2_core import Pose

        q = np.asarray(q, dtype=np.float64)
        return Pose(np.array([q[0] * 0.1, q[1] * 0.1, q[2] * 0.1]), np.array([1.0, 0, 0, 0]))

    def base_quat_world(self, arm_id):
        return np.array([1.0, 0.0, 0.0, 0.0])


class FakeIK:
    """IK that moves joint 1..3 by the requested delta position (x, y, z) * 10."""

    def solve(self, arm_id, target, q_last):
        q = np.array(q_last, dtype=np.float64)
        q[0] = target.position[0] * 10.0
        q[1] = target.position[1] * 10.0
        q[2] = target.position[2] * 10.0
        return SimpleNamespace(q=q, diverged=False, pos_err_m=0.0, rot_err_rad=0.0)

    def reset(self, arm_id, q):
        pass

    def sync_passive(self, states):
        pass


class FakeVpSource:
    """A live ``ExternalPolicySource`` stand-in: a constant view-block row."""

    def __init__(self, row, period=1.0 / 15.0) -> None:
        self.row = np.asarray(row, dtype=np.float32)
        self.period = period
        self.paused = False
        self.started = 0
        self.stopped = 0
        self.t = 0.0

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    def latest(self):
        return PolicyOutput(actions=self.row, version=1, t_mono=self.t, chunk_remaining=0), self.t

    def staleness_scale(self, now):
        return 1.0

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def drop_and_requery(self, reason="handback"):
        pass

    def current_version(self):
        return 1


def _cfg(**over) -> GelloConfig:
    return GelloConfig(backend="fake", **over)


def make_loop(tmp_path, *, cell=None, viewpoint=None, anchor=None, launch_pending=False, gcfg=None):
    cell = cell or FakeWorkcell(
        {
            "grip": FakeArm("grip", has_rail=True, q0=GRIP_Q0.copy(), max_joint_speed_rad_s=5.0),
            "view": FakeArm("view", has_rail=True, q0=VIEW_Q0.copy(), max_joint_speed_rad_s=5.0),
        }
    )
    cell.start()
    bus = RuntimeBus()
    supervisor = SafetySupervisor(NullGate(), InputWatchdog())
    reader = FakeReader()
    gcfg = gcfg or _cfg()
    loop = GelloLoop(
        cell,
        ControlConfig(),
        bus,
        supervisor,
        ["view", "grip"],
        gello_cfg=gcfg,
        reader=reader,
        viewpoint=viewpoint,
        anchor=anchor,
        launch_pending=launch_pending,
        profile_store=ProfileStore(tmp_path / "profiles"),
        workcell_kind="sim",
        gripper_arms=["grip"],
        speed_scale=1.0,
    )
    return cell, bus, loop, reader


def keys(bus, held: list[str], t: float) -> None:
    from apollo_mavis_v2_core import HeldState

    bus.held_keys.put(HeldState(held=frozenset(held), seq=int(t * 1000) + 1, rx_mono=t))


def ack(loop, bus, op: str, **args):
    """Submit one action and run ONE deterministic tick (the loop's own clock continues:
    a wall-clock tick here would read as a process stall and hold every arm)."""
    fut = bus.commands.submit(Command(op=op, args=args, source="ws"))
    last = loop._last_tick_now
    loop.run_tick(last + loop.dt if last is not None else 0.01)
    return fut.result(timeout=1.0)


def q_of(cell, arm) -> np.ndarray:
    return cell.arms[arm].get_state().q


# -- construction ---------------------------------------------------------------------------------
def test_active_arm_pinned_no_tracker_and_sim_cap_lowered(tmp_path):
    cell, bus, loop, reader = make_loop(tmp_path)
    assert loop.active_arm == "grip" and loop.tracker is None
    # 16-gello §6.2: 0.6 rad/s -> 0.006 rad/tick at 100 % on the SIM loop - and the
    # PlanExecutor's slew with it (2026-09-09 review: the executor was left at 0.02 and cut
    # every waypoint corner); the rail slew is untouched (no driver track speed in sim)
    assert loop.cfg.dq_max_rad == pytest.approx(0.006)
    assert loop.cfg.jog.slew_rad_per_tick == pytest.approx(0.006)
    assert loop.plans.cfg is loop.cfg.jog and loop.jog.cfg is loop.cfg.jog
    assert loop.cfg.jog.rail_m_per_tick == ControlConfig().jog.rail_m_per_tick
    assert loop.engage.state == "no_leader"
    snap = loop.run_tick(0.01)
    g = snap.session_extra["gello"]
    assert g["state"] == "no_leader" and g["engaged_arm"] is None and g["lag_rad"] is None
    assert g["viewpoint"] is None and "no leader sample yet" in g["state_detail"]
    assert g["paused_latched"] is False


def test_sim_speed_scale_lowers_both_caps_and_a_hardware_loop_is_left_alone(tmp_path):
    from apollo_mavis_v2_runtime.gello.loop import sim_gello_caps

    cfg = ControlConfig()
    half = sim_gello_caps(cfg, _cfg(), 0.5)
    assert half.dq_max_rad == pytest.approx(0.003) and half.jog.slew_rad_per_tick == 0.003
    # a cap already below the GELLO value is kept (min, never raised)
    low_jog = cfg.jog.model_copy(update={"slew_rad_per_tick": 0.001})
    low = ControlConfig(dq_max_rad=0.002, jog=low_jog)
    assert sim_gello_caps(low, _cfg(), 1.0) is low
    # the constructor applies the pairing from the speed_scale kwarg
    cell, bus, loop, reader = make_loop(tmp_path)
    assert loop.speed_scale == 1.0 and loop.plans.cfg.slew_rad_per_tick == pytest.approx(0.006)


def test_sim_planned_motion_reaches_every_waypoint_before_the_next_segment(tmp_path):
    """2026-09-09 review: a two-waypoint follower plan (w1 = +0.018 on joint 2, w2 = +0.018
    on joint 3) must walk joint 2 to w1 EXACTLY (three 0.006 steps) before joint 3 moves -
    the executor's slew and the loop's cap agree, so no planned step is ever clamped and the
    commanded path never leaves the validated polyline."""
    cell, bus, loop, reader = make_loop(tmp_path)
    loop.run_tick(0.01)
    w1 = GRIP_Q0.copy()
    w1[1] += 0.018
    w2 = w1.copy()
    w2[2] += 0.018
    res = ack(loop, bus, "execute_plan", waypoints={"grip": [w1.tolist(), w2.tolist()]})
    assert res.ok and res.detail == "executing"
    trace = [loop._last_cmd["grip"].copy()]
    t = loop._last_tick_now
    while loop.plans.active("grip") and len(trace) < 40:
        t = run_ticks(loop, cell, 1, t)
        trace.append(loop._last_cmd["grip"].copy())
    assert not loop.plans.active("grip")
    steps = [b[1] - a[1] for a, b in zip(trace, trace[1:], strict=False)]
    at_w1 = [i for i, q in enumerate(trace) if np.allclose(q, w1, atol=1e-9)]
    assert at_w1, trace  # the intermediate waypoint IS reached (not cut)
    i = at_w1[0]
    assert i == 2  # 0.018 / 0.006 = three equal steps (the ack's tick was the first)
    assert all(q[2] == GRIP_Q0[2] for q in trace[: i + 1])  # joint 3 waits for the corner
    assert np.allclose(trace[-1], w2, atol=1e-9)
    assert max(abs(x) for x in steps) <= 0.006 + 1e-9
    assert loop.clamp_ticks == 0  # the cap never had to bind a planned step
    assert loop.engage.state == "motion"  # the executor owned the arm on the last tick
    t = run_ticks(loop, cell, 2, t)
    assert loop.engage.state == "no_leader"  # no leader: nothing else moves the arm
    assert np.allclose(loop._last_cmd["grip"], w2, atol=1e-9)


def test_leader_within_tolerance_engages_and_the_follower_streams_at_the_cap(tmp_path):
    cell, bus, loop, reader = make_loop(tmp_path)
    t = 0.0
    reader.set(GRIP_Q0[:7], 1.0, t=0.0)
    t = run_ticks(loop, cell, 2, t)
    assert loop.engage.state == "tracking"
    assert loop._arm_source["grip"] is CommandSource.GELLO
    # move the leader 0.3 rad on joint 2: the follower walks at dq_max per tick, never jumps
    target = GRIP_Q0[:7].copy()
    target[1] += 0.3
    reader.set(target, 1.0, t=t)
    before = loop._last_cmd["grip"].copy()
    t = run_ticks(loop, cell, 1, t)
    step = loop._last_cmd["grip"][1] - before[1]
    assert step == pytest.approx(loop.cfg.dq_max_rad, rel=1e-6)
    assert loop.clamp_ticks >= 1
    for _ in range(80):
        reader.set(target, 1.0, t=t)  # keep the sample fresh (stale_s 0.2)
        t = run_ticks(loop, cell, 1, t)
    assert loop._last_cmd["grip"][1] == pytest.approx(target[1], abs=1e-6)
    assert np.allclose(loop._last_cmd["grip"][[0, 2, 3, 4, 5, 6]], GRIP_Q0[[0, 2, 3, 4, 5, 6]])
    assert loop._last_cmd["grip"][7] == pytest.approx(0.65)  # rail untouched


def test_stale_sample_holds_and_out_of_sync_needs_the_engage_rule(tmp_path):
    cell, bus, loop, reader = make_loop(tmp_path)
    far = GRIP_Q0[:7].copy()
    far[2] += 0.5
    reader.set(far, 1.0, t=0.0)
    t = run_ticks(loop, cell, 2, 0.0)
    assert loop.engage.state == "out_of_sync" and "joint 3" in loop.engage.detail
    assert np.allclose(loop._last_cmd["grip"], GRIP_Q0)  # holds: never starts from far away
    reader.set(GRIP_Q0[:7], 1.0, t=t)
    t = run_ticks(loop, cell, 1, t)
    assert loop.engage.state == "tracking"
    t = run_ticks(loop, cell, 40, t)  # no new sample: 0.4 s > stale_s -> no_leader, hold
    assert loop.engage.state == "no_leader" and "stale" in loop.engage.detail
    assert np.allclose(loop._last_cmd["grip"], GRIP_Q0)


# -- keys ----------------------------------------------------------------------------------------
def test_rail_keys_move_only_the_rail_slot_while_tracking_and_while_not(tmp_path):
    cell, bus, loop, reader = make_loop(tmp_path)
    t = 0.0
    reader.set(GRIP_Q0[:7], 1.0, t=t)
    t = run_ticks(loop, cell, 2, t)
    assert loop.engage.state == "tracking"
    for _ in range(20):
        keys(bus, ["ArrowLeft"], t)
        reader.set(GRIP_Q0[:7], 1.0, t=t)
        t = run_ticks(loop, cell, 1, t)
    q = loop._last_cmd["grip"]
    assert q[7] < 0.65 - 0.01 and np.allclose(q[:7], GRIP_Q0[:7])
    rail_tracking = q[7]
    # not tracking (leader stale) the rail keys STILL move the carriage (D9: every source)
    for _ in range(40):
        keys(bus, ["ArrowLeft"], t)
        t = run_ticks(loop, cell, 1, t)
    assert loop.engage.state == "no_leader"
    assert loop._last_cmd["grip"][7] < rail_tracking - 0.01
    assert np.allclose(loop._last_cmd["grip"][:7], GRIP_Q0[:7])
    # the Perception Arm never moves on a rail key (the follower's rail only)
    assert np.allclose(loop._last_cmd["view"], VIEW_Q0)


def test_translate_rotate_gripper_and_clutch_keys_do_nothing_to_the_manipulation_arm(tmp_path):
    cell, bus, loop, reader = make_loop(tmp_path)
    t = 0.0
    reader.set(GRIP_Q0[:7], 1.0, t=t)
    t = run_ticks(loop, cell, 2, t)
    assert loop.engage.state == "tracking"
    for code in ("KeyW", "KeyA", "KeyE", "KeyI", "KeyJ", "KeyU", "KeyF", "KeyH", "KeyC"):
        for _ in range(10):
            keys(bus, [code], t)
            reader.set(GRIP_Q0[:7], 1.0, t=t)
            t = run_ticks(loop, cell, 1, t)
    assert np.allclose(loop._last_cmd["grip"], GRIP_Q0)
    assert np.allclose(loop._last_cmd["view"], VIEW_Q0)
    assert loop._grip_frac["grip"] == 1.0  # F/H ignored: the trigger owns the gripper
    assert loop.ik_slips == 0 and loop.integrator.get("grip") is None  # no teleop seed


# -- gripper -------------------------------------------------------------------------------------
def test_gripper_follows_the_trigger_quantised_on_the_send_cadence(tmp_path):
    cell, bus, loop, reader = make_loop(tmp_path)
    loop.start()  # ArmSender threads: put_gripper reaches the fake arm
    try:
        t = 0.0
        reader.set(GRIP_Q0[:7], 1.0, t=t)
        for _ in range(3):
            t += loop.dt
            loop.run_tick(t)
        assert loop.engage.state == "tracking"
        reader.set(GRIP_Q0[:7], 0.4237, t=t)
        for _ in range(2 * GRIPPER_SEND_EVERY_N_TICKS + 1):
            t += loop.dt
            reader.set(GRIP_Q0[:7], 0.4237, t=t)
            loop.run_tick(t)
        assert loop._grip_frac["grip"] == pytest.approx(0.42)  # quantum 0.01
        assert loop.gripper_sends == 1  # once per change, never per tick
        import time as _time

        arm = cell.arms["grip"]

        def wait_cmds(n: int) -> None:
            deadline = _time.monotonic() + 2.0
            while _time.monotonic() < deadline and len(arm.gripper_commands) < n:
                _time.sleep(0.01)
            assert len(arm.gripper_commands) >= n, arm.gripper_commands

        wait_cmds(1)
        assert arm.gripper_commands[-1].open_frac == pytest.approx(0.42)
        # no gripper channel: nothing changes
        reader.set(GRIP_Q0[:7], None, t=t)
        for _ in range(GRIPPER_SEND_EVERY_N_TICKS + 1):
            t += loop.dt
            reader.set(GRIP_Q0[:7], None, t=t)
            loop.run_tick(t)
        assert loop._grip_frac["grip"] == pytest.approx(0.42) and loop.gripper_sends == 1
        # 2026-09-09 review: a profile gripper applied on arrival (Go to profile / R) closes
        # the real gripper; the leader's UNCHANGED trigger must be re-sent on the next
        # cadence tick - the comparison is against the value in force, not a private copy
        loop._apply_gripper_target("grip", 0.0)
        wait_cmds(2)
        assert arm.gripper_commands[-1].open_frac == 0.0 and loop._grip_frac["grip"] == 0.0
        for _ in range(GRIPPER_SEND_EVERY_N_TICKS + 1):
            t += loop.dt
            reader.set(GRIP_Q0[:7], 0.4237, t=t)
            loop.run_tick(t)
        assert loop.gripper_sends == 2 and loop._grip_frac["grip"] == pytest.approx(0.42)
        wait_cmds(3)
        assert arm.gripper_commands[-1].open_frac == pytest.approx(0.42)
        # ... and after a recovery re-seed: the MEASURED opening becomes the value in force
        # (here the gripper was moved behind the loop's back) and the leader is re-sent
        # against it on the next cadence tick (the ArmSender itself still drops a value equal
        # to the last one IT dispatched, so the leader is moved to 0.9 for the arm-side check)
        arm.command_gripper(type(arm.gripper_commands[-1])(open_frac=0.75))
        loop.reseed_arm("grip")
        assert loop._grip_frac["grip"] == pytest.approx(0.75)
        for _ in range(GRIPPER_SEND_EVERY_N_TICKS + 1):
            t += loop.dt
            reader.set(GRIP_Q0[:7], 0.9, t=t)
            loop.run_tick(t)
        assert loop.gripper_sends == 3 and loop._grip_frac["grip"] == pytest.approx(0.9)
        wait_cmds(5)
        assert arm.gripper_commands[-1].open_frac == pytest.approx(0.9)
    finally:
        loop.stop()


# -- pause / resume / fault ------------------------------------------------------------------------
def test_pause_and_resume_are_idempotent_and_the_pause_is_sticky(tmp_path):
    cell, bus, loop, reader = make_loop(tmp_path)
    t = 0.0
    reader.set(GRIP_Q0[:7], 1.0, t=t)
    t = run_ticks(loop, cell, 2, t)
    assert loop.engage.state == "tracking"
    res = ack(loop, bus, "gello_pause")
    assert (res.ok, res.detail) == (True, "paused")
    assert ack(loop, bus, "gello_pause").detail == "already paused"
    assert loop.engage.state == "paused"
    moved = GRIP_Q0[:7].copy()
    moved[1] += 0.05  # within tolerance, but paused: the follower does NOT move
    for _ in range(30):
        reader.set(moved, 1.0, t=t)
        t = run_ticks(loop, cell, 1, t)
    assert loop.engage.state == "paused" and np.allclose(loop._last_cmd["grip"], GRIP_Q0)
    snap = loop.run_tick(t + loop.dt)
    assert snap.session_extra["gello"]["state"] == "paused"
    assert snap.session_extra["gello"]["paused_latched"] is True
    assert snap.session_extra["gello"]["lag_rad"][1] == pytest.approx(0.05, abs=1e-6)
    res = ack(loop, bus, "gello_resume")
    assert (res.ok, res.detail) == (True, "resumed")
    assert ack(loop, bus, "gello_resume").detail == "not paused"
    for _ in range(3):
        reader.set(moved, 1.0, t=t)
        t = run_ticks(loop, cell, 1, t)
    assert loop.engage.state == "tracking"
    assert loop._last_cmd["grip"][1] > GRIP_Q0[1]  # following again


def test_a_driver_fault_on_the_manipulation_arm_forces_paused_until_resume(tmp_path):
    cell = EventFakeWorkcell(
        {
            "grip": FakeArm("grip", has_rail=True, q0=GRIP_Q0.copy(), max_joint_speed_rad_s=5.0),
            "view": FakeArm("view", has_rail=True, q0=VIEW_Q0.copy(), max_joint_speed_rad_s=5.0),
        },
        kind="sim",
    )
    cell, bus, loop, reader = make_loop(tmp_path, cell=cell)
    t = 0.0
    reader.set(GRIP_Q0[:7], 1.0, t=t)
    t = run_ticks(loop, cell, 2, t)
    assert loop.engage.state == "tracking"
    cell.fault("grip", 24)
    reader.set(GRIP_Q0[:7], 1.0, t=t)
    t = run_ticks(loop, cell, 1, t)
    assert loop.engage.state == "paused" and "paused:" in loop.engage.detail
    assert "grip" in loop.faulted_arms
    # recovery: re-seed + RECOVERING; a paused GELLO is "no live input", so RECOVERING
    # ends on the first tick with every key up; the pause itself stays until Resume
    cell.request_recovery("grip")
    for _ in range(3):
        reader.set(GRIP_Q0[:7], 1.0, t=t)
        t = run_ticks(loop, cell, 1, t)
    assert not loop.faulted_arms and not loop.recovering_arms
    assert loop.engage.state == "paused"
    assert ack(loop, bus, "gello_resume").ok
    reader.set(GRIP_Q0[:7], 1.0, t=t)
    t = run_ticks(loop, cell, 2, t)
    assert loop.engage.state == "tracking"


# -- nacks ---------------------------------------------------------------------------------------
def test_mode_nacks(tmp_path):
    cell, bus, loop, reader = make_loop(tmp_path)
    loop.run_tick()
    assert ack(loop, bus, "switch_arm").detail == GELLO_NO_ARM_SWITCH
    assert ack(loop, bus, "switch_arm", arm_id="view").detail == GELLO_NO_ARM_SWITCH
    assert ack(loop, bus, "switch_arm_prev").detail == GELLO_NO_ARM_SWITCH
    assert loop.active_arm == "grip"
    for op in ("takeover_toggle", "takeover", "handback"):
        res = ack(loop, bus, op)
        assert (res.ok, res.detail) == (False, GELLO_NO_TAKEOVER), op
    assert ack(loop, bus, "train_now").detail == "not an Online DAgger session"
    for op in ("episode_new", "episode_save", "episode_discard"):
        assert ack(loop, bus, op).detail == GELLO_NO_RECORDING
    res = ack(loop, bus, "joint_target", arm_id="grip", positions=[0.0] * 8, mode="jog")
    assert (res.ok, res.detail) == (False, GELLO_NO_JOINT_PANEL)
    assert ack(loop, bus, "tracker_settings", yaw_deg=1.0).detail == (
        "no tracker provider in this session"
    )


# -- the motion window + R --------------------------------------------------------------------
def test_motion_window_reads_motion_and_engages_on_release(tmp_path):
    cell, bus, loop, reader = make_loop(tmp_path, launch_pending=True)
    t = 0.0
    reader.set(GRIP_Q0[:7], 1.0, t=t)
    t = run_ticks(loop, cell, 2, t)
    assert loop.engage.state == "motion"  # the launch window is open from the first tick
    moved = GRIP_Q0[:7].copy()
    moved[1] += 0.05
    for _ in range(5):
        reader.set(moved, 1.0, t=t)
        t = run_ticks(loop, cell, 1, t)
    assert np.allclose(loop._last_cmd["grip"], GRIP_Q0)  # nothing follows inside the window
    res = ack(loop, bus, "gello_motion", active=False, reason="launch over")
    assert res.ok
    reader.set(moved, 1.0, t=t)
    t = run_ticks(loop, cell, 1, t)
    assert loop.engage.state == "tracking"  # the engage rule ran once on release
    t = run_ticks(loop, cell, 1, t)
    assert loop._last_cmd["grip"][1] > GRIP_Q0[1]


def test_reset_to_initial_pauses_only_when_accepted_and_a_refusal_changes_nothing(tmp_path):
    """2026-09-09 review: the pause used to be forced BEFORE the base op validated and
    undone with ``resume()`` on a nack - which re-ran the engage rule against the MEASURED
    arm and dropped a lagging follower to out_of_sync. Now a refused R / Go to profile
    leaves the state, the unwrap branch and the transition count exactly as they were."""
    cell, bus, loop, reader = make_loop(tmp_path)
    t = 0.0
    reader.set(GRIP_Q0[:7], 1.0, t=t)
    t = run_ticks(loop, cell, 2, t)
    assert loop.engage.state == "tracking"
    # the follower lags the leader by 0.2 rad (inside the 0.8 leash, outside the 0.10
    # engage tolerance): a resume() here would have dropped it to out_of_sync
    lead = GRIP_Q0[:7].copy()
    lead[1] += 0.2
    reader.set(lead, 1.0, t=t)
    t = run_ticks(loop, cell, 1, t)
    assert loop.engage.state == "tracking" and loop.engage.max_lag_rad() > 0.15
    k_before = loop.engage._k.copy()
    transitions = loop.engage.transitions
    # no initial condition designated: the base op nacks and GELLO is left as it was
    res = ack(loop, bus, "reset_to_initial")
    assert not res.ok and "no initial condition designated" in res.detail
    assert loop.engage.state == "tracking" and not loop.engage.paused
    assert np.array_equal(loop.engage._k, k_before) and loop.engage.transitions == transitions
    for _ in range(3):
        reader.set(lead, 1.0, t=t)
        t = run_ticks(loop, cell, 1, t)
    assert loop.engage.state == "tracking"  # still following, no out_of_sync detour
    assert loop._last_cmd["grip"][1] > GRIP_Q0[1] + 0.01
    reader.set(GRIP_Q0[:7], 1.0, t=t)
    for _ in range(40):
        reader.set(loop._last_cmd["grip"][:7], 1.0, t=t)
        t = run_ticks(loop, cell, 1, t)
    # with a manager hook that accepts, the pause sticks (the manager's motion follows)
    from apollo_mavis_v2_core import ArmPosture, StateProfile

    prof = loop.profile_store.save(
        StateProfile(
            name="init",
            workcell_kind="sim",
            is_initial_condition=True,
            arms={a: ArmPosture(q=[0.1] * 7) for a in ("grip", "view")},
        )
    )
    loop.profile_store.set_initial(prof.profile_id)
    loop.on_reset_to_initial = lambda profile: (True, "returning")
    res = ack(loop, bus, "reset_to_initial")
    assert res.ok
    assert loop.engage.state == "paused" and "planned motion" in loop.engage.detail
    loop.on_goto_profile = lambda profile: (False, "a planned motion is already running")
    res = ack(loop, bus, "goto_profile", profile_id=prof.profile_id)
    assert not res.ok and loop.engage.state == "paused"  # an earlier pause is never undone
    assert "planned motion" in loop.engage.detail  # ... and keeps its own reason


def test_gello_resume_is_nacked_while_a_planned_motion_owns_the_arm(tmp_path):
    """2026-09-09 review: a Resume inside the manager's motion window (or an executor plan)
    used to clear the latch invisibly and let the follower engage the instant the return
    ended, against 16-gello §5.3 "paused afterwards". Pause still latches meanwhile."""
    cell, bus, loop, reader = make_loop(tmp_path)
    t = 0.0
    reader.set(GRIP_Q0[:7], 1.0, t=t)
    t = run_ticks(loop, cell, 2, t)
    assert loop.engage.state == "tracking"
    assert ack(loop, bus, "gello_motion", active=True, reason="R").ok
    assert loop.engage.state == "motion" and not loop.engage.paused
    res = ack(loop, bus, "gello_resume")
    assert (res.ok, res.detail) == (False, RESUME_DURING_MOTION)
    assert not loop.engage.paused  # a nack has no side effect
    res = ack(loop, bus, "gello_pause")  # accepted: it latches, the display stays `motion`
    assert (res.ok, res.detail) == (True, "paused")
    assert loop.engage.state == "motion" and loop.engage.paused
    snap = loop.run_tick(loop._last_tick_now + loop.dt)
    assert snap.session_extra["gello"]["state"] == "motion"
    assert snap.session_extra["gello"]["paused_latched"] is True
    res = ack(loop, bus, "gello_resume")
    assert (res.ok, res.detail) == (False, RESUME_DURING_MOTION)
    assert loop.engage.paused  # the latch survives the refused Resume
    assert ack(loop, bus, "gello_motion", active=False, reason="R over").ok
    t = loop._last_tick_now
    reader.set(GRIP_Q0[:7], 1.0, t=t)
    t = run_ticks(loop, cell, 2, t)
    assert loop.engage.state == "paused"  # the window closed paused, as §5.3 promises
    res = ack(loop, bus, "gello_resume")
    assert (res.ok, res.detail) == (True, "resumed")
    reader.set(GRIP_Q0[:7], 1.0, t=t)
    t = run_ticks(loop, cell, 2, t)
    assert loop.engage.state == "tracking"
    # an executor-owned plan on the follower is a motion window too
    w = GRIP_Q0.copy()
    w[1] += 0.03
    assert ack(loop, bus, "execute_plan", waypoints={"grip": [w.tolist()]}).ok
    assert loop.plans.active("grip")
    res = ack(loop, bus, "gello_resume")
    assert (res.ok, res.detail) == (False, RESUME_DURING_MOTION)


# -- the Perception Arm -----------------------------------------------------------------------
def _viewpoint(mode="auto", hub=None, src=None):
    vp = ViewpointSource(
        hub,
        None,
        session_id="s",
        view_frame="arm_base:view",
        mode=mode,
        cfg=SimpleNamespace(spec_stale_s=3.0, max_obs_age_s=0.5),
        source_factory=(lambda *a, **k: src) if src is not None else None,
    )
    return vp


class FakeHub:
    """``ExternalPolicyHub.spec(now)`` with a fresh compatible announce."""

    def __init__(self, ann):
        self.ann = ann
        self.bridge = SimpleNamespace(attached=True)

    def spec(self, now=None):
        return self.ann


def _announce(names, frame="arm_base:view"):
    from apollo_mavis_v2_core.protocol.external import PolicySpecAnnounce, PolicySpecModel

    return PolicySpecAnnounce(
        policy_id="viewer",
        policy_version=1,
        node_version="t",
        rate_hz=15.0,
        spec=PolicySpecModel(
            action_space="delta_ee",
            action_frame=frame,
            action_names=list(names),
            state_names=["view_joint1.pos"],
        ),
    )


def test_viewpoint_drives_the_perception_arm_only_and_hold_never_polls(tmp_path):
    from apollo_mavis_v2_runtime.dagger.policy_runner import ActionAnchor, SlewLimits
    from apollo_mavis_v2_runtime.gello.viewpoint import view_action_names

    row = [0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.002]  # +x delta per period, rail +2 mm
    src = FakeVpSource(row)
    hub = FakeHub(_announce(view_action_names(True)))
    vp = _viewpoint("auto", hub, src)
    anchor = ActionAnchor(FakeIK(), FakeKin(), SlewLimits(), action_space="delta_ee")
    cell, bus, loop, reader = make_loop(tmp_path, viewpoint=vp, anchor=anchor)
    t = 0.0
    reader.set(GRIP_Q0[:7], 1.0, t=t)
    t = run_ticks(loop, cell, 3, t)
    assert vp.attached and src.started == 1
    assert loop._arm_source["view"] is CommandSource.POLICY
    q_view = loop._last_cmd["view"]
    assert q_view[7] > VIEW_Q0[7]  # the rail delta landed on the view block
    assert q_view[0] != VIEW_Q0[0]  # the IK moved the view joints
    assert np.allclose(loop._last_cmd["grip"], GRIP_Q0)  # the follower is untouched
    snap = loop.run_tick(t + loop.dt)
    g = snap.session_extra["gello"]
    assert g["viewpoint"]["attached"] and g["viewpoint"]["policy_id"] == "viewer"
    assert g["viewpoint"]["mode"] == "auto"
    # GELLO drove the tick: the source is GELLO (the follower streams), not POLICY
    # (checked through the gate report's stamp: the supervisor filter got GELLO)
    assert loop._gello_drove
    # a planned motion window detaches the source for its duration
    t = loop._last_tick_now
    ack(loop, bus, "gello_motion", active=True, reason="R")
    t = run_ticks(loop, cell, 1, t + loop.dt)
    assert not vp.attached and src.stopped == 1
    ack(loop, bus, "gello_motion", active=False, reason="R over")
    reader.set(GRIP_Q0[:7], 1.0, t=t)
    run_ticks(loop, cell, 2, t + loop.dt)
    assert vp.attached and src.started == 2
    # incompatible layout (both arms) -> never attaches; the detail says so
    hub2 = FakeHub(_announce(["grip_ee.dx"] + view_action_names(True)))
    vp2 = _viewpoint("auto", hub2, FakeVpSource(row))
    cell2, bus2, loop2, reader2 = make_loop(tmp_path / "b", viewpoint=vp2, anchor=anchor)
    run_ticks(loop2, cell2, 3, 0.0)
    assert not vp2.attached and "!= view layout" in vp2.detail
    assert np.allclose(loop2._last_cmd["view"], VIEW_Q0)
    # hold: never polls the hub, the announce says so in telemetry
    vp3 = _viewpoint("hold", hub, FakeVpSource(row))
    cell3, bus3, loop3, reader3 = make_loop(tmp_path / "c", viewpoint=vp3, anchor=anchor)
    snap = loop3.run_tick(0.01)
    run_ticks(loop3, cell3, 3, 0.01)
    assert not vp3.attached and np.allclose(loop3._last_cmd["view"], VIEW_Q0)
    assert snap.session_extra["gello"]["viewpoint"]["mode"] == "hold"


def test_viewpoint_nan_three_strike_pauses_until_resume(tmp_path):
    from apollo_mavis_v2_runtime.dagger.policy_runner import ActionAnchor, SlewLimits
    from apollo_mavis_v2_runtime.gello.viewpoint import view_action_names

    src = FakeVpSource([np.nan] * 8)
    vp = _viewpoint("auto", FakeHub(_announce(view_action_names(True))), src)
    anchor = ActionAnchor(FakeIK(), FakeKin(), SlewLimits(), action_space="delta_ee")
    cell, bus, loop, reader = make_loop(tmp_path, viewpoint=vp, anchor=anchor)
    t = 0.0
    for i in range(6):
        src.t = float(i)  # a NEW output every tick
        t = run_ticks(loop, cell, 1, t)
    assert vp.paused and src.paused
    assert np.allclose(loop._last_cmd["view"], VIEW_Q0)
    # 2026-09-09 review: the pause is ON THE WIRE - `paused` plus the reason as the detail,
    # which poll() does not overwrite with "attached" while the pause lasts
    snap = loop.run_tick(t + loop.dt)
    vpt = snap.session_extra["gello"]["viewpoint"]
    assert vpt["paused"] is True and vpt["attached"] is True and vpt["policy_id"] == "viewer"
    assert vpt["detail"] == PAUSED_DETAIL == "paused after 3 NaN actions - press Resume"
    t = run_ticks(loop, cell, 3, t + loop.dt)
    assert vp.telemetry()["detail"] == PAUSED_DETAIL and vp.paused
    assert "viewpoint=paused(3 NaN)" in loop._mode_health(t)
    assert snap.session_extra["gello"]["paused_latched"] is False  # the FOLLOWER is not paused
    res = ack(loop, bus, "gello_resume")  # lifts the NaN pause too (the follower is not paused)
    assert (res.ok, res.detail) == (True, "not paused")
    assert not vp.paused and not src.paused
    assert vp.telemetry()["paused"] is False and vp.telemetry()["detail"] == (
        "external node 'viewer' attached"
    )
    # a detach (the node restarted / a motion window) drops the pause and the strike count,
    # so a re-attached node starts clean
    for i in range(6, 12):
        src.t = float(i)
        t = run_ticks(loop, cell, 1, t)
    assert vp.paused and loop._nan_strikes >= 3
    ack(loop, bus, "gello_motion", active=True, reason="R")
    t = run_ticks(loop, cell, 1, loop._last_tick_now)
    assert not vp.attached and not vp.paused and loop._nan_strikes == 0
    ack(loop, bus, "gello_motion", active=False, reason="R over")
    src.t = 100.0
    t = run_ticks(loop, cell, 2, loop._last_tick_now)
    assert vp.attached and not vp.paused and src.started == 2  # a clean (re)attach
    # the fake keeps emitting NaN: ONE fresh strike since the re-attach, counted from zero
    assert vp.telemetry()["paused"] is False and loop._nan_strikes == 1
