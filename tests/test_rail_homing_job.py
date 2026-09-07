"""``devices/rail_homing.py`` (phase-09d): ``home_rail`` with a planned pre-positioning
motion, over fakes only - a module-scoped ``Runtime`` with the mavis_v2 hardware
workcell (both arms, no cameras, overlay off), a FAKE read-only monitor whose
Manipulation Arm sample is the table-dive posture (sweep-blocked at rail 0 m but
plannable to the keyframe posture), and ``tests/fakes.HardwareFakeWorkcell`` with
``FakeRailDriverArm`` (the ``XArmDriver`` rail surface: ``rail_position_known``,
``home_rail()``) through the ``SessionManager.workcell_factory`` seam. No control
box anywhere (``conftest`` also guards the real workcell factory).

Covers the contract's acceptance: dry run -> ``pre_position.needed`` + waypoints;
the real op -> 202 ``accepted`` + ``job_id`` -> the phases queued / sweeping /
planning / connecting / positioning / homing / verifying / done on
``telemetry.hardware_monitor.arms[].maintenance`` with ``maintenance_busy`` true
throughout -> ``POST /api/session`` and every other op 409 "rail homing in progress"
meanwhile -> the driver's ``home_rail`` called ONLY after the arm reached the
planned posture, the loop stopped, the rail slot never commanded -> teardown (D6)
-> monitor resumed -> ``GET .../maintenance/last`` with the same ``job_id``. Plus
the failure rollback (homing fails / the arm moved since the sweep), the refusal
without a rail-safe plan, the synchronous 09c path for a clear posture and the
``RailHoldArm`` adapter."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import replace

import numpy as np
import pytest
from apollo_mavis_v2_core import CommandError, WorkcellConfig
from apollo_mavis_v2_core.protocol import ArmMaintenanceResult
from apollo_mavis_v2_core.testing import FakeArm
from conftest import FakeMonitorFactory, FakeMonitorSample, make_runtime_config
from fakes import FakeRailDriverArm, FakeServoLimits, HardwareFakeWorkcell
from starlette.testclient import TestClient

from apollo_mavis_v2_runtime.config import (
    HardwareMonitorConfig,
    HardwareProbeConfig,
    MicrophoneConfig,
    TwinOverlayConfig,
)
from apollo_mavis_v2_runtime.devices import rail_homing
from apollo_mavis_v2_runtime.devices.rail_homing import PHASE_ORDER, RailHomingJob
from apollo_mavis_v2_runtime.runtime import Runtime
from apollo_mavis_v2_runtime.server.app import create_app
from apollo_mavis_v2_runtime.session.hardware import (
    RailHoldArm,
    RailHoldWorkcell,
    SessionStateProvider,
    plan_duration_s,
    sample_from_state,
    servo_executor_caps,
)

PI = math.pi
KEYFRAME_Q = (PI, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
TABLE_DIVE_Q = (PI, 0.8, 0.0, 0.5, 0.0, 0.3, 0.0)  # blocked at rail 0 m, plannable to keyframe
REACH_SIDE_Q = (PI / 2, 0.9, 0.0, 0.9, 0.0, 0.0, 0.0)  # blocked, NO rail-safe plan
URL = "/api/hardware/arms/{}/maintenance"
HW_SPEC = {
    "mode": "teleop",
    "kind": "hardware",
    "arms": ["grip", "view"],
    "frames": {"grip": "arm_base:grip", "view": "arm_base:view"},
    "digital_twin_scene": "mavis_v2",
    "speed_scale": 0.1,
}
SIM_SPEC = {  # the runtime config's sim workcell (single_rail / arm0): ANY kind is refused
    "mode": "teleop",
    "kind": "sim",
    "arms": ["arm0"],
    "frames": {},
    "sim_scene": "single_rail",
}
HW_WORKCELL = {
    "kind": "hardware",
    "digital_twin_scene": "mavis_v2",
    "arms": [
        {
            "id": "grip",
            "ip": "192.168.1.201",
            "base_in_world": {},
            "gripper": "xarm_g2",
            "tcp_load_kg": 0.95,
            "tcp_load_cog_mm": [0, 0, 60],
            "collision_sensitivity": 3,
        },
        {
            "id": "view",
            "ip": "192.168.2.219",
            "base_in_world": {},
            "gripper": "none",
            "microphone": True,
            "tcp_load_kg": 0.55,
            "tcp_load_cog_mm": [0, 0, 90],
            "collision_sensitivity": 3,
        },
    ],
    "cameras": [],
    "safety": {"enabled": True},
}


def _sample(arm_id: str, q, *, seq: int = 1, homed: bool = False) -> FakeMonitorSample:
    return FakeMonitorSample(
        arm_id,
        seq=seq,
        t_mono=time.monotonic(),
        q=tuple(q),
        rail_present=True,
        rail_homed=homed,
        rail_enabled=homed,
        rail_pos_m=0.0 if homed else None,
        gripper_open_frac=1.0 if arm_id == "grip" else None,
        collision_sensitivity=3,
        tcp_load_kg=0.95 if arm_id == "grip" else 0.55,
        tcp_load_cog_mm=(0.0, 0.0, 60.0 if arm_id == "grip" else 90.0),
    )


def _config(tmp_path):
    cfg = make_runtime_config(tmp_path)
    return cfg.model_copy(
        update={
            "workcells": {**cfg.workcells, "hardware": WorkcellConfig.model_validate(HW_WORKCELL)},
            "microphone": MicrophoneConfig(enabled=False),
            "hardware_probe": HardwareProbeConfig(enabled=False),
            "hardware_monitor": HardwareMonitorConfig(),
            "twin_overlay": TwinOverlayConfig(enabled=False),
        }
    )


def _wait(pred, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


@pytest.fixture(scope="module")
def factory():
    return FakeMonitorFactory(
        {"grip": _sample("grip", TABLE_DIVE_Q, seq=3), "view": _sample("view", KEYFRAME_Q, seq=5)}
    )


@pytest.fixture(scope="module")
def rt(tmp_path_factory, factory):
    return Runtime(_config(tmp_path_factory.mktemp("rt")), monitor_factory=factory)


@pytest.fixture(scope="module")
def client(rt):
    with TestClient(create_app(rt)) as c:
        assert _wait(lambda: rt.hardware_monitor.status_of("grip")[0] == "running")
        yield c
        c.delete("/api/session")


class Cells:
    """``workcell_factory`` seam: a ``HardwareFakeWorkcell`` whose railed arms are
    ``FakeRailDriverArm``s starting at the MONITOR's posture (``q_from`` overrides)
    with the track unhomed (``homed``), homing in ``homing_duration_s`` or failing
    with ``homing_fails``; every built cell is kept."""

    def __init__(self, factory, **arm_kw) -> None:
        self.factory = factory
        self.arm_kw = arm_kw
        self.built: list[HardwareFakeWorkcell] = []
        self.q_from: dict[str, tuple] = {}

    def __call__(self, session_cfg, driver_factory):
        arms = {}
        for a in session_cfg.arms:
            q7 = self.q_from.get(a.id) or self.factory.monitors[a.id].sample.q
            rail = 0.65 if a.id == "grip" else 0.0
            arms[a.id] = FakeRailDriverArm(a.id, q0=[*q7, rail], **self.arm_kw)
        cell = HardwareFakeWorkcell(arms, kind="hardware")
        self.built.append(cell)
        return cell

    @property
    def cell(self) -> HardwareFakeWorkcell:
        return self.built[-1]


@pytest.fixture()
def cells(rt, factory):
    """The seam + a reset of the module-scoped fake samples afterwards (a test that
    fails half-way must not leak a homed / moved sample into the next one)."""
    seam = Cells(factory, homing_duration_s=0.2)
    rt.manager.workcell_factory = seam
    grip, view = factory.monitors["grip"], factory.monitors["view"]
    samples = (grip.sample, view.sample)
    yield seam
    rt.manager.workcell_factory = None
    rt.rail_homing.stop()
    rt.manager.teardown()
    grip.sample = replace(samples[0], seq=grip.sample.seq + 1)
    view.sample = replace(samples[1], seq=view.sample.seq + 1)


def _telemetry(client, pred, frames: int = 80) -> dict:
    with client.websocket_connect("/ws/telemetry") as ws:
        for _ in range(frames):
            msg = ws.receive_json()
            if pred(msg):
                return msg
    raise AssertionError(f"telemetry never satisfied the predicate; last frame {msg}")


def _hw_arm(msg: dict, arm_id: str) -> dict:
    return next(a for a in msg["hardware_monitor"]["arms"] if a["arm_id"] == arm_id)


def _follow_homing(factory, cell, arm_id: str = "grip") -> threading.Thread:
    """A watcher that turns the fake monitor's sample homed once the fake driver
    homed (the real monitor re-reads the registers after the resume)."""

    def watch() -> None:
        mon = factory.monitors[arm_id]
        arm = cell.arms[arm_id]
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and not arm.rail_position_known:
            time.sleep(0.02)
        if arm.rail_position_known:
            mon.sample = replace(
                mon.sample,
                seq=mon.sample.seq + 10,
                q=tuple(float(v) for v in arm.get_state().q[:7]),
                rail_homed=True,
                rail_enabled=True,
                rail_pos_m=0.0,
                rail_raw_mm=0.0,
            )

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    return t


# -- adapters / helpers ---------------------------------------------------------------------------
def test_rail_hold_arm_shows_the_fallback_and_never_commands_the_rail():
    arm = FakeRailDriverArm("grip", q0=[*TABLE_DIVE_Q, 0.4], instant=True)
    arm.connect()
    hold = RailHoldArm(arm, 0.65)
    st = hold.get_state()
    assert st.q[7] == pytest.approx(0.65) and st.rail_pos_m == pytest.approx(0.65)  # fallback
    assert arm.get_state().q[7] == 0.0  # the driver's placeholder underneath
    assert st.q[:7] == pytest.approx(list(TABLE_DIVE_Q)) and hold.dof == 8 and hold.has_rail
    hold.command_joints(np.array([*KEYFRAME_Q, 0.65]))  # the loop's hold at the fallback
    assert hold.rail_commands_dropped == 1 and arm.command_log[-1][7] == 0.0  # pinned
    assert arm.get_state().q[:7] == pytest.approx(list(KEYFRAME_Q))  # joints forwarded
    with pytest.raises(CommandError, match="rail locked"):
        hold.command_rail(0.3)
    assert hold.rail_position_known is False and hold.rail_phase == "DETECTED"  # forwarded
    out = arm.home_rail()
    assert out.ok and hold.position_known and hold.get_state().q[7] == 0.0  # measured now
    hold.command_joints(np.array([*KEYFRAME_Q, 0.65]))
    assert hold.rail_commands_dropped == 2 and arm.get_state().q[7] == 0.0  # still pinned
    cell = HardwareFakeWorkcell({"grip": arm, "view": FakeArm("view")}, kind="hardware")
    wrapped = RailHoldWorkcell(cell, {"grip": 0.65, "view": 0.0})
    assert isinstance(wrapped.arms["grip"], RailHoldArm)
    assert wrapped.arms["view"] is cell.arms["view"]  # rail-less arms pass through
    assert wrapped.states()["grip"].q.shape == (8,)


def test_plan_duration_estimate_follows_the_executor_slew():
    wps = [[*TABLE_DIVE_Q, 0.65], [*KEYFRAME_Q, 0.65]]
    # max |dq| = 0.8 rad at 0.002 rad/tick x 100 Hz = 4 s (the rail slot is ignored)
    assert plan_duration_s(wps, 0.002, 100.0) == pytest.approx(4.0)
    assert plan_duration_s(wps, 0.02, 100.0) == pytest.approx(0.4)
    assert plan_duration_s([wps[0]], 0.002, 100.0) == 0.0
    # the hardware executor caps: the lever-weighted Cartesian step binds when it is the
    # slower bound - sum|dq| lever = 0.8*1.2 + 0.5*0.75 + 0.3*0.3 = 1.425 m over 0.0002 m/tick
    lever = (1.20, 1.20, 1.00, 0.75, 0.44, 0.30, 0.10)
    assert plan_duration_s(wps, 0.002, 100.0, cart_step_m=0.0002, lever_arm_m=lever) == (
        pytest.approx(1.425 / 0.0002 / 100.0)
    )
    assert plan_duration_s(wps, 0.002, 100.0, cart_step_m=1.0, lever_arm_m=lever) == (
        pytest.approx(4.0)
    )


def test_servo_executor_caps_bound_the_host_slew_by_the_driver_stream():
    """The job's executor caps come from the driver's ServoLimits (already scaled):
    per-joint slew = min(host slew, min max_joint_vel / rate), cart step per loop
    tick, the levers; a faster driver leaves the host slew alone."""
    servo = FakeServoLimits()  # the hardware defaults at scale 0.1
    caps = servo_executor_caps(servo, 100.0, 0.002)
    assert caps.source == "servo" and caps.slew_rad_per_tick == pytest.approx(0.0003)
    assert caps.cart_step_m == pytest.approx(0.0002) and caps.lever_arm_m == servo.lever_arm_m
    # the same caps from the UNscaled defaults x 0.1 (what the request-time estimate uses)
    caps2 = servo_executor_caps(
        FakeServoLimits(max_joint_vel=(0.3,) * 7, max_cart_step_m=0.002), 100.0, 0.002, scale=0.1
    )
    assert caps2.slew_rad_per_tick == pytest.approx(0.0003)
    assert caps2.cart_step_m == pytest.approx(0.0002)
    fast = servo_executor_caps(FakeServoLimits(max_joint_vel=(2.0,) * 7), 100.0, 0.002)
    assert fast.slew_rad_per_tick == pytest.approx(0.002)  # the host slew stays the bound
    # a streamer at 200 Hz gets two ticks per loop tick: the cart bound doubles
    assert servo_executor_caps(FakeServoLimits(rate_hz=200.0), 100.0, 0.002).cart_step_m == (
        pytest.approx(0.0004)
    )


def test_overlay_sample_reports_an_unhomed_driver_rail_as_unknown():
    """The job's alignment overlay is fed from the raw driver: while its track is
    unhomed the published 0.0 rail slot is a placeholder, so the sample says
    rail present but NOT homed / enabled with ``rail_pos_m None`` (the overlay
    then applies its own fallback + caption); once homed the position passes."""
    arm = FakeRailDriverArm("grip", q0=[*TABLE_DIVE_Q, 0.4], instant=True)
    arm.connect()
    st = arm.get_state()
    assert sample_from_state(st, 1, has_gripper=True).rail_pos_m == 0.0  # default: homed track
    raw = sample_from_state(st, 1, has_gripper=True, rail_known=False)
    assert raw.rail_present is True and raw.rail_homed is False and raw.rail_enabled is False
    assert (
        raw.rail_pos_m is None and raw.rail_raw_mm == 0.0 and raw.q == pytest.approx(TABLE_DIVE_Q)
    )
    cell = HardwareFakeWorkcell({"grip": arm}, kind="hardware")
    provider = SessionStateProvider(cell, ["grip"], {}, ["grip"])
    sample = provider.samples()["grip"]
    assert sample.rail_homed is False and sample.rail_pos_m is None
    assert provider.status_of("grip") == ("running", "")
    assert arm.home_rail().ok
    sample = provider.samples()["grip"]
    assert sample.rail_homed is True and sample.rail_enabled is True and sample.rail_pos_m == 0.0


# -- the full chain -------------------------------------------------------------------------------
def test_dry_run_then_accepted_job_runs_every_phase_and_homes_after_positioning(
    client, rt, factory, cells
):
    grip, view = factory.monitors["grip"], factory.monitors["view"]
    n_grip, n_view = len(grip.calls), len(view.calls)
    # 1. dry run: a plan is needed and exists; nothing written, no job
    r = client.post(URL.format("grip"), json={"op": "home_rail", "dry_run": True})
    assert r.status_code == 200, r.text
    dry = ArmMaintenanceResult.model_validate(r.json())
    plan = dry.rail_sweep.pre_position
    assert dry.status == "done" and dry.ok and plan.needed and plan.clear
    assert plan.source == "keyframe" and plan.waypoints >= 2 and plan.checked_rail_positions == 131
    assert plan.target_q == pytest.approx(list(KEYFRAME_Q))
    assert cells.built == [] and not rt.rail_homing.active and grip.homing_calls == []
    # the estimate is at the HARDWARE executor caps (ServoLimits defaults x 0.1: the cart
    # step binds for this move), not the host's bare 0.002 rad/tick (which would say 4 s)
    assert plan.duration_s > 30.0
    cached = rt.rail_homing.cached_plan("grip")
    assert cached is not None and cached.plan == plan and len(cached.waypoints) == plan.waypoints
    n_eval, n_reval = rt.rail_homing.planner.evaluations, rt.rail_homing.planner.revalidations

    # 2. the real op: 202 accepted + job_id
    r = client.post(URL.format("grip"), json={"op": "home_rail"})
    assert r.status_code == 202, r.text
    acc = ArmMaintenanceResult.model_validate(r.json())
    assert acc.status == "accepted" and acc.ok and acc.job_id and acc.sdk_codes == {}
    assert acc.detail.startswith("accepted: rail sweep blocked at the current posture; the arm")
    assert acc.rail_sweep.pre_position == plan and acc.before is not None
    job = rt.rail_homing.job("grip")
    assert isinstance(job, RailHomingJob) and job.job_id == acc.job_id
    assert rt.hardware_monitor.maintenance_busy("grip") and rt.rail_homing.active
    # a second homing (either arm), any other op and a session are refused meanwhile
    r = client.post(URL.format("view"), json={"op": "home_rail", "dry_run": True})
    assert r.status_code == 409 and "rail homing in progress on the Manipulation Arm" in r.text
    r = client.post(URL.format("view"), json={"op": "clear_errors"})
    assert r.status_code == 409
    assert r.json()["detail"].startswith("clear_errors refused: rail homing in progress")
    r = client.post("/api/session", json=HW_SPEC)
    assert r.status_code == 409 and r.json()["detail"].startswith(
        "rail homing in progress on the Manipulation Arm"
    )
    r = client.post("/api/session", json=SIM_SPEC)  # ANY session kind is refused meanwhile
    assert r.status_code == 409 and r.json()["detail"].startswith(
        "rail homing in progress on the Manipulation Arm"
    )
    # /last carries the 202's accepted result until the job ends (core docstring)
    r = client.get(URL.format("grip") + "/last")
    assert r.status_code == 200 and r.json()["status"] == "accepted"
    assert r.json()["job_id"] == acc.job_id
    # the confirm executed the plan the dry run displayed: re-validated, not re-planned
    assert rt.rail_homing.planner.evaluations == n_eval
    assert rt.rail_homing.planner.revalidations == n_reval + 1
    assert job.waypoints == cached.waypoints and rt.rail_homing.cached_plan("grip") is None
    # the monitor is paused while the job's driver owns the box; telemetry shows the job
    assert _wait(lambda: rt.hardware_monitor.paused, 10.0) and rt.rail_homing.owns_boxes
    msg = _telemetry(client, lambda m: _hw_arm(m, "grip")["maintenance"] is not None)
    row = _hw_arm(msg, "grip")
    assert row["maintenance_busy"] is True and row["maintenance"]["job_id"] == acc.job_id
    assert row["maintenance"]["op"] == "home_rail" and row["maintenance"]["phase"] in PHASE_ORDER
    assert msg["hardware_monitor"]["paused"] is True and msg["session"]["state"] == "idle"
    assert _wait(lambda: len(cells.built) == 1, 10.0)
    cell = cells.cell
    watcher = _follow_homing(factory, cell)
    assert _wait(lambda: not rt.rail_homing.active, 45.0), job.progress
    watcher.join(5.0)

    # 3. the phases, in order; the fake driver homed exactly once, AFTER positioning
    assert job.phases_seen == list(PHASE_ORDER), job.phases_seen
    assert job.progress.phase == "done" and job.progress.progress == 1.0
    arm = cell.arms["grip"]
    assert len(arm.home_rail_calls) == 1
    assert arm.home_rail_calls[0] == pytest.approx(list(KEYFRAME_Q), abs=1e-6)  # at the target
    assert arm.rail_position_known and arm.get_state().q[:7] == pytest.approx(list(KEYFRAME_Q))
    assert set(cell.arms) == {"grip"}  # this arm's driver only; the Perception Arm stays braked
    hold = job.rig.workcell.arms["grip"]
    assert isinstance(hold, RailHoldArm) and hold.rail_commands_dropped > 0  # never a rail move
    assert job.rig.loop.active_arm is None and job.rig.frozen.keys() == {"view"}  # D1
    assert job.rig.speed_scale == 0.1 and job.rig.loop.cfg.dq_max_rad == pytest.approx(0.004)
    assert job.rig.loop._thread is None  # the loop was stopped before home_rail()
    # teardown (D6) + monitor resumed; nothing else was touched
    assert cell.stop_calls == 1 and not cell.started
    assert _wait(lambda: not rt.hardware_monitor.paused, 5.0)
    assert grip.calls[n_grip:] == ["disconnect", "join", "start"]
    assert view.calls[n_view:] == ["disconnect", "join", "start"]  # all monitors pause together
    assert not rt.hardware_monitor.maintenance_busy("grip") and rt.manager.session is None
    assert grip.homing_calls == []  # the MONITOR path never ran: the driver homed

    # 4. the final result at /last, same job_id
    r = client.get(URL.format("grip") + "/last")
    assert r.status_code == 200, r.text
    last = ArmMaintenanceResult.model_validate(r.json())
    assert (last.status, last.ok, last.job_id, last.path) == ("done", True, acc.job_id, "session")
    assert last.detail.startswith("rail homed after a planned pre-positioning motion (")
    assert "the arm holds that posture" in last.detail
    assert list(last.sdk_codes) == [
        "set_linear_track_back_origin",
        "set_linear_track_enable",
        "set_linear_track_speed",
    ]
    assert last.rail_sweep.pre_position == plan and last.before is not None
    assert last.after is not None and last.after.rail_homed and last.after.rail_pos_m == 0.0
    # the terminal phase lingers on telemetry; busy is false
    msg = _telemetry(client, lambda m: _hw_arm(m, "grip")["maintenance_busy"] is False)
    assert _hw_arm(msg, "grip")["maintenance"]["phase"] == "done"
    # a session can start again now (both tracks homed in the fake samples)
    view.sample = replace(view.sample, rail_homed=True, rail_enabled=True, rail_pos_m=0.0)
    cells.q_from = {"grip": KEYFRAME_Q, "view": KEYFRAME_Q}
    cells.arm_kw = {"rail_homed": True}
    try:
        r = client.post("/api/session", json=HW_SPEC)
        assert r.status_code == 200, r.text
    finally:
        assert client.delete("/api/session").status_code == 204  # samples reset by the fixture


def test_clear_posture_still_homes_synchronously_on_the_monitor_path(client, rt, factory, cells):
    grip = factory.monitors["grip"]
    original = grip.sample
    grip.sample = replace(original, q=KEYFRAME_Q, seq=original.seq + 1)
    try:
        r = client.post(URL.format("grip"), json={"op": "home_rail"})
        assert r.status_code == 200, r.text
        res = ArmMaintenanceResult.model_validate(r.json())
        assert (res.status, res.path, res.ok, res.job_id) == ("done", "monitor", True, None)
        assert not res.rail_sweep.pre_position.needed
        assert grip.homing_calls == [tuple(KEYFRAME_Q)] and cells.built == []  # no driver
        assert rt.rail_homing.job("grip") is None or rt.rail_homing.job("grip").terminal
        assert client.get(URL.format("grip") + "/last").json()["path"] == "monitor"
    finally:
        grip.sample = original


def test_no_rail_safe_plan_is_refused_without_a_job(client, rt, factory, cells):
    grip = factory.monitors["grip"]
    original = grip.sample
    n_homings = len(grip.homing_calls)
    grip.sample = replace(original, q=REACH_SIDE_Q, seq=original.seq + 1)
    try:
        r = client.post(URL.format("grip"), json={"op": "home_rail"})
        assert r.status_code == 200, r.text
        res = ArmMaintenanceResult.model_validate(r.json())
        assert res.status == "refused" and not res.ok and res.job_id is None
        assert not res.rail_sweep.pre_position.clear and res.rail_sweep.pre_position.needed
        assert "no rail-safe pre-positioning path" in res.detail
        assert cells.built == [] and len(grip.homing_calls) == n_homings  # zero writes
        assert not rt.rail_homing.active
    finally:
        grip.sample = original


def test_homing_failure_rolls_back_and_reports_failed(client, rt, factory, cells):
    cells.arm_kw = {"homing_duration_s": 0.05, "homing_fails": "linear track error 25"}
    grip = factory.monitors["grip"]
    n0 = len(grip.calls)
    r = client.post(URL.format("grip"), json={"op": "home_rail"})
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert _wait(lambda: not rt.rail_homing.active, 45.0)
    job = rt.rail_homing.job("grip")
    assert job.job_id == job_id and job.progress.phase == "failed"
    assert job.phases_seen[-2:] == ["homing", "failed"]
    cell = cells.cell
    arm = cell.arms["grip"]
    assert len(arm.home_rail_calls) == 1 and not arm.rail_position_known
    assert arm.get_state().q[:7] == pytest.approx(list(KEYFRAME_Q))  # posture held where it is
    assert cell.stop_calls == 1 and not rt.rail_homing.owns_boxes
    assert _wait(lambda: not rt.hardware_monitor.paused, 5.0) and grip.calls[n0:][-1] == "start"
    last = ArmMaintenanceResult.model_validate(client.get(URL.format("grip") + "/last").json())
    assert (last.status, last.ok, last.job_id) == ("done", False, job_id)
    assert last.detail.startswith("rail homing job failed during homing: rail homing failed:")
    assert "linear track error 25" in last.detail
    assert last.sdk_codes["set_linear_track_back_origin"] == 80
    assert not rt.hardware_monitor.maintenance_busy("grip")


def test_arm_moved_since_the_sweep_fails_during_connecting(client, rt, factory, cells):
    cells.q_from = {"grip": KEYFRAME_Q}  # the driver measures a posture the sweep never saw
    grip = factory.monitors["grip"]
    r = client.post(URL.format("grip"), json={"op": "home_rail"})
    assert r.status_code == 202, r.text
    assert _wait(lambda: not rt.rail_homing.active, 30.0)
    job = rt.rail_homing.job("grip")
    assert job.progress.phase == "failed" and job.phases_seen[-2:] == ["connecting", "failed"]
    cell = cells.cell
    assert cell.arms["grip"].home_rail_calls == [] and cell.stop_calls == 1
    last = client.get(URL.format("grip") + "/last").json()
    assert last["ok"] is False and "the arm moved since the sweep" in last["detail"]
    assert _wait(lambda: not rt.hardware_monitor.paused, 5.0) and grip.calls[-1] == "start"


def test_home_rail_is_refused_while_any_session_is_active_or_starting(
    client, rt, factory, cells, monkeypatch
):
    """Contract 注意事項 1: the maintenance motion excludes EVERY session - a sim
    one and a create() still validating (its kind is not recorded yet) included."""
    monkeypatch.setattr(rt.manager, "_creating", True, raising=True)  # a create() in flight
    for body in ({"op": "home_rail", "dry_run": True}, {"op": "home_rail"}):
        r = client.post(URL.format("grip"), json=body)
        assert r.status_code == 409, r.text
        assert "while a session is active or starting" in r.json()["detail"]
    assert cells.built == [] and not rt.rail_homing.active and not rt.hardware_monitor.paused


def test_job_refuses_to_connect_while_a_session_is_active(client, rt, factory, cells, monkeypatch):
    """Belt and braces for the window between the 202 and the connect (create()
    itself is refused while the job is registered): a session that slipped in
    first wins - the job fails during ``connecting``, never pauses the monitor and
    therefore never resumes it either (a resume would reconnect the read-only
    clients onto boxes the session's drivers own)."""
    real_run = RailHomingJob.run

    def run_with_session(self):  # the create() lands right before the job's connect
        rt.manager._creating = True
        real_run(self)

    monkeypatch.setattr(RailHomingJob, "run", run_with_session)
    resumes = []
    real_resume = rt.hardware_monitor.resume
    monkeypatch.setattr(rt.hardware_monitor, "resume", lambda: resumes.append(1) or real_resume())
    r = client.post(URL.format("grip"), json={"op": "home_rail"})
    assert r.status_code == 202, r.text
    try:
        assert _wait(lambda: not rt.rail_homing.active, 30.0)
    finally:
        rt.manager._creating = False
    job = rt.rail_homing.job("grip")
    assert job.progress.phase == "failed" and cells.built == []
    assert "a session is active or starting" in job.progress.detail
    assert resumes == []  # this job never paused the monitor -> it does not resume it
    assert not rt.hardware_monitor.paused
    last = client.get(URL.format("grip") + "/last").json()
    assert last["job_id"] == job.job_id and last["ok"] is False and last["status"] == "done"


def test_concurrent_requests_reserve_the_cell_atomically(client, rt, factory, cells, monkeypatch):
    """Two overlapping ``home_rail`` POSTs (a double-click, or one per arm via
    curl): the first reserves the cell BEFORE its seconds of planning, the second
    is 409 "rail homing in progress" - and so is any session meanwhile. Exactly
    one job starts."""
    planner = rt.rail_homing.planner
    real_evaluate = planner.evaluate
    entered = threading.Event()
    release = threading.Event()

    def slow_evaluate(*args, **kwargs):
        entered.set()
        release.wait(5.0)
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(planner, "evaluate", slow_evaluate)
    results: dict[str, tuple[int, dict]] = {}

    def post(name, arm_id, body):
        r = client.post(URL.format(arm_id), json=body)
        results[name] = (r.status_code, r.json())

    first = threading.Thread(target=post, args=("first", "grip", {"op": "home_rail"}))
    first.start()
    assert entered.wait(10.0)  # the first request is inside the (slowed) planning
    assert rt.rail_homing.active_arm == "grip" and rt.rail_homing.active
    # meanwhile: a second request (either arm, dry run or not), any other op, any session
    post("second", "view", {"op": "home_rail", "dry_run": True})
    post("third", "grip", {"op": "home_rail"})
    r = client.post(URL.format("view"), json={"op": "clear_errors"})
    assert r.status_code == 409 and "rail homing in progress" in r.json()["detail"]
    r = client.post("/api/session", json=SIM_SPEC)
    assert r.status_code == 409 and r.json()["detail"].startswith(
        "rail homing in progress on the Manipulation Arm"
    )
    monkeypatch.setattr(planner, "evaluate", real_evaluate)  # the job's own path is unslowed
    release.set()
    first.join(20.0)
    assert results["first"][0] == 202, results["first"]
    for name in ("second", "third"):
        code, body = results[name]
        assert code == 409 and body["detail"].startswith(
            "home_rail refused: rail homing in progress on the Manipulation Arm"
        ), results[name]
    job = rt.rail_homing.job("grip")
    assert job is not None and job.job_id == results["first"][1]["job_id"]
    assert rt.rail_homing.job("view") is None or rt.rail_homing.job("view").terminal
    assert _wait(lambda: len(cells.built) == 1, 10.0)
    watcher = _follow_homing(factory, cells.cell)
    assert _wait(lambda: not rt.rail_homing.active, 45.0), job.progress
    watcher.join(5.0)
    assert job.progress.phase == "done" and len(cells.built) == 1
    assert _wait(lambda: not rt.hardware_monitor.paused, 5.0)


def test_confirm_after_the_arm_moved_refuses_a_changed_plan(
    client, rt, factory, cells, monkeypatch
):
    """The confirm executes the plan the dry run displayed. When the arm moved
    beyond the start tolerance a fresh plan is made - and refused if it is not the
    one the operator confirmed ("plan changed - re-confirm"); nothing connects."""
    grip = factory.monitors["grip"]
    original = grip.sample
    r = client.post(URL.format("grip"), json={"op": "home_rail", "dry_run": True})
    assert r.status_code == 200 and r.json()["rail_sweep"]["pre_position"]["needed"]
    cached = rt.rail_homing.cached_plan("grip")
    assert cached is not None
    planner = rt.rail_homing.planner
    real_evaluate = planner.evaluate

    def different_plan(*args, **kwargs):  # a fresh plan with one waypoint more
        plan, wps = real_evaluate(*args, **kwargs)
        return plan.model_copy(update={"waypoints": plan.waypoints + 1}), wps

    monkeypatch.setattr(planner, "evaluate", different_plan)
    # 0.05 rad on joint 5 > START_POSTURE_TOL_RAD, and still blocked-but-plannable on the
    # measured mavis_v2 (2026-09-06): the same nudge on joint 7 now puts the finger 6.8 mm
    # from the table and the straight path to the keyframe fails the position-agnostic check
    moved = (PI, 0.8, 0.0, 0.5, 0.05, 0.3, 0.0)
    grip.sample = replace(original, q=moved, seq=original.seq + 1)
    try:
        r = client.post(URL.format("grip"), json={"op": "home_rail"})
        assert r.status_code == 200, r.text
        res = ArmMaintenanceResult.model_validate(r.json())
        assert res.status == "refused" and not res.ok and res.job_id is None
        assert "plan changed since the dry run" in res.detail and "re-confirm" in res.detail
        assert cells.built == [] and not rt.rail_homing.active
        assert rt.rail_homing.cached_plan("grip") is not None  # the dry run stays displayed
    finally:
        grip.sample = original


def test_streamed_driver_follows_the_validated_straight_segments(client, rt, factory, cells):
    """With a fake that models the driver's servo streamer (per-joint velocity clip,
    lever-weighted Cartesian scaling) the host executor is capped by the driver's
    caps: the MEASURED path stays on the planner's straight joint-space segments
    (the ones ``check_path`` validated) instead of the L-inf bend a faster host
    command would produce, and the job still reaches the posture and homes."""
    servo = FakeServoLimits(max_joint_vel=(0.1,) * 7, max_cart_step_m=0.02)  # 0.001 rad/tick
    cells.arm_kw = {"homing_duration_s": 0.1, "instant": False, "servo": servo}
    grip = factory.monitors["grip"]
    n_monitor_homings = len(grip.homing_calls)  # the module-scoped fake monitor's history
    r = client.post(URL.format("grip"), json={"op": "home_rail"})
    assert r.status_code == 202, r.text
    job = rt.rail_homing.job("grip")
    assert _wait(lambda: len(cells.built) == 1, 10.0)
    cell = cells.cell
    watcher = _follow_homing(factory, cell)
    assert _wait(lambda: not rt.rail_homing.active, 60.0), job.progress
    watcher.join(5.0)
    assert job.progress.phase == "done", job.progress
    jog = job.rig.loop.cfg.jog
    assert job.rig.executor_caps.source == "servo"
    assert jog.slew_rad_per_tick == pytest.approx(0.001)  # the host 0.002 capped by the driver
    assert jog.plan_cart_step_m == pytest.approx(0.02) and jog.plan_lever_arm_m == servo.lever_arm_m
    arm = cell.arms["grip"]
    assert len(arm.home_rail_calls) == 1
    assert arm.home_rail_calls[0] == pytest.approx(list(KEYFRAME_Q), abs=0.01)
    # every streamed configuration lies on the polyline of the planned waypoints
    wps = [np.asarray(w[:7]) for w in job.waypoints]
    assert len(arm.path_log) > 100

    def off_path(q7: np.ndarray) -> float:
        best = np.inf
        for a, b in zip(wps[:-1], wps[1:], strict=True):
            d = b - a
            t = 0.0 if not np.any(d) else float(np.clip(np.dot(q7 - a, d) / np.dot(d, d), 0.0, 1.0))
            best = min(best, float(np.max(np.abs(q7 - (a + t * d)))))
        return best

    devs = [off_path(np.asarray(q[:7])) for q in arm.path_log]
    # the host loop's tick jitter (a short burst of catch-up ticks) lets the command run a
    # few ticks ahead, so the streamer's per-joint clip may bend by ~ticks x slew (mrad);
    # the UNCAPPED executor (0.002 vs the driver's 0.001 rad/tick) bends this move by
    # > 0.1 rad - the small-delta joints arrive first
    assert max(devs) < 0.02, max(devs)
    assert float(np.median(devs)) < 0.005, float(np.median(devs))
    assert not rt.hardware_monitor.paused
    assert len(grip.homing_calls) == n_monitor_homings  # the DRIVER homed, not the monitor path


def test_runtime_stop_cancels_a_positioning_job_and_hands_the_arm_back(client, rt, factory, cells):
    """``RailHomingService.stop`` (Runtime.stop) cancels a job at its next safe
    point: the positioning motion ends where it is (on the validated path), the
    rig is torn down (D6), the track is NOT homed, the monitor resumes."""
    servo = FakeServoLimits(max_joint_vel=(0.02,) * 7, max_cart_step_m=0.02)  # slow: ~40 s move
    cells.arm_kw = {"homing_duration_s": 0.1, "instant": False, "servo": servo}
    r = client.post(URL.format("grip"), json={"op": "home_rail"})
    assert r.status_code == 202, r.text
    job = rt.rail_homing.job("grip")
    assert _wait(lambda: job.progress.phase == "positioning", 20.0), job.progress
    time.sleep(0.5)  # a few hundred streamer ticks into the motion
    t0 = time.monotonic()
    rt.rail_homing.stop(timeout_s=20.0)
    assert time.monotonic() - t0 < 15.0 and not job.is_alive()
    assert job.cancelled and job.progress.phase == "failed"
    assert "cancelled during the pre-positioning motion" in job.progress.detail
    cell = cells.cell
    arm = cell.arms["grip"]
    assert arm.home_rail_calls == [] and not arm.rail_position_known
    assert cell.stop_calls == 1 and not rt.rail_homing.owns_boxes
    q = arm.get_state().q[:7]
    assert 0.0 < abs(q[1] - TABLE_DIVE_Q[1]) < 0.8  # stopped part-way along the segment
    assert _wait(lambda: not rt.hardware_monitor.paused, 5.0)
    last = client.get(URL.format("grip") + "/last").json()
    assert last["ok"] is False and "cancelled" in last["detail"] and last["job_id"] == job.job_id


def test_progress_lingers_then_clears(rt, factory, cells, monkeypatch):
    job = rt.rail_homing.job("grip")
    assert job is not None and job.terminal
    assert rt.rail_homing.progress("grip") is not None
    monkeypatch.setattr(rail_homing, "PROGRESS_LINGER_S", 0.0)
    assert rt.rail_homing.progress("grip") is None and not rt.rail_homing.busy("grip")
