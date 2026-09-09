"""``POST /api/hardware/arms/{arm_id}/maintenance`` (phase-09b/09c; 04-runtime
§13.1 / §15) over a Runtime with a hardware workcell config, a FAKE read-only
monitor and, for the session path, a REAL ``POST /api/session kind=hardware``
over ``tests/fakes.HardwareFakeWorkcell`` (the ``workcell_factory`` seam) - no
control box anywhere.

Routes: the monitor path (no session: clear_errors / apply_backstops on the arm
monitor's thread, ``recover`` -> 409, ``home_rail`` = the twin-gated homing:
dry-run verdict, blocked sweep = ok false + zero writes, clear sweep = the
fake's exact write set), the session path (a hardware session owns the boxes:
clear_errors / recover = the driver's user recovery, apply_backstops and
home_rail -> 409) and the 409 matrix (unknown arm 404, monitor off / paused /
erroring, busy; ``POST /api/session`` while a homing is in flight). Plus the
fake full chain of the phase-09c acceptance (unhomed -> session 409 -> home_rail
-> session running) and the session state machine FAULT -> RECOVERING ->
RUNNING driven by the scripted driver events."""

from __future__ import annotations

import math
import time
from dataclasses import replace

import pytest
from apollo_mavis_v2_core import WorkcellConfig
from apollo_mavis_v2_core.protocol import ArmMaintenanceResult
from apollo_mavis_v2_core.testing import FakeArm
from conftest import (
    BACKSTOP_SEQUENCE,
    FakeMaintenanceOutcome,
    FakeMonitorFactory,
    FakeMonitorSample,
    make_runtime_config,
)
from fakes import HardwareFakeWorkcell
from starlette.testclient import TestClient

from apollo_mavis_v2_runtime.config import (
    HardwareMonitorConfig,
    HardwareProbeConfig,
    MicrophoneConfig,
    TwinOverlayConfig,
)
from apollo_mavis_v2_runtime.runtime import Runtime
from apollo_mavis_v2_runtime.server.app import create_app
from apollo_mavis_v2_runtime.session.types import SessionState

PI = math.pi
HOME_RAIL_WRITES = [
    "set_linear_track_back_origin",
    "set_linear_track_enable",
    "set_linear_track_speed",
]
TABLE_DIVE_Q = (PI, 0.8, 0.0, 0.5, 0.0, 0.3, 0.0)  # gripper below the table top (test_rail_sweep)
# Manipulation Arm reaching sideways along the rail: the sweep is blocked (the gripper
# meets the rail base near 0 m) AND no candidate posture has a position-agnostic path
# (every path would first move the gripper deeper into the rail base at rail 0 m).
REACH_SIDE_Q = (PI / 2, 0.9, 0.0, 0.9, 0.0, 0.0, 0.0)
HW_SPEC = {
    "mode": "teleop",
    "kind": "hardware",
    "arms": ["grip", "view"],  # phase-09d: every configured arm, always
    "frames": {"grip": "arm_base:grip", "view": "arm_base:view"},
    "digital_twin_scene": "mavis_v2",
    "speed_scale": 0.1,
}
GRIP_Q0 = [PI, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.65]  # keyframe posture, rail at the right end
VIEW_Q0 = [PI, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

HW_WORKCELL = {
    "kind": "hardware",
    "digital_twin_scene": "mavis_v2",
    "arms": [
        {
            "id": "grip",  # Manipulation Arm
            "ip": "192.168.1.201",
            "base_in_world": {},
            "gripper": "xarm_g2",
            "tcp_load_kg": 0.95,
            "tcp_load_cog_mm": [0, 0, 60],
            "collision_sensitivity": 3,
        },
        {
            "id": "view",  # Perception Arm
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
URL = "/api/hardware/arms/{}/maintenance"


def _samples() -> dict[str, FakeMonitorSample]:
    """The lab as found on 2026-09-04: payload 0 kg on both boxes, sensitivity 3
    (grip) / 1 (view), C19 latched on the Perception Arm."""
    now = time.monotonic()
    return {
        "grip": FakeMonitorSample(  # rail present, NOT homed (as found): rail_pos_m None
            "grip",
            seq=3,
            t_mono=now,
            collision_sensitivity=3,
            tcp_load_kg=0.0,
            tcp_load_cog_mm=(0.0, 0.0, 0.0),
            gripper_open_frac=1.0,
            gripper_raw=84.0,
        ),
        "view": FakeMonitorSample(
            "view",
            seq=5,
            t_mono=now,
            error_code=19,
            collision_sensitivity=1,
            tcp_load_kg=0.0,
            tcp_load_cog_mm=(0.0, 0.0, 0.0),
        ),
    }


def _config(tmp_path):
    cfg = make_runtime_config(tmp_path)  # sim single_rail previews (light)
    return cfg.model_copy(
        update={
            "workcells": {**cfg.workcells, "hardware": WorkcellConfig.model_validate(HW_WORKCELL)},
            "microphone": MicrophoneConfig(enabled=False),
            "hardware_probe": HardwareProbeConfig(enabled=False),
            "hardware_monitor": HardwareMonitorConfig(),
            "twin_overlay": TwinOverlayConfig(enabled=False),
        }
    )


def _wait(pred, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


@pytest.fixture(scope="module")
def factory():
    return FakeMonitorFactory(_samples())


@pytest.fixture(scope="module")
def rt(tmp_path_factory, factory):
    return Runtime(_config(tmp_path_factory.mktemp("rt")), monitor_factory=factory)


@pytest.fixture(scope="module")
def client(rt):
    with TestClient(create_app(rt)) as c:
        assert _wait(lambda: rt.hardware_monitor.status_of("grip")[0] == "running")
        yield c
        c.delete("/api/session")


def _telemetry(client, pred, frames: int = 50) -> dict:
    with client.websocket_connect("/ws/telemetry") as ws:
        for _ in range(frames):
            msg = ws.receive_json()
            if pred(msg):
                return msg
    raise AssertionError(f"telemetry never satisfied the predicate; last frame {msg}")


def _hw_arm(msg: dict, arm_id: str) -> dict:
    return next(a for a in msg["hardware_monitor"]["arms"] if a["arm_id"] == arm_id)


# -- 404 / 422 ---------------------------------------------------------------------------------
def test_unknown_arm_404_and_bad_op_422(client):
    r = client.post(URL.format("arm9"), json={"op": "clear_errors"})
    assert r.status_code == 404 and "arm9" in r.json()["detail"]
    assert client.post(URL.format("grip"), json={"op": "go_home"}).status_code == 422
    assert client.post(URL.format("grip"), json={}).status_code == 422
    body = {"op": "home_rail", "dry_run": "yes?"}
    assert client.post(URL.format("grip"), json=body).status_code == 422


# -- monitor path (no session) -------------------------------------------------------------
def test_telemetry_read_backs_and_backstops_match_before_any_op(client):
    msg = _telemetry(client, lambda m: m["hardware_monitor"]["enabled"])
    grip, view = _hw_arm(msg, "grip"), _hw_arm(msg, "view")
    assert (grip["collision_sensitivity"], grip["tcp_load_kg"]) == (3, 0.0)
    assert grip["tcp_load_cog_mm"] == [0.0, 0.0, 0.0]
    assert grip["backstops_match"] is False  # payload 0 kg != 0.95 kg
    assert view["backstops_match"] is False  # sensitivity 1 != 3
    assert grip["maintenance_busy"] is False and view["maintenance_busy"] is False
    assert view["error_code"] == 19


def test_clear_errors_on_the_perception_arm_via_the_monitor(client, factory):
    r = client.post(URL.format("view"), json={"op": "clear_errors"})
    assert r.status_code == 200, r.text
    res = ArmMaintenanceResult.model_validate(r.json())
    assert (res.arm_id, res.op, res.path, res.ok) == ("view", "clear_errors", "monitor", True)
    assert list(res.sdk_codes) == ["clean_error", "clean_warn"]  # never motion_enable
    assert res.detail == "cleared controller error 19" and res.warnings == []
    assert res.before is not None and res.before.error_code == 19
    assert res.after is not None and res.after.error_code == 0 and res.after.arm_id == "view"
    assert res.after.status == "running" and res.after.maintenance_busy is False
    assert res.after.backstops_match is False  # clearing errors does not touch the backstops
    op, driver_cfg, timeout_s = factory.monitors["view"].maintenance_calls[-1]
    assert op == "clear_errors" and timeout_s == 10.0
    # The monitor's newest sample reflects the op -> Welcome page error code goes to 0.
    assert rt_error_code(client, "view") == 0


def rt_error_code(client, arm_id: str) -> int:
    msg = _telemetry(client, lambda m: m["hardware_monitor"]["enabled"])
    return _hw_arm(msg, arm_id)["error_code"]


def test_apply_backstops_on_the_manipulation_arm_via_the_monitor(client, factory):
    r = client.post(URL.format("grip"), json={"op": "apply_backstops"})
    assert r.status_code == 200, r.text
    res = ArmMaintenanceResult.model_validate(r.json())
    assert (res.path, res.ok) == ("monitor", True)
    assert list(res.sdk_codes) == list(BACKSTOP_SEQUENCE)  # backstops.py order, no reduced mode
    assert res.detail == "safety settings applied: sensitivity 3, payload 0.95 kg at (0, 0, 60) mm"
    assert res.before is not None and res.before.backstops_match is False
    assert res.after is not None and res.after.backstops_match is True
    assert (res.after.collision_sensitivity, res.after.tcp_load_kg) == (3, 0.95)
    assert res.after.tcp_load_cog_mm == [0.0, 0.0, 60.0]
    # The driver config handed to the monitor is the hardware package's ArmConfig mapping.
    _, driver_cfg, _ = factory.monitors["grip"].maintenance_calls[-1]
    assert (driver_cfg.arm_id, driver_cfg.ip, driver_cfg.gripper) == (
        "grip",
        "192.168.1.201",
        "xarm_g2",
    )
    assert (driver_cfg.tcp_load_kg, tuple(driver_cfg.tcp_load_cog_mm)) == (0.95, (0.0, 0.0, 60.0))
    assert driver_cfg.collision_sensitivity == 3 and driver_cfg.reduced_tcp_boundary_mm is None
    assert driver_cfg.expected_sn is None
    # Telemetry now reports the match on the Manipulation Arm only.
    msg = _telemetry(client, lambda m: _hw_arm(m, "grip")["backstops_match"] is True)
    assert _hw_arm(msg, "view")["backstops_match"] is False


def test_backstop_sequence_pinned_against_the_hardware_package():
    hw = pytest.importorskip("apollo_mavis_v2_hardware")
    cfg = hw.XArmDriverConfig(arm_id="grip", ip="192.168.1.201", gripper="xarm_g2")
    assert hw.expected_backstop_sequence(cfg) == BACKSTOP_SEQUENCE
    boxed = cfg.model_copy(update={"reduced_tcp_boundary_mm": (500, -500, 500, -500, 600, 0)})
    from conftest import fake_backstop_sequence

    assert hw.expected_backstop_sequence(boxed) == fake_backstop_sequence(boxed)


def test_recover_without_a_session_is_409(client):
    r = client.post(URL.format("grip"), json={"op": "recover"})
    assert r.status_code == 409 and r.json()["detail"] == "no hardware session - use clear_errors"


def test_monitor_not_connected_or_busy_is_409(client, factory):
    view = factory.monitors["view"]
    view.forced_status = "error"
    view.detail_text = "connect failed: timeout"
    try:
        r = client.post(URL.format("view"), json={"op": "clear_errors"})
        assert r.status_code == 409 and "monitor error: connect failed" in r.json()["detail"]
    finally:
        view.forced_status = None
        view.detail_text = ""
    view.maintenance_busy = True
    try:
        r = client.post(URL.format("view"), json={"op": "clear_errors"})
        assert r.status_code == 409 and "already running" in r.json()["detail"]
    finally:
        view.maintenance_busy = False
    # A failed op is still a 200 with ok=false (the detail is the operator's message).
    view.scripted_outcome = FakeMaintenanceOutcome(
        "view",
        "clear_errors",
        False,
        "controller error 19: End Effector Communication Error re-latched right after clearing",
        {"clean_error": 0, "clean_warn": 0},
    )
    try:
        r = client.post(URL.format("view"), json={"op": "clear_errors"})
        assert r.status_code == 200 and r.json()["ok"] is False
        assert r.json()["detail"].endswith("re-latched right after clearing")
        assert r.json()["before"] is None and r.json()["after"] is None
    finally:
        view.scripted_outcome = None


# -- home_rail: THE ONE motion op (phase-09c; monitor path, twin-gated) ----------------------
def test_home_rail_dry_run_returns_the_sweep_verdict_and_writes_nothing(client, factory):
    grip = factory.monitors["grip"]
    assert grip.sample.rail_present and not grip.sample.rail_homed  # as found in the lab
    r = client.post(URL.format("grip"), json={"op": "home_rail", "dry_run": True})
    assert r.status_code == 200, r.text
    res = ArmMaintenanceResult.model_validate(r.json())
    assert (res.arm_id, res.op, res.path, res.ok) == ("grip", "home_rail", "monitor", True)
    assert res.sdk_codes == {} and res.after is None  # zero writes
    assert res.before is not None and res.before.rail_homed is False
    v = res.rail_sweep
    assert v is not None and v.clear
    assert (v.scene_id, v.inflation_m, v.step_m) == ("mavis_v2", 0.025, 0.005)
    assert v.q_checked == pytest.approx(list(grip.sample.q)) and v.sample_seq == grip.sample.seq
    assert v.assumptions == ["view rail unknown - used fallback 0.00 m"]
    assert v.min_clearance_pair == ["table", "view_microphone"]
    assert res.detail.startswith("rail sweep clear")
    assert res.detail.endswith("(dry run, nothing written)")
    assert grip.homing_calls == [] and all(c[0] != "home_rail" for c in grip.maintenance_calls)
    assert grip.sample.rail_homed is False  # untouched


def test_home_rail_blocked_sweep_without_a_plan_is_refused_with_zero_writes(client, rt, factory):
    """Phase-09d: a blocked sweep is refused ONLY when no rail-safe pre-positioning
    path exists (the position-agnostic check rejects every candidate) - status
    ``refused``, ``ok`` false, nothing written, no job started."""
    grip = factory.monitors["grip"]
    original = grip.sample
    grip.sample = replace(original, q=REACH_SIDE_Q, seq=original.seq + 1)
    try:
        for body in ({"op": "home_rail", "dry_run": True}, {"op": "home_rail"}):
            r = client.post(URL.format("grip"), json=body)  # dry run, then the REAL op
            assert r.status_code == 200, r.text
            res = ArmMaintenanceResult.model_validate(r.json())
            assert res.status == "refused" and res.job_id is None
            assert res.ok is False and res.sdk_codes == {} and res.after is None
            assert res.rail_sweep is not None and not res.rail_sweep.clear
            assert res.rail_sweep.first_blocked_m == 0.0
            assert "grip_rail_base" in res.rail_sweep.first_blocked_pair
            plan = res.rail_sweep.pre_position
            assert plan is not None and plan.needed and not plan.clear
            assert plan.waypoints == 0 and plan.target_q == [] and plan.source == "current"
            assert plan.detail.startswith("no rail-safe pre-positioning path: keyframe:")
            assert "home:" in plan.detail and plan.detail.endswith(
                "fold the arm toward the factory zero posture in xArm Studio (joints 2-7 near 0) "
                "and retry"
            )
            assert res.detail.startswith("home_rail refused: home_rail refused: rail sweep blocked")
            assert grip.homing_calls == []  # the monitor never saw the op
            assert all(c[0] != "home_rail" for c in grip.maintenance_calls)
        assert not rt.rail_homing.active and rt.rail_homing.job("grip") is None
        # the refused real op is the arm's last result; the dry run is not
        assert client.get(URL.format("grip") + "/last").json()["status"] == "refused"
    finally:
        grip.sample = original


def test_home_rail_dry_run_with_a_plannable_posture_offers_the_pre_position_plan(
    client, rt, factory
):
    """The table-dive posture is blocked at rail 0 m but a straight joint path to the
    keyframe posture is clear for EVERY rail position: the dry run says so
    (``needed`` + ``clear``, waypoints, ~duration at 10 %) and writes nothing."""
    grip = factory.monitors["grip"]
    original = grip.sample
    grip.sample = replace(original, q=TABLE_DIVE_Q, seq=original.seq + 1)
    try:
        r = client.post(URL.format("grip"), json={"op": "home_rail", "dry_run": True})
        assert r.status_code == 200, r.text
        res = ArmMaintenanceResult.model_validate(r.json())
        assert (res.status, res.ok, res.job_id, res.sdk_codes) == ("done", True, None, {})
        v = res.rail_sweep
        assert v is not None and not v.clear and "table" in v.first_blocked_pair
        plan = v.pre_position
        assert plan is not None and plan.needed and plan.clear and plan.source == "keyframe"
        assert plan.target_q == pytest.approx([PI, 0, 0, 0, 0, 0, 0])
        assert plan.waypoints >= 2 and plan.checked_rail_positions == 131
        # at the DRIVER's caps x 0.1 (the servo stream's Cartesian bound over the lever arms
        # binds, not the host slew's 4 s). Halved from ~71 s on 2026-09-07 when
        # `max_cart_step_m` went 0.002 -> 0.004 m/tick: 0.4 mm/tick at scale 0.1 over
        # sum|dq| lever = 1.425 m -> ~36 s.
        assert 30.0 < plan.duration_s < 45.0
        assert plan.detail.startswith("the arm first moves along a planned path (")
        assert res.detail.endswith("(dry run, nothing written)")
        assert "then the rail homes and the arm holds that posture" in res.detail
        assert grip.homing_calls == [] and not rt.rail_homing.active
        assert rt.rail_homing.job("grip") is None  # a dry run never starts the job
    finally:
        grip.sample = original


def test_home_rail_refusals_are_409_before_any_write(client, factory):
    grip, view = factory.monitors["grip"], factory.monitors["view"]
    original = grip.sample
    # controller error latched -> clear errors first
    grip.sample = replace(original, error_code=24)
    try:
        r = client.post(URL.format("grip"), json={"op": "home_rail", "dry_run": True})
        assert r.status_code == 409 and "controller error 24" in r.json()["detail"]
        assert "clear errors first" in r.json()["detail"]
        # no linear track on this arm
        grip.sample = replace(original, rail_present=False)
        r = client.post(URL.format("grip"), json={"op": "home_rail", "dry_run": True})
        assert r.status_code == 409 and "no linear track detected" in r.json()["detail"]
    finally:
        grip.sample = original
    # another op running on the arm (the homing itself, typically)
    grip.maintenance_busy = True
    try:
        r = client.post(URL.format("grip"), json={"op": "home_rail", "dry_run": True})
        assert r.status_code == 409 and "already running" in r.json()["detail"]
        # ... and no hardware session may start while the carriage moves
        r = client.post("/api/session", json=HW_SPEC)
        assert r.status_code == 409, r.text
        assert r.json()["detail"].startswith("rail homing in progress on the Manipulation Arm")
    finally:
        grip.maintenance_busy = False
    # a homing in flight on the OTHER arm: its monitor publishes nothing while the
    # carriage travels, so its sample still shows the pre-homing carriage - the sweep
    # would be judged against a wrong posture (two carriages moving, unverified geometry)
    view.maintenance_busy = True
    try:
        r = client.post(URL.format("grip"), json={"op": "home_rail", "dry_run": True})
        assert r.status_code == 409, r.text
        assert r.json()["detail"] == (
            "home_rail refused: rail homing in progress on the Perception Arm - wait for it "
            "to finish"
        )
    finally:
        view.maintenance_busy = False
    # the target's own sample is stale: the sweep must use the CURRENT posture
    grip.forced_status = "stale"
    try:
        r = client.post(URL.format("grip"), json={"op": "home_rail", "dry_run": True})
        assert r.status_code == 409, r.text
        assert "sample of 'grip' is stale" in r.json()["detail"]
        assert "retry when it reads running" in r.json()["detail"]
    finally:
        grip.forced_status = None
    # the monitor not connected
    view.forced_status = "error"
    try:
        r = client.post(URL.format("view"), json={"op": "home_rail", "dry_run": True})
        assert r.status_code == 409 and "monitor error" in r.json()["detail"]
    finally:
        view.forced_status = None
    assert grip.homing_calls == [] and view.homing_calls == []


def test_home_rail_records_a_non_live_other_arm_monitor_as_an_assumption(client, factory):
    """The other arm is still posed from its last sample when its monitor is not live
    (stale / paused / error), but the verdict says so - the HomeRailSheet shows it."""
    grip, view = factory.monitors["grip"], factory.monitors["view"]
    view.forced_status = "stale"
    try:
        r = client.post(URL.format("grip"), json={"op": "home_rail", "dry_run": True})
        assert r.status_code == 200, r.text
        res = ArmMaintenanceResult.model_validate(r.json())
        assert res.ok and res.rail_sweep is not None and res.rail_sweep.clear
        assert res.rail_sweep.assumptions == [
            "view rail unknown - used fallback 0.00 m",
            "view: monitor stale - posed from its last sample, which may not be its current "
            "posture",
        ]
    finally:
        view.forced_status = None
    assert grip.homing_calls == [] and res.sdk_codes == {}


def _hardware_cells(recovery_latency_s: float = 0.0):
    """``workcell_factory`` seam + the cells it built (one per POST)."""
    cells: list[HardwareFakeWorkcell] = []

    def workcell_factory(session_cfg, driver_factory):
        cell = HardwareFakeWorkcell(
            {
                a.id: FakeArm(a.id, has_rail=True, q0=GRIP_Q0 if a.id == "grip" else VIEW_Q0)
                for a in session_cfg.arms
            },
            kind="hardware",
            recovery_latency_s=recovery_latency_s,
        )
        cells.append(cell)
        return cell

    return workcell_factory, cells


def test_fake_full_chain_unhomed_409_then_home_rail_then_session_running(client, rt, factory):
    """Phase-09c acceptance over fakes: unhomed -> POST /api/session 409 "rail not
    homed" -> home_rail on BOTH arms (sweep-clear postures: the synchronous 09c
    monitor path, exact write set, judged from the registers; ``pre_position.needed``
    false) -> the session comes up to RUNNING with speed_scale applied, the
    monitor paused and, after DELETE, resumed."""
    grip, view = factory.monitors["grip"], factory.monitors["view"]
    view.sample = replace(view.sample, error_code=0, warn_code=0)  # C19 cleared (09b op)
    assert not grip.sample.rail_homed and not view.sample.rail_homed
    r = client.post("/api/session", json=HW_SPEC)
    assert r.status_code == 409, r.text
    assert r.json()["detail"].startswith("Manipulation Arm: rail not homed - home it from the")
    assert rt.manager.session is None and not rt.hardware_monitor.paused

    r = client.post(URL.format("grip"), json={"op": "home_rail"})
    assert r.status_code == 200, r.text
    res = ArmMaintenanceResult.model_validate(r.json())
    assert (res.path, res.ok, res.status, res.job_id) == ("monitor", True, "done", None)
    assert list(res.sdk_codes) == HOME_RAIL_WRITES  # never motion_enable
    assert res.detail.startswith("rail homed: carriage at 0.000 m (register 0 mm), track enabled")
    assert res.rail_sweep is not None and res.rail_sweep.clear
    plan = res.rail_sweep.pre_position
    assert plan is not None and not plan.needed and plan.clear and plan.source == "current"
    assert plan.target_q == pytest.approx(res.rail_sweep.q_checked) and plan.waypoints == 0
    assert res.after is not None and res.after.rail_homed and res.after.rail_enabled
    assert res.after.rail_pos_m == 0.0
    assert grip.homing_calls == [tuple(res.rail_sweep.q_checked)]  # expected_q hand-off
    op, driver_cfg, timeout_s = grip.maintenance_calls[-1]
    assert op == "home_rail" and timeout_s == 45.0  # D3: the long REST budget
    assert driver_cfg.rail_speed_mm_s == 50 and driver_cfg.arm_id == "grip"
    msg = _telemetry(client, lambda m: _hw_arm(m, "grip")["rail_homed"] is True)
    assert _hw_arm(msg, "grip")["rail_pos_m"] == 0.0
    assert _hw_arm(msg, "grip")["maintenance"] is None  # no async job ran
    # the synchronous result is the arm's "last" too
    last = ArmMaintenanceResult.model_validate(client.get(URL.format("grip") + "/last").json())
    assert last.detail == res.detail and last.status == "done"
    assert client.get(URL.format("view") + "/last").status_code == 404  # nothing yet
    assert client.get(URL.format("arm9") + "/last").status_code == 404

    r = client.post("/api/session", json=HW_SPEC)
    assert r.status_code == 409 and r.json()["detail"].startswith("Perception Arm: rail not homed")
    r = client.post(URL.format("view"), json={"op": "home_rail"})
    assert r.status_code == 200 and r.json()["ok"] is True, r.text
    assert view.homing_calls == [tuple(view.sample.q)] and view.sample.rail_homed

    workcell_factory, cells = _hardware_cells()
    rt.manager.workcell_factory = workcell_factory
    try:
        r = client.post("/api/session", json=HW_SPEC)
        assert r.status_code == 200, r.text
        info = r.json()
        assert (info["kind"], info["speed_scale"], info["arms"]) == (
            "hardware",
            0.1,
            ["grip", "view"],
        )
        assert info["streams"] == []  # no cameras configured here -> nothing adopted
        assert _wait(lambda: rt.manager.state is SessionState.RUNNING)
        assert rt.hardware_monitor.paused and grip.calls[-1] == "join"
        assert len(cells) == 1 and cells[0].started and cells[0].bringup_calls == [60.0]
        assert set(cells[0].arms) == {"grip", "view"}  # both arms (phase-09d)
        assert rt.manager.session.loop.cfg.dq_max_rad == pytest.approx(0.004)
        # home_rail is session-less only
        r = client.post(URL.format("grip"), json={"op": "home_rail", "dry_run": True})
        assert r.status_code == 409 and r.json()["detail"].endswith("end the session first")
        msg = _telemetry(client, lambda m: m["session"]["state"] == "running")
        assert msg["session"]["bringup"] is None and msg["hardware_monitor"]["paused"] is True
    finally:
        rt.manager.workcell_factory = None
        assert client.delete("/api/session").status_code == 204
    assert cells[0].stop_calls == 1 and rt.manager.session is None
    assert _wait(lambda: not rt.hardware_monitor.paused) and grip.calls[-1] == "start"


# -- session path (a hardware session owns the boxes) ----------------------------------------
@pytest.fixture()
def hw_session(rt, client, factory):
    """A hardware session brought up by the REAL ``POST /api/session`` path over
    ``HardwareFakeWorkcell`` (``workcell_factory`` seam) + a real ControlLoop with
    the SafetyGate on the mavis_v2 twin; the fake monitor's Manipulation Arm reads
    homed + enabled (the phase-09c refusal matrix requires it)."""
    grip, view = factory.monitors["grip"], factory.monitors["view"]
    grip.sample = replace(
        grip.sample, rail_homed=True, rail_enabled=True, rail_pos_m=0.0, error_code=0
    )
    view.sample = replace(
        view.sample, rail_homed=True, rail_enabled=True, rail_pos_m=0.0, error_code=0, warn_code=0
    )
    workcell_factory, cells = _hardware_cells(recovery_latency_s=0.05)
    rt.manager.workcell_factory = workcell_factory
    r = client.post("/api/session", json=HW_SPEC)
    assert r.status_code == 200, r.text
    session = rt.manager.session
    cell, loop = cells[0], session.loop
    assert _wait(lambda: rt.manager.state is SessionState.RUNNING)
    assert rt.hardware_monitor.paused  # the monitor released the boxes
    yield cell, loop, session
    rt.manager.workcell_factory = None
    assert client.delete("/api/session").status_code == 204
    assert rt.manager.session is None
    assert _wait(lambda: not rt.hardware_monitor.paused)


def test_session_path_apply_backstops_409_and_arm_outside_session_409(client, hw_session):
    r = client.post(URL.format("grip"), json={"op": "apply_backstops"})
    assert r.status_code == 409 and "hardware session" in r.json()["detail"]
    r = client.post(URL.format("grip"), json={"op": "home_rail"})
    assert r.status_code == 409 and "end the session first" in r.json()["detail"]
    r = client.post(URL.format("arm9"), json={"op": "recover"})
    assert r.status_code == 404


def test_session_fault_recover_and_state_machine(client, rt, hw_session):
    cell, loop, session = hw_session
    assert rt.manager.state is SessionState.RUNNING
    cell.fault("grip", 24)  # the driver latched C24
    assert _wait(lambda: rt.manager.state is SessionState.FAULT)
    assert session.fault_detail.startswith("grip: controller error 24")
    msg = _telemetry(client, lambda m: m["session"]["state"] == "fault")
    grip = next(a for a in msg["arms"] if a["arm_id"] == "grip")
    assert grip["error_code"] == 24 and grip["recovering"] is False
    assert grip["fault_detail"].startswith("controller error 24")
    assert loop._senders["grip"].paused

    r = client.post(URL.format("grip"), json={"op": "recover"})  # "Clear errors & resume"
    assert r.status_code == 200, r.text
    res = ArmMaintenanceResult.model_validate(r.json())
    assert (res.path, res.ok) == ("session", True)
    assert res.detail.startswith("recovered from controller error 24")
    assert res.before is None and res.after is None and res.sdk_codes == {}
    assert cell.recovery_requests == ["grip"]
    # No input is held -> RECOVERING is left within a tick; the loop re-seeded from measured.
    assert _wait(lambda: rt.manager.state is SessionState.RUNNING)
    assert session.fault_detail == "" and not loop._senders["grip"].paused
    assert loop.recoveries == 1
    msg = _telemetry(client, lambda m: m["session"]["state"] == "running")
    grip = next(a for a in msg["arms"] if a["arm_id"] == "grip")
    assert grip["fault_detail"] == "" and grip["recovering"] is False and grip["error_code"] == 0

    # clear_errors inside a session is the same driver recovery (path "session").
    r = client.post(URL.format("grip"), json={"op": "clear_errors"})
    assert r.status_code == 200 and r.json()["path"] == "session" and r.json()["ok"] is True
    assert r.json()["detail"] == "re-seeded from the measured position"
    assert _wait(lambda: rt.manager.state is SessionState.RUNNING and loop.recoveries == 2)

    # The e-stop is still engaged: the driver latches, the session stays in FAULT.
    cell.fault("grip", 1)
    assert _wait(lambda: rt.manager.state is SessionState.FAULT)
    cell.latch_next["grip"] = "motion_enable failed (release the physical e-stop?)"
    r = client.post(URL.format("grip"), json={"op": "recover"})
    assert r.status_code == 200 and r.json()["ok"] is False and r.json()["path"] == "session"
    assert r.json()["detail"].startswith("controller error 1")
    assert r.json()["detail"].endswith("motion_enable failed (release the physical e-stop?)")
    time.sleep(0.1)
    assert rt.manager.state is SessionState.FAULT
    msg = _telemetry(client, lambda m: m["session"]["state"] == "fault")
    grip = next(a for a in msg["arms"] if a["arm_id"] == "grip")
    assert "motion_enable failed" in grip["fault_detail"]
    # E-stop released: the next click recovers.
    r = client.post(URL.format("grip"), json={"op": "recover"})
    assert r.status_code == 200 and r.json()["ok"] is True
    assert _wait(lambda: rt.manager.state is SessionState.RUNNING)


def test_session_path_skips_an_auto_recovery_result_that_lands_first(client, rt, hw_session):
    """The driver bumps ``RecoveryResult.seq`` for its own auto recoveries too. An
    auto sequence already running when the operator clicks completes first; its
    result (``user_initiated=False``, here a budget latch) must NOT be reported
    as the operator's outcome - the reply describes the user recovery."""
    from fakes import RecoveryResult

    cell, _loop, _session = hw_session
    cell.fault("grip", 24)
    assert _wait(lambda: rt.manager.state is SessionState.FAULT)
    real_request = cell.request_recovery

    def request_with_auto_first(arm_id: str) -> None:
        # the monitor thread finishes its auto sequence (latched: budget) first ...
        seq = cell._seq.get(arm_id, 0) + 1
        cell._seq[arm_id] = seq
        cell._results[arm_id] = RecoveryResult(
            seq, False, 24, "recovery budget exhausted (3 in 30 s)", False, cell._clock()
        )
        real_request(arm_id)  # ... then runs the operator's (recovery_latency_s later)

    cell.request_recovery = request_with_auto_first  # type: ignore[method-assign]
    r = client.post(URL.format("grip"), json={"op": "recover"})
    assert r.status_code == 200, r.text
    res = ArmMaintenanceResult.model_validate(r.json())
    assert (res.path, res.ok) == ("session", True)  # the USER outcome, not the auto latch
    assert res.detail.startswith("recovered from controller error 24")
    assert cell.recovery_result("grip").user_initiated is True
    assert _wait(lambda: rt.manager.state is SessionState.RUNNING)


def test_session_path_times_out_when_the_driver_never_answers(client, rt, hw_session, monkeypatch):
    cell, _loop, _session = hw_session
    cell._complete_recovery = lambda arm_id: None  # the monitor thread never finishes
    monkeypatch.setattr(
        "apollo_mavis_v2_runtime.devices.hardware_monitor.MAINTENANCE_TIMEOUT_S", 0.2
    )
    monkeypatch.setattr("apollo_mavis_v2_runtime.runtime.MAINTENANCE_TIMEOUT_S", 0.2)
    r = client.post(URL.format("grip"), json={"op": "recover"})
    assert r.status_code == 200 and r.json()["ok"] is False
    assert (
        r.json()["detail"] == "recover timed out after 0.2 s (no recovery result from the driver)"
    )


def test_session_path_without_a_recovery_channel_is_409(client, rt, hw_session):
    cell, _loop, session = hw_session
    from apollo_mavis_v2_core.testing import FakeWorkcell

    plain = FakeWorkcell({"grip": FakeArm("grip", has_rail=True)}, {}, kind="hardware")
    plain.start()
    session.workcell = plain  # a workcell without request_recovery / recovery_result
    try:
        r = client.post(URL.format("grip"), json={"op": "recover"})
        assert r.status_code == 409 and "no recovery channel" in r.json()["detail"]
    finally:
        session.workcell = cell
