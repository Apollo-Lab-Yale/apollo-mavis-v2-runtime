"""``POST /api/hardware/arms/{arm_id}/maintenance`` (phase-09b; 04-runtime §13.1 /
§15) over a Runtime with a hardware workcell config, a FAKE read-only monitor and,
for the session path, a hand-installed hardware session over
``tests/fakes.EventFakeWorkcell`` - no control box anywhere.

Three routes: the monitor path (no session: clear_errors / apply_backstops on
the arm monitor's thread, ``recover`` -> 409), the session path (a hardware
session owns the boxes: clear_errors / recover = the driver's user recovery,
apply_backstops -> 409) and the 409 matrix (unknown arm 404, monitor off /
paused / erroring, busy). Plus the session state machine FAULT -> RECOVERING ->
RUNNING driven by the scripted driver events, and the telemetry fields
``hardware_monitor.arms[*]`` read-backs / ``backstops_match`` /
``maintenance_busy`` and ``arms[*].fault_detail`` / ``recovering``."""

from __future__ import annotations

import time

import pytest
from apollo_mavis_v2_core import WorkcellConfig
from apollo_mavis_v2_core.protocol import ArmMaintenanceResult, SessionSpec
from apollo_mavis_v2_core.testing import FakeArm
from conftest import (
    BACKSTOP_SEQUENCE,
    FakeMaintenanceOutcome,
    FakeMonitorFactory,
    FakeMonitorSample,
    make_runtime_config,
)
from fakes import EventFakeWorkcell
from starlette.testclient import TestClient

from apollo_mavis_v2_runtime.config import (
    ControlConfig,
    HardwareMonitorConfig,
    HardwareProbeConfig,
    MicrophoneConfig,
    TwinOverlayConfig,
)
from apollo_mavis_v2_runtime.control.loop import ControlLoop
from apollo_mavis_v2_runtime.runtime import Runtime
from apollo_mavis_v2_runtime.safety.gate import NullGate
from apollo_mavis_v2_runtime.safety.supervisor import SafetySupervisor
from apollo_mavis_v2_runtime.safety.watchdog import InputWatchdog
from apollo_mavis_v2_runtime.server.app import create_app
from apollo_mavis_v2_runtime.session.manager import ActiveSession
from apollo_mavis_v2_runtime.session.types import SessionState

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
        "grip": FakeMonitorSample(
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
    assert client.post(URL.format("grip"), json={"op": "home_rail"}).status_code == 422
    assert client.post(URL.format("grip"), json={}).status_code == 422


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


# -- session path (a hardware session owns the boxes) ----------------------------------------
@pytest.fixture()
def hw_session(rt, client):
    """A hand-installed hardware session (POST /api/session kind=hardware is still
    409 until phase-09): EventFakeWorkcell + a real ControlLoop, wired to the
    manager exactly like ``_bringup_*`` does."""
    cell = EventFakeWorkcell(
        {"grip": FakeArm("grip", has_rail=True), "view": FakeArm("view", has_rail=True)},
        kind="hardware",
        recovery_latency_s=0.05,
    )
    cell.start()
    loop = ControlLoop(
        cell,
        ControlConfig(),
        rt.bus,
        SafetySupervisor(NullGate(), InputWatchdog()),
        ["grip"],
        profile_store=rt.profile_store,
        workcell_kind="hardware",
    )
    spec = SessionSpec(
        mode="teleop",
        kind="hardware",
        arms=["grip"],
        frames={"grip": "arm_base:grip"},
        digital_twin_scene="mavis_v2",
    )
    session = ActiveSession(
        session_id="hw-test",
        spec=spec,
        state=SessionState.RUNNING,
        workcell=cell,
        loop=loop,
        supervisor=loop.supervisor,
        twin=None,
        render_service=None,
    )
    rt.manager.attach_fault_state(session)
    loop.start()
    rt.manager.session = session
    assert _wait(lambda: rt.hardware_monitor.paused)  # the monitor released the boxes
    yield cell, loop, session
    assert client.delete("/api/session").status_code == 204
    assert rt.manager.session is None
    assert _wait(lambda: not rt.hardware_monitor.paused)


def test_session_path_apply_backstops_409_and_arm_outside_session_409(client, hw_session):
    r = client.post(URL.format("grip"), json={"op": "apply_backstops"})
    assert r.status_code == 409 and "hardware session" in r.json()["detail"]
    r = client.post(URL.format("view"), json={"op": "recover"})  # configured, not in the session
    assert r.status_code == 409 and "not part of the session" in r.json()["detail"]
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
