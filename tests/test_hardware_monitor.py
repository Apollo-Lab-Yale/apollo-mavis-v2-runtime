"""HardwareStateMonitor (phase-09a): config defaults, inert modes, factory kwargs,
pause = disconnect every box / resume = reconnect, telemetry rows."""

from __future__ import annotations

import time

import pytest
from apollo_mavis_v2_core import WorkcellConfig
from apollo_mavis_v2_core.protocol import HardwareMonitorTelemetry
from conftest import FakeArmMonitor, FakeMonitorFactory, FakeMonitorSample
from pydantic import ValidationError

from apollo_mavis_v2_runtime.config import (
    HardwareMonitorConfig,
    RuntimeConfig,
    TwinOverlayConfig,
)
from apollo_mavis_v2_runtime.devices.hardware_monitor import (
    HardwareStateMonitor,
    default_monitor_factory,
    sample_to_telemetry,
)

HW = WorkcellConfig.model_validate({
    "kind": "hardware",
    "digital_twin_scene": "mavis_v2",
    "arms": [
        {"id": "grip", "ip": "192.168.1.201", "base_in_world": {}, "gripper": "xarm_g2"},
        {"id": "view", "ip": "192.168.2.219", "base_in_world": {}, "gripper": "none",
         "microphone": True, "expect_rail": "yes"},
    ],
    "cameras": [],
    "safety": {"enabled": True},
})


def _wait(pred, timeout_s: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


# -- config -------------------------------------------------------------------------------
def test_config_defaults_match_contract():
    mon = HardwareMonitorConfig()
    assert (mon.enabled, mon.poll_hz, mon.stale_s, mon.reconnect_s) == (True, 10.0, 0.5, 2.0)
    ov = TwinOverlayConfig()
    assert (ov.enabled, ov.fps, ov.alpha) == (True, 12.0, 0.5)
    assert ov.tint_rgb == (255, 235, 140) and ov.edge_rgb == (255, 220, 60)
    assert ov.env_outline is True and ov.env_rgb == (90, 200, 250)
    assert ov.stale_tint_rgb == (170, 170, 170)
    assert ov.joint1_offset_rad == 0.0 and ov.rail_flip is False
    assert ov.rail_fallback_m == {"grip": 0.65, "view": 0.0} and ov.stream_suffix == "_align"
    rt = RuntimeConfig()
    assert rt.hardware_monitor == HardwareMonitorConfig() and rt.twin_overlay == TwinOverlayConfig()
    # YAML lists become the RGB tuples.
    assert TwinOverlayConfig.model_validate({"tint_rgb": [1, 2, 3]}).tint_rgb == (1, 2, 3)
    # phase-09c: hardware_session block defaults (D2 / D4) and the rail_flip alias.
    hs = rt.hardware_session
    assert hs.default_speed_scale == 1.0  # 100 % since 2026-09-08 evening (50 % was 09-07)
    assert not hasattr(hs, "default_arms")  # 09d: gone
    assert RuntimeConfig.model_validate({"hardware_session": {"default_arms": ["grip"]}})
    assert (hs.rail_flip, hs.home_rail_inflation_m, hs.home_rail_step_m) == (False, 0.025, 0.005)
    assert hs.bringup_timeout_s == 60.0
    aliased = RuntimeConfig.model_validate({"twin_overlay": {"rail_flip": True}})
    assert aliased.hardware_session.rail_flip is True and aliased.twin_overlay.rail_flip is True
    new = RuntimeConfig.model_validate({"hardware_session": {"rail_flip": True}})
    assert new.twin_overlay.rail_flip is True  # the overlay reads one convention
    with pytest.raises(ValidationError):
        RuntimeConfig.model_validate({"hardware_session": {"default_speed_scale": 0}})


# -- inert modes ----------------------------------------------------------------------------
def test_inert_without_workcell_disabled_or_hardware_package(monkeypatch):
    none = HardwareStateMonitor(HardwareMonitorConfig(), None)
    assert not none.enabled and "no hardware workcell" in none.detail
    none.start()
    assert none._thread is None and none.snapshot() == {} and none.error_code("grip") == 0
    assert none.telemetry() == HardwareMonitorTelemetry()  # the valid no-hardware block

    off = HardwareStateMonitor(HardwareMonitorConfig(enabled=False), HW)
    assert not off.enabled and "enabled: false" in off.detail
    off.start()
    rows = off.telemetry()
    assert rows.enabled is False and rows.paused is False and rows.overlays == []
    assert [a.arm_id for a in rows.arms] == ["grip", "view"]
    assert all(a.status == "off" and "disabled" in a.detail for a in rows.arms)

    def broken():
        raise ImportError("No module named 'apollo_mavis_v2_hardware'")

    import apollo_mavis_v2_runtime.devices.hardware_monitor as mod

    monkeypatch.setattr(mod, "default_monitor_factory", broken)
    missing = HardwareStateMonitor(HardwareMonitorConfig(), HW)  # no factory seam -> default
    assert not missing.enabled and "not importable" in missing.detail
    assert missing.status_of("view") == ("off", missing.detail)
    missing.start()
    assert missing._thread is None
    assert all(a.status == "off" for a in missing.telemetry().arms)


def test_default_factory_is_the_hardware_arm_state_monitor_when_importable():
    try:
        from apollo_mavis_v2_hardware import ArmStateMonitor
    except Exception:  # noqa: BLE001 - [hardware] extra absent
        return
    assert default_monitor_factory() is ArmStateMonitor


# -- construction ---------------------------------------------------------------------------
def test_factory_receives_config_and_arm_facts():
    factory = FakeMonitorFactory()
    cfg = HardwareMonitorConfig(poll_hz=7.0, stale_s=0.9, reconnect_s=1.5)
    mon = HardwareStateMonitor(cfg, HW, monitor_factory=factory)
    assert mon.enabled and mon.detail == "" and mon.arm_ids == ["grip", "view"]
    grip, view = factory.monitors["grip"], factory.monitors["view"]
    assert (grip.ip, grip.gripper, grip.expect_rail) == ("192.168.1.201", "xarm_g2", True)  # auto
    assert (view.ip, view.gripper, view.expect_rail) == ("192.168.2.219", "none", True)  # yes
    assert (grip.poll_hz, grip.stale_s, grip.reconnect_s) == (7.0, 0.9, 1.5)
    no_rail = HW.model_copy(update={
        "arms": [HW.arms[0].model_copy(update={"expect_rail": "no"})],
    })
    f2 = FakeMonitorFactory()
    HardwareStateMonitor(cfg, no_rail, monitor_factory=f2)
    assert f2.monitors["grip"].expect_rail is False
    assert grip.calls == []  # nothing connects before start()


# -- pause / resume -------------------------------------------------------------------------
def test_pause_disconnects_every_box_and_resume_reconnects():
    factory = FakeMonitorFactory()
    paused = {"on": False}
    mon = HardwareStateMonitor(
        HardwareMonitorConfig(), HW, paused=lambda: paused["on"],
        monitor_factory=factory, check_period_s=0.02,
    )
    mon.start()
    try:
        grip, view = factory.monitors["grip"], factory.monitors["view"]
        assert grip.calls == ["start"] and view.calls == ["start"]
        assert mon.paused is False and mon.status_of("grip") == ("running", "")
        paused["on"] = True  # a hardware session takes the boxes
        assert _wait(lambda: mon.paused)
        assert _wait(lambda: grip.calls == ["start", "disconnect"])
        assert view.calls == ["start", "disconnect"]
        assert grip.status == "paused" and mon.status_of("view")[0] == "paused"
        time.sleep(0.1)
        assert grip.calls.count("disconnect") == 1  # edge-triggered, not repeated
        block = mon.telemetry()
        assert block.enabled and block.paused and {a.status for a in block.arms} == {"paused"}
        paused["on"] = False  # session over
        assert _wait(lambda: grip.calls == ["start", "disconnect", "start"])
        assert view.calls[-1] == "start" and mon.paused is False
        assert mon.transitions == 2
    finally:
        mon.stop()
    assert grip.calls[-1] == "stop" and view.calls[-1] == "stop"
    assert grip.status == "off" and mon.status_of("grip") == ("off", "monitor not started")
    assert mon._thread is None
    mon.stop()  # idempotent


def test_start_while_already_paused_only_releases_and_sync_seams():
    factory = FakeMonitorFactory()
    paused = {"on": True}
    mon = HardwareStateMonitor(
        HardwareMonitorConfig(), HW, paused=lambda: paused["on"],
        monitor_factory=factory, check_period_s=5.0,  # supervisor effectively idle
    )
    mon.start()
    try:
        grip = factory.monitors["grip"]
        assert grip.calls == ["disconnect"] and mon.paused is True  # never connected
        paused["on"] = False  # the session / job released the boxes ...
        mon.resume()  # ... and hands back explicitly (phase-09 bring-up seam)
        assert grip.calls == ["disconnect", "start"] and mon.paused is False
        mon.pause()
        mon.pause()  # idempotent
        assert grip.calls == ["disconnect", "start", "disconnect"] and mon.paused is True
    finally:
        mon.stop()
    mon.resume()  # after stop: no reconnect
    assert grip.calls[-1] == "stop"


def test_resume_is_a_no_op_while_the_hand_over_predicate_still_holds():
    """``resume()`` re-applies the predicate instead of blindly reconnecting: a
    caller that never paused the monitor (a rail-homing job failing before its
    connect while a hardware create() owns the boxes) must not put a second SDK
    client on a box the session's drivers hold."""
    factory = FakeMonitorFactory()
    paused = {"on": True}
    mon = HardwareStateMonitor(
        HardwareMonitorConfig(), HW, paused=lambda: paused["on"],
        monitor_factory=factory, check_period_s=5.0,
    )
    mon.start()
    try:
        grip = factory.monitors["grip"]
        assert grip.calls == ["disconnect"] and mon.paused is True
        mon.resume()  # predicate still true -> nothing reconnects
        assert grip.calls == ["disconnect"] and mon.paused is True
        paused["on"] = False
        mon.resume()  # predicate released -> the explicit hand-back reconnects
        assert grip.calls == ["disconnect", "start"] and mon.paused is False
    finally:
        mon.stop()


def test_predicate_errors_do_not_kill_the_supervisor():
    factory = FakeMonitorFactory()

    def bad():
        raise RuntimeError("boom")

    mon = HardwareStateMonitor(
        HardwareMonitorConfig(), HW, paused=bad, monitor_factory=factory, check_period_s=0.01,
    )
    mon.start()
    try:
        time.sleep(0.05)
        assert mon._thread is not None and mon._thread.is_alive()
        assert factory.monitors["grip"].calls == ["start"]  # treated as not paused
    finally:
        mon.stop()


# -- samples -> telemetry ---------------------------------------------------------------------
def test_snapshot_error_code_and_telemetry_rows():
    now = time.monotonic()
    samples = {
        "grip": FakeMonitorSample(
            "grip", seq=7, t_mono=now, q=(3.14159, 0.1, -0.2, 0.3, 0.0, 0.5, 0.0),
            tcp_pose=(0.2, -0.05, 0.11, 3.14, 0.0, 0.0), gripper_open_frac=0.25,
            gripper_raw=21.0, rail_raw_mm=0.0,
        ),
        "view": FakeMonitorSample("view", seq=9, t_mono=now, error_code=19, warn_code=0),
    }
    factory = FakeMonitorFactory(samples)
    mon = HardwareStateMonitor(HardwareMonitorConfig(), HW, monitor_factory=factory)
    mon.start()
    try:
        factory.monitors["view"].detail_text = (
            "controller error 19: End Effector Communication Error"
        )
        snap = mon.snapshot()
        assert set(snap) == {"grip", "view"} and snap["view"].error_code == 19
        assert mon.error_code("view") == 19 and mon.error_code("grip") == 0
        assert mon.error_code("nope") == 0
        block = mon.telemetry()
        assert block.enabled is True and block.paused is False
        grip, view = block.arms
        assert grip.arm_id == "grip" and grip.status == "running" and grip.seq == 7
        assert grip.q == [3.14159, 0.1, -0.2, 0.3, 0.0, 0.5, 0.0]
        assert grip.tcp_pose == [0.2, -0.05, 0.11, 3.14, 0.0, 0.0]
        assert (grip.rail_present, grip.rail_homed, grip.rail_enabled) == (True, False, False)
        assert grip.rail_pos_m is None and grip.rail_raw_mm == 0.0  # not homed: raw only
        assert (grip.gripper_open_frac, grip.gripper_raw) == (0.25, 21.0)
        assert (grip.state, grip.mode, grip.error_code, grip.warn_code) == (4, 0, 0, 0)
        assert grip.age_s is not None and grip.age_s < 1.0
        assert view.error_code == 19 and view.detail.startswith("controller error 19")
        assert view.gripper_open_frac is None and view.q[0] == 3.141592653589793
        # a forced stale status flows through unchanged
        factory.monitors["grip"].forced_status = "stale"
        assert mon.status_of("grip")[0] == "stale"
        assert mon.telemetry().arms[0].status == "stale"
    finally:
        mon.stop()


def test_sample_to_telemetry_without_sample_keeps_defaults():
    row = sample_to_telemetry("grip", "connecting", "connecting to 192.168.1.201", None, None)
    assert row.arm_id == "grip" and row.status == "connecting" and row.q == []
    assert row.rail_pos_m is None and row.error_code == 0 and row.seq == 0


def test_fake_monitor_surface_matches_protocol():
    """The conftest fake exposes exactly what the runtime relies on."""
    fake = FakeArmMonitor("grip", "1.2.3.4")
    for name in ("start", "stop", "disconnect", "snapshot", "status", "detail", "age_s"):
        assert hasattr(fake, name)


# -- phase-09b: safety read-backs, backstops_match, maintenance channel ----------------------------
def test_backstops_match_tolerances():
    from apollo_mavis_v2_runtime.devices.hardware_monitor import (
        TCP_COG_MATCH_MM,
        TCP_LOAD_MATCH_KG,
        backstops_match,
    )

    grip = HW.arms[0].model_copy(update={
        "tcp_load_kg": 0.95, "tcp_load_cog_mm": (0.0, 0.0, 60.0), "collision_sensitivity": 3,
    })
    assert (TCP_LOAD_MATCH_KG, TCP_COG_MATCH_MM) == (0.05, 10.0)
    assert backstops_match(None, grip) is None
    assert backstops_match(FakeMonitorSample("grip"), grip) is None  # nothing read back yet
    ok = FakeMonitorSample(
        "grip", collision_sensitivity=3, tcp_load_kg=0.95, tcp_load_cog_mm=(0.0, 0.0, 60.0)
    )
    assert backstops_match(ok, grip) is True
    from dataclasses import replace

    assert backstops_match(replace(ok, tcp_load_kg=0.95 + 0.05), grip) is True  # at tolerance
    assert backstops_match(replace(ok, tcp_load_kg=0.95 + 0.0501), grip) is False
    assert backstops_match(replace(ok, tcp_load_cog_mm=(10.0, 0.0, 50.0)), grip) is True
    assert backstops_match(replace(ok, tcp_load_cog_mm=(0.0, 0.0, 49.9)), grip) is False
    assert backstops_match(replace(ok, collision_sensitivity=4), grip) is False
    assert backstops_match(replace(ok, tcp_load_cog_mm=()), grip) is False  # cog unreadable
    assert backstops_match(replace(ok, collision_sensitivity=None), grip) is None
    # 2026-09-11: the operator's requested level replaces the config as the expectation
    # (an obeyed set_collision_sensitivity is not "differs from config").
    assert backstops_match(ok, grip, expected_sensitivity=2) is False
    assert backstops_match(replace(ok, collision_sensitivity=2), grip, expected_sensitivity=2)
    assert backstops_match(replace(ok, collision_sensitivity=2), grip) is False
    assert backstops_match(replace(ok, collision_sensitivity=2), grip, 3) is False
    assert backstops_match(replace(ok, collision_sensitivity=None), grip, 2) is None
    # The lab as found (2026-09-04): 0 kg / sensitivity 1 on the Perception Arm.
    view = HW.arms[1].model_copy(update={"tcp_load_kg": 0.55, "tcp_load_cog_mm": (0.0, 0.0, 90.0)})
    found = FakeMonitorSample(
        "view", collision_sensitivity=1, tcp_load_kg=0.0, tcp_load_cog_mm=(0.0, 0.0, 0.0)
    )
    assert backstops_match(found, view) is False


def test_telemetry_rows_carry_read_backs_match_and_busy():
    now = time.monotonic()
    cfg = HW.model_copy(update={"arms": [
        HW.arms[0].model_copy(update={"tcp_load_kg": 0.95, "tcp_load_cog_mm": (0.0, 0.0, 60.0)}),
        HW.arms[1],
    ]})
    samples = {
        "grip": FakeMonitorSample(
            "grip", seq=2, t_mono=now, collision_sensitivity=3, tcp_load_kg=0.97,
            tcp_load_cog_mm=(1.0, -2.0, 65.0),
        ),
        "view": FakeMonitorSample("view", seq=1, t_mono=now),  # rich frame not read yet
    }
    factory = FakeMonitorFactory(samples)
    mon = HardwareStateMonitor(HardwareMonitorConfig(), cfg, monitor_factory=factory)
    mon.start()
    try:
        grip, view = mon.telemetry().arms
        assert (grip.collision_sensitivity, grip.tcp_load_kg) == (3, 0.97)
        assert grip.tcp_load_cog_mm == [1.0, -2.0, 65.0]
        assert grip.backstops_match is True and grip.maintenance_busy is False
        assert view.collision_sensitivity is None and view.tcp_load_kg is None
        assert view.tcp_load_cog_mm == [] and view.backstops_match is None
        factory.monitors["view"].maintenance_busy = True
        assert mon.telemetry().arms[1].maintenance_busy is True
        assert mon.maintenance_busy("view") is True and mon.maintenance_busy("nope") is False
        assert mon.telemetry().arms[1].maintenance is None  # no async job (phase-09d)
        # phase-09d: an attached job registry contributes busy + the progress block
        from apollo_mavis_v2_core.protocol import MaintenanceProgress

        class Jobs:
            def busy(self, arm_id):
                return arm_id == "grip"

            def progress(self, arm_id):
                if arm_id != "grip":
                    return None
                return MaintenanceProgress(
                    op="home_rail", job_id="j1", phase="positioning", detail="wp 1", progress=0.5
                )

        mon.jobs = Jobs()
        grip_row = mon.telemetry().arms[0]
        assert mon.maintenance_busy("grip") is True and grip_row.maintenance_busy is True
        assert grip_row.maintenance is not None and grip_row.maintenance.phase == "positioning"
        assert grip_row.maintenance.job_id == "j1" and mon.job_progress("view") is None
        mon.jobs = None
        assert mon.maintenance_busy("grip") is False
        # a row without a sample keeps the defaults but still reports the busy flag
        row = sample_to_telemetry("grip", "connecting", "", None, None, maintenance_busy=True)
        assert row.backstops_match is None and row.maintenance_busy is True
    finally:
        mon.stop()


def test_inert_monitor_reports_the_hand_over_flag_from_the_predicate():
    """``telemetry.hardware_monitor.paused`` is the UI's only "a hardware session
    exists" signal (05-ui §8.2): with the monitor switched off (or the hardware
    package missing) nothing is released, but the flag must still follow the
    runtime's predicate - otherwise the Cockpit loses its recover button."""
    paused = {"on": False}
    off = HardwareStateMonitor(
        HardwareMonitorConfig(enabled=False), HW, paused=lambda: paused["on"]
    )
    off.start()  # no-op while inert
    assert off._thread is None and off.paused is False
    assert off.telemetry().paused is False
    paused["on"] = True  # a hardware session owns the boxes
    assert off.paused is True and off.telemetry().paused is True  # immediate, no supervisor
    assert off.telemetry().enabled is False
    paused["on"] = False
    assert off.telemetry().paused is False

    def bad():
        raise RuntimeError("predicate exploded")

    broken = HardwareStateMonitor(HardwareMonitorConfig(enabled=False), HW, paused=bad)
    assert broken.paused is False  # predicate errors never poison the flag

    # Enabled: the flag is the APPLIED state (the supervisor's disconnect walk), unchanged.
    factory = FakeMonitorFactory()
    live = HardwareStateMonitor(
        HardwareMonitorConfig(), HW, paused=lambda: paused["on"], monitor_factory=factory,
        check_period_s=0.02,
    )
    assert live.paused is False
    paused["on"] = True
    assert live.paused is False  # nothing applied before start()
    live.start()
    try:
        assert live.paused is True  # start() applies the predicate synchronously
        assert factory.monitors["grip"].calls == ["disconnect"]
    finally:
        live.stop()
        paused["on"] = False


def test_maintenance_refusals_on_the_runtime_monitor():
    import pytest

    from apollo_mavis_v2_runtime.devices.hardware_monitor import MAINTENANCE_OPS
    from apollo_mavis_v2_runtime.errors import MaintenanceUnavailableError

    assert MAINTENANCE_OPS == (
        "clear_errors", "apply_backstops", "recover", "home_rail", "set_collision_sensitivity",
    )
    factory = FakeMonitorFactory()
    mon = HardwareStateMonitor(HardwareMonitorConfig(), HW, monitor_factory=factory)
    with pytest.raises(KeyError):
        mon.maintenance("arm9", "clear_errors")
    with pytest.raises(ValueError):
        mon.maintenance("grip", "go_home")
    with pytest.raises(MaintenanceUnavailableError, match="no hardware session - use clear_errors"):
        mon.maintenance("grip", "recover")
    with pytest.raises(MaintenanceUnavailableError, match="monitor off"):
        mon.maintenance("grip", "clear_errors")  # not started
    with pytest.raises(MaintenanceUnavailableError, match="monitor off"):
        mon.maintenance("grip", "home_rail")  # phase-09c: a real op now, same gate
    # set_collision_sensitivity validates the level BEFORE any gate (422 upstream).
    for bad in (None, 0, 4, 5, 2.5, True, "2"):
        with pytest.raises(ValueError, match="1, 2 or 3"):
            mon.maintenance("grip", "set_collision_sensitivity", collision_sensitivity=bad)
    with pytest.raises(MaintenanceUnavailableError, match="monitor off"):
        mon.maintenance("grip", "set_collision_sensitivity", collision_sensitivity=2)
    mon.start()
    try:
        grip = factory.monitors["grip"]
        grip.disconnect()  # paused (hand-over)
        with pytest.raises(MaintenanceUnavailableError, match="monitor paused"):
            mon.maintenance("grip", "clear_errors")
        grip.start()
        grip.forced_status = "connecting"
        with pytest.raises(MaintenanceUnavailableError, match="monitor connecting"):
            mon.maintenance("grip", "apply_backstops")
        grip.forced_status = "stale"  # connected, slow sample: the op may run
        res = mon.maintenance("grip", "clear_errors")
        assert res.ok and res.path == "monitor" and list(res.sdk_codes) == [
            "clean_error", "clean_warn",
        ]
        assert res.before is None and res.after is None  # never sampled
        grip.forced_status = None
        grip.maintenance_busy = True
        with pytest.raises(MaintenanceUnavailableError, match="already running"):
            mon.maintenance("grip", "clear_errors")
        grip.maintenance_busy = False
        assert grip.maintenance_calls[-1][0] == "clear_errors"
        # home_rail without a sweep checker (no twin scene / [sim] extra) is refused
        # BEFORE any write: the monitor never sees the op.
        with pytest.raises(MaintenanceUnavailableError, match="needs the digital twin"):
            mon.maintenance("grip", "home_rail")
        assert grip.maintenance_calls[-1][0] == "clear_errors" and grip.homing_calls == []
        # join(): the hand-over waits for the poll threads; a thread still inside the
        # SDK is reported by arm id.
        assert mon.join(0.1) == []
        grip.join_result = False
        assert mon.join(0.1) == ["grip"]
        grip.join_result = True
    finally:
        mon.stop()

    # Inert monitor (no hardware workcell): every op is 409, never a crash.
    none = HardwareStateMonitor(HardwareMonitorConfig(), None)
    with pytest.raises(KeyError):
        none.note_requested_sensitivity("grip", 2)
    assert none.requested_sensitivity("grip") is None
    none.forget_requested_sensitivity("grip")  # unknown arm: a no-op
    with pytest.raises(KeyError):
        none.maintenance("grip", "clear_errors")
    off = HardwareStateMonitor(HardwareMonitorConfig(enabled=False), HW)
    off.start()
    with pytest.raises(MaintenanceUnavailableError, match="monitor off"):
        off.maintenance("grip", "clear_errors")


def test_set_collision_sensitivity_on_the_runtime_monitor_and_the_requested_level():
    """2026-09-11: the monitor path writes once and judges by read-back; the level
    is remembered per arm (``backstops_match`` expects it, the row publishes it while
    paused) until a hand-over (driver connect) or ``apply_backstops`` forgets it."""
    from dataclasses import replace

    from apollo_mavis_v2_runtime.errors import MaintenanceUnavailableError

    now = time.monotonic()
    cfg = HW.model_copy(update={"arms": [
        HW.arms[0].model_copy(update={"tcp_load_kg": 0.95, "tcp_load_cog_mm": (0.0, 0.0, 60.0)}),
        HW.arms[1],
    ]})
    samples = {
        "grip": FakeMonitorSample(
            "grip", seq=2, t_mono=now, collision_sensitivity=3, tcp_load_kg=0.95,
            tcp_load_cog_mm=(0.0, 0.0, 60.0),
        ),
    }
    factory = FakeMonitorFactory(samples)
    paused = {"on": False}
    mon = HardwareStateMonitor(
        HardwareMonitorConfig(), cfg, lambda: paused["on"], monitor_factory=factory,
        driver_cfg_factory=lambda arm: arm, check_period_s=0.02,
    )
    mon.start()
    try:
        grip = factory.monitors["grip"]
        assert _wait(lambda: mon.status_of("grip")[0] == "running")
        res = mon.maintenance("grip", "set_collision_sensitivity", collision_sensitivity=2)
        assert res.ok and res.path == "monitor" and res.op == "set_collision_sensitivity"
        assert res.sdk_codes == {"set_collision_sensitivity": 0} and res.warnings == []
        assert res.collision_sensitivity == 2 and grip.sensitivity_calls == [2]
        assert grip.maintenance_calls[-1][0] == "set_collision_sensitivity"
        assert res.detail.startswith("collision sensitivity set to 2 (was 3; the config value 3")
        assert res.before is not None and res.before.collision_sensitivity == 3
        assert res.before.backstops_match is True  # judged against the level in force: config
        assert res.after is not None and res.after.collision_sensitivity == 2
        assert res.after.backstops_match is True  # judged against the level written
        assert mon.requested_sensitivity("grip") == 2 and mon.requested_sensitivity("view") is None
        row = next(r for r in mon.arm_telemetry() if r.arm_id == "grip")
        assert row.collision_sensitivity == 2 and row.backstops_match is True
        # A status echo (fault latched) is ok with the note in warnings; the read-back rules.
        grip.sensitivity_code = 2
        res = mon.maintenance("grip", "set_collision_sensitivity", collision_sensitivity=1)
        assert res.ok and res.sdk_codes == {"set_collision_sensitivity": 2}
        assert res.collision_sensitivity == 1 and mon.requested_sensitivity("grip") == 1
        assert "status echo" in res.detail
        # The box did not take the value: not ok, the read-back is what is reported.
        grip.sensitivity_code = 0
        grip.sensitivity_readback = 1
        res = mon.maintenance("grip", "set_collision_sensitivity", collision_sensitivity=3)
        assert not res.ok and res.collision_sensitivity == 1
        assert res.detail == "collision sensitivity still reads 1 after writing 3"
        assert mon.requested_sensitivity("grip") == 1  # a failed write changes nothing
        grip.sensitivity_readback = None
        # Paused (a driver connected: the config value is back): the level is forgotten ...
        paused["on"] = True
        assert _wait(lambda: mon.paused)
        assert mon.requested_sensitivity("grip") is None
        # ... and a session-path write recorded now is PUBLISHED on the paused row (the
        # stale sample reads 1; the payload alone is judged).
        mon.note_requested_sensitivity("grip", 2)
        row = next(r for r in mon.arm_telemetry() if r.arm_id == "grip")
        assert row.status == "paused" and row.collision_sensitivity == 2
        assert row.backstops_match is True
        with pytest.raises(MaintenanceUnavailableError, match="monitor paused"):
            mon.maintenance("grip", "set_collision_sensitivity", collision_sensitivity=3)
        paused["on"] = False
        assert _wait(lambda: not mon.paused)
        assert mon.requested_sensitivity("grip") == 2  # a resume alone keeps the override
        row = next(r for r in mon.arm_telemetry() if r.arm_id == "grip")
        assert row.collision_sensitivity == 1  # running again: the read-back is published
        assert row.backstops_match is False  # ... and 1 != the requested 2
        grip.sample = replace(grip.sample, collision_sensitivity=2, seq=grip.sample.seq + 1)
        row = next(r for r in mon.arm_telemetry() if r.arm_id == "grip")
        assert row.collision_sensitivity == 2 and row.backstops_match is True
        # apply_backstops rewrites the config value: the override is gone.
        res = mon.maintenance("grip", "apply_backstops")
        assert res.ok and res.after is not None and res.after.collision_sensitivity == 3
        assert res.after.backstops_match is True and res.collision_sensitivity is None
        assert mon.requested_sensitivity("grip") is None
        for bad in (0, 4, 2.5):
            with pytest.raises(ValueError):
                mon.note_requested_sensitivity("grip", bad)
        with pytest.raises(KeyError):
            mon.note_requested_sensitivity("arm9", 2)
    finally:
        mon.stop()
