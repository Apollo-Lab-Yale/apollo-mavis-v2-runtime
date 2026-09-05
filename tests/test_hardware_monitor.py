"""HardwareStateMonitor (phase-09a): config defaults, inert modes, factory kwargs,
pause = disconnect every box / resume = reconnect, telemetry rows."""

from __future__ import annotations

import time

from apollo_mavis_v2_core import WorkcellConfig
from apollo_mavis_v2_core.protocol import HardwareMonitorTelemetry
from conftest import FakeArmMonitor, FakeMonitorFactory, FakeMonitorSample

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
        mon.resume()  # explicit hand-back (phase-09 bring-up seam)
        assert grip.calls == ["disconnect", "start"] and mon.paused is False
        mon.pause()
        mon.pause()  # idempotent
        assert grip.calls == ["disconnect", "start", "disconnect"] and mon.paused is True
    finally:
        mon.stop()
    mon.resume()  # after stop: no reconnect
    assert grip.calls[-1] == "stop"


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
