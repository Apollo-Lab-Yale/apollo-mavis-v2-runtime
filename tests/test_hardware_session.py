"""Hardware teleop session bring-up / teardown (phase-09c/09d; 04-runtime §5, §13.1,
§13.4) over fakes only: a module-scoped ``Runtime`` with the mavis_v2 hardware
workcell (both arms, two fake wrist cameras, the twin alignment overlays on
EGL), a FAKE read-only monitor whose samples say both tracks are homed, and
``tests/fakes.HardwareFakeWorkcell`` through the ``SessionManager.workcell_factory``
seam. No control box, no UVC node.

Covers the contract's test plan: the hand-over call order (pause + join ->
bring_up -> ... -> stop -> resume) and that the monitor's supervisor never
reconnects mid-bring-up (the ``hardware_session_active`` predicate), camera
adoption (hub ids unchanged, fps switched and restored, the same preview object),
the unconditional SafetyGate, the D2 speed scale on the host and the driver side,
the bring-up telemetry rows, the failure path (fps restored, monitor resumed, no
half-connected session), the §3 refusal matrix with its detail strings - incl.
phase-09d's "hardware sessions include every configured arm" (D1 freezing is
now the rail-homing job's alone) - the phase-09d ``start_from=profile`` path
(planned on the gate twin inside bring-up, executed through ``_op_execute_plan``,
409 on a plan failure) and - optional, skipped without the hardware package's
test fakes - the REAL ``HardwareWorkcell`` + ``XArmDriver`` over ``FakeXArmAPI``."""

from __future__ import annotations

import importlib
import importlib.util
import math
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from apollo_mavis_v2_core import WorkcellConfig
from apollo_mavis_v2_core.testing import FakeArm, FakeCamera
from conftest import FakeMonitorFactory, FakeMonitorSample, make_runtime_config
from fakes import FakeServoLimits, HardwareFakeWorkcell
from starlette.testclient import TestClient

from apollo_mavis_v2_runtime.config import (
    ControlConfig,
    HardwareMonitorConfig,
    HardwareProbeConfig,
    HardwareSessionConfig,
    MicrophoneConfig,
    TwinOverlayConfig,
)
from apollo_mavis_v2_runtime.control.loop import ControlLoop
from apollo_mavis_v2_runtime.errors import SafetyConfigError
from apollo_mavis_v2_runtime.runtime import Runtime
from apollo_mavis_v2_runtime.safety.gate import NullGate, SafetyGate
from apollo_mavis_v2_runtime.safety.supervisor import SafetySupervisor
from apollo_mavis_v2_runtime.safety.watchdog import InputWatchdog
from apollo_mavis_v2_runtime.server.app import create_app
from apollo_mavis_v2_runtime.session.hardware import (
    ExecutorCaps,
    RailFlipWorkcell,
    apply_teleop_caps,
    bringup_rows,
    frozen_state,
    scale_control_config,
    scale_driver_config,
    teleop_rate_caps,
)
from apollo_mavis_v2_runtime.session.types import SessionState

pytestmark = pytest.mark.egl

PI = math.pi
SCENE = "mavis_v2"
KEYFRAME_Q = (PI, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
GRIP_Q0 = [PI, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.65]  # keyframe: rail at the operator's right
VIEW_Q0 = [PI, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
URL = "/api/hardware/arms/{}/maintenance"
HW_TESTS = Path(__file__).resolve().parents[2] / "apollo-mavis-v2-hardware" / "tests"

HW_WORKCELL = {
    "kind": "hardware",
    "digital_twin_scene": SCENE,
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
    "cameras": [  # small frames: the overlay renders at the camera resolution
        {
            "id": "grip_wrist",
            "kind": "v4l2",
            "serial": "349643062582",
            "fourcc": "YUYV",
            "resolution": [64, 48],
            "fps": 30,
        },
        {
            "id": "view_wrist",
            "kind": "v4l2",
            "serial": "322143060792",
            "fourcc": "YUYV",
            "resolution": [64, 48],
            "fps": 30,
        },
    ],
    "safety": {"enabled": True},
}


ALL_ARMS = ["grip", "view"]  # phase-09d: a hardware session always includes every arm
ALL_FRAMES = {"grip": "arm_base:grip", "view": "arm_base:view"}
TABLE_DIVE_Q = (PI, 0.8, 0.0, 0.5, 0.0, 0.3, 0.0)  # gripper below the table top


def spec(**over) -> dict:
    base = {
        "mode": "teleop",
        "kind": "hardware",
        "arms": list(ALL_ARMS),
        "frames": dict(ALL_FRAMES),
        "digital_twin_scene": SCENE,
        "speed_scale": 0.1,
    }
    base.update(over)
    return base


def _samples() -> dict[str, FakeMonitorSample]:
    """Both tracks homed + enabled at the keyframe posture (after the operator's
    home_rail); the Manipulation Arm's carriage at 0.65 m, the Perception Arm's at 0."""
    now = time.monotonic()
    return {
        "grip": FakeMonitorSample(
            "grip",
            seq=3,
            t_mono=now,
            q=KEYFRAME_Q,
            rail_homed=True,
            rail_enabled=True,
            rail_pos_m=0.65,
            rail_raw_mm=650.0,
            gripper_open_frac=1.0,
            gripper_raw=84.0,
            collision_sensitivity=3,
            tcp_load_kg=0.95,
            tcp_load_cog_mm=(0.0, 0.0, 60.0),
        ),
        "view": FakeMonitorSample(
            "view",
            seq=5,
            t_mono=now,
            q=KEYFRAME_Q,
            rail_homed=True,
            rail_enabled=True,
            rail_pos_m=0.0,
            rail_raw_mm=0.0,
            collision_sensitivity=3,
            tcp_load_kg=0.55,
            tcp_load_cog_mm=(0.0, 0.0, 90.0),
        ),
    }


def _config(tmp_path):
    cfg = make_runtime_config(tmp_path, SCENE)
    return cfg.model_copy(
        update={
            "workcells": {**cfg.workcells, "hardware": WorkcellConfig.model_validate(HW_WORKCELL)},
            "microphone": MicrophoneConfig(enabled=False),
            "hardware_probe": HardwareProbeConfig(enabled=False),
            "hardware_monitor": HardwareMonitorConfig(),
            "twin_overlay": TwinOverlayConfig(fps=12.0, env_outline=False),
            "hardware_session": HardwareSessionConfig(armed=True, bringup_timeout_s=20.0),
        }
    )


def _wait(pred, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


@pytest.fixture(scope="module")
def cams():
    return {cid: FakeCamera(cid, (64, 48), 30.0) for cid in ("grip_wrist", "view_wrist")}


@pytest.fixture(scope="module")
def factory():
    return FakeMonitorFactory(_samples())


@pytest.fixture(scope="module")
def rt(tmp_path_factory, cams, factory):
    runtime = Runtime(_config(tmp_path_factory.mktemp("rt")), monitor_factory=factory)
    runtime.manager.camera_factory = lambda cam_cfg: cams[cam_cfg.id]
    return runtime


@pytest.fixture(scope="module")
def client(rt):
    with TestClient(create_app(rt)) as c:
        assert _wait(lambda: rt.hardware_monitor.status_of("grip")[0] == "running")
        assert rt.twin_overlay is not None and rt.twin_overlay.wait_ready(60.0)
        assert _wait(lambda: rt.twin_overlay.status_of("grip_wrist_align") == "live", 20.0)
        yield c
        c.delete("/api/session")


class Cells:
    """``workcell_factory`` seam: builds a ``HardwareFakeWorkcell`` per POST and keeps it."""

    def __init__(self, **kw) -> None:
        self.kw = kw
        self.built: list[HardwareFakeWorkcell] = []
        self.driver_factories: list = []
        self.configs: list[WorkcellConfig] = []

    def __call__(self, session_cfg, driver_factory):
        arms = {
            a.id: FakeArm(a.id, has_rail=True, q0=GRIP_Q0 if a.id == "grip" else VIEW_Q0)
            for a in session_cfg.arms
        }
        cell = HardwareFakeWorkcell(arms, kind="hardware", **self.kw)
        self.built.append(cell)
        self.driver_factories.append(driver_factory)
        self.configs.append(session_cfg)
        return cell

    @property
    def cell(self) -> HardwareFakeWorkcell:
        return self.built[-1]


@pytest.fixture()
def cells(rt):
    seam = Cells()
    rt.manager.workcell_factory = seam
    yield seam
    rt.manager.workcell_factory = None
    rt.manager.driver_api_factory = None
    rt.manager.teardown()


def _telemetry(client, pred, frames: int = 60) -> dict:
    with client.websocket_connect("/ws/telemetry") as ws:
        for _ in range(frames):
            msg = ws.receive_json()
            if pred(msg):
                return msg
    raise AssertionError(f"telemetry never satisfied the predicate; last frame {msg}")


# -- pure helpers ---------------------------------------------------------------------------------
def test_speed_scale_on_the_host_side_config():
    cfg = ControlConfig()
    scaled = scale_control_config(cfg, 0.1)
    assert scaled.teleop.linear_mps == pytest.approx(0.012)
    assert scaled.teleop.angular_rps == pytest.approx(0.06)
    assert scaled.teleop.rail_mps == pytest.approx(0.010)
    assert scaled.target_rate.v_mps == pytest.approx(0.1)
    assert scaled.target_rate.w_radps == pytest.approx(0.2)
    assert scaled.dq_max_rad == pytest.approx(0.004)
    assert scaled.jog.slew_rad_per_tick == pytest.approx(0.002)
    assert scaled.jog.rail_m_per_tick == pytest.approx(0.0002)
    # not speeds: untouched
    assert scaled.teleop.gripper_frac_ps == cfg.teleop.gripper_frac_ps
    assert scaled.leash == cfg.leash and scaled.watchdog == cfg.watchdog
    assert scaled.jog.goto_threshold_rad == cfg.jog.goto_threshold_rad
    assert scale_control_config(cfg, 1.0) == cfg
    with pytest.raises(ValueError):
        scale_control_config(cfg, 0.0)


def test_teleop_chain_is_capped_at_what_the_servo_stream_executes():
    """2026-09-07: the tracker target chain ran at target_rate 1.0 m/s against a
    streamer that executed 0.2 m/s (0.02 m/s at speed_scale 0.1; the caps are 0.4
    m/s / 0.6 rad/s since the same day); every faster hand motion hit the leash and
    the truncation was folded into the anchor - hand travel silently DISCARDED, and
    the remainder still arriving up to a leash after the hand stopped. The caps
    make the mapping faithful; no bound is lowered below what the streamer enforced
    anyway. The numbers below are explicit ExecutorCaps, not the driver defaults."""
    cfg = ControlConfig()  # 100 Hz, target_rate 1.0 m/s / 2.0 rad/s, dq_max 0.04, linear 0.12
    caps = ExecutorCaps(
        slew_rad_per_tick=0.003,
        cart_step_m=0.002,
        lever_arm_m=(1.0,) * 7,
        source="servo",
        joint_step_rad=0.003,
    )
    assert teleop_rate_caps(cfg, caps) == (pytest.approx(0.2), pytest.approx(0.3))
    out = apply_teleop_caps(cfg, caps)
    assert out.target_rate.v_mps == pytest.approx(0.2)  # 1.0 -> the streamer's 0.2 m/s
    assert out.target_rate.w_radps == pytest.approx(0.3)  # 2.0 -> max_joint_vel 0.3 rad/s
    assert out.dq_max_rad == pytest.approx(0.003)  # 0.04 -> 0.3 rad/s / 100 Hz
    assert out.teleop.linear_mps == pytest.approx(0.12)  # already below the cap: kept
    assert out.teleop.angular_rps == pytest.approx(0.3)  # 0.6 rad/s -> 0.3
    # Not a speed of the servo stream: untouched.
    assert out.teleop.rail_mps == cfg.teleop.rail_mps
    assert out.teleop.gripper_frac_ps == cfg.teleop.gripper_frac_ps
    assert out.leash == cfg.leash and out.jog == cfg.jog
    # At speed_scale 0.1 the streamer executes 0.02 m/s: the chain follows.
    scaled = scale_control_config(cfg, 0.1)
    caps01 = ExecutorCaps(
        slew_rad_per_tick=0.0003,
        cart_step_m=0.0002,
        lever_arm_m=(1.0,) * 7,
        source="servo",
        joint_step_rad=0.0003,
    )
    out01 = apply_teleop_caps(scaled, caps01)
    assert out01.target_rate.v_mps == pytest.approx(0.02)
    assert out01.teleop.linear_mps == pytest.approx(0.012)  # 0.12 * 0.1 < 0.02: kept
    assert out01.dq_max_rad == pytest.approx(0.0003)
    # No Cartesian bound published: only the joint-rate caps apply.
    joint_only = ExecutorCaps(
        slew_rad_per_tick=0.003, cart_step_m=None, lever_arm_m=None, joint_step_rad=0.003
    )
    assert teleop_rate_caps(cfg, joint_only) == (None, pytest.approx(0.3))
    o = apply_teleop_caps(cfg, joint_only)
    assert o.target_rate.v_mps == cfg.target_rate.v_mps and o.teleop.linear_mps == 0.12
    assert o.target_rate.w_radps == pytest.approx(0.3)
    # HOST-only caps (fakes without ServoLimits): the jog slew is NOT a servo bound
    # and must not leak into teleop - the config comes back untouched.
    host = ExecutorCaps(slew_rad_per_tick=0.002, cart_step_m=None, lever_arm_m=None)
    assert teleop_rate_caps(cfg, host) == (None, None) and apply_teleop_caps(cfg, host) == cfg
    # Caps looser than the host config change nothing either.
    loose = ExecutorCaps(
        slew_rad_per_tick=1.0, cart_step_m=1.0, lever_arm_m=None, source="servo", joint_step_rad=1.0
    )
    assert apply_teleop_caps(cfg, loose) == cfg
    # The real driver's caps: joint_step_rad is the servo bound, slew the jog-bounded one.
    from apollo_mavis_v2_runtime.session.hardware import servo_executor_caps

    servo_caps = servo_executor_caps(FakeServoLimits(), 100.0, 0.002)  # scale-0.1 fakes
    assert servo_caps.joint_step_rad == pytest.approx(0.0003)
    assert servo_caps.slew_rad_per_tick == pytest.approx(0.0003)
    assert teleop_rate_caps(cfg, servo_caps) == (pytest.approx(0.02), pytest.approx(0.03))


def test_speed_scale_on_the_driver_side_config():
    hw = pytest.importorskip("apollo_mavis_v2_hardware")
    cfg = hw.XArmDriverConfig(arm_id="grip", ip="192.168.1.201", gripper="xarm_g2")
    # D2 caps at scale 1.0, raised 2026-09-07 (0.3 rad/s / 2 mm were "over-conservative")
    assert cfg.servo.max_joint_vel == (0.6,) * 7 and cfg.servo.max_cart_step_m == 0.004
    assert cfg.rail_speed_mm_s == 50
    scaled = scale_driver_config(cfg, 0.1)
    assert scaled.servo.max_joint_vel == pytest.approx((0.06,) * 7)
    assert scaled.servo.max_cart_step_m == pytest.approx(0.0004)
    assert scaled.rail_speed_mm_s == 5 and isinstance(scaled.rail_speed_mm_s, int)
    assert scaled.servo.max_joint_acc == cfg.servo.max_joint_acc  # not a speed cap
    assert (scaled.arm_id, scaled.ip, scaled.tcp_load_kg) == ("grip", "192.168.1.201", 0.82)
    assert scale_driver_config(cfg, 0.01).rail_speed_mm_s == 1  # floor 1 mm/s
    assert scale_driver_config(cfg, 1.0) == cfg


def test_rail_flip_workcell_mirrors_states_and_commands():
    inner = HardwareFakeWorkcell(
        {"grip": FakeArm("grip", has_rail=True, q0=GRIP_Q0), "view": FakeArm("view")},
        kind="hardware",
    )
    inner.start()
    cell = RailFlipWorkcell(inner)
    st = cell.states()["grip"]
    assert st.q[7] == pytest.approx(0.0) and st.rail_pos_m == pytest.approx(0.0)  # 0.65 - 0.65
    assert cell.states()["view"].q.shape == (7,)  # rail-less arm passes through
    cell.arms["grip"].command_joints(np.array([PI, 0, 0, 0, 0, 0, 0, 0.1]))
    assert inner.arms["grip"]._target[7] == pytest.approx(0.55)  # back to track coordinates
    cell.arms["grip"].command_rail(0.65)
    assert inner.arms["grip"]._target[7] == pytest.approx(0.0)
    assert cell.arms["grip"].dof == 8 and cell.arms["grip"].has_rail
    assert cell.kind == "hardware" and cell.drain_events() == []
    assert cell.request_recovery is not None  # forwarded surface
    cell.stop()
    assert inner.stop_calls == 1


def test_frozen_state_uses_the_sample_rail_fallback_and_flip():
    sample = _samples()["view"]
    state, note = frozen_state("view", sample, has_rail=True, rail_fallback_m={"view": 0.2})
    assert note is None and state.q.shape == (8,) and state.q[7] == 0.0 and not state.stale
    unhomed = replace(sample, rail_pos_m=None, rail_homed=False, rail_enabled=False)
    state, note = frozen_state("view", unhomed, has_rail=True, rail_fallback_m={"view": 0.2})
    assert state.q[7] == pytest.approx(0.2) and note == "rail not homed - twin assumes 0.20 m"
    state, note = frozen_state(
        "view", replace(sample, rail_pos_m=0.1), has_rail=True, rail_fallback_m={}, rail_flip=True
    )
    assert state.q[7] == pytest.approx(0.55)
    state, _ = frozen_state("view", sample, has_rail=False, rail_fallback_m={})
    assert state.q.shape == (7,) and state.rail_pos_m is None


def test_bringup_rows_map_the_status_stages():
    from fakes import FakeBringupStatus

    rows = {r.step: r for r in bringup_rows(FakeBringupStatus("grip"))}
    assert {r.status for r in rows.values()} == {"pending"}
    ok = FakeBringupStatus(
        "grip",
        network="ok",
        connected=True,
        fw_version="1.12.10",
        sn="X",
        rail="ready",
        gripper="xarm_g2",
        warnings=["netsetup skipped (no NetSetup wired)"],
    )
    rows = {r.step: r for r in bringup_rows(ok)}
    assert (
        rows["network"].status == "ok" and rows["connect"].detail == "connected (fw 1.12.10, sn X)"
    )
    assert rows["rail"].detail == "linear track homed and enabled" and rows["report"].status == "ok"
    assert rows["gripper"].detail == "gripper xarm_g2" and rows["warnings"].status == "warning"
    unhomed = FakeBringupStatus(
        "grip",
        network="ok",
        rail="unhomed",
        error="[rail] grip: linear track not homed (on_zero == 0)",
    )
    rows = {r.step: r for r in bringup_rows(unhomed)}
    assert rows["rail"].status == "error" and "not homed" in rows["rail"].detail
    assert rows["connect"].status == "ok"  # connect itself returned; the rail stage refused
    assert rows["report"].status == "pending"


def test_control_loop_refuses_hardware_kind_without_a_twin_gate(rt):
    cell = HardwareFakeWorkcell(
        {"grip": FakeArm("grip", has_rail=True, q0=GRIP_Q0)}, kind="hardware"
    )
    cell.start()
    with pytest.raises(SafetyConfigError, match="SafetyGate bound to a live digital twin"):
        ControlLoop(
            cell,
            ControlConfig(),
            rt.bus,
            SafetySupervisor(NullGate(), InputWatchdog()),
            ["grip"],
            workcell_kind="hardware",
        )
    cell.stop()


# -- the happy path -----------------------------------------------------------------------------
def test_bringup_order_camera_adoption_gate_scale_and_teardown(client, rt, factory, cams, cells):
    grip, view = factory.monitors["grip"], factory.monitors["view"]
    starts = {a: m.calls.count("start") for a, m in factory.monitors.items()}
    n_grip, n_view = len(grip.calls), len(view.calls)
    hub_ids_before = set(rt.hub.ids())
    assert {"grip_wrist", "view_wrist", "grip_wrist_align", "view_wrist_align"} <= hub_ids_before
    assert rt.hub._workers["grip_wrist"].fps == 15.0  # preview fps
    grip_cam = rt.manager.hardware_camera("grip_wrist")
    assert grip_cam is cams["grip_wrist"]

    r = client.post("/api/session", json=spec())
    assert r.status_code == 200, r.text
    info = r.json()
    assert (info["kind"], info["speed_scale"], info["arms"]) == ("hardware", 0.1, ALL_ARMS)
    assert info["streams"] == [] and info["state"] in ("bringup", "running")
    session = rt.manager.session
    cell = cells.cell
    try:
        # (a) hand-over order: every monitor disconnected + joined BEFORE the drivers connect,
        #     and the supervisor did not start() anything since.
        assert grip.calls[n_grip:] == ["disconnect", "join"]
        assert view.calls[n_view:] == ["disconnect", "join"]  # D1: all monitors pause together
        assert rt.hardware_monitor.paused and rt.manager.hardware_session_active
        assert cell.started and cell.bringup_calls == [20.0]  # bring_up, with the config budget
        assert set(cell.arms) == set(ALL_ARMS) and cells.configs[0].cameras == []  # no cameras
        assert [a.id for a in cells.configs[0].arms] == ALL_ARMS  # both arms, config order
        # (b) camera adoption: same hub ids, same preview objects, fps switched to session fps
        assert set(rt.hub.ids()) == hub_ids_before
        assert (
            rt.hub._workers["grip_wrist"].fps == 30.0 and rt.hub._workers["view_wrist"].fps == 30.0
        )
        assert rt.manager.hardware_camera("grip_wrist") is grip_cam
        assert session.adopted_streams == ["grip_wrist", "view_wrist"]
        rows = {c["camera_id"]: c for c in client.get("/api/cameras").json()}
        assert rows["grip_wrist"]["live"] is True and rows["view_wrist"]["live"] is True
        # (d) the gate is unconditional: a SafetyGate on a live twin; hardware kind loop
        assert isinstance(session.supervisor.gate, SafetyGate)
        assert session.supervisor.twin is session.twin and session.twin is not None
        assert session.loop.workcell_kind == "hardware" and session.loop.gripper_arms == {"grip"}
        assert session.workcell is cell and session.inner_workcell is cell  # rail_flip off
        # (f) speed scale 0.1 on the host side. These fakes publish no ServoLimits, so
        #     the caps are host-only and the teleop chain is NOT capped further (the
        #     servo-capped case is test_driver_factory_seam_receives_the_speed_scaled_caps).
        assert session.loop.cfg.dq_max_rad == pytest.approx(0.004)
        assert session.loop.cfg.teleop.linear_mps == pytest.approx(0.012)
        assert session.loop.cfg.target_rate.v_mps == pytest.approx(0.1)
        # (h) phase-09d: both arms are session arms - nothing is frozen (D1 is the
        #     rail-homing job's mechanism now); the twin tracks both from the drivers
        assert session.frozen_arms == [] and session.planned_start is None
        assert _wait(lambda: rt.manager.state is SessionState.RUNNING)
        twin = session.twin
        q_view = twin.data.qpos[twin.addr["view"].qpos_adr]
        assert q_view[:7] == pytest.approx(list(KEYFRAME_Q)) and q_view[7] == pytest.approx(0.0)
        msg = _telemetry(client, lambda m: m["session"]["state"] == "running")
        assert msg["session"]["bringup"] is None  # rows shown until running, then cleared
        assert msg["hardware_monitor"]["paused"] is True
        assert [a["arm_id"] for a in msg["arms"]] == ALL_ARMS
        assert msg["collision"]["blocked"] is False  # fresh states, keyframe posture clear
        # the overlay reads the session provider now: both arms from the drivers (live)
        assert rt.twin_overlay.state_provider is not None
        assert _wait(lambda: rt.twin_overlay.status_of("grip_wrist_align") == "live", 15.0)
        assert _wait(lambda: rt.twin_overlay.status_of("view_wrist_align") == "live", 15.0)
        assert rows_of(client)["grip_wrist_align"]["live"] is True
        # (e) a second session is refused, home_rail too
        assert client.post("/api/session", json=spec()).status_code == 409
        r = client.post(URL.format("grip"), json={"op": "home_rail", "dry_run": True})
        assert r.status_code == 409 and r.json()["detail"].endswith("end the session first")
    finally:
        assert client.delete("/api/session").status_code == 204
    # teardown mirror: loop -> workcell.stop (D6 hand-back) -> fps back -> monitor resumed
    assert cell.stop_calls == 1 and rt.manager.session is None
    assert rt.hub._workers["grip_wrist"].fps == 15.0 and rt.hub._workers["view_wrist"].fps == 15.0
    assert set(rt.hub.ids()) == hub_ids_before and rt.hub.has("grip_wrist")
    assert rt.manager.hardware_camera("grip_wrist") is grip_cam  # never re-opened
    assert rt.twin_overlay.state_provider is None
    assert _wait(lambda: not rt.hardware_monitor.paused)
    for a, m in factory.monitors.items():
        assert m.calls.count("start") == starts[a] + 1 and m.calls[-1] == "start"
    assert not rt.manager.hardware_session_active
    assert client.get("/api/session").status_code == 404
    assert _wait(lambda: rt.twin_overlay.status_of("grip_wrist_align") == "live", 15.0)


def rows_of(client) -> dict:
    return {c["camera_id"]: c for c in client.get("/api/cameras").json()}


def test_predicate_holds_and_rows_stream_while_bringup_is_in_flight(client, rt, factory):
    """The monitor's supervisor polls the predicate every 0.5 s: with the connect held
    for 1.3 s it must NOT reconnect the boxes (``hardware_session_active`` is true
    from the first line of create()); meanwhile GET /api/session answers
    ``state: bringup`` and telemetry carries the rows."""
    from apollo_mavis_v2_core.protocol import SessionSpec

    grip = factory.monitors["grip"]
    seen: dict = {}

    def probe() -> None:  # runs inside bring_up, on the POST thread, mid-connect
        seen["active"] = rt.manager.hardware_session_active
        seen["paused"] = rt.hardware_monitor.paused
        seen["grip_status"] = grip.status
        seen["info"] = rt.manager.info().model_dump()
        seen["rows"] = [r.model_dump() for r in rt.manager.bringup_telemetry()]
        seen["state"] = rt.manager.state

    seam = Cells(bringup_delay_s=1.3, on_bring_up=probe)
    rt.manager.workcell_factory = seam
    n0 = len(grip.calls)
    errors: list[BaseException] = []

    def create() -> None:
        try:
            rt.manager.create(SessionSpec.model_validate(spec()))
        except BaseException as e:  # noqa: BLE001 - reported below
            errors.append(e)

    worker = threading.Thread(target=create, daemon=True)
    worker.start()
    try:
        assert _wait(lambda: "info" in seen, 10.0)
        worker.join(15.0)
        assert not worker.is_alive() and not errors, errors
        assert seen["active"] is True and seen["paused"] is True and seen["grip_status"] == "paused"
        assert seen["state"] is SessionState.BRINGUP and seen["info"]["state"] == "bringup"
        assert seen["info"]["kind"] == "hardware" and seen["info"]["streams"] == []
        steps = {(r["arm_id"], r["step"], r["status"]) for r in seen["rows"]}
        assert ("grip", "monitor", "ok") in steps and ("grip", "network", "ok") in steps
        # no start() slipped in during the 1.3 s hold (two supervisor rounds)
        assert grip.calls[n0:] == ["disconnect", "join"]
        assert rt.manager.session is not None
        assert _wait(lambda: rt.manager.state is SessionState.RUNNING)
        assert grip.calls[n0:] == ["disconnect", "join"]
    finally:
        rt.manager.workcell_factory = None
        rt.manager.teardown()
    assert grip.calls[-1] == "start" and _wait(lambda: not rt.hardware_monitor.paused)


def test_driver_factory_seam_receives_the_speed_scaled_caps(client, rt, cells):
    hw = pytest.importorskip("apollo_mavis_v2_hardware")
    r = client.post("/api/session", json=spec(speed_scale=0.3))
    assert r.status_code == 200, r.text
    try:
        assert r.json()["speed_scale"] == 0.3
        driver_factory = cells.driver_factories[0]
        base = hw.XArmDriverConfig(arm_id="grip", ip="192.168.1.201", gripper="xarm_g2")
        driver = driver_factory(base)  # an inert XArmDriver (nothing connects in the ctor)
        assert isinstance(driver, hw.XArmDriver)
        assert driver.cfg.servo.max_joint_vel == pytest.approx((0.18,) * 7)
        assert driver.cfg.servo.max_cart_step_m == pytest.approx(0.0012)
        assert driver.cfg.rail_speed_mm_s == 15
        # The session's cell is the FAKE workcell (its arms carry no ServoLimits), so the
        # loop sees host-only caps and the teleop chain stays at the plain speed scale;
        # the servo-capped values are pinned by test_teleop_chain_is_capped_... above.
        assert rt.manager.session.loop.cfg.dq_max_rad == pytest.approx(0.012)
        assert rt.manager.session.loop.cfg.target_rate.v_mps == pytest.approx(0.3)
    finally:
        assert client.delete("/api/session").status_code == 204


def test_subset_of_the_configured_arms_is_refused(client, rt, factory, cells):
    """Phase-09d user decision 1: both arms are always part of a hardware session."""
    grip = factory.monitors["grip"]
    n0 = len(grip.calls)
    for arms in (["grip"], ["view"], ["grip", "grip"]):
        body = spec(arms=arms, frames={a: f"arm_base:{a}" for a in arms})
        r = client.post("/api/session", json=body)
        assert r.status_code == 409, r.text
        detail = r.json()["detail"]
        assert detail.startswith(
            "hardware sessions include every configured arm (Manipulation Arm, Perception Arm)"
        ), detail
    body = spec(arms=["grip"], frames={"grip": "arm_base:grip"})
    assert "missing ['view']" in client.post("/api/session", json=body).json()["detail"]
    assert grip.calls[n0:] == [] and cells.built == [] and rt.manager.session is None
    assert not rt.hardware_monitor.paused


def _profile(rt, name: str, q7, *, rail: dict | None = None):
    from apollo_mavis_v2_core import ArmPosture, StateProfile

    rail = rail or {}
    return rt.profile_store.save(
        StateProfile(
            name=name,
            workcell_kind="hardware",
            arms={
                a: ArmPosture(q=list(q7), rail_pos_m=rail.get(a), gripper_open_frac=1.0)
                for a in ALL_ARMS
            },
        )
    )


def test_start_from_profile_plans_on_the_gate_twin_and_executes_the_waypoints(
    client, rt, cells, monkeypatch
):
    """Phase-09d user decision 3: the hardware ``start_from=profile`` motion goes
    through the SAME path as sim - ``twin.plan`` (RRT-Connect on the gate twin,
    inside bring-up) -> ``ControlLoop._op_execute_plan`` with the planned waypoints."""
    from apollo_mavis_v2_sim import DigitalTwin

    plans: list = []
    real_plan = DigitalTwin.plan

    def spy_plan(self, req):
        result = real_plan(self, req)
        plans.append((req, result))
        return result

    executed: list = []
    real_exec = ControlLoop._op_execute_plan

    def spy_exec(self, cmd):
        executed.append(cmd.args)
        return real_exec(self, cmd)

    monkeypatch.setattr(DigitalTwin, "plan", spy_plan)
    monkeypatch.setattr(ControlLoop, "_op_execute_plan", spy_exec)
    target = (PI, -0.3, 0.0, 0.4, 0.0, 0.7, 0.0)  # a lifted, still folded posture
    profile = _profile(rt, "lifted", target)
    r = client.post("/api/session", json=spec(start_from=f"profile:{profile.profile_id}"))
    assert r.status_code == 200, r.text
    try:
        session = rt.manager.session
        assert session.planned_start is not None
        waypoints, grippers = session.planned_start
        assert set(waypoints) == set(ALL_ARMS) and grippers == {"grip": 1.0}
        assert len(plans) == 1  # planned ONCE, inside bring-up, before the loop started
        req, result = plans[0]
        assert result.ok and set(req.q_goal) == set(ALL_ARMS)
        assert req.q_goal["grip"][:7] == pytest.approx(list(target))
        assert req.q_goal["grip"][7] == pytest.approx(0.65)  # rail slot kept (profile: None)
        assert req.q_start["grip"] == pytest.approx(GRIP_Q0)  # measured start
        assert all(len(w) >= 2 for w in waypoints.values())
        assert _wait(lambda: rt.manager.state is SessionState.RUNNING, 20.0), (
            rt.manager.state,
            session.fault_detail,
            session.start_from_progress,
            session.supervisor.merged_report(),
        )
        assert executed and executed[0]["waypoints"] == waypoints  # the same waypoints
        assert executed[0]["gripper"] == grippers
        assert len(plans) == 1  # the worker did NOT plan again
        # the loop drove the commanded posture to the profile target
        loop = session.loop
        assert loop._last_cmd["grip"][:7] == pytest.approx(list(target), abs=1e-6)
        assert loop._last_cmd["view"][:7] == pytest.approx(list(target), abs=1e-6)
    finally:
        assert client.delete("/api/session").status_code == 204
    assert _wait(lambda: not rt.hardware_monitor.paused)


def test_start_from_profile_plan_failure_is_409_and_tears_down(client, rt, factory, cells):
    """A profile posture the gate twin cannot reach collision-free (the gripper
    below the table top) refuses the session with the planner's reason; the
    drivers are stopped and the monitor resumed like any bring-up failure."""
    grip = factory.monitors["grip"]
    profile = _profile(rt, "table-dive", TABLE_DIVE_Q)
    r = client.post("/api/session", json=spec(start_from=f"profile:{profile.profile_id}"))
    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert detail.startswith("profile motion not collision-free: goal_in_collision ("), detail
    assert "table" in detail and profile.profile_id in detail
    cell = cells.cell
    assert cell.started is False and cell.stop_calls == 1  # connected, then torn down
    assert rt.manager.session is None and client.get("/api/session").status_code == 404
    assert rt.hub._workers["grip_wrist"].fps == 15.0 and rt.twin_overlay.state_provider is None
    assert not rt.manager.hardware_session_active and rt.manager.bringup_telemetry() is None
    assert _wait(lambda: not rt.hardware_monitor.paused) and grip.calls[-1] == "start"


# -- failure path -----------------------------------------------------------------------------
def test_bringup_failure_tears_down_restores_fps_and_resumes_the_monitor(client, rt, factory):
    grip = factory.monitors["grip"]
    seam = Cells(
        fail={
            "grip": "[rail] grip: linear track not homed (on_zero == 0): carriage "
            "position unknown; home it from the Hardware tab (Home rail)"
        }
    )
    rt.manager.workcell_factory = seam
    try:
        r = client.post("/api/session", json=spec())
    finally:
        rt.manager.workcell_factory = None
    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert detail.startswith("hardware bring-up failed: Manipulation Arm: rail - [rail] grip:")
    cell = seam.cell
    assert cell.stop_calls == 1  # the half-connected workcell was shut down
    assert rt.manager.session is None and client.get("/api/session").status_code == 404
    assert rt.hub._workers["grip_wrist"].fps == 15.0  # adoption never happened / restored
    assert rt.hub.has("grip_wrist") and rt.twin_overlay.state_provider is None
    assert not rt.manager.hardware_session_active
    assert _wait(lambda: not rt.hardware_monitor.paused) and grip.calls[-1] == "start"
    assert rt.manager.bringup_telemetry() is None
    msg = _telemetry(client, lambda m: m["session"]["state"] == "idle")
    assert msg["session"]["bringup"] is None and msg["hardware_monitor"]["paused"] is False


def test_bringup_refuses_a_connected_arm_whose_rail_is_not_ready(client, rt, factory):
    """A driver may come back CONNECTED with the track not READY (before the review
    fix a failed gate register read latched RAIL_ERROR and connect() continued:
    ``connected True``, ``error None``, ``q[7] == 0.0`` while the carriage may sit at
    0.65 m). The runtime must not gate on a guessed carriage position."""
    grip = factory.monitors["grip"]
    seam = Cells(rail_status={"grip": "error"})
    rt.manager.workcell_factory = seam
    try:
        r = client.post("/api/session", json=spec())
    finally:
        rt.manager.workcell_factory = None
    assert r.status_code == 409, r.text
    assert r.json()["detail"] == (
        "hardware bring-up failed: Manipulation Arm: rail - linear track error after connect "
        "(carriage position unknown, the digital twin cannot gate it)"
    )
    cell = seam.cell
    st = cell.statuses["grip"]
    assert st.connected and st.error is None and st.rail == "error"  # the trap, as reported
    assert cell.stop_calls == 1 and rt.manager.session is None  # torn down like any failure
    assert rt.hub._workers["grip_wrist"].fps == 15.0 and rt.twin_overlay.state_provider is None
    assert not rt.manager.hardware_session_active and rt.manager.bringup_telemetry() is None
    assert _wait(lambda: not rt.hardware_monitor.paused) and grip.calls[-1] == "start"


def test_rail_flip_start_check_and_loop_share_one_rail_convention(client, rt, cells, monkeypatch):
    """With ``hardware_session.rail_flip`` the start-posture ``twin.check`` must see
    the FLIPPED states (before the fix it used the pre-wrap ``workcell.states()``:
    track 0.65 m instead of twin 0.0 m, so the warning was judged at the mirrored
    carriage and the twin stayed mirrored until the loop's first sync)."""
    from apollo_mavis_v2_sim import DigitalTwin

    checks: list[dict] = []
    real_check = DigitalTwin.check

    def spy(self, states, *a, **kw):
        checks.append({k: np.array(v, dtype=float) for k, v in states.items()})
        return real_check(self, states, *a, **kw)

    monkeypatch.setattr(DigitalTwin, "check", spy)
    cfg0 = rt.manager.cfg
    rt.manager.cfg = cfg0.model_copy(
        update={"hardware_session": cfg0.hardware_session.model_copy(update={"rail_flip": True})}
    )
    try:
        r = client.post("/api/session", json=spec())
        assert r.status_code == 200, r.text
        session = rt.manager.session
        assert isinstance(session.workcell, RailFlipWorkcell)
        assert session.inner_workcell is cells.cell
        start = next(c for c in checks if "grip" in c)  # the bring-up's start-posture check
        assert start["grip"].shape == (8,)
        assert start["grip"][7] == pytest.approx(0.0)  # twin convention: 0.65 - 0.65, not 0.65
        assert session.workcell.states()["grip"].q[7] == pytest.approx(0.0)
        assert cells.cell.states()["grip"].q[7] == pytest.approx(0.65)  # the track's own value
        assert _wait(lambda: rt.manager.state is SessionState.RUNNING)
        twin = session.twin
        assert twin.data.qpos[twin.addr["grip"].qpos_adr[7]] == pytest.approx(0.0)
    finally:
        rt.manager.cfg = cfg0
        assert client.delete("/api/session").status_code == 204
    assert _wait(lambda: not rt.hardware_monitor.paused)


def test_hardware_session_active_stays_false_while_the_refusal_matrix_runs(
    client, rt, factory, monkeypatch
):
    """``_creating_kind`` is recorded only after ``_validate_hardware`` passes: a
    supervisor round landing inside the validation window must NOT see the
    hand-over predicate, disconnect every arm monitor and turn a valid-looking
    request into a spurious "no monitor sample (monitor paused)" 409."""
    grip, view = factory.monitors["grip"], factory.monitors["view"]
    n_grip, n_view = len(grip.calls), len(view.calls)
    seen: dict = {}
    real = rt.manager._validate_hardware

    def validate_with_a_supervisor_round(spec_, wc):
        seen["active"] = rt.manager.hardware_session_active
        rt.hardware_monitor._apply_paused(rt.hardware_monitor._safe_paused())  # the 0.5 s round
        seen["paused"] = rt.hardware_monitor.paused
        return real(spec_, wc)

    monkeypatch.setattr(rt.manager, "_validate_hardware", validate_with_a_supervisor_round)
    r = client.post("/api/session", json=spec(start_from="profile:nope"))  # refused LAST
    assert r.status_code == 409 and "unknown profile 'nope'" in r.json()["detail"], r.text
    assert seen == {"active": False, "paused": False}
    assert grip.calls[n_grip:] == [] and view.calls[n_view:] == []  # never disconnected
    assert rt.manager.session is None and not rt.hardware_monitor.paused
    assert rt.manager._creating_kind is None


def test_bringup_failure_on_a_dof_mismatch(client, rt, factory):
    """A 7-dof driver on a railed twin arm would make every twin.sync raise and
    hold ALL arms silently - refused at bring-up instead."""

    def seam(session_cfg, driver_factory):
        return HardwareFakeWorkcell(
            {"grip": FakeArm("grip", has_rail=False), "view": FakeArm("view", has_rail=True)},
            kind="hardware",
        )

    rt.manager.workcell_factory = seam
    try:
        r = client.post("/api/session", json=spec())
    finally:
        rt.manager.workcell_factory = None
    assert (
        r.status_code == 409 and "driver reports 7 dof but the digital twin" in r.json()["detail"]
    )
    assert rt.manager.session is None and _wait(lambda: not rt.hardware_monitor.paused)


# -- the refusal matrix (§3) ---------------------------------------------------------------------
def test_refusal_matrix_details(client, rt, factory):
    grip, view = factory.monitors["grip"], factory.monitors["view"]
    n0 = len(grip.calls)

    def refused(body: dict, needle: str) -> None:
        r = client.post("/api/session", json=body)
        assert r.status_code == 409, r.text
        assert needle in r.json()["detail"], (needle, r.json()["detail"])

    refused(spec(mode="collect", task="t"), "hardware sessions support teleop only (phase-09c)")
    refused(spec(arms=[], frames={}), "session needs at least one arm")
    refused(spec(arms=["arm9"], frames={}), "arms ['arm9'] not in the hardware workcell")
    refused(spec(arms=["view"], frames={}), "hardware sessions include every configured arm")
    refused(spec(digital_twin_scene="not_a_scene"), "not_a_scene")
    refused(spec(start_from="profile:nope"), "unknown profile 'nope'")
    original = grip.sample
    try:
        grip.sample = replace(original, rail_homed=False, rail_enabled=False, rail_pos_m=None)
        refused(spec(), "Manipulation Arm: rail not homed - home it from the Hardware tab")
        grip.sample = replace(original, rail_enabled=False, rail_pos_m=None)  # homed, not enabled
        refused(spec(), "Manipulation Arm: rail not homed")
        grip.sample = replace(original, error_code=31)
        refused(spec(), "Manipulation Arm: controller error 31 is latched - clear errors first")
        grip.sample = replace(original, rail_present=False)
        refused(spec(), "expects a linear track but the monitor found none")
        grip.sample = None
        refused(spec(), "Manipulation Arm: no monitor sample (monitor running)")
    finally:
        grip.sample = original
    grip.forced_status = "error"
    grip.detail_text = "connect failed: timeout"
    try:
        refused(spec(), "no monitor sample (monitor error: connect failed: timeout)")
    finally:
        grip.forced_status = None
        grip.detail_text = ""
    view.maintenance_busy = True  # a homing in flight on the OTHER arm blocks too
    try:
        refused(spec(), "rail homing in progress on the Perception Arm")
    finally:
        view.maintenance_busy = False
    probe = rt.hardware_probe
    real = probe.reachable
    probe.reachable = lambda arm_id: "unreachable"  # type: ignore[method-assign]
    try:
        refused(spec(), "Manipulation Arm: control box 192.168.1.201 is unreachable")
    finally:
        probe.reachable = real  # type: ignore[method-assign]
    # 422s are pydantic's: speed_scale outside (0, 1]
    assert client.post("/api/session", json=spec(speed_scale=0)).status_code == 422
    assert client.post("/api/session", json=spec(speed_scale=1.5)).status_code == 422
    # nothing was touched by any refusal: no pause, no session, previews untouched
    assert grip.calls[n0:] == [] and rt.manager.session is None
    assert not rt.hardware_monitor.paused and rt.hub._workers["grip_wrist"].fps == 15.0


def test_no_hardware_workcell_configured_is_409(tmp_path):
    cfg = make_runtime_config(tmp_path, SCENE)  # sim only
    runtime = Runtime(cfg)
    with TestClient(create_app(runtime)) as c:
        r = c.post("/api/session", json=spec())
        assert r.status_code == 409 and "no 'hardware' workcell config" in r.json()["detail"]


# -- optional: the REAL HardwareWorkcell + XArmDriver over the hardware package's FakeXArmAPI ----
def _load_fake_xarm_api():
    """The hardware package ships its fake in ``tests/fakes`` (not in the wheel):
    load it as a private package so it does not clash with this suite's ``fakes``."""
    path = HW_TESTS / "fakes"
    if not (path / "fake_xarm_api.py").exists():
        pytest.skip("apollo-mavis-v2-hardware/tests/fakes not checked out next to the runtime")
    name = "hw_test_fakes"
    if name not in sys.modules:
        pkg_spec = importlib.util.spec_from_file_location(
            name, path / "__init__.py", submodule_search_locations=[str(path)]
        )
        pkg = importlib.util.module_from_spec(pkg_spec)
        sys.modules[name] = pkg
        pkg_spec.loader.exec_module(pkg)
    return importlib.import_module(f"{name}.fake_xarm_api").FakeXArmAPI


def test_real_hardware_workcell_and_driver_over_fake_xarm_api(client, rt, factory):
    """No seam: ``HardwareWorkcell(subset cfg, driver_factory=<scaled XArmDriver>,
    netsetup=None)`` with every ``XArmDriver`` talking to a ``FakeXArmAPI`` whose
    track is homed - the driver connects (clean -> backstops -> enable -> mode 0 ->
    mode 1 -> 100 Hz stream), the session runs, ``stop()`` hands the arm back
    stopped + braked (D6) with the track's homed flag kept, and the scaled rail
    speed reached the fake."""
    pytest.importorskip("apollo_mavis_v2_hardware")
    FakeXArmAPI = _load_fake_xarm_api()
    apis: dict[str, object] = {}

    def api_factory(ip, **kw):
        api = FakeXArmAPI(
            ip,
            auto_report_hz=100.0,
            has_rail=True,
            rail_homed=True,
            rail_enabled=True,
            initial_q=list(KEYFRAME_Q),
            **kw,
        )
        apis[ip] = api
        return api

    rt.manager.driver_api_factory = api_factory
    try:
        r = client.post("/api/session", json=spec())
        assert r.status_code == 200, r.text
        session = rt.manager.session
        api = apis["192.168.1.201"]
        assert set(apis) == {"192.168.1.201", "192.168.2.219"}  # both arms (phase-09d)
        assert type(session.inner_workcell).__name__ == "HardwareWorkcell"
        driver = session.inner_workcell.arms["grip"]
        assert driver.dof == 8 and driver.has_rail
        assert driver.cfg.rail_speed_mm_s == 5 and driver.cfg.servo.max_joint_vel[
            0
        ] == pytest.approx(0.06)
        assert driver.cfg.rail_homing == "require_homed"  # sessions never allow_unhomed
        assert api.rail_speed == 5 and api.homing_started == 0  # never homes at connect
        assert "set_linear_track_back_origin" not in api.call_names()
        assert _wait(lambda: rt.manager.state is SessionState.RUNNING)
        st = session.workcell.states()["grip"]
        assert st.q.shape == (8,) and not st.stale and st.q[:7] == pytest.approx(list(KEYFRAME_Q))
        assert rt.hardware_monitor.paused
    finally:
        rt.manager.driver_api_factory = None
        assert client.delete("/api/session").status_code == 204
    for api in apis.values():
        assert api.motion_enabled is False and api.state == 4  # D6 hand-back
        assert api.rail_homed is True and api.homing_started == 0
    assert rt.manager.session is None and _wait(lambda: not rt.hardware_monitor.paused)
    assert factory.monitors["grip"].calls[-1] == "start"


def test_unarmed_config_refuses_real_hardware_before_touching_anything(tmp_path):
    """Arming switch (2026-09-05): with the repo default ``hardware_session.armed`` False
    and NO ``workcell_factory`` seam, a hardware session and a real ``home_rail`` are
    refused with 409 before the monitor is paused or any SDK client is built (the
    conftest guard would raise its own message if the real factory were reached);
    the zero-write dry run stays available."""
    cfg = _config(tmp_path).model_copy(
        update={
            "hardware_session": HardwareSessionConfig(armed=False, bringup_timeout_s=20.0),
            "twin_overlay": TwinOverlayConfig(enabled=False),
        }
    )
    runtime = Runtime(cfg, monitor_factory=FakeMonitorFactory(_samples()))
    assert runtime.manager.workcell_factory is None
    with TestClient(create_app(runtime)) as c:
        assert _wait(lambda: runtime.hardware_monitor.status_of("grip")[0] == "running")
        r = c.post("/api/session", json=spec())
        assert r.status_code == 409 and "not armed" in r.json()["detail"], r.text
        r = c.post(URL.format("grip"), json={"op": "home_rail"})
        assert r.status_code == 409 and "not armed" in r.json()["detail"], r.text
        r = c.post(URL.format("grip"), json={"op": "home_rail", "dry_run": True})
        assert r.status_code == 200, r.text  # kinematics only, zero writes
        assert runtime.manager.session is None
        assert runtime.hardware_monitor.paused is False
