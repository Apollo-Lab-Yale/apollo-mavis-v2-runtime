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
(planned on the gate twin inside bring-up, executed through ``_op_execute_plan`` one
arm at a time in the planner's ``arm_order`` - 2026-09-08 evening - 409 on a plan
failure), the 2026-09-12 admission of POLICY-DRIVEN motion behind
``hardware_session.policy_modes`` (refused with the knob named while it is false; with it
true an external inference session and an Online DAgger session come up over the same
``GatedPolicyExecutor`` stack as sim, the speed scale still bounding the drivers, the
rollouts recorded from the adopted wrist cameras) and - optional, skipped without the
hardware package's test fakes - the REAL ``HardwareWorkcell`` + ``XArmDriver`` over
``FakeXArmAPI``."""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from apollo_mavis_v2_core import HeldState, WorkcellConfig
from apollo_mavis_v2_core.schemas.config import CameraIntrinsics
from apollo_mavis_v2_core.testing import FakeArm, FakeCamera
from conftest import FakeMonitorFactory, FakeMonitorSample, make_runtime_config
from fakes import FakeServoLimits, HardwareFakeWorkcell
from starlette.testclient import TestClient

from apollo_mavis_v2_runtime.config import (
    ControlConfig,
    DatasetNamespaceConfig,
    DatasetsConfig,
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
            # the lab's D435 colour intrinsics scaled to the 64x48 test frames (x 0.1)
            "intrinsics": {"fx": 60.819, "fy": 60.823, "cx": 32.739, "cy": 24.790},
        },
        {
            "id": "view_wrist",
            "kind": "v4l2",
            "serial": "322143060792",
            "fourcc": "YUYV",
            "resolution": [64, 48],
            "fps": 30,
            "intrinsics": {"fx": 60.636, "fy": 60.638, "cx": 31.190, "cy": 24.945},
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
            # the shipped Online DAgger layout (<root>/<s>/rollouts, 15-online-dagger D5) for
            # the 2026-09-12 policy-mode tests; demonstrations keep the tmp datasets_root
            "datasets": DatasetsConfig(
                default_namespace="apollo",
                namespaces={
                    "online_dagger": DatasetNamespaceConfig(root=tmp_path / "od", subdir="rollouts")
                },
            ),
        }
    )


def _wait(pred, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


class LiveClockCamera(FakeCamera):
    """FakeCamera whose frames carry the LIVE monotonic clock: the recorder drops a
    frame older than 2/fps, so the deterministic internal clock would starve it."""

    def latest(self):
        frame = super().latest()
        if frame is None:
            return None
        from apollo_mavis_v2_core import CameraFrame

        return CameraFrame(
            camera_id=frame.camera_id, rgb=frame.rgb, t_mono=time.monotonic(),
            wallclock_ns=time.time_ns(), seq=frame.seq,
        )


@pytest.fixture(scope="module")
def cams():
    return {cid: LiveClockCamera(cid, (64, 48), 30.0) for cid in ("grip_wrist", "view_wrist")}


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


def test_executor_rail_slew_is_capped_at_the_tracks_positioning_speed():
    """2026-09-08 review: the executor slewed the rail slot at ``jog.rail_m_per_tick`` x
    scale = 0.2 m/s at 100 % while the track positions at ``rail_speed_mm_s`` (50 x scale)
    toward latest-wins targets, so an arm was declared arrived up to ~10 s before its
    carriage physically got there. The connected driver's (already speed-scaled)
    ``rail_speed_mm_s`` now bounds the rail slew per loop tick; a driver without one
    (fakes, sim) leaves the host value alone."""
    from types import SimpleNamespace

    from apollo_mavis_v2_runtime.session.hardware import apply_executor_caps, executor_caps_for

    cfg = ControlConfig()  # 100 Hz, host rail 0.002 m/tick
    railed = SimpleNamespace(cfg=SimpleNamespace(servo=FakeServoLimits(), rail_speed_mm_s=50))
    caps = executor_caps_for([railed], cfg)
    assert caps.source == "servo" and caps.rail_m_per_tick == pytest.approx(0.0005)
    out = apply_executor_caps(cfg, caps)
    assert out.jog.rail_m_per_tick == pytest.approx(0.0005)  # 0.002 -> 50 mm/s / 100 Hz
    assert out.jog.slew_rad_per_tick == pytest.approx(0.0003)  # the joint cap as before
    # two drivers: the slower track wins; a scaled driver config (5 mm/s at 10 %) follows
    slow = SimpleNamespace(cfg=SimpleNamespace(servo=FakeServoLimits(), rail_speed_mm_s=5))
    assert executor_caps_for([railed, slow], cfg).rail_m_per_tick == pytest.approx(0.00005)
    # a host value already below the cap stands (min, never raised)
    scaled = scale_control_config(cfg, 0.1)  # host rail 0.0002
    assert apply_executor_caps(scaled, caps).jog.rail_m_per_tick == pytest.approx(0.0002)
    # no driver publishes a rail speed: None, and the host value is untouched
    plain = SimpleNamespace(cfg=SimpleNamespace(servo=FakeServoLimits()))
    none_caps = executor_caps_for([plain], cfg)
    assert none_caps.rail_m_per_tick is None
    assert apply_executor_caps(cfg, none_caps).jog.rail_m_per_tick == cfg.jog.rail_m_per_tick
    host_only = executor_caps_for([SimpleNamespace()], cfg)
    assert host_only.source == "host" and host_only.rail_m_per_tick is None
    # a rail speed without servo limits (host joint caps) still caps the rail
    rail_only = executor_caps_for([SimpleNamespace(cfg=SimpleNamespace(rail_speed_mm_s=50))], cfg)
    assert rail_only.source == "host" and rail_only.rail_m_per_tick == pytest.approx(0.0005)


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


def test_workcell_status_kind_sim_survives_a_hardware_session(client, rt, cells):
    """`GET /api/workcell?kind=sim` 500ed while a hardware session ran: the sim arm
    rows reused `session.workcell.scene` and `HardwareWorkcell` has none (seen live
    2026-09-07, the Welcome page polls this endpoint). Sim rows now come from the
    preview scene, with connected=False - no sim session owns those arms."""
    r = client.post("/api/session", json=spec())
    assert r.status_code == 200, r.text
    try:
        for params in (None, {"kind": "sim"}, {"kind": "hardware"}):
            resp = client.get("/api/workcell", params=params)
            assert resp.status_code == 200, (params, resp.text)
        sim = client.get("/api/workcell", params={"kind": "sim"}).json()
        assert sim["kind"] == "sim" and all(a["connected"] is False for a in sim["arms"])
        hw = client.get("/api/workcell", params={"kind": "hardware"}).json()
        assert all(a["connected"] is True for a in hw["arms"])  # the session owns these
        assert client.get("/api/workcell").json()["kind"] == "hardware"  # no kind: the session's
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


_KEY_SEQ = [0]


def drive_keys(rt, code: str, seconds: float) -> None:
    """Hold one keyboard code on the running session the way the Cockpit does: an
    EMPTY set first (clears a deadman latch left by the silence since the last hold),
    the code at 50 Hz, an EMPTY set last; one monotonic seq across calls."""
    session = rt.manager.session
    wd = session.supervisor.watchdog

    def send(held: set[str]) -> None:
        _KEY_SEQ[0] += 1
        hs = HeldState(held=frozenset(held), seq=_KEY_SEQ[0], rx_mono=time.monotonic())
        if wd.on_keys(hs):
            rt.bus.held_keys.put(hs)

    send(set())
    time.sleep(0.03)
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        send({code})
        time.sleep(0.02)
    send(set())


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
        # the same pre-planned waypoints, ONE ARM PER execute_plan in the planner's
        # arm_order (2026-09-08 evening), the gripper targets on the last arm
        assert [list(c["waypoints"]) for c in executed] == [[a] for a in waypoints]
        assert {a: w for c in executed for a, w in c["waypoints"].items()} == waypoints
        assert [c["gripper"] for c in executed] == [{}] * (len(executed) - 1) + [grippers]
        assert len(plans) == 1  # the worker did NOT plan again
        # the loop drove the commanded posture to the profile target
        loop = session.loop
        assert loop._last_cmd["grip"][:7] == pytest.approx(list(target), abs=1e-6)
        assert loop._last_cmd["view"][:7] == pytest.approx(list(target), abs=1e-6)
    finally:
        assert client.delete("/api/session").status_code == 204
    assert _wait(lambda: not rt.hardware_monitor.paused)


def test_start_from_refused_by_a_latched_fault_reaches_the_wire_and_survives_recovery(
    client, rt, cells, monkeypatch
):
    """Operator problem #2 (2026-09-08): after a refused profile start the Cockpit showed
    a running session, held arms and no message - the refusal lived on the manager only.
    Over the real ``/ws/telemetry`` (and ``GET /api/session``): the exact text rides
    ``session.fault_detail`` while the arm is faulted, is STILL there after "Clear errors
    & resume" (the fault callback wipes its own text, not the notice), and is gone once
    "Go to profile" - the retry the text asks for - arrives."""
    monkeypatch.setattr(
        rt.manager,
        "cfg",
        rt.manager.cfg.model_copy(
            update={
                "hardware_session": HardwareSessionConfig(
                    armed=True, bringup_timeout_s=20.0, start_from_fault_grace_s=0.3
                )
            }
        ),
    )
    target = (PI, -0.3, 0.0, 0.4, 0.0, 0.7, 0.0)  # a lifted, still folded posture
    profile = _profile(rt, "lifted-refused", target)
    real_exec = ControlLoop._op_execute_plan
    latched: list[str] = []

    def latch_c24_as_the_plan_arrives(self, cmd):
        if not latched:  # the driver latches C24 on the Manipulation Arm exactly then
            latched.append("grip")
            cells.cell.fault("grip", 24)  # error_code 24 in the driver's view + the event
            self._faulted.add("grip")  # what the event's drain leaves behind, a tick early
        return real_exec(self, cmd)

    monkeypatch.setattr(ControlLoop, "_op_execute_plan", latch_c24_as_the_plan_arrives)
    r = client.post("/api/session", json=spec(start_from=f"profile:{profile.profile_id}"))
    assert r.status_code == 200, r.text
    session = rt.manager.session
    ctrl_state = cells.cell.states()["grip"].state
    expected = (
        f"start_from refused: Manipulation Arm faulted (controller state {ctrl_state}, code "
        "C24) - use Clear errors & resume, then Go to profile"
    )
    msg = _telemetry(client, lambda m: m["session"]["fault_detail"] != "", frames=250)
    assert msg["session"]["fault_detail"] == expected
    assert session.motion_detail == expected and latched == ["grip"]
    assert _wait(lambda: rt.manager.state is SessionState.FAULT)  # the event was drained
    assert client.get("/api/session").json()["fault_detail"] == expected
    assert session.start_from_progress is None and rt.manager.bringup_telemetry() is None
    assert not session.loop.plans.active_arms  # nothing moved
    # the arm row carries the C24; the session row is the refusal, never a duplicate
    msg = _telemetry(client, lambda m: m["session"]["state"] == "fault")
    grip = next(a for a in msg["arms"] if a["arm_id"] == "grip")
    assert grip["error_code"] == 24 and grip["fault_detail"].startswith("controller error 24")
    assert msg["session"]["fault_detail"] == expected
    # "Clear errors & resume": RUNNING again, the hint is STILL on the wire
    r = client.post(URL.format("grip"), json={"op": "recover"})
    assert r.status_code == 200 and r.json()["ok"] is True, r.text
    assert _wait(lambda: rt.manager.state is SessionState.RUNNING)
    assert session.fault_detail == "" and session.motion_detail == expected
    msg = _telemetry(client, lambda m: m["session"]["state"] == "running")
    assert msg["session"]["fault_detail"] == expected
    grip = next(a for a in msg["arms"] if a["arm_id"] == "grip")
    assert grip["fault_detail"] == "" and grip["error_code"] == 0
    # "Go to profile" is the retry: on arrival the notice is gone
    assert rt.manager.request_goto_profile(profile) == (
        True, "going to profile 'lifted-refused'"
    )
    assert _wait(lambda: rt.manager.profile_motion_in_flight is None, 60.0)
    assert session.motion_detail == "", session.motion_detail
    assert session.loop._last_cmd["grip"][:7] == pytest.approx(list(target), abs=1e-6)
    assert session.loop._last_cmd["view"][:7] == pytest.approx(list(target), abs=1e-6)
    msg = _telemetry(client, lambda m: m["session"]["fault_detail"] == "")
    assert client.get("/api/session").json()["fault_detail"] == ""


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

    refused(spec(mode="dagger", task="t"), "hardware sessions support teleop and data collection")
    refused(spec(mode="inference"), "hardware sessions support teleop and data collection")
    # collect is admitted (2026-09-07); its own contract still refuses before any box is touched
    refused(spec(mode="collect", task="t", dataset="x"), "return_to_start needs a start_from")
    refused(spec(mode="collect", task="t", dataset="x", return_to_start=False, dataset_resume=True),
            "unknown dataset")
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


def test_collect_session_records_episode_directories_from_the_adopted_cameras(
    client, rt, cams, cells
):
    """Data collection on the real cell (2026-09-07; 04-runtime §10.5): the recorder is
    built inside bring-up from the ADOPTED preview cameras and the gate twin's
    kinematics; both wrist cameras land in ``episodes/<id>/video/``; the dataset is
    ``in_use`` and a previous episode can be deleted mid-session; the export is
    refused while the session records."""
    from apollo_mavis_v2_core import Command

    r = client.post(
        "/api/session",
        json=spec(mode="collect", task="hw pick", dataset="hw_pick", return_to_start=False),
    )
    assert r.status_code == 200, r.text
    session = rt.manager.session
    rt_thread = session.recorder_thread
    root = rt.cfg.datasets_root / "apollo" / "hw_pick"
    try:
        assert rt_thread is not None and rt_thread.repo_id == "apollo/hw_pick"
        assert _wait(lambda: rt.manager.state is SessionState.RUNNING, 20.0)
        assert (root / "manifest.json").exists() and (root / "sessions").is_dir()
        rows = {d["repo_id"]: d for d in client.get("/api/datasets").json()}
        row = rows["apollo/hw_pick"]
        assert row["in_use"] is True and row["kind"] == "hardware"
        assert row["cameras"] == ["grip_wrist", "view_wrist"]
        assert client.post("/api/datasets/apollo/hw_pick/export", json={}).status_code == 409

        def episode(op: str):
            return rt.bus.commands.submit(Command(op=op, source="ws")).result(timeout=5.0)

        saved = []
        for code in ("KeyE", "KeyQ"):
            ack = episode("episode_new")
            assert ack.ok, ack
            time.sleep(0.6)  # prepare (encoder open) before the first key
            drive_keys(rt, code, 2.0)  # a still arm records nothing: the idle filter is ON
            #   (speed scale 0.1: ~0.5 mm per capture -> one kept frame per ~2 captures)
            assert _wait(lambda: rt_thread.status().frames >= 12, 15.0), (
                rt_thread.status(), rt_thread.filter.summary(),
            )
            ack = episode("episode_save")
            assert ack.ok, ack
            assert _wait(lambda: rt_thread.status().state == "idle", 15.0)
            saved.append(rt_thread.last_saved_id)
        assert rt_thread.status().total_episodes == 2
        eps = client.get("/api/datasets/apollo/hw_pick/episodes").json()
        assert [e["episode_id"] for e in eps] == saved and all(e["export_ok"] for e in eps)
        d = root / "episodes" / saved[0]
        videos = sorted(p.name for p in (d / "video").iterdir())
        assert videos == ["grip_wrist.mp4", "view_wrist.mp4"]
        ep = json.loads((d / "episode.json").read_text())
        assert ep["length"] >= 12 and set(ep["video"]) == {"grip_wrist", "view_wrist"}
        assert set(ep["extrinsics"]) == {"grip_wrist", "view_wrist"}
        assert ep["extrinsics"]["grip_wrist"]["twin_camera"] == "grip_wrist_cam"
        # the configured D435 intrinsics travel into the sidecar verbatim
        cams_cfg = {c["id"]: c["intrinsics"] for c in HW_WORKCELL["cameras"]}
        for cam_id in ("grip_wrist", "view_wrist"):
            assert ep["extrinsics"][cam_id]["intrinsics"] == (
                CameraIntrinsics(**cams_cfg[cam_id]).model_dump()
            )
        assert ep["filter"]["enabled"] is True
        assert ep["frames"] == ALL_FRAMES and ep["arm_bases"]["grip"]["has_rail"] is True
        # a previous episode can go while the session records the next one
        assert episode("episode_new").ok
        r = client.delete(f"/api/datasets/apollo/hw_pick/episodes/{saved[0]}")
        assert r.status_code == 204 and not d.exists()
        assert episode("episode_discard").ok
        assert _wait(lambda: rt_thread.status().state == "idle", 15.0)
    finally:
        assert client.delete("/api/session").status_code == 204
    assert _wait(lambda: not rt.hardware_monitor.paused)
    assert sorted(p.name for p in (root / "episodes").iterdir()) == [saved[1]]
    assert client.get("/api/datasets/apollo/hw_pick").json()["in_use"] is False
    assert client.post("/api/datasets/apollo/hw_pick/export", json={}).status_code == 202
    assert rt.manager.dataset_store.wait_export(120.0)
    assert client.get("/api/datasets/apollo/hw_pick").json()["export"]["state"] == "fresh"


def test_collect_without_a_live_camera_is_refused_before_any_box_is_touched(
    client, rt, cells, monkeypatch
):
    """Review item 14: the camera check moved into the refusal matrix — no workcell is
    built, no monitor paused, when no preview is live."""
    monkeypatch.setattr(rt.manager, "hardware_camera", lambda cid: None)
    r = client.post(
        "/api/session",
        json=spec(mode="collect", task="t", dataset="no_cam", return_to_start=False),
    )
    assert r.status_code == 409, r.text
    assert "needs at least one live hardware camera" in r.json()["detail"]
    assert cells.built == [] and not rt.hardware_monitor.paused and rt.manager.session is None


def test_collect_session_records_audio_from_a_fake_microphone(client, rt, cams, cells):
    """D8: the Runtime-owned MicrophoneReader feeds the per-episode audio sink. With a
    fake-backend reader assigned (the lab wiring is Runtime.__init__), every saved
    episode carries audio.wav + the alignment block."""
    from apollo_mavis_v2_core import Command, LatestSlot

    from apollo_mavis_v2_runtime.devices.microphone import MicrophoneReader

    mic_cfg = MicrophoneConfig(enabled=True, backend="fake")
    mic = MicrophoneReader(mic_cfg, LatestSlot(), frame_hz=25)
    mic.start()
    previous = rt.manager.microphone
    rt.manager.microphone = mic
    root = rt.cfg.datasets_root / "apollo" / "hw_audio"
    try:
        r = client.post(
            "/api/session",
            json=spec(
                mode="collect", task="hw audio", dataset="hw_audio", return_to_start=False,
                action_filter={"enabled": False},  # a still arm: nothing would be recorded
            ),
        )
        assert r.status_code == 200, r.text
        session = rt.manager.session
        rt_thread = session.recorder_thread
        assert rt_thread.audio is not None
        assert _wait(lambda: rt.manager.state is SessionState.RUNNING, 20.0)

        def episode(op: str):
            return rt.bus.commands.submit(Command(op=op, source="ws")).result(timeout=5.0)

        assert episode("episode_new").ok
        assert _wait(lambda: rt_thread.status().frames >= 12, 15.0), (
            rt_thread.status(), rt_thread.filter.summary(),
        )
        assert episode("episode_save").ok
        assert _wait(lambda: rt_thread.status().state == "idle", 15.0)
        eid = rt_thread.last_saved_id
        d = root / "episodes" / eid
        ep = json.loads((d / "episode.json").read_text())
        assert ep["audio"] is not None and (d / "audio.wav").exists()
        assert ep["audio"]["sample_rate"] == 48000 and ep["audio"]["samples"] > 48000 // 4
        assert abs(ep["audio"]["duration_s"] - ep["duration_s"]) < 1.5  # the look-ahead drift
        assert client.get("/api/datasets/apollo/hw_audio/episodes").json()[0]["audio"] is True
    finally:
        client.delete("/api/session")
        rt.manager.microphone = previous
        mic.stop()
    assert _wait(lambda: not rt.hardware_monitor.paused)


def test_collect_start_from_profile_returns_after_save_and_a_fault_cancels_it(
    client, rt, cams, cells
):
    """Return-to-start on the fake cell at speed scale 0.5: start_from profile, drive
    with keyboard codes during the episode, save -> saving -> returning -> idle and the
    arm is back; a driver fault mid-return cancels the return on EVERY arm ("return
    cancelled: driver fault"); the loop keeps its tick rate throughout."""
    from apollo_mavis_v2_core import Command, HeldState

    target = (PI, -0.3, 0.0, 0.4, 0.0, 0.7, 0.0)
    profile = _profile(rt, "lifted-collect", target)
    r = client.post(
        "/api/session",
        json=spec(
            mode="collect", task="hw return", dataset="hw_return",
            start_from=f"profile:{profile.profile_id}", speed_scale=0.5,
        ),
    )
    assert r.status_code == 200, r.text
    assert r.json()["speed_scale"] == 0.5
    session = rt.manager.session
    loop, rt_thread, cell = session.loop, session.recorder_thread, cells.cell
    try:
        assert _wait(lambda: rt.manager.state is SessionState.RUNNING, 20.0)
        assert _wait(lambda: not loop.plans.active_arms, 20.0)
        assert loop._last_cmd["grip"][:7] == pytest.approx(list(target), abs=1e-6)

        def episode(op: str):
            return rt.bus.commands.submit(Command(op=op, source="ws")).result(timeout=5.0)

        def drive(code: str, seconds: float) -> None:
            drive_keys(rt, code, seconds)

        states: list[str] = []

        def watch(until: str, timeout: float = 30.0) -> None:
            t0 = time.monotonic()
            while time.monotonic() - t0 < timeout:
                st = rt_thread.status().state
                if not states or states[-1] != st:
                    states.append(st)
                if st == until and len(states) > 1:
                    return
                time.sleep(0.005)
            raise AssertionError(f"never reached {until}: {states}")

        # -- episode 1: drive up, save, return to the profile ---------------------------
        ack = episode("episode_new")
        assert ack.ok, (ack.detail, rt_thread.status())
        time.sleep(0.6)  # prepare (encoder open) before the first key
        ticks0, t0 = loop.tick_count, time.monotonic()
        drive("KeyE", 1.5)
        rate = (loop.tick_count - ticks0) / (time.monotonic() - t0)
        assert 95.0 <= rate <= 103.0, f"control loop {rate:.1f} Hz during hardware collect"
        print(f"REPORT hardware fake collect: control loop {rate:.1f} Hz at speed 0.5")
        moved = float(np.max(np.abs(loop._last_cmd["grip"][:7] - np.asarray(target))))
        assert moved > 0.02, moved
        states.clear()
        beat = threading.Event()

        def heartbeat() -> None:  # the Cockpit's 25 Hz empty heartbeat: no deadman latch
            while not beat.is_set():
                _KEY_SEQ[0] += 1
                hs = HeldState(held=frozenset(), seq=_KEY_SEQ[0], rx_mono=time.monotonic())
                if session.supervisor.watchdog.on_keys(hs):
                    rt.bus.held_keys.put(hs)
                beat.wait(0.04)

        hb = threading.Thread(target=heartbeat, daemon=True)
        hb.start()
        try:
            assert episode("episode_save").ok
            watch("idle")
        finally:
            beat.set()
            hb.join(1.0)
        assert states[states.index("saving"):] == ["saving", "returning", "idle"], states
        assert loop._last_cmd["grip"][:7] == pytest.approx(list(target), abs=0.02)
        assert rt_thread.status().detail == ""

        # -- episode 2: a driver fault on the Perception Arm mid-return -----------------
        assert episode("episode_new").ok
        time.sleep(0.6)
        drive("KeyE", 1.5)
        states.clear()
        beat = threading.Event()
        hb = threading.Thread(target=heartbeat, daemon=True)
        hb.start()
        try:
            assert episode("episode_save").ok
            # Sequential execution (2026-09-08 evening): the return walks ONE arm at a
            # time in the planner's order. The fake cell's MEASURED postures follow the
            # command (``HardwareFakeWorkcell._follow``; the manager hands over on
            # measured arrival), so the Perception Arm - never moved by the keys - is
            # already at the profile and is not submitted at all: the Manipulation Arm
            # is the only moving arm. Wait for its plan, then fault the sibling.
            t0 = time.monotonic()
            while rt_thread.status().state != "returning" or "grip" not in loop.plans.active_arms:
                assert time.monotonic() - t0 < 30.0, (rt_thread.status(), loop.plans.active_arms)
                time.sleep(0.002)
            assert loop.plans.active_arms == ["grip"]  # never two arms in the executor
            cell.fault("view", 31, source="monitor")  # FaultScript: the sibling faults
            watch("idle")
        finally:
            beat.set()
            hb.join(1.0)
        assert loop.plans.active_arms == []  # an interruptible plan stops on ANY arm's fault
        # a single moving arm: the detail reads exactly as before the sequential change
        assert rt_thread.status().detail == "return cancelled: driver fault"
        assert rt.manager.state in (SessionState.FAULT, SessionState.RECOVERING)
    finally:
        assert client.delete("/api/session").status_code == 204
    assert _wait(lambda: not rt.hardware_monitor.paused)


# -- policy-driven motion on the real arms (operator decision 2026-09-12; 15-online-dagger D7) --
POLICY_MODES_REFUSAL = "hardware sessions support teleop and data collection only"


@pytest.fixture()
def policy_modes(rt):
    """``hardware_session.policy_modes`` ON for one test (what the lab render's
    ``HARDWARE_POLICY_MODES=true`` does); the module config keeps the repo default False."""
    hs = rt.manager.cfg.hardware_session
    assert hs.policy_modes is False
    hs.policy_modes = True
    try:
        yield hs
    finally:
        hs.policy_modes = False


def _both_arm_announce(version: int = 1, capabilities=()):
    """A ``PolicySpecAnnounce`` whose policy drives BOTH cell arms in their base frames
    (the hubfakes default names a single ``arm0``)."""
    from dora_bridge.hubfakes import announce

    from apollo_mavis_v2_runtime.recorder.features import arm_action_names, arm_state_names

    names = [n for a in ALL_ARMS for n in arm_action_names(a, True, "delta_ee")]
    states = [n for a in ALL_ARMS for n in arm_state_names(a, True)]
    ann = announce(
        version=version,
        frame="arm_base:grip",
        names=names,
        arms=list(ALL_ARMS),
        action_frames=dict(ALL_FRAMES),
        capabilities=list(capabilities),
    )
    return ann.model_copy(update={"spec": ann.spec.model_copy(update={"state_names": states})})


def _announce(dora, version: int = 1, *capabilities: str) -> None:
    from dora_bridge.hubfakes import spec_event

    dora.policy_hub._on_spec(spec_event(_both_arm_announce(version, capabilities)))


@pytest.fixture()
def external_policy(rt, tmp_path):
    """A FAKE dora wiring (no bus thread; ``test_online_dagger_session.FakeDora``) on the
    manager - the seam every ``policy_source: external`` session goes through."""
    from test_online_dagger_session import FakeDora

    dora = FakeDora(tmp_path)
    rt.manager.dora = dora
    try:
        yield dora
    finally:
        rt.manager.dora = None


def _action(rt, op: str):
    from apollo_mavis_v2_core import Command

    return rt.bus.commands.submit(Command(op=op, source="ws")).result(timeout=5.0)


def test_policy_modes_off_refuses_inference_and_dagger_before_touching_anything(
    client, rt, factory, cells
):
    """D7 as shipped (repo default ``policy_modes: false``): inference / dagger / an Online
    DAgger body are 409 with the knob NAMED, before the monitor is paused or a box is
    enabled - the mode line stays FIRST in the refusal matrix (no session directory either)."""
    grip = factory.monitors["grip"]
    n0 = len(grip.calls)
    assert rt.manager.cfg.hardware_session.policy_modes is False
    bodies = (
        spec(mode="inference"),
        spec(mode="inference", policy_source="external"),
        spec(mode="dagger", task="t", return_to_start=False),
        spec(
            mode="dagger",
            task="t",
            policy_source="external",
            return_to_start=False,
            online_dagger={"session_name": "never"},
        ),
    )
    for body in bodies:
        r = client.post("/api/session", json=body)
        assert r.status_code == 409, r.text
        detail = r.json()["detail"]
        assert POLICY_MODES_REFUSAL in detail and "hardware_session.policy_modes" in detail, detail
    assert cells.built == [] and grip.calls[n0:] == [] and not rt.hardware_monitor.paused
    assert rt.manager.session is None
    assert not (rt.manager.online_dagger_root() / "never").exists()


def test_policy_modes_on_still_resolves_the_policy_before_any_box_is_touched(
    client, rt, factory, cells, policy_modes
):
    """With the knob on the policy is resolved INSIDE the refusal matrix - an unknown
    checkpoint or no attached policy node is 409 before the monitor is paused."""
    grip = factory.monitors["grip"]
    n0 = len(grip.calls)
    r = client.post("/api/session", json=spec(mode="inference", policy="nope/deploy/v001"))
    assert r.status_code == 409 and "unknown policy 'nope/deploy/v001'" in r.json()["detail"]
    r = client.post(
        "/api/session",
        json=spec(mode="dagger", task="t", return_to_start=False, policy="nope/v000001"),
    )
    assert r.status_code == 409 and "unknown policy 'nope/v000001'" in r.json()["detail"]
    r = client.post("/api/session", json=spec(mode="inference", policy_source="external"))
    assert r.status_code == 409, r.text
    assert "no external policy attached (dora bridge is not attached)" in r.json()["detail"]
    assert cells.built == [] and grip.calls[n0:] == [] and not rt.hardware_monitor.paused
    assert rt.manager.session is None


def test_external_inference_session_is_admitted_with_policy_modes(
    client, rt, factory, cams, cells, policy_modes, external_policy
):
    """Operator decision 2026-09-12: with the knob ON an inference session over the external
    policy node comes up on the real cell with the SAME stack as sim - a
    ``GatedPolicyExecutor`` of kind hardware over the rig's speed-scaled, servo-capped
    control config and the gate twin; the speed scale still bounds the drivers; the
    explicit ``takeover`` / ``handback`` publish ``events.gate``; teardown is the D6
    hand-back like any teleop session (loop -> drivers -> fps back -> monitor resumed)."""
    from apollo_mavis_v2_runtime.dagger.loop import GatedPolicyExecutor, InferenceSession
    from apollo_mavis_v2_runtime.dora_bridge.policy_source import ExternalPolicySource

    dora = external_policy
    _announce(dora, 1)
    grip = factory.monitors["grip"]
    hub_ids_before = set(rt.hub.ids())
    r = client.post("/api/session", json=spec(mode="inference", policy_source="external"))
    assert r.status_code == 200, r.text
    info = r.json()
    assert (info["mode"], info["kind"], info["policy_source"]) == (
        "inference",
        "hardware",
        "external",
    )
    assert info["speed_scale"] == 0.1 and info["streams"] == []
    session = rt.manager.session
    cell = cells.cell
    try:
        loop = session.loop
        assert isinstance(loop, GatedPolicyExecutor) and loop.session_mode == "inference"
        assert loop.workcell_kind == "hardware" and loop.gripper_arms == {"grip"}
        assert isinstance(session.supervisor.gate, SafetyGate)
        assert loop.supervisor is session.supervisor and loop.planner is session.twin
        assert loop.ik is not None and loop.kin is not None and loop.workcell is cell
        assert isinstance(session.policy_session, InferenceSession)
        assert isinstance(loop.runner, ExternalPolicySource)
        assert session.recorder_thread is None and session.online_dagger is None
        # the speed scale (0.1) bounds the loop exactly like a teleop session's (host-only
        # caps on these fakes), the anchor leash stays the unscaled one, and the driver
        # factory carries the scaled servo caps to the (fake) drivers
        assert loop.cfg.dq_max_rad == pytest.approx(0.004)
        assert loop.cfg.teleop.linear_mps == pytest.approx(0.012)
        assert loop.cfg.target_rate.v_mps == pytest.approx(0.1)
        assert loop.cfg.leash == rt.cfg.control.leash
        assert loop.anchor.leash_pos_m == pytest.approx(rt.cfg.control.leash.pos_m)
        if importlib.util.find_spec("apollo_mavis_v2_hardware") is not None:
            import apollo_mavis_v2_hardware as hw

            base = hw.XArmDriverConfig(arm_id="grip", ip="192.168.1.201", gripper="xarm_g2")
            driver = cells.driver_factories[0](base)  # inert: nothing connects in the ctor
            assert driver.cfg.servo.max_joint_vel == pytest.approx((0.06,) * 7)
            assert driver.cfg.servo.max_cart_step_m == pytest.approx(0.0004)
            assert driver.cfg.rail_speed_mm_s == 5
        assert rt.hardware_monitor.paused and rt.manager.hardware_session_active
        assert session.adopted_streams == ["grip_wrist", "view_wrist"]
        assert _wait(lambda: rt.manager.state is SessionState.RUNNING, 20.0)
        msg = _telemetry(
            client, lambda m: m["session"]["state"] == "running" and m.get("inference")
        )
        assert msg["dagger"] is None and msg["episode"] is None
        assert msg["inference"]["control_mode"] == "policy"
        assert msg["collision"] is not None  # the gate twin is live (unconditional)
        assert dora.calls[:2] == ["before_bringup", "after_session_start"]
        assert dora.facts.kind == "hardware" and dora.facts.policy_source == "external"
        assert dora.facts.camera_ids == ["grip_wrist", "view_wrist"]
        for op in ("episode_new", "episode_save", "episode_discard"):
            assert _action(rt, op).ok is False  # no recorder in inference
        assert _action(rt, "takeover").ok
        assert _wait(lambda: any(g["mode"] == "human" for g in dora.publisher.of("gate")), 5.0)
        _telemetry(client, lambda m: (m.get("inference") or {}).get("control_mode") == "human")
        assert _action(rt, "handback").ok
        assert _wait(lambda: dora.publisher.of("gate")[-1]["mode"] == "policy", 5.0)
        assert all(g["arm_id"] == "grip" for g in dora.publisher.of("gate"))
    finally:
        assert client.delete("/api/session").status_code == 204
    assert cell.stop_calls == 1 and rt.manager.session is None
    assert rt.hub._workers["grip_wrist"].fps == 15.0 and set(rt.hub.ids()) == hub_ids_before
    assert _wait(lambda: not rt.hardware_monitor.paused)
    assert not rt.manager.hardware_session_active and grip.calls[-1] == "start"
    assert dora.calls[-1] == "after_teardown"


def test_online_dagger_session_is_admitted_with_policy_modes_and_records_rollouts(
    client, rt, factory, cams, cells, policy_modes, external_policy
):
    """D7 amended: an Online DAgger body comes up on the real cell - the executor is a
    ``GatedPolicyExecutor`` of kind hardware carrying the ``OnlineDaggerCoordinator``, the
    rollouts are recorded from the ADOPTED wrist cameras into ``online_dagger/<s>/rollouts``
    (both videos, the twin camera extrinsics), ``episode_new`` waits for the trainer's
    ``ready`` exactly as in sim, a kept rollout publishes ``episode_saved`` with the
    ``online_dagger`` block, and teardown leaves ``session.json`` behind."""
    from apollo_mavis_v2_runtime.dagger.loop import DaggerSession, GatedPolicyExecutor

    dora = external_policy
    _announce(dora, 1, "online_dagger")
    sdir = rt.manager.online_dagger_root() / "hw1"
    body = spec(
        mode="dagger",
        task="hw rollout",
        policy_source="external",
        return_to_start=False,
        action_filter={"enabled": False},  # a held arm records every frame
        online_dagger={"session_name": "hw1"},
    )
    r = client.post("/api/session", json=body)
    assert r.status_code == 200, r.text
    sid = r.json()["session_id"]
    session = rt.manager.session
    cell = cells.cell
    rt_thread = session.recorder_thread
    try:
        loop = session.loop
        assert isinstance(loop, GatedPolicyExecutor) and loop.session_mode == "dagger"
        assert loop.workcell_kind == "hardware" and loop.gripper_arms == {"grip"}
        assert isinstance(session.policy_session, DaggerSession)
        assert session.online_dagger is not None
        assert loop.coordinator is session.online_dagger.coordinator
        assert rt_thread is not None and loop.recorder is rt_thread
        assert rt_thread.repo_id == "online_dagger/hw1"
        assert sorted(p.name for p in sdir.iterdir()) == ["rollouts", "session.json"]
        assert loop.cfg.dq_max_rad == pytest.approx(0.004)  # the speed scale, as in teleop
        assert _wait(lambda: rt.manager.state is SessionState.RUNNING, 20.0)
        rows = {d["repo_id"]: d for d in client.get("/api/datasets").json()}
        row = rows["online_dagger/hw1"]
        assert row["kind"] == "hardware" and row["in_use"] is True
        assert row["cameras"] == ["grip_wrist", "view_wrist"]  # the adopted previews
        # the trainer gate, exactly as in sim: no status yet -> "no trainer"; a trainer that
        # serves this session but is not ready -> "waiting"; its first `ready` opens rollouts
        from apollo_mavis_v2_runtime.dagger.online_dagger import NO_TRAINER

        ack = _action(rt, "episode_new")
        assert ack.ok is False and ack.detail == NO_TRAINER, ack
        dora.beat("preparing", progress=0.5, detail="offline pool 1/2", session_id=sid)
        ack = _action(rt, "episode_new")
        assert ack.ok is False, ack
        assert ack.detail == "waiting for the trainer to report ready (offline pool 1/2)"
        dora.beat("ready", session_id=sid)
        assert _wait(lambda: _action(rt, "episode_new").ok, 5.0)
        assert _wait(lambda: rt_thread.status().frames >= 10, 15.0), rt_thread.status()
        assert _action(rt, "episode_save").ok
        assert _wait(lambda: rt_thread.status().state == "idle", 15.0)
        saved = rt_thread.last_saved_id
        assert _wait(lambda: dora.publisher.of("episode_saved"), 5.0)
        [ev] = dora.publisher.of("episode_saved")
        assert ev["summary"]["episode_id"] == saved and ev["summary"]["n_frames"] >= 10
        assert ev["summary"]["n_novice_frames"] == ev["summary"]["n_frames"]  # policy held
        od = ev["online_dagger"]
        assert od["episode_id"] == saved and od["rollouts_saved"] == 1
        assert od["actor_counts"] == {"novice": ev["summary"]["n_frames"], "expert": 0}
        assert ev["dataset_root"] == str(sdir / "rollouts") and ev["run_id"] == sid[:8]
        d = sdir / "rollouts" / "episodes" / saved
        assert sorted(p.name for p in (d / "video").iterdir()) == [
            "grip_wrist.mp4",
            "view_wrist.mp4",
        ]
        ep = json.loads((d / "episode.json").read_text())
        assert ep["extrinsics"]["grip_wrist"]["twin_camera"] == "grip_wrist_cam"
        assert ep["frames"] == ALL_FRAMES and ep["arm_bases"]["grip"]["has_rail"] is True
    finally:
        assert client.delete("/api/session").status_code == 204
    assert cell.stop_calls == 1 and rt.manager.session is None
    assert (sdir / "session.json").is_file() and (sdir / "rollouts" / "episodes" / saved).is_dir()
    assert _wait(lambda: not rt.hardware_monitor.paused)
    assert not rt.manager.hardware_session_active
    assert client.get("/api/datasets/online_dagger/hw1").json()["in_use"] is False


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
