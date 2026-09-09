"""``SessionManager`` GELLO path (16-gello §5, §9.2; phase-15): the launch 409 matrix in order
(both arms, leader not available, joint limits, the collision text against the kitchen twin
with a leader posture inside the fridge shell, ``keep_current`` only, the viewpoint node for
``viewpoint: external``) evaluated BEFORE any side effect; the launch motion on a hand-built
fake session (plan -> the loop's motion window -> ONE ARM AT A TIME -> engage on arrival;
a plan failure leaves the arms, the session RUNNING, GELLO out_of_sync and the reason on the
wire); and the hardware admission (D8) over ``HardwareFakeWorkcell`` with the twin's
graspable handles whitelisted against the gripper."""

from __future__ import annotations

import math
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from apollo_mavis_v2_core import ProfileStore, WorkcellConfig
from apollo_mavis_v2_core.protocol import SessionSpec
from apollo_mavis_v2_core.protocol.external import PolicySpecAnnounce, PolicySpecModel
from apollo_mavis_v2_core.testing import FakeArm, FakeWorkcell
from conftest import FakeMonitorFactory, FakeMonitorSample, make_runtime_config, run_ticks
from fakes import HardwareFakeWorkcell
from starlette.testclient import TestClient
from test_gello_loop import GRIP_Q0, VIEW_Q0, FakeReader, make_loop
from test_return_manager_units import FakeTwin

from apollo_mavis_v2_runtime.config import (
    GelloConfig,
    HardwareMonitorConfig,
    HardwareProbeConfig,
    MicrophoneConfig,
    TwinOverlayConfig,
)
from apollo_mavis_v2_runtime.devices.gello import FAKE_DEFAULT_Q
from apollo_mavis_v2_runtime.errors import SessionError
from apollo_mavis_v2_runtime.gello.loop import GelloLoop
from apollo_mavis_v2_runtime.gello.preview import GelloPreviewService
from apollo_mavis_v2_runtime.gello.viewpoint import view_action_names, view_state_names
from apollo_mavis_v2_runtime.runtime import Runtime
from apollo_mavis_v2_runtime.server.app import create_app
from apollo_mavis_v2_runtime.session.manager import SessionManager, _GelloSession
from apollo_mavis_v2_runtime.session.types import SessionState
from apollo_mavis_v2_runtime.streams.hub import VideoHub

pytestmark = pytest.mark.egl

PI = math.pi
KITCHEN = "mavis_v2_kitchen"
HOLD = [2.646, -1.598, 0.018, 1.637, 0.25, 2.007, 0.029]
# A leader posture whose right finger penetrates fridge_body by 76 mm at the keyframe rail
# (0.65) and touches nothing else, mic on or off (random search on the kitchen twin,
# 2026-09-09; joints 1/3/5/7 inside (-pi, pi) of the measured arm so the unwrap keeps k = 0).
FRIDGE_HIT = [3.145, 1.606, -1.426, 2.844, -0.467, 0.133, -1.952]
SIM_WORKCELL = {
    "kind": "sim",
    "sim_scene": KITCHEN,
    "arms": [
        {"id": "view", "base_in_world": {}, "gripper": "none"},
        {"id": "grip", "base_in_world": {}},
    ],
    "cameras": [],
}
HW_WORKCELL = {
    "kind": "hardware",
    "digital_twin_scene": "mavis_v2",
    "arms": [
        {"id": "grip", "ip": "192.168.1.201", "base_in_world": {}, "gripper": "xarm_g2"},
        {"id": "view", "ip": "192.168.2.219", "base_in_world": {}, "gripper": "none"},
    ],
    "cameras": [],
    "safety": {"enabled": True},
}
FRAMES = {"view": "arm_base:view", "grip": "arm_base:grip"}


def gello_spec(**over) -> SessionSpec:
    body = {
        "mode": "gello",
        "kind": "sim",
        "arms": ["view", "grip"],
        "frames": dict(FRAMES),
        "sim_scene": KITCHEN,
        "gello": {},
    }
    body.update(over)
    return SessionSpec.model_validate(body)


class StatusReader(FakeReader):
    """``FakeReader`` + the ``status`` / ``fresh_sample`` surface the manager and the
    preview service read; ``forced`` pins the device status."""

    def __init__(self) -> None:
        super().__init__()
        self.forced: str | None = None
        self.calibrated = True
        self.stale_s = 0.2

    def status(self, now=None):
        st = self.forced or ("connected" if self.sample is not None else "starting")
        detail = "" if st == "connected" else f"{st} detail"
        return SimpleNamespace(status=st, detail=detail, calibrated=self.calibrated)

    def fresh_sample(self, now=None):
        s = self.sample
        if s is None or not s.valid:
            return None
        now = time.monotonic() if now is None else now
        return s if now - s.rx_mono <= self.stale_s else None


def make_manager(tmp_path, *, reader=None, with_preview=True, scene=KITCHEN):
    cfg = make_runtime_config(tmp_path, scene).model_copy(
        update={
            "workcells": {
                "sim": WorkcellConfig.model_validate({**SIM_WORKCELL, "sim_scene": scene})
            },
            "gello": GelloConfig(backend="fake", calibration_path=tmp_path / "cal.json"),
        }
    )
    from apollo_mavis_v2_runtime.bus import RuntimeBus

    bus = RuntimeBus()
    manager = SessionManager(cfg, bus, VideoHub(bus), ProfileStore(tmp_path / "profiles"), "e")
    if reader is not None:
        manager.gello_reader = reader
        if with_preview:
            manager.gello_preview = GelloPreviewService(cfg, reader, render=False)
    return manager, cfg


def _announce(names, frame="arm_base:view", state_names=None):
    return PolicySpecAnnounce(
        policy_id="viewer",
        policy_version=1,
        node_version="t",
        rate_hz=15.0,
        spec=PolicySpecModel(
            action_space="delta_ee",
            action_frame=frame,
            action_names=list(names),
            state_names=list(state_names or ["view_joint1.pos"]),
        ),
    )


def _fake_dora(ann, attached=True):
    return SimpleNamespace(
        enabled=True,
        policy_hub=SimpleNamespace(spec=lambda now=None: ann),
        bridge=SimpleNamespace(attached=attached),
        publisher=None,
    )


# -- the 409 matrix (before any side effect) -----------------------------------------------------
def test_gello_409_matrix_in_order(tmp_path):
    reader = StatusReader()
    manager, cfg = make_manager(tmp_path, reader=reader)
    wc = cfg.workcells["sim"]
    now = time.monotonic()

    def refused(spec, needle: str, samples=None):
        with pytest.raises(SessionError) as e:
            manager._check_gello(spec, wc, samples)
        assert needle in str(e.value), (needle, str(e.value))
        assert manager.session is None

    # 1. both arms (the Perception Arm is the viewpoint)
    refused(gello_spec(arms=["grip"], frames={"grip": FRAMES["grip"]}), "needs both arms")
    # 1b. start_from keep_current only (core 422s a profile start; the manager 409s a spec
    #     that bypasses validation, belt and braces)
    bypass = SessionSpec.model_construct(**{**gello_spec().model_dump(), "start_from": "profile:x"})
    refused(bypass, "requires start_from 'keep_current'")
    # 2. the leader
    bare, _ = make_manager(tmp_path / "bare")  # no reader / preview wired
    with pytest.raises(SessionError) as e:
        bare._check_gello(gello_spec(), wc, None)
    assert "GELLO leader not available (no leader reader" in str(e.value)
    refused(gello_spec(), "GELLO leader not available (starting")  # no sample yet
    reader.set(FAKE_DEFAULT_Q, 1.0, t=now - 5.0)  # stale
    refused(gello_spec(), "GELLO leader not available (sample stale")
    reader.set(FAKE_DEFAULT_Q, 1.0, t=now, valid=False)
    refused(gello_spec(), "GELLO leader not available (no valid sample")
    reader.forced = "stale"
    reader.set(FAKE_DEFAULT_Q, 1.0, t=time.monotonic())
    refused(gello_spec(), "GELLO leader not available (stale: stale detail)")
    reader.forced = None
    reader.calibrated = False
    refused(gello_spec(), "GELLO leader not available (not calibrated")
    reader.calibrated = True
    # 3. joint limits (joint 2 beyond +120 deg less the 0.5 deg margin)
    over = list(FAKE_DEFAULT_Q)
    over[1] = 2.3
    reader.set(over, 1.0, t=time.monotonic())
    refused(
        gello_spec(),
        "GELLO posture outside the Manipulation Arm's joint limits (joint 2 = 2.30 rad, "
        "limit [-2.05, 2.09])",
    )
    # 4. the kitchen twin: the fridge shell (every pair, tightest first; mm; the retry hint)
    reader.set(FRIDGE_HIT, 1.0, t=time.monotonic())
    with pytest.raises(SessionError) as e:
        manager._check_gello(gello_spec(), wc, None)
    text = str(e.value)
    assert text.startswith("GELLO posture collides: fridge_body / grip_right_finger at -7")
    assert text.endswith(" mm - move GELLO and retry") and "grip_right_finger" in text
    # 5. viewpoint external: no bridge / stale spec / incompatible layout / compatible
    reader.set(FAKE_DEFAULT_Q, 1.0, t=time.monotonic())
    ext = gello_spec(gello={"viewpoint": "external"})
    refused(ext, "no external viewpoint node attached (dora bridge is not attached)")
    manager.dora = _fake_dora(None)
    refused(ext, "no external viewpoint node attached (no policy_spec heartbeat within 3 s)")
    manager.dora = _fake_dora(_announce(["grip_ee.dx"] + view_action_names(True)))
    refused(ext, "!= view layout")
    manager.dora = _fake_dora(_announce(view_action_names(True), frame="world"))
    refused(ext, "action_frame 'world' != the session's view frame 'arm_base:view'")
    manager.dora = _fake_dora(_announce(view_action_names(True), state_names=["grip_joint1.pos"]))
    refused(ext, "state_names outside the view state layout")
    manager.dora = _fake_dora(
        _announce(view_action_names(True), state_names=view_state_names(True))
    )
    ev = manager._check_gello(ext, wc, None)
    assert ev.status == "clear" and ev.ok
    assert ev.q_goal["grip"] == pytest.approx([*FAKE_DEFAULT_Q, 0.65])  # the keyframe rail
    assert ev.q_goal["view"] == pytest.approx([*HOLD, 0.0])
    # auto never needs a node at launch
    assert manager._check_gello(gello_spec(), wc, None).status == "clear"
    # the preview service answers the same verdicts (render off here)
    from apollo_mavis_v2_core.protocol import GelloPreviewRequest

    res = manager.gello_preview.preview(GelloPreviewRequest(kind="sim"))
    assert res.status == "clear" and res.ok and res.image_png_b64 is None
    assert res.camera == "cam_kitchen" and res.leader_q == pytest.approx(list(FAKE_DEFAULT_Q))
    reader.set(FRIDGE_HIT, 1.0, t=time.monotonic())
    res = manager.gello_preview.preview(GelloPreviewRequest(kind="sim"))
    assert res.status == "collision" and not res.ok
    assert (res.pairs[0].a, res.pairs[0].b) == ("fridge_body", "grip_right_finger")
    assert res.pairs[0].dist_m < -0.07 and res.detail.startswith("collides: fridge_body")
    with pytest.raises(Exception) as e:  # no hardware workcell configured: 409, not a status
        manager.gello_preview.preview(GelloPreviewRequest(kind="hardware"))
    assert "no hardware workcell is configured" in str(e.value)
    res = manager.gello_preview.preview(GelloPreviewRequest(kind="sim", scene="not_a_scene"))
    assert res.status == "scene_error" and "not_a_scene" in res.detail


def test_scene_less_gello_spec_defaults_to_gello_scene_id_for_check_and_bringup(tmp_path):
    """2026-09-09 review (16-gello §5.1 item 1): a gello spec without a scene used to fall
    back to the WORKCELL's bare-cell twin while the session-less preview judged the kitchen,
    so the same posture got two verdicts. Both kinds now default to ``gello.scene_id`` and
    the resolved id is written into the spec at ``create()``, so the validation, the launch
    check, the bring-up twin and the announce all see the same scene."""
    reader = StatusReader()
    manager, cfg = make_manager(tmp_path, reader=reader, scene="mavis_v2")  # the bare cell
    wc = cfg.workcells["sim"]
    assert wc.sim_scene == "mavis_v2" and cfg.gello.scene_id == KITCHEN
    bare = gello_spec(sim_scene=None)
    assert bare.sim_scene is None
    assert manager._gello_scene_id(bare, wc) == KITCHEN
    assert manager._gello_scene_id(gello_spec(), wc) == KITCHEN  # explicit wins
    assert manager._gello_scene_id(gello_spec(sim_scene="mavis_v2"), wc) == "mavis_v2"
    hw = SessionSpec.model_validate(
        {
            "mode": "gello",
            "kind": "hardware",
            "arms": ["grip", "view"],
            "frames": dict(FRAMES),
            "gello": {"viewpoint": "hold"},
        }
    )
    assert hw.digital_twin_scene is None and manager._gello_scene_id(hw, wc) == KITCHEN
    assert manager._gello_spec_with_scene(hw).digital_twin_scene == KITCHEN
    assert manager._gello_spec_with_scene(bare).sim_scene == KITCHEN
    assert manager._gello_spec_with_scene(gello_spec()) is not None
    # create(): the check runs on the kitchen and the bring-up receives the resolved spec
    reader.set(FAKE_DEFAULT_Q, 1.0, t=time.monotonic())
    seen: list = []

    def bringup(spec, wc, gello_check=None):
        seen.append((spec, gello_check))
        raise SessionError("stop here (test seam)")

    manager._bringup_sim = bringup
    with pytest.raises(SessionError, match="stop here"):
        manager.create(bare)
    assert manager.session is None and not manager.session_active
    assert len(seen) == 1
    spec_seen, ev = seen[0]
    assert spec_seen.sim_scene == KITCHEN and ev.status == "clear"
    assert set(manager.gello_preview._twins) == {("sim", KITCHEN)}  # never the bare cell
    # the fridge posture is refused on the kitchen twin although the workcell's own scene
    # has no fridge - check and preview agree
    reader.set(FRIDGE_HIT, 1.0, t=time.monotonic())
    with pytest.raises(SessionError, match="GELLO posture collides: fridge_body"):
        manager.create(bare)
    assert len(seen) == 1  # refused before the bring-up
    from apollo_mavis_v2_core.protocol import GelloPreviewRequest

    assert manager.gello_preview.preview(GelloPreviewRequest(kind="sim")).status == "collision"


def test_create_refuses_before_any_side_effect(tmp_path):
    reader = StatusReader()
    manager, cfg = make_manager(tmp_path, reader=reader)
    reader.set(FRIDGE_HIT, 1.0, t=time.monotonic())
    with pytest.raises(SessionError) as e:
        manager.create(gello_spec())
    assert "GELLO posture collides" in str(e.value)
    assert manager.session is None and not manager.session_active
    assert manager._preview_service is None  # the sim previews were never stopped / started


# -- the launch motion on a hand-built fake session ------------------------------------------------
def _fake_session(manager, loop, cell, twin, q_goal, state=SessionState.BRINGUP):
    spec = gello_spec()
    session = SimpleNamespace(
        spec=spec,
        state=state,
        recorder_thread=None,
        loop=loop,
        supervisor=loop.supervisor,
        workcell=cell,
        twin=twin,
        session_id="s",
        fault_detail="",
        motion_detail="",
        start_from_progress=None,
        planned_start=None,
        gello=_GelloSession(scene_id=KITCHEN, q_goal=q_goal),
        online_dagger=None,
        policy_session=None,
        streams=[],
        sources=[],
        render_service=None,
        adopted_streams=[],
        frozen_arms=[],
        inner_workcell=None,
    )
    session.notice = lambda arms_carry_faults=False: session.motion_detail or session.fault_detail
    manager.session = session
    manager.attach_fault_state(session)
    return session


def test_launch_motion_runs_one_arm_at_a_time_inside_the_motion_window_then_engages(tmp_path):
    reader = StatusReader()
    manager, cfg = make_manager(tmp_path, reader=reader)
    # the Perception Arm starts 0.2 rad off its hold posture too (else the planner parks it)
    view_start = VIEW_Q0.copy()
    view_start[1] += 0.2
    cell = FakeWorkcell(
        {
            "grip": FakeArm("grip", has_rail=True, q0=GRIP_Q0.copy(), max_joint_speed_rad_s=5.0),
            "view": FakeArm("view", has_rail=True, q0=view_start, max_joint_speed_rad_s=5.0),
        }
    )
    cell, bus, loop, _ = make_loop(tmp_path, cell=cell, launch_pending=True)
    manager.bus = bus
    loop.reader = reader  # the loop's leader = the manager's reader
    # the leader stands 0.2 rad off the arm on joint 2 (out of tolerance): the launch
    # motion walks the arm THERE, then the engage rule passes
    leader = GRIP_Q0[:7].copy()
    leader[1] += 0.2
    reader.set(leader, 1.0, t=0.0)
    q_goal = {"grip": [*leader, 0.65], "view": [*HOLD, 0.0]}
    twin = FakeTwin()
    session = _fake_session(manager, loop, cell, twin, q_goal)
    submitted: list[list[str]] = []
    orig = manager._submit_execute_plan

    def spy(wps, grip, *, interruptible):
        submitted.append(sorted(wps))
        return orig(wps, grip, interruptible=interruptible)

    manager._submit_execute_plan = spy
    states_seen: set[str] = set()
    t = [0.0]

    def tick():
        reader.set(leader, 1.0, t=t[0])  # keep the sample fresh in the loop's clock
        t[0] = run_ticks(loop, cell, 1, t[0])
        states_seen.add(loop.engage.state)

    worker = threading.Thread(target=manager._start_from_worker, args=(session,), daemon=True)
    worker.start()
    deadline = time.monotonic() + 20.0
    while worker.is_alive() and time.monotonic() < deadline:
        tick()
        time.sleep(0.002)
    assert not worker.is_alive(), "start_from worker did not finish"
    for _ in range(5):
        tick()
    assert session.state is SessionState.RUNNING and session.motion_detail == ""
    assert twin.plans and twin.plans[0].q_goal["grip"] == pytest.approx(q_goal["grip"])
    assert twin.plans[0].q_goal["view"] == pytest.approx(q_goal["view"])
    # one arm per execute_plan, in the planner's order (FakeTwin: dict order = spec order)
    assert submitted == [["view"], ["grip"]]
    assert "motion" in states_seen  # the window kept GELLO in `motion` throughout
    assert not loop._motion_hold
    assert loop.engage.state == "tracking"  # engaged on arrival: the leader is within tolerance
    assert np.allclose(cell.arms["grip"].get_state().q[:7], leader, atol=2e-3)
    assert np.allclose(cell.arms["view"].get_state().q[:7], HOLD, atol=2e-3)
    assert loop._arm_source["grip"].value == "gello"


def test_launch_plan_failure_leaves_the_arms_running_out_of_sync_with_the_reason(tmp_path):
    reader = StatusReader()
    manager, cfg = make_manager(tmp_path, reader=reader)
    cell, bus, loop, _ = make_loop(tmp_path, launch_pending=True)
    manager.bus = bus
    loop.reader = reader
    leader = GRIP_Q0[:7].copy()
    leader[1] += 0.4
    reader.set(leader, 1.0, t=0.0)
    session = _fake_session(
        manager, loop, cell, FakeTwin(ok=False), {"grip": [*leader, 0.65], "view": [*HOLD, 0.0]}
    )
    worker = threading.Thread(target=manager._start_from_worker, args=(session,), daemon=True)
    worker.start()
    t = 0.0
    deadline = time.monotonic() + 10.0
    while worker.is_alive() and time.monotonic() < deadline:
        reader.set(leader, 1.0, t=t)
        t = run_ticks(loop, cell, 1, t)
        time.sleep(0.002)
    assert not worker.is_alive()
    reader.set(leader, 1.0, t=t)
    t = run_ticks(loop, cell, 2, t)
    assert session.state is SessionState.RUNNING
    detail = session.motion_detail
    assert detail.startswith("GELLO launch motion not planned: goal_in_collision")
    assert "arm0_link6 / table" in detail and "have not moved" in detail
    assert np.allclose(cell.arms["grip"].get_state().q, GRIP_Q0)  # nothing moved
    assert np.allclose(cell.arms["view"].get_state().q, VIEW_Q0)
    assert not loop._motion_hold and loop.engage.state == "out_of_sync"
    assert "joint 2" in loop.engage.detail


def test_return_to_initial_forces_paused_and_keeps_it_after_the_motion(tmp_path):
    from apollo_mavis_v2_core import ArmPosture, StateProfile

    reader = StatusReader()
    manager, cfg = make_manager(tmp_path, reader=reader)
    cell, bus, loop, _ = make_loop(tmp_path)
    manager.bus = bus
    loop.reader = reader
    reader.set(GRIP_Q0[:7], 1.0, t=0.0)
    _fake_session(
        manager,
        loop,
        cell,
        FakeTwin(),
        {"grip": list(GRIP_Q0), "view": list(VIEW_Q0)},
        state=SessionState.RUNNING,
    )
    t = run_ticks(loop, cell, 3, 0.0)
    assert loop.engage.state == "tracking"
    prof = manager.profile_store.save(
        StateProfile(
            name="initial",
            workcell_kind="sim",
            is_initial_condition=True,
            arms={
                "grip": ArmPosture(q=[x + 0.1 for x in GRIP_Q0[:7]], rail_pos_m=None),
                "view": ArmPosture(q=[x + 0.1 for x in VIEW_Q0[:7]], rail_pos_m=None),
            },
        )
    )
    manager.profile_store.set_initial(prof.profile_id)
    result_box: list = []
    worker = threading.Thread(
        target=lambda: result_box.append(manager.return_to_initial()), daemon=True
    )
    worker.start()
    deadline = time.monotonic() + 20.0
    seen: set[str] = set()
    while worker.is_alive() and time.monotonic() < deadline:
        reader.set(GRIP_Q0[:7], 1.0, t=t)
        t = run_ticks(loop, cell, 1, t)
        seen.add(loop.engage.state)
        time.sleep(0.002)
    assert not worker.is_alive() and result_box and result_box[0].status == "done", result_box
    reader.set(GRIP_Q0[:7], 1.0, t=t)
    t = run_ticks(loop, cell, 3, t)
    assert "motion" in seen
    assert loop.engage.state == "paused"
    assert "Return to the initial condition" in loop.engage.detail
    assert np.allclose(cell.arms["grip"].get_state().q[:7], GRIP_Q0[:7] + 0.1, atol=2e-3)
    # the leader is back at the old posture: paused means the follower stays at the profile
    assert np.allclose(loop._last_cmd["grip"][:7], GRIP_Q0[:7] + 0.1, atol=1e-6)


# -- hardware admission (16-gello D8) over HardwareFakeWorkcell ------------------------------------
KEYFRAME_Q = (PI, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _hw_samples():
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
        ),
    }


class Cells:
    def __init__(self) -> None:
        self.built: list[HardwareFakeWorkcell] = []

    def __call__(self, session_cfg, driver_factory):
        arms = {
            a.id: FakeArm(
                a.id,
                has_rail=True,
                q0=np.array([*KEYFRAME_Q, 0.65 if a.id == "grip" else 0.0]),
                max_joint_speed_rad_s=3.0,
            )
            for a in session_cfg.arms
        }
        cell = HardwareFakeWorkcell(arms, kind="hardware")
        self.built.append(cell)
        return cell


def _wait(pred, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


def _telemetry(client, pred, frames: int = 150) -> dict:
    with client.websocket_connect("/ws/telemetry") as ws:
        for _ in range(frames):
            msg = ws.receive_json()
            if pred(msg):
                return msg
    raise AssertionError(f"telemetry never satisfied the predicate; last frame {msg['gello']}")


def test_hardware_admits_gello_and_the_twin_whitelists_the_handles(tmp_path):
    cfg = make_runtime_config(tmp_path, "mavis_v2").model_copy(
        update={
            "workcells": {
                "sim": WorkcellConfig.model_validate({**SIM_WORKCELL, "sim_scene": "mavis_v2"}),
                "hardware": WorkcellConfig.model_validate(HW_WORKCELL),
            },
            "gello": GelloConfig(backend="fake", calibration_path=tmp_path / "cal.json"),
            "microphone": MicrophoneConfig(enabled=False),
            "hardware_probe": HardwareProbeConfig(enabled=False),
            "twin_overlay": TwinOverlayConfig(enabled=False),
            "hardware_monitor": HardwareMonitorConfig(),
        }
    )
    factory = FakeMonitorFactory(_hw_samples())
    rt = Runtime(cfg, monitor_factory=factory)
    cells = Cells()
    rt.manager.workcell_factory = cells
    with TestClient(create_app(rt)) as client:
        assert _wait(lambda: rt.hardware_monitor.status_of("grip")[0] == "running")
        assert _wait(lambda: rt.gello.status().status == "connected")
        body = {
            "mode": "gello",
            "kind": "hardware",
            "arms": ["grip", "view"],
            "frames": dict(FRAMES),
            "digital_twin_scene": KITCHEN,
            "speed_scale": 0.5,
            "gello": {"viewpoint": "hold"},
        }
        # D7 stays: Online DAgger / inference are refused with the new text
        r = client.post("/api/session", json={**body, "mode": "inference", "gello": None})
        assert r.status_code == 409 and (
            "hardware sessions support teleop, data collection and GELLO Manipulation only "
            "(inference on hardware: not yet)" == r.json()["detail"]
        )
        # the leader inside the fridge shell: refused BEFORE the monitor hand-over
        n_calls = len(factory.monitors["grip"].calls)
        rt.gello.fake_set(FRIDGE_HIT, 1.0)
        assert _wait(lambda: np.allclose(rt.gello.latest().q, FRIDGE_HIT))
        r = client.post("/api/session", json=body)
        assert r.status_code == 409 and r.json()["detail"].startswith(
            "GELLO posture collides: fridge_body / grip_right_finger at -7"
        ), r.text
        assert len(factory.monitors["grip"].calls) == n_calls  # never paused
        assert not cells.built
        # synced leader (the fake's default = the keyframe = the monitor sample): admitted
        rt.gello.fake_set(FAKE_DEFAULT_Q, 1.0)
        assert _wait(lambda: np.allclose(rt.gello.latest().q, FAKE_DEFAULT_Q))
        r = client.post("/api/session", json=body)
        assert r.status_code == 200, r.text
        info = r.json()
        assert info["mode"] == "gello" and info["kind"] == "hardware"
        assert info["gello"] == {"viewpoint": "hold"}
        try:
            assert _wait(lambda: client.get("/api/session").json()["state"] == "running", 20.0)
            session = rt.manager.session
            assert isinstance(session.loop, GelloLoop) and session.loop.tracker is None
            assert session.loop.active_arm == "grip" and session.loop.workcell_kind == "hardware"
            assert session.gello is not None and session.gello.plan_failure is None
            # D7: the handles are whitelisted against the gripper, the links stay gated
            allowed = session.twin.allowed
            assert allowed.allows_labels("grip_left_finger", "fridge_door_handle")
            assert allowed.allows_labels("grip_right_finger", "range_handle")
            assert not allowed.allows_labels("grip_link6", "fridge_door_handle")
            assert not allowed.allows_labels("grip_left_finger", "fridge_body")
            msg = _telemetry(client, lambda m: m["gello"]["state"] == "tracking")
            g = msg["gello"]
            assert g["engaged_arm"] == "grip" and g["viewpoint"]["mode"] == "hold"
            assert g["viewpoint"]["attached"] is False
            assert msg["active_arm"] == "grip"
            assert client.get("/api/session").json()["fault_detail"] == ""
            # the hardware loop keeps the streamer's caps (no sim lowering)
            assert session.loop.cfg.dq_max_rad > 0.0
        finally:
            client.delete("/api/session")
        assert _wait(lambda: rt.manager.session is None)
        assert cells.built and cells.built[-1].stop_calls >= 1  # D6 hand-back
