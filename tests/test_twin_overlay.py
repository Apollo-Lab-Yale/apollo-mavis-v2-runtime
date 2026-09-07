"""Twin alignment overlays (phase-09a): ``<camera_id>_align`` streams over a
module-scoped ``Runtime`` with fake wrist cameras (black 640x480 frames) and a
fake read-only monitor. Needs MUJOCO_GL=egl (conftest sets it)."""

from __future__ import annotations

import math
import os
import time

import cv2
import numpy as np
import pytest
from apollo_mavis_v2_core import CameraFrame, WorkcellConfig
from apollo_mavis_v2_core.protocol import HEADER_SIZE, unpack_header
from apollo_mavis_v2_core.schemas import CameraIntrinsics, PoseModel
from conftest import FakeMonitorFactory, FakeMonitorSample, make_runtime_config
from starlette.testclient import TestClient

from apollo_mavis_v2_runtime.config import (
    HardwareMonitorConfig,
    HardwareProbeConfig,
    MicrophoneConfig,
    TwinOverlayConfig,
)
from apollo_mavis_v2_runtime.runtime import Runtime
from apollo_mavis_v2_runtime.server.app import create_app
from apollo_mavis_v2_runtime.streams.twin_overlay import (
    DEFAULT_COLOUR_FOVY_DEG,
    TwinOverlayRenderer,
    apply_intrinsics,
    arm_for_camera,
    base_pose_overrides,
    composite,
    default_fovy_deg,
    overlay_label,
    principal_pixel,
)

pytestmark = pytest.mark.egl

SCENE = "mavis_v2"
W, H = 640, 480
PI = math.pi
GRIP_INTR = {"fx": 608.19, "fy": 608.23, "cx": 327.39, "cy": 247.90}
VIEW_INTR = {"fx": 606.36, "fy": 606.38, "cx": 311.90, "cy": 249.45}
YELLOW = (255, 235, 140)
GREY = (170, 170, 170)
ENV_BLUE = (90, 200, 250)

HW_WORKCELL = {
    "kind": "hardware",
    "digital_twin_scene": SCENE,
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
    "cameras": [
        {
            "id": "grip_wrist",
            "kind": "v4l2",
            "serial": "349643062582",
            "fourcc": "YUYV",
            "resolution": [W, H],
            "fps": 30,
            "intrinsics": GRIP_INTR,
        },
        {
            "id": "view_wrist",
            "kind": "v4l2",
            "serial": "322143060792",
            "fourcc": "YUYV",
            "resolution": [W, H],
            "fps": 30,
            "intrinsics": VIEW_INTR,
        },
    ],
    "safety": {"enabled": True},
}

# Fake monitor samples: the Perception Arm (view) at the mavis_v2 keyframe (its mic
# body fills the bottom of its own camera image), the Manipulation Arm (grip) with
# joint 6 bent so its wrist camera sees its own gripper / forearm (at the folded
# keyframe the grip camera sees only floor with the D435i intrinsics). Both tracks
# unhomed like the lab (rail_pos_m None, raw 0) -> rail fallback in use.
SAMPLES = {
    "grip": FakeMonitorSample(
        "grip",
        seq=3,
        t_mono=time.monotonic(),
        q=(PI, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0),
        gripper_open_frac=1.0,
        gripper_raw=84.0,
    ),
    "view": FakeMonitorSample("view", seq=5, t_mono=time.monotonic(), error_code=19),
}


class StaticCamera:
    """Hardware camera stand-in: a black 640x480 frame per ``latest()`` with an
    incrementing seq; ``blackout`` makes it return ``None`` (no frame)."""

    def __init__(self, camera_id: str, resolution=(W, H), fps: float = 30.0) -> None:
        self._camera_id = camera_id
        self._resolution = tuple(resolution)
        self._fps = float(fps)
        self._seq = 0
        self.blackout = False
        self.started = False
        self.failed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def latest(self) -> CameraFrame | None:
        if not self.started or self.blackout:
            return None
        self._seq += 1
        w, h = self._resolution
        return CameraFrame(
            camera_id=self._camera_id,
            rgb=np.zeros((h, w, 3), dtype=np.uint8),
            t_mono=time.monotonic(),
            wallclock_ns=time.time_ns(),
            seq=self._seq,
        )

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution

    @property
    def fps(self) -> float:
        return self._fps


def _wait(pred, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


def _config(tmp_path, **overlay_kw):
    cfg = make_runtime_config(tmp_path, SCENE)
    return cfg.model_copy(
        update={
            "workcells": {**cfg.workcells, "hardware": WorkcellConfig.model_validate(HW_WORKCELL)},
            "microphone": MicrophoneConfig(enabled=False),
            "hardware_probe": HardwareProbeConfig(enabled=False),
            "hardware_monitor": HardwareMonitorConfig(),
            "twin_overlay": TwinOverlayConfig(fps=12.0, **overlay_kw),
        }
    )


@pytest.fixture(scope="module")
def cams():
    return {cid: StaticCamera(cid) for cid in ("grip_wrist", "view_wrist")}


@pytest.fixture(scope="module")
def factory():
    return FakeMonitorFactory(dict(SAMPLES))


@pytest.fixture(scope="module")
def rt(tmp_path_factory, cams, factory):
    runtime = Runtime(_config(tmp_path_factory.mktemp("rt")), monitor_factory=factory)
    runtime.manager.camera_factory = lambda cam_cfg: cams[cam_cfg.id]
    return runtime


@pytest.fixture(scope="module")
def client(rt):
    with TestClient(create_app(rt)) as c:
        assert rt.twin_overlay is not None and rt.twin_overlay.wait_ready(60.0)
        assert _wait(lambda: rt.twin_overlay.status_of("grip_wrist_align") == "live"), (
            rt.twin_overlay.telemetry()
        )
        assert _wait(lambda: rt.twin_overlay.status_of("view_wrist_align") == "live")
        yield c


def _robot_pixels(frame: CameraFrame) -> np.ndarray:
    """Non-black pixels that are not the environment outline (real frame is black)."""
    rgb = np.asarray(frame.rgb)
    nonblack = rgb.any(axis=2)
    env = np.all(rgb == np.asarray(ENV_BLUE, dtype=np.uint8), axis=2)
    return rgb[nonblack & ~env]


def _next_frame(src, after_seq: int) -> CameraFrame:
    assert _wait(lambda: src.seq > after_seq)
    frame = src.latest()
    assert frame is not None
    return frame


# -- pure helpers ---------------------------------------------------------------------------------
def test_principal_point_sign_and_intrinsics_on_a_real_mjs_camera():
    import mujoco

    intr = CameraIntrinsics(**GRIP_INTR)
    assert principal_pixel(intr, W, H) == pytest.approx((320 - 327.39, 240 - 247.90))
    spec = mujoco.MjSpec()
    cam = spec.worldbody.add_camera(name="c")
    apply_intrinsics(cam, intr, W, H)
    assert list(cam.resolution) == [W, H]
    assert list(cam.focal_pixel) == pytest.approx([608.19, 608.23])
    assert list(cam.principal_pixel) == pytest.approx([-7.39, -7.90])  # MuJoCo = -OpenCV
    assert list(cam.sensor_size) == pytest.approx([W * 1e-5, H * 1e-5])
    plain = spec.worldbody.add_camera(name="p")
    apply_intrinsics(plain, None, W, H)
    assert plain.fovy == pytest.approx(DEFAULT_COLOUR_FOVY_DEG)
    assert default_fovy_deg(608.23, H) == pytest.approx(43.1, abs=0.15)  # ~43.2, not the MJCF 57


def test_principal_offset_shifts_only_the_rendered_principal_point():
    # A per-camera overlay-only nudge (cx += du, cy += dv) leaves the true intrinsics
    # untouched but moves the rendered principal point by exactly (-du, -dv) in MuJoCo's
    # opposite-sign convention. Aligns a wrist camera mounted off the shared wrist_cam pose.
    import mujoco

    intr = CameraIntrinsics(**GRIP_INTR)
    base = principal_pixel(intr, W, H)
    shifted = principal_pixel(intr, W, H, (21.0, 13.0))
    assert shifted == pytest.approx((base[0] - 21.0, base[1] - 13.0))
    assert intr.cx == pytest.approx(327.39) and intr.cy == pytest.approx(247.90)  # unchanged
    spec = mujoco.MjSpec()
    cam = spec.worldbody.add_camera(name="c")
    apply_intrinsics(cam, intr, W, H, (21.0, 13.0))
    expected = [320 - (327.39 + 21.0), 240 - (247.90 + 13.0)]
    assert list(cam.principal_pixel) == pytest.approx(expected)
    assert list(cam.focal_pixel) == pytest.approx([608.19, 608.23])  # focal untouched


def test_config_principal_offset_reaches_the_view_stream(rt):
    # twin_overlay.principal_offset_px keyed by camera id is carried onto the matching
    # TwinOverlaySource so the view overlay renders with the corrected principal point;
    # a camera with no entry keeps the (0, 0) default.
    wc = WorkcellConfig.model_validate(HW_WORKCELL)
    renderer = TwinOverlayRenderer(
        TwinOverlayConfig(principal_offset_px={"view_wrist": (21.0, 13.0)}),
        wc,
        SCENE,
        rt.hardware_monitor,
        lambda cid: None,
        rt.hub,
    )
    view = next(s for s in renderer.streams.values() if s.camera_id == "view_wrist")
    grip = next(s for s in renderer.streams.values() if s.camera_id == "grip_wrist")
    assert view.principal_offset == pytest.approx((21.0, 13.0))
    assert grip.principal_offset == pytest.approx((0.0, 0.0))  # default: no nudge


def test_camera_to_arm_mapping_labels_and_base_pose_overrides():
    wc = WorkcellConfig.model_validate(HW_WORKCELL)
    ids = [a.id for a in wc.arms]
    assert [arm_for_camera(c, ids) for c in wc.cameras] == ["grip", "view"]
    other = wc.cameras[0].model_copy(update={"id": "camera1"})
    assert arm_for_camera(other, ids) is None
    ee = wc.cameras[0].model_copy(update={"id": "cam9", "extrinsics_frame": "ee:view"})
    assert arm_for_camera(ee, ids) == "view"
    assert overlay_label("grip") == "Manipulation · twin overlay"
    assert overlay_label("view") == "Perception · twin overlay"
    assert overlay_label("arm0") == "arm0 · twin overlay"
    assert base_pose_overrides(wc) == {}  # identity base_in_world -> scene places the arms
    placed = wc.model_copy(
        update={
            "arms": [
                wc.arms[0].model_copy(
                    update={"base_in_world": PoseModel(position=(0.1, 0.2, 0.3))}
                ),
                wc.arms[1],
            ]
        }
    )
    over = base_pose_overrides(placed)
    assert set(over) == {"grip"} and list(over["grip"].position) == [0.1, 0.2, 0.3]


def test_composite_tint_outline_env_edges_and_stale_grey():
    cfg = TwinOverlayConfig(alpha=0.5)
    real = np.zeros((40, 60, 3), dtype=np.uint8)
    twin = np.full((40, 60, 3), 200, dtype=np.uint8)
    mask = np.zeros((40, 60), dtype=bool)
    mask[10:30, 20:40] = True
    env_ids = np.full((40, 60), -1, dtype=np.int32)
    env_ids[35:, :] = 1  # "table" across the bottom
    out = composite(real, twin, mask, cfg, env_ids=env_ids)
    assert out.shape == real.shape and out.dtype == np.uint8
    inner = out[15, 30]
    shade = (200 * 0.299 + 200 * 0.587 + 200 * 0.114) / 255.0
    expect = 0.5 * np.asarray(YELLOW) * (0.55 + 0.45 * shade)
    assert inner == pytest.approx(expect, abs=1.0)  # alpha * shaded tint over black
    assert tuple(out[10, 30]) == (255, 220, 60)  # 1 px outline on the mask boundary
    assert not out[5, 5].any()  # untouched real pixel
    assert (out[34:36, 5] == np.asarray(ENV_BLUE)).all(axis=1).any()  # table edge drawn
    stale = composite(real, twin, mask, cfg, stale=True, env_ids=None)
    px = stale[mask]
    assert px.any() and np.all(px[:, 0] == px[:, 1]) and np.all(px[:, 1] == px[:, 2])  # grey
    assert tuple(stale[10, 30]) == GREY
    untouched = composite(real, twin, np.zeros_like(mask), cfg)
    assert not untouched.any()


def test_renderer_inert_when_disabled_or_without_wrist_cameras(rt):
    wc = WorkcellConfig.model_validate(HW_WORKCELL)
    off = TwinOverlayRenderer(
        TwinOverlayConfig(enabled=False),
        wc,
        SCENE,
        rt.hardware_monitor,
        lambda cid: None,
        rt.hub,
    )
    assert not off.enabled and "enabled: false" in off.detail and off.streams == {}
    off.start()
    assert off._thread is None and off.telemetry() == []
    plain = WorkcellConfig.model_validate(
        {
            **HW_WORKCELL,
            "cameras": [{"id": "camera1", "kind": "v4l2", "device_path": "/dev/video9"}],
        }
    )
    none = TwinOverlayRenderer(
        TwinOverlayConfig(),
        plain,
        SCENE,
        rt.hardware_monitor,
        lambda cid: None,
        rt.hub,
    )
    assert not none.enabled and "no hardware wrist camera" in none.detail
    no_twin = TwinOverlayRenderer(
        TwinOverlayConfig(),
        wc,
        None,
        rt.hardware_monitor,
        lambda cid: None,
        rt.hub,
    )
    assert not no_twin.enabled and "digital_twin_scene" in no_twin.detail
    # A wrist camera without intrinsics still gets a stream (fovy fallback + warning).
    bare = WorkcellConfig.model_validate(
        {
            **HW_WORKCELL,
            "cameras": [{"id": "grip_wrist", "kind": "v4l2", "serial": "1", "fourcc": "YUYV"}],
        }
    )
    lone = TwinOverlayRenderer(
        TwinOverlayConfig(),
        bare,
        SCENE,
        rt.hardware_monitor,
        lambda cid: None,
        rt.hub,
    )
    assert lone.enabled and list(lone.streams) == ["grip_wrist_align"]
    src = lone.streams["grip_wrist_align"]
    assert (src.camera_id, src.arm_id, src.twin_camera) == ("grip_wrist", "grip", "grip_wrist_cam")
    assert src.intrinsics is None and src.status == "off"


# -- live runtime -------------------------------------------------------------------------------
def test_streams_on_hub_and_cameras_rows_kind_twin(client, rt):
    ov = rt.twin_overlay
    assert set(ov.streams) == {"grip_wrist_align", "view_wrist_align"}
    assert rt.hub.has("grip_wrist_align") and rt.hub.has("view_wrist_align")
    for sid in ov.streams:
        assert not sid.endswith("_wrist_cam") and sid not in ("sim", "twin")  # reserved rules
    cams = {c["camera_id"]: c for c in client.get("/api/cameras").json()}
    assert cams["grip_wrist"]["live"] is True and cams["grip_wrist"]["kind"] == "v4l2"
    for sid, label in (
        ("grip_wrist_align", "Manipulation · twin overlay"),
        ("view_wrist_align", "Perception · twin overlay"),
    ):
        row = cams[sid]
        assert row["kind"] == "twin" and row["live"] is True and row["label"] == label
        assert row["resolution"] == [W, H] and row["fps"] == 12
    hw = client.get("/api/workcell", params={"kind": "hardware"}).json()
    hw_cams = {c["camera_id"]: c for c in hw["cameras"]}
    assert {"grip_wrist", "view_wrist", "grip_wrist_align", "view_wrist_align"} == set(hw_cams)
    by_id = {a["arm_id"]: a for a in hw["arms"]}
    assert by_id["view"]["error_code"] == 19 and by_id["grip"]["error_code"] == 0  # monitor
    # the sim rows / previews are untouched
    assert cams["cam_front"]["kind"] == "sim" and cams["cam_front"]["live"] is True


def test_ws_video_align_stream_serves_jpeg_frames_at_640x480(client, rt):
    with client.websocket_connect("/ws/video/grip_wrist_align") as ws:
        buf = ws.receive_bytes()
    ts, n = unpack_header(buf)
    assert n == len(buf) - HEADER_SIZE and buf[HEADER_SIZE : HEADER_SIZE + 2] == b"\xff\xd8"
    img = cv2.imdecode(np.frombuffer(buf[HEADER_SIZE:], dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (H, W, 3)
    assert img.any()  # not a black frame: the twin is drawn on it


def test_mask_fraction_rail_fallback_and_telemetry_block(client, rt):
    ov = rt.twin_overlay
    assert _wait(lambda: all(t.fps > 0 for t in ov.telemetry()))
    time.sleep(1.2)  # let the 1 s fps window fill (the first live frame was just published)
    rows = {t.stream_id: t for t in ov.telemetry()}
    grip, view = rows["grip_wrist_align"], rows["view_wrist_align"]
    assert (grip.camera_id, grip.arm_id) == ("grip_wrist", "grip")
    assert (view.camera_id, view.arm_id) == ("view_wrist", "view")
    assert grip.status == "live" and view.status == "live"
    assert grip.mask_fraction > 0.05  # own gripper / forearm in view (joint 6 bent)
    assert view.mask_fraction > 0.05  # the microphone body at the keyframe
    assert grip.rail_fallback_m == 0.65 and view.rail_fallback_m == 0.0  # tracks unhomed
    assert grip.detail == "rail not homed - twin assumes 0.65 m"
    assert view.detail == "rail not homed - twin assumes 0.00 m"
    assert grip.joint1_offset_rad == 0.0
    assert 6.0 <= grip.fps <= 20.0, grip.fps  # ~12 fps (loose band: shared CI GPU)
    with client.websocket_connect("/ws/telemetry") as ws:
        msg = ws.receive_json()
    hm = msg["hardware_monitor"]
    assert hm["enabled"] is True and hm["paused"] is False
    arms = {a["arm_id"]: a for a in hm["arms"]}
    assert arms["grip"]["status"] == "running" and arms["grip"]["q"][5] == 1.5
    assert arms["grip"]["rail_pos_m"] is None and arms["grip"]["rail_raw_mm"] == 0.0
    assert arms["grip"]["gripper_open_frac"] == 1.0 and arms["view"]["gripper_open_frac"] is None
    assert arms["view"]["error_code"] == 19 and arms["view"]["state"] == 4
    ovs = {o["stream_id"]: o for o in hm["overlays"]}
    assert set(ovs) == {"grip_wrist_align", "view_wrist_align"}
    assert ovs["grip_wrist_align"]["rail_fallback_m"] == 0.65
    assert ovs["grip_wrist_align"]["mask_fraction"] > 0.05
    assert ovs["view_wrist_align"]["detail"] == "rail not homed - twin assumes 0.00 m"


def test_frames_are_yellow_when_live_and_grey_when_monitor_stale(client, rt, factory):
    src = rt.twin_overlay.streams["grip_wrist_align"]
    frame = _next_frame(src, src.seq)
    assert frame.camera_id == "grip_wrist_align" and frame.rgb.shape == (H, W, 3)
    px = _robot_pixels(frame)
    assert len(px) > 1000
    assert np.all(px[:, 0].astype(int) > px[:, 2].astype(int))  # yellow: R > B everywhere
    # Monitor goes stale for the Manipulation Arm -> grey twin + status/detail.
    factory.monitors["grip"].forced_status = "stale"
    factory.monitors["grip"].detail_text = "no fresh sample for 1.2 s"
    try:
        assert _wait(lambda: rt.twin_overlay.status_of("grip_wrist_align") == "stale")
        frame = _next_frame(src, src.seq)
        px = _robot_pixels(frame)
        assert len(px) > 1000
        assert np.all(px[:, 0] == px[:, 1]) and np.all(px[:, 1] == px[:, 2])  # grey
        tele = {t.stream_id: t for t in rt.twin_overlay.telemetry()}["grip_wrist_align"]
        assert tele.status == "stale"
        assert tele.detail == (
            "monitor stale: no fresh sample for 1.2 s; rail not homed - twin assumes 0.65 m"
        )
        cams = {c["camera_id"]: c for c in client.get("/api/cameras").json()}
        assert cams["grip_wrist_align"]["live"] is True  # stale still composites
        assert rt.twin_overlay.status_of("view_wrist_align") == "live"  # per-arm
    finally:
        factory.monitors["grip"].forced_status = None
        factory.monitors["grip"].detail_text = ""
    assert _wait(lambda: rt.twin_overlay.status_of("grip_wrist_align") == "live")


def test_arm_never_sampled_means_waiting_not_a_keyframe_twin(client, rt, factory):
    """A box that is off (monitor error / connecting, no sample EVER for that arm)
    must not be drawn: the twin at the keyframe posture was never measured. The
    stream waits (nothing published, telemetry fps 0, `/api/cameras` live false)
    and the other arm's overlay is unaffected. `stale` is reserved for an arm whose
    last real sample aged out."""
    view = factory.monitors["view"]
    src = rt.twin_overlay.streams["view_wrist_align"]
    view.sample = None
    view.forced_status = "error"
    view.detail_text = "connect failed: [Errno 113] No route to host"
    try:
        assert _wait(lambda: rt.twin_overlay.status_of("view_wrist_align") == "waiting")
        seq = src.seq
        time.sleep(0.3)  # > 3 overlay periods
        assert src.seq == seq  # nothing published while waiting
        tele = {t.stream_id: t for t in rt.twin_overlay.telemetry()}["view_wrist_align"]
        assert tele.detail == (
            "no sample from view yet - monitor error: connect failed: [Errno 113] No route to host"
        )
        assert tele.fps == 0.0
        rows = {c["camera_id"]: c for c in client.get("/api/cameras").json()}
        assert rows["view_wrist_align"]["live"] is False
        assert rows["grip_wrist_align"]["live"] is True  # per-arm
        assert rt.twin_overlay.status_of("grip_wrist_align") == "live"
    finally:
        view.sample = SAMPLES["view"]
        view.forced_status = None
        view.detail_text = ""
    assert _wait(lambda: rt.twin_overlay.status_of("view_wrist_align") == "live")


def test_no_real_frame_means_waiting_and_not_live(client, rt, cams):
    cams["view_wrist"].blackout = True
    try:
        assert _wait(lambda: rt.twin_overlay.status_of("view_wrist_align") == "waiting")
        tele = {t.stream_id: t for t in rt.twin_overlay.telemetry()}["view_wrist_align"]
        assert tele.detail == "no frame from view_wrist" and tele.fps == 0.0
        rows = {c["camera_id"]: c for c in client.get("/api/cameras").json()}
        assert rows["view_wrist_align"]["live"] is False
        assert rows["grip_wrist_align"]["live"] is True
    finally:
        cams["view_wrist"].blackout = False
    assert _wait(lambda: rt.twin_overlay.status_of("view_wrist_align") == "live")


def test_hardware_session_predicate_pauses_monitor_and_overlays(client, rt, factory):
    grip = factory.monitors["grip"]
    starts = grip.calls.count("start")
    rt._hardware_session_active = lambda: True  # a hardware session owns the boxes
    try:
        assert _wait(lambda: rt.hardware_monitor.paused)  # supervisor edge (0.5 s poll)
        assert grip.calls[-1] == "disconnect" and grip.status == "paused"
        assert _wait(lambda: rt.twin_overlay.status_of("grip_wrist_align") == "waiting")
        tele = {t.stream_id: t for t in rt.twin_overlay.telemetry()}
        assert all("hardware session" in t.detail for t in tele.values())
        rows = {c["camera_id"]: c for c in client.get("/api/cameras").json()}
        assert rows["grip_wrist_align"]["live"] is False
        assert rows["view_wrist_align"]["live"] is False
        with client.websocket_connect("/ws/telemetry") as ws:
            hm = ws.receive_json()["hardware_monitor"]
        assert hm["paused"] is True and {a["status"] for a in hm["arms"]} == {"paused"}
        assert {o["status"] for o in hm["overlays"]} == {"waiting"}
    finally:
        rt._hardware_session_active = lambda: False
    assert _wait(lambda: not rt.hardware_monitor.paused)
    assert grip.calls.count("start") == starts + 1  # reconnected once
    assert _wait(lambda: rt.twin_overlay.status_of("grip_wrist_align") == "live")


def test_stop_order_closes_overlay_then_monitor(tmp_path, cams):
    """Runtime.stop(): overlay first (renderers closed on its thread, streams
    removed), then the monitor (boxes released), then the rest."""
    factory = FakeMonitorFactory(dict(SAMPLES))
    runtime = Runtime(_config(tmp_path, env_outline=False), monitor_factory=factory)
    local = {cid: StaticCamera(cid) for cid in cams}
    runtime.manager.camera_factory = lambda cam_cfg: local[cam_cfg.id]
    runtime.start()
    try:
        assert runtime.twin_overlay.wait_ready(60.0)
        assert _wait(lambda: runtime.twin_overlay.status_of("grip_wrist_align") == "live")
        assert runtime.hub.has("grip_wrist_align")
        assert factory.monitors["grip"].calls == ["start"]
    finally:
        runtime.stop()
    assert not runtime.hub.has("grip_wrist_align") and not runtime.hub.has("view_wrist_align")
    assert runtime.twin_overlay._thread is None and runtime.twin_overlay._registered == []
    assert runtime.twin_overlay.status_of("grip_wrist_align") == "off"
    assert factory.monitors["grip"].calls[-1] == "stop" and runtime.hardware_monitor._thread is None
    assert os.environ.get("MUJOCO_GL") == "egl"
