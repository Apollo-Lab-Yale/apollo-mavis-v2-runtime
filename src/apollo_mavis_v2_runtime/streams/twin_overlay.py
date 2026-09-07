"""Digital-twin alignment overlays (phase-09a; 04-runtime §13.4 ``<camera_id>_align``).

One :class:`TwinOverlayRenderer` thread owns a PRIVATE copy of the hardware
workcell's digital-twin scene (``REGISTRY.build(twin_scene,
SceneOverrides(microphones=..., base_pose=...))`` — never the phase-09 gate's
``DigitalTwin``) and, at ``cfg.fps``, composites for every hardware wrist
camera the REAL frame (the preview ``OpenCVCamera`` already open in
``SessionManager``; UVC nodes cannot be opened twice) with the twin rendered
from the SAME wrist camera:

* the twin is posed from the read-only monitor's samples — ``qpos[j1..j7] =
  q + [joint1_offset_rad, 0, ...]`` (identity convention verified
  2026-09-04), ``qpos[rail] = rail_pos_m`` (``rail_flip`` -> ``0.65 - pos``)
  or ``rail_fallback_m[arm]`` while the track is not homed / enabled,
  gripper fingers from ``gripper_open_frac`` via the sim gripper mapping
  (``(1 - f) * DRIVER_CLOSED_RAD``);
* the spec copy sets ``visual.quality.offsamples = 0`` (multisampling blends
  segmentation ids at edges into OTHER valid geom ids), the wrist camera's
  ``resolution / sensor_size / focal_pixel / principal_pixel`` from
  ``CameraConfig.intrinsics`` (MuJoCo's principal-point offset has the
  OPPOSITE sign of OpenCV's: ``principal_pixel = [W/2 - cx, H/2 - cy]``, plus
  the per-camera overlay-only ``principal_offset_px`` nudge - ``cx += du``,
  ``cy += dv`` - that aligns a wrist camera mounted slightly off the shared
  ``wrist_cam`` pose WITHOUT changing the recorded intrinsics), and
  moves the environment geoms (floor / table / obstacle) to geom group 4 so
  ``MjvOption.geomgroup[4] = 0`` hides them in the robot passes;
* an RGB pass gives the twin's shading, a segmentation pass the robot mask
  (robot = every non-environment geom, mic and camera bodies included); the
  robot pixels become ``alpha * tint + (1 - alpha) * real`` with ``tint =
  tint_rgb * (0.55 + 0.45 * luminance)`` plus a 1 px ``edge_rgb`` outline;
  with ``env_outline`` an env-only segmentation pass draws the table /
  obstacle edges (Canny on the id image) in ``env_rgb`` as the base-placement
  cue; a stale / erroring / paused monitor swaps the tint for
  ``stale_tint_rgb`` — but only for an arm that HAS a sample: an arm the
  monitor never read (box off, still connecting) leaves its stream
  ``waiting`` (nothing published, ``/api/cameras`` ``live: false``) rather
  than drawing the keyframe posture as if it had been measured.

During a hardware session (phase-09c) the monitor is paused, so
:meth:`TwinOverlayRenderer.set_state_provider` swaps the sample source: the
session installs a provider that serves the session arms from the driver's
``workcell.states()`` (track rail convention; ``rail_flip`` is applied here as
usual) and the unselected arms from their frozen last sample (grey tint,
"frozen at last sample"); the streams keep their ids and ``paused()`` is
ignored while a provider is installed. Teardown restores the monitor source.

GL contexts are thread-affine: both ``mujoco.Renderer`` instances (RGB and
segmentation) and the ``MjData`` are created AND closed inside the overlay
thread. The published ``CameraFrame`` reuses the real frame's ``t_mono`` /
``wallclock_ns``; each stream is a plain VideoHub FrameSource
(:class:`TwinOverlaySource`) so ``/ws/video/<id>_align`` serves the usual
``<dI`` JPEG frames. Telemetry per stream: ``TwinOverlayTelemetry`` (fps over
a 1 s window, ``mask_fraction``, the rail fallback in use, the joint-1 knob).
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from apollo_mavis_v2_core import CameraConfig, CameraFrame, LatestSlot, WorkcellConfig
from apollo_mavis_v2_core.protocol import (
    RESERVED_STREAM_IDS,
    TwinOverlayStatus,
    TwinOverlayTelemetry,
)
from apollo_mavis_v2_core.schemas import CameraIntrinsics, PoseModel

from ..config import TwinOverlayConfig
from .hub import VideoHub

logger = logging.getLogger(__name__)

RAIL_TRAVEL_M = 0.65  # linear track travel; rail_flip maps q_sim = RAIL_TRAVEL_M - q_track
DEFAULT_COLOUR_FOVY_DEG = 43.2  # D435i colour imager at 640x480: 2*atan(240/607) (NOT the depth 57)
ENV_GEOM_GROUP = 4  # environment geoms are moved here so geomgroup[4] hides them
FPS_WINDOW_S = 1.0
STOP_JOIN_TIMEOUT_S = 5.0
IDLE_SLEEP_S = 0.05
OVERLAY_LABELS = {  # user-facing tile labels (ids stay internal)
    "grip": "Manipulation · twin overlay",
    "view": "Perception · twin overlay",
}
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def overlay_label(arm_id: str) -> str:
    return OVERLAY_LABELS.get(arm_id, f"{arm_id} · twin overlay")


def arm_for_camera(cam: CameraConfig, arm_ids: list[str]) -> str | None:
    """Arm a hardware camera is mounted on: ``extrinsics_frame: ee:<arm>`` when
    declared, else the longest arm id that prefixes the camera id
    (``grip_wrist`` -> ``grip``). ``None`` = not a wrist camera."""
    frame = cam.extrinsics_frame
    if frame and frame.startswith("ee:"):
        ident = frame.split(":", 1)[1]
        return ident if ident in arm_ids else None
    matches = [a for a in arm_ids if cam.id.startswith(f"{a}_")]
    return max(matches, key=len) if matches else None


def principal_pixel(
    intr: CameraIntrinsics,
    width: int,
    height: int,
    offset: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float]:
    """MuJoCo principal-point offset from OpenCV intrinsics: the SIGN is
    opposite (verified < 0.5 px on this box), so ``[W/2 - cx, H/2 - cy]``.

    ``offset`` is a per-camera overlay-only nudge ``(du, dv)`` applied as
    ``cx += du``, ``cy += dv`` (module docstring / ``principal_offset_px``): it
    aligns a wrist camera whose physical mount differs from the shared
    ``wrist_cam`` pose without touching the true recorded intrinsics."""
    cx = float(intr.cx) + float(offset[0])
    cy = float(intr.cy) + float(offset[1])
    return (width / 2.0 - cx, height / 2.0 - cy)


def apply_intrinsics(
    cam: Any,
    intr: CameraIntrinsics | None,
    width: int,
    height: int,
    principal_offset: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Configure an ``MjsCamera`` to render exactly ``width x height`` pixels
    with the given pinhole intrinsics (``fovy`` fallback without them).

    ``principal_offset`` nudges only the rendered principal point (see
    :func:`principal_pixel`); it does not change ``intr``."""
    if intr is None:
        cam.fovy = DEFAULT_COLOUR_FOVY_DEG
        return
    cam.resolution = [int(width), int(height)]
    cam.sensor_size = [width * 1e-5, height * 1e-5]
    cam.focal_pixel = [float(intr.fx), float(intr.fy)]
    cam.principal_pixel = list(principal_pixel(intr, width, height, principal_offset))


def subtree_bodies(model: Any, root: int) -> set[int]:
    """Body ids of ``root`` and every descendant (``body_parentid`` walk)."""
    ids = {int(root)}
    for b in range(int(model.nbody)):  # parents precede children in MuJoCo's body order
        if b != root and int(model.body_parentid[b]) in ids:
            ids.add(b)
    return ids


def base_pose_overrides(workcell: WorkcellConfig) -> dict[str, Any]:
    """``SceneOverrides.base_pose`` entries for arms whose ``base_in_world`` is
    configured (differs from the identity default); the scene descriptor
    places the others."""
    identity = PoseModel()
    return {a.id: a.base_in_world.to_pose() for a in workcell.arms if a.base_in_world != identity}


def composite(
    real: np.ndarray,
    twin: np.ndarray,
    mask: np.ndarray,
    cfg: TwinOverlayConfig,
    *,
    stale: bool = False,
    env_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Blend the twin's robot pixels into the real frame (see module docstring).

    ``real`` / ``twin``: uint8 (H, W, 3) RGB; ``mask``: bool (H, W) robot
    pixels; ``env_ids``: int (H, W) env-only segmentation geom ids (-1 =
    background) or ``None`` to skip the environment outline.
    """
    out = real.copy()
    tint_rgb = cfg.stale_tint_rgb if stale else cfg.tint_rgb
    edge_rgb = cfg.stale_tint_rgb if stale else cfg.edge_rgb
    if env_ids is not None:
        _, inverse = np.unique(env_ids, return_inverse=True)
        n = int(inverse.max()) + 1 if inverse.size else 1
        step = 255 // max(1, n - 1)
        levels = (inverse.reshape(env_ids.shape) * step).astype(np.uint8)
        edges = cv2.Canny(levels, 20, 60) > 0
        edges &= ~mask  # the robot occludes the environment
        out[edges] = np.asarray(cfg.env_rgb, dtype=np.uint8)
    if mask.any():
        shade = (twin[mask].astype(np.float32) @ LUMA) / 255.0  # (N,)
        tint = np.asarray(tint_rgb, dtype=np.float32)[None, :] * (0.55 + 0.45 * shade[:, None])
        blended = cfg.alpha * tint + (1.0 - cfg.alpha) * real[mask].astype(np.float32)
        out[mask] = np.clip(blended, 0.0, 255.0).astype(np.uint8)
        m8 = mask.astype(np.uint8)
        eroded = cv2.erode(m8, np.ones((3, 3), dtype=np.uint8)) > 0
        out[mask & ~eroded] = np.asarray(edge_rgb, dtype=np.uint8)  # 1 px inner outline
    return out


class TwinOverlaySource:
    """VideoHub FrameSource + telemetry state of one ``<camera_id>_align`` stream."""

    def __init__(
        self,
        stream_id: str,
        camera_id: str,
        arm_id: str,
        twin_camera: str,
        resolution: tuple[int, int],
        fps: float,
        intrinsics: CameraIntrinsics | None,
        principal_offset: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.stream_id = stream_id
        self.camera_id = camera_id
        self.arm_id = arm_id
        self.twin_camera = twin_camera
        self.resolution = (int(resolution[0]), int(resolution[1]))  # (W, H)
        self.fps = float(fps)
        self.intrinsics = intrinsics
        self.principal_offset = (float(principal_offset[0]), float(principal_offset[1]))
        self.label = overlay_label(arm_id)
        self.slot: LatestSlot[CameraFrame] = LatestSlot()
        self._lock = threading.Lock()
        self._status: TwinOverlayStatus = "off"
        self._detail = "overlay not started"
        self._rail_fallback_m: float | None = None
        self._mask_fraction = 0.0
        self._seq = 0
        self._times: deque[float] = deque(maxlen=256)

    # -- FrameSource -----------------------------------------------------------------------
    def latest(self) -> CameraFrame | None:
        got = self.slot.get()
        return got[0] if got is not None else None

    # -- state (overlay thread writes, REST / telemetry threads read) ---------------------------
    def set_status(
        self,
        status: TwinOverlayStatus,
        detail: str = "",
        *,
        rail_fallback_m: float | None = None,
        mask_fraction: float | None = None,
    ) -> None:
        with self._lock:
            self._status = status
            self._detail = detail
            self._rail_fallback_m = rail_fallback_m
            if mask_fraction is not None:
                self._mask_fraction = float(mask_fraction)

    def publish(self, rgb: np.ndarray, real: CameraFrame, now: float) -> CameraFrame:
        with self._lock:
            self._seq += 1
            seq = self._seq
            self._times.append(now)
        frame = CameraFrame(
            camera_id=self.stream_id,
            rgb=rgb,
            t_mono=real.t_mono,
            wallclock_ns=real.wallclock_ns,
            seq=seq,
        )
        self.slot.put(frame)
        return frame

    @property
    def status(self) -> TwinOverlayStatus:
        with self._lock:
            return self._status

    @property
    def detail(self) -> str:
        with self._lock:
            return self._detail

    @property
    def seq(self) -> int:
        with self._lock:
            return self._seq

    def measured_fps(self, now: float) -> float:
        with self._lock:
            recent = [t for t in self._times if t > now - FPS_WINDOW_S]
        return len(recent) / FPS_WINDOW_S

    def telemetry(self, now: float, joint1_offset_rad: float) -> TwinOverlayTelemetry:
        with self._lock:
            status, detail = self._status, self._detail
            fallback, frac = self._rail_fallback_m, self._mask_fraction
        return TwinOverlayTelemetry(
            stream_id=self.stream_id,
            camera_id=self.camera_id,
            arm_id=self.arm_id,
            status=status,
            detail=detail,
            fps=self.measured_fps(now) if status in ("live", "stale") else 0.0,
            rail_fallback_m=fallback,
            joint1_offset_rad=joint1_offset_rad,
            mask_fraction=frac,
        )


@dataclass
class _ArmSlots:
    """Precomputed qpos indices of one twin arm (core order, rail last)."""

    joint_adr: np.ndarray  # (7,)
    rail_adr: int | None
    gripper_adr: np.ndarray  # (k,) finger joints, all set to the driver angle


@dataclass
class _Context:
    """Everything the overlay thread owns (never touched from outside it)."""

    mujoco: Any
    model: Any
    data: Any
    arms: dict[str, _ArmSlots]
    driver_closed_rad: float
    opt_robot: Any  # MjvOption: env group hidden
    opt_env: Any  # MjvOption: env group only
    renderers: dict[tuple[int, int], tuple[Any, Any]]  # (H, W) -> (rgb, seg)

    def renderer_pair(self, height: int, width: int) -> tuple[Any, Any]:
        key = (height, width)
        pair = self.renderers.get(key)
        if pair is None:
            rgb = self.mujoco.Renderer(self.model, height=height, width=width)
            seg = self.mujoco.Renderer(self.model, height=height, width=width)
            seg.enable_segmentation_rendering()
            pair = (rgb, seg)
            self.renderers[key] = pair
        return pair

    def close(self) -> None:
        for rgb, seg in self.renderers.values():
            for r in (rgb, seg):
                try:
                    r.close()
                except Exception:  # noqa: BLE001 - best-effort teardown
                    logger.warning("overlay renderer close failed", exc_info=True)
        self.renderers.clear()


class TwinOverlayRenderer:
    """Owner of the overlay thread and its ``<camera_id>_align`` streams.

    ``frame_source(camera_id)`` returns the newest REAL ``CameraFrame`` of a
    hardware camera (``SessionManager.hardware_camera(...).latest()``) or
    ``None``; ``monitor`` supplies the arm samples and per-arm status;
    ``paused()`` (default: ``monitor.paused``) stops publishing while a
    hardware session owns the arms. ``start()`` registers the streams on the
    hub immediately (status ``waiting``), the thread builds the scene, then
    renders at ``cfg.fps``; ``stop()`` ends the thread (renderers closed
    in-thread) and removes the streams. Inert (``enabled`` False + ``detail``)
    when disabled, without wrist cameras or without the ``[sim]`` extra.
    """

    def __init__(
        self,
        cfg: TwinOverlayConfig,
        workcell: WorkcellConfig,
        twin_scene: str | None,
        monitor,  # HardwareStateMonitor
        frame_source: Callable[[str], CameraFrame | None],
        hub: VideoHub,
        *,
        paused: Callable[[], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self.workcell = workcell
        self.twin_scene = twin_scene
        self.monitor = monitor
        self.frame_source = frame_source
        self.hub = hub
        self.paused_fn = paused or (lambda: bool(getattr(monitor, "paused", False)))
        self._clock = clock
        self.streams: dict[str, TwinOverlaySource] = {}
        self.detail = ""  # why inert ('' = enabled)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()  # scene built (or build failed)
        self._registered: list[str] = []
        self.frames_rendered = 0
        # phase-09c: alternative sample source while a hardware session owns the arms
        # (``samples() -> {arm_id: sample}``, ``status_of(arm_id) -> (status, detail)``)
        self._provider: Any = None
        self._plan_streams()

    # -- construction --------------------------------------------------------------------
    def _plan_streams(self) -> None:
        if not self.cfg.enabled:
            self.detail = "twin overlay disabled (twin_overlay.enabled: false)"
            return
        if not self.twin_scene:
            self.detail = "hardware workcell has no digital_twin_scene"
            return
        try:
            import mujoco  # noqa: F401 - presence check only; used in-thread
            from apollo_mavis_v2_sim import REGISTRY  # noqa: F401
        except Exception as e:  # noqa: BLE001 - [sim] extra absent or broken
            self.detail = f"sim package not importable ({type(e).__name__}: {e})"
            return
        arm_ids = [a.id for a in self.workcell.arms]
        camera_ids = {c.id for c in self.workcell.cameras}
        for cam in self.workcell.cameras:
            arm_id = arm_for_camera(cam, arm_ids)
            if arm_id is None:
                continue
            stream_id = f"{cam.id}{self.cfg.stream_suffix}"
            if (
                stream_id in camera_ids
                or stream_id in RESERVED_STREAM_IDS
                or stream_id.endswith("_wrist_cam")
            ):
                logger.error("twin overlay stream id %r is not allowed; skipped", stream_id)
                continue
            offset = self.cfg.principal_offset_px.get(cam.id, (0.0, 0.0))
            self.streams[stream_id] = TwinOverlaySource(
                stream_id,
                cam.id,
                arm_id,
                f"{arm_id}_wrist_cam",
                tuple(cam.resolution),
                self.cfg.fps,
                cam.intrinsics,
                (float(offset[0]), float(offset[1])),
            )
            if cam.intrinsics is None:
                logger.warning(
                    "camera %r has no intrinsics: twin overlay %r renders with fovy %.1f deg",
                    cam.id,
                    stream_id,
                    DEFAULT_COLOUR_FOVY_DEG,
                )
        if not self.streams:
            self.detail = "no hardware wrist camera matches a configured arm"

    @property
    def enabled(self) -> bool:
        return not self.detail and bool(self.streams)

    # -- lifecycle -----------------------------------------------------------------------
    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._stop.clear()
        self._ready.clear()
        for sid, src in self.streams.items():
            if self.hub.has(sid):
                src.set_status("error", f"stream id {sid!r} already exists on the hub")
                continue
            self.hub.add_stream(sid, src, self.cfg.fps)
            self._registered.append(sid)
            src.set_status("waiting", "building the digital twin")
        self._thread = threading.Thread(target=self._run, name="twin-overlay", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = STOP_JOIN_TIMEOUT_S) -> None:
        """Stop the thread FIRST (renderers are closed on it), then drop the streams."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("twin overlay thread did not stop within %.1f s", timeout)
            self._thread = None
        for sid in self._registered:
            self.hub.remove_stream(sid)
        self._registered = []
        for src in self.streams.values():
            src.set_status("off", "overlay stopped")

    def wait_ready(self, timeout: float = 30.0) -> bool:
        """Block until the twin scene is built (or failed); tests."""
        return self._ready.wait(timeout)

    def set_state_provider(self, provider: Any) -> None:
        """Install (``provider``) or remove (``None``) the hardware-session sample
        source (module docstring). The provider is read once per frame on the
        overlay thread; assignment is atomic."""
        self._provider = provider

    @property
    def state_provider(self) -> Any:
        return self._provider

    # -- readers -------------------------------------------------------------------------
    def status_of(self, stream_id: str) -> TwinOverlayStatus:
        src = self.streams.get(stream_id)
        return src.status if src is not None else "off"

    def stream_for_camera(self, camera_id: str) -> TwinOverlaySource | None:
        for src in self.streams.values():
            if src.camera_id == camera_id:
                return src
        return None

    def telemetry(self) -> list[TwinOverlayTelemetry]:
        now = self._clock()
        return [s.telemetry(now, self.cfg.joint1_offset_rad) for s in self.streams.values()]

    # -- thread body -----------------------------------------------------------------------
    def _run(self) -> None:
        ctx: _Context | None = None
        try:
            try:
                ctx = self._build()
            except Exception as e:  # noqa: BLE001 - the scene/GL failed: streams go off
                logger.exception("twin overlay: digital twin build failed")
                for src in self.streams.values():
                    src.set_status("off", f"twin scene build failed: {type(e).__name__}: {e}")
                return
            finally:
                self._ready.set()
            for src in self.streams.values():
                src.set_status("waiting", f"no frame from {src.camera_id}")
            period = 1.0 / self.cfg.fps
            next_t = self._clock()
            while not self._stop.is_set():
                self._tick(ctx)
                next_t += period
                lag = self._clock() - next_t
                if lag > period:
                    next_t = self._clock()
                elif lag < 0.0:
                    self._stop.wait(-lag)
        except Exception:  # last line of defence
            logger.exception("twin overlay thread crashed")
            for src in self.streams.values():
                src.set_status("error", "overlay thread crashed (see log)")
        finally:
            if ctx is not None:
                ctx.close()

    def _build(self) -> _Context:
        import mujoco
        from apollo_mavis_v2_sim import DRIVER_CLOSED_RAD, REGISTRY, Addressing, SceneOverrides

        overrides = SceneOverrides(
            microphones={a.id: bool(a.microphone) for a in self.workcell.arms},
            base_pose=base_pose_overrides(self.workcell),
        )
        built = REGISTRY.build(self.twin_scene, overrides)
        spec = built.spec
        spec.visual.quality.offsamples = 0  # REQUIRED: multisampling corrupts segmentation ids
        for src in self.streams.values():
            width, height = src.resolution
            try:
                cam = spec.camera(src.twin_camera)
            except Exception as e:  # noqa: BLE001 - camera-less arm in the scene
                src.set_status("error", f"twin camera {src.twin_camera!r} missing: {e}")
                continue
            apply_intrinsics(cam, src.intrinsics, width, height, src.principal_offset)
            spec.visual.global_.offwidth = max(int(spec.visual.global_.offwidth), width)
            spec.visual.global_.offheight = max(int(spec.visual.global_.offheight), height)
        model = spec.compile()
        addr = Addressing(model, built.meta)
        robot_bodies: set[int] = set()
        arms: dict[str, _ArmSlots] = {}
        for arm_id in built.meta.arm_ids:
            a = addr[arm_id]
            bodies = subtree_bodies(model, a.root_body_id)
            robot_bodies |= bodies
            chain = {int(x) for x in a.qpos_adr}
            fingers = [
                int(model.jnt_qposadr[j])
                for j in range(model.njnt)
                if int(model.jnt_bodyid[j]) in bodies and int(model.jnt_qposadr[j]) not in chain
            ]
            arms[arm_id] = _ArmSlots(
                joint_adr=np.asarray(a.qpos_adr[:7], dtype=np.intp),
                rail_adr=int(a.qpos_adr[7]) if a.has_rail else None,
                gripper_adr=np.asarray(fingers, dtype=np.intp),
            )
        env_geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) not in robot_bodies]
        model.geom_group[env_geoms] = ENV_GEOM_GROUP
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        opt_robot = mujoco.MjvOption()
        opt_robot.geomgroup[ENV_GEOM_GROUP] = 0
        opt_env = mujoco.MjvOption()
        opt_env.geomgroup[:] = 0
        opt_env.geomgroup[ENV_GEOM_GROUP] = 1
        ctx = _Context(
            mujoco=mujoco,
            model=model,
            data=data,
            arms=arms,
            driver_closed_rad=float(DRIVER_CLOSED_RAD),
            opt_robot=opt_robot,
            opt_env=opt_env,
            renderers={},
        )
        for src in self.streams.values():  # create the GL contexts in-thread, up front
            if src.status != "error":
                ctx.renderer_pair(src.resolution[1], src.resolution[0])
        return ctx

    # -- one frame for every stream -----------------------------------------------------------
    def _pose_arms(self, ctx: _Context, samples: dict[str, Any]) -> dict[str, tuple[float, str]]:
        """Write every sampled arm into ``ctx.data.qpos``; returns
        ``{arm_id: (fallback_m, detail)}`` for arms whose rail used the fallback."""
        fallbacks: dict[str, tuple[float, str]] = {}
        qpos = ctx.data.qpos
        for arm_id, slots in ctx.arms.items():
            sample = samples.get(arm_id)
            if sample is None or len(sample.q) < 7:
                continue
            q = np.asarray(sample.q[:7], dtype=np.float64)
            q[0] += self.cfg.joint1_offset_rad
            qpos[slots.joint_adr] = q
            if slots.rail_adr is not None:
                pos = sample.rail_pos_m
                if pos is None:
                    fb = float(self.cfg.rail_fallback_m.get(arm_id, 0.0))
                    why = "rail not enabled" if sample.rail_homed else "rail not homed"
                    fallbacks[arm_id] = (fb, f"{why} - twin assumes {fb:.2f} m")
                    pos = fb
                elif self.cfg.rail_flip:
                    pos = RAIL_TRAVEL_M - float(pos)
                qpos[slots.rail_adr] = min(RAIL_TRAVEL_M, max(0.0, float(pos)))
            if slots.gripper_adr.size and sample.gripper_open_frac is not None:
                f = min(1.0, max(0.0, float(sample.gripper_open_frac)))
                qpos[slots.gripper_adr] = (1.0 - f) * ctx.driver_closed_rad
        ctx.mujoco.mj_forward(ctx.model, ctx.data)
        return fallbacks

    def _tick(self, ctx: _Context) -> None:
        mj = ctx.mujoco
        provider = self._provider
        if provider is None:
            if self.paused_fn():
                for src in self.streams.values():
                    if src.status != "error":
                        src.set_status("waiting", "paused - a hardware session owns the arms")
                return
            samples = self.monitor.snapshot()
            status_of = self.monitor.status_of
        else:  # hardware session: driver states + frozen arms (phase-09c)
            try:
                samples = provider.samples()
            except Exception:  # noqa: BLE001 - a provider failure never kills the overlay
                logger.exception("twin overlay: session state provider failed")
                samples = {}
            status_of = provider.status_of
        fallbacks = self._pose_arms(ctx, samples)
        now = self._clock()
        for src in self.streams.values():
            if src.status == "error":
                continue
            try:
                real = self.frame_source(src.camera_id)
            except Exception as e:  # noqa: BLE001 - camera errors never kill the overlay
                logger.debug("twin overlay: frame source %r failed: %r", src.camera_id, e)
                real = None
            if real is None:
                src.set_status("waiting", f"no frame from {src.camera_id}")
                continue
            mon_status, mon_detail = status_of(src.arm_id)
            mon_text = f"monitor {mon_status}" + (f": {mon_detail}" if mon_detail else "")
            if src.arm_id not in samples:
                # Never measured (box off / connecting): publish nothing rather than a
                # grey twin at the keyframe posture that was never read from the arm.
                src.set_status("waiting", f"no sample from {src.arm_id} yet - {mon_text}")
                continue
            stale = mon_status != "running"
            parts: list[str] = []
            if stale:
                parts.append(mon_text)
            fb = fallbacks.get(src.arm_id)
            if fb is not None:
                parts.append(fb[1])
            width, height = src.resolution
            rgb_r, seg_r = ctx.renderer_pair(height, width)
            rgb_r.update_scene(ctx.data, camera=src.twin_camera, scene_option=ctx.opt_robot)
            twin = rgb_r.render()
            seg_r.update_scene(ctx.data, camera=src.twin_camera, scene_option=ctx.opt_robot)
            seg = seg_r.render()
            mask = seg[..., 1] == int(mj.mjtObj.mjOBJ_GEOM)
            env_ids = None
            if self.cfg.env_outline:
                seg_r.update_scene(ctx.data, camera=src.twin_camera, scene_option=ctx.opt_env)
                env_seg = seg_r.render()
                env_ids = np.where(
                    env_seg[..., 1] == int(mj.mjtObj.mjOBJ_GEOM), env_seg[..., 0], -1
                )
            real_rgb = np.asarray(real.rgb)
            if real_rgb.shape[:2] != (height, width):
                real_rgb = cv2.resize(real_rgb, (width, height), interpolation=cv2.INTER_AREA)
            out = composite(real_rgb, twin, mask, self.cfg, stale=stale, env_ids=env_ids)
            src.publish(out, real, now)
            self.frames_rendered += 1
            src.set_status(
                "stale" if stale else "live",
                "; ".join(parts),
                rail_fallback_m=fb[0] if fb is not None else None,
                mask_fraction=float(mask.mean()),
            )


def default_fovy_deg(fy: float, height: int) -> float:
    """Vertical field of view (deg) of a pinhole camera: ``2 * atan(H / 2 / fy)``."""
    return math.degrees(2.0 * math.atan((height / 2.0) / float(fy)))


__all__ = [
    "DEFAULT_COLOUR_FOVY_DEG",
    "ENV_GEOM_GROUP",
    "OVERLAY_LABELS",
    "RAIL_TRAVEL_M",
    "TwinOverlayRenderer",
    "TwinOverlaySource",
    "apply_intrinsics",
    "arm_for_camera",
    "base_pose_overrides",
    "composite",
    "default_fovy_deg",
    "overlay_label",
    "principal_pixel",
]
