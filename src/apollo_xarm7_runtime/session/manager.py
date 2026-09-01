"""SessionManager — one session at a time (04-runtime §5).

IDLE -> BRINGUP -> START_FROM -> RUNNING -> TEARDOWN -> IDLE (+ FAULT).
``create()`` returns after BRINGUP; START_FROM progress rides telemetry.
Phase-05 implements teleop over the sim workcell; collect/dagger/inference
and the hardware workcell path return clean 409s until phases 07/08.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from apollo_xarm7_core import (
    ArmConfig,
    Command,
    ProfileNotFoundError,
    ProfileStore,
    WorkcellConfig,
)
from apollo_xarm7_core.protocol import SessionInfo, SessionSpec

from ..config import RuntimeConfig
from ..control.loop import ControlLoop
from ..errors import SessionError, SessionNotFoundError
from ..safety.gate import NullGate, SafetyGate
from ..safety.supervisor import SafetySupervisor
from ..safety.watchdog import ArmReportWatchdog, InputWatchdog
from ..streams.hub import VideoHub
from .types import SessionState

if TYPE_CHECKING:
    from ..bus import RuntimeBus

logger = logging.getLogger(__name__)

_UNIMPLEMENTED_MODES = {"collect": "phase-07", "dagger": "phase-08", "inference": "phase-08"}


def _servo_faithful_scene(scene_id: str):
    """Session workcell scene with hardware-grade servo fidelity.

    Mirrors ``guardrail_check._build_real_robot_scene``: the menagerie
    position actuators sag 1-2 cm under gravity and the rail spring-servo
    lags, while a real xArm7 + linear track hold commanded positions stiffly.
    Compile-time spec edits: gravity compensation on every body + a 40x
    stiffer rail servo. Twin/IK/previews keep the stock scene.
    """
    from apollo_xarm7_sim import REGISTRY, Addressing, BuiltScene

    scene = REGISTRY.build(scene_id)
    spec = scene.spec
    for body in spec.bodies:
        body.gravcomp = 1.0
    for joint in spec.joints:
        if joint.name.endswith("rail_joint"):
            joint.stiffness *= 40.0
            joint.damping *= 4.0
    for act in spec.actuators:
        if act.name.endswith("_rail"):
            act.gear[0] *= 40.0
    model = spec.compile()
    return BuiltScene(scene.meta, spec, model, spec.to_xml(), Addressing(model, scene.meta))


@dataclass
class ActiveSession:
    """Everything one running session owns (torn down in reverse)."""

    session_id: str
    spec: SessionSpec
    state: SessionState
    workcell: object
    loop: ControlLoop
    supervisor: SafetySupervisor
    twin: object | None
    render_service: object | None
    streams: list[str] = field(default_factory=list)  # video ids
    sources: list[object] = field(default_factory=list)  # started FrameSources
    start_from_progress: float | None = None
    fault_detail: str = ""


class SessionManager:
    """Owns the singleton session + the pre-session camera previews."""

    def __init__(
        self,
        cfg: RuntimeConfig,
        bus: RuntimeBus,
        hub: VideoHub,
        profile_store: ProfileStore,
        epoch: str,
    ) -> None:
        self.cfg = cfg
        self.bus = bus
        self.hub = hub
        self.profile_store = profile_store
        self.epoch = epoch
        self.session: ActiveSession | None = None
        self._lock = threading.Lock()
        self._preview_service = None
        self._preview_sources: list[object] = []
        self._preview_ids: list[str] = []

    # -- info ------------------------------------------------------------------
    @property
    def state(self) -> SessionState:
        return self.session.state if self.session else SessionState.IDLE

    def info(self) -> SessionInfo:
        s = self.session
        if s is None:
            raise SessionNotFoundError("no active session")
        return SessionInfo(
            session_id=s.session_id,
            epoch=self.epoch,
            mode=s.spec.mode,
            arms=list(s.spec.arms),
            streams=list(s.streams),
            state=s.state.value,
        )

    # -- validation ---------------------------------------------------------------
    def _validate(self, spec: SessionSpec) -> WorkcellConfig:
        if self.session is not None:
            raise SessionError("a session already exists")
        if spec.mode in _UNIMPLEMENTED_MODES:
            raise SessionError(
                f"mode {spec.mode!r} is not available yet ({_UNIMPLEMENTED_MODES[spec.mode]})"
            )
        wc = self.cfg.workcell_config(spec.kind)
        if wc is None:
            raise SessionError(f"no {spec.kind!r} workcell config available")
        if spec.kind == "hardware":
            raise SessionError("hardware sessions land with phase-09 integration")
        if not spec.arms:
            raise SessionError("session needs at least one arm")
        scene_id = spec.sim_scene or wc.sim_scene
        if not scene_id:
            raise SessionError("sim session needs a sim_scene")
        from apollo_xarm7_sim import REGISTRY, SceneNotFoundError

        try:
            meta = REGISTRY.meta(scene_id)
        except SceneNotFoundError as e:
            raise SessionError(str(e)) from e
        missing = [a for a in spec.arms if a not in meta.arm_ids]
        if missing:
            raise SessionError(
                f"arms {missing} not in scene {scene_id!r} (has {list(meta.arm_ids)})"
            )
        if spec.start_from.startswith("profile:"):
            pid = spec.start_from.split(":", 1)[1]
            try:
                profile = self.profile_store.get(pid)
            except ProfileNotFoundError as e:
                raise SessionError(f"unknown profile {pid!r}") from e
            uncovered = [a for a in spec.arms if a not in profile.arms]
            if uncovered:
                raise SessionError(f"profile {pid!r} does not cover arms {uncovered}")
        return wc

    # -- create (BRINGUP; returns while START_FROM runs) ---------------------------
    def create(self, spec: SessionSpec) -> SessionInfo:
        with self._lock:
            wc = self._validate(spec)
            session = self._bringup_sim(spec, wc)
            self.session = session
        threading.Thread(
            target=self._start_from_worker, args=(session,), name="start-from", daemon=True
        ).start()
        return self.info()

    def _bringup_sim(self, spec: SessionSpec, wc: WorkcellConfig) -> ActiveSession:
        from apollo_xarm7_sim import (
            REGISTRY,
            DigitalTwin,
            IKParams,
            MinkIKSolver,
            RenderService,
            SimWorkcell,
            default_collision_pairs,
        )

        from ..control.fk import SceneKinematics
        from ..streams.render_source import RenderStreamSource

        self.stop_previews()
        scene_id = spec.sim_scene or wc.sim_scene
        safety = wc.safety
        session_cfg = self._session_workcell_config(spec, wc, scene_id)
        rs = RenderService()
        rs.start()
        scene = _servo_faithful_scene(scene_id)
        workcell = SimWorkcell(scene, session_cfg, render_service=rs)
        workcell.start()

        # Twin: always built for planning (goto / start_from); it is the GATE
        # twin only under safety_debug (11-safety §5).
        twin = DigitalTwin(
            REGISTRY.build(scene_id),
            inflation_m=safety.geom_inflation_m,
            render_service=rs if safety.safety_debug else None,
            allowed_pairs_extra=safety.allowed_pairs_extra,
        )
        if safety.safety_debug:
            gate = SafetyGate(twin, safety)
            pairs = default_collision_pairs(twin.scene, twin.allowed)
        else:
            gate = NullGate()
            pairs = None
        ik = MinkIKSolver(
            twin.scene,
            IKParams(min_distance_m=safety.geom_inflation_m + 0.002),
            collision_pairs=pairs,
        )
        kin = SceneKinematics(twin.scene)
        supervisor = SafetySupervisor(
            gate,
            InputWatchdog(self.cfg.control.watchdog.stale_s, self.cfg.control.watchdog.ramp_s),
            twin=twin if safety.safety_debug else None,
            report_watchdog=ArmReportWatchdog(safety.twin_staleness_s),
            warn_clearance_m=safety.warn_clearance_m,
        )
        loop = ControlLoop(
            workcell,
            self.cfg.control,
            self.bus,
            supervisor,
            list(spec.arms),
            ik=ik,
            kin=kin,
            planner=twin,
            profile_store=self.profile_store,
            workcell_kind="sim",
        )
        loop.start()

        # Video: session cameras at session fps + reserved "sim"/"twin".
        session = ActiveSession(
            session_id=uuid.uuid4().hex,
            spec=spec,
            state=SessionState.BRINGUP,
            workcell=workcell,
            loop=loop,
            supervisor=supervisor,
            twin=twin,
            render_service=rs,
        )
        fps = self.cfg.video.session_fps
        for cam_id, cam in workcell.cameras.items():
            self.hub.add_stream(cam_id, cam, fps)
            session.streams.append(cam_id)
        sim_src = RenderStreamSource(rs, "sim", "sim", camera=None, fps=fps)
        sim_src.start()
        self.hub.add_stream("sim", sim_src, fps)
        session.sources.append(sim_src)
        session.streams.append("sim")
        if safety.safety_debug:
            twin_src = RenderStreamSource(rs, "twin", "twin", camera=None, fps=fps)
            twin_src.start()
            self.hub.add_stream("twin", twin_src, fps)
            session.sources.append(twin_src)
            session.streams.append("twin")
        return session

    def _session_workcell_config(
        self, spec: SessionSpec, wc: WorkcellConfig, scene_id: str
    ) -> WorkcellConfig:
        by_id = {a.id: a for a in wc.arms}
        arms = [
            by_id.get(a) or ArmConfig(id=a, base_in_world={})  # scene places sim arms
            for a in spec.arms
        ]
        cams = [c for c in wc.cameras if c.kind == "sim"]
        return wc.model_copy(update={"arms": arms, "cameras": cams, "sim_scene": scene_id})

    # -- START_FROM (background; progress via telemetry) ----------------------------
    def _start_from_worker(self, session: ActiveSession) -> None:
        try:
            if not session.spec.start_from.startswith("profile:"):
                session.state = SessionState.RUNNING  # keep_current: no motion
                return
            session.state = SessionState.START_FROM
            session.start_from_progress = 0.0
            pid = session.spec.start_from.split(":", 1)[1]
            profile = self.profile_store.get(pid)
            states = session.workcell.states()
            q_start: dict[str, list[float]] = {}
            q_goal: dict[str, list[float]] = {}
            grippers: dict[str, float] = {}
            for arm_id in session.spec.arms:
                st = states[arm_id]
                posture = profile.arms[arm_id]
                goal = list(posture.q)
                if st.q.shape[0] > 7:  # rail slot LAST
                    rail = posture.rail_pos_m
                    goal.append(float(st.q[7]) if rail is None else float(rail))
                q_start[arm_id] = [float(x) for x in st.q]
                q_goal[arm_id] = goal
                grippers[arm_id] = float(posture.gripper_open_frac)
            from apollo_xarm7_core import PlanRequest

            if session.supervisor.twin is None:  # plain sim: keep the plan twin fresh
                session.twin.sync(states)
            result = session.twin.plan(PlanRequest(q_start=q_start, q_goal=q_goal))
            if not result.ok:
                logger.error("start_from plan failed: %s %s", result.failure,
                             result.failing_pair)
                session.fault_detail = f"start_from plan failed: {result.failure}"
                session.state = SessionState.RUNNING  # arms stay held; operator decides
                session.start_from_progress = None
                self.bus.commands.submit(Command(
                    op="_plan_ready",
                    args={"arms": list(q_goal), "result": result, "detail": ""},
                    source="internal",
                ))
                return
            total = sum(len(w) for w in result.waypoints.values()) or 1
            self.bus.commands.submit(Command(
                op="execute_plan",
                args={"waypoints": result.waypoints, "gripper": grippers},
                source="internal",
            ))
            plans = session.loop.plans
            deadline = time.monotonic() + 120.0
            while time.monotonic() < deadline and session.state == SessionState.START_FROM:
                active = plans.active_arms
                remaining = sum(
                    len(plans._waypoints.get(a, ())) - plans._index.get(a, 0)
                    for a in active
                )
                session.start_from_progress = 1.0 - remaining / total
                if not active and session.loop.tick_count > 1:
                    break
                time.sleep(0.05)
            session.start_from_progress = None
            session.state = SessionState.RUNNING
        except Exception as e:
            logger.exception("start_from worker failed")
            session.fault_detail = repr(e)
            session.state = SessionState.FAULT

    # -- teardown -------------------------------------------------------------------
    def teardown(self) -> None:
        """DELETE /api/session (idempotent); also SIGTERM/fatal-error path."""
        with self._lock:
            session = self.session
            if session is None:
                return
            session.state = SessionState.TEARDOWN
            try:
                session.loop.stop()
                for sid in session.streams:
                    self.hub.remove_stream(sid)
                for src in session.sources:
                    src.stop()
                if session.render_service is not None:
                    session.render_service.stop()
                session.workcell.stop()
            finally:
                self.session = None
        self.start_previews()

    # -- pre-session camera previews (~15 fps, 04-runtime §13.4) --------------------
    def start_previews(self) -> None:
        if self._preview_service is not None:
            return
        wc = self.cfg.workcell_config("sim")
        if wc is None or not wc.sim_scene:
            return
        try:
            from apollo_xarm7_sim import REGISTRY, RenderService, SimCamera
        except ImportError:
            return
        rs = RenderService()
        rs.start()
        scene = REGISTRY.build(wc.sim_scene)
        rs.register_source("preview", scene.model)
        self._preview_scene = scene
        fps = self.cfg.video.preview_fps
        for name in scene.meta.cameras:
            cam = SimCamera(
                camera_id=name, render_service=rs, mjcf_camera=name,
                fps=fps, source="preview",
            )
            cam.start()
            self.hub.add_stream(name, cam, fps)
            self._preview_sources.append(cam)
            self._preview_ids.append(name)
        self._preview_service = rs

    def stop_previews(self) -> None:
        if self._preview_service is None:
            return
        for sid in self._preview_ids:
            self.hub.remove_stream(sid)
        for cam in self._preview_sources:
            cam.stop()
        self._preview_service.stop()
        self._preview_service = None
        self._preview_sources = []
        self._preview_ids = []

    # -- REST discovery helpers (server/rest.py) -------------------------------------
    def camera_infos(self) -> list:
        from apollo_xarm7_core.protocol import CameraInfo

        out = []
        if self.session is not None:
            for cam_id, cam in self.session.workcell.cameras.items():
                out.append(CameraInfo(
                    camera_id=cam_id, kind="sim", label=cam_id,
                    resolution=cam.resolution, fps=int(cam.fps), live=True,
                ))
            return out
        for cam in self._preview_sources:
            out.append(CameraInfo(
                camera_id=cam.camera_id, kind="sim", label=cam.camera_id,
                resolution=cam.resolution, fps=int(cam.fps), live=True,
            ))
        return out

    def scene_infos(self, kind: str) -> list:
        from apollo_xarm7_core.protocol import SceneInfo

        try:
            from apollo_xarm7_sim import REGISTRY
        except ImportError:
            return []
        out = []
        for meta in REGISTRY.list():
            if kind not in meta.suitable_for:
                continue
            out.append(SceneInfo(
                scene_id=meta.id, label=meta.description, num_arms=meta.n_arms,
                rail_flags=[meta.rail[a] for a in meta.arm_ids],
                cameras=list(meta.cameras), kind=kind,  # type: ignore[arg-type]
            ))
        return out

    def workcell_status(self):
        from apollo_xarm7_core.protocol import ArmStatusInfo, WorkcellStatus

        available = [k for k in ("hardware", "sim") if k in self.cfg.workcells]
        kind = self.session.spec.kind if self.session else (
            "sim" if "sim" in available else (available[0] if available else "sim")
        )
        arms: list[ArmStatusInfo] = []
        scene = None
        connected = False
        states = {}
        if self.session is not None:
            scene = self.session.workcell.scene
            connected = True
            states = self.session.workcell.states()
        elif self._preview_service is not None:
            scene = self._preview_scene
        if scene is not None:
            for arm_id in scene.meta.arm_ids:
                limits = self._joint_limits(scene, arm_id)
                arms.append(ArmStatusInfo(
                    arm_id=arm_id, ip=None, connected=connected,
                    has_rail=scene.meta.rail[arm_id], gripper="xarm",
                    gripper_force_capable=False,
                    error_code=states[arm_id].error_code if arm_id in states else 0,
                    joint_limits=limits,
                ))
        return WorkcellStatus(
            kind=kind, available_kinds=available, arms=arms,
            cameras=self.camera_infos(), policies_available=False,
        )

    @staticmethod
    def _joint_limits(scene, arm_id: str) -> list[tuple[float, float]]:
        limits = []
        for i in range(1, 8):
            rng = scene.model.joint(f"{arm_id}_joint{i}").range
            limits.append((float(rng[0]), float(rng[1])))
        if scene.meta.rail[arm_id]:
            limits.append((0.0, 0.65))
        return limits


__all__ = ["ActiveSession", "SessionManager"]
