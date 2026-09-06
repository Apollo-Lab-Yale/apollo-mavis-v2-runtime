"""SessionManager — one session at a time (04-runtime §5).

IDLE -> BRINGUP -> START_FROM -> RUNNING -> TEARDOWN -> IDLE (+ FAULT).
``create()`` returns after BRINGUP; START_FROM progress rides telemetry.
Phase-05 implements teleop over the sim workcell (collect / dagger / inference
followed in phases 07/08); phase-09c adds the HARDWARE teleop session
(:meth:`SessionManager._bringup_hardware`, 04-runtime §5): the refusal matrix
(:meth:`SessionManager._validate_hardware` - rail homed, monitor sample,
no latched error, no homing in flight, teleop only, EVERY configured arm -
phase-09d), the read-only monitor hand-over (``pause()`` + ``join``; the
``hardware_session_active`` predicate is true from the first line of
``create()`` so the monitor's supervisor never reconnects mid-bring-up), a
session ``WorkcellConfig`` (``cameras: []`` - the preview cameras are ADOPTED
via ``hub.set_fps`` and never re-opened), a speed-scaled driver factory
(``SessionSpec.speed_scale``, D2), ``HardwareWorkcell.bring_up`` progress into
``SessionTelemetry.bringup``, a FRESH gate twin with an unconditional
``SafetyGate``, and the mirror-image teardown (drivers hand the arms back
stopped + braked, D6; camera fps restored; monitor resumed). Rail homing is
NOT part of bring-up: a session is refused while an arm's track is unhomed and
the operator homes it from the Hardware tab (``home_rail`` maintenance op,
twin-gated).

Phase-09d: the connect sequence is factored into
:meth:`SessionManager.connect_hardware_rig` (-> :class:`HardwareRig`) so the
rail-homing maintenance job (``devices/rail_homing.py``) reuses it to connect
ONE arm with its track unhomed at speed scale 0.1 while the other arm is
frozen in the gate twin at its last monitor sample (09c D1 - now used by the
maintenance motion only: teleop sessions always include both arms).
``start_from=profile:<id>`` on hardware plans the motion with the gate twin
INSIDE bring-up (``twin.plan`` -> the same ``execute_plan`` path as sim); a
plan failure tears the session down (409 "profile motion not collision-free").

FAULT / RECOVERING (phase-09b; 04-runtime §15): the control loop consumes the
workcell's driver events every tick and reports the aggregate per-arm state
through ``ControlLoop.on_fault_state`` -> :meth:`SessionManager._on_arm_fault_state`,
which moves ``session.state`` RUNNING -> FAULT (an arm stopped by a driver
fault) -> RECOVERING (re-seeded, waiting for the operator to release every
input) -> RUNNING. Nothing recovers without an operator click:
:meth:`SessionManager.session_recovery` is the session path of
``POST /api/hardware/arms/{arm_id}/maintenance`` (``request_recovery`` on the
hardware workcell, outcome awaited via ``recovery_result``).
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from apollo_mavis_v2_core import (
    ArmConfig,
    Command,
    ProfileNotFoundError,
    ProfileStore,
    WorkcellConfig,
)
from apollo_mavis_v2_core.protocol import (
    ArmBringupTelemetry,
    ArmMaintenanceResult,
    SessionInfo,
    SessionSpec,
)

from ..config import RuntimeConfig
from ..control.loop import DEFAULT_ACTIVE_ARM, ControlLoop, controller_error_title
from ..control.pose_filter import PoseFilterConfig
from ..control.tracker_teleop import TrackerTeleop
from ..devices.hardware_monitor import CONNECTED_STATUSES, MONITOR_JOIN_TIMEOUT_S
from ..devices.tracker import TrackerSettings
from ..errors import MaintenanceUnavailableError, SessionError, SessionNotFoundError
from ..safety.gate import NullGate, SafetyGate
from ..safety.supervisor import SafetySupervisor
from ..safety.watchdog import ArmReportWatchdog, InputWatchdog
from ..streams.hub import VideoHub
from .hardware import (
    RailFlipWorkcell,
    RailHoldWorkcell,
    SessionStateProvider,
    apply_executor_caps,
    arm_label,
    bringup_rows,
    executor_caps_for,
    frozen_state,
    scale_control_config,
    scale_driver_config,
)
from .types import SessionState

if TYPE_CHECKING:
    from ..bus import RuntimeBus
    from ..devices.hardware_monitor import HardwareStateMonitor
    from ..devices.hardware_probe import HardwareProbe
    from ..streams.twin_overlay import TwinOverlayRenderer

logger = logging.getLogger(__name__)

RECOVERY_WAIT_S = 10.0  # session path: wait this long for the driver's RecoveryResult
RECOVERY_POLL_S = 0.02


def _microphone_overrides(wc: WorkcellConfig, scene_id: str | None = None):
    """``SceneOverrides`` carrying ``ArmConfig.microphone`` per arm (03-sim §4):
    every build of a workcell's scene / digital twin passes it so the Perception
    Arm's microphone body exists exactly when the config says so.

    With ``scene_id`` the mapping is restricted to arms the scene actually has:
    the SIM workcell's arm ids are placeholders until a session picks (and
    validates against) a scene, so the pre-session preview build must not fail
    on an id the scene lacks (``SceneArmMismatchError``)."""
    from apollo_mavis_v2_sim import REGISTRY, SceneOverrides

    mics = {a.id: bool(a.microphone) for a in wc.arms}
    if scene_id is not None:
        known = {a.id for a in REGISTRY.descriptor(scene_id).arms}
        mics = {k: v for k, v in mics.items() if k in known}
    return SceneOverrides(microphones=mics)


def _servo_faithful_scene(scene_id: str, overrides=None):
    """Session workcell scene with hardware-grade servo fidelity.

    Mirrors ``guardrail_check._build_real_robot_scene``: the menagerie
    position actuators sag 1-2 cm under gravity and the rail spring-servo
    lags, while a real xArm7 + linear track hold commanded positions stiffly.
    Compile-time spec edits: gravity compensation on every body + a 40x
    stiffer rail servo. Twin/IK/previews keep the stock scene.
    """
    from apollo_mavis_v2_sim import REGISTRY, Addressing, BuiltScene

    scene = REGISTRY.build(scene_id, overrides)
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


def _bringup_step(status) -> str:
    """Stage name of a failed ``ArmBringupStatus`` (core ``BringupError`` prefixes
    its message with ``[step]``; fall back to the status fields)."""
    error = str(getattr(status, "error", "") or "")
    if error.startswith("["):
        return error[1 : error.index("]")] if "]" in error else "connect"
    if getattr(status, "network", "ok") == "failed":
        return "network"
    if getattr(status, "rail", "") in ("unhomed", "error"):
        return "rail"
    if getattr(status, "gripper", "") == "error":
        return "gripper"
    if "report" in error:
        return "report"
    return "connect"


def _manipulation_first(ids):
    """Status-row order (``GET /api/workcell``): the Manipulation Arm
    (``grip`` = :data:`DEFAULT_ACTIVE_ARM`, the default teleop arm) first, then
    the scene / config order (stable sort)."""
    return sorted(ids, key=lambda a: a != DEFAULT_ACTIVE_ARM)


def _gripper_arms(scene, arm_ids: list[str]) -> list[str]:
    """Session arms that carry a gripper (the scene is the truth in sim;
    camera-only arms have no gripper actuator)."""
    return [a for a in arm_ids if scene.addressing[a].has_gripper]


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
    recorder_thread: object | None = None  # RecorderThread (collect/dagger)
    policy_session: object | None = None  # DaggerSession | InferenceSession (phase-08)
    streams: list[str] = field(default_factory=list)  # video ids
    sources: list[object] = field(default_factory=list)  # started FrameSources
    start_from_progress: float | None = None
    fault_detail: str = ""
    # phase-09c (hardware): preview cameras adopted at session fps (NOT in ``streams``:
    # teardown restores their fps instead of removing them), arms frozen in the gate
    # twin at their last monitor sample (D1), and the unwrapped HardwareWorkcell when
    # ``workcell`` is the rail_flip adapter (the overlay reads track coordinates).
    adopted_streams: list[str] = field(default_factory=list)
    frozen_arms: list[str] = field(default_factory=list)
    inner_workcell: object | None = None
    # phase-09d (hardware): ``start_from=profile`` waypoints planned INSIDE bring-up
    # (``(waypoints, grippers)``); the start_from worker executes them instead of planning
    planned_start: tuple[dict, dict] | None = None


@dataclass
class HardwareRig:
    """Everything :meth:`SessionManager.connect_hardware_rig` builds (phase-09d): the
    connected drivers behind their adapters, the FRESH gate twin + unconditional
    ``SafetyGate``, the supervisor and a NOT yet started ``ControlLoop``. Shared by
    the hardware session bring-up and the rail-homing maintenance job."""

    arms: list[str]
    session_cfg: WorkcellConfig
    workcell: object  # what the loop drives (RailFlip / RailHold adapters applied)
    inner: object  # the unwrapped HardwareWorkcell (track rail convention)
    twin: object
    gate: SafetyGate
    supervisor: SafetySupervisor
    loop: ControlLoop
    frozen: dict  # arm_id -> ArmState posed once in the twin (09c D1)
    speed_scale: float
    executor_caps: Any = None  # ExecutorCaps applied to the loop (phase-09d; None = host slew)


@dataclass
class _BringupProgress:
    """Hardware bring-up in flight (phase-09c, D5): ``GET /api/session`` answers
    ``state: bringup`` and telemetry carries one row per (arm, step)."""

    session_id: str
    spec: SessionSpec | None  # None: the rail-homing job's connect (rows not exposed)
    rows: list[ArmBringupTelemetry] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def set(self, arm_id: str, step: str, status: str, detail: str = "") -> None:
        self.update([ArmBringupTelemetry(arm_id=arm_id, step=step, status=status, detail=detail)])

    def update(self, rows: list[ArmBringupTelemetry]) -> None:
        with self.lock:
            for new in rows:
                for i, old in enumerate(self.rows):
                    if old.arm_id == new.arm_id and old.step == new.step:
                        self.rows[i] = new
                        break
                else:
                    self.rows.append(new)

    def snapshot(self) -> list[ArmBringupTelemetry]:
        with self.lock:
            return list(self.rows)


class SessionManager:
    """Owns the singleton session + the pre-session camera previews (sim scene
    renders AND the hardware workcell's cameras, phase-11)."""

    def __init__(
        self,
        cfg: RuntimeConfig,
        bus: RuntimeBus,
        hub: VideoHub,
        profile_store: ProfileStore,
        epoch: str,
        tracker_settings: TrackerSettings | None = None,  # Runtime-owned live settings
        hardware_probe: HardwareProbe | None = None,  # Runtime-owned reachability snapshot
        hardware_monitor: HardwareStateMonitor | None = None,  # Runtime-owned read-only monitor
    ) -> None:
        self.cfg = cfg
        self.bus = bus
        self.hub = hub
        self.profile_store = profile_store
        self.epoch = epoch
        self.tracker_settings = tracker_settings or TrackerSettings.from_config(cfg.tracker)
        self.hardware_probe = hardware_probe
        self.hardware_monitor = hardware_monitor
        # Twin alignment overlays (phase-09a): Runtime-owned, assigned after construction
        # (the renderer reads this manager's hardware camera frames).
        self.twin_overlay: TwinOverlayRenderer | None = None
        self.session: ActiveSession | None = None
        self._lock = threading.Lock()
        self._creating = False  # create() is validating / bringing a session up (under _lock)
        self._creating_kind: str | None = None  # spec.kind of the create() in flight
        self._bringup: _BringupProgress | None = None  # hardware bring-up rows (phase-09c)
        # phase-09d: the RailHomingService registers itself here (``active_arm``) so
        # create() refuses ANY session kind while a rail-homing job / request is live
        self.maintenance_guard: Any = None
        # phase-09c test seams: ``workcell_factory(session_cfg, driver_factory)`` replaces
        # the hardware package's HardwareWorkcell; ``driver_api_factory`` is handed to
        # every XArmDriver (a FakeXArmAPI in tests, the real SDK when None).
        self.workcell_factory: Callable[[WorkcellConfig, Callable], object] | None = None
        self.driver_api_factory: Callable[..., object] | None = None
        self._preview_service = None
        self._preview_scene = None
        self._preview_sources: list[object] = []
        self._preview_ids: list[str] = []
        # Hardware camera previews (phase-11): camera id -> started CameraInterface;
        # cameras that failed to open are listed with live=False and retried on the
        # next start_previews() (after every session). Test seam: camera_factory.
        self._hw_cameras: dict[str, object] = {}
        self._hw_camera_errors: dict[str, str] = {}
        self.camera_factory: Callable[[object], object] | None = None
        self._twin_scenes: dict[str, object] = {}  # digital-twin scene cache (status rows)

    # -- info ------------------------------------------------------------------
    @property
    def state(self) -> SessionState:
        if self.session is not None:
            return self.session.state
        if self._bringup is not None:
            return SessionState.BRINGUP  # a hardware bring-up is in flight (phase-09c)
        return SessionState.IDLE

    @property
    def session_active(self) -> bool:
        """A session exists OR :meth:`create` is bringing one up. ``session`` is
        assigned only after bringup returns (seconds of MuJoCo/bringup under
        ``_lock``), so a guard reading it alone — the tracker calibration's
        "stop the session first" (13-tracker §4) — could pass while a session is
        coming up; ``_creating`` is raised under ``_lock`` before validation."""
        return self._creating or self.session is not None

    @property
    def hardware_session_active(self) -> bool:
        """A HARDWARE session exists or is being brought up (phase-09c): the
        hand-over predicate of the read-only monitor, the probe and the overlay.
        ``_creating_kind`` is recorded in :meth:`create` once the refusal matrix
        (:meth:`_validate_hardware`) has passed and right before
        :meth:`_bringup_hardware` - i.e. before the first side effect (its
        ``monitor.pause()``) - so the monitor's 0.5 s supervisor never reconnects
        the control boxes in the middle of a hardware bring-up (its ``pause()``
        alone would be undone), while a request that is REFUSED never flips the
        predicate: a supervisor round landing inside the validation window would
        otherwise disconnect every arm monitor, make the validation itself fail
        with a spurious "monitor paused" 409 and leave the monitors down until the
        next round."""
        kind = self._creating_kind
        session = self.session
        if kind is None and session is not None:
            kind = session.spec.kind
        return self.session_active and kind == "hardware"

    def info(self) -> SessionInfo:
        s = self.session
        if s is None:
            bp = self._bringup
            if bp is not None and bp.spec is not None:  # hardware bring-up in flight (D5)
                return SessionInfo(
                    session_id=bp.session_id,
                    epoch=self.epoch,
                    mode=bp.spec.mode,
                    arms=list(bp.spec.arms),
                    streams=[],
                    state=SessionState.BRINGUP.value,
                    kind=bp.spec.kind,
                    speed_scale=bp.spec.speed_scale,
                )
            raise SessionNotFoundError("no active session")
        return SessionInfo(
            session_id=s.session_id,
            epoch=self.epoch,
            mode=s.spec.mode,
            arms=list(s.spec.arms),
            streams=list(s.streams),
            state=s.state.value,
            kind=s.spec.kind,
            speed_scale=s.spec.speed_scale,
        )

    def bringup_telemetry(self) -> list[ArmBringupTelemetry] | None:
        """``SessionTelemetry.bringup`` (phase-09c): the hardware bring-up rows
        while one is in flight / until the session is RUNNING; ``None`` otherwise."""
        bp = self._bringup
        return bp.snapshot() if bp is not None else None

    def _tracker_provider(self) -> TrackerTeleop:
        """Per-session clutch/anchor state over the process-wide tracker slot
        (13-tracker §4); built for every session so ``tracker_settings`` and
        the clutch behave uniformly (no samples with backend ``none``)."""
        return TrackerTeleop(
            self.bus.tracker,
            self.tracker_settings,
            stale_s=self.cfg.tracker.stale_s,
            leash_pos_m=self.cfg.control.leash.pos_m,
            leash_rot_rad=self.cfg.control.leash.rot_rad,
            filter_cfg=PoseFilterConfig(**self.cfg.tracker.filter.model_dump()),
        )

    # -- validation ---------------------------------------------------------------
    def _validate(self, spec: SessionSpec) -> WorkcellConfig:
        if self.session is not None:
            raise SessionError("a session already exists")
        if spec.kind != "hardware" and spec.mode in ("dagger", "inference"):
            from ..dagger.registry import resolve_policy

            resolve_policy(self.cfg.checkpoints_root, spec.mode, spec.policy)  # 409 early
        wc = self.cfg.workcell_config(spec.kind)
        if wc is None:
            raise SessionError(f"no {spec.kind!r} workcell config available")
        if spec.kind == "hardware":
            return wc  # the hardware refusal matrix follows in _validate_hardware
        if not spec.arms:
            raise SessionError("session needs at least one arm")
        scene_id = spec.sim_scene or wc.sim_scene
        if not scene_id:
            raise SessionError("sim session needs a sim_scene")
        from apollo_mavis_v2_sim import REGISTRY, SceneNotFoundError

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
            self._creating = True  # session_active() is True from here on
            try:
                # phase-09d: a rail-homing job (or a home_rail request being evaluated)
                # excludes EVERY session kind - contract §3 REST "409 rail homing in
                # progress" - not only the hardware refusal matrix's maintenance_busy
                guard = self.maintenance_guard
                busy_arm = guard.active_arm if guard is not None else None
                if busy_arm is not None:
                    raise SessionError(
                        f"rail homing in progress on the {arm_label(busy_arm)} - wait for it "
                        "to finish"
                    )
                wc = self._validate(spec)
                if spec.kind == "hardware":
                    twin_scene, samples = self._validate_hardware(spec, wc)
                    # hardware_session_active from here on (monitor hand-over): after the
                    # refusal matrix, before the first side effect (monitor.pause())
                    self._creating_kind = spec.kind
                    session = self._bringup_hardware(spec, wc, twin_scene, samples)
                else:
                    session = self._bringup_sim(spec, wc)
                self.session = session
            finally:
                self._creating = False
                self._creating_kind = None
        threading.Thread(
            target=self._start_from_worker, args=(session,), name="start-from", daemon=True
        ).start()
        return self.info()

    def _bringup_sim(self, spec: SessionSpec, wc: WorkcellConfig) -> ActiveSession:
        from apollo_mavis_v2_sim import (
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
        overrides = _microphone_overrides(session_cfg)
        scene = _servo_faithful_scene(scene_id, overrides)
        workcell = SimWorkcell(scene, session_cfg, render_service=rs)
        workcell.start()

        # Twin: always built for planning (goto / start_from); it is the GATE
        # twin only under safety_debug (11-safety §5).
        twin = DigitalTwin(
            REGISTRY.build(scene_id, overrides),
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
            IKParams(
                min_distance_m=safety.geom_inflation_m + 0.002,
                lock_rail=not self.cfg.control.rail_in_ik,  # 04-runtime §6 "Rail"
            ),
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
        session_id = uuid.uuid4().hex
        recorder_thread = None
        policy_session = None
        try:
            if spec.mode == "collect":
                recorder_thread = self._build_collect_recorder(
                    spec, session_cfg, workcell, scene, session_id
                )
            if spec.mode in ("dagger", "inference"):
                loop, recorder_thread, policy_session = self._build_policy_stack(
                    spec, session_cfg, workcell, scene, session_id,
                    ik, kin, twin, supervisor,
                )
            else:
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
                    recorder=recorder_thread,
                    gripper_arms=_gripper_arms(scene, spec.arms),
                    tracker=self._tracker_provider(),
                )
        except Exception:
            workcell.stop()
            rs.stop()
            self.start_previews()
            raise
        loop.start()
        if recorder_thread is not None:
            recorder_thread.start()
        if policy_session is not None:
            policy_session.start()

        # Video: session cameras at session fps + reserved "sim"/"twin".
        session = ActiveSession(
            session_id=session_id,
            spec=spec,
            state=SessionState.BRINGUP,
            workcell=workcell,
            loop=loop,
            supervisor=supervisor,
            twin=twin,
            render_service=rs,
            recorder_thread=recorder_thread,
            policy_session=policy_session,
        )
        self.attach_fault_state(session)
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

    # -- hardware session (phase-09c/09d; 04-runtime §5 BRINGUP, §13.1 refusal matrix) -----
    def _validate_hardware(self, spec: SessionSpec, wc: WorkcellConfig) -> tuple[str, dict]:
        """The hardware refusal matrix (409 via :class:`SessionError`), evaluated
        BEFORE anything is touched; returns ``(twin_scene, monitor samples)``.

        teleop only (phase-09c) -> at least one arm, every arm in the hardware
        config AND in the twin scene, and - phase-09d - EVERY configured arm in
        the session ("hardware sessions include every configured arm") -> a
        resolvable ``digital_twin_scene`` -> no rail homing in flight on ANY arm
        (monitor op or rail-homing job) -> per arm: the read-only monitor
        connected with a sample (the twin cannot be posed otherwise), the
        control box reachable, no latched controller error ("clear errors
        first"), a linear track where the twin has one, and that track homed +
        enabled ("rail not homed" - the operator homes it from the Hardware tab)
        -> ``start_from`` profile covers the arms.
        """
        self._require_armed()
        if spec.mode != "teleop":
            raise SessionError("hardware sessions support teleop only (phase-09c)")
        if not spec.arms:
            raise SessionError("session needs at least one arm")
        ids = [a.id for a in wc.arms]
        missing = [a for a in spec.arms if a not in ids]
        if missing:
            raise SessionError(f"arms {missing} not in the hardware workcell (has {ids})")
        if sorted(spec.arms) != sorted(ids):
            absent = [a for a in ids if a not in spec.arms]
            names = ", ".join(arm_label(a) for a in ids)
            raise SessionError(
                f"hardware sessions include every configured arm ({names}) - "
                f"missing {absent} (phase-09d: both arms are always part of the session)"
            )
        twin_scene = spec.digital_twin_scene or wc.digital_twin_scene
        if not twin_scene:
            raise SessionError("hardware session needs a digital_twin_scene")
        try:
            from apollo_mavis_v2_sim import REGISTRY, SceneNotFoundError
        except ImportError as e:
            raise SessionError(f"hardware sessions need the [sim] extra for the twin: {e}") from e
        try:
            meta = REGISTRY.meta(twin_scene)
        except SceneNotFoundError as e:
            raise SessionError(str(e)) from e
        missing = [a for a in spec.arms if a not in meta.arm_ids]
        if missing:
            raise SessionError(
                f"arms {missing} not in scene {twin_scene!r} (has {list(meta.arm_ids)})"
            )
        monitor = self.hardware_monitor
        if monitor is None:
            raise SessionError(
                "no read-only hardware monitor - the digital twin cannot be posed for the gate"
            )
        for arm in wc.arms:  # a homing in flight on ANY arm (the carriage is moving)
            if monitor.maintenance_busy(arm.id):
                raise SessionError(
                    f"rail homing in progress on the {arm_label(arm.id)} - wait for it to finish"
                )
        samples = monitor.snapshot()
        probe = self.hardware_probe
        by_id = {a.id: a for a in wc.arms}
        for arm_id in spec.arms:
            name = arm_label(arm_id)
            status, detail = monitor.status_of(arm_id)
            sample = samples.get(arm_id)
            if status not in CONNECTED_STATUSES or sample is None:
                why = f"monitor {status}" + (f": {detail}" if detail else "")
                raise SessionError(
                    f"{name}: no monitor sample ({why}) - the read-only monitor must be "
                    "connected before a hardware session (the digital twin cannot be posed)"
                )
            if probe is not None:
                reach = probe.reachable(arm_id)
                if reach in ("refused", "unreachable"):
                    raise SessionError(
                        f"{name}: control box {by_id[arm_id].ip} is {reach} - power it on / "
                        "check the network first"
                    )
            code = int(getattr(sample, "error_code", 0) or 0)
            if code:
                raise SessionError(
                    f"{name}: controller error {code} is latched - clear errors first"
                )
            rail_present = bool(getattr(sample, "rail_present", False))
            if meta.rail.get(arm_id, False) and not rail_present:
                raise SessionError(
                    f"{name}: the digital twin {twin_scene!r} expects a linear track but the "
                    "monitor found none"
                )
            if rail_present and not (
                getattr(sample, "rail_homed", False) and getattr(sample, "rail_enabled", False)
            ):
                raise SessionError(
                    f"{name}: rail not homed - home it from the Hardware tab (Home rail) "
                    "before starting a session (carriage position unknown)"
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
        return twin_scene, samples

    def _hardware_driver_factory(self, scale: float, rail_homing: str | None = None) -> Callable:
        """``XArmDriverConfig -> XArmDriver`` with the D2 speed scale applied to
        the driver-side caps (``servo.max_joint_vel`` / ``max_cart_step_m`` /
        ``rail_speed_mm_s``); ``driver_api_factory`` (tests) is forwarded.
        ``rail_homing="allow_unhomed"`` (phase-09d, the rail-homing job ONLY) lets
        the driver connect with an unhomed track (position unknown); normal
        sessions keep the config default ``require_homed``."""
        api_factory = self.driver_api_factory

        def factory(driver_cfg):
            from apollo_mavis_v2_hardware import XArmDriver  # [hardware] extra

            scaled = scale_driver_config(driver_cfg, scale)
            if rail_homing is not None:
                scaled = scaled.model_copy(update={"rail_homing": rail_homing})
            if api_factory is None:
                return XArmDriver(scaled)
            return XArmDriver(scaled, api_factory=api_factory)

        return factory

    def _default_workcell_factory(self, session_cfg: WorkcellConfig, driver_factory: Callable):
        """The hardware package's ``HardwareWorkcell`` over the session config;
        ``netsetup=None``: NIC matching would run nmcli mutations inside the POST,
        reachability comes from the probe instead (the network stage reports
        ``ok`` + a warning)."""
        try:
            from apollo_mavis_v2_hardware import HardwareWorkcell  # [hardware] extra
        except ImportError as e:
            raise SessionError(
                f"hardware package not importable ({type(e).__name__}: {e}); install the "
                "[hardware] extra"
            ) from e
        return HardwareWorkcell(session_cfg, driver_factory=driver_factory, netsetup=None)

    def _on_bringup_status(self, status) -> None:
        """``HardwareWorkcell.bring_up`` status_cb (bring-up threads) -> telemetry rows."""
        bp = self._bringup
        if bp is not None:
            try:
                bp.update(bringup_rows(status))
            except Exception:  # noqa: BLE001 - telemetry must never break bring-up
                logger.exception("bring-up status row failed")

    def _require_armed(self) -> None:
        """Arming switch (2026-09-05, after a test process reached a real control box):
        the REAL xArm drivers are connected only when ``hardware_session.armed`` is
        true. Test seams (``workcell_factory``) never reach the SDK and bypass it."""
        if self.workcell_factory is None and not self.cfg.hardware_session.armed:
            raise SessionError(
                "hardware not armed: set hardware_session.armed: true in the lab config before "
                "connecting the real arms (the repo default is false so tests and dev instances "
                "never enable a control box)"
            )

    def connect_hardware_rig(
        self,
        *,
        arms: list[str],
        wc: WorkcellConfig,
        twin_scene: str,
        samples: dict,
        speed_scale: float,
        progress: _BringupProgress,
        bus=None,
        tracker: bool = True,
        allow_unhomed_rail: bool = False,
        rail_hold: bool = False,
        status_cb: Callable | None = None,
    ) -> HardwareRig:
        """Connect ``arms`` of the hardware workcell and build the gated control
        stack around them (04-runtime §5 steps 2-11; the loop is NOT started).

        Shared by :meth:`_bringup_hardware` (every configured arm, the session's
        ``speed_scale``) and the phase-09d rail-homing job (ONE arm,
        ``speed_scale`` 0.1, ``allow_unhomed_rail`` + ``rail_hold``: the driver
        connects with the track unhomed and reports ``rail: unhomed``, the
        ``RailHoldWorkcell`` adapter shows the twin the configured
        ``rail_fallback_m`` instead of the driver's 0.0 placeholder and never
        commands the carriage). Twin arms NOT in ``arms`` are frozen at their
        last monitor sample (09c D1). ``bus`` defaults to the runtime bus (the
        job passes a private one so no WS input can reach its loop);
        ``tracker=False`` builds the loop without the tracker provider. The
        monitor is paused + joined here; on ANY failure the workcell is stopped
        again and the exception propagates - the caller resumes the monitor.
        """
        self._require_armed()
        from apollo_mavis_v2_sim import (
            REGISTRY,
            DigitalTwin,
            IKParams,
            MinkIKSolver,
            SceneOverrides,
            default_collision_pairs,
        )

        from ..control.fk import SceneKinematics
        from ..streams.twin_overlay import base_pose_overrides

        hs = self.cfg.hardware_session
        monitor = self.hardware_monitor
        assert monitor is not None  # _validate_hardware / the job's preflight
        safety = wc.safety
        scale = float(speed_scale)
        meta = REGISTRY.meta(twin_scene)
        workcell = None
        try:
            # 1. monitor hand-over (rule 3): release every box, then wait for the poll
            #    threads to really exit (a disconnect() may return mid-SDK-call).
            for a in arms:
                progress.set(a, "monitor", "pending", "releasing the read-only monitor")
            monitor.pause()
            alive = monitor.join(MONITOR_JOIN_TIMEOUT_S)
            if alive:
                names = ", ".join(arm_label(a) for a in alive)
                raise SessionError(
                    f"read-only monitor of the {names} is still inside the SDK after "
                    f"{MONITOR_JOIN_TIMEOUT_S:g} s - retry in a moment"
                )
            for a in arms:
                progress.set(a, "monitor", "ok", "read-only monitor released the control box")

            # 2. WorkcellConfig for these arms only, NO cameras (rule 4)
            session_cfg = wc.model_copy(
                update={"arms": [a for a in wc.arms if a.id in arms], "cameras": []}
            )

            # 3. HardwareWorkcell with the speed-scaled driver factory (D2), netsetup None
            factory = self.workcell_factory or self._default_workcell_factory
            workcell = factory(
                session_cfg,
                self._hardware_driver_factory(
                    scale, rail_homing="allow_unhomed" if allow_unhomed_rail else None
                ),
            )
            inner = workcell

            # 4. bring_up (never start(): one arm failing must not abort the report)
            bring_up = getattr(workcell, "bring_up", None)
            if bring_up is None:
                raise SessionError("the hardware workcell has no bring_up()")
            statuses = bring_up(
                status_cb=status_cb or self._on_bringup_status, timeout_s=hs.bringup_timeout_s
            )
            failures = []
            accepted_rail = ("ready", "none") + (("unhomed",) if allow_unhomed_rail else ())
            for arm_id in arms:
                st = statuses.get(arm_id)
                if st is None:
                    failures.append(f"{arm_label(arm_id)}: bring-up reported no status")
                    continue
                error = getattr(st, "error", None)
                rail = str(getattr(st, "rail", "unknown"))
                if error or not getattr(st, "connected", False):
                    failures.append(
                        f"{arm_label(arm_id)}: {_bringup_step(st)} - "
                        f"{error or 'not connected'}"
                    )
                elif meta.rail.get(arm_id, False) and rail not in accepted_rail:
                    # connected with a track that is not READY (e.g. the driver latched
                    # RAIL_ERROR at connect): its position is unverifiable, so the gate
                    # twin would place the carriage at a guess - refuse. ("none" on a
                    # railed twin arm is the dof mismatch caught right below.) The
                    # rail-homing job accepts "unhomed" (it homes the track itself).
                    failures.append(
                        f"{arm_label(arm_id)}: rail - linear track {rail} after connect "
                        "(carriage position unknown, the digital twin cannot gate it)"
                    )
            if failures:
                raise SessionError("hardware bring-up failed: " + "; ".join(failures))

            # 5. consistency: driver dof == twin dof, first state not stale
            states = workcell.states()
            for arm_id in arms:
                expected = 8 if meta.rail.get(arm_id, False) else 7
                dof = int(workcell.arms[arm_id].dof)
                if dof != expected:
                    raise SessionError(
                        f"{arm_label(arm_id)}: driver reports {dof} dof but the digital twin "
                        f"{twin_scene!r} has {expected} (rail detection disagrees)"
                    )
                st = states.get(arm_id)
                if st is None or st.stale:
                    raise SessionError(
                        f"{arm_label(arm_id)}: first state is stale (30003 report stream silent)"
                    )
            if hs.rail_flip:  # one rail convention for twin, IK, gate and loop
                workcell = RailFlipWorkcell(inner)
            if rail_hold:  # phase-09d: unknown carriage -> the twin sees the fallback
                workcell = RailHoldWorkcell(workcell, self.cfg.twin_overlay.rail_fallback_m)
            if workcell is not inner:
                states = workcell.states()  # the start-posture check below sees it too

            # 6. FRESH gate twin (never the cached status scene: inflation mutates the model)
            overrides = SceneOverrides(
                microphones={a.id: bool(a.microphone) for a in wc.arms if a.id in meta.arm_ids},
                base_pose={
                    k: v for k, v in base_pose_overrides(wc).items() if k in meta.arm_ids
                },
            )
            try:
                twin = DigitalTwin(
                    REGISTRY.build(twin_scene, overrides),
                    inflation_m=safety.geom_inflation_m,
                    allowed_pairs_extra=safety.allowed_pairs_extra,
                )
            except Exception as e:  # noqa: BLE001 - TwinAuditError / scene build
                raise SessionError(f"digital twin {twin_scene!r} unavailable: {e}") from e
            # 7. the gate is UNCONDITIONAL on hardware (11-safety §4; ControlLoop re-checks)
            gate = SafetyGate(twin, safety)
            pairs = default_collision_pairs(twin.scene, twin.allowed)
            for a in arms:
                progress.set(a, "gate", "ok", f"SafetyGate on twin {twin_scene!r}")

            # 8. D1: arms not connected here are frozen at their last monitor sample
            frozen = self._freeze_unselected_arms(twin, meta, arms, samples, progress)

            # 9. IK / kinematics / supervisor
            ik = MinkIKSolver(
                twin.scene,
                IKParams(
                    min_distance_m=safety.geom_inflation_m + 0.002,
                    lock_rail=not self.cfg.control.rail_in_ik,
                ),
                collision_pairs=pairs,
            )
            if frozen:
                ik.sync_passive(frozen)
            kin = SceneKinematics(twin.scene)
            supervisor = SafetySupervisor(
                gate,
                InputWatchdog(self.cfg.control.watchdog.stale_s, self.cfg.control.watchdog.ramp_s),
                twin=twin,
                report_watchdog=ArmReportWatchdog(safety.twin_staleness_s),
                warn_clearance_m=safety.warn_clearance_m,
            )
            start_report = twin.check({a: states[a].q for a in arms})
            if start_report.blocked:
                for a in arms:
                    progress.set(
                        a,
                        "gate",
                        "warning",
                        f"twin reports {start_report.pairs} within the inflation at the start "
                        "posture - the gate holds until the clearance opens",
                    )

            # 10. control loop at the scaled host-side caps; workcell_kind hardware. The
            #     PlanExecutor is additionally bounded by the connected drivers' servo
            #     caps (per-joint velocity + lever-weighted Cartesian step; phase-09d):
            #     a host step the streamer must clip would bend the physical path off
            #     the validated straight segment while the gate sees only the command.
            control_cfg = scale_control_config(self.cfg.control, scale)
            caps = executor_caps_for((inner.arms[a] for a in arms), control_cfg)
            control_cfg = apply_executor_caps(control_cfg, caps)
            if caps.source == "servo":
                logger.info(
                    "hardware loop: plan executor capped by the servo stream - slew %.5f rad/tick, "
                    "cart %.5f m/tick (host slew %.5f)",
                    caps.slew_rad_per_tick,
                    caps.cart_step_m if caps.cart_step_m is not None else float("nan"),
                    scale_control_config(self.cfg.control, scale).jog.slew_rad_per_tick,
                )
            loop = ControlLoop(
                workcell,
                control_cfg,
                bus or self.bus,
                supervisor,
                list(arms),
                ik=ik,
                kin=kin,
                planner=twin,
                profile_store=self.profile_store,
                workcell_kind="hardware",
                gripper_arms=[a.id for a in session_cfg.arms if a.gripper != "none"],
                tracker=self._tracker_provider() if tracker else None,
            )
            return HardwareRig(
                arms=list(arms),
                session_cfg=session_cfg,
                workcell=workcell,
                inner=inner,
                twin=twin,
                gate=gate,
                supervisor=supervisor,
                loop=loop,
                frozen=frozen,
                speed_scale=scale,
                executor_caps=caps,
            )
        except BaseException:
            if workcell is not None:
                try:
                    workcell.stop()  # nothing half-connected (D6 hand-back)
                except Exception:  # noqa: BLE001
                    logger.exception("hardware connect abort: workcell.stop failed")
            raise

    @staticmethod
    def stop_rig(loop, workcell) -> None:
        """Stop a :class:`HardwareRig`'s loop (senders) then its drivers
        (``HardwareWorkcell.shutdown`` sees no cameras; D6 hand-back: mode 0,
        state 4, brakes engaged, posture kept). Never raises."""
        try:
            if loop is not None:
                loop.stop()
        except Exception:  # noqa: BLE001
            logger.exception("hardware rig: loop.stop failed")
        try:
            if workcell is not None:
                workcell.stop()
        except Exception:  # noqa: BLE001
            logger.exception("hardware rig: workcell.stop failed")

    def _bringup_hardware(
        self, spec: SessionSpec, wc: WorkcellConfig, twin_scene: str, samples: dict
    ) -> ActiveSession:
        """Mirror of :meth:`_bringup_sim` for the real cell (module docstring;
        04-runtime §5): :meth:`connect_hardware_rig` for every configured arm,
        the ``start_from`` profile motion planned on the gate twin BEFORE the
        loop starts (phase-09d: a plan failure is a 409, never a silent hold),
        then the loop, the session, the camera hand-over and the overlay
        provider. Every failure path tears down what exists, restores the
        camera fps and resumes the monitor before the ``SessionError`` (409)
        leaves; nothing stays half-connected."""
        scale = float(spec.speed_scale)
        session_id = uuid.uuid4().hex
        progress = _BringupProgress(session_id, spec)
        self._bringup = progress
        labels = ", ".join(arm_label(a) for a in spec.arms)
        logger.info(
            "hardware session %s: bring-up of %s at speed scale %.2f (twin %r)",
            session_id,
            labels,
            scale,
            twin_scene,
        )
        rig: HardwareRig | None = None
        session: ActiveSession | None = None
        try:
            rig = self.connect_hardware_rig(
                arms=list(spec.arms),
                wc=wc,
                twin_scene=twin_scene,
                samples=samples,
                speed_scale=scale,
                progress=progress,
            )
            # 11. start_from=profile: plan on the gate twin NOW (measured start, frozen
            #     arms as obstacles); the worker executes the waypoints (§5.2 path)
            planned = None
            if spec.start_from.startswith("profile:"):
                planned = self._plan_profile_start(spec, rig, progress)
            rig.loop.start()
            loop, workcell = rig.loop, rig.workcell
            session = ActiveSession(
                session_id=session_id,
                spec=spec,
                state=SessionState.BRINGUP,
                workcell=workcell,
                loop=loop,
                supervisor=rig.supervisor,
                twin=rig.twin,
                render_service=None,
                frozen_arms=sorted(rig.frozen),
                inner_workcell=rig.inner,
                planned_start=planned,
            )
            self.attach_fault_state(session)

            # 12. camera hand-over (rule 4): the previews keep their UVC nodes and hub
            #     ids, only their encoder fps changes; teardown switches it back.
            for cam_id in list(self._hw_cameras):
                if self.hardware_camera(cam_id) is None:
                    continue
                self.hub.set_fps(cam_id, self.cfg.video.session_fps)
                session.adopted_streams.append(cam_id)

            # 13. overlay: the monitor is paused, so the alignment streams read the
            #     session arms from the driver (and any frozen arm from its sample).
            if self.twin_overlay is not None:
                self.twin_overlay.set_state_provider(
                    SessionStateProvider(
                        rig.inner,
                        spec.arms,
                        {a: samples[a] for a in rig.frozen if a in samples},
                        loop.gripper_arms,
                    )
                )
            for a in spec.arms:
                progress.set(a, "loop", "ok", f"control loop running at speed scale {scale:g}")
            logger.info("hardware session %s: bring-up complete (%s)", session_id, labels)
            return session
        except BaseException:
            self._abort_hardware_bringup(
                rig.loop if rig is not None else None,
                rig.workcell if rig is not None else None,
                session,
            )
            raise

    def _plan_profile_start(
        self, spec: SessionSpec, rig: HardwareRig, progress: _BringupProgress
    ) -> tuple[dict, dict]:
        """``start_from=profile:<id>`` on hardware (phase-09d): plan every arm's
        motion from its MEASURED posture to the profile posture on the gate twin
        (``twin.plan`` - RRT-Connect, frozen arms as static obstacles; the rail
        slot follows the profile when it has one, else stays) before the loop
        starts; returns ``(waypoints, grippers)`` for ``execute_plan``. A failed
        plan raises ``SessionError`` ("profile motion not collision-free") and the
        caller tears the session down."""
        from apollo_mavis_v2_core import PlanRequest

        pid = spec.start_from.split(":", 1)[1]
        profile = self.profile_store.get(pid)  # validated by _validate_hardware
        states = rig.workcell.states()
        q_start: dict[str, list[float]] = {}
        q_goal: dict[str, list[float]] = {}
        grippers: dict[str, float] = {}
        for arm_id in rig.arms:
            st = states[arm_id]
            posture = profile.arms[arm_id]
            goal = list(posture.q)
            if st.q.shape[0] > 7:  # rail slot LAST
                rail = posture.rail_pos_m
                goal.append(float(st.q[7]) if rail is None else float(rail))
            q_start[arm_id] = [float(x) for x in st.q]
            q_goal[arm_id] = goal
            if arm_id in rig.loop.gripper_arms:
                grippers[arm_id] = float(posture.gripper_open_frac)
        for a in rig.arms:
            progress.set(a, "start_from", "pending", f"planning the motion to profile {pid!r}")
        rig.twin.sync({a: states[a] for a in rig.arms})  # measured context for the planner
        result = rig.twin.plan(PlanRequest(q_start=q_start, q_goal=q_goal))
        if not result.ok:
            pair = f" ({' / '.join(result.failing_pair)})" if result.failing_pair else ""
            raise SessionError(
                f"profile motion not collision-free: {result.failure}{pair} - the digital twin "
                f"found no safe path from the measured posture to profile {pid!r}"
            )
        for a in rig.arms:
            progress.set(
                a,
                "start_from",
                "ok",
                f"{len(result.waypoints.get(a, ()))} waypoints planned to profile {pid!r}",
            )
        return result.waypoints, grippers

    def _freeze_unselected_arms(
        self, twin, meta, arms: list[str], samples: dict, progress: _BringupProgress
    ) -> dict:
        """D1: pose every twin arm NOT in ``arms`` once from its last monitor sample
        (q7 + rail position / ``rail_fallback_m``, ``rail_flip`` applied) and leave
        it there - brakes engaged, never commanded; a row says so. An arm the
        monitor never sampled keeps the scene keyframe (warning). Since phase-09d
        only the rail-homing maintenance motion connects a subset of the arms."""
        hs = self.cfg.hardware_session
        frozen: dict = {}
        for arm_id in meta.arm_ids:
            if arm_id in arms:
                continue
            name = arm_label(arm_id)
            sample = samples.get(arm_id)
            if sample is None or len(getattr(sample, "q", ())) < 7:
                progress.set(
                    arm_id,
                    "frozen",
                    "warning",
                    f"{name}: no monitor sample - the twin keeps its keyframe posture; "
                    "do not move it from Studio",
                )
                continue
            state, note = frozen_state(
                arm_id,
                sample,
                has_rail=bool(meta.rail.get(arm_id, False)),
                rail_fallback_m=self.cfg.twin_overlay.rail_fallback_m,
                rail_flip=hs.rail_flip,
            )
            twin.sync({arm_id: state})
            frozen[arm_id] = state
            detail = f"{name} frozen at last sample (monitor seq {getattr(sample, 'seq', 0)})"
            if note:
                detail += f"; {note}"
            progress.set(arm_id, "frozen", "warning" if note else "ok", detail)
            logger.info("hardware maintenance/session: %s", detail)
        return frozen

    def _abort_hardware_bringup(self, loop, workcell, session: ActiveSession | None) -> None:
        """Failure path of :meth:`_bringup_hardware`: stop what started (loop ->
        drivers; ``HardwareWorkcell.shutdown`` sees no cameras), restore the
        adopted previews' fps, drop the overlay provider, resume the monitor
        (the hand-over predicate goes false first so the supervisor agrees)."""
        self.stop_rig(loop, workcell)
        if session is not None:
            for cam_id in session.adopted_streams:
                self.hub.set_fps(cam_id, self.cfg.video.preview_fps)
        if self.twin_overlay is not None:
            self.twin_overlay.set_state_provider(None)
        self._bringup = None
        self._creating_kind = None  # hardware_session_active false before the monitor resumes
        if self.hardware_monitor is not None:
            try:
                self.hardware_monitor.resume()
            except Exception:  # noqa: BLE001
                logger.exception("hardware bring-up abort: monitor resume failed")

    def _build_collect_recorder(
        self,
        spec: SessionSpec,
        session_cfg: WorkcellConfig,
        workcell,
        scene,
        session_id: str,
        dagger_ctx: dict | None = None,  # {"run_id", "gate"} -> DaggerRecorderThread
    ):
        """Collect/DAgger recorder stack (04-runtime §10; 10-frames §5-§9).

        Heavy imports (lerobot -> torch) happen inside the recorder ctor —
        only collect/dagger sessions ever pay them. With ``dagger_ctx`` the
        schema gains the 12-dagger §4 columns and the repo id the
        ``_dagger_{run_id}`` suffix (dedicated repo; seed data never mutated).
        """
        import hashlib

        import numpy as np
        from apollo_mavis_v2_core import parse_frame

        from ..recorder.episode_recorder import LeRobotEpisodeRecorder
        from ..recorder.features import ArmMeta, build_features, build_repo_id, build_robot_type
        from ..recorder.frames import RecordingFrameConverter
        from ..recorder.kinematics import RecorderKinematics
        from ..recorder.sidecars import SidecarWriter, pose_json
        from ..recorder.thread import RecorderThread

        action_space = "delta_ee"  # canonical (10-frames §1.2); per-session later
        arms = [ArmMeta(a, bool(scene.meta.rail[a])) for a in spec.arms]
        frames = {a: spec.frames.get(a, f"arm_base:{a}") for a in spec.arms}
        kin = RecorderKinematics(scene)

        # camera:<id> recording frames need a declared camera with static T_W_C
        # (10-frames §5.1); wrist/moving cameras are rejected here (-> 409).
        camera_poses = {}
        for arm_id, ref in frames.items():
            parsed = parse_frame(ref)
            if parsed.kind != "camera":
                continue
            if parsed.ident not in workcell.cameras:
                raise SessionError(f"frames[{arm_id!r}]: unknown camera {parsed.ident!r}")
            if not kin.camera_static(parsed.ident):
                raise SessionError(
                    f"frames[{arm_id!r}]: camera {parsed.ident!r} is not static in world"
                )
            camera_poses[parsed.ident] = kin.camera_world(parsed.ident)
        converter = RecordingFrameConverter(frames, camera_poses)

        cam_res = {cid: cam.resolution for cid, cam in workcell.cameras.items()}
        features = build_features(arms, frames, cam_res, action_space)
        repo_id = build_repo_id(spec.task or "task", arms, frames, action_space)
        if dagger_ctx is not None:
            from ..dagger.recorder import dagger_features

            features = dagger_features(features, dagger_ctx["run_id"])
            repo_id = f"{repo_id}_dagger_{dagger_ctx['run_id']}"
        root = self.cfg.datasets_root / repo_id
        recorder = LeRobotEpisodeRecorder(
            self.cfg.recorder,
            features,
            root,
            repo_id,
            build_robot_type(len(arms), sim=True),
            default_task=spec.task or "",
        )

        sidecars = SidecarWriter(root)
        scene_sha = sidecars.archive_scene_xml(scene.xml)
        sidecars.write_session(
            session_id,
            spec.mode,
            spec.model_dump(mode="json"),
            {
                "kind": "sim",
                "config_sha256": hashlib.sha256(
                    session_cfg.model_dump_json().encode()
                ).hexdigest(),
                "arm_ids": list(spec.arms),
                "rail": {a.arm_id: a.has_rail for a in arms},
            },
            session_cfg.safety.model_dump(mode="json"),
        )
        states = workcell.states()
        arm_bases = {}
        for arm in arms:
            q0 = np.array(states[arm.arm_id].q, dtype=np.float64)
            if arm.has_rail:
                q0[7] = 0.0  # rail origin = base pose at zero rail travel (§2.2)
            arm_bases[arm.arm_id] = {
                "rail_origin_in_world": pose_json(kin.base_world(arm.arm_id, q0)),
                "has_rail": arm.has_rail,
            }
        initial = self.profile_store.initial_for("sim")
        profile_snapshot = None
        if spec.start_from.startswith("profile:"):
            profile_snapshot = self.profile_store.get(
                spec.start_from.split(":", 1)[1]
            ).model_dump(mode="json")
        meta_base = {
            "session_id": session_id,
            "scene_xml_sha256": scene_sha,
            "start_from": spec.start_from,
            "initial_condition_profile_id": initial.profile_id if initial else None,
            "profile_snapshot": profile_snapshot,
            "frames": dict(frames),
            "arm_bases": arm_bases,
        }
        cam_cfgs = {c.id: c for c in session_cfg.cameras}

        def extrinsics_fn(arm_states):  # runs on the recorder thread (owns kin)
            q_by_arm = {a: np.asarray(arm_states[a].q) for a in spec.arms}
            out = {}
            for cam_id in workcell.cameras:
                cfg = cam_cfgs.get(cam_id)
                out[cam_id] = {
                    "T_W_C": pose_json(kin.camera_world(cam_id, q_by_arm)),
                    "extrinsics_frame": "world" if kin.camera_static(cam_id) else None,
                    "intrinsics": (
                        cfg.intrinsics.model_dump() if cfg and cfg.intrinsics else None
                    ),
                    "calibration_file": None,  # sim: pose comes from the MJCF
                    "calibration_sha256": None,
                }
            return out

        if dagger_ctx is not None:
            from ..dagger.recorder import DaggerRecorderThread

            return DaggerRecorderThread(
                recorder,
                self.bus,
                workcell.cameras,
                arms,
                converter,
                kin,
                fps=self.cfg.recorder.fps,
                sidecars=sidecars,
                episode_meta_base=meta_base,
                extrinsics_fn=extrinsics_fn,
                gate=dagger_ctx["gate"],
                run_id=dagger_ctx["run_id"],
                dataset_root=root,
            )
        return RecorderThread(
            recorder,
            self.bus,
            workcell.cameras,
            arms,
            converter,
            kin,
            fps=self.cfg.recorder.fps,
            sidecars=sidecars,
            episode_meta_base=meta_base,
            extrinsics_fn=extrinsics_fn,
        )

    def _build_policy_stack(
        self, spec: SessionSpec, session_cfg: WorkcellConfig, workcell, scene,
        session_id: str, ik, kin, twin, supervisor,
    ):
        """DAgger/inference bringup (12-dagger §1): -> (loop, recorder, session)."""
        import shutil

        from apollo_mavis_v2_core.dagger import CheckpointInfo

        from ..dagger.gate import TakeoverGateImpl
        from ..dagger.loop import (
            DaggerSession,
            GatedPolicyExecutor,
            InferenceSession,
            make_obs_fn,
        )
        from ..dagger.policy_runner import ActionAnchor, PolicyRunner, SlewLimits
        from ..dagger.registry import resolve_policy
        from ..dagger.trainer.checkpoints import STATE_DICT, CheckpointStore, sha256_file
        from ..recorder.features import arm_action_names, arm_state_names
        from ..recorder.frames import RecordingFrameConverter
        from ..recorder.kinematics import RecorderKinematics

        dcfg = self.cfg.dagger
        resolved = resolve_policy(self.cfg.checkpoints_root, spec.mode, spec.policy)
        info = resolved.info
        frames = {a: spec.frames.get(a, f"arm_base:{a}") for a in spec.arms}
        if info.action_space != "delta_ee" or any(
            f != info.action_frame for f in frames.values()
        ):
            raise SessionError("policy/dataset frame mismatch")
        from ..dagger.policies import MLPPolicy, resolve_device

        device = resolve_device(dcfg.policy_device)
        try:
            policy = MLPPolicy.from_bundle(
                str(resolved.state_dict_path), info.version, device
            )
        except Exception as e:
            raise SessionError(f"policy load failed: {e!r}") from e
        gate = TakeoverGateImpl(list(spec.arms), dcfg.t_blend_s)
        arms_meta = [(a, bool(scene.meta.rail[a])) for a in spec.arms]
        policy_lock = threading.Lock()
        runner = PolicyRunner(
            policy,
            make_obs_fn(self.bus, arms_meta,
                        RecordingFrameConverter(frames, {}), RecorderKinematics(scene)),
            rate_hz=dcfg.policy_rate_hz,
            policy_lock=policy_lock,
        )
        anchor = ActionAnchor(ik, kin, SlewLimits(window_s=dcfg.slew_window_s),
                              action_space=info.action_space)
        common = dict(ik=ik, kin=kin, planner=twin, profile_store=self.profile_store,
                      workcell_kind="sim", gripper_arms=_gripper_arms(scene, spec.arms),
                      tracker=self._tracker_provider())
        if spec.mode == "inference":
            loop = GatedPolicyExecutor(
                workcell, self.cfg.control, self.bus, supervisor, list(spec.arms),
                gate=gate, runner=runner, anchor=anchor, arms_meta=arms_meta,
                session_mode="inference", version_label=resolved.policy_id,
                recorder=None, recorder_fps=self.cfg.recorder.fps, **common,
            )
            return loop, None, InferenceSession(loop, runner)

        # -- dagger: fresh run store; v000000 = seed (rollback target, §12) ------
        from ..dagger.client import AsyncTrainerClientImpl, TrainerConfig
        from ..dagger.reloader import PolicyReloaderImpl

        run_id = session_id[:8]
        store = CheckpointStore(self.cfg.checkpoints_root, run_id)
        v0 = store.version_dir(0)
        v0.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(resolved.state_dict_path, v0 / STATE_DICT)
        store.write_manifest(CheckpointInfo(
            run_id=run_id, version=0, path=str(v0), parent_version=None,
            trained_on_frames=info.trained_on_frames, trained_on_episodes=[],
            action_frame=info.action_frame, action_space=info.action_space,
            sanity_ok=True, mean_loss=info.mean_loss,
            sha256=sha256_file(v0 / STATE_DICT), created_wallclock_ns=time.time_ns(),
        ))
        policy.set_version(0)
        recorder_thread = self._build_collect_recorder(
            spec, session_cfg, workcell, scene, session_id,
            dagger_ctx={"run_id": run_id, "gate": gate},
        )
        state_dim = sum(len(arm_state_names(a, r)) for a, r in arms_meta)
        action_dim = sum(len(arm_action_names(a, r, "delta_ee")) for a, r in arms_meta)
        tcfg = TrainerConfig(
            run_id=run_id, checkpoints_root=str(self.cfg.checkpoints_root),
            spool_dir=str(recorder_thread.spool_dir),
            seed_bundle=str(v0 / STATE_DICT),
            port=dcfg.trainer.port, device=dcfg.trainer.device,
            cuda_visible_devices=dcfg.trainer.cuda_visible_devices,
            action_frame=info.action_frame, action_space=info.action_space,
            state_dim=state_dim, action_dim=action_dim,
            min_new_labels=dcfg.trainer.min_new_labels,
            push_period_s=dcfg.trainer.push_period_s,
            batch_size=dcfg.trainer.batch_size, lr=dcfg.trainer.lr,
        )
        client = AsyncTrainerClientImpl(tcfg, workdir=store.root)
        reloader = PolicyReloaderImpl(
            policy, store, info.action_frame, info.action_space,
            policy_lock=policy_lock, current_version=0,
            on_rollback=client.notify_rollback,
        )
        loop = GatedPolicyExecutor(
            workcell, self.cfg.control, self.bus, supervisor, list(spec.arms),
            gate=gate, runner=runner, anchor=anchor, arms_meta=arms_meta,
            session_mode="dagger", run_id=run_id, reloader=reloader,
            trainer_client=client, recorder=recorder_thread,
            recorder_fps=self.cfg.recorder.fps, **common,
        )
        recorder_thread.on_episode_saved = loop.on_episode_saved
        return loop, recorder_thread, DaggerSession(loop, runner, reloader, client)

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
                if session.state is SessionState.BRINGUP:  # a fault may already hold it
                    session.state = SessionState.RUNNING  # keep_current: no motion
                    self._bringup = None  # hardware bring-up rows shown until running
                return
            if session.state is not SessionState.BRINGUP:
                return  # faulted during bring-up: no start_from motion
            session.state = SessionState.START_FROM
            session.start_from_progress = 0.0
            if session.planned_start is not None:
                # hardware (phase-09d): planned on the gate twin inside bring-up
                waypoints, grippers = session.planned_start
                q_goal = dict(waypoints)
                result = None
            else:
                pid = session.spec.start_from.split(":", 1)[1]
                profile = self.profile_store.get(pid)
                states = session.workcell.states()
                q_start: dict[str, list[float]] = {}
                q_goal = {}
                grippers = {}
                for arm_id in session.spec.arms:
                    st = states[arm_id]
                    posture = profile.arms[arm_id]
                    goal = list(posture.q)
                    if st.q.shape[0] > 7:  # rail slot LAST
                        rail = posture.rail_pos_m
                        goal.append(float(st.q[7]) if rail is None else float(rail))
                    q_start[arm_id] = [float(x) for x in st.q]
                    q_goal[arm_id] = goal
                    if arm_id in session.loop.gripper_arms:
                        grippers[arm_id] = float(posture.gripper_open_frac)
                from apollo_mavis_v2_core import PlanRequest

                if session.supervisor.twin is None:  # plain sim: keep the plan twin fresh
                    session.twin.sync(states)
                result = session.twin.plan(PlanRequest(q_start=q_start, q_goal=q_goal))
                waypoints = result.waypoints
            if result is not None and not result.ok:
                logger.error("start_from plan failed: %s %s", result.failure,
                             result.failing_pair)
                session.fault_detail = f"start_from plan failed: {result.failure}"
                if session.state is SessionState.START_FROM:
                    session.state = SessionState.RUNNING  # arms stay held; operator decides
                session.start_from_progress = None
                self.bus.commands.submit(Command(
                    op="_plan_ready",
                    args={"arms": list(q_goal), "result": result, "detail": ""},
                    source="internal",
                ))
                return
            total = sum(len(w) for w in waypoints.values()) or 1
            self.bus.commands.submit(Command(
                op="execute_plan",
                args={"waypoints": waypoints, "gripper": grippers},
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
            if session.state is SessionState.START_FROM:  # a driver fault may own it now
                session.state = SessionState.RUNNING
                self._bringup = None
        except Exception as e:
            logger.exception("start_from worker failed")
            session.fault_detail = repr(e)
            session.state = SessionState.FAULT

    # -- driver faults: FAULT -> RECOVERING -> RUNNING (phase-09b; 04-runtime §15) -----------
    def attach_fault_state(self, session: ActiveSession) -> None:
        """Wire ``session.loop.on_fault_state`` to this session's state (also the
        seam tests use to install a hand-built session)."""
        session.loop.on_fault_state = lambda state, s=session: self._on_arm_fault_state(s, state)

    def _on_arm_fault_state(self, session: ActiveSession, state: str | None) -> None:
        """Control-loop callback (loop thread): the aggregate per-arm fault state
        changed. ``"fault"`` -> FAULT (an arm was stopped by a driver fault;
        siblings keep running in the loop), ``"recovering"`` -> RECOVERING (the
        arm was re-seeded from the measured position and waits for the operator
        to release every input), ``None`` -> back to RUNNING. TEARDOWN is never
        overridden; BRINGUP / START_FROM give way to a fault (the start_from
        worker checks before it writes RUNNING)."""
        if self.session is not session:
            return
        current = session.state
        if current is SessionState.TEARDOWN:
            return
        if state == "fault":
            if current in (
                SessionState.BRINGUP,
                SessionState.START_FROM,
                SessionState.RUNNING,
                SessionState.RECOVERING,
            ):
                session.start_from_progress = None
                session.state = SessionState.FAULT
                faults = session.loop._arm_fault_details(time.monotonic())
                stopped = session.loop.faulted_arms
                session.fault_detail = "; ".join(
                    f"{arm}: {text}" for arm, text in faults.items() if arm in stopped
                )
                logger.warning("session %s FAULT: %s", session.session_id, session.fault_detail)
        elif state == "recovering":
            if current in (
                SessionState.BRINGUP,
                SessionState.START_FROM,
                SessionState.RUNNING,
                SessionState.FAULT,
            ):
                session.start_from_progress = None
                session.state = SessionState.RECOVERING
                logger.info("session %s RECOVERING (release every input to resume)",
                            session.session_id)
        elif state is None:
            if current in (SessionState.FAULT, SessionState.RECOVERING):
                session.state = SessionState.RUNNING
                session.fault_detail = ""
                logger.info("session %s RUNNING again", session.session_id)

    def session_recovery(
        self, arm_id: str, op: str, timeout_s: float = RECOVERY_WAIT_S
    ) -> ArmMaintenanceResult:
        """Session path of ``POST /api/hardware/arms/{arm_id}/maintenance``
        (04-runtime §13.1): inside a hardware session ``clear_errors`` and
        ``recover`` both run the driver's user-initiated recovery on ITS monitor
        thread (``workcell.request_recovery(arm_id)``: clean errors -> enable ->
        servo mode -> ready -> re-seed from the MEASURED position; no motion) and
        wait <= ``timeout_s`` for the outcome via ``workcell.recovery_result``
        (the loop consumes the events themselves: FaultEvent -> ReseedEvent ->
        RecoveredEvent, or a latch FaultEvent). ``apply_backstops`` is refused
        (409) - the driver applied the volatile settings at connect. Returns
        ``path="session"``; ``before`` / ``after`` are ``None`` (no monitor
        sample while the session owns the box)."""
        session = self.session
        if session is None or session.spec.kind != "hardware":
            raise MaintenanceUnavailableError("no hardware session")
        if op == "apply_backstops":
            raise MaintenanceUnavailableError(
                "apply_backstops is not available while a hardware session owns the arm "
                "(the driver applied the safety settings at connect; end the session first)"
            )
        if op not in ("clear_errors", "recover"):
            raise ValueError(f"unknown maintenance op {op!r}")
        if arm_id not in session.spec.arms:
            raise MaintenanceUnavailableError(f"arm {arm_id!r} is not part of the session")
        workcell = session.workcell
        request = getattr(workcell, "request_recovery", None)
        result_of = getattr(workcell, "recovery_result", None)
        if request is None or result_of is None:
            raise MaintenanceUnavailableError("the session workcell has no recovery channel")
        previous = result_of(arm_id)
        seq0 = int(getattr(previous, "seq", 0) or 0) if previous is not None else 0
        try:
            request(arm_id)
        except KeyError:
            raise MaintenanceUnavailableError(f"unknown arm {arm_id!r}") from None
        except Exception as e:  # noqa: BLE001 - CommandError: driver not connected / no channel
            raise MaintenanceUnavailableError(f"{arm_id}: {e}") from e
        deadline = time.monotonic() + float(timeout_s)
        res = None
        while time.monotonic() < deadline:
            res = result_of(arm_id)
            # The driver bumps ``seq`` for its own auto recoveries too: an auto
            # sequence already running when we asked completes first, so wait for
            # the result that is ours (``user_initiated``), not merely a newer one.
            if (
                res is not None
                and int(getattr(res, "seq", 0) or 0) > seq0
                and bool(getattr(res, "user_initiated", False))
            ):
                break
            time.sleep(RECOVERY_POLL_S)
        else:
            return ArmMaintenanceResult(
                arm_id=arm_id,
                op=op,  # type: ignore[arg-type]
                path="session",
                ok=False,
                detail=(
                    f"{op} timed out after {float(timeout_s):g} s "
                    "(no recovery result from the driver)"
                ),
            )
        code = int(getattr(res, "error_code", 0) or 0)
        title = controller_error_title(code)
        if res.ok:
            detail = (
                f"recovered from {title}" if title else "re-seeded from the measured position"
            )
        else:
            reason = str(getattr(res, "detail", "") or "recovery failed")
            detail = f"{title} - {reason}" if title else reason
        return ArmMaintenanceResult(
            arm_id=arm_id,
            op=op,  # type: ignore[arg-type]
            path="session",
            ok=bool(res.ok),
            detail=detail,
        )

    # -- teardown -------------------------------------------------------------------
    def teardown(self) -> None:
        """DELETE /api/session (idempotent); also SIGTERM/fatal-error path.

        Hardware (phase-09c): after ``workcell.stop()`` (drivers hand the arms
        back stopped + braked, D6; the workcell owns no cameras) the adopted
        previews go back to ``preview_fps``, the overlay reads the monitor
        again and the read-only monitor is resumed (after ``session`` is
        cleared so its predicate agrees)."""
        with self._lock:
            session = self.session
            if session is None:
                return
            session.state = SessionState.TEARDOWN
            hardware = session.spec.kind == "hardware"
            try:
                if session.policy_session is not None:
                    session.policy_session.stop()  # trainer stop -> reloader -> runner
                if session.recorder_thread is not None:
                    session.recorder_thread.stop()  # discard + finalize (§10.4)
                session.loop.stop()
                for sid in session.streams:
                    self.hub.remove_stream(sid)
                for src in session.sources:
                    src.stop()
                if session.render_service is not None:
                    session.render_service.stop()
                session.workcell.stop()
            finally:
                if hardware:
                    for cam_id in session.adopted_streams:
                        self.hub.set_fps(cam_id, self.cfg.video.preview_fps)
                    if self.twin_overlay is not None:
                        self.twin_overlay.set_state_provider(None)
                self.session = None
                self._bringup = None
            if hardware and self.hardware_monitor is not None:
                self.hardware_monitor.resume()
        self.start_previews()

    # -- pre-session camera previews (~15 fps, 04-runtime §13.4) --------------------
    def start_previews(self) -> None:
        """Sim scene previews (one static render source + one stream per scene
        camera) and the hardware workcell's camera previews. The sim part is
        torn down by :meth:`stop_previews` when a sim session starts; hardware
        previews stay up across sim sessions (phase-11) and are stopped only by
        :meth:`stop_hardware_previews` (process exit) or a hardware session."""
        self._start_sim_previews()
        self.start_hardware_previews()

    def _start_sim_previews(self) -> None:
        if self._preview_service is not None:
            return
        wc = self.cfg.workcell_config("sim")
        if wc is None or not wc.sim_scene:
            return
        try:
            from apollo_mavis_v2_sim import REGISTRY, RenderService, SimCamera
        except ImportError:
            return
        rs = RenderService()
        rs.start()
        scene = REGISTRY.build(wc.sim_scene, _microphone_overrides(wc, wc.sim_scene))
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
        """Stop the SIM preview sources only; hardware camera previews are kept
        (no side effect while no camera is connected)."""
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

    # -- hardware camera previews (phase-11; failure-isolated) ------------------------
    def _default_camera_factory(self) -> Callable[[object], object]:
        from apollo_mavis_v2_hardware.cameras import make_camera  # [hardware] extra

        return make_camera

    def start_hardware_previews(self) -> None:
        """Try to open every hardware camera and stream it at ``preview_fps``.
        Each camera is isolated: ``CameraInitError`` / any exception (device
        absent, cv2 missing, hardware extra not installed, duplicate stream id)
        marks that camera ``live: false`` and never touches the others or the
        sim previews. A camera that failed mid-run (``failed``) is re-opened."""
        wc = self.cfg.workcell_config("hardware")
        if wc is None:
            return
        if self.session is not None and self.session.spec.kind == "hardware":
            return  # the session owns the cameras (04-runtime §13.4)
        fps = self.cfg.video.preview_fps
        for cam_cfg in wc.cameras:
            existing = self._hw_cameras.get(cam_cfg.id)
            if existing is not None:
                if not getattr(existing, "failed", False):
                    continue
                self._stop_hardware_camera(cam_cfg.id)
            cam = None
            try:
                factory = self.camera_factory or self._default_camera_factory()
                cam = factory(cam_cfg)
                cam.start()
                self.hub.add_stream(cam_cfg.id, cam, fps)
            except Exception as e:  # noqa: BLE001 - isolation: absent -> live false
                if cam is not None:
                    try:
                        cam.stop()
                    except Exception:  # noqa: BLE001
                        pass
                self._hw_camera_errors[cam_cfg.id] = f"{type(e).__name__}: {e}"
                logger.info("hardware camera %r preview unavailable: %s", cam_cfg.id, e)
                continue
            self._hw_cameras[cam_cfg.id] = cam
            self._hw_camera_errors.pop(cam_cfg.id, None)

    def _stop_hardware_camera(self, cam_id: str) -> None:
        cam = self._hw_cameras.pop(cam_id, None)
        if cam is None:
            return
        self.hub.remove_stream(cam_id)
        try:
            cam.stop()
        except Exception:  # noqa: BLE001
            logger.exception("hardware camera %r did not stop cleanly", cam_id)

    def stop_hardware_previews(self) -> None:
        for cam_id in list(self._hw_cameras):
            self._stop_hardware_camera(cam_id)

    def hardware_camera_errors(self) -> dict[str, str]:
        """Last open failure per hardware camera id (diagnostics)."""
        return dict(self._hw_camera_errors)

    def hardware_camera(self, cam_id: str):
        """The started hardware camera preview (core ``CameraInterface``) for
        ``cam_id``, or ``None`` when it is not open / has failed. The twin
        overlay (phase-09a) reads ``latest()`` from it: a UVC node cannot be
        opened twice, so the preview capture is the ONLY reader of the device."""
        cam = self._hw_cameras.get(cam_id)
        if cam is None or getattr(cam, "failed", False):
            return None
        return cam

    def hardware_camera_frame(self, cam_id: str):
        """Newest real frame of a hardware camera (``None`` when unavailable)."""
        cam = self.hardware_camera(cam_id)
        return cam.latest() if cam is not None else None

    # -- REST discovery helpers (server/rest.py) -------------------------------------
    def camera_infos(self) -> list:
        """``GET /api/cameras``: sim cameras (session or preview) + hardware cameras."""
        return self.sim_camera_infos() + self.hardware_camera_infos()

    def sim_camera_infos(self) -> list:
        from apollo_mavis_v2_core.protocol import CameraInfo

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

    def hardware_camera_infos(self) -> list:
        """Every configured hardware camera; ``live`` iff its preview opened
        (and has not failed since). The UI draws ``live: false`` as a black
        "no signal" tile without opening a WebSocket."""
        from apollo_mavis_v2_core.protocol import CameraInfo

        wc = self.cfg.workcell_config("hardware")
        if wc is None:
            return []
        out = []
        live_cams: dict[str, bool] = {}
        for cam_cfg in wc.cameras:
            live = self.hardware_camera(cam_cfg.id) is not None
            live_cams[cam_cfg.id] = live
            out.append(CameraInfo(
                camera_id=cam_cfg.id, kind=cam_cfg.kind, label=cam_cfg.id,
                resolution=tuple(cam_cfg.resolution), fps=int(cam_cfg.fps), live=live,
            ))
        # Twin alignment overlays (phase-09a): kind "twin", live iff the real camera
        # underneath is live AND the overlay is compositing (live / stale tint).
        overlay = self.twin_overlay
        if overlay is not None:
            for src in overlay.streams.values():
                out.append(CameraInfo(
                    camera_id=src.stream_id, kind="twin", label=src.label,
                    resolution=src.resolution, fps=int(overlay.cfg.fps),
                    live=bool(live_cams.get(src.camera_id)) and src.status in ("live", "stale"),
                ))
        return out

    def scene_infos(self, kind: str) -> list:
        """``GET /api/scenes``: the registry's default listing (hidden scenes
        filtered by the sim registry itself, phase-11), labelled by the scene's
        display ``title`` when it has one, else its description."""
        from apollo_mavis_v2_core.protocol import SceneInfo

        try:
            from apollo_mavis_v2_sim import REGISTRY
        except ImportError:
            return []
        out = []
        for meta in REGISTRY.list():
            if getattr(meta, "hidden", False):  # defensive: older registries list all
                continue
            if kind not in meta.suitable_for:
                continue
            out.append(SceneInfo(
                scene_id=meta.id, label=getattr(meta, "title", None) or meta.description,
                num_arms=meta.n_arms,
                rail_flags=[meta.rail[a] for a in meta.arm_ids],
                cameras=list(meta.cameras), kind=kind,  # type: ignore[arg-type]
            ))
        return out

    def workcell_status(self, kind: str | None = None):
        """``GET /api/workcell[?kind=hardware|sim]`` (phase-11).

        ``kind=None`` keeps the legacy behaviour: the session's kind, else sim;
        cameras = everything previewed. ``kind="sim"``: arms from the preview /
        session scene, sim cameras. ``kind="hardware"``: arms from the hardware
        config (``ip`` filled, ``reachable`` from the probe, ``connected`` =
        a hardware session exists), hardware cameras (``live`` = preview
        opened). ``hardware_ready`` (all configured hardware arms ``open``) is
        reported on every response."""
        from apollo_mavis_v2_core.protocol import ArmStatusInfo, WorkcellStatus

        from ..dagger.registry import scan_policies

        available = [k for k in ("hardware", "sim") if k in self.cfg.workcells]
        requested = kind
        if kind is None:
            kind = self.session.spec.kind if self.session else (
                "sim" if "sim" in available else (available[0] if available else "sim")
            )
        arms: list[ArmStatusInfo]
        if kind == "hardware":
            arms = self._hardware_arm_infos()
            cameras = self.hardware_camera_infos()
        else:
            arms = self._sim_arm_infos()
            cameras = self.camera_infos() if requested is None else self.sim_camera_infos()
        probe = self.hardware_probe
        return WorkcellStatus(
            kind=kind, available_kinds=available, arms=arms, cameras=cameras,
            policies_available=bool(scan_policies(self.cfg.checkpoints_root)),
            hardware_ready=bool(probe is not None and probe.hardware_ready),
        )

    def _sim_arm_infos(self) -> list:
        from apollo_mavis_v2_core.protocol import ArmStatusInfo

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
            for arm_id in _manipulation_first(scene.meta.arm_ids):
                limits = self._joint_limits(scene, arm_id)
                arms.append(ArmStatusInfo(
                    arm_id=arm_id, ip=None, connected=connected,
                    has_rail=scene.meta.rail[arm_id],
                    gripper="xarm" if scene.addressing[arm_id].has_gripper else "none",
                    gripper_force_capable=False,  # sim grippers are position-only
                    error_code=states[arm_id].error_code if arm_id in states else 0,
                    joint_limits=limits,
                ))
        return arms

    def _twin_scene(self, scene_id: str | None):
        """Built digital-twin scene of the hardware workcell for the status rows
        (rail flags, joint limits): a one-off build cached per id, carrying the
        hardware arms' ``microphone`` flags (``SceneOverrides.microphones``, so
        it is the same twin the overlay / gate use); ``None`` when the sim extra
        / scene is missing."""
        if not scene_id:
            return None
        if scene_id not in self._twin_scenes:
            try:
                from apollo_mavis_v2_sim import REGISTRY

                wc = self.cfg.workcell_config("hardware")
                overrides = _microphone_overrides(wc) if wc is not None else None
                self._twin_scenes[scene_id] = REGISTRY.build(scene_id, overrides)
            except Exception as e:  # noqa: BLE001 - status rows degrade gracefully
                logger.warning("digital twin scene %r unavailable for status: %r", scene_id, e)
                self._twin_scenes[scene_id] = None
        return self._twin_scenes[scene_id]

    def _hardware_arm_infos(self) -> list:
        from apollo_mavis_v2_core.protocol import ArmStatusInfo

        wc = self.cfg.workcell_config("hardware")
        if wc is None:
            return []
        connected = self.session is not None and self.session.spec.kind == "hardware"
        probe = self.hardware_probe
        monitor = self.hardware_monitor  # read-only controller state (phase-09a)
        twin = self._twin_scene(wc.digital_twin_scene)
        arms: list[ArmStatusInfo] = []
        for arm in sorted(wc.arms, key=lambda a: a.id != DEFAULT_ACTIVE_ARM):  # grip first
            in_twin = twin is not None and arm.id in twin.meta.arm_ids
            arms.append(ArmStatusInfo(
                arm_id=arm.id, ip=arm.ip, connected=connected,
                reachable=probe.reachable(arm.id) if probe is not None else "unknown",
                has_rail=twin.meta.rail[arm.id] if in_twin else arm.expect_rail == "yes",
                gripper=arm.gripper,
                gripper_force_capable=arm.gripper == "xarm_g2",
                error_code=monitor.error_code(arm.id) if monitor is not None else 0,
                joint_limits=self._joint_limits(twin, arm.id) if in_twin else [],
            ))
        return arms

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
