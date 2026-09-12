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

DATA COLLECTION on both workcells (2026-09-07; 04-runtime §10.5/§10.6; 10-frames
§11): the hardware refusal matrix admits ``mode: collect`` (DAgger / inference stay
409), :meth:`SessionManager._build_collect_recorder` records from the twin's
kinematics + the ADOPTED preview cameras (``self._hw_cameras``) with the real
D435 intrinsics into an ``EpisodeDirRecorder`` (one directory per saved episode;
LeRobot v3 is an export), ``SessionSpec.dataset`` / ``dataset_resume`` name the
repo (create-must-not-exist / resume-must-exist-and-match, legacy v3 trees and a
repo being exported are 409 - all checked BEFORE any box is touched), the
Perception Arm microphone is recorded as ``audio.wav`` inside the episode
directory, and - ``SessionSpec.return_to_start`` (DEFAULT ON, D6) - every save /
discard drives the arms back to the return profile (the ``start_from`` profile,
else the kind's initial-condition profile; 409 at POST when neither exists):
:meth:`SessionManager._return_home_worker` plans on the session's twin like
``start_from`` itself, executes through the gated ``execute_plan`` path as an
INTERRUPTIBLE plan (any held key / clutch / jog / arm switch cancels it and the
arm holds where it is) while the recorder reports ``returning``. Episode
deletion is immediate (``DatasetStore.delete_episode`` - one directory) and
allowed during the session except for the open episode.

ONLINE DAGGER (phase-14; 15-online-dagger §3, §7): ``SessionSpec.online_dagger`` on an
external dagger session. :meth:`SessionManager._check_online_dagger` runs at ``create()``
BEFORE any side effect (the session-directory rule, a trainer-capable policy node
attached), then :meth:`SessionManager._build_online_dagger` creates ``<online_dagger
root>/<s>/rollouts``, writes ``session.json`` and builds the ``OnlineDaggerCoordinator``
whose side effects (``events`` + the session file) run on its own serial worker; the
rollouts are recorded by the ``DaggerRecorderThread`` into repo id ``online_dagger/<s>``.
The runtime is the algorithm-agnostic shell: no offline-dataset check, no trainer
directory, no hyper-parameters. Every ``TakeoverGate`` event of a policy session is
published as ``events.gate`` (through the coordinator's worker, else the publisher's
queue). Return-to-start (D6) applies to dagger sessions too. No new motion path: the
coordinator owns no arm.

2026-09-08 (two operator-reported problems, not yet in the design docs): the
``start_from`` worker waits ``hardware_session.start_from_fault_grace_s`` for a
TRANSIENT post-bring-up fault to clear before it submits the pre-planned motion
(:meth:`SessionManager._await_arms_clear`) and names the arm / controller state when
the loop still refuses; ``goto_profile`` (:meth:`SessionManager.request_goto_profile`)
walks the arms to a CHOSEN saved profile through the reset-to-initial machinery.

SEQUENTIAL EXECUTION (2026-09-08 evening; 04-runtime §10.5, 11-safety §9): every
twin-planned multi-arm motion - the return phases, the per-episode return,
``start_from`` on both workcells, ``goto_profile`` - is executed ONE ARM AT A TIME in
``PlanResult.arm_order`` (:meth:`SessionManager._execute_arms`): the sequential
planner validated arm k only with the arms before it AT their goals and the arms
after it AT their starts, so the paths are collision-free in that order and in no
other. On the real cell at 23:16:52 that day a ``reset_to_initial`` submitted both
arms' waypoints as ONE ``execute_plan``, the executor moved them simultaneously
through combinations nobody had checked and the gate held the return at 5.2 mm
(``grip_right_finger`` / ``view_link3``) until the budget. The gripper targets ride
the LAST arm that moves; any refusal / cancel / fault / timeout stops the sequence
(the remaining arms never start and the report names the arm); the loop's gate-held
abort (``hardware_session.plan_gate_hold_s``) turns a held motion into an immediate
"held by the safety gate (<pair>)" report. The hand-over criterion is MEASURED arrival
(2026-09-08 review): the executor retiring an arm's waypoints only means the COMMAND
reached the goal, and a carriage follows its latest-wins targets at the track's own
speed, so the manager waits until ``workcell.states()[arm].q`` is within
``PLAN_ARRIVAL_TOL_RAD`` / ``PLAN_ARRIVAL_TOL_RAIL_M`` of the last waypoint
(:meth:`SessionManager._await_arrival`) before the next arm is submitted - and reports
``stalled`` when it never gets there. The loop refuses a plan carrying more than one
arm, so the incident can not recur through ``execute_plan`` either.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from apollo_mavis_v2_core import (
    ArmConfig,
    ArmPosture,
    Command,
    ProfileNotFoundError,
    ProfileStore,
    StateProfile,
    WorkcellConfig,
)
from apollo_mavis_v2_core.protocol import (
    ArmBringupTelemetry,
    ArmMaintenanceResult,
    EpisodeStatus,
    ReturnHomeResult,
    SessionInfo,
    SessionSpec,
)

from ..config import RuntimeConfig
from ..control.loop import (
    DEFAULT_ACTIVE_ARM,
    GATE_HOLD_PREFIX,
    RAIL_TRAVEL_M,
    ControlLoop,
    controller_error_title,
)
from ..control.pose_filter import PoseFilterConfig
from ..control.tracker_teleop import TrackerTeleop
from ..devices.hardware_monitor import CONNECTED_STATUSES, MONITOR_JOIN_TIMEOUT_S
from ..devices.tracker import TrackerSettings
from ..errors import MaintenanceUnavailableError, SessionError, SessionNotFoundError
from ..recorder.datasets import DatasetError, DatasetStore
from ..safety.gate import NullGate, SafetyGate
from ..safety.supervisor import SafetySupervisor
from ..safety.watchdog import ArmReportWatchdog, InputWatchdog
from ..streams.hub import VideoHub
from .hardware import (
    RailFlipWorkcell,
    RailHoldWorkcell,
    SessionStateProvider,
    apply_executor_caps,
    apply_teleop_caps,
    arm_label,
    bringup_rows,
    executor_caps_for,
    frozen_state,
    scale_control_config,
    scale_driver_config,
    teleop_rate_caps,
)
from .types import SessionState

if TYPE_CHECKING:
    from ..bus import RuntimeBus
    from ..devices.hardware_monitor import HardwareStateMonitor
    from ..devices.hardware_probe import HardwareProbe
    from ..dora_bridge.wiring import DoraWiring
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


MOTION_BUSY = "a planned motion is already running"
# Measured-arrival criterion of a sequentially executed plan (2026-09-08 review): arm k
# has ARRIVED when every joint of ``workcell.states()[k].q`` is within
# ``PLAN_ARRIVAL_TOL_RAD`` of its last waypoint and the rail slot within
# ``PLAN_ARRIVAL_TOL_RAIL_M``; the wait ends at the arm's budget deadline or, when that
# is nearer, ``PLAN_ARRIVAL_GRACE_S`` after the executor retired the waypoints. The real
# arms hold the servo command to < 1e-4 rad at rest (``cmd-meas`` on the health line);
# the servo-faithful sim settles a small step to < 1e-3 rad in ~0.6 s and the rail to
# < 1 mm in ~1 s (measured 2026-09-09 on ``mavis_v2``). The same tolerance is the
# "already there" / parked-arm threshold, so an arm the planner left within it is not
# submitted at all (``_moving_arms``).
PLAN_ARRIVAL_TOL_RAD = 1e-3
PLAN_ARRIVAL_TOL_RAIL_M = 2e-3
PLAN_ARRIVAL_GRACE_S = 2.0
PLAN_ARRIVAL_POLL_S = 0.02


PLAYBACK_INITIAL_LABEL = "episode_playback_initial"
PLAYBACK_LABEL = "episode_playback"
#: How far an arm may be from the episode's first frame and still be replayed from
#: there (rad, per joint). 1 deg: tight enough that a drifted cell is caught, loose
#: enough that a settled arm is not refused for a tenth of a degree.
PLAYBACK_START_TOL_RAD = math.radians(1.0)
#: Playback deadline = its own length x this + the grace, so a gate hold has room.
PLAYBACK_BUDGET_FACTOR = 3.0
PLAYBACK_BUDGET_GRACE_S = 20.0
#: An ACTION replay (``delta_ee`` / ``abs_ee``, 2026-09-11) is judged on the TCP, not the
#: joints: the executor tracks the Cartesian command through IK, and a 7-DOF arm has a null
#: space, so the measured joints legitimately end a few mrad off the recorded ones while
#: the tool is exactly where the recording says (the first sim replay: 8.8 mrad joints,
#: 1.7 mm TCP). "Done" = the measured TCP within these of the last recorded frame's TCP
#: once the arm has settled (no joint moving more than ``REPLAY_SETTLE_EPS`` over
#: ``REPLAY_SETTLE_S``); the joint / carriage residual is still reported.
REPLAY_TCP_TOL_M = 0.005
REPLAY_TCP_TOL_RAD = 0.02
REPLAY_SETTLE_S = 0.3
REPLAY_SETTLE_EPS = 1e-4


def _motion_title(label: str, profile) -> str:
    """How the Cockpit names the profile motion whose outcome it reports."""
    if label == "goto_profile":
        return f"Go to profile '{profile.name}'"
    if label == PLAYBACK_INITIAL_LABEL:
        return f"Return to {profile.name}"  # "... episode <id>'s initial state"
    return "Return to the initial condition"


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
    fault_detail: str = ""  # the arms' fault text while FAULT / RECOVERING (or a worker
    #   crash); written by the fault callback, cleared when the session runs again
    motion_detail: str = ""  # 2026-09-08: the operator-facing outcome of the LAST profile
    #   motion that did not arrive - a ``start_from`` plan the loop refused or could not
    #   plan, a "Go to profile" / `R` return that failed / was cancelled / was skipped.
    #   Survives a fault cycle (unlike ``fault_detail``) so the "then Go to profile"
    #   hint is still there once the operator has cleared the error; replaced by the
    #   next profile motion, "" once one arrives. On the wire as
    #   ``SessionTelemetry.fault_detail`` / ``SessionInfo.fault_detail``.
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
    # phase-14: the Online DAgger coordinator + its serial I/O worker (None otherwise)
    online_dagger: _OnlineDagger | None = None

    def notice(self, arms_carry_faults: bool = False) -> str:
        """``SessionTelemetry.fault_detail`` / ``SessionInfo.fault_detail``: the
        session-level text the per-arm rows do not already say. The motion notice
        first; else the fault text, but only while no arm row carries a fault (the
        FaultBanner lists those per arm - a bring-up fault before the first snapshot
        is the case that needs it here)."""
        if self.motion_detail:
            return self.motion_detail
        return "" if arms_carry_faults else self.fault_detail


@dataclass
class _ProfileMotion:
    """The claim one twin-planned profile motion holds while it plans and runs
    (:meth:`SessionManager._claim_profile_motion`)."""

    label: str


@dataclass
class _OnlineDagger:
    """What an Online DAgger session owns besides the DAgger stack (15-online-dagger §3):
    the coordinator, the serial worker its events / ``session.json`` writes run on,
    the hub it is attached to as ``trainer_sink`` and the session directory."""

    coordinator: Any  # OnlineDaggerCoordinator
    worker: Any  # SerialWorker
    hub: Any  # ExternalPolicyHub
    paths: Any  # OnlineDaggerPaths
    repo_id: str  # "online_dagger/<session_name>"
    created_fresh: bool  # this create() made the session directory (not a resume)

    def start(self) -> None:
        """Session announced: the session file, then the hub replays its cached trainer
        status / policy version into the coordinator (bus-thread path from here on)."""
        self.coordinator.on_session_start()
        self.hub.attach_trainer_sink(self.coordinator)

    def close(self) -> None:
        """Teardown (after the recorder stopped): detach from the hub, write the last
        ``session.json`` (``last_used_at``) and drain the worker."""
        try:
            self.hub.detach_trainer_sink(self.coordinator)
        finally:
            try:
                self.coordinator.close()
            finally:
                self.worker.close(wait=True)

    def discard_fresh(self) -> None:
        """A FRESH session whose bring-up failed right after the directory was created:
        remove it again so the name stays usable (never on a resume)."""
        if not self.created_fresh:
            return
        import shutil

        try:
            shutil.rmtree(self.paths.session_dir, ignore_errors=True)
        except Exception:  # noqa: BLE001
            logger.exception("Online DAgger session dir cleanup failed")

    def abandon(self) -> None:
        """Bring-up failed ANYWHERE after the coordinator was built (recorder, executor,
        a ``start()``): drop the worker (the coordinator never started, nothing to drain)
        and remove a fresh directory so the next POST with the name is not a 409."""
        try:
            self.worker.close(wait=False)
        finally:
            self.discard_fresh()


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
        # 2026-09-08: at most ONE twin-planned profile motion (per-episode return, `R`
        # return, Go to profile, the exit return) plans and runs at a time. The loop's
        # "plan executing" blocker runs BEFORE a worker plans (tens to hundreds of ms on
        # the twin), so two workers could both pass it and the second execute_plan
        # would replace the first mid-motion; this claim closes that window.
        self._profile_motion: _ProfileMotion | None = None
        self._profile_motion_lock = threading.Lock()
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
        # Data collection (2026-09-07): the recorded datasets (the store asks this
        # manager which repo the running session records into and which episode is
        # open, so its listing / deletion can mark and protect them) and the
        # Runtime-owned microphone reader (assigned after construction by
        # ``Runtime.__init__``; None = no audio sidecars). 2026-09-08 (15-online-dagger
        # §7 / D5): per-namespace roots - ``cfg.datasets_root`` stays the generic root,
        # ``cfg.datasets`` maps bc_demo / online_dagger to the operator's ~/data folders;
        # ``dataset_store.root_of`` is the only spelling of a dataset directory here.
        self.dataset_store = DatasetStore(
            cfg.datasets_root,
            default_namespace=cfg.datasets.default_namespace,
            namespaces=cfg.datasets.namespaces,
        )
        self.dataset_store.in_use_repo = self.in_use_repo
        self.dataset_store.open_episode = self.open_episode
        self.dataset_store.episode_delete_refusal = self.episode_delete_refusal
        self.microphone: Any = None
        # phase-12 (14-dora §4.2 / §7): the dora wiring (Runtime-owned, assigned after
        # construction) is told about bring-up / session start / teardown, and the sim
        # preview keeps the LAST session's final joint vector when the preview scene equals
        # the session scene (the parked Perception Arm survives session end in sim too).
        self.dora: DoraWiring | None = None
        self._parked_q: dict[str, Any] = {}  # arm_id -> q (incl. rail) at the last teardown
        self._parked_scene: str | None = None
        self._preview_scene_cache: dict[str, Any] = {}  # scene_id -> BuiltScene (previews)
        self._pending_preview: Any = None  # (RenderService, scene, cameras) warmed in teardown

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
                    policy_source=bp.spec.policy_source,
                    online_dagger=bp.spec.online_dagger,
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
            policy_source=s.spec.policy_source,  # phase-12 echo
            fault_detail=s.notice(),  # 2026-09-08: refused start_from / Go-to outcome
            online_dagger=s.spec.online_dagger,  # phase-14 echo
        )

    def bringup_telemetry(self) -> list[ArmBringupTelemetry] | None:
        """``SessionTelemetry.bringup`` (phase-09c): the hardware bring-up rows
        while one is in flight / until the session is RUNNING; ``None`` otherwise."""
        bp = self._bringup
        return bp.snapshot() if bp is not None else None

    def session_identity(self) -> tuple[str | None, str | None, str | None]:
        """``(session_id, mode, kind)`` of the live session, ``(None, None, None)``
        with none (2026-09-09; ``SessionTelemetry.session_id`` / ``mode`` / ``kind``).

        A hardware bring-up counts, exactly as it does for ``GET /api/session``: the
        session object does not exist until :meth:`_bringup_hardware` returns, and a
        Welcome page that offered a launcher during that window would 409. This is
        how a freshly loaded Welcome page — which never opens ``/ws/control`` and so
        has no ``SessionInfo`` — learns that a session is running and which Cockpit
        route to offer instead of a disabled launch card."""
        s = self.session
        if s is not None:
            return s.session_id, s.spec.mode, s.spec.kind
        bp = self._bringup
        if bp is not None and bp.spec is not None:
            return bp.session_id, bp.spec.mode, bp.spec.kind
        return None, None, None

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
        if (
            spec.kind != "hardware"
            and spec.mode in ("dagger", "inference")
            and spec.policy_source != "external"  # phase-12: no checkpoint is resolved
        ):
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
                if spec.mode in ("collect", "dagger"):
                    self._check_dataset_spec(spec, wc)  # before any box / camera is touched
                if spec.kind == "hardware":
                    twin_scene, samples = self._validate_hardware(spec, wc)
                if spec.online_dagger is not None:
                    self._check_online_dagger(spec)  # 15-online-dagger §3/§7: before side effects
                if spec.mode in ("collect", "dagger"):  # D6: dagger rollouts return too
                    self._check_return_to_start(spec)  # after the profile checks above
                # phase-12: every 409 above is evaluated BEFORE the idle arm reader is paused
                # (the refusal matrix reads monitor samples only, so it never needs the
                # reader's connections released); the bring-up itself is bracketed
                dora = self.dora if (self.dora is not None and self.dora.enabled) else None
                if dora is not None:
                    dora.before_bringup()  # idle arm reader releases its connections
                try:
                    if spec.kind == "hardware":
                        # hardware_session_active from here on (monitor hand-over): after the
                        # refusal matrix, before the first side effect (monitor.pause())
                        self._creating_kind = spec.kind
                        session = self._bringup_hardware(spec, wc, twin_scene, samples)
                    else:
                        session = self._bringup_sim(spec, wc)
                except Exception:
                    if dora is not None:
                        dora.after_teardown()  # nothing came up: idle publishing resumes
                    raise
                self.session = session
                if dora is not None:
                    try:
                        dora.after_session_start(self._session_facts(session, wc))
                    except Exception:  # noqa: BLE001 - the bus never blocks a session
                        logger.exception("dora session announce failed")
                if session.online_dagger is not None:
                    try:
                        session.online_dagger.start()  # session.json + trainer sink attach
                    except Exception:  # noqa: BLE001 - never blocks the session
                        logger.exception("Online DAgger coordinator start failed")
            finally:
                self._creating = False
                self._creating_kind = None
        threading.Thread(
            target=self._start_from_worker, args=(session,), name="start-from", daemon=True
        ).start()
        return self.info()

    def _task_repo_id(self, spec: SessionSpec, wc: WorkcellConfig) -> str | None:
        """The phase-07 task-derived repo id (10-frames §8.1) for ``dataset: None``, from
        the scene's rail flags; None when the scene cannot be resolved here (the
        recorder build refuses later)."""
        from ..recorder.features import ArmMeta, build_repo_id

        scene_id = (
            spec.sim_scene or wc.sim_scene
            if spec.kind == "sim"
            else spec.digital_twin_scene or wc.digital_twin_scene
        )
        if not scene_id:
            return None
        try:
            from apollo_mavis_v2_sim import REGISTRY

            meta = REGISTRY.meta(scene_id)
            arms = [ArmMeta(a, bool(meta.rail.get(a, False))) for a in spec.arms]
            frames = {a: spec.frames.get(a, f"arm_base:{a}") for a in spec.arms}
            return build_repo_id(spec.task or "task", arms, frames, "delta_ee")
        except Exception:  # noqa: BLE001 - unknown scene / bad frame: refused downstream
            return None

    def _check_dataset_spec(self, spec: SessionSpec, wc: WorkcellConfig | None = None) -> None:
        """``SessionSpec.dataset`` contract (04-runtime §10.5): a NEW dataset must not
        exist yet, a RESUMED one must (its schema is checked once the features are
        known, in :meth:`_build_collect_recorder`); a legacy phase-07 LeRobot v3 tree
        is read-only and a repo the export job is rewriting is refused too — those
        two checks also cover the task-derived repo of ``dataset: None`` (and a
        DAgger run's ``_dagger_<run_id>`` repo is always fresh)."""
        if spec.dataset is None:
            if spec.mode != "collect" or wc is None:
                return
            repo_id = self._task_repo_id(spec, wc)
            if repo_id is None:
                return
            self._refuse_exporting_or_legacy(repo_id)
            return
        repo_id = self.dataset_store.resolve(spec.dataset)
        self._refuse_exporting_or_legacy(repo_id)
        layout = self.dataset_store.layout_of(repo_id)
        exists = layout is not None
        if spec.dataset_resume and not exists:
            raise SessionError(
                f"unknown dataset {repo_id!r} - it has no recorded episode yet; start it as a "
                "new dataset instead"
            )
        if not spec.dataset_resume and exists:
            raise SessionError(
                f"dataset {repo_id!r} already exists - choose 'Continue existing' to append "
                "to it, or another name"
            )

    def _refuse_exporting_or_legacy(self, repo_id: str) -> None:
        if self.dataset_store.exporting == repo_id:
            raise SessionError(f"dataset {repo_id!r} is being exported - retry in a moment")
        if self.dataset_store.layout_of(repo_id) == "lerobot_v3":
            raise SessionError(
                f"dataset {repo_id!r} is a legacy LeRobot v3 dataset (read-only; already "
                "trainable as-is) - record into a new dataset"
            )

    def _return_profile_for(self, spec: SessionSpec):
        """The return-to-start profile (04-runtime §10.5): the ``start_from`` profile
        when one was chosen, else the kind's designated initial-condition profile,
        else None."""
        if spec.start_from.startswith("profile:"):
            try:
                return self.profile_store.get(spec.start_from.split(":", 1)[1])
            except ProfileNotFoundError:
                return None
        return self.profile_store.initial_for(spec.kind)

    def _check_return_to_start(self, spec: SessionSpec) -> None:
        """D6: the flag is on by default; with neither a ``start_from`` profile nor an
        initial-condition profile the POST is refused (the LaunchSheet disables Start
        with the same reason first)."""
        if spec.return_to_start and self._return_profile_for(spec) is None:
            raise SessionError(
                "return_to_start needs a start_from profile or an initial-condition profile - "
                "set an initial condition or untick 'Return to start'"
            )

    # -- Online DAgger (phase-14; 15-online-dagger §3, §7) -------------------------------------
    def online_dagger_root(self):
        """Where Online DAgger session directories live: the mapped ``online_dagger``
        namespace root (``~/data/online_dagger``), else ``<datasets_root>/online_dagger``."""
        mapped = self.dataset_store.namespaces.get("online_dagger")
        return mapped.root if mapped is not None else self.dataset_store.root / "online_dagger"

    def online_dagger_sessions(self):
        """``GET /api/online_dagger/sessions``: every ``<root>/*/session.json`` (newest
        first); a missing root lists nothing."""
        from ..dagger.online_dagger import OnlineDaggerCoordinator

        return OnlineDaggerCoordinator.scan_sessions(self.online_dagger_root())

    def _online_dagger_paths(self, od):
        """The session directory (15-online-dagger §0 item 6): the rollouts dataset is
        ``root_of("online_dagger/<s>")`` so REST / the store address it like any dataset;
        the session directory is its parent when the namespace maps a ``subdir`` (the
        shipped layout ``<root>/<s>/rollouts``), else the dataset directory itself."""
        from ..dagger.online_dagger import OnlineDaggerPaths

        repo_id = f"online_dagger/{od.session_name}"
        rollouts = self.dataset_store.root_of(repo_id)
        mapped = self.dataset_store.namespaces.get("online_dagger")
        session_dir = rollouts.parent if (mapped is not None and mapped.subdir) else rollouts
        return OnlineDaggerPaths(session_dir=session_dir, rollouts_dir=rollouts)

    def _external_hub_or_409(self):
        """The dora hub with a FRESH policy spec, or the 409 every external session
        shares (14-dora §6.1): ``(dora, hub, announce)``."""
        dora = self.dora
        hub = dora.policy_hub if dora is not None else None
        if hub is None or not dora.bridge.attached:
            raise SessionError("no external policy attached (dora bridge is not attached)")
        ann = hub.spec()
        if ann is None:
            raise SessionError(
                "no external policy attached (no policy_spec heartbeat within "
                f"{self.cfg.dora.policy.spec_stale_s:g} s)"
            )
        return dora, hub, ann

    def _check_online_dagger(self, spec: SessionSpec) -> None:
        """The Online DAgger refusal matrix (15-online-dagger §3, §7), evaluated at
        ``create()`` BEFORE any side effect: the session-directory rule (``resume: false``
        on an existing name / ``resume: true`` on a missing one are 409; a resume with an
        unreadable ``session.json`` is 409 rather than overwritten), the rollouts dataset
        must be neither exporting nor a legacy tree, then a trainer-capable policy node
        must be attached (a fresh spec that lacks the ``online_dagger`` capability is its
        own 409). There is NO offline-dataset check: the trainer configures its own anchor
        (operator decision 2026-09-08, §0 item 2)."""
        od = spec.online_dagger
        assert od is not None
        paths = self._online_dagger_paths(od)
        exists = paths.session_dir.is_dir()
        if exists and not od.resume:
            raise SessionError(
                f"Online DAgger session '{od.session_name}' already exists - resume it or pick "
                "another name"
            )
        if od.resume and not exists:
            raise SessionError(f"Online DAgger session '{od.session_name}' not found")
        if od.resume and paths.session_json.exists():
            # a corrupt record must never be overwritten by a fresh document: the rollouts
            # rows and the counters a resume continues live only here
            from ..recorder.manifest import read_json

            if not isinstance(read_json(paths.session_json), dict):
                raise SessionError(
                    f"Online DAgger session '{od.session_name}': session.json is unreadable - "
                    "fix or remove it"
                )
        # the rollouts dataset is a dataset like any other: a running export of it or a
        # legacy LeRobot v3 tree at its path is refused here, before any side effect
        # (the same 409s every collect / dagger POST gets from _check_dataset_spec)
        self._refuse_exporting_or_legacy(f"online_dagger/{od.session_name}")
        _dora, hub, _ann = self._external_hub_or_409()
        if not hub.trainer_capable():
            raise SessionError(
                "no Online DAgger trainer attached (the policy node does not report the "
                "online_dagger capability)"
            )

    def _build_online_dagger(self, spec: SessionSpec, session_id: str, run_id: str, ann, hub, dora):
        """Session directory + ``session.json`` + the coordinator (15-online-dagger §3).
        Runs inside the sim bring-up before the recorder / loop exist; no motion. Creates
        ``<root>/<s>/rollouts`` ONLY (the trainer owns whatever else it puts there) and
        writes the first ``session.json`` of a FRESH directory (a resumed one is left as
        it is until the session is RUNNING). The coordinator's ``publish`` /
        ``write_session`` go through ONE serial worker so no event or file write ever
        runs on the tick, and the events keep their order."""
        from ..dagger.online_dagger import (
            OnlineDaggerCoordinator,
            SerialWorker,
            write_session_json_atomic,
        )
        from ..recorder.manifest import read_json

        od = spec.online_dagger
        assert od is not None
        paths = self._online_dagger_paths(od)
        created_fresh = not paths.session_dir.is_dir()
        resume_doc = None
        if od.resume:
            resume_doc = read_json(paths.session_json)
            if not isinstance(resume_doc, dict) and paths.session_json.exists():
                # _check_online_dagger refused this already; belt and braces before side effects
                raise SessionError(
                    f"Online DAgger session '{od.session_name}': session.json is unreadable - "
                    "fix or remove it"
                )
        paths.mkdirs()
        worker = SerialWorker()
        publisher = dora.publisher

        def publish(kind: str, payload: dict) -> None:
            # the id is pinned HERE: the publisher resolves a missing one from its live
            # facts on its worker thread, so an event still queued at session_ended() (a
            # teardown discard) or published during BRINGUP would spell "" and trainers
            # filter by session id (the same reason enqueue_event captures it eagerly)
            worker.submit(publisher.publish_event, kind, payload, session_id)

        def write_session(doc: dict) -> None:
            worker.submit(write_session_json_atomic, paths.session_json, doc)

        coordinator = OnlineDaggerCoordinator(
            od,
            session_id=session_id,
            paths=paths,
            spec=spec,
            task=spec.task,
            run_id=run_id,
            spec_stale_s=self.cfg.dora.policy.spec_stale_s,
            policy_version=int(ann.policy_version),
            publish=publish,
            write_session=write_session,
            session_file_hz=self.cfg.online_dagger.session_file_hz,
        )
        if od.resume:
            if isinstance(resume_doc, dict):
                coordinator.from_session_json(resume_doc)
            else:
                logger.warning(
                    "Online DAgger resume of %r: no session.json - starting the count at 0",
                    od.session_name,
                )
        else:
            # a FRESH directory gets its first document now (listed at once; abandon()
            # removes the whole directory if the bring-up fails). A RESUMED record is NOT
            # rewritten here: the bring-up can still 409 after this point (frame mismatch,
            # recorder schema) and the file must not then point at a session that never
            # ran, nor move in the last_used_at listing; on_session_start() writes it once
            # the session is RUNNING.
            write_session_json_atomic(paths.session_json, coordinator.to_session_json())
        logger.info(
            "Online DAgger session %r (%s): dir %s, rollouts %s",
            od.session_name,
            "resume" if od.resume else "new",
            paths.session_dir,
            paths.rollouts_dir,
        )
        return _OnlineDagger(
            coordinator=coordinator,
            worker=worker,
            hub=hub,
            paths=paths,
            repo_id=f"online_dagger/{od.session_name}",
            created_fresh=created_fresh,
        )

    @staticmethod
    def _online_dagger_saved_hook(inner, coordinator):
        """Recorder-thread hook of an Online DAgger session: the coordinator FIRST
        (``events.episode_saved`` + ``online_dagger`` block, the ``session.json`` rollouts
        row - one worker submit), THEN the executor's boundary callback. The callback arms
        the boundary the next tick consumes, whose ``gate.reset()`` events go through the
        same worker: this order puts ``episode_saved`` before the boundary's
        ``events.gate`` on the wire, never after."""

        def hook(index: int, summary, spool_path: str) -> None:
            try:
                coordinator.on_episode_saved(summary.episode_id, index, summary, spool_path)
            except Exception:  # noqa: BLE001 - the coordinator never breaks a save
                logger.exception("Online DAgger on_episode_saved failed")
            if inner is not None:
                inner(index, summary, spool_path)

        return hook

    @staticmethod
    def _online_dagger_discard_hook(coordinator):
        """Recorder-thread hook: a discarded rollout -> ``events.episode_discarded`` (the
        recorder already removed the episode's temp directory; nothing is persisted)."""

        def hook(index: int | None, episode_id: str | None, reason: str) -> None:
            try:
                coordinator.on_episode_discarded(episode_id, index, reason)
            except Exception:  # noqa: BLE001
                logger.exception("Online DAgger on_episode_discarded failed")

        return hook

    def _gate_events_hook(self):
        """``GatedPolicyExecutor.on_gate_events`` for a policy session WITHOUT a
        coordinator (15-online-dagger §3): the dora publisher's queue (drained on the
        ``dora-publisher`` thread), None when the bridge is off."""
        dora = self.dora
        if dora is None or not dora.enabled:
            return None
        return dora.publish_gate_events

    # -- dataset store hooks (04-runtime §10.6) ---------------------------------------------
    def in_use_repo(self) -> str | None:
        """The repo id the running collect / DAgger session records into, else None."""
        session = self.session
        if session is None or session.recorder_thread is None:
            return None
        return getattr(session.recorder_thread, "repo_id", None)

    def open_episode(self) -> str | None:
        """The id of the episode being recorded right now (deletion -> 409), else None."""
        session = self.session
        if session is None or session.recorder_thread is None:
            return None
        return getattr(session.recorder_thread, "open_episode_id", None)

    def episode_delete_refusal(self, repo_id: str) -> str | None:
        """``DatasetStore.episode_delete_refusal``: a SAVED rollout of the running Online
        DAgger session may not be deleted while it runs (15-online-dagger §3) — the
        coordinator's ``rollouts_saved`` / ``session.json`` rows and the trainer's buffer
        (told about discards only, never deletions) would silently diverge from the
        dataset. Every other dataset keeps the phase-13 rule (only the open episode is
        protected). None = allowed."""
        session = self.session
        od = session.online_dagger if session is not None else None
        if od is None or od.repo_id != repo_id:
            return None
        return (
            f"dataset {repo_id!r} is in use by the running Online DAgger session - end the "
            "session first (the trainer is told about discards, not deletions)"
        )

    def episode_status(self) -> EpisodeStatus | None:
        """The recorder's status (the deprecated ``GET /api/episodes`` alias)."""
        session = self.session
        if session is None or session.recorder_thread is None:
            return None
        return session.recorder_thread.status()

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
        # phase-12 (14-dora §7): a STANDBY preview render service is warmed now (its EGL
        # renderer set-up stalls the GPU driver ~150 ms, which is fine at session start and
        # unacceptable at teardown) so the camera streams hand over within one frame period
        if self.dora is not None and self.dora.enabled and self._pending_preview is None:
            try:
                self._pending_preview = self._prepare_sim_previews()
            except Exception:  # noqa: BLE001 - the standby is an optimisation
                logger.exception("standby preview warm-up failed")
        rs = RenderService()
        rs.start()
        overrides = _microphone_overrides(session_cfg)
        scene = _servo_faithful_scene(scene_id, overrides)
        workcell = SimWorkcell(
            scene, session_cfg, render_service=rs, depth_cameras=self._depth_cameras()
        )
        workcell.start()

        # Twin: always built for planning (goto / start_from); it is the GATE
        # twin only under safety_debug (11-safety §5).
        twin = DigitalTwin(
            REGISTRY.build(scene_id, overrides),
            inflation_m=safety.geom_inflation_m,
            render_service=rs if safety.safety_debug else None,
            allowed_pairs_extra=safety.allowed_pairs_extra,
            hysteresis_m=safety.hysteresis_m,  # the planner escapes the gate's band too
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
            clearance_sweep_m=safety.clearance_sweep_m,
        )
        session_id = uuid.uuid4().hex
        recorder_thread = None
        policy_session = None
        started: list = []  # what to stop, in reverse, if the bring-up fails after a start()
        try:
            if spec.mode == "collect":
                recorder_thread = self._build_collect_recorder(
                    spec, session_cfg, workcell, scene, session_id
                )
            if spec.mode in ("dagger", "inference"):
                loop, recorder_thread, policy_session = self._build_policy_stack(
                    spec,
                    session_cfg,
                    workcell,
                    scene,
                    session_id,
                    ik,
                    kin,
                    twin,
                    supervisor,
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
                    plan_gate_hold_s=self.cfg.hardware_session.plan_gate_hold_s,
                    speed_scale=spec.speed_scale,
                )
            # the three starts sit INSIDE the rollback scope: a start that raises must
            # stop what came up before it and, for Online DAgger, drop a fresh session dir
            loop.start()
            started.append(loop)
            if recorder_thread is not None:
                recorder_thread.start()
                started.append(recorder_thread)
            if policy_session is not None:
                policy_session.start()
                started.append(policy_session)
        except Exception:
            for obj in reversed(started):
                try:
                    obj.stop()
                except Exception:  # noqa: BLE001 - keep unwinding
                    logger.exception("bring-up rollback: %s.stop() failed", type(obj).__name__)
            online_dagger = getattr(policy_session, "online_dagger", None)
            if online_dagger is not None:
                online_dagger.abandon()  # a fresh name stays usable (15-online-dagger §3)
            workcell.stop()
            rs.stop()
            self.start_previews()
            raise

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
            online_dagger=getattr(policy_session, "online_dagger", None),
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

        teleop or collect (phase-09c; data collection since 2026-09-07 - DAgger /
        inference on hardware stay 409) -> at least one arm, every arm in the hardware
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
        if spec.mode not in ("teleop", "collect"):
            raise SessionError(
                "hardware sessions support teleop and data collection only "
                f"({spec.mode} on hardware: not yet)"
            )
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
        if spec.mode == "collect" and not self._live_hardware_cameras():
            # checked HERE, before any box is enabled (2026-09-07 review)
            raise SessionError(
                "data collection needs at least one live hardware camera - none is open "
                "(see the Hardware tab camera tiles)"
            )
        return twin_scene, samples

    def _live_hardware_cameras(self) -> dict:
        return {
            cid: cam
            for cid in list(self._hw_cameras)
            if (cam := self.hardware_camera(cid)) is not None
        }

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
                        f"{arm_label(arm_id)}: {_bringup_step(st)} - {error or 'not connected'}"
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
                base_pose={k: v for k, v in base_pose_overrides(wc).items() if k in meta.arm_ids},
            )
            try:
                twin = DigitalTwin(
                    REGISTRY.build(twin_scene, overrides),
                    inflation_m=safety.geom_inflation_m,
                    allowed_pairs_extra=safety.allowed_pairs_extra,
                    hysteresis_m=safety.hysteresis_m,  # the planner escapes the gate's band too
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
                clearance_sweep_m=safety.clearance_sweep_m,
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
            # TELEOP is capped the same way (2026-09-07): the tracker target chain ran
            # at target_rate (1.0 m/s) against a streamer that executes 0.2 m/s at
            # scale 1.0, so every fast hand motion hit the 0.025 m leash and the
            # truncation was folded into the anchor - hand travel silently DISCARDED,
            # and the remainder still arriving up to a leash after the hand stopped.
            control_cfg = apply_teleop_caps(control_cfg, caps)
            if caps.source == "servo":
                unscaled = scale_control_config(self.cfg.control, scale)
                tcp_mps, joint_radps = teleop_rate_caps(unscaled, caps)
                logger.info(
                    "hardware loop: plan executor capped by the servo stream - slew %.5f rad/tick, "
                    "cart %.5f m/tick, rail %.5f m/tick (host slew %.5f, host rail %.5f)",
                    caps.slew_rad_per_tick,
                    caps.cart_step_m if caps.cart_step_m is not None else float("nan"),
                    control_cfg.jog.rail_m_per_tick,
                    unscaled.jog.slew_rad_per_tick,
                    unscaled.jog.rail_m_per_tick,
                )
                logger.info(
                    "hardware loop: teleop capped by the servo stream - tcp %.4f m/s, "
                    "joint %.4f rad/s, dq_max %.5f rad/tick (requested target_rate %.3f m/s / "
                    "%.3f rad/s, dq_max %.5f)",
                    tcp_mps if tcp_mps is not None else float("nan"),
                    joint_radps if joint_radps is not None else float("nan"),
                    control_cfg.dq_max_rad,
                    unscaled.target_rate.v_mps,
                    unscaled.target_rate.w_radps,
                    unscaled.dq_max_rad,
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
                plan_gate_hold_s=self.cfg.hardware_session.plan_gate_hold_s,
                speed_scale=scale,
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
        recorder_thread = None
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
            # 11b. data collection (2026-09-07, §10.5): the recorder reads the ADOPTED
            #      preview cameras (the UVC nodes are open exactly once) and the twin's
            #      kinematics; built before the loop starts so a dataset refusal (409)
            #      never leaves a running loop behind.
            if spec.mode == "collect":
                cams = self._live_hardware_cameras()
                if not cams:  # re-checked: a preview may have died during the connect
                    raise SessionError(
                        "data collection needs at least one live hardware camera - none is "
                        "open (see the Hardware tab camera tiles)"
                    )
                for a in spec.arms:
                    progress.set(a, "recorder", "pending", "opening the dataset writer")
                recorder_thread = self._build_collect_recorder(
                    spec,
                    rig.session_cfg,
                    rig.workcell,
                    rig.twin.scene,
                    session_id,
                    kind="hardware",
                    cameras=cams,
                    camera_cfgs={c.id: c for c in wc.cameras},
                )
                rig.loop.recorder = recorder_thread
                for a in spec.arms:
                    progress.set(a, "recorder", "ok", f"recording into {recorder_thread.repo_id}")
            rig.loop.start()
            if recorder_thread is not None:
                recorder_thread.start()
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
                recorder_thread=recorder_thread,
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
            if recorder_thread is not None:
                try:
                    recorder_thread.stop()  # finalize the (empty) dataset, never half-open
                except Exception:  # noqa: BLE001
                    logger.exception("hardware bring-up abort: recorder stop failed")
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
        starts; returns ``(waypoints, grippers)`` for ``execute_plan``, the
        waypoints keyed in the planner's ``arm_order`` (the start_from worker
        executes them one arm at a time in that order). A failed plan raises
        ``SessionError`` ("profile motion not collision-free") and the caller tears
        the session down."""
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
        result = rig.twin.plan(
            PlanRequest(q_start=q_start, q_goal=q_goal, speed_scale=spec.speed_scale)
        )
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
        return self._ordered_waypoints(result), grippers

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
        dagger_ctx: dict | None = None,  # {"run_id", "gate"[, "repo_id", "coordinator"]}
        #   -> DaggerRecorderThread; "repo_id" = record into THIS repo verbatim (Online DAgger
        #   rollouts "online_dagger/<s>", resumed when it exists) instead of _dagger_<run_id>
        *,
        kind: str = "sim",
        cameras: dict | None = None,  # hardware: the adopted preview cameras (id -> camera)
        camera_cfgs: dict | None = None,  # hardware: CameraConfig by id (intrinsics)
    ):
        """Collect/DAgger recorder stack (04-runtime §10; 10-frames §5-§9).

        Heavy imports (lerobot -> torch) happen inside the recorder ctor —
        only collect/dagger sessions ever pay them. With ``dagger_ctx`` the
        schema gains the 12-dagger §4 columns and the repo id the
        ``_dagger_{run_id}`` suffix (dedicated repo; seed data never mutated).

        ``kind == "hardware"`` (2026-09-07): ``scene`` is the digital-twin
        ``BuiltScene`` (kinematics for the recording-frame conversion and the
        wrist-camera extrinsics), ``cameras`` the live UVC previews and
        ``camera_cfgs`` their configs — the factory D435 intrinsics go into the
        sidecar, the twin camera of the same name (``<id>`` or ``<id>_cam``) gives
        ``T_W_C``. The dataset repo follows ``SessionSpec.dataset`` (resume =
        schema-checked against ``manifest.json`` incl. the feature info blocks, 409
        on a mismatch) and gets ``audio.wav`` inside every episode directory when
        the microphone reader is live (hardware; sim only with the fake backend).
        """
        import hashlib

        import numpy as np
        from apollo_mavis_v2_core import parse_frame

        from ..recorder.episode_recorder import EpisodeDirRecorder, dataset_incompatibility
        from ..recorder.features import ArmMeta, build_features, build_repo_id, build_robot_type
        from ..recorder.frames import RecordingFrameConverter
        from ..recorder.kinematics import RecorderKinematics
        from ..recorder.manifest import is_legacy_v3, read_manifest
        from ..recorder.sidecars import SidecarWriter, pose_json
        from ..recorder.thread import RecorderThread

        sim = kind == "sim"
        cams = dict(workcell.cameras) if cameras is None else dict(cameras)
        cam_cfgs = (
            {c.id: c for c in session_cfg.cameras} if camera_cfgs is None else dict(camera_cfgs)
        )
        action_space = "delta_ee"  # canonical (10-frames §1.2); per-session later
        arms = [ArmMeta(a, bool(scene.meta.rail[a])) for a in spec.arms]
        frames = {a: spec.frames.get(a, f"arm_base:{a}") for a in spec.arms}
        kin = RecorderKinematics(scene)

        def twin_camera(cam_id: str) -> str | None:
            """MJCF camera that models ``cam_id`` (sim: itself; hardware: ``<id>_cam``)."""
            for name in (cam_id, f"{cam_id}_cam"):
                try:
                    kin.camera_static(name)
                except KeyError:
                    continue
                return name
            return None

        # camera:<id> recording frames need a declared camera with static T_W_C
        # (10-frames §5.1); wrist/moving cameras are rejected here (-> 409).
        camera_poses = {}
        for arm_id, ref in frames.items():
            parsed = parse_frame(ref)
            if parsed.kind != "camera":
                continue
            if parsed.ident not in cams:
                raise SessionError(f"frames[{arm_id!r}]: unknown camera {parsed.ident!r}")
            twin_cam = twin_camera(parsed.ident)
            if twin_cam is None or not kin.camera_static(twin_cam):
                raise SessionError(
                    f"frames[{arm_id!r}]: camera {parsed.ident!r} is not static in world"
                )
            camera_poses[parsed.ident] = kin.camera_world(twin_cam)
        converter = RecordingFrameConverter(frames, camera_poses)

        cam_res = {cid: tuple(cam.resolution) for cid, cam in cams.items()}
        features = build_features(arms, frames, cam_res, action_space)
        if spec.dataset is not None:
            repo_id = self.dataset_store.resolve(spec.dataset)
        elif dagger_ctx is not None and dagger_ctx.get("repo_id"):
            repo_id = str(dagger_ctx["repo_id"])  # Online DAgger rollouts (15-online-dagger §4)
        else:
            repo_id = build_repo_id(spec.task or "task", arms, frames, action_space)
        if dagger_ctx is not None:
            from ..dagger.recorder import dagger_features

            features = dagger_features(features, dagger_ctx["run_id"])
            if not dagger_ctx.get("repo_id"):
                repo_id = f"{repo_id}_dagger_{dagger_ctx['run_id']}"
        root = self.dataset_store.root_of(repo_id)  # per-namespace roots (D4)
        robot_type = build_robot_type(len(arms), sim=sim)
        if is_legacy_v3(root):
            raise SessionError(
                f"dataset {repo_id!r} is a legacy LeRobot v3 dataset (read-only; already "
                "trainable as-is) - record into a new dataset"
            )
        try:
            manifest = read_manifest(root)
        except ValueError as e:  # unsupported layout major
            raise SessionError(f"dataset {repo_id!r}: {e}") from e
        why = dataset_incompatibility(manifest, features, self.cfg.recorder.fps, robot_type)
        if why is not None:
            raise SessionError(
                f"dataset {repo_id!r} cannot be continued by this session: {why} - record "
                "into a new dataset"
            )
        try:
            recorder = EpisodeDirRecorder(
                self.cfg.recorder,
                features,
                root,
                repo_id,
                robot_type,
                default_task=spec.task or "",
            )
        except (ValueError, RuntimeError) as e:  # codec family / encoder refusals (§7.5)
            raise SessionError(f"dataset {repo_id!r}: {e}") from e

        sidecars = SidecarWriter(root)
        scene_sha = sidecars.archive_scene_xml(scene.xml)
        sidecars.write_session(
            session_id,
            spec.mode,
            spec.model_dump(mode="json"),
            {
                "kind": kind,
                "config_sha256": hashlib.sha256(session_cfg.model_dump_json().encode()).hexdigest(),
                "arm_ids": list(spec.arms),
                "rail": {a.arm_id: a.has_rail for a in arms},
                "cameras": {
                    cid: {
                        "resolution": list(cam_res[cid]),
                        "kind": getattr(cam_cfgs.get(cid), "kind", "sim"),
                        "twin_camera": twin_camera(cid),
                    }
                    for cid in cams
                },
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
        initial = self.profile_store.initial_for(kind)  # type: ignore[arg-type]
        profile_snapshot = None
        if spec.start_from.startswith("profile:"):
            profile_snapshot = self.profile_store.get(spec.start_from.split(":", 1)[1]).model_dump(
                mode="json"
            )
        return_profile = self._return_profile_for(spec) if spec.return_to_start else None
        meta_base = {
            "session_id": session_id,
            "scene_xml_sha256": scene_sha,
            "start_from": spec.start_from,
            "initial_condition_profile_id": initial.profile_id if initial else None,
            "return_profile_id": return_profile.profile_id if return_profile else None,
            "profile_snapshot": profile_snapshot,
            "frames": dict(frames),
            "arm_bases": arm_bases,
        }

        def extrinsics_fn(arm_states):  # runs on the recorder thread (owns kin)
            q_by_arm = {a: np.asarray(arm_states[a].q) for a in spec.arms}
            out = {}
            for cam_id in cams:
                cfg = cam_cfgs.get(cam_id)
                twin_cam = twin_camera(cam_id)
                static = kin.camera_static(twin_cam) if twin_cam else False
                out[cam_id] = {
                    "T_W_C": pose_json(kin.camera_world(twin_cam, q_by_arm)) if twin_cam else None,
                    "extrinsics_frame": "world" if static else None,
                    "twin_camera": twin_cam,  # hardware: the MJCF camera standing in for it
                    "intrinsics": (cfg.intrinsics.model_dump() if cfg and cfg.intrinsics else None),
                    "calibration_file": None,  # pose comes from the (twin) MJCF
                    "calibration_sha256": None,
                }
            return out

        audio = None
        mic = self.microphone
        mic_backend = getattr(getattr(mic, "cfg", None), "backend", None)
        if (
            self.cfg.recorder.audio
            and mic is not None
            and getattr(mic, "active", False)
            and (not sim or mic_backend == "fake")  # 04-runtime §10.5: sim only with a fake reader
        ):
            from ..recorder.audio import EpisodeAudioSink

            audio = EpisodeAudioSink(mic)

        if dagger_ctx is not None:
            from ..dagger.recorder import DaggerRecorderThread

            dagger_thread = DaggerRecorderThread(
                recorder,
                self.bus,
                cams,
                arms,
                converter,
                kin,
                fps=self.cfg.recorder.fps,
                episode_meta_base=meta_base,
                extrinsics_fn=extrinsics_fn,
                audio=audio,
                action_filter=spec.action_filter,
                gate=dagger_ctx["gate"],
                run_id=dagger_ctx["run_id"],
                dataset_root=root,
                coordinator=dagger_ctx.get("coordinator"),
            )
            dagger_thread.on_episode_done = self._on_episode_done  # D6: dagger returns too
            return dagger_thread
        thread = RecorderThread(
            recorder,
            self.bus,
            cams,
            arms,
            converter,
            kin,
            fps=self.cfg.recorder.fps,
            episode_meta_base=meta_base,
            extrinsics_fn=extrinsics_fn,
            audio=audio,
            repo_id=repo_id,
            dataset_root=root,
            action_filter=spec.action_filter,
        )
        thread.on_episode_done = self._on_episode_done
        return thread

    # -- return-to-start (D6; 04-runtime §10.5) ---------------------------------------------
    def _on_episode_done(self, outcome: str, index: int | None) -> None:
        """Recorder-thread hook fired after every save / discard (BEFORE the recorder
        flips back to idle, so telemetry reads ``saving -> returning -> idle``): start
        the return motion when the session asked for it. Never blocks the recorder."""
        session = self.session
        if (
            session is None
            or session.spec.mode not in ("collect", "dagger")  # D6 (15-online-dagger)
            or not session.spec.return_to_start
            or session.recorder_thread is None
        ):
            return
        rt = session.recorder_thread
        if session.state is not SessionState.RUNNING:  # faulted / recovering / tearing down
            rt.set_returning(False, f"return skipped: session {session.state.value}")
            return
        if session.supervisor.watchdog.tripped:
            # The browser's deadman is latched (keys down / stale): a planned motion
            # would start under an operator who is not in control - hold instead.
            rt.set_returning(False, "return skipped: browser input latched - release every key")
            return
        profile = self._return_profile_for(session.spec)
        if profile is None:
            rt.set_returning(False, "return skipped: no return profile")
            return
        token = self._claim_profile_motion("return_to_start")
        if token is None:  # an `R` / Go-to-profile motion is planning or running
            rt.set_returning(False, f"return skipped: {MOTION_BUSY}")
            return
        rt.set_returning(True, f"returning to profile '{profile.name}'")
        threading.Thread(
            target=self._return_home_worker,
            args=(session, profile, outcome, token),
            name="return-to-start",
            daemon=True,
        ).start()

    # -- one profile motion at a time ---------------------------------------------------------
    def _claim_profile_motion(self, label: str) -> _ProfileMotion | None:
        """Claim the single profile-motion slot for ``label``; None when another
        motion (per-episode return, `R`, Go to profile, exit return) holds it. The
        holder releases it with :meth:`_release_profile_motion` and ITS token only, so a
        late release after a teardown can never drop a newer session's claim."""
        with self._profile_motion_lock:
            if self._profile_motion is not None:
                return None
            token = _ProfileMotion(label)
            self._profile_motion = token
            return token

    def _release_profile_motion(self, token: _ProfileMotion | None) -> None:
        with self._profile_motion_lock:
            if token is not None and self._profile_motion is token:
                self._profile_motion = None

    @property
    def profile_motion_in_flight(self) -> str | None:
        """The label of the profile motion planning / running right now, else None."""
        claim = self._profile_motion
        return claim.label if claim is not None else None

    # -- shared return machinery: goals, one planned phase, the wait -----------------------
    def _return_goals(self, session: ActiveSession, profile, arms):
        """Per-arm return targets toward ``profile`` (04-runtime §10.5).

        Returns ``(states, q_start, q_joint, q_full, grippers)`` keyed by the arms
        the profile actually covers (an arm it does not cover simply stays where
        it is). ``q_joint`` carries the profile's 7 joints with the arm's CURRENT
        rail value in the rail slot; ``q_full`` carries the profile's rail too
        (identical to ``q_joint`` when the profile stores no rail for that arm, i.e.
        ``rail_pos_m is None`` = "keep the carriage"). Splitting the two is what
        lets the exit return move the joints FIRST and the carriage after
        (2026-09-08 operator decision: sliding a rail with the arm extended sweeps
        it through the cell, folding first does not).

        Rail GOALS are snapped into ``[0, RAIL_TRAVEL_M]`` (2026-09-09): a carriage
        that settled a few tenths of a mm past an end stop reads e.g. -0.0004 m, and a
        goal copied verbatim from such a reading (the joints phase keeps the current
        carriage; a profile saved from it stores the same value) can never be reached
        by the loop's command path, whose per-tick clamp pins the rail slot to the
        travel. The START stays the raw measurement (the planner is given the arm
        where it is); the goal a hair inside the travel is within the arrival
        tolerance either way, so nothing observable changes except that the plan's
        goal is reachable by construction (``ControlLoop._confirm_plan_arrivals`` keeps
        its reachable-limit fallback for goals from other sources).
        """
        states = session.workcell.states()
        q_start: dict[str, list[float]] = {}
        q_joint: dict[str, list[float]] = {}
        q_full: dict[str, list[float]] = {}
        grippers: dict[str, float] = {}
        for arm_id in arms:
            posture = profile.arms.get(arm_id)
            if posture is None:
                continue  # the profile does not cover this arm: it stays
            st = states[arm_id]
            joints = [float(x) for x in posture.q]
            goal_joint = list(joints)
            goal_full = list(joints)
            if st.q.shape[0] > 7:  # rail slot LAST; None = keep the current carriage
                current_rail = min(max(float(st.q[7]), 0.0), RAIL_TRAVEL_M)
                rail = posture.rail_pos_m
                goal_joint.append(current_rail)
                goal_full.append(
                    current_rail if rail is None else min(max(float(rail), 0.0), RAIL_TRAVEL_M)
                )
            q_start[arm_id] = [float(x) for x in st.q]
            q_joint[arm_id] = goal_joint
            q_full[arm_id] = goal_full
            if arm_id in session.loop.gripper_arms:
                grippers[arm_id] = float(posture.gripper_open_frac)
        return states, q_start, q_joint, q_full, grippers

    def _plan_return(self, session: ActiveSession, states, q_start: dict, q_goal: dict):
        """``twin.plan`` for one return phase (plain sim: keep the plan twin fresh)."""
        from apollo_mavis_v2_core import PlanRequest

        if session.supervisor.twin is None:
            session.twin.sync(states)
        # the escape from a pinched start is judged at the speed the plan will run at
        return session.twin.plan(
            PlanRequest(q_start=q_start, q_goal=q_goal, speed_scale=session.spec.speed_scale)
        )

    @staticmethod
    def _plan_failure_text(result) -> str:
        pair = f" ({' / '.join(result.failing_pair)})" if result.failing_pair else ""
        return f"{result.failure}{pair}"

    # -- sequential execution of a planned multi-arm motion (2026-09-08 evening) --------
    @staticmethod
    def _ordered_waypoints(result) -> dict:
        """``result.waypoints`` re-keyed in ``result.arm_order`` - the order the
        sequential planner validated (arm k with arms < k at their GOALS and arms > k
        at their STARTS) and therefore the ONLY order the paths may be executed in.
        Falls back to the dict's insertion order for a planner that does not fill
        ``arm_order`` (test fakes); arms the order does not name come last."""
        waypoints = dict(result.waypoints)
        order = [a for a in (getattr(result, "arm_order", None) or []) if a in waypoints]
        order += [a for a in waypoints if a not in order]
        return {a: waypoints[a] for a in order}

    @staticmethod
    def _moving_arms(waypoints: dict) -> list[str]:
        """The arms whose waypoints actually go somewhere (in ``waypoints`` order): an
        arm the planner left at its start (every point within the arrival tolerance of
        the first - 1e-3 rad / m, the manager's "already there" threshold; a hardware
        start that differs from the goal by the SDK's ~1e-4 rad read-back noise is
        parked, not a 1-2 tick micro-plan) is not submitted at all - it has nothing to
        arrive at."""
        import numpy as np

        out = []
        for arm_id, wps in waypoints.items():
            pts = [np.asarray(w, dtype=np.float64) for w in wps]
            if pts and any(np.max(np.abs(q - pts[0])) > PLAN_ARRIVAL_TOL_RAD for q in pts[1:]):
                out.append(arm_id)
        return out

    @staticmethod
    def _arrival_error(q_meas, goal) -> tuple[float, float | None]:
        """``(max joint error rad, rail error m | None)`` of a measured q against a
        goal (the rail slot compared only when both carry one)."""
        import numpy as np

        q = np.asarray(q_meas, dtype=np.float64)
        g = np.asarray(goal, dtype=np.float64)
        n = min(7, q.shape[0], g.shape[0])
        joints = float(np.max(np.abs(q[:n] - g[:n]))) if n else 0.0
        rail = float(abs(q[7] - g[7])) if q.shape[0] > 7 and g.shape[0] > 7 else None
        return joints, rail

    @classmethod
    def _arrived(cls, q_meas, goal) -> bool:
        joints, rail = cls._arrival_error(q_meas, goal)
        return joints <= PLAN_ARRIVAL_TOL_RAD and (rail is None or rail <= PLAN_ARRIVAL_TOL_RAIL_M)

    def _await_arrival(
        self,
        session: ActiveSession,
        arm_id: str,
        goal,
        *,
        running: Callable[[], bool],
        deadline: float,
    ) -> tuple[str, str]:
        """Wait until ``arm_id``'s MEASURED q is within the arrival tolerance of ``goal``
        (the arm's last waypoint). ``("done", "")`` on arrival; ``("cancelled", ...)`` when
        the session leaves its motion state or the arm faults meanwhile; ``("stalled",
        "<how far off after how long>")`` at ``deadline`` (monotonic). Polls the driver's
        cached states every ``PLAN_ARRIVAL_POLL_S``; never commands anything."""
        t0 = time.monotonic()
        while True:
            try:
                q = session.workcell.states()[arm_id].q
            except Exception:  # noqa: BLE001 - a read hiccup is not an arrival
                q = None
            if q is not None and self._arrived(q, goal):
                return "done", ""
            if not running():
                return "cancelled", f"session {session.state.value}"
            if arm_id in session.loop.faulted_arms:
                return "cancelled", "driver fault"
            now = time.monotonic()
            if now >= deadline:
                if q is None:
                    return "stalled", f"no state read-back after {now - t0:.1f} s"
                joints, rail = self._arrival_error(q, goal)
                off = f"joints off by {joints * 1e3:.1f} mrad"
                if rail is not None and rail > PLAN_ARRIVAL_TOL_RAIL_M:
                    off += f", carriage off by {rail * 1e3:.0f} mm"
                return "stalled", f"did not arrive: {off} after {now - t0:.1f} s"
            time.sleep(PLAN_ARRIVAL_POLL_S)

    @staticmethod
    def _span(wps) -> float:
        """Largest |q - q[0]| over an arm's waypoints (rad / m; log line only)."""
        import numpy as np

        pts = [np.asarray(w, dtype=np.float64) for w in wps]
        return max((float(np.max(np.abs(q - pts[0]))) for q in pts[1:]), default=0.0)

    @staticmethod
    def _sequence_detail(detail: str, order: list[str], i: int) -> str:
        """Name the arm a multi-arm sequence stopped at and the arms that never
        started; a single-arm sequence reports exactly as before."""
        if len(order) < 2:
            return detail
        rest = [arm_label(a) for a in order[i + 1 :]]
        text = f"{detail} - {arm_label(order[i])}"
        return f"{text}; {', '.join(rest)} not moved" if rest else text

    def _submit_execute_plan(self, waypoints: dict, grippers: dict, *, interruptible: bool):
        """One ``execute_plan`` through the bus; returns the loop's ack."""
        return self.bus.commands.submit(
            Command(
                op="execute_plan",
                args={
                    "waypoints": waypoints,
                    "gripper": grippers,
                    "interruptible": interruptible,
                },
                source="internal",
            )
        ).result(timeout=5.0)

    def _execute_arms(
        self,
        session: ActiveSession,
        waypoints: dict,
        grippers: dict,
        *,
        interruptible: bool,
        running: Callable[[], bool],
        budget_s: Callable[[str, list], float],
        tag: str,
        timeout_reason: str,
        submit: Callable[[dict, dict], Any] | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> tuple[str, str]:
        """Execute a planned multi-arm motion ONE ARM AT A TIME in ``waypoints`` order
        (= ``PlanResult.arm_order``, see :meth:`_ordered_waypoints`) through the gated
        ``execute_plan`` path: submit one arm's waypoints, wait until the executor has
        retired them, then the next arm. The gripper targets ride the LAST arm that
        moves (an interruptible plan applies them on that arm's arrival). Any refusal /
        cancel / fault / timeout stops the sequence - the remaining arms never start.
        2026-09-08: a two-arm return submitted as ONE plan moved both arms at once
        through never-validated combinations and the gate held it at 5.2 mm.

        ``running()`` false stops the wait (the motion is cancelled through the loop);
        ``budget_s(arm, wps)`` is the per-arm deadline; ``submit(wps, grippers)`` may
        wrap :meth:`_submit_execute_plan` (start_from's retry-once); ``on_progress``
        receives the fraction of ALL waypoints retired so far. Returns ``(status,
        detail)`` with ``detail`` naming the arm for a multi-arm sequence:

        ``done`` - every arm reached its last waypoint AND its measured posture is
        there (``_await_arrival``: the executor retiring the waypoints only says the
        COMMAND arrived; a carriage trails its latest-wins targets at the track's speed);
        ``refused`` - the loop nacked an arm's plan (``detail`` = its reason, plus the
        arms that did / did not move when it was not the first arm);
        ``timeout`` - an arm ran past its budget, the motion was STOPPED through the
        loop (``detail`` = the budget);
        ``held`` - the loop's gate-held abort cancelled the plan (``detail`` = the
        blocking pair and distance, ``GATE_HOLD_PREFIX`` stripped);
        ``stalled`` - the command arrived but the measured arm did not settle at the
        goal within the arm's deadline (``detail`` = how far off, after how long); the
        remaining arms never start;
        ``cancelled`` - operator input / a driver fault / teardown interrupted it
        (``detail`` = ``loop.plan_cancel_reason``) and the arms hold where they are.

        A profile that moves no arm but sets a gripper still submits ONE gripper-only
        ``execute_plan`` (``waypoints={}``), so the targets are not dropped.
        """
        loop = session.loop
        plans = loop.plans
        submit = submit or (
            lambda wps, grip: self._submit_execute_plan(wps, grip, interruptible=interruptible)
        )
        order = self._moving_arms(waypoints)
        total = sum(len(w) for w in waypoints.values()) or 1
        retired = sum(len(waypoints[a]) for a in waypoints if a not in order)  # parked arms
        if not order:  # nothing moves: nothing to arrive at - only the gripper targets
            if grippers:
                ack = submit({}, grippers)
                if not ack.ok:
                    return "refused", ack.detail
            if on_progress is not None:
                on_progress(1.0)
            return "done", ""
        logger.info(
            "%s: executing one arm at a time in planner order %s (parked: %s)",
            tag,
            [
                f"{a} ({len(waypoints[a])} waypoints, max |dq| {self._span(waypoints[a]):.4f})"
                for a in order
            ],
            [a for a in waypoints if a not in order] or "none",
        )
        for i, arm_id in enumerate(order):
            wps = waypoints[arm_id]
            last = i == len(order) - 1
            if not running():  # the session left its motion state between two arms
                return "cancelled", self._sequence_detail(
                    f"session {session.state.value}", order, i
                )
            ack = submit({arm_id: wps}, grippers if last else {})
            if not ack.ok:
                # the first arm: nothing moved, the loop's words verbatim; a later arm:
                # say which arm was refused and that the earlier ones DID move
                detail = ack.detail if i == 0 else self._sequence_detail(ack.detail, order, i)
                return "refused", detail
            budget = float(budget_s(arm_id, wps))
            deadline = time.monotonic() + budget
            timed_out = False
            while plans.active_arms:
                if not running():
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                if on_progress is not None:
                    left = len(plans._waypoints.get(arm_id, ())) - plans._index.get(arm_id, 0)
                    on_progress(min(1.0, (retired + len(wps) - max(0, left)) / total))
                time.sleep(0.02)
            if plans.active_arms:
                # deadline / a fault / teardown: STOP the motion through the loop first
                why = timeout_reason if timed_out else f"session {session.state.value}"
                self._cancel_plan_via_loop(f"{tag}: {why}")
            if timed_out:
                return "timeout", self._sequence_detail(f"budget {budget:.1f} s", order, i)
            reason = loop.plan_cancel_reason
            if reason:
                if reason.startswith(GATE_HOLD_PREFIX):
                    pair = reason[len(GATE_HOLD_PREFIX) :].lstrip(": ")
                    return "held", self._sequence_detail(pair, order, i)
                return "cancelled", self._sequence_detail(reason, order, i)
            # The command is at the goal; now the ARM has to be (2026-09-08 review): the
            # next arm's path was validated against this arm AT its goal, and the gate
            # checks commanded postures, so a carriage still travelling would be invisible
            # to it. The wait shares the arm's budget, with at least the grace.
            status, detail = self._await_arrival(
                session,
                arm_id,
                wps[-1],
                running=running,
                deadline=max(deadline, time.monotonic() + PLAN_ARRIVAL_GRACE_S),
            )
            if status != "done":
                logger.warning("%s: %s %s - %s", tag, arm_label(arm_id), status, detail)
                return status, self._sequence_detail(detail, order, i)
            retired += len(wps)
            if on_progress is not None:
                on_progress(min(1.0, retired / total))
        return "done", ""

    def _run_return_plan(
        self, session: ActiveSession, waypoints: dict, grippers: dict, *, interruptible: bool
    ) -> tuple[str, str]:
        """Execute ONE planned return phase - one arm at a time in the planner's order
        (:meth:`_execute_arms`) - and wait for it; the per-arm budget is
        :meth:`_return_budget_s` of that arm's waypoints. Returns ``(status, detail)``
        as :meth:`_execute_arms` does (``done`` / ``refused`` / ``timeout`` / ``held``
        / ``stalled`` / ``cancelled``); the wait ends when the session leaves RUNNING."""
        return self._execute_arms(
            session,
            waypoints,
            grippers,
            interruptible=interruptible,
            running=lambda: session.state is SessionState.RUNNING,
            budget_s=lambda arm_id, wps: self._return_budget_s(session, {arm_id: wps}),
            tag="return-to-start",
            timeout_reason="return timed out",
        )

    def _return_home_worker(
        self, session: ActiveSession, profile, outcome: str, token: _ProfileMotion | None = None
    ) -> None:
        """The ``start_from`` machinery, per episode: ``twin.plan`` from the MEASURED
        posture to the profile, executed as an INTERRUPTIBLE ``execute_plan`` at the
        session's speed scale. Any operator input cancels it (the loop reports the
        reason: movement key / clutch / jog / arm switch) and the arm holds where it
        is; a planning failure produces no motion. ``EpisodeStatus.detail`` carries
        the outcome until the next ``episode_new``. ONE phase (joints and carriage
        together) — the two-phase split is the exit / reset path
        (:meth:`_return_to_initial_motion`), not the per-episode one. Both arms are
        executed one after the other in the planner's ``arm_order`` (module
        docstring)."""
        import numpy as np

        rt = session.recorder_thread
        try:
            states, q_start, _q_joint, q_goal, grippers = self._return_goals(
                session, profile, session.spec.arms
            )
            if not q_goal:
                rt.set_returning(
                    False, f"return skipped: profile '{profile.name}' covers no session arm"
                )
                return
            if all(np.allclose(q_start[a], q_goal[a], atol=1e-3) for a in q_goal):
                rt.set_returning(False, "")  # already at the return profile
                return
            result = self._plan_return(session, states, q_start, q_goal)
            if not result.ok:
                failure = self._plan_failure_text(result)
                logger.warning("return-to-start plan failed: %s", failure)
                rt.set_returning(False, f"return failed: {failure} - arm holds")
                return
            if session.state is not SessionState.RUNNING:
                rt.set_returning(False, f"return skipped: session {session.state.value}")
                return
            status, detail = self._run_return_plan(
                session, self._ordered_waypoints(result), grippers, interruptible=True
            )
            if status == "refused":
                rt.set_returning(False, f"return refused: {detail}")
            elif status == "timeout":
                rt.set_returning(False, "return timed out - held by the gate; arm stopped")
                logger.warning("return-to-start after %s timed out (%s)", outcome, detail)
            elif status == "held":  # the loop's gate-held abort (plan_gate_hold_s)
                rt.set_returning(False, f"return stopped - {GATE_HOLD_PREFIX}: {detail}")
                logger.warning("return-to-start after %s held by the gate: %s", outcome, detail)
            elif status == "stalled":  # the command arrived, the measured arm did not
                rt.set_returning(False, f"return stopped - {detail}")
                logger.warning("return-to-start after %s stalled: %s", outcome, detail)
            elif status == "cancelled":
                rt.set_returning(False, f"return cancelled: {detail}")
                logger.info("return-to-start after %s cancelled: %s", outcome, detail)
            else:
                rt.set_returning(False, "")
                session.motion_detail = ""  # arrived at the return profile: notice resolved
                logger.info("return-to-start after %s complete (profile %r)", outcome, profile.name)
        except Exception as e:
            logger.exception("return-to-start worker failed")
            try:
                rt.set_returning(False, f"return failed: {e!r}")
            except Exception:  # noqa: BLE001
                pass
        finally:
            self._release_profile_motion(token)

    def _return_budget_s(self, session: ActiveSession, waypoints: dict) -> float:
        """Executor-time estimate of the return (every joint incl. the rail slot at the
        loop's slew) x 3 for gate holds, min 30 s — like ``rail_homing``'s
        ``plan_duration_s``, never a flat 120 s."""
        import numpy as np

        cfg = session.loop.cfg.jog
        rate = 1.0 / session.loop.dt if session.loop.dt > 0 else 100.0
        longest = 0.0
        for wps in waypoints.values():
            qs = [np.asarray(list(w), dtype=np.float64) for w in wps]
            ticks = 0.0
            for a, b in zip(qs[:-1], qs[1:], strict=False):
                dq = np.abs(b - a)
                limit = np.full(dq.shape, float(cfg.slew_rad_per_tick))
                if dq.shape[0] > 7:
                    limit[7:] = float(cfg.rail_m_per_tick)
                ticks += float(np.max(dq / limit)) if dq.size else 0.0
            longest = max(longest, ticks / rate)
        return max(30.0, 3.0 * longest + 10.0)

    def _cancel_plan_via_loop(self, reason: str) -> None:
        session = self.session
        if session is None or not session.loop.motion_active:
            return
        try:
            self.bus.commands.submit(
                Command(op="cancel_plan", args={"reason": reason}, source="internal")
            ).result(timeout=1.0)
        except Exception:  # noqa: BLE001 - the loop may already be stopping
            logger.warning("cancel_plan (%s) did not complete within 1 s", reason)

    # -- return to the initial condition: the `R` key and the Cockpit's exit ----------------
    # 2026-09-08 (operator request). ONE motion, two entry points:
    #   * `reset_to_initial` (key `R`) — fire-and-forget, the ack only says it started;
    #   * POST /api/session/return_home — synchronous, the Cockpit runs it BEFORE it
    #     tears the session down and shows a dialog when it does not arrive.
    # Both walk to the workcell kind's designated initial-condition profile and both
    # are no-ops (a reason, no motion) when no initial condition is designated. The
    # motion is twin-planned, gated like every other command and INTERRUPTIBLE: any
    # movement key / clutch / jog / arm switch cancels it and the arms hold.
    def _initial_profile(self, kind: str) -> ReturnHomeResult | Any:
        """This kind's designated initial-condition profile, or the terminal
        ``ReturnHomeResult`` to report instead. A missing designation is a SUCCESS
        (``skipped``): the operator has simply never set one, and both entry points
        are specified to do nothing then."""
        try:
            profile = self.profile_store.initial_for(kind)  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001 - an unreadable store is a refusal, not a 500
            return ReturnHomeResult(
                ok=False, status="failed", detail=f"profile store unreadable: {e}"
            )
        if profile is None:
            return ReturnHomeResult(
                ok=True,
                status="skipped",
                detail=(
                    f"no initial condition designated for the {kind} workcell - save a "
                    "profile with 'use as initial condition' to enable this"
                ),
            )
        return profile

    def _reset_blockers(self, session: ActiveSession, profile) -> ReturnHomeResult | None:
        """What forbids the motion once a session and a target profile exist; None =
        go ahead. Shared by both entry points so they cannot refuse differently."""
        result = ReturnHomeResult(
            ok=False, status="refused", detail="", profile_id=profile.profile_id
        )
        if self.open_episode() is not None:
            # ``open_episode_id`` is set while recording AND while saving (the store
            # keeps the directory open until the writer is done): say which.
            if self._recorder_state(session) == "saving":
                return result.model_copy(
                    update={"detail": "an episode is still saving - wait for it to finish"}
                )
            return result.model_copy(
                update={"detail": "an episode is still recording - save or discard it first"}
            )
        if session.state is not SessionState.RUNNING:
            return result.model_copy(
                update={
                    "status": "failed",
                    "detail": f"the session is {session.state.value}, not running",
                }
            )
        faulted = sorted(session.loop.faulted_arms)
        recovering = sorted(session.loop.recovering_arms - session.loop.faulted_arms)
        if faulted:
            names = " / ".join(arm_label(a) for a in faulted)
            return result.model_copy(
                update={
                    "status": "failed",
                    "detail": f"{names} is faulted - clear the error and resume first",
                }
            )
        if recovering:
            # RECOVERING is lifted by releasing every live input, not by clearing errors
            names = " / ".join(arm_label(a) for a in recovering)
            return result.model_copy(
                update={
                    "status": "failed",
                    "detail": f"{names} is recovering - release every input (clutch / keys) first",
                }
            )
        if session.loop.motion_active or self._profile_motion is not None:
            return result.model_copy(update={"detail": MOTION_BUSY})
        return None

    @staticmethod
    def _recorder_state(session: ActiveSession) -> str:
        """``EpisodeStatus.state`` of the session's recorder ("idle" without one)."""
        rt = session.recorder_thread
        status = getattr(rt, "status", None)
        if rt is None or not callable(status):
            return "idle"
        try:
            return str(status().state)
        except Exception:  # noqa: BLE001 - a recorder hiccup must not hide the refusal
            return "idle"

    def request_reset_to_initial(self, profile) -> tuple[bool, str]:
        """``ControlLoop.on_reset_to_initial`` (LOOP THREAD): validate, then hand the
        motion to a worker thread. Returns the ack ``(ok, detail)`` immediately — the
        loop must never block on planning. ``profile`` is the initial-condition profile
        the loop already resolved, so this does no disk I/O on the loop thread."""
        return self._start_profile_motion(
            profile, ack=f"returning to '{profile.name}'", label="reset_to_initial"
        )

    def request_goto_profile(self, profile) -> tuple[bool, str]:
        """``ControlLoop.on_goto_profile`` (LOOP THREAD; ``ActionMsg goto_profile``,
        2026-09-08): the reset-to-initial motion toward a CHOSEN saved profile instead
        of the designated initial condition — the same two twin-planned, gated,
        INTERRUPTIBLE phases (joints with the carriages held, then the carriages when
        the profile stores a rail slot), the same blockers (an open episode, a faulted
        / recovering arm, a plan already running). The loop resolved ``profile`` from
        the store; a profile of another workcell kind is refused here as well."""
        session = self.session
        if session is not None and profile.workcell_kind != session.spec.kind:
            return False, f"profile '{profile.name}' is for the {profile.workcell_kind} workcell"
        return self._start_profile_motion(
            profile, ack=f"going to profile '{profile.name}'", label="goto_profile"
        )

    def _start_profile_motion(self, profile, *, ack: str, label: str) -> tuple[bool, str]:
        """Shared by the two fire-and-forget entry points: the blockers, then the
        worker thread; ``ack`` is the success detail, ``label`` names the op in logs."""
        session = self.session
        if session is None:
            return False, "no active session"
        blocked = self._reset_blockers(session, profile)
        if blocked is not None:
            return blocked.ok, blocked.detail
        token = self._claim_profile_motion(label)
        if token is None:  # lost the race against another motion's claim
            return False, MOTION_BUSY
        session.motion_detail = ""  # this motion's outcome replaces the last notice
        threading.Thread(
            target=self._profile_motion_worker,
            args=(session, profile, label, token),
            name=label.replace("_", "-"),
            daemon=True,
        ).start()
        return True, ack

    def _profile_motion_worker(
        self, session: ActiveSession, profile, label: str, token: _ProfileMotion
    ) -> None:
        try:
            self._profile_motion_reported(session, profile, label)
        finally:
            self._release_profile_motion(token)

    def _profile_motion_reported(
        self, session: ActiveSession, profile, label: str
    ) -> ReturnHomeResult:
        """Run the two-phase motion and leave its outcome where the operator can see
        it: ``session.motion_detail`` (telemetry ``session.fault_detail``, the Cockpit's
        SESSION banner row) carries every outcome but an arrival - the fire-and-forget
        entry points (`R`, Go to profile) ack "started" and would otherwise stay silent
        when the twin cannot plan the path or the operator's own key cancels it."""
        try:
            result = self._return_to_initial_motion(session, profile, label=label)
        except Exception as e:  # noqa: BLE001 - a worker bug must not kill the session
            logger.exception("%s worker failed", label)
            result = ReturnHomeResult(
                ok=False, status="failed", detail=repr(e), profile_id=profile.profile_id
            )
        session.motion_detail = (
            "" if result.status == "done" else f"{_motion_title(label, profile)}: {result.detail}"
        )
        log = logger.info if result.ok else logger.warning
        log("%s: %s%s", label, result.status, f" - {result.detail}" if result.detail else "")
        return result

    def return_to_initial(self) -> ReturnHomeResult:
        """``POST /api/session/return_home``: run the motion SYNCHRONOUSLY (the caller
        is a REST request the Cockpit awaits before it tears the session down) and
        report where the arms ended up. Never raises for an operational refusal — the
        UI branches on ``ok`` and shows ``detail``."""
        session = self.session
        if session is None:
            return ReturnHomeResult(ok=False, status="refused", detail="no active session")
        profile = self._initial_profile(session.spec.kind)
        if isinstance(profile, ReturnHomeResult):
            return profile
        blocked = self._reset_blockers(session, profile)
        if blocked is not None:
            return blocked
        token = self._claim_profile_motion("return_home")
        if token is None:
            return ReturnHomeResult(
                ok=False, status="refused", detail=MOTION_BUSY, profile_id=profile.profile_id
            )
        try:
            session.motion_detail = ""
            return self._profile_motion_reported(session, profile, "return_home")
        finally:
            self._release_profile_motion(token)

    # -- episode playback (2026-09-10; 04-runtime §10.8, 05-ui §8.1 item 7) ------------
    # Operator request: a Playback button on every episode row of the Welcome page's
    # Datasets panel, opening a dialog with "return to this episode's initial state" and
    # "play back the whole episode", the second disabled until the first has run. Both
    # motions are ordinary twin-planned, gated, interruptible profile motions - the
    # trajectory replay is built on top of the same machinery, so nothing here can move
    # an arm the gate has not approved.
    def _playback_trajectory(self, repo_id: str, episode_id: str):
        """Read one episode's measured trajectory; ``DatasetError`` / ``PlaybackError``
        carry an operator-facing reason for the REST layer to turn into 404 / 409."""
        from ..recorder.playback import PlaybackError, load_trajectory

        layout = self.dataset_store.layout_of(repo_id)
        if layout is None:
            raise DatasetError(f"unknown dataset {repo_id!r}", not_found=True)
        if layout != "episode_dirs":  # the only other value is "lerobot_v3" (legacy)
            raise PlaybackError(
                f"dataset {repo_id!r} is a legacy LeRobot tree - playback needs the "
                "episode-directory layout (10-frames §11)"
            )
        return load_trajectory(repo_id, self.dataset_store.root_of(repo_id), episode_id)

    def _playback_refusal(self, traj) -> str:
        """Why this episode cannot be replayed onto the LIVE cell right now; "" = it can.

        Deliberately NOT a check of the workcell kind: the sim twin and the real cell
        share their kinematics, so replaying a hardware recording in sim (the sensible
        way to preview one before it touches the arms) is allowed. What must match is the
        ARM SET - an episode that names an arm this session does not drive cannot be
        placed at all.
        """
        session = self.session
        if session is None:
            return (
                "Start a session first - playback drives the arms through the twin "
                "planner and the safety gate, which only exist inside a session"
            )
        missing = [a for a in traj.arm_ids if a not in session.spec.arms]
        if missing:
            names = " / ".join(arm_label(a) for a in missing)
            return f"the episode was recorded with {names}, which this session does not drive"
        return ""

    def episode_playback_info(self, repo_id: str, episode_id: str):
        """``GET /api/datasets/{ns}/{name}/episodes/{id}/playback``: what a playback of
        this episode would do, plus whether it can run right now and why not.

        Session-less on purpose - the dialog opens and explains itself before anything
        moves, so the operator never meets a bare 409.
        """
        from apollo_mavis_v2_core.protocol import EpisodePlaybackInfo

        traj = self._playback_trajectory(repo_id, episode_id)
        reason = self._playback_refusal(traj)
        return EpisodePlaybackInfo(
            repo_id=repo_id,
            episode_id=episode_id,
            frames=traj.frames,
            fps=traj.fps,
            duration_s=traj.duration_s,
            arms=list(traj.initial_state().values()),
            playable=not reason,
            reason=reason,
            sources=traj.sources,
            action_space=traj.action_space,
        )

    def _episode_initial_profile(self, session: ActiveSession, traj) -> StateProfile:
        """A TRANSIENT profile holding the episode's first frame.

        It is never saved: giving the goto a ``StateProfile`` is what lets it reuse the
        whole return-to-initial path unchanged - two separately planned and gated phases
        (joints with the carriages held, then the carriages), one arm at a time in the
        planner's ``arm_order``, cancelled by any operator input.
        """
        state = traj.initial_state()
        return StateProfile(
            name=f"episode {traj.episode_id}'s initial state",
            notes=(
                f"Frame 0 of {traj.repo_id} episode {traj.episode_id} "
                f"({traj.frames} frames at {traj.fps:g} fps). Transient - never stored."
            ),
            workcell_kind=session.spec.kind,  # type: ignore[arg-type]
            arms={
                arm_id: ArmPosture(
                    q=list(st.q),
                    rail_pos_m=st.rail_pos_m,
                    gripper_open_frac=(
                        1.0
                        if st.gripper_open_frac is None
                        else min(1.0, max(0.0, st.gripper_open_frac))
                    ),
                )
                for arm_id, st in state.items()
                if arm_id in session.spec.arms
            },
        )

    def episode_playback_goto_initial(self, repo_id: str, episode_id: str) -> ReturnHomeResult:
        """``POST /api/session/playback {action: "goto_initial"}``: walk the arms to the
        episode's FIRST recorded frame, SYNCHRONOUSLY.

        Synchronous like ``return_home`` because the dialog awaits it and only then
        enables **Playback** - the operator's rule, and the right one: replaying a
        trajectory from the wrong place is exactly how you drive an arm into something.
        Every refusal is ``ok: false`` + ``detail``, never an exception.
        """
        from ..recorder.playback import PlaybackError

        session = self.session
        if session is None:
            return ReturnHomeResult(ok=False, status="refused", detail="no active session")
        try:
            traj = self._playback_trajectory(repo_id, episode_id)
        except (DatasetError, PlaybackError) as e:
            return ReturnHomeResult(ok=False, status="refused", detail=str(e))
        reason = self._playback_refusal(traj)
        if reason:
            return ReturnHomeResult(ok=False, status="refused", detail=reason)
        profile = self._episode_initial_profile(session, traj)
        blocked = self._reset_blockers(session, profile)
        if blocked is not None:
            return blocked
        token = self._claim_profile_motion(PLAYBACK_INITIAL_LABEL)
        if token is None:
            return ReturnHomeResult(ok=False, status="refused", detail=MOTION_BUSY)
        try:
            session.motion_detail = ""
            return self._profile_motion_reported(session, profile, PLAYBACK_INITIAL_LABEL)
        finally:
            self._release_profile_motion(token)

    def _verify_playback(self, session: ActiveSession, plan) -> str:
        """Check EVERY posture of a resampled playback in the twin, all arms jointly; ""
        = clear, else the operator-facing reason naming the first violating waypoint.

        This is what earns :meth:`ControlLoop._op_playback_path` the right to move both
        arms at once (see its docstring). It is checked at the resolution the postures
        will be COMMANDED - one waypoint per tick - not at the recorded frame rate, so
        there is no un-checked interpolation between two approved postures. The model's
        own inflation applies, i.e. the cell's raised 25 mm shell on hardware: this is
        the same twin the gate uses, so it cannot disagree with it.

        **It must NOT touch the twin's own ``MjData``** (2026-09-10, learned the hard
        way): that belongs to the 100 Hz control loop, which is gating ticks through
        ``twin.check`` while this runs on a REST thread. Writing qpos and running
        ``mj_collision`` on it from here corrupts MuJoCo's contact bookkeeping and
        raises ``mujoco.FatalError: collisionTask: collision function returned 0
        contacts for geom pair (22, 24), expected at most -75 from mj_maxContact`` - a
        negative budget, i.e. an inconsistent contact buffer, and the reason a real
        hardware playback died at waypoint 244. So the whole verification runs on a
        PRIVATE ``MjData`` (``twin.new_data()``), the same way every planner worker
        does, and reads shared state exactly once: one snapshot of the measured qpos for
        the non-arm degrees of freedom (props), whose slots nothing else writes.

        A recorded trajectory can legitimately fail this: the episode was recorded with
        the objects present, and the twin does not model them (03-sim §4.5), so a grasp
        that was safe in the room may read as a collision against a bare table - and the
        reverse, which is why the live gate stays the authority per tick.
        """
        twin = session.twin
        if twin is None:
            return ""  # sim NullGate session: no twin to verify against (11-safety §5)
        try:
            data = twin.new_data()
            base = np.array(twin.data.qpos, dtype=np.float64)
        except Exception as e:  # noqa: BLE001 - an older sim without new_data(): say so
            logger.exception("playback verification could not take a private twin data")
            return f"the digital twin could not be prepared for verification: {e}"
        slots = {}
        for arm_id in plan.waypoints:
            try:
                slots[arm_id] = twin.addr[arm_id].qpos_adr
            except Exception:  # noqa: BLE001
                return f"the digital twin does not model arm {arm_id!r}"
            want, got = len(slots[arm_id]), len(plan.waypoints[arm_id][0])
            if want != got:
                return (
                    f"the episode gives {got} values for {arm_label(arm_id)} but the twin "
                    f"needs {want} (a recording made with a different rail fitment?)"
                )
        rate = max(1.0, float(session.loop.cfg.rate_hz))
        for k, posture in plan.postures():
            q_full = base.copy()
            for arm_id, q in posture.items():
                q_full[slots[arm_id]] = q
            try:
                violations = twin.check_config_violations(q_full, data=data)
            except Exception as e:  # noqa: BLE001 - a twin problem is a refusal, not a 500
                logger.exception("playback verification failed at waypoint %d", k)
                return f"the digital twin could not check waypoint {k}: {e}"
            if violations:
                pairs = ", ".join(" <-> ".join(p) for p, _ in violations[:2])
                # Waypoint -> seconds: the resampler emits one waypoint per loop tick.
                return (
                    f"the digital twin blocks this episode {k / rate:.1f} s in "
                    f"(waypoint {k} of {plan.length}): {pairs}. The twin does not model "
                    "the objects the episode was recorded with, so a grasp can read as a "
                    "collision here - re-measure the cell or replay it in sim first"
                )
        return ""

    def episode_playback_play(
        self, repo_id: str, episode_id: str, source: str = "state"
    ) -> ReturnHomeResult:
        """``POST /api/session/playback {action: "play", source}``: replay the episode.

        SYNCHRONOUS, like ``goto_initial`` and ``return_home``: the dialog awaits it and
        shows the outcome. Interruptible throughout - any operator input, a driver fault
        or a teardown cancels it and the arms hold where they are.

        ``source: "state"`` (the default; this path is unchanged since 2026-09-10) replays
        the MEASURED joint / rail / gripper trajectory (``observation.state``), resampled
        onto the loop tick with ONE global time scale so the arms keep their recorded
        relative timing, verified posture by posture in the twin, then streamed through
        the gated executor as one ``playback_path`` command.

        ``source: "delta_ee" | "abs_ee"`` (2026-09-11) replays the recorded ``action`` /
        ``action.abs_ee`` column through the executor path a policy drives
        (``dagger/step.policy_step`` over a ``ReplayActionSource`` + ``ActionAnchor``)
        inside this session's control loop (``replay_actions``). There is no pre-planned
        joint path, so ``_verify_playback`` does not apply: every row is gated on the tick
        it is due, and a gate hold longer than ``plan_gate_hold_s`` cancels it with the
        pair. Admitted in SIM only until the operator says otherwise (hardware answers
        ``ok: false``), refused when the episode lacks the column (naming the backfill),
        and refused in dagger / inference sessions (their loop drives its own policy).
        The outcome names the terminal residual against the last measured frame - the
        executor-fidelity metric these two sources exist for.

        Neither path re-places the arms first: the operator's dialog requires "return to
        the initial state" to have run, and doing it again silently here would hide a
        cell that has drifted since. The starting posture is instead CHECKED against
        frame 0 below, and a mismatch is refused with the distance.
        """
        from ..recorder.playback import ExecutorCaps, PlaybackError, resample

        session = self.session
        if session is None:
            return ReturnHomeResult(ok=False, status="refused", detail="no active session")
        try:
            traj = self._playback_trajectory(repo_id, episode_id)
        except (DatasetError, PlaybackError) as e:
            return ReturnHomeResult(ok=False, status="refused", detail=str(e))
        reason = self._playback_refusal(traj)
        if reason:
            return ReturnHomeResult(ok=False, status="refused", detail=reason)
        if source != "state":
            reason = self._action_replay_refusal(session, traj, source)
            if reason:
                return ReturnHomeResult(ok=False, status="refused", detail=reason)
        profile = self._episode_initial_profile(session, traj)
        blocked = self._reset_blockers(session, profile)
        if blocked is not None:
            return blocked
        arms = [a for a in traj.arm_ids if a in session.spec.arms]
        states = session.workcell.states()
        # An arm whose recording has no rail column keeps its carriage where it is.
        rail_hold = {
            a: float(getattr(states[a], "rail_pos_m", 0.0) or 0.0) for a in arms if a in states
        }
        # The arms must already BE at frame 0 (the dialog's rule). Refuse with the
        # distance rather than quietly walking there: a drifted cell is news.
        start = traj.initial_state()
        for arm_id in arms:
            measured = np.asarray(states[arm_id].q, dtype=np.float64)[:7]
            want = np.asarray(start[arm_id].q, dtype=np.float64)
            off = float(np.max(np.abs(measured - want)))
            if off > PLAYBACK_START_TOL_RAD:
                return ReturnHomeResult(
                    ok=False,
                    status="refused",
                    detail=(
                        f"{arm_label(arm_id)} is {math.degrees(off):.1f}deg away from the "
                        "episode's first frame - run 'Return to the initial state' first"
                    ),
                )
        plan = None
        if source == "state":
            try:
                plan = resample(
                    traj,
                    ExecutorCaps.from_jog(session.loop.cfg.jog),
                    loop_hz=session.loop.cfg.rate_hz,
                    arms=arms,
                    rail_hold=rail_hold,
                )
            except PlaybackError as e:
                return ReturnHomeResult(ok=False, status="refused", detail=str(e))
        token = self._claim_profile_motion(PLAYBACK_LABEL)
        if token is None:
            return ReturnHomeResult(ok=False, status="refused", detail=MOTION_BUSY)
        try:
            session.motion_detail = ""
            if source == "state":
                result = self._playback_motion(session, traj, plan)
            else:
                result = self._replay_motion(session, traj, source, arms, rail_hold)
        except Exception as e:  # noqa: BLE001 - a worker bug must not kill the session
            logger.exception("episode playback failed")
            result = ReturnHomeResult(ok=False, status="failed", detail=repr(e))
        finally:
            self._release_profile_motion(token)
        session.motion_detail = (
            "" if result.status == "done" else f"Playback of episode {episode_id}: {result.detail}"
        )
        log = logger.info if result.ok else logger.warning
        log("episode playback %s: %s - %s", episode_id, result.status, result.detail)
        return result

    def _playback_motion(self, session: ActiveSession, traj, plan) -> ReturnHomeResult:
        """Verify, submit and wait; the status vocabulary of :meth:`_execute_arms`."""
        arms = sorted(plan.waypoints)
        base = ReturnHomeResult(ok=False, status="failed", detail="", arms=arms)
        blocked = self._verify_playback(session, plan)
        if blocked:
            return base.model_copy(update={"status": "refused", "detail": blocked})
        loop, plans = session.loop, session.loop.plans
        ack = self.bus.commands.submit(
            Command(
                op="playback_path",
                args={"waypoints": plan.waypoints, "gripper_track": plan.gripper},
                source="internal",
            )
        ).result(timeout=5.0)
        if not ack.ok:
            return base.model_copy(update={"status": "refused", "detail": ack.detail})
        # Budget: the replay's own length plus generous slack for gate holds. `slowdown`
        # is already folded into the waypoint count, so this scales with the real motion.
        expected_s = plan.length / max(1.0, session.loop.cfg.rate_hz)
        budget = expected_s * PLAYBACK_BUDGET_FACTOR + PLAYBACK_BUDGET_GRACE_S
        deadline = time.monotonic() + budget
        timed_out = False
        while plans.active_arms:
            if session.state is not SessionState.RUNNING:
                break
            if time.monotonic() >= deadline:
                timed_out = True
                break
            time.sleep(0.02)
        if plans.active_arms:
            why = f"budget {budget:.0f} s" if timed_out else f"session {session.state.value}"
            self._cancel_plan_via_loop(f"playback: {why}")
        if timed_out:
            return base.model_copy(update={"status": "timeout", "detail": f"budget {budget:.0f} s"})
        cancel = loop.plan_cancel_reason
        if cancel:
            if cancel.startswith(GATE_HOLD_PREFIX):
                pair = cancel[len(GATE_HOLD_PREFIX) :].lstrip(": ")
                return base.model_copy(update={"status": "held", "detail": pair})
            return base.model_copy(update={"status": "cancelled", "detail": cancel})
        # The command reached the last waypoint; now the ARM has to be there — the same
        # rule _execute_arms follows, because the executor retiring waypoints only says
        # the COMMAND arrived, and a carriage trails its latest-wins targets at the
        # track's own speed. Without this a replay reports "done" while an arm still moves.
        for arm_id in arms:
            status, detail = self._await_arrival(
                session,
                arm_id,
                plan.waypoints[arm_id][-1],
                running=lambda: session.state is SessionState.RUNNING,
                deadline=time.monotonic() + max(PLAN_ARRIVAL_GRACE_S, expected_s),
            )
            if status != "done":
                return base.model_copy(
                    update={"status": status, "detail": f"{arm_label(arm_id)}: {detail}"}
                )
        slower = f" ({plan.slowdown:.1f}x slower than recorded)" if plan.slowdown > 1.05 else ""
        return base.model_copy(
            update={
                "ok": True,
                "status": "done",
                "detail": (f"replayed {traj.frames} frames in {expected_s:.1f} s{slower}"),
            }
        )

    def _action_replay_refusal(self, session: ActiveSession, traj, source: str) -> str:
        """Why an ACTION replay (``delta_ee`` / ``abs_ee``) cannot run here; "" = it can.
        Checked before the shared blockers so the operator reads the specific reason."""
        from ..recorder.playback import ACTION_COLUMNS, PlaybackError

        if source not in ACTION_COLUMNS:
            return f"unknown playback source {source!r} (state, delta_ee or abs_ee)"
        if session.spec.kind == "hardware" and not self.cfg.hardware_session.policy_modes:
            # The per-tick gate is the only check an action replay gets - no whole-path
            # twin verification - and the executor's fidelity is what it MEASURES. The real
            # arms get it only with hardware_session.policy_modes (operator decision
            # 2026-09-12; the lab render knob HARDWARE_POLICY_MODES).
            return "action replay is admitted in sim only (hardware_session.policy_modes is false)"
        if session.spec.mode in ("dagger", "inference"):
            return (
                "action replay runs in teleop / collect sessions only - this session's loop "
                "drives its own policy"
            )
        try:
            traj.action_column(source)
        except PlaybackError as e:
            return str(e)
        if session.loop.ik is None or session.loop.kin is None:
            return "this session has no IK / kinematics to drive Cartesian actions with"
        return ""

    def _replay_motion(
        self,
        session: ActiveSession,
        traj,
        source: str,
        arms: list[str],
        rail_hold: dict[str, float],
    ) -> ReturnHomeResult:
        """Submit ``replay_actions`` and wait; the status vocabulary of :meth:`_execute_arms`.

        Mirrors :meth:`_playback_motion` - the same budget rule, the same cancel path, the
        same measured-arrival wait against the LAST recorded frame - with two differences:
        nothing is verified up front (there is no path yet), and the outcome carries the
        residual against that last frame (max joint / carriage error, and for an episode
        with an ``action.abs_ee`` column the TCP error against the last COMMANDED pose),
        because that residual is the point of replaying an action column at all.
        """
        from ..dagger.policy_runner import ActionAnchor, SlewLimits, anchor_leash_kwargs
        from ..dagger.replay_source import ReplayActionSource
        from ..recorder.playback import PlaybackError

        base = ReturnHomeResult(ok=False, status="failed", detail="", arms=sorted(arms))
        loop = session.loop
        states = session.workcell.states()
        arms_meta = [(a, len(states[a].q) > 7) for a in session.spec.arms if a in states]
        try:
            replay = ReplayActionSource(
                traj, source, arms_meta, clock=loop._clock, rail_hold=rail_hold
            )
        except PlaybackError as e:
            return base.model_copy(update={"status": "refused", "detail": str(e)})
        anchor = ActionAnchor(
            loop.ik,
            loop.kin,
            SlewLimits(window_s=self.cfg.dagger.slew_window_s),
            action_space=source,
            **anchor_leash_kwargs(self.cfg.dagger, self.cfg.control),
        )
        ack = self.bus.commands.submit(
            Command(
                op="replay_actions",
                args={"source": replay, "anchor": anchor},
                source="internal",
            )
        ).result(timeout=5.0)
        if not ack.ok:
            return base.model_copy(update={"status": "refused", "detail": ack.detail})
        expected_s = traj.frames / max(1e-6, traj.fps)
        budget = expected_s * PLAYBACK_BUDGET_FACTOR + PLAYBACK_BUDGET_GRACE_S
        deadline = time.monotonic() + budget
        timed_out = False
        while loop.motion_active:
            if session.state is not SessionState.RUNNING:
                break
            if time.monotonic() >= deadline:
                timed_out = True
                break
            time.sleep(0.02)
        if loop.motion_active:
            why = f"budget {budget:.0f} s" if timed_out else f"session {session.state.value}"
            self._cancel_plan_via_loop(f"playback: {why}")
        if timed_out:
            return base.model_copy(update={"status": "timeout", "detail": f"budget {budget:.0f} s"})
        cancel = loop.plan_cancel_reason
        if cancel:
            if cancel.startswith(GATE_HOLD_PREFIX):
                pair = cancel[len(GATE_HOLD_PREFIX) :].lstrip(": ")
                return base.model_copy(update={"status": "held", "detail": pair})
            return base.model_copy(update={"status": "cancelled", "detail": cancel})
        # The rows ran out; now the ARMS have to be where the recording ended. Wait for
        # each arm to settle (arrival in joint space is not required - see REPLAY_TCP_TOL_M),
        # then judge the measured TCP against the last recorded frame's TCP. That verdict IS
        # the fidelity report, so a miss is reported with the distance.
        goals = self._replay_goals(traj, arms, rail_hold, arms_meta)
        settle_by = time.monotonic() + max(PLAN_ARRIVAL_GRACE_S, expected_s)
        for arm_id in arms:
            status, detail = self._await_settle(
                session,
                arm_id,
                goals[arm_id],
                running=lambda: session.state is SessionState.RUNNING,
                deadline=settle_by,
            )
            if status == "cancelled":
                return base.model_copy(
                    update={"status": status, "detail": f"{arm_label(arm_id)}: {detail}"}
                )
        residuals = self._replay_residuals(session, traj, arms, goals)
        report = "; ".join(text for _, text, _ in residuals)
        summary = f"replayed {traj.frames} {source} rows in {expected_s:.1f} s; {report}"
        misses = [arm_label(a) for a, _, ok in residuals if not ok]
        if misses:
            return base.model_copy(
                update={
                    "status": "stalled",
                    "detail": f"{' / '.join(misses)} did not reach the last frame - {summary}",
                }
            )
        return base.model_copy(update={"ok": True, "status": "done", "detail": summary})

    def _await_settle(
        self,
        session: ActiveSession,
        arm_id: str,
        goal,
        *,
        running: Callable[[], bool],
        deadline: float,
    ) -> tuple[str, str]:
        """Wait until ``arm_id`` has ARRIVED at ``goal`` (the plan rule) or has SETTLED (no
        joint / carriage moved more than ``REPLAY_SETTLE_EPS`` over ``REPLAY_SETTLE_S``) -
        ``("done", "")`` / ``("settled", "")``; ``("cancelled", ...)`` when the session leaves
        RUNNING or the arm faults; ``("stalled", "")`` at ``deadline``. Never commands."""
        last_q = None
        still_since = None
        while True:
            try:
                q = np.asarray(session.workcell.states()[arm_id].q, dtype=np.float64)
            except Exception:  # noqa: BLE001 - a read hiccup is not a settle
                q = None
            now = time.monotonic()
            if q is not None:
                if self._arrived(q, goal):
                    return "done", ""
                if last_q is not None and np.max(np.abs(q - last_q)) <= REPLAY_SETTLE_EPS:
                    if still_since is None:
                        still_since = now
                    elif now - still_since >= REPLAY_SETTLE_S:
                        return "settled", ""
                else:
                    still_since = None
                last_q = q
            if not running():
                return "cancelled", f"session {session.state.value}"
            if arm_id in session.loop.faulted_arms:
                return "cancelled", "driver fault"
            if now >= deadline:
                return "stalled", ""
            time.sleep(PLAN_ARRIVAL_POLL_S)

    @staticmethod
    def _replay_goals(traj, arms, rail_hold, arms_meta) -> dict[str, list[float]]:
        """Per arm, the LAST ``observation.state`` row as a full q (rail slot from the
        recording, else the held carriage position, only for a railed session arm)."""
        last = traj.state_at(traj.frames - 1)
        railed = dict(arms_meta)
        goals: dict[str, list[float]] = {}
        for arm_id in arms:
            st = last[arm_id]
            goal = [float(v) for v in st.q]
            if railed.get(arm_id):
                rail = st.rail_pos_m if st.rail_pos_m is not None else rail_hold.get(arm_id)
                if rail is not None:
                    goal.append(float(rail))
            goals[arm_id] = goal
        return goals

    def _replay_residuals(
        self, session: ActiveSession, traj, arms, goals
    ) -> list[tuple[str, str, bool]]:
        """Per arm ``(arm_id, text, within_tolerance)``: the max joint error (mrad), the
        carriage error (mm), the measured TCP against the last recorded frame's TCP (mm /
        mrad; the verdict, ``REPLAY_TCP_TOL_M`` / ``_RAD``) and, when the episode carries
        ``action.abs_ee``, against the last COMMANDED pose - the executor-fidelity metric.
        A recording without TCP dims falls back to the joint verdict."""
        from apollo_mavis_v2_core import Pose, se3

        try:
            states = session.workcell.states()
        except Exception:  # noqa: BLE001 - a read hiccup only costs the report
            return [(a, f"{arm_label(a)}: no state read-back", False) for a in arms]
        abs_col = traj.actions.get("abs_ee")
        out: list[tuple[str, str, bool]] = []
        for arm_id in arms:
            st = states.get(arm_id)
            if st is None:
                out.append((arm_id, f"{arm_label(arm_id)}: no state", False))
                continue
            joints, rail = self._arrival_error(st.q, goals[arm_id])
            ok = self._arrived(st.q, goals[arm_id])
            text = f"{arm_label(arm_id)} joints {joints * 1e3:.1f} mrad"
            if rail is not None:
                text += f", carriage {rail * 1e3:.1f} mm"
            recorded = traj.tcp_at(traj.frames - 1, arm_id)
            if recorded is not None:
                dp, dr = se3.pose_error(recorded, st.ee_pose)
                ok = dp <= REPLAY_TCP_TOL_M and dr <= REPLAY_TCP_TOL_RAD
                text += f", TCP {dp * 1e3:.1f} mm / {dr * 1e3:.0f} mrad off the last frame"
            block = abs_col.block(arm_id) if abs_col is not None else None
            if block is not None:
                row = abs_col.rows[traj.frames - 1, block.start : block.stop]
                try:
                    want = Pose(
                        np.asarray(row[:3], dtype=np.float64),
                        se3.rot6d_to_quat(np.asarray(row[3:9], dtype=np.float64)),
                    )
                    dp, dr = se3.pose_error(want, st.ee_pose)
                    text += f" and {dp * 1e3:.1f} mm / {dr * 1e3:.0f} mrad off the last command"
                except Exception:  # noqa: BLE001 - a malformed row only costs the report
                    text += ", command residual unavailable"
            out.append((arm_id, text, ok))
        return out

    def episode_playback_stop(self) -> ReturnHomeResult:
        """``POST /api/session/playback {action: "stop"}``: cancel a replay in flight.

        The Welcome page's dialog never opens ``/ws/control``, so the operator has no key
        to cancel with — this is their stop button. Idempotent: nothing in flight is a
        success, because the wanted state (no replay running) already holds.
        """
        session = self.session
        if session is None:
            return ReturnHomeResult(ok=False, status="refused", detail="no active session")
        if not session.loop.motion_active:
            return ReturnHomeResult(ok=True, status="skipped", detail="nothing is playing")
        self._cancel_plan_via_loop("playback stopped by the operator")
        return ReturnHomeResult(ok=True, status="done", detail="playback stopped; the arms hold")

    def _return_to_initial_motion(
        self, session: ActiveSession, profile, *, label: str = "reset_to_initial"
    ) -> ReturnHomeResult:
        """Two twin-planned, gated phases (2026-09-08 operator decision):

        1. the JOINTS to the profile's posture with each carriage held where it is;
        2. the CARRIAGES to the profile's rail positions with the joints held.

        A phase whose start already equals its goal is skipped. Phase 2 does not run
        if phase 1 did not arrive, and it is skipped entirely for a profile that
        stores no rail position (``rail_pos_m is None`` = keep the carriage), which is
        how the seeded default postures are stored. The profile's gripper target rides
        the LAST phase that runs and is applied on arrival only.

        Within a phase the arms move ONE AT A TIME in the planner's ``arm_order``
        (:meth:`_execute_arms`; 2026-09-08 evening - the first live run of this
        motion executed both arms' sequentially planned paths simultaneously and the
        gate held them at 5.2 mm).
        """
        import numpy as np

        _states, q_start, q_joint, q_full, grippers = self._return_goals(
            session, profile, session.spec.arms
        )
        arms = sorted(q_joint)
        base = ReturnHomeResult(
            ok=False, status="failed", detail="", arms=arms, profile_id=profile.profile_id
        )
        if not q_joint:
            return base.model_copy(
                update={
                    "ok": True,
                    "status": "skipped",
                    "detail": f"profile '{profile.name}' covers no arm of this session",
                }
            )
        joints_move = any(
            not np.allclose(q_start[a][:7], q_joint[a][:7], atol=1e-3) for a in q_joint
        )
        rail_move = any(not np.allclose(q_joint[a], q_full[a], atol=1e-3) for a in q_full)
        if not joints_move and not rail_move:
            return base.model_copy(
                update={
                    "ok": True,
                    "status": "skipped",
                    "detail": (
                        f"already at profile '{profile.name}'"
                        if label == "goto_profile"
                        else "already at the initial condition"
                    ),
                }
            )
        phases = []
        if joints_move:
            phases.append(("joints", q_joint))
        if rail_move:
            phases.append(("carriage", q_full))
        for i, (label, goal) in enumerate(phases):
            last = i == len(phases) - 1
            status, detail = self._return_phase(
                session, goal, grippers if last else {}, label=label
            )
            if status == "skipped":
                continue
            if status != "done":
                return base.model_copy(
                    update={
                        # the wire's ``timeout`` = "stopped where they are, not at the goal";
                        # the loop's gate-held abort (``held``) is that, just reported at
                        # once, and so is an arm whose measured posture never settled at
                        # the commanded goal (``stalled``)
                        "status": "timeout" if status in ("held", "stalled") else status,
                        "detail": self._return_phase_text(label, status, detail, len(phases) > 1),
                    }
                )
        return base.model_copy(update={"ok": True, "status": "done", "detail": ""})

    def _return_phase(
        self, session: ActiveSession, goal: dict, grippers: dict, *, label: str
    ) -> tuple[str, str]:
        """Plan + run one phase from the arms' MEASURED posture to ``goal``.
        ``("skipped", "")`` when they are already there."""
        import numpy as np

        states = session.workcell.states()
        q_start = {a: [float(x) for x in states[a].q] for a in goal}
        if all(np.allclose(q_start[a], goal[a], atol=1e-3) for a in goal):
            return "skipped", ""
        if session.state is not SessionState.RUNNING:
            return "cancelled", f"session {session.state.value}"
        result = self._plan_return(session, states, q_start, goal)
        if not result.ok:
            failure = self._plan_failure_text(result)
            logger.warning("return-to-initial (%s) plan failed: %s", label, failure)
            return "failed", failure
        return self._run_return_plan(
            session, self._ordered_waypoints(result), grippers, interruptible=True
        )

    @staticmethod
    def _return_phase_text(label: str, status: str, detail: str, two_phase: bool) -> str:
        """Operator-facing sentence for a phase that did not arrive. The wording is
        what the Cockpit's dialog shows, so it names the phase only when there are
        two of them (otherwise "the joints" is noise)."""
        where = f" while moving the {label}" if two_phase else ""
        if status == "failed":
            return (
                f"the digital twin could not plan a collision-free path{where}: {detail}. "
                "The arms have not moved."
            )
        if status == "timeout":
            return f"the motion was held by the safety gate{where} and stopped part-way ({detail})."
        if status == "held":  # the loop's gate-held abort (plan_gate_hold_s)
            return (
                f"the motion was held by the safety gate{where} and stopped ({detail}). "
                "The arms hold where they are."
            )
        if status == "stalled":  # the command arrived, the measured arm did not settle
            return (
                f"an arm did not settle at its goal{where} ({detail}). "
                "The arms hold where they are."
            )
        if status == "cancelled":
            return f"the motion was cancelled{where}: {detail}. The arms hold where they are."
        return f"the motion was refused{where}: {detail}."

    def _build_policy_stack(
        self,
        spec: SessionSpec,
        session_cfg: WorkcellConfig,
        workcell,
        scene,
        session_id: str,
        ik,
        kin,
        twin,
        supervisor,
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
        from ..dagger.policy_runner import (
            ActionAnchor,
            PolicyRunner,
            SlewLimits,
            anchor_leash_kwargs,
        )
        from ..dagger.registry import resolve_policy
        from ..dagger.trainer.checkpoints import STATE_DICT, CheckpointStore, sha256_file
        from ..recorder.features import arm_action_names, arm_state_names
        from ..recorder.frames import RecordingFrameConverter
        from ..recorder.kinematics import RecorderKinematics

        dcfg = self.cfg.dagger
        frames = {a: spec.frames.get(a, f"arm_base:{a}") for a in spec.arms}
        if spec.policy_source == "external":  # phase-12 (14-dora §6.1)
            return self._build_external_policy_stack(
                spec, session_cfg, workcell, scene, session_id, ik, kin, twin, supervisor, frames
            )
        resolved = resolve_policy(self.cfg.checkpoints_root, spec.mode, spec.policy)
        info = resolved.info
        # 2026-09-11: the executor drives ``delta_ee`` AND ``abs_ee`` (dagger/step.py); the
        # space refusal has its own text, the FRAME refusal keeps the one the UI knows
        if info.action_space not in ("delta_ee", "abs_ee"):
            raise SessionError(
                f"unsupported policy action_space {info.action_space!r} (delta_ee or abs_ee)"
            )
        if any(f != info.action_frame for f in frames.values()):
            raise SessionError("policy/dataset frame mismatch")
        from ..dagger.policies import MLPPolicy, resolve_device

        device = resolve_device(dcfg.policy_device)
        try:
            policy = MLPPolicy.from_bundle(str(resolved.state_dict_path), info.version, device)
        except Exception as e:
            raise SessionError(f"policy load failed: {e!r}") from e
        gate = TakeoverGateImpl(list(spec.arms), dcfg.t_blend_s)
        arms_meta = [(a, bool(scene.meta.rail[a])) for a in spec.arms]
        policy_lock = threading.Lock()
        runner = PolicyRunner(
            policy,
            make_obs_fn(
                self.bus, arms_meta, RecordingFrameConverter(frames, {}), RecorderKinematics(scene)
            ),
            rate_hz=dcfg.policy_rate_hz,
            policy_lock=policy_lock,
        )
        anchor = ActionAnchor(
            ik,
            kin,
            SlewLimits(window_s=dcfg.slew_window_s),
            action_space=info.action_space,
            **anchor_leash_kwargs(dcfg, self.cfg.control),
        )
        common = dict(
            ik=ik,
            kin=kin,
            planner=twin,
            profile_store=self.profile_store,
            workcell_kind="sim",
            gripper_arms=_gripper_arms(scene, spec.arms),
            tracker=self._tracker_provider(),
            plan_gate_hold_s=self.cfg.hardware_session.plan_gate_hold_s,
        )
        if spec.mode == "inference":
            loop = GatedPolicyExecutor(
                workcell,
                self.cfg.control,
                self.bus,
                supervisor,
                list(spec.arms),
                gate=gate,
                runner=runner,
                anchor=anchor,
                arms_meta=arms_meta,
                session_mode="inference",
                version_label=resolved.policy_id,
                recorder=None,
                recorder_fps=self.cfg.recorder.fps,
                on_gate_events=self._gate_events_hook(),
                **common,
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
        store.write_manifest(
            CheckpointInfo(
                run_id=run_id,
                version=0,
                path=str(v0),
                parent_version=None,
                trained_on_frames=info.trained_on_frames,
                trained_on_episodes=[],
                action_frame=info.action_frame,
                action_space=info.action_space,
                sanity_ok=True,
                mean_loss=info.mean_loss,
                sha256=sha256_file(v0 / STATE_DICT),
                created_wallclock_ns=time.time_ns(),
            )
        )
        policy.set_version(0)
        recorder_thread = self._build_collect_recorder(
            spec,
            session_cfg,
            workcell,
            scene,
            session_id,
            dagger_ctx={"run_id": run_id, "gate": gate},
        )
        state_dim = sum(len(arm_state_names(a, r)) for a, r in arms_meta)
        action_dim = sum(len(arm_action_names(a, r, info.action_space)) for a, r in arms_meta)
        tcfg = TrainerConfig(
            run_id=run_id,
            checkpoints_root=str(self.cfg.checkpoints_root),
            spool_dir=str(recorder_thread.spool_dir),
            seed_bundle=str(v0 / STATE_DICT),
            port=dcfg.trainer.port,
            device=dcfg.trainer.device,
            cuda_visible_devices=dcfg.trainer.cuda_visible_devices,
            action_frame=info.action_frame,
            action_space=info.action_space,
            state_dim=state_dim,
            action_dim=action_dim,
            min_new_labels=dcfg.trainer.min_new_labels,
            push_period_s=dcfg.trainer.push_period_s,
            batch_size=dcfg.trainer.batch_size,
            lr=dcfg.trainer.lr,
        )
        client = AsyncTrainerClientImpl(tcfg, workdir=store.root)
        reloader = PolicyReloaderImpl(
            policy,
            store,
            info.action_frame,
            info.action_space,
            policy_lock=policy_lock,
            current_version=0,
            on_rollback=client.notify_rollback,
        )
        loop = GatedPolicyExecutor(
            workcell,
            self.cfg.control,
            self.bus,
            supervisor,
            list(spec.arms),
            gate=gate,
            runner=runner,
            anchor=anchor,
            arms_meta=arms_meta,
            session_mode="dagger",
            run_id=run_id,
            reloader=reloader,
            trainer_client=client,
            recorder=recorder_thread,
            recorder_fps=self.cfg.recorder.fps,
            on_gate_events=self._gate_events_hook(),
            **common,
        )
        recorder_thread.on_episode_saved = self._episode_saved_hook(
            loop.on_episode_saved, recorder_thread, run_id
        )
        return loop, recorder_thread, DaggerSession(loop, runner, reloader, client)

    def _episode_saved_hook(self, inner, recorder_thread, run_id: str):
        """Chain the executor's boundary callback with the dora ``events.episode_saved``
        publish (phase-12; 14-dora §4.2) - a no-op when the bridge is off."""
        dora = self.dora
        if dora is None or not dora.enabled:
            return inner
        root = str(getattr(getattr(recorder_thread, "recorder", None), "root", "") or "") or None

        def hook(index: int, summary, spool_path: str) -> None:
            if inner is not None:
                inner(index, summary, spool_path)
            try:
                dora.publish_episode_saved(index, summary, spool_path, root, run_id)
            except Exception:  # noqa: BLE001 - the bus never breaks a save
                logger.exception("dora episode_saved publish failed")

        return hook

    def _build_external_policy_stack(
        self, spec, session_cfg, workcell, scene, session_id, ik, kin, twin, supervisor, frames
    ):
        """``policy_source: external`` (phase-12; 14-dora §6.1): the policy node drives
        through the dora bus. Requires the bridge ``attached`` and a fresh ``policy_spec``
        (409 ``no external policy attached``, no waiting); frame / space checked like a
        checkpoint (409 ``policy/dataset frame mismatch``); ``spec.action_names`` must
        equal the session's action layout and ``state_names`` be a subset of its state
        layout. No ``resolve_policy`` / ``MLPPolicy`` / ``PolicyReloaderImpl`` /
        ``AsyncTrainerClientImpl``: DAgger keeps the recorder and publishes
        ``events.episode_saved``; ``trainer_alive`` stays null. Every ``TakeoverGate``
        event is published as ``events.gate`` (15-online-dagger §3).

        Online DAgger (``spec.online_dagger``; phase-14): the rollouts are recorded into
        ``online_dagger/<session_name>`` (resumed when ``online_dagger.resume``), the
        ``OnlineDaggerCoordinator`` is handed to the executor and hooked to the recorder's
        save / discard and to the gate, and ``trainer_alive`` follows the trainer's status
        freshness."""
        from ..dagger.gate import TakeoverGateImpl
        from ..dagger.loop import (
            DaggerSession,
            GatedPolicyExecutor,
            InferenceSession,
        )
        from ..dagger.policy_runner import ActionAnchor, SlewLimits, anchor_leash_kwargs
        from ..dora_bridge.policy_source import (
            DrivenArmsError,
            ExternalPolicySource,
            resolve_driven_arms,
            spec_from_announce,
        )
        from ..recorder.features import arm_state_names

        dora, hub, ann = self._external_hub_or_409()
        pspec = spec_from_announce(ann)
        arms_meta = [(a, bool(scene.meta.rail[a])) for a in spec.arms]
        # v1.3 (14-dora §6.1): the policy drives the arms its spec names (or every arm whose
        # whole block its action_names cover); every other session arm holds. The frame
        # check is per DRIVEN arm - a per-arm policy may record each arm in its own base
        # frame (``action_frames``), as the cell's datasets do
        if pspec.action_space not in ("delta_ee", "abs_ee"):
            raise SessionError(
                f"unsupported external policy action_space {pspec.action_space!r} "
                "(delta_ee or abs_ee)"
            )
        try:
            driven = resolve_driven_arms(
                ann.spec.arms, pspec.action_names, arms_meta, pspec.action_space
            )
        except DrivenArmsError as e:
            raise SessionError(str(e)) from e
        for a in driven:
            want = ann.spec.action_frames.get(a, pspec.action_frame)
            if frames[a] != want:
                logger.warning(
                    "external policy frame for arm %s is %r, the session records %r",
                    a,
                    want,
                    frames[a],
                )
                raise SessionError("policy/dataset frame mismatch")
        state_names = [n for a, r in arms_meta for n in arm_state_names(a, r)]
        if not set(pspec.state_names) <= set(state_names):
            raise SessionError(
                "external policy state_names are not a subset of the session state layout: "
                f"{sorted(set(pspec.state_names) - set(state_names))}"
            )
        dcfg = self.cfg.dagger
        source = ExternalPolicySource(
            hub,
            dora.publisher,
            session_id=session_id,
            spec=pspec,
            policy_id=ann.policy_id,
            arms_meta=arms_meta,
            rate_hz=float(ann.rate_hz) if ann.rate_hz else dcfg.policy_rate_hz,
            chunk_dt_s=ann.chunk_dt_s,
            cfg=self.cfg.dora.policy,
            driven_arms=driven,
        )
        if len(driven) < len(arms_meta):
            logger.info(
                "external policy %s drives %s; %s hold",
                ann.policy_id,
                driven,
                [a for a, _ in arms_meta if a not in driven],
            )
        gate = TakeoverGateImpl(list(spec.arms), dcfg.t_blend_s)
        anchor = ActionAnchor(
            ik,
            kin,
            SlewLimits(window_s=dcfg.slew_window_s),
            action_space=pspec.action_space,
            **anchor_leash_kwargs(dcfg, self.cfg.control),
        )
        common = dict(
            ik=ik,
            kin=kin,
            planner=twin,
            profile_store=self.profile_store,
            workcell_kind="sim",
            gripper_arms=_gripper_arms(scene, spec.arms),
            tracker=self._tracker_provider(),
            plan_gate_hold_s=self.cfg.hardware_session.plan_gate_hold_s,
        )
        if spec.mode == "inference":
            loop = GatedPolicyExecutor(
                workcell,
                self.cfg.control,
                self.bus,
                supervisor,
                list(spec.arms),
                gate=gate,
                runner=source,
                anchor=anchor,
                arms_meta=arms_meta,
                session_mode="inference",
                recorder=None,
                recorder_fps=self.cfg.recorder.fps,
                on_gate_events=self._gate_events_hook(),
                **common,
            )
            return loop, None, InferenceSession(loop, source)
        run_id = session_id[:8]
        dagger_ctx: dict = {"run_id": run_id, "gate": gate}
        online_dagger = None
        if spec.online_dagger is not None:
            online_dagger = self._build_online_dagger(spec, session_id, run_id, ann, hub, dora)
            dagger_ctx.update(repo_id=online_dagger.repo_id, coordinator=online_dagger.coordinator)
        try:
            recorder_thread = self._build_collect_recorder(
                spec, session_cfg, workcell, scene, session_id, dagger_ctx=dagger_ctx
            )
            loop = GatedPolicyExecutor(
                workcell,
                self.cfg.control,
                self.bus,
                supervisor,
                list(spec.arms),
                gate=gate,
                runner=source,
                anchor=anchor,
                arms_meta=arms_meta,
                session_mode="dagger",
                run_id=run_id,
                reloader=None,
                trainer_client=None,
                recorder=recorder_thread,
                recorder_fps=self.cfg.recorder.fps,
                coordinator=online_dagger.coordinator if online_dagger is not None else None,
                # gate events: the coordinator's serial worker, else the publisher's queue
                on_gate_events=(
                    online_dagger.coordinator.on_gate_events
                    if online_dagger is not None
                    else self._gate_events_hook()
                ),
                **common,
            )
            if online_dagger is not None:
                recorder_thread.on_episode_saved = self._online_dagger_saved_hook(
                    loop.on_episode_saved, online_dagger.coordinator
                )
                recorder_thread.on_episode_discarded = self._online_dagger_discard_hook(
                    online_dagger.coordinator
                )
            else:
                recorder_thread.on_episode_saved = self._episode_saved_hook(
                    loop.on_episode_saved, recorder_thread, run_id
                )
            policy_session = DaggerSession(loop, source, None, None)
        except BaseException:
            if online_dagger is not None:
                online_dagger.abandon()  # a fresh name stays usable after a 409 / 500
            raise
        policy_session.online_dagger = online_dagger  # picked up by _bringup_sim -> ActiveSession
        return loop, recorder_thread, policy_session

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
            if session.state is SessionState.TEARDOWN:
                return  # ended before the motion started
            # 2026-09-08: an arm can be RECOVERING for a tick or two right after the
            # driver enabled it (controller state 4, transient). Give such a fault
            # ``start_from_fault_grace_s`` to clear BEFORE the plan is handed to the loop
            # instead of letting the loop refuse it and dropping the plan for good. A
            # fault that persists still refuses below (never move a faulted arm).
            session.start_from_progress = 0.0
            grace_deadline = time.monotonic() + float(
                self.cfg.hardware_session.start_from_fault_grace_s
            )
            self._await_arms_clear(session, grace_deadline)
            if session.state is SessionState.TEARDOWN:
                return
            if session.state in (SessionState.BRINGUP, SessionState.RUNNING):
                # RUNNING: the fault callback already walked RECOVERING -> RUNNING
                session.state = SessionState.START_FROM
            # else FAULT / RECOVERING persists: the loop refuses the plan below and the
            # refusal is reported specifically instead of silently dropping the motion
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
                result = session.twin.plan(
                    PlanRequest(
                        q_start=q_start, q_goal=q_goal, speed_scale=session.spec.speed_scale
                    )
                )
                waypoints = self._ordered_waypoints(result)  # executed in arm_order
            if result is not None and not result.ok:
                logger.error("start_from plan failed: %s %s", result.failure, result.failing_pair)
                session.motion_detail = (
                    f"start_from plan failed: {self._plan_failure_text(result)} - the arms have "
                    "not moved; Go to profile retries it"
                )
                if session.state is SessionState.START_FROM:
                    session.state = SessionState.RUNNING  # arms stay held; operator decides
                session.start_from_progress = None
                self._bringup = None  # the bring-up is over either way
                self.bus.commands.submit(
                    Command(
                        op="_plan_ready",
                        args={"arms": list(q_goal), "result": result, "detail": ""},
                        source="internal",
                    )
                )
                return

            def submit(wps: dict, grip: dict):
                ack = self._submit_execute_plan(wps, grip, interruptible=False)
                if not ack.ok and self._await_arms_clear(session, grace_deadline):
                    # refused, but the arms cleared within the remaining grace: ONE retry
                    logger.info(
                        "start_from refused (%s) - arms clear now, retrying once", ack.detail
                    )
                    ack = self._submit_execute_plan(wps, grip, interruptible=False)
                return ack

            def progress(frac: float) -> None:
                session.start_from_progress = frac

            # ONE ARM AT A TIME in the planner's order (2026-09-08 evening; module
            # docstring): ``waypoints`` is keyed in ``PlanResult.arm_order`` by
            # ``_ordered_waypoints`` / ``_plan_profile_start``. Once a plan is in the loop
            # the wait follows the EXECUTOR, not ``session.state``: a transient driver
            # fault mid-plan walks the session START_FROM -> FAULT -> RECOVERING ->
            # RUNNING through the callback while the arm's waypoints keep executing;
            # keyed on the state this used to exit at once, leaving
            # ``start_from_progress`` None and the bring-up rows on screen for good.
            # ``start_from_progress`` counts over ALL arms' waypoints.
            status, detail = self._execute_arms(
                session,
                waypoints,
                grippers,
                interruptible=False,
                running=lambda: session.state is not SessionState.TEARDOWN,
                budget_s=lambda arm_id, wps: max(
                    120.0, self._return_budget_s(session, {arm_id: wps})
                ),
                tag="start_from",
                timeout_reason="timed out",
                submit=submit,
                on_progress=progress,
            )
            if status == "refused":  # a faulted arm refuses the plan: say so, never a silent hold
                detail = self._start_from_refusal(session, detail)
                logger.warning("start_from refused by the loop: %s", detail)
                session.motion_detail = detail  # survives the fault cycle (unlike fault_detail)
            elif status != "done":
                # a cancel / gate hold / timeout part-way: which arm, why, and that the
                # remaining arms never started (``_sequence_detail``)
                verb = {
                    "held": GATE_HOLD_PREFIX,
                    "timeout": "timed out",
                    "stalled": "stalled (the arm did not settle at its goal)",
                }.get(status, status)
                session.motion_detail = f"start_from {verb}: {detail} (Go to profile retries it)"
                logger.warning("start_from %s: %s", verb, detail)
            session.start_from_progress = None
            if session.state is SessionState.START_FROM:  # a driver fault may own it now
                session.state = SessionState.RUNNING
            self._bringup = None  # the motion is over either way: no more bring-up rows
        except Exception as e:
            logger.exception("start_from worker failed")
            session.fault_detail = repr(e)
            session.state = SessionState.FAULT
            self._bringup = None

    START_FROM_FAULT_POLL_S = 0.05

    @staticmethod
    def _stuck_arms(session: ActiveSession) -> list[str]:
        """Session arms the loop would refuse a plan for: FAULTED or RECOVERING."""
        loop = session.loop
        return sorted(loop.faulted_arms | loop.recovering_arms)

    def _await_arms_clear(self, session: ActiveSession, deadline: float) -> bool:
        """Poll (``START_FROM_FAULT_POLL_S``) until no session arm is FAULTED / RECOVERING
        in the loop AND the session itself has left FAULT / RECOVERING, or ``deadline``
        (monotonic) passes, or a teardown starts. True = clear. Returns at once when
        the arms are already clear or the deadline is in the past, so callers can bound
        every wait by one grace budget."""
        while True:
            if session.state is SessionState.TEARDOWN:
                return False
            if not self._stuck_arms(session) and session.state not in (
                SessionState.FAULT,
                SessionState.RECOVERING,
            ):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(self.START_FROM_FAULT_POLL_S)

    def _start_from_refusal(self, session: ActiveSession, ack_detail: str) -> str:
        """Operator-facing ``fault_detail`` for a start_from plan the loop refused:
        WHICH arm (user-facing name), its controller state and error code, and what to
        do about it. Falls back to the loop's own words when no arm can be named."""
        loop = session.loop
        faulted = sorted(loop.faulted_arms)
        # RECOVERING is lifted by releasing every live input (clutch / keys), not by
        # clearing errors - a clutch held through bring-up must not be reported as a C0
        recovering = sorted(loop.recovering_arms - loop.faulted_arms)
        if not faulted and not recovering:  # cleared between the refusal and now
            m = re.search(r"arm '([^']+)' is faulted", ack_detail)
            faulted = [m.group(1)] if m else []  # the loop's nack names the arm
        if not faulted and not recovering:
            return f"start_from refused: {ack_detail}"
        parts = []
        if faulted:
            try:
                states = session.workcell.states()
            except Exception:  # noqa: BLE001 - a dead driver must not hide the refusal
                states = {}
            names = ", ".join(arm_label(a) for a in faulted)
            ctrl = " / ".join(str(getattr(states.get(a), "state", "?")) for a in faulted)
            codes = " / ".join(f"C{getattr(states.get(a), 'error_code', '?')}" for a in faulted)
            parts.append(
                f"{names} faulted (controller state {ctrl}, code {codes}) - "
                "use Clear errors & resume"
            )
        if recovering:
            names = ", ".join(arm_label(a) for a in recovering)
            parts.append(f"{names} is recovering - release every input (clutch / keys)")
        return f"start_from refused: {'; '.join(parts)}, then Go to profile"

    # -- driver faults: FAULT -> RECOVERING -> RUNNING (phase-09b; 04-runtime §15) -----------
    def attach_fault_state(self, session: ActiveSession) -> None:
        """Wire the loop's SessionManager hooks — ``on_fault_state`` to this session's
        state, ``on_reset_to_initial`` to the ``R``-key return and ``on_goto_profile``
        to the "Go to profile" op (both 2026-09-08). Also the seam the tests use to
        install a hand-built session."""
        session.loop.on_fault_state = lambda state, s=session: self._on_arm_fault_state(s, state)
        session.loop.on_reset_to_initial = self.request_reset_to_initial
        session.loop.on_goto_profile = self.request_goto_profile

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
                logger.info(
                    "session %s RECOVERING (release every input to resume)", session.session_id
                )
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
            detail = f"recovered from {title}" if title else "re-seeded from the measured position"
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

    def session_set_collision_sensitivity(
        self, arm_id: str, level: int, timeout_s: float = RECOVERY_WAIT_S
    ) -> ArmMaintenanceResult:
        """Session path of ``set_collision_sensitivity`` (2026-09-11; 04-runtime
        §13.1): inside a hardware session the ONE write goes to the SESSION driver's
        monitor thread (``workcell.request_set_collision_sensitivity(arm_id, level)``,
        the sibling of ``request_recovery``: never an SDK call on this thread) and
        the outcome is awaited <= ``timeout_s`` via ``workcell.setting_result`` (a
        ``SettingResult`` with a higher ``seq``). No motion: the SDK call runs
        ``wait_move()`` (immediate in servo mode 1), ``set_collis_sens`` and an
        idempotent ``set_state(0)``; the driver's phase / budget / streamer are
        untouched. ``ok`` = the SDK accepted the write (code 0, or a 1 / 2 / 9
        status echo while a fault is latched - reported in ``warnings``); the
        ``real`` report stream carries no sensitivity read-back, so ``before`` /
        ``after`` are ``None``, the result's ``collision_sensitivity`` echoes the
        level written and the read-only monitor verifies the value once it holds
        the box again. On success the level is recorded in the hardware monitor's
        requested map so telemetry shows it for the rest of the session; the
        next driver connect re-applies the config value. 409
        (:class:`MaintenanceUnavailableError`): no hardware session, an arm
        outside the session, a workcell without the channel, the driver's
        ``CommandError`` (not connected / read-only / level refused)."""
        session = self.session
        if session is None or session.spec.kind != "hardware":
            raise MaintenanceUnavailableError("no hardware session")
        if arm_id not in session.spec.arms:
            raise MaintenanceUnavailableError(f"arm {arm_id!r} is not part of the session")
        workcell = session.workcell
        request = getattr(workcell, "request_set_collision_sensitivity", None)
        result_of = getattr(workcell, "setting_result", None)
        if request is None or result_of is None:
            raise MaintenanceUnavailableError(
                "the session workcell has no collision-sensitivity channel"
            )
        previous = result_of(arm_id)
        seq0 = int(getattr(previous, "seq", 0) or 0) if previous is not None else -1
        try:
            request(arm_id, int(level))
        except KeyError:
            raise MaintenanceUnavailableError(f"unknown arm {arm_id!r}") from None
        except Exception as e:  # noqa: BLE001 - CommandError: driver not connected / refused
            raise MaintenanceUnavailableError(f"{arm_id}: {e}") from e
        deadline = time.monotonic() + float(timeout_s)
        res = None
        while time.monotonic() < deadline:
            res = result_of(arm_id)
            if res is not None and int(getattr(res, "seq", 0) or 0) > seq0:
                break
            time.sleep(RECOVERY_POLL_S)
        else:
            return ArmMaintenanceResult(
                arm_id=arm_id,
                op="set_collision_sensitivity",
                path="session",
                ok=False,
                detail=(
                    "set_collision_sensitivity: no result from the session driver within "
                    f"{float(timeout_s):g} s"
                ),
            )
        ok = bool(res.ok)
        code = int(getattr(res, "code", 0) or 0)
        note = str(getattr(res, "detail", "") or "")
        detail = note or (
            f"collision sensitivity set to {level} on the session driver (verified by the "
            "monitor after the session)"
        )
        if ok and self.hardware_monitor is not None:
            try:
                self.hardware_monitor.note_requested_sensitivity(arm_id, int(level))
            except (KeyError, ValueError):  # an arm the monitor does not know: nothing to show
                logger.warning("requested sensitivity of %r not recorded", arm_id, exc_info=True)
        return ArmMaintenanceResult(
            arm_id=arm_id,
            op="set_collision_sensitivity",
            path="session",
            ok=ok,
            detail=detail,
            sdk_codes={"set_collision_sensitivity": code},
            warnings=[note] if ok and code != 0 and note else [],
            collision_sensitivity=int(level) if ok else None,
            before=None,
            after=None,
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
            # A return-to-start (or any plan) in flight: stop it through the loop so the
            # arm HOLDS, instead of workcell.stop() cutting the stream mid-move.
            self._cancel_plan_via_loop("session teardown")
            try:
                if session.policy_session is not None:
                    session.policy_session.stop()  # trainer stop -> reloader -> runner
                if session.recorder_thread is not None:
                    session.recorder_thread.stop()  # discard + finalize (§10.4)
                session.loop.stop()
                if session.online_dagger is not None:  # phase-14: last session.json, sink off
                    try:
                        session.online_dagger.close()
                    except Exception:  # noqa: BLE001
                        logger.exception("Online DAgger coordinator close failed")
                if not hardware:  # phase-12: the parked pose survives into the sim preview
                    try:
                        # the LAST loop snapshot's joints (what the last in-session camera frames
                        # were stamped with), falling back to the workcell's measured state
                        got = self.bus.snapshot.get()
                        snap = got[0] if got is not None and got[0].tick >= 0 else None
                        states = snap.arms if snap is not None else session.workcell.states()
                        self._parked_q = {a: np.array(st.q) for a, st in states.items()}
                        self._parked_scene = (
                            session.spec.sim_scene
                            or (
                                self.cfg.workcell_config("sim") or ArmConfig(id="x")  # type: ignore[union-attr]
                            ).sim_scene
                        )
                        # the standby previews (warmed at session start) take the parked pose now;
                        # wait for one fresh frame so the hand-over shows the parked arm, not the
                        # keyframe (<= one preview period)
                        if self._preview_service is None and self._pending_preview is None:
                            self._pending_preview = self._prepare_sim_previews()
                        self._repose_pending_preview()
                    except Exception:  # noqa: BLE001
                        logger.exception("parked pose capture failed")
                # phase-12: hand each camera stream straight over to its warmed preview twin
                # (same stream id) so the dora publisher misses at most one frame period
                warmed = self._pending_preview[2] if self._pending_preview else []
                pending_cams = {c.camera_id: c for c in warmed}
                for sid in session.streams:
                    t_rm = time.monotonic()
                    self.hub.remove_stream(sid)
                    cam = pending_cams.get(sid)
                    if cam is not None and not self.hub.has(sid):
                        self.hub.add_stream(sid, cam, self.cfg.video.preview_fps)
                        self._preview_sources.append(cam)
                        self._preview_ids.append(sid)
                        logger.info(
                            "teardown: stream %s handed to the warmed preview in %.0f ms",
                            sid,
                            (time.monotonic() - t_rm) * 1e3,
                        )
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
        if self.dora is not None and self.dora.enabled:
            self.dora.after_teardown()  # session announce -> idle; idle arm reader resumes

    # -- phase-12: parked pose + session facts for the dora publisher (14-dora §4.2/§7) ----
    def _depth_cameras(self) -> set[str]:
        """Sim cameras rendered WITH a depth sibling: the dora ``publish.depth_cameras`` when
        the bridge is enabled (03-sim §7 lifts "depth off in v1" for exactly these)."""
        dora = self.dora
        if dora is None or not dora.enabled:
            return set()
        return set(dora.depth_camera_ids)

    def idle_sim_q(self) -> dict[str, np.ndarray]:
        """Joint vectors (incl. rail) the sim preview shows between sessions: the last
        session's final ``q`` when its scene equals the preview scene, else keyframe 0."""
        wc = self.cfg.workcell_config("sim")
        scene = self._preview_scene
        if wc is None or scene is None:
            return {}
        if self._parked_q and self._parked_scene == wc.sim_scene:
            return {a: np.array(q) for a, q in self._parked_q.items() if a in scene.meta.arm_ids}
        out: dict[str, np.ndarray] = {}
        key = scene.model.key_qpos[0] if scene.model.nkey > 0 else scene.model.qpos0
        for arm_id in scene.meta.arm_ids:
            out[arm_id] = np.array(key[scene.addressing[arm_id].qpos_adr], dtype=np.float64)
        return out

    def _preview_qpos(self, scene, scene_id: str | None) -> np.ndarray | None:
        if not self._parked_q or self._parked_scene != scene_id:
            return None
        qpos = np.array(
            scene.model.key_qpos[0] if scene.model.nkey > 0 else scene.model.qpos0,
            dtype=np.float64,
        )
        for arm_id, q in self._parked_q.items():
            if arm_id in scene.meta.arm_ids:
                adr = scene.addressing[arm_id].qpos_adr
                qpos[adr] = np.asarray(q, dtype=np.float64)[: len(adr)]
        return qpos

    def _session_facts(self, session: ActiveSession, wc: WorkcellConfig):
        """``SessionFacts`` for the dora ``session`` announce + ``obs_state`` layout."""
        from ..dora_bridge.publishers import SessionFacts
        from ..recorder.features import arm_action_names, arm_state_names
        from ..recorder.frames import RecordingFrameConverter

        spec = session.spec
        arms = list(spec.arms)
        has_rail = {a: bool(session.workcell.arms[a].has_rail) for a in arms}
        frames = {a: spec.frames.get(a, f"arm_base:{a}") for a in arms}
        scene_id = (
            (spec.sim_scene or wc.sim_scene)
            if spec.kind == "sim"
            else (spec.digital_twin_scene or wc.digital_twin_scene)
        )
        if spec.kind == "sim":
            cams = list(session.workcell.cameras)
        else:
            cams = [c.id for c in wc.cameras]
        q_by_arm = {}
        try:
            q_by_arm = {a: np.asarray(st.q) for a, st in session.workcell.states().items()}
        except Exception:  # noqa: BLE001
            pass
        announces = self.dora.camera_announces(cams, scene_id, q_by_arm) if self.dora else {}
        loop = session.loop
        runner = getattr(loop, "runner", None)
        gate = getattr(loop, "gate", None)
        recorder = session.recorder_thread
        return SessionFacts(
            session_id=session.session_id,
            spec=spec,
            kind=spec.kind,
            scene_id=scene_id,
            arm_ids=arms,
            has_rail=has_rail,
            frames=frames,
            action_names=[n for a in arms for n in arm_action_names(a, has_rail[a], "delta_ee")],
            state_names=[n for a in arms for n in arm_state_names(a, has_rail[a])],
            camera_ids=cams,
            cameras=announces,
            policy_source=spec.policy_source,
            dataset_root=(
                str(getattr(getattr(recorder, "recorder", None), "root", "") or "") or None
            ),
            run_id=getattr(loop, "run_id", "") or None,
            converter=RecordingFrameConverter(frames, {}),
            engaged_arm=(lambda: gate.engaged_arm()) if gate is not None else (lambda: None),
            gate_events=lambda: list(session.supervisor.events),
            policy_version=(
                (lambda: int(runner.current_version()))
                if runner is not None and hasattr(runner, "current_version")
                else (lambda: None)
            ),
            online_dagger=(
                session.online_dagger.coordinator.announce()
                if session.online_dagger is not None
                else None
            ),
        )

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
        pending = self._pending_preview or self._prepare_sim_previews()
        self._pending_preview = None
        if pending is None:
            return
        rs, scene, cams = pending
        fps = self.cfg.video.preview_fps
        for cam in cams:  # already rendering: the encoders pick a fresh frame within a period
            if cam.camera_id in self._preview_ids:
                continue  # handed over during teardown already
            self.hub.add_stream(cam.camera_id, cam, fps)
            self._preview_sources.append(cam)
            self._preview_ids.append(cam.camera_id)
        self._preview_scene = scene
        self._preview_service = rs

    def _repose_pending_preview(self, wait_s: float = 0.15) -> None:
        pending = self._pending_preview
        wc = self.cfg.workcell_config("sim")
        if pending is None or wc is None:
            return
        rs, scene, cams = pending
        qpos = self._preview_qpos(scene, wc.sim_scene)
        if qpos is None:
            return
        before = {c.camera_id: (c.latest().seq if c.latest() is not None else -1) for c in cams}
        rs.submit_state("preview", qpos, 0.0)
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:  # every camera rendered the parked pose once
            fresh = all(
                (c.latest() is not None and c.latest().seq > before[c.camera_id]) for c in cams
            )
            if fresh:
                break
            time.sleep(0.005)

    def _prepare_sim_previews(self):
        """Build + START the sim preview render service and its cameras WITHOUT publishing
        them on the hub -> ``(render_service, scene, cameras)`` or ``None``. ``teardown()``
        calls this while the session streams still run so the previews are warm (renderer,
        MjData, first frames) when the session cameras leave the hub — the fixed-viewpoint
        consumer sees no gap beyond one frame period (phase-12; 14-dora §7)."""
        wc = self.cfg.workcell_config("sim")
        if wc is None or not wc.sim_scene:
            return None
        try:
            from apollo_mavis_v2_sim import REGISTRY, RenderService, SimCamera
        except ImportError:
            return None
        rs = RenderService()
        rs.start()
        scene = self._preview_scene_cache.get(wc.sim_scene)
        if scene is None:  # a MuJoCo compile (~0.3-0.5 s): built once per process
            scene = REGISTRY.build(wc.sim_scene, _microphone_overrides(wc, wc.sim_scene))
            self._preview_scene_cache[wc.sim_scene] = scene
        rs.register_source("preview", scene.model)
        qpos = self._preview_qpos(scene, wc.sim_scene)
        if qpos is not None:
            rs.submit_state("preview", qpos, 0.0)  # phase-12: parked pose, not the keyframe
        fps = self.cfg.video.preview_fps
        depth = self._depth_cameras()
        cams = []
        for name in scene.meta.cameras:
            cam = SimCamera(
                camera_id=name,
                render_service=rs,
                mjcf_camera=name,
                fps=fps,
                source="preview",
                depth=name in depth,  # phase-12: cam_<id>_depth for the dora depth list
            )
            cam.start()
            cams.append(cam)
        return rs, scene, cams

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
                out.append(
                    CameraInfo(
                        camera_id=cam_id,
                        kind="sim",
                        label=cam_id,
                        resolution=cam.resolution,
                        fps=int(cam.fps),
                        live=True,
                    )
                )
            return out
        for cam in self._preview_sources:
            out.append(
                CameraInfo(
                    camera_id=cam.camera_id,
                    kind="sim",
                    label=cam.camera_id,
                    resolution=cam.resolution,
                    fps=int(cam.fps),
                    live=True,
                )
            )
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
            out.append(
                CameraInfo(
                    camera_id=cam_cfg.id,
                    kind=cam_cfg.kind,
                    label=cam_cfg.id,
                    resolution=tuple(cam_cfg.resolution),
                    fps=int(cam_cfg.fps),
                    live=live,
                )
            )
        # Twin alignment overlays (phase-09a): kind "twin", live iff the real camera
        # underneath is live AND the overlay is compositing (live / stale tint).
        overlay = self.twin_overlay
        if overlay is not None:
            for src in overlay.streams.values():
                out.append(
                    CameraInfo(
                        camera_id=src.stream_id,
                        kind="twin",
                        label=src.label,
                        resolution=src.resolution,
                        fps=int(overlay.cfg.fps),
                        live=bool(live_cams.get(src.camera_id)) and src.status in ("live", "stale"),
                    )
                )
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
            out.append(
                SceneInfo(
                    scene_id=meta.id,
                    label=getattr(meta, "title", None) or meta.description,
                    num_arms=meta.n_arms,
                    rail_flags=[meta.rail[a] for a in meta.arm_ids],
                    cameras=list(meta.cameras),
                    kind=kind,  # type: ignore[arg-type]
                )
            )
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
            kind = (
                self.session.spec.kind
                if self.session
                else ("sim" if "sim" in available else (available[0] if available else "sim"))
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
            kind=kind,
            available_kinds=available,
            arms=arms,
            cameras=cameras,
            policies_available=bool(scan_policies(self.cfg.checkpoints_root)),
            hardware_ready=bool(probe is not None and probe.hardware_ready),
        )

    def _sim_arm_infos(self) -> list:
        from apollo_mavis_v2_core.protocol import ArmStatusInfo

        arms: list[ArmStatusInfo] = []
        scene = None
        connected = False
        states = {}
        # Only a SIM session owns a scene. `HardwareWorkcell` has no `.scene` at all, so
        # an explicit `?kind=sim` while a hardware session runs (the Welcome page polls it)
        # used to 500 here; fall back to the preview scene and report connected=False,
        # which is the truth — no sim session owns those arms (fixed 2026-09-07).
        if self.session is not None and self.session.spec.kind == "sim":
            scene = self.session.workcell.scene
            connected = True
            states = self.session.workcell.states()
        elif self._preview_service is not None:
            scene = self._preview_scene
        if scene is not None:
            for arm_id in _manipulation_first(scene.meta.arm_ids):
                limits = self._joint_limits(scene, arm_id)
                arms.append(
                    ArmStatusInfo(
                        arm_id=arm_id,
                        ip=None,
                        connected=connected,
                        has_rail=scene.meta.rail[arm_id],
                        gripper="xarm" if scene.addressing[arm_id].has_gripper else "none",
                        gripper_force_capable=False,  # sim grippers are position-only
                        error_code=states[arm_id].error_code if arm_id in states else 0,
                        joint_limits=limits,
                    )
                )
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
            arms.append(
                ArmStatusInfo(
                    arm_id=arm.id,
                    ip=arm.ip,
                    connected=connected,
                    reachable=probe.reachable(arm.id) if probe is not None else "unknown",
                    has_rail=twin.meta.rail[arm.id] if in_twin else arm.expect_rail == "yes",
                    gripper=arm.gripper,
                    gripper_force_capable=arm.gripper == "xarm_g2",
                    error_code=monitor.error_code(arm.id) if monitor is not None else 0,
                    joint_limits=self._joint_limits(twin, arm.id) if in_twin else [],
                )
            )
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
