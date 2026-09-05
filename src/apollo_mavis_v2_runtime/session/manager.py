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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from apollo_mavis_v2_core import (
    ArmConfig,
    Command,
    ProfileNotFoundError,
    ProfileStore,
    WorkcellConfig,
)
from apollo_mavis_v2_core.protocol import SessionInfo, SessionSpec

from ..config import RuntimeConfig
from ..control.loop import DEFAULT_ACTIVE_ARM, ControlLoop
from ..control.pose_filter import PoseFilterConfig
from ..control.tracker_teleop import TrackerTeleop
from ..devices.tracker import TrackerSettings
from ..errors import SessionError, SessionNotFoundError
from ..safety.gate import NullGate, SafetyGate
from ..safety.supervisor import SafetySupervisor
from ..safety.watchdog import ArmReportWatchdog, InputWatchdog
from ..streams.hub import VideoHub
from .types import SessionState

if TYPE_CHECKING:
    from ..bus import RuntimeBus
    from ..devices.hardware_monitor import HardwareStateMonitor
    from ..devices.hardware_probe import HardwareProbe
    from ..streams.twin_overlay import TwinOverlayRenderer

logger = logging.getLogger(__name__)


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
        return self.session.state if self.session else SessionState.IDLE

    @property
    def session_active(self) -> bool:
        """A session exists OR :meth:`create` is bringing one up. ``session`` is
        assigned only after bringup returns (seconds of MuJoCo/bringup under
        ``_lock``), so a guard reading it alone — the tracker calibration's
        "stop the session first" (13-tracker §4) — could pass while a session is
        coming up; ``_creating`` is raised under ``_lock`` before validation."""
        return self._creating or self.session is not None

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
        if spec.mode in ("dagger", "inference"):
            from ..dagger.registry import resolve_policy

            resolve_policy(self.cfg.checkpoints_root, spec.mode, spec.policy)  # 409 early
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
                wc = self._validate(spec)
                session = self._bringup_sim(spec, wc)
                self.session = session
            finally:
                self._creating = False
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
                if arm_id in session.loop.gripper_arms:
                    grippers[arm_id] = float(posture.gripper_open_frac)
            from apollo_mavis_v2_core import PlanRequest

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
                self.session = None
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
