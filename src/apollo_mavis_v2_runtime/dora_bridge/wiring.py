"""``DoraWiring`` — how ``Runtime`` composes the dora bridge (14-dora §2, §4, §13).

One object owned by ``Runtime`` for the process lifetime. With
``dora.enabled: false`` it is inert (no thread, no tap, ``external.state ==
"disabled"``). Otherwise it builds:

- the :class:`DoraBridge` (control plane + node, ``dora-bus`` thread);
- the :class:`ExternalPolicyHub` (spec cache / action router);
- the :class:`PoseStamper` + one :class:`CameraTap` per published camera stream
  (attached to existing ``VideoHub`` streams at start and to later ones through
  ``hub.stream_hooks``: session cameras, hardware previews);
- the :class:`MicTap` on the runtime's ``MicrophoneReader``;
- the :class:`SnapshotPublisher` (``dora-publisher`` thread) and the
  :class:`IdleArmReader` (arm states between sessions).

``SessionManager`` calls :meth:`before_bringup` / :meth:`after_session_start` /
:meth:`after_teardown`; the REST layer reads :meth:`info` and calls
:meth:`request_join`; telemetry reads :meth:`external_status`.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from apollo_mavis_v2_core import se3
from apollo_mavis_v2_core.protocol.external import CameraAnnounce, DoraInfo, ExternalStatus

from ..config import RuntimeConfig
from .bridge import DoraBridge
from .idle_state import DriverIdleSource, IdleArmReader, MonitorIdleSource, SimIdleSource
from .policy_source import ExternalPolicyHub
from .publishers import CameraTap, MicTap, PoseStamper, SessionFacts, SnapshotPublisher, camera_arm

if TYPE_CHECKING:
    from ..runtime import Runtime

logger = logging.getLogger(__name__)

DEFAULT_RES = (640, 480)


def sim_intrinsics(fovy_deg: float, resolution: tuple[int, int] = DEFAULT_RES) -> list[float]:
    """MuJoCo ``fovy`` -> ``[fx, fy, cx, cy]`` (fx = fy = (H/2) / tan(fovy/2); 03-sim §7)."""
    w, h = resolution
    f = (h / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    return [f, f, w / 2.0, h / 2.0]


def _build_scene(scene_id: str) -> Any:
    from apollo_mavis_v2_sim import REGISTRY  # [sim] extra

    return REGISTRY.build(scene_id)


def make_kin_factory() -> Callable[[str], Any]:
    def factory(scene_id: str) -> Any:
        from ..recorder.kinematics import RecorderKinematics

        return RecorderKinematics(_build_scene(scene_id))

    return factory


class DoraWiring:
    """See the module docstring."""

    def __init__(self, runtime: Runtime) -> None:
        self.rt = runtime
        self.cfg: RuntimeConfig = runtime.cfg
        dcfg = self.cfg.dora
        hw = self.cfg.workcell_config("hardware")
        sim = self.cfg.workcell_config("sim")
        self.hw = hw
        self.sim = sim
        arm_ips = tuple(a.ip for a in hw.arms) if hw is not None else ()
        self.bridge = DoraBridge(
            dcfg, epoch=runtime.epoch, arm_ips=arm_ips, session_id=self._current_session_id
        )
        self.hub_hook_installed = False
        self.camera_ids: list[str] = []
        self.depth_camera_ids: list[str] = []
        self.intrinsics: dict[str, list[float]] = {}
        self.mjcf_camera: dict[str, str] = {}
        self.camera_res: dict[str, tuple[int, int]] = {}
        self.camera_fps: dict[str, float] = {}
        self.arm_ids: list[str] = []
        self.has_rail: dict[str, bool] = {}
        self.idle_scene: str | None = None
        self.camera_seen: dict[str, int] = {}
        self.taps: dict[str, CameraTap] = {}
        self.policy_hub: ExternalPolicyHub | None = None
        self.publisher: SnapshotPublisher | None = None
        self.stamper: PoseStamper | None = None
        self.idle_reader: IdleArmReader | None = None
        self.mic_tap: MicTap | None = None
        self._kin_factory = make_kin_factory()
        self._announce_kin: Any = None
        self._telemetry_seq = 0
        self.enabled = bool(dcfg.enabled)
        if dcfg.enabled:
            self._discover()

    # -- discovery of the publishable cameras / arms -----------------------------------------------
    def _discover(self) -> None:
        cams: list[str] = []
        if self.sim is not None and self.sim.sim_scene:
            try:
                from apollo_mavis_v2_sim import REGISTRY

                meta = REGISTRY.meta(self.sim.sim_scene)
                scene = REGISTRY.build(self.sim.sim_scene)
                import mujoco

                for cam in meta.cameras:
                    cams.append(cam)
                    cid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
                    fovy = float(scene.model.cam_fovy[cid]) if cid >= 0 else 57.0
                    self.intrinsics[cam] = sim_intrinsics(fovy)
                    self.camera_res[cam] = DEFAULT_RES
                    self.camera_fps[cam] = self.cfg.video.preview_fps
                self.arm_ids = list(meta.arm_ids)
                self.has_rail = {a: bool(meta.rail[a]) for a in meta.arm_ids}
                self.idle_scene = self.sim.sim_scene
            except Exception:  # noqa: BLE001 - sim extra absent
                logger.exception("dora: sim scene discovery failed")
        if self.hw is not None:
            hw_arms = [a.id for a in self.hw.arms]
            for cam in self.hw.cameras:
                cams.append(cam.id)
                if cam.intrinsics is not None:
                    k = cam.intrinsics
                    self.intrinsics[cam.id] = [float(k.fx), float(k.fy), float(k.cx), float(k.cy)]
                self.camera_res[cam.id] = tuple(cam.resolution)
                self.camera_fps[cam.id] = float(cam.fps)
                arm = camera_arm(cam.id, hw_arms)
                if arm is not None:
                    self.mjcf_camera[cam.id] = f"{arm}_wrist_cam"
                if cam.depth:
                    self.depth_camera_ids.append(cam.id)
            self.arm_ids = hw_arms  # the hardware cell is the lab's arm set
            self.has_rail = {a: True for a in hw_arms}
            if self.hw.digital_twin_scene:
                self.idle_scene = self.hw.digital_twin_scene
        pub = self.cfg.dora.publish
        self.camera_ids = cams if pub.cameras == "all" else [c for c in cams if c in pub.cameras]
        for c in pub.depth_cameras:
            if c in self.camera_ids and c not in self.depth_camera_ids:
                self.depth_camera_ids.append(c)
        mic_id = self.cfg.microphone.mic_id if self.cfg.microphone.enabled else None
        # v1.3 (2026-09-11): the configured arms get a per-arm policy_action_<arm> input each
        self.bridge.set_outputs(self.camera_ids, self.depth_camera_ids, mic_id, self.arm_ids)

    # -- lifecycle ---------------------------------------------------------------------------------
    def start(self) -> None:
        dcfg = self.cfg.dora
        if not dcfg.enabled:
            self.bridge.start()  # -> disabled with detail
            return
        rt = self.rt
        self.policy_hub = ExternalPolicyHub(self.bridge, dcfg.policy, arm_ids=self.arm_ids)
        self.stamper = PoseStamper(
            self._kin_factory,
            intrinsics=self.intrinsics,
            mjcf_camera=self.mjcf_camera,
            arm_ids=self.arm_ids,
            image_pose=dcfg.publish.image_pose,
        )
        self.stamper.set_scene(self.idle_scene)
        self.publisher = SnapshotPublisher(
            dcfg,
            self.bridge,
            rt.bus,
            kin_factory=self._kin_factory,
            telemetry_json=self._telemetry_json,
            session_state=lambda: rt.manager.state.value,
            epoch=rt.epoch,
            telemetry_hz=self.cfg.telemetry_hz,
            camera_seen=self.camera_seen,
            idle_arm_ids=self.arm_ids,
            idle_has_rail=self.has_rail,
            idle_scene=self.idle_scene,
            stamper=self.stamper,
        )
        # camera taps: existing streams now, later ones through the hub hook
        rt.hub.stream_hooks.append(self._on_stream_added)
        self.hub_hook_installed = True
        for sid in rt.hub.ids():
            self._attach_tap(sid)
        if self.cfg.microphone.enabled and dcfg.publish.audio:
            self.mic_tap = MicTap(
                self.bridge,
                self.cfg.microphone.mic_id,
                self.cfg.microphone.sample_rate,
                lambda: rt.microphone.status().status,
            )
            rt.microphone.taps.append(self.mic_tap)
        self.bridge.start()
        self.publisher.start()
        self.idle_reader = self._build_idle_reader()
        if self.idle_reader is not None and not rt.manager.session_active:
            self.idle_reader.start()
        elif self.idle_reader is not None:
            self.idle_reader.start()
            self.idle_reader.pause()
        logger.info(
            "dora wiring up: cameras %s (depth %s), arms %s, idle scene %s, idle source %s",
            self.camera_ids,
            self.depth_camera_ids,
            self.arm_ids,
            self.idle_scene,
            type(self.idle_reader.source).__name__ if self.idle_reader else "none",
        )

    def stop(self) -> None:
        if self.idle_reader is not None:
            self.idle_reader.stop()
        if self.publisher is not None:
            self.publisher.stop()
        if self.hub_hook_installed:
            try:
                self.rt.hub.stream_hooks.remove(self._on_stream_added)
            except ValueError:
                pass
        if self.mic_tap is not None:
            try:
                self.rt.microphone.taps.remove(self.mic_tap)
            except ValueError:
                pass
        self.bridge.stop()

    # -- taps --------------------------------------------------------------------------------------
    def _on_stream_added(self, stream_id: str, worker: Any) -> None:
        # a camera stream is re-created at every session boundary (preview <-> session fps);
        # each new EncoderWorker gets its own tap (the old worker is gone with the old stream)
        if stream_id in self.camera_ids and self.stamper is not None:
            tap = CameraTap(
                self.bridge,
                stream_id,
                self.stamper,
                publish_depth=stream_id in self.depth_camera_ids,
                seen=self.camera_seen,
            )
            self.taps[stream_id] = tap
            worker.taps.append(tap)

    def _attach_tap(self, stream_id: str) -> None:
        if stream_id in self.camera_ids and stream_id not in self.taps and self.stamper is not None:
            tap = CameraTap(
                self.bridge,
                stream_id,
                self.stamper,
                publish_depth=stream_id in self.depth_camera_ids,
                seen=self.camera_seen,
            )
            if self.rt.hub.add_tap(stream_id, tap):
                self.taps[stream_id] = tap

    # -- idle reader -------------------------------------------------------------------------------
    def _build_idle_reader(self) -> IdleArmReader | None:
        rt = self.rt
        dcfg = self.cfg.dora
        if not self.arm_ids or self.idle_scene is None:
            return None
        kin = None
        try:
            kin = self._kin_factory(self.idle_scene)
        except Exception:  # noqa: BLE001 - sim extra absent: ee_pose identity
            logger.exception("dora idle reader: kinematics unavailable")
        if self.hw is not None:
            gripper_arms = [a.id for a in self.hw.arms if a.gripper != "none"]
            monitor = getattr(rt, "hardware_monitor", None)
            use_monitor = (
                dcfg.publish.idle_source == "monitor" and monitor is not None and monitor.enabled
            )
            if use_monitor:
                source: Any = MonitorIdleSource(
                    monitor,
                    kin,
                    self.has_rail,
                    self.cfg.twin_overlay.rail_fallback_m,
                    rail_flip=self.cfg.hardware_session.rail_flip,
                )
            else:
                source = DriverIdleSource(
                    self._readonly_driver_factory(), self.arm_ids, kin, self.has_rail, gripper_arms
                )
            kind = "hardware"
        else:
            source = SimIdleSource(rt.manager.idle_sim_q, kin, self.has_rail)
            kind = "sim"
        return IdleArmReader(rt.bus, source, hz=dcfg.publish.idle_state_hz, kind=kind)

    def _readonly_driver_factory(self) -> Callable[[str], Any]:
        rt = self.rt
        arms = {a.id: a for a in self.hw.arms} if self.hw is not None else {}

        def factory(arm_id: str) -> Any:
            from apollo_mavis_v2_hardware import XArmDriver  # [hardware] extra

            from ..devices.hardware_monitor import default_driver_cfg_factory

            driver_cfg = default_driver_cfg_factory()(arms[arm_id])
            api_factory = rt.manager.driver_api_factory
            if api_factory is None:
                if not self.cfg.hardware_session.armed:
                    raise RuntimeError(
                        "hardware not armed: the read-only idle reader never opens a real "
                        "control box while hardware_session.armed is false"
                    )
                return XArmDriver(driver_cfg)
            return XArmDriver(driver_cfg, api_factory=api_factory)

        return factory

    # -- manager hooks -----------------------------------------------------------------------------
    def before_bringup(self) -> None:
        if self.idle_reader is not None:
            self.idle_reader.pause()

    def after_session_start(self, facts: SessionFacts) -> None:
        if self.publisher is not None:
            self.publisher.session_started(facts)

    def after_teardown(self) -> None:
        if self.publisher is not None:
            self.publisher.session_ended()
        if self.idle_reader is not None:
            self.idle_reader.resume()

    def publish_episode_saved(
        self,
        index: int,
        summary: Any,
        spool_path: str,
        dataset_root: str | None,
        run_id: str | None,
    ) -> None:
        if self.publisher is None:
            return
        import dataclasses

        payload = {
            "episode_index": int(index),
            "summary": dataclasses.asdict(summary)
            if dataclasses.is_dataclass(summary)
            else summary,
            "dataset_root": dataset_root,
            "spool_path": spool_path,
            "run_id": run_id,
        }
        self.publisher.publish_event("episode_saved", payload)

    def publish_gate_events(self, payloads: list[dict[str, Any]]) -> None:
        """``events.gate`` for a plain external session (15-online-dagger §3): the
        executor hands the payloads over ON THE TICK, so they only go into the
        publisher's queue here; the ``dora-publisher`` thread publishes them in order.
        (An Online DAgger session routes them through its coordinator's serial worker.)"""
        if self.publisher is None:
            return
        for payload in payloads:
            self.publisher.enqueue_event("gate", payload)

    # -- announces ---------------------------------------------------------------------------------
    def camera_announces(
        self, camera_ids: list[str], scene_id: str | None, q_by_arm: dict[str, np.ndarray]
    ) -> dict[str, CameraAnnounce]:
        out: dict[str, CameraAnnounce] = {}
        kin = None
        if scene_id is not None:
            try:
                if self._announce_kin is None or self._announce_kin[0] != scene_id:
                    self._announce_kin = (scene_id, self._kin_factory(scene_id))
                kin = self._announce_kin[1]
            except Exception:  # noqa: BLE001
                kin = None
        for cam in camera_ids:
            arm = camera_arm(cam, self.arm_ids)
            t_e_c = t_w_c = None
            if kin is not None:
                mj = self.mjcf_camera.get(cam, cam)
                try:
                    cw = kin.camera_world(mj, q_by_arm)
                    if arm is not None and arm in q_by_arm:
                        q = q_by_arm[arm]
                        tcp_w = se3.pose_mul(kin.base_world(arm, q), kin.tcp_base(arm, q))
                        rel = se3.pose_mul(se3.pose_inv(tcp_w), cw)
                        t_e_c = [float(v) for v in rel.position] + [
                            float(v) for v in rel.orientation
                        ]
                    elif kin.camera_static(mj):
                        t_w_c = [float(v) for v in cw.position] + [float(v) for v in cw.orientation]
                except Exception:  # noqa: BLE001 - camera not in this scene
                    pass
            out[cam] = CameraAnnounce(
                resolution=tuple(self.camera_res.get(cam, DEFAULT_RES)),
                fps=float(self.camera_fps.get(cam, self.cfg.video.session_fps)),
                frame_ref=f"camera:{cam}",
                mount=f"ee:{arm}" if arm else "world",
                intrinsics=self.intrinsics.get(cam),
                T_E_C=t_e_c,
                T_W_C=t_w_c,
                depth=cam in self.depth_camera_ids,
            )
        return out

    # -- status / REST -----------------------------------------------------------------------------
    def _current_session_id(self) -> str:
        s = self.rt.manager.session
        return s.session_id if s is not None else ""

    def _telemetry_json(self) -> str:
        from ..server.ws_telemetry import build_telemetry

        self._telemetry_seq += 1
        return build_telemetry(self.rt, self._telemetry_seq).model_dump_json()

    def external_status(self, now: float | None = None) -> ExternalStatus:
        now = time.monotonic() if now is None else now
        st = self.bridge.status()
        hub = self.policy_hub
        if hub is not None:
            spec = hub.spec(now)
            st.policy_attached = hub.policy_attached
            st.spec_age_s = hub.spec_age_s(now)
            if spec is not None:
                st.policy_id = spec.policy_id
                st.policy_version = int(spec.policy_version)
                st.policy_rate_hz = float(spec.rate_hz)
                # phase-14 (15-online-dagger §6/§8): the FRESH spec's capabilities so the
                # launcher can gate "Start Online DAgger" before a session exists
                st.capabilities = list(spec.capabilities)
                # v1.3 (14-dora §6.1): the arms the fresh spec drives (declared or inferred
                # from its action_names) so the launcher can say "drives: Manipulation Arm"
                st.policy_arms = hub.policy_arms(now)
            # the newest trainer_status while fresh (session-less trainer pill); None once
            # the node detaches or falls silent for > spec_stale_s
            st.trainer_status = hub.trainer_status(now)
            src = hub.source
            if src is not None:
                st.actions_late = int(src.actions_late)
                st.action_age_s = src.action_age_s(now)
                st.version_changes_mid_episode = int(
                    self.publisher.version_changes_mid_episode if self.publisher else 0
                )
                st.policy_version = src.current_version()
        if self.idle_reader is not None:
            st.idle_reader = self.idle_reader.status  # type: ignore[assignment]
        return st

    def info(self) -> DoraInfo:
        return self.bridge.info()

    def request_join(self, machine_id: str) -> bool:
        return self.bridge.request_join(machine_id)


__all__ = ["DoraWiring", "make_kin_factory", "sim_intrinsics"]
