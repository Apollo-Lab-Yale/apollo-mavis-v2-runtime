"""Runtime — the composition root; owns everything (04-runtime §2)."""

from __future__ import annotations

import time
import uuid
from concurrent.futures import Future

import numpy as np
from apollo_mavis_v2_core import Command, CommandResult, HeldState, ProfileStore
from apollo_mavis_v2_core.protocol import (
    ArmMaintenanceResult,
    GelloCalibrateRequest,
    GelloCalibrateResult,
    GelloInfo,
    GelloPreviewRequest,
    GelloPreviewResult,
    MicrophoneInfo,
)

from .bus import RuntimeBus
from .config import RuntimeConfig
from .devices.gello import GelloReader, device_telemetry_fields
from .devices.hardware_monitor import (
    CONNECTED_STATUSES,
    MAINTENANCE_TIMEOUT_S,
    HardwareStateMonitor,
)
from .devices.hardware_probe import HardwareProbe
from .devices.microphone import MicrophoneReader, to_info
from .devices.rail_homing import RailHomingService
from .devices.rail_sweep import RailSweepChecker
from .devices.tracker import TrackerReader, TrackerSettings
from .devices.tracker_calibration import TrackerCalibration, apply_persisted_yaw
from .dora_bridge.wiring import DoraWiring
from .errors import GelloUnavailableError, MaintenanceUnavailableError
from .gello import FOLLOWER_ARM_ID
from .gello.calibration import GelloCalibrationStore, match_arm_offsets
from .gello.preview import GelloPreviewService
from .session.manager import SessionManager
from .streams.hub import VideoHub
from .streams.twin_overlay import TwinOverlayRenderer


class Runtime:
    """Process-singleton: bus, video hub, profile store, tracker reader,
    microphone reader, GELLO leader reader (phase-15), hardware probe, read-only
    hardware monitor + twin alignment overlays (phase-09a), tracker calibration
    FSM, session manager."""

    def __init__(self, cfg: RuntimeConfig, *, monitor_factory=None) -> None:
        """``monitor_factory`` is the phase-09a test seam: replaces the hardware
        package's ``ArmStateMonitor`` class for the read-only monitor."""
        self.cfg = cfg
        self.epoch = uuid.uuid4().hex  # new epoch per process; UI detects restarts
        self.bus = RuntimeBus()
        self.hub = VideoHub(self.bus, jpeg_quality=cfg.video.jpeg_quality)
        self.profile_store = ProfileStore(cfg.profiles_dir)
        # Tracker device + live settings live for the whole process (13-tracker
        # §4): telemetry shows the device before any session exists. A persisted,
        # still-valid yaw alignment (calibration_dir/tracker_calibration.json)
        # overrides the YAML default before the settings are seeded (phase-10).
        apply_persisted_yaw(cfg)
        self.tracker_settings = TrackerSettings.from_config(cfg.tracker)
        self.tracker = TrackerReader(cfg.tracker, self.bus.tracker)
        if cfg.tracker.backend != "none":
            self.tracker.start()
        # Perception Arm microphone (phase-11): Runtime-owned like the tracker so the
        # Hardware tab shows the live waveform with no session and no arms.
        # Frames are aligned to telemetry_hz (one new frame per telemetry tick).
        self.microphone = MicrophoneReader(
            cfg.microphone, self.bus.microphone, frame_hz=cfg.telemetry_hz
        )
        self.microphone.start()  # no-op unless enabled with a backend
        # GELLO leader arm (phase-15; 16-gello §4 / D2): Runtime-owned like the tracker so
        # GET /api/gello and telemetry.gello show the leader before any session exists.
        # Started iff backend != none; the calibration store is the file the session-less
        # POST /api/gello/calibrate ops write and the reader maps with.
        self.gello_calibration = GelloCalibrationStore(cfg.gello.calibration_path)
        self.gello = GelloReader(
            cfg.gello, self.bus.gello, calibration_store=self.gello_calibration
        )
        if cfg.gello.backend != "none":
            self.gello.start()
        # Hardware reachability probe (phase-11): TCP 502 connect-and-close per
        # configured hardware arm; paused while a hardware session runs.
        hw = cfg.workcell_config("hardware")
        self.hardware_probe = HardwareProbe(
            {a.id: a.ip for a in hw.arms} if hw is not None else {},
            cfg.hardware_probe,
            paused=self._hardware_session_active,
        )
        # Read-only controller state monitor (phase-09a): one read-only SDK client
        # per hardware arm, session-less; PAUSED (connections released) while a
        # hardware session owns the boxes. Started in start() after the previews.
        # phase-09c: the monitor's ``home_rail`` op is gated by a full-travel sweep
        # of a dedicated digital twin (RailSweepChecker; built lazily on first use).
        self.rail_sweep: RailSweepChecker | None = None
        if hw is not None and hw.digital_twin_scene:
            self.rail_sweep = RailSweepChecker(
                hw,
                hw.digital_twin_scene,
                inflation_m=cfg.hardware_session.home_rail_inflation_m,
                step_m=cfg.hardware_session.home_rail_step_m,
                rail_flip=cfg.hardware_session.rail_flip,
            )
        self.hardware_monitor = HardwareStateMonitor(
            cfg.hardware_monitor,
            hw,
            paused=lambda: self._hardware_session_active(),  # late-bound (tests patch it)
            monitor_factory=monitor_factory,
            rail_sweep=self.rail_sweep,
            rail_fallback_m=cfg.twin_overlay.rail_fallback_m,
        )
        self.manager = SessionManager(
            cfg,
            self.bus,
            self.hub,
            self.profile_store,
            self.epoch,
            tracker_settings=self.tracker_settings,
            hardware_probe=self.hardware_probe,
            hardware_monitor=self.hardware_monitor,
        )
        # Data collection (04-runtime §10.5, D8): the recorder builds its per-episode
        # audio sink from THIS reader; without this line no episode ever gets audio.
        self.manager.microphone = self.microphone
        # GELLO Manipulation (phase-15; 16-gello §5): the manager's launch check and the
        # session-less POST /api/gello/preview share ONE cached kitchen twin per (kind,
        # scene) and the leader reader; the hardware posture comes from the read-only
        # monitor's sample (rail flip / fallback like the frozen twin arms).
        self.gello_preview = GelloPreviewService(
            cfg, self.gello, hardware_arm_q=self._gello_hardware_arm_q
        )
        self.manager.gello_reader = self.gello
        self.manager.gello_preview = self.gello_preview
        # phase-09d: home_rail with a planned pre-positioning motion. The service decides
        # dry-run / synchronous 09c homing / asynchronous RailHomingJob (202) / refused,
        # registers itself as the monitor's ``jobs`` (maintenance_busy + progress) and
        # keeps the final result per arm (GET .../maintenance/last).
        self.rail_homing = RailHomingService(
            cfg, self.hardware_monitor, self.rail_sweep, self.manager
        )
        # Twin alignment overlays (phase-09a): <camera_id>_align streams built from
        # the manager's hardware camera previews + the monitor's samples. phase-15
        # (16-gello D6 / §10): ``twin_overlay.scene`` overrides the workcell's twin scene
        # (the lab render points it at ``mavis_v2_kitchen`` so the appliance outlines can
        # be aligned session-less); the gate / sweep twins are NOT affected.
        self.twin_overlay: TwinOverlayRenderer | None = None
        if hw is not None:
            self.twin_overlay = TwinOverlayRenderer(
                cfg.twin_overlay,
                hw,
                cfg.twin_overlay.scene or hw.digital_twin_scene,
                self.hardware_monitor,
                self.manager.hardware_camera_frame,
                self.hub,
                paused=lambda: self._hardware_session_active(),
            )
            self.manager.twin_overlay = self.twin_overlay
            self.rail_homing.twin_overlay = self.twin_overlay
        self.hardware_probe.start()  # no-op without a hardware workcell
        # External interface over dora (phase-12; 14-dora §2): process-lifetime bridge +
        # publishers + idle arm reader. Inert (state "disabled") unless cfg.dora.enabled.
        self.dora = DoraWiring(self)
        self.manager.dora = self.dora
        # Calibration wizard back end (13-tracker §4 "Calibration modes"): Runtime-
        # owned, session-less; REST /api/tracker/calibration + telemetry.
        self.tracker_calibration = TrackerCalibration(
            self.tracker,
            self.tracker_settings,
            cfg,
            self.bus.tracker,
            lambda: self.manager.session_active,  # True during bringup too (rest.py re-checks)
        )
        self.controller_connected = False  # maintained by server/ws_control

    # -- lifecycle (server lifespan) ------------------------------------------
    def start(self) -> None:
        # Unclean prior shutdown: an episode that was being recorded when the process
        # died is one ``episodes/.tmp-*`` directory; sweep it before serving
        # (10-frames §11.6 step 5) - under the generic datasets_root AND every mapped
        # namespace root (15-online-dagger §7, D5), the same layout the store lists.
        # Filesystem only, lerobot stays unimported.
        try:
            self.manager.dataset_store.sweep()
        except Exception:  # never block serving on sweep problems
            import logging

            logging.getLogger(__name__).exception("dataset startup sweep failed")
        self.manager.start_previews()
        # Phase-09a: monitor -> overlay, after the hardware camera previews exist
        # (the overlay composites onto their frames). Both no-ops when inert.
        self.hardware_monitor.start()
        if self.twin_overlay is not None:
            self.twin_overlay.start()
        # phase-12: after the previews exist (camera taps attach to their encoders) and
        # the monitor is up (the hardware idle reader re-publishes its samples).
        self.dora.start()

    def stop(self) -> None:
        # Reverse of start(): dora (idle reader -> dora stop -> dora down -> reap) ->
        # overlay (renderers closed on its thread) -> monitor (boxes released) -> the rest.
        self.dora.stop()
        if self.twin_overlay is not None:
            self.twin_overlay.stop()
        self.rail_homing.stop()  # a homing in flight is never interrupted; wait for it
        self.hardware_monitor.stop()
        self.manager.teardown()
        self.manager.stop_previews()
        self.manager.stop_hardware_previews()
        self.hub.stop()
        self.tracker_calibration.close()  # restores normal libsurvive args if mid-calibration
        self.tracker.stop()
        self.hardware_probe.stop()
        self.microphone.stop()
        self.gello_preview.close()  # the render thread's GL contexts, closed in-thread
        self.gello.stop()

    def _hardware_session_active(self) -> bool:
        """Hand-over predicate for the monitor / probe / overlay: a hardware
        session exists OR ``create()`` is bringing one up (phase-09c: true from
        the first line of a hardware ``create()``, otherwise the monitor's 0.5 s
        supervisor round would reconnect the boxes in the middle of bring-up) OR
        a rail-homing job's driver holds a control box (phase-09d)."""
        return self.manager.hardware_session_active or self.rail_homing.owns_boxes

    # -- arm maintenance routing (REST; phase-09b/09c, 04-runtime §13.1) -------------------
    def arm_maintenance(
        self, arm_id: str, op: str, *, dry_run: bool = False
    ) -> ArmMaintenanceResult:
        """``POST /api/hardware/arms/{arm_id}/maintenance``: unknown hardware arm ->
        ``KeyError`` (404); a hardware session owns the boxes -> the session path
        (``clear_errors`` / ``recover`` = the driver's user recovery,
        ``apply_backstops`` and ``home_rail`` refused); otherwise the read-only
        monitor's maintenance queue (``recover`` refused: "no hardware session -
        use clear_errors"). ``home_rail`` (phase-09c) is THE ONE op that moves a
        mechanical part - the carriage drives to the track's zero end - so it is
        session-less only ("end the session first"), twin-gated (``rail_sweep``)
        and routed to the :class:`RailHomingService` (phase-09d): ``dry_run``
        returns the sweep verdict + ``pre_position`` (zero writes); a sweep-clear
        posture homes synchronously on the monitor's thread (<= 45 s, ``status:
        done``); a posture that needs the planned pre-positioning motion starts a
        ``RailHomingJob`` (``status: accepted`` + ``job_id`` -> REST 202); no
        plan -> ``status: refused``. While a job runs EVERY op is refused ("rail
        homing in progress"). Refusals raise :class:`MaintenanceUnavailableError`
        (409). Blocks on the calling (threadpool) thread; the SDK work happens on
        the driver's / monitor's / job's own thread."""
        from .session.hardware import arm_label

        hw = self.cfg.workcell_config("hardware")
        if hw is None or all(a.id != arm_id for a in hw.arms):
            raise KeyError(arm_id)
        busy_arm = self.rail_homing.active_arm  # phase-09d: a job owns the cell
        if busy_arm is not None:
            raise MaintenanceUnavailableError(
                f"{op} refused: rail homing in progress on the {arm_label(busy_arm)} - "
                "wait for it to finish"
            )
        if self._hardware_session_active():
            if op == "home_rail":
                raise MaintenanceUnavailableError(
                    "home_rail is not available while a hardware session owns the arms - "
                    "end the session first"
                )
            return self.manager.session_recovery(arm_id, op, timeout_s=MAINTENANCE_TIMEOUT_S)
        if op == "home_rail" and not dry_run and not self.cfg.hardware_session.armed:
            raise MaintenanceUnavailableError(
                "home_rail refused: hardware not armed (set hardware_session.armed: true in the "
                "lab config; the repo default is false so tests and dev instances never move a "
                "rail)"
            )
        if op == "home_rail":
            # phase-09d: the maintenance motion excludes EVERY session (contract 注意事项 1),
            # including a sim one and a create() still validating (its kind is not
            # recorded yet, so _hardware_session_active() is false in that window)
            if self.manager.session_active:
                raise MaintenanceUnavailableError(
                    "home_rail is not available while a session is active or starting - "
                    "end the session first"
                )
            return self.rail_homing.request(arm_id, dry_run=dry_run)
        return self.hardware_monitor.maintenance(
            arm_id, op, timeout_s=MAINTENANCE_TIMEOUT_S, dry_run=dry_run
        )

    def last_maintenance(self, arm_id: str) -> ArmMaintenanceResult | None:
        """``GET /api/hardware/arms/{arm_id}/maintenance/last`` (phase-09d): the
        final result of the last ``home_rail`` on this arm (a finished
        ``RailHomingJob`` or a synchronous homing); ``None`` -> 404. Unknown
        hardware arm -> ``KeyError`` (404)."""
        hw = self.cfg.workcell_config("hardware")
        if hw is None or all(a.id != arm_id for a in hw.arms):
            raise KeyError(arm_id)
        return self.rail_homing.last(arm_id)

    # -- external interface over dora (REST; 14-dora §2.6) ---------------------------------
    def dora_info(self):
        """``GET /api/dora``: connection facts for foreign clients (never the token)."""
        return self.dora.info()

    def dora_join(self, machine_id: str) -> bool:
        """``POST /api/dora/machines/{id}/join``: rescan the registered daemons now
        (False = the machine is not in ``dora.machines`` -> 404)."""
        return self.dora.request_join(machine_id)

    # -- session-less device discovery (REST; 04-runtime §13.1) -----------------------
    def microphone_infos(self) -> list[MicrophoneInfo]:
        """``GET /api/microphones``: the configured microphone is always listed
        (``live: false`` while absent / stalled / erroring); empty when
        ``microphone.enabled`` is false."""
        if not self.cfg.microphone.enabled:
            return []
        return [to_info(self.microphone.status())]

    # -- GELLO leader (REST; phase-15, 16-gello §4 / §9.2 / D10) ----------------------------
    def gello_scene_label(self) -> str:
        """Title of ``gello.scene_id`` from the sim registry (``mavis_v2_kitchen`` ->
        "APOLLO MAVIS V2 Kitchen (GELLO)"); the id itself when the sim extra / scene is
        missing."""
        scene_id = self.cfg.gello.scene_id
        try:
            from apollo_mavis_v2_sim import REGISTRY

            meta = REGISTRY.meta(scene_id)
        except Exception:  # noqa: BLE001 - no sim extra / unknown id: never a 500
            return scene_id
        return getattr(meta, "title", None) or getattr(meta, "description", None) or scene_id

    def gello_info(self) -> GelloInfo:
        """``GET /api/gello``: the device half of the GELLO block (16-gello §8.3) plus the
        launch-sheet facts - the twin scene the GELLO card launches, the Perception Arm's
        hold posture, the calibration file and its echo, ``hardware_admitted`` (D8)."""
        st = self.gello.status()
        g = self.cfg.gello
        return GelloInfo(
            **device_telemetry_fields(st),
            scene_id=g.scene_id,
            scene_label=self.gello_scene_label(),
            view_posture_rad=[float(x) for x in g.view_posture_rad],
            view_rail_m=float(g.view_rail_m),
            calibration_path=str(g.calibration_path),
            hardware_admitted=True,  # 16-gello D8: gello is admitted on hardware
            gripper_open_rad=st.gripper_open_rad,
            gripper_closed_rad=st.gripper_closed_rad,
        )

    def _gello_monitor_sample(self, arm: str):
        """The read-only monitor's FRESH sample of ``arm`` for the GELLO preview / launch
        check / ``match_arm`` on the hardware kind: ``(sample, "")`` only while the monitor
        reports a connected status (``CONNECTED_STATUSES``, the ``_validate_hardware`` rule)
        AND the sample is younger than ``hardware_monitor.stale_s``; else ``(None, why)``.
        2026-09-09 review: the hardware ``ArmStateMonitor`` never clears its last sample,
        so after a lost box / a paused monitor ``snapshot()`` kept returning an OUTDATED
        posture and the preview said ``clear`` while the launch 409'd (``match_arm`` would
        have stored offsets against a stale posture)."""
        monitor = self.hardware_monitor
        status, detail = monitor.status_of(arm)
        sample = monitor.snapshot().get(arm)
        t_mono = getattr(sample, "t_mono", None) if sample is not None else None
        age = None if t_mono is None else time.monotonic() - float(t_mono)
        stale_s = float(self.cfg.hardware_monitor.stale_s)
        if (
            status not in CONNECTED_STATUSES
            or sample is None
            or getattr(sample, "q", None) is None
            or age is None
            or age > stale_s
        ):
            if sample is not None and age is not None and age > stale_s:
                detail = (detail + "; " if detail else "") + f"sample {age:.1f} s old"
            return None, (
                "no fresh monitor sample of the Manipulation Arm "
                f"(monitor {status}{': ' + detail if detail else ''})"
            )
        return sample, ""

    def _gello_hardware_arm_q(self) -> tuple[np.ndarray | None, str]:
        """The Manipulation Arm's CURRENT joints + rail slot (twin convention) from the
        read-only monitor's newest FRESH sample (:meth:`_gello_monitor_sample`), for the
        GELLO launch check / preview on the hardware kind (16-gello §5.1 / §5.4);
        ``(None, why)`` when unknown - the preview then reports ``no_workcell``."""
        from .session.hardware import frozen_state

        arm = FOLLOWER_ARM_ID
        hw = self.cfg.workcell_config("hardware")
        if hw is None:
            return None, "no hardware workcell is configured"
        sample, why = self._gello_monitor_sample(arm)
        if sample is None:
            return None, why
        has_rail = True
        scene_id = self.cfg.gello.scene_id
        try:
            from apollo_mavis_v2_sim import REGISTRY

            has_rail = bool(REGISTRY.meta(scene_id).rail.get(arm, True))
        except Exception:  # noqa: BLE001 - no sim extra: assume the lab's railed arm
            pass
        state, _note = frozen_state(
            arm,
            sample,
            has_rail=has_rail,
            rail_fallback_m=self.cfg.twin_overlay.rail_fallback_m,
            rail_flip=self.cfg.hardware_session.rail_flip,
        )
        return np.asarray(state.q, dtype=np.float64), ""

    def gello_preview_result(self, req: GelloPreviewRequest) -> GelloPreviewResult:
        """``POST /api/gello/preview`` (16-gello §5.4): the launch check on the cached kitchen
        twin + the PNG; 200 with the status for every posture, :class:`GelloUnavailableError`
        (409) only when the runtime has no workcell of ``req.kind``. Never moves anything."""
        return self.gello_preview.preview(req)

    def _gello_arm_joints(self, kind: str) -> np.ndarray:
        """The Manipulation Arm's CURRENT 7 joints for ``match_arm``: the read-only
        monitor's newest FRESH sample on hardware (:meth:`_gello_monitor_sample`), the
        parked / idle sim posture the previews show in sim. Raises
        :class:`GelloUnavailableError` when unknown."""
        arm = FOLLOWER_ARM_ID
        if kind == "hardware":
            if self.cfg.workcell_config("hardware") is None:
                raise GelloUnavailableError("no hardware workcell is configured")
            sample, why = self._gello_monitor_sample(arm)
            if sample is None:
                raise GelloUnavailableError(f"{why} - nothing to match against")
            return np.asarray(sample.q, dtype=np.float64)[:7]
        if self.cfg.workcell_config("sim") is None:
            raise GelloUnavailableError("no sim workcell is configured")
        q = self.manager.idle_sim_q().get(arm)
        if q is None:
            raise GelloUnavailableError(
                "the sim workcell has no Manipulation Arm posture to match against "
                f"(arm {arm!r} not in the preview scene, or the previews are not running)"
            )
        return np.asarray(q, dtype=np.float64)[:7]

    def gello_calibrate(self, req: GelloCalibrateRequest) -> GelloCalibrateResult:
        """``POST /api/gello/calibrate`` (16-gello §4 / D10), session-less.

        Refused (:class:`GelloUnavailableError`, 409) while a session exists or is
        starting, and - for the three reading ops - when the leader has no fresh sample
        (``clear`` needs no leader). The reading ops accept an UNCALIBRATED sample
        (``fresh_sample(require_calibrated=False)``: they exist to create the calibration
        - 2026-09-09 review; a stale or jump-flagged sample still 409s). ``match_arm`` stores
        ``round((raw - sign * q_arm) / (pi/2)) * pi/2`` per joint from the leader's raw
        joints and the Manipulation Arm's current joints (``kind`` picks the source);
        ``gripper_open`` / ``gripper_closed`` store the raw gripper reading; ``clear``
        deletes the file. The reader reloads the file at once; the result echoes the
        calibration now in force (also in ``GET /api/gello``)."""
        if self.manager.session_active:
            raise GelloUnavailableError(
                "GELLO calibration is not available while a session is active or starting - "
                "end the session first"
            )
        store = self.gello_calibration
        if req.op == "clear":
            existed = store.clear()
            self.gello.reload_calibration()
            return GelloCalibrateResult(
                ok=True,
                detail=(
                    f"calibration cleared ({store.path})"
                    if existed
                    else f"no calibration file to clear ({store.path})"
                ),
            )
        sample = self.gello.fresh_sample(require_calibrated=False)
        if sample is None:
            st = self.gello.status()
            raise GelloUnavailableError(
                f"no fresh GELLO leader sample (status {st.status}"
                f"{': ' + st.detail if st.detail else ''})"
            )
        if req.op == "match_arm":
            q_arm = self._gello_arm_joints(req.kind)
            offsets = match_arm_offsets(sample.q_raw, q_arm, self.cfg.gello.joint_signs)
            cal = store.update(joint_offsets_rad=tuple(float(x) for x in offsets))
            self.gello.reload_calibration()
            residual = np.asarray(self.cfg.gello.joint_signs, dtype=np.float64) * (
                sample.q_raw - offsets
            ) - q_arm
            detail = (
                f"joint offsets stored ({store.path}); residual vs the {req.kind} Manipulation "
                f"Arm max {float(np.max(np.abs(residual))):.3f} rad"
            )
            if self.cfg.gello.joint_offsets_rad is not None:
                detail += (
                    " - NOTE gello.joint_offsets_rad is set in the config and overrides the file"
                )
        else:  # gripper_open / gripper_closed
            if sample.gripper_raw is None:
                raise GelloUnavailableError(
                    "the leader has no gripper channel (gello.gripper_id is null)"
                )
            field = "gripper_open_rad" if req.op == "gripper_open" else "gripper_closed_rad"
            cal = store.update(**{field: float(sample.gripper_raw)})
            self.gello.reload_calibration()
            detail = f"{field} = {float(sample.gripper_raw):.4f} rad stored ({store.path})"
        return GelloCalibrateResult(
            ok=True,
            detail=detail,
            joint_offsets_rad=(
                [float(x) for x in cal.joint_offsets_rad]
                if cal.joint_offsets_rad is not None
                else None
            ),
            gripper_open_rad=cal.gripper_open_rad,
            gripper_closed_rad=cal.gripper_closed_rad,
        )

    # -- control-WS plumbing (no motion work here; 04-runtime §13.2) -------------
    def on_keys(self, seq: int, held: list[str]) -> None:
        """Controller KeysMsg: seq-checked, stamped with SERVER rx time."""
        session = self.manager.session
        if session is None:
            return
        state = HeldState(held=frozenset(held), seq=seq, rx_mono=time.monotonic())
        if session.supervisor.watchdog.on_keys(state):  # False = stale seq: drop
            self.bus.held_keys.put(state)

    def on_controller_disconnect(self) -> None:
        self.controller_connected = False
        session = self.manager.session
        if session is not None:
            session.supervisor.watchdog.on_disconnect()
            session.loop.note_controller_disconnect()  # cancels an interruptible plan

    def submit_action(self, name: str, args: dict) -> Future[CommandResult]:
        return self.bus.commands.submit(Command(op=name, args=dict(args), source="ws"))


__all__ = ["Runtime"]
