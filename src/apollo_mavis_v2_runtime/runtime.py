"""Runtime — the composition root; owns everything (04-runtime §2)."""

from __future__ import annotations

import time
import uuid
from concurrent.futures import Future

from apollo_mavis_v2_core import Command, CommandResult, HeldState, ProfileStore
from apollo_mavis_v2_core.protocol import ArmMaintenanceResult, MicrophoneInfo

from .bus import RuntimeBus
from .config import RuntimeConfig
from .devices.hardware_monitor import MAINTENANCE_TIMEOUT_S, HardwareStateMonitor
from .devices.hardware_probe import HardwareProbe
from .devices.microphone import MicrophoneReader, to_info
from .devices.rail_homing import RailHomingService
from .devices.rail_sweep import RailSweepChecker
from .devices.tracker import TrackerReader, TrackerSettings
from .devices.tracker_calibration import TrackerCalibration, apply_persisted_yaw
from .dora_bridge.wiring import DoraWiring
from .errors import MaintenanceUnavailableError
from .session.manager import SessionManager
from .session.orphan import OrphanSessionWatch
from .streams.hub import VideoHub
from .streams.twin_overlay import TwinOverlayRenderer


class Runtime:
    """Process-singleton: bus, video hub, profile store, tracker reader,
    microphone reader, hardware probe, read-only hardware monitor + twin
    alignment overlays (phase-09a), tracker calibration FSM, session manager."""

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
        # phase-09d: home_rail with a planned pre-positioning motion. The service decides
        # dry-run / synchronous 09c homing / asynchronous RailHomingJob (202) / refused,
        # registers itself as the monitor's ``jobs`` (maintenance_busy + progress) and
        # keeps the final result per arm (GET .../maintenance/last).
        self.rail_homing = RailHomingService(
            cfg, self.hardware_monitor, self.rail_sweep, self.manager
        )
        # Twin alignment overlays (phase-09a): <camera_id>_align streams built from
        # the manager's hardware camera previews + the monitor's samples.
        self.twin_overlay: TwinOverlayRenderer | None = None
        if hw is not None:
            self.twin_overlay = TwinOverlayRenderer(
                cfg.twin_overlay,
                hw,
                hw.digital_twin_scene,
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
        # Orphaned-session watch (2026-09-09 evening; 04-runtime §13.2): a session whose
        # controller /ws/control connection has been gone for
        # control.orphan_session_grace_s is ended by the runtime itself through the
        # no-motion teardown, and why is published on telemetry.session.auto_ended.
        # `controller_connected` is read live rather than mirrored, so there is only ever
        # one copy of the connection state (ws_control's).
        self.orphan_watch = OrphanSessionWatch(
            self.manager,
            cfg.control.orphan_session_grace_s,
            lambda: self.controller_connected,
        )

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
        # 2026-09-09: last, so nothing can be orphaned before the process can serve
        # the Welcome page that explains it (no-op when the grace period is 0).
        self.orphan_watch.start()

    def stop(self) -> None:
        # Reverse of start(): the orphan watch first (its teardown must never race the
        # one below) -> dora (idle reader -> dora stop -> dora down -> reap) ->
        # overlay (renderers closed on its thread) -> monitor (boxes released) -> the rest.
        self.orphan_watch.stop()
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
