"""Runtime — the composition root; owns everything (04-runtime §2)."""

from __future__ import annotations

import time
import uuid
from concurrent.futures import Future

from apollo_mavis_v2_core import Command, CommandResult, HeldState, ProfileStore
from apollo_mavis_v2_core.protocol import MicrophoneInfo

from .bus import RuntimeBus
from .config import RuntimeConfig
from .devices.hardware_monitor import HardwareStateMonitor
from .devices.hardware_probe import HardwareProbe
from .devices.microphone import MicrophoneReader, to_info
from .devices.tracker import TrackerReader, TrackerSettings
from .devices.tracker_calibration import TrackerCalibration, apply_persisted_yaw
from .session.manager import SessionManager
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
        self.hardware_monitor = HardwareStateMonitor(
            cfg.hardware_monitor, hw,
            paused=lambda: self._hardware_session_active(),  # late-bound (tests patch it)
            monitor_factory=monitor_factory,
        )
        self.manager = SessionManager(
            cfg, self.bus, self.hub, self.profile_store, self.epoch,
            tracker_settings=self.tracker_settings,
            hardware_probe=self.hardware_probe,
            hardware_monitor=self.hardware_monitor,
        )
        # Twin alignment overlays (phase-09a): <camera_id>_align streams built from
        # the manager's hardware camera previews + the monitor's samples.
        self.twin_overlay: TwinOverlayRenderer | None = None
        if hw is not None:
            self.twin_overlay = TwinOverlayRenderer(
                cfg.twin_overlay, hw, hw.digital_twin_scene, self.hardware_monitor,
                self.manager.hardware_camera_frame, self.hub,
                paused=lambda: self._hardware_session_active(),
            )
            self.manager.twin_overlay = self.twin_overlay
        self.hardware_probe.start()  # no-op without a hardware workcell
        # Calibration wizard back end (13-tracker §4 "Calibration modes"): Runtime-
        # owned, session-less; REST /api/tracker/calibration + telemetry.
        self.tracker_calibration = TrackerCalibration(
            self.tracker, self.tracker_settings, cfg, self.bus.tracker,
            lambda: self.manager.session_active,  # True during bringup too (rest.py re-checks)
        )
        self.controller_connected = False  # maintained by server/ws_control

    # -- lifecycle (server lifespan) ------------------------------------------
    def start(self) -> None:
        # Unclean prior shutdown: resume() + finalize() unfinalized datasets
        # before serving (04-runtime §15). Filesystem scan only unless a
        # repair is actually needed (lerobot stays unimported).
        try:
            from .recorder.episode_recorder import repair_unfinalized_datasets

            repair_unfinalized_datasets(self.cfg.datasets_root)
        except Exception:  # never block serving on repair problems
            import logging

            logging.getLogger(__name__).exception("dataset startup repair failed")
        self.manager.start_previews()
        # Phase-09a: monitor -> overlay, after the hardware camera previews exist
        # (the overlay composites onto their frames). Both no-ops when inert.
        self.hardware_monitor.start()
        if self.twin_overlay is not None:
            self.twin_overlay.start()

    def stop(self) -> None:
        # Reverse of start(): overlay (renderers closed on its thread) -> monitor
        # (boxes released) -> the rest.
        if self.twin_overlay is not None:
            self.twin_overlay.stop()
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
        session = self.manager.session
        return session is not None and session.spec.kind == "hardware"

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

    def submit_action(self, name: str, args: dict) -> Future[CommandResult]:
        return self.bus.commands.submit(Command(op=name, args=dict(args), source="ws"))


__all__ = ["Runtime"]
