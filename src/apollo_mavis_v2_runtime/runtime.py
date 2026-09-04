"""Runtime — the composition root; owns everything (04-runtime §2)."""

from __future__ import annotations

import time
import uuid
from concurrent.futures import Future

from apollo_mavis_v2_core import Command, CommandResult, HeldState, ProfileStore

from .bus import RuntimeBus
from .config import RuntimeConfig
from .devices.tracker import TrackerReader, TrackerSettings
from .devices.tracker_calibration import TrackerCalibration, apply_persisted_yaw
from .session.manager import SessionManager
from .streams.hub import VideoHub


class Runtime:
    """Process-singleton: bus, video hub, profile store, tracker reader,
    tracker calibration FSM, session manager."""

    def __init__(self, cfg: RuntimeConfig) -> None:
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
        self.manager = SessionManager(
            cfg, self.bus, self.hub, self.profile_store, self.epoch,
            tracker_settings=self.tracker_settings,
        )
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

    def stop(self) -> None:
        self.manager.teardown()
        self.manager.stop_previews()
        self.hub.stop()
        self.tracker_calibration.close()  # restores normal libsurvive args if mid-calibration
        self.tracker.stop()

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
