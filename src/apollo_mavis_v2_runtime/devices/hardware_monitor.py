"""Read-only hardware state monitor (phase-09a; 04-runtime §13.3 ``hardware_monitor``).

``Runtime`` owns one :class:`HardwareStateMonitor` when a ``hardware``
workcell is configured. It wraps one
:class:`apollo_mavis_v2_hardware.ArmStateMonitor` per configured arm — a
read-only SDK client that polls joint angles, flange pose, error/warn codes,
controller state/mode, linear-track and gripper registers and NEVER sends a
motion / mode / state / ``clean_*`` / ``set_*`` command (the allowlist lives in
the hardware module; its tests assert zero writes against a call-logging fake).
The samples feed the Welcome page's real error codes
(``ArmStatusInfo.error_code``), the telemetry block and the digital-twin
alignment overlays (``streams/twin_overlay.py``).

**Pause = release the connection.** A supervisor thread polls ``paused()``
(``Runtime._hardware_session_active``) every ``check_period_s``: when it turns
true every arm monitor is ``disconnect()``-ed (status ``paused``, last sample
kept); when it turns false they are ``start()``-ed again. Two SDK clients on
one control box are unevidenced, so a hardware session and the monitor never
hold the same box at the same time; :meth:`pause` / :meth:`resume` are the
synchronous seams a hardware bring-up (phase-09) calls around its connect.

Without the ``[hardware]`` extra (or with ``enabled: false``) the monitor is
inert: ``enabled`` is False, every arm reports status ``off`` with a
``detail`` that says why, and ``telemetry()`` still returns a valid block.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

from apollo_mavis_v2_core import WorkcellConfig
from apollo_mavis_v2_core.protocol import ArmMonitorTelemetry, HardwareMonitorTelemetry

from ..config import HardwareMonitorConfig

logger = logging.getLogger(__name__)

CHECK_PERIOD_S = 0.5  # paused() is polled this often
STOP_JOIN_TIMEOUT_S = 5.0


class ArmMonitorLike(Protocol):
    """Surface of ``apollo_mavis_v2_hardware.ArmStateMonitor`` the runtime relies on
    (also what a test fake must provide)."""

    arm_id: str

    def start(self) -> None: ...

    def stop(self, timeout: float = 2.0) -> None: ...

    def disconnect(self, timeout: float = 2.0) -> bool | None: ...  # False = not yet released

    def snapshot(self) -> Any: ...  # ArmMonitorSample | None

    @property
    def status(self) -> str: ...

    @property
    def detail(self) -> str: ...

    @property
    def age_s(self) -> float | None: ...


MonitorFactory = Callable[..., ArmMonitorLike]
"""``factory(arm_id, ip, *, gripper, expect_rail, poll_hz, stale_s, reconnect_s)``."""


def default_monitor_factory() -> MonitorFactory:
    """The hardware package's ``ArmStateMonitor`` (raises ``ImportError`` without
    the ``[hardware]`` extra)."""
    from apollo_mavis_v2_hardware import ArmStateMonitor  # [hardware] extra

    return ArmStateMonitor


def sample_to_telemetry(
    arm_id: str, status: str, detail: str, age_s: float | None, sample: Any
) -> ArmMonitorTelemetry:
    """Core ``ArmMonitorTelemetry`` from a monitor's status + its last
    ``ArmMonitorSample`` (``None`` -> data fields at their defaults)."""
    if sample is None:
        return ArmMonitorTelemetry(arm_id=arm_id, status=status, detail=detail, age_s=age_s)
    return ArmMonitorTelemetry(
        arm_id=arm_id,
        status=status,
        detail=detail,
        seq=int(sample.seq),
        age_s=age_s,
        q=[float(v) for v in sample.q],
        tcp_pose=[float(v) for v in sample.tcp_pose],
        rail_present=sample.rail_present,
        rail_homed=sample.rail_homed,
        rail_enabled=sample.rail_enabled,
        rail_pos_m=sample.rail_pos_m,
        rail_raw_mm=sample.rail_raw_mm,
        gripper_open_frac=sample.gripper_open_frac,
        gripper_raw=sample.gripper_raw,
        error_code=int(sample.error_code),
        warn_code=int(sample.warn_code),
        state=sample.state,
        mode=sample.mode,
    )


class HardwareStateMonitor:
    """One read-only ``ArmStateMonitor`` per configured hardware arm + the
    pause/resume supervisor (see module docstring).

    ``workcell`` is the ``hardware`` workcell config (``None`` = no hardware
    configured -> inert). ``paused()`` is the hand-over predicate. Test seams:
    ``monitor_factory`` (default: the hardware package's ``ArmStateMonitor``),
    ``check_period_s``, ``clock``.
    """

    def __init__(
        self,
        cfg: HardwareMonitorConfig,
        workcell: WorkcellConfig | None,
        paused: Callable[[], bool] | None = None,
        *,
        monitor_factory: MonitorFactory | None = None,
        check_period_s: float = CHECK_PERIOD_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self.workcell = workcell
        self.paused_fn = paused or (lambda: False)
        self._factory = monitor_factory
        self.check_period_s = float(check_period_s)
        self._clock = clock
        self.arm_ids: list[str] = [a.id for a in workcell.arms] if workcell is not None else []
        self._monitors: dict[str, ArmMonitorLike] = {}
        self._disabled_detail = ""  # why the monitor is inert ("" = it is not)
        self._paused = False
        self._lock = threading.Lock()
        # Serializes whole pause/resume transitions: the supervisor edge and the
        # synchronous pause()/resume() seams must not interleave their per-arm
        # disconnect()/start() walks (a start() racing a still-finishing
        # disconnect() would leave that arm paused while `paused` reads False).
        self._apply_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self.transitions = 0  # pause/resume edges applied (tests)
        if workcell is None:
            self._disabled_detail = "no hardware workcell configured"
        elif not cfg.enabled:
            self._disabled_detail = "hardware monitor disabled (hardware_monitor.enabled: false)"
        else:
            self._build_monitors()

    # -- construction --------------------------------------------------------------------
    def _build_monitors(self) -> None:
        assert self.workcell is not None
        try:
            factory = self._factory or default_monitor_factory()
        except Exception as e:  # noqa: BLE001 - [hardware] extra absent or broken
            self._disabled_detail = (
                f"hardware package not importable ({type(e).__name__}: {e}); "
                "install the [hardware] extra"
            )
            logger.info("hardware monitor off: %s", self._disabled_detail)
            return
        for arm in self.workcell.arms:
            if not arm.ip:
                continue
            try:
                self._monitors[arm.id] = factory(
                    arm.id,
                    arm.ip,
                    gripper=arm.gripper,
                    expect_rail=arm.expect_rail != "no",  # auto | yes -> read the registers
                    poll_hz=self.cfg.poll_hz,
                    stale_s=self.cfg.stale_s,
                    reconnect_s=self.cfg.reconnect_s,
                )
            except Exception as e:  # noqa: BLE001 - one bad arm must not kill the rest
                logger.warning("hardware monitor for %r not created: %r", arm.id, e)
        if not self._monitors:
            self._disabled_detail = "no hardware arm with an ip to monitor"

    # -- lifecycle -----------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        """Monitors exist and may connect (config on, hardware extra importable,
        at least one arm with an ip)."""
        return not self._disabled_detail and bool(self._monitors)

    @property
    def detail(self) -> str:
        """Why the monitor is inert ('' when enabled)."""
        return self._disabled_detail

    @property
    def paused(self) -> bool:
        """Last applied hand-over state (connections released)."""
        with self._lock:
            return self._paused

    def start(self) -> None:
        """Start every arm monitor (unless ``paused()`` already holds) and the
        supervisor thread. No-op when inert or already started."""
        if not self.enabled or self._thread is not None:
            return
        self._stop.clear()
        self._started = True
        self._apply_paused(bool(self._safe_paused()), initial=True)
        self._thread = threading.Thread(target=self._run, name="hardware-monitor", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = STOP_JOIN_TIMEOUT_S) -> None:
        """Stop the supervisor and every arm monitor (boxes released, status
        ``off``). Idempotent."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("hardware monitor thread did not stop within %.1f s", timeout)
            self._thread = None
        for mon in self._monitors.values():
            try:
                mon.stop()
            except Exception:  # noqa: BLE001 - release never raises
                logger.exception("arm monitor %r did not stop cleanly", mon.arm_id)
        self._started = False

    def pause(self) -> None:
        """Synchronous hand-over: release every box now (status ``paused``).
        The supervisor keeps it released while ``paused()`` stays true."""
        self._apply_paused(True)

    def resume(self) -> None:
        """Synchronous resume after a hand-over (reconnects every arm)."""
        if self._started:
            self._apply_paused(False)

    def _safe_paused(self) -> bool:
        try:
            return bool(self.paused_fn())
        except Exception:  # noqa: BLE001 - predicate errors never stop the monitor
            logger.exception("hardware monitor paused() predicate failed")
            return False

    def _apply_paused(self, flag: bool, *, initial: bool = False) -> None:
        with self._apply_lock:
            with self._lock:
                if not initial and flag == self._paused:
                    return
                self._paused = flag
                self.transitions += 0 if initial else 1
            for mon in self._monitors.values():
                try:
                    if flag:
                        # False = the poll thread is still inside an SDK call; it
                        # releases the box itself on return and never re-publishes
                        # (hardware monitor.py hand-over guarantee).
                        if mon.disconnect() is False:
                            logger.warning(
                                "arm monitor %r: box not yet released (thread still in the "
                                "SDK); it is released when the call returns",
                                mon.arm_id,
                            )
                    else:
                        mon.start()
                except Exception:  # noqa: BLE001 - one arm must not block the others
                    logger.exception(
                        "arm monitor %r %s failed", mon.arm_id, "disconnect" if flag else "start"
                    )

    def _run(self) -> None:
        while not self._stop.wait(self.check_period_s):
            try:
                self._apply_paused(self._safe_paused())
            except Exception:  # last line of defence
                logger.exception("hardware monitor supervisor round failed")

    # -- readers -------------------------------------------------------------------------
    def monitor(self, arm_id: str) -> ArmMonitorLike | None:
        return self._monitors.get(arm_id)

    def snapshot(self) -> dict[str, Any]:
        """``{arm_id: ArmMonitorSample}`` for every arm that has a sample."""
        out: dict[str, Any] = {}
        for arm_id, mon in self._monitors.items():
            sample = mon.snapshot()
            if sample is not None:
                out[arm_id] = sample
        return out

    def status_of(self, arm_id: str) -> tuple[str, str]:
        """``(status, detail)`` of one arm; ``("off", <why>)`` when inert / unknown."""
        mon = self._monitors.get(arm_id)
        if mon is None:
            return "off", self._disabled_detail or f"arm {arm_id!r} is not monitored"
        if not self._started:
            return "off", "monitor not started"
        return str(mon.status), str(mon.detail)

    def error_code(self, arm_id: str) -> int:
        """Controller error code of the last sample (0 without one)."""
        mon = self._monitors.get(arm_id)
        sample = mon.snapshot() if mon is not None else None
        return int(sample.error_code) if sample is not None else 0

    def arm_telemetry(self) -> list[ArmMonitorTelemetry]:
        rows: list[ArmMonitorTelemetry] = []
        for arm_id in self.arm_ids:
            mon = self._monitors.get(arm_id)
            status, detail = self.status_of(arm_id)
            rows.append(
                sample_to_telemetry(
                    arm_id,
                    status,
                    detail,
                    mon.age_s if mon is not None else None,
                    mon.snapshot() if mon is not None else None,
                )
            )
        return rows

    def telemetry(self) -> HardwareMonitorTelemetry:
        """The ``arms`` part of ``telemetry.hardware_monitor`` (``overlays`` is
        filled by the caller from the ``TwinOverlayRenderer``)."""
        return HardwareMonitorTelemetry(
            enabled=self.enabled, paused=self.paused, arms=self.arm_telemetry()
        )


__all__ = [
    "ArmMonitorLike",
    "HardwareStateMonitor",
    "MonitorFactory",
    "default_monitor_factory",
    "sample_to_telemetry",
]
