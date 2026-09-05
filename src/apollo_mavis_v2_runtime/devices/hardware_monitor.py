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

**Maintenance channel (phase-09b; 04-runtime §13.1 / §15).** The zero-write
guarantee becomes "zero writes unless an explicit maintenance request":
:meth:`HardwareStateMonitor.maintenance` forwards the operator's
``clear_errors`` / ``apply_backstops`` to the arm's ``ArmStateMonitor``, which
queues it for ITS poll thread (one ``XArmAPI`` is never driven from two
threads; this thread only waits), and returns the core
``ArmMaintenanceResult`` with ``path="monitor"``. ``apply_backstops`` writes
the arm's ``ArmConfig`` safety parameters through the hardware package's own
``ArmConfig -> XArmDriverConfig`` mapping, so the UI button and the driver's
connect-time ``apply_backstops`` send identical values. Every slow poll the
monitor reads the controller's CURRENT ``collision_sensitivity`` / ``tcp_load``
back; :func:`backstops_match` compares them with the config (sensitivity
equal, |d load| <= 0.05 kg, |d cog| <= 10 mm) for ``ArmMonitorTelemetry`` and
the ``before`` / ``after`` rows of a result. ``recover`` (enable + servo mode)
needs a session driver and is refused here (409: "no hardware session - use
clear_errors"); so is any op while the monitor is off / paused / not connected
or another op is still running on that arm (:class:`MaintenanceUnavailableError`).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

from apollo_mavis_v2_core import ArmConfig, WorkcellConfig
from apollo_mavis_v2_core.protocol import (
    ArmMaintenanceResult,
    ArmMonitorTelemetry,
    HardwareMonitorTelemetry,
)

from ..config import HardwareMonitorConfig
from ..errors import MaintenanceUnavailableError

logger = logging.getLogger(__name__)

CHECK_PERIOD_S = 0.5  # paused() is polled this often
STOP_JOIN_TIMEOUT_S = 5.0
# phase-09b: backstops_match tolerances (contract; the hardware monitor's own settle
# check uses the same 0.05 kg) and the REST wait budget for one maintenance op.
TCP_LOAD_MATCH_KG = 0.05
TCP_COG_MATCH_MM = 10.0
MAINTENANCE_TIMEOUT_S = 10.0
MAINTENANCE_OPS: tuple[str, ...] = ("clear_errors", "apply_backstops", "recover")
CONNECTED_STATUSES: frozenset[str] = frozenset({"running", "stale"})  # a maintenance op may run


class ArmMonitorLike(Protocol):
    """Surface of ``apollo_mavis_v2_hardware.ArmStateMonitor`` the runtime relies on
    (also what a test fake must provide)."""

    arm_id: str

    def start(self) -> None: ...

    def stop(self, timeout: float = 2.0) -> None: ...

    def disconnect(self, timeout: float = 2.0) -> bool | None: ...  # False = not yet released

    def snapshot(self) -> Any: ...  # ArmMonitorSample | None

    def maintenance(
        self, op: str, driver_cfg: Any = None, timeout_s: float = 10.0
    ) -> Any: ...  # MaintenanceOutcome (phase-09b)

    @property
    def maintenance_busy(self) -> bool: ...

    @property
    def status(self) -> str: ...

    @property
    def detail(self) -> str: ...

    @property
    def age_s(self) -> float | None: ...


MonitorFactory = Callable[..., ArmMonitorLike]
"""``factory(arm_id, ip, *, gripper, expect_rail, poll_hz, stale_s, reconnect_s)``."""

DriverConfigFactory = Callable[[ArmConfig], Any]
"""``ArmConfig -> XArmDriverConfig`` (the hardware package's ``workcell._driver_cfg``)."""


def default_monitor_factory() -> MonitorFactory:
    """The hardware package's ``ArmStateMonitor`` (raises ``ImportError`` without
    the ``[hardware]`` extra)."""
    from apollo_mavis_v2_hardware import ArmStateMonitor  # [hardware] extra

    return ArmStateMonitor


def default_driver_cfg_factory() -> DriverConfigFactory:
    """The hardware package's ``ArmConfig -> XArmDriverConfig`` mapping — the ONE
    place the controller-side backstop parameters (tcp_load, collision
    sensitivity, reduced-mode boundary, expected SN) are translated, so the UI's
    ``apply_backstops`` and the driver's connect-time call write the same values
    (02-hardware §6). Raises ``ImportError`` without the ``[hardware]`` extra."""
    from apollo_mavis_v2_hardware.workcell import _driver_cfg  # [hardware] extra

    return _driver_cfg


def backstops_match(sample: Any, arm: ArmConfig) -> bool | None:
    """Controller read-back == ``ArmConfig`` backstops (contract tolerances):
    sensitivity equal, ``|d tcp_load| <= 0.05 kg`` and every centre-of-gravity
    component within 10 mm. ``None`` when the sample carries no read-back yet
    (older sample shape, or the monitor never read the rich report frame)."""
    if sample is None:
        return None
    sens = getattr(sample, "collision_sensitivity", None)
    kg = getattr(sample, "tcp_load_kg", None)
    if sens is None or kg is None:
        return None
    if int(sens) != int(arm.collision_sensitivity):
        return False
    eps = 1e-9  # the tolerances are inclusive; keep 0.95 + 0.05 vs 1.0 from flipping on rounding
    if abs(float(kg) - float(arm.tcp_load_kg)) > TCP_LOAD_MATCH_KG + eps:
        return False
    cog = tuple(getattr(sample, "tcp_load_cog_mm", ()) or ())
    if len(cog) != 3:
        return False
    return all(
        abs(float(a) - float(b)) <= TCP_COG_MATCH_MM + eps
        for a, b in zip(cog, arm.tcp_load_cog_mm, strict=True)
    )


def sample_to_telemetry(
    arm_id: str,
    status: str,
    detail: str,
    age_s: float | None,
    sample: Any,
    *,
    backstops_match: bool | None = None,
    maintenance_busy: bool = False,
) -> ArmMonitorTelemetry:
    """Core ``ArmMonitorTelemetry`` from a monitor's status + its last
    ``ArmMonitorSample`` (``None`` -> data fields at their defaults). The
    phase-09b read-backs are read duck-typed (absent on an older sample ->
    their defaults); ``backstops_match`` / ``maintenance_busy`` are the
    runtime's (:func:`backstops_match`, the monitor's busy flag)."""
    if sample is None:
        return ArmMonitorTelemetry(
            arm_id=arm_id,
            status=status,
            detail=detail,
            age_s=age_s,
            maintenance_busy=maintenance_busy,
        )
    sens = getattr(sample, "collision_sensitivity", None)
    kg = getattr(sample, "tcp_load_kg", None)
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
        collision_sensitivity=None if sens is None else int(sens),
        tcp_load_kg=None if kg is None else float(kg),
        tcp_load_cog_mm=[float(v) for v in (getattr(sample, "tcp_load_cog_mm", ()) or ())],
        backstops_match=backstops_match,
        maintenance_busy=maintenance_busy,
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
        driver_cfg_factory: DriverConfigFactory | None = None,
        check_period_s: float = CHECK_PERIOD_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self.workcell = workcell
        self.paused_fn = paused or (lambda: False)
        self._factory = monitor_factory
        self._driver_cfg_factory = driver_cfg_factory  # None = the hardware package's mapping
        self.check_period_s = float(check_period_s)
        self._clock = clock
        self.arm_ids: list[str] = [a.id for a in workcell.arms] if workcell is not None else []
        self._arm_cfgs: dict[str, ArmConfig] = (
            {a.id: a for a in workcell.arms} if workcell is not None else {}
        )
        self._monitors: dict[str, ArmMonitorLike] = {}
        # phase-09b: one maintenance op at a time per arm (a second concurrent
        # request is refused with 409 semantics instead of queueing behind the first)
        self._maintenance_locks: dict[str, threading.Lock] = {
            arm_id: threading.Lock() for arm_id in self.arm_ids
        }
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
        """Hand-over state: ``True`` while a hardware session owns the boxes.

        Enabled: the last state the supervisor APPLIED (connections released).
        Inert (``enabled: false`` / no hardware package): nothing to release, so
        the flag follows the predicate directly - the UI reads
        ``telemetry.hardware_monitor.paused`` as "a hardware session exists"
        (05-ui §8.2) and must not lose that signal with the monitor switched off."""
        if not self.enabled:
            return self._safe_paused()
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

    def maintenance_busy(self, arm_id: str) -> bool:
        """A maintenance op is queued / executing on this arm's monitor (phase-09b)."""
        mon = self._monitors.get(arm_id)
        return bool(getattr(mon, "maintenance_busy", False)) if mon is not None else False

    def arm_telemetry(self) -> list[ArmMonitorTelemetry]:
        rows: list[ArmMonitorTelemetry] = []
        for arm_id in self.arm_ids:
            mon = self._monitors.get(arm_id)
            status, detail = self.status_of(arm_id)
            sample = mon.snapshot() if mon is not None else None
            rows.append(
                sample_to_telemetry(
                    arm_id,
                    status,
                    detail,
                    mon.age_s if mon is not None else None,
                    sample,
                    backstops_match=backstops_match(sample, self._arm_cfgs[arm_id]),
                    maintenance_busy=self.maintenance_busy(arm_id),
                )
            )
        return rows

    def telemetry(self) -> HardwareMonitorTelemetry:
        """The ``arms`` part of ``telemetry.hardware_monitor`` (``overlays`` is
        filled by the caller from the ``TwinOverlayRenderer``)."""
        return HardwareMonitorTelemetry(
            enabled=self.enabled, paused=self.paused, arms=self.arm_telemetry()
        )

    # -- maintenance channel (phase-09b; 04-runtime §13.1 monitor path) -----------------
    def maintenance(
        self, arm_id: str, op: str, timeout_s: float = MAINTENANCE_TIMEOUT_S
    ) -> ArmMaintenanceResult:
        """Run one operator-triggered maintenance op on ``arm_id``'s READ-ONLY
        monitor (``path="monitor"``) and wait for its outcome (<= ``timeout_s``).

        ``clear_errors`` = ``clean_error`` + ``clean_warn`` (never
        ``motion_enable``); ``apply_backstops`` = ``backstops.apply_backstops``
        with the arm's ``ArmConfig`` mapped through the hardware package's
        ``XArmDriverConfig`` mapping. Both execute on the arm monitor's poll
        thread; this thread only waits. Refusals (HTTP 409 semantics,
        :class:`MaintenanceUnavailableError`): ``recover`` (needs a hardware
        session), monitor inert / not started / ``off`` / ``paused`` /
        ``connecting`` / ``error`` (the box is not connected), or an op already
        running on that arm. Unknown ``arm_id`` -> ``KeyError`` (404), unknown
        ``op`` -> ``ValueError`` (422 is FastAPI's job upstream). The result's
        ``before`` / ``after`` rows carry :func:`backstops_match` against the
        config; 200 whether or not ``ok``.
        """
        if op not in MAINTENANCE_OPS:
            raise ValueError(f"unknown maintenance op {op!r}; expected one of {MAINTENANCE_OPS}")
        arm = self._arm_cfgs.get(arm_id)
        if arm is None:
            raise KeyError(arm_id)
        if op == "recover":
            raise MaintenanceUnavailableError("no hardware session - use clear_errors")
        mon = self._monitors.get(arm_id)
        status, detail = self.status_of(arm_id)
        if mon is None or not self._started or status not in CONNECTED_STATUSES:
            why = f"monitor {status}" + (f": {detail}" if detail else "")
            raise MaintenanceUnavailableError(
                f"{op} needs the read-only monitor connected to {arm_id!r} ({why})"
            )
        lock = self._maintenance_locks[arm_id]
        if not lock.acquire(blocking=False):
            raise MaintenanceUnavailableError(f"a maintenance op is already running on {arm_id!r}")
        try:
            if self.maintenance_busy(arm_id):
                raise MaintenanceUnavailableError(
                    f"a maintenance op is already running on {arm_id!r}"
                )
            driver_cfg = self._driver_cfg(arm)
            outcome = mon.maintenance(op, driver_cfg, timeout_s=float(timeout_s))
        finally:
            lock.release()
        return self._result(arm, mon, op, outcome)

    def _driver_cfg(self, arm: ArmConfig) -> Any:
        try:
            factory = self._driver_cfg_factory or default_driver_cfg_factory()
        except Exception as e:  # noqa: BLE001 - [hardware] extra absent or broken
            raise MaintenanceUnavailableError(
                f"hardware package not importable ({type(e).__name__}: {e}); "
                "install the [hardware] extra"
            ) from e
        return factory(arm)

    def _result(
        self, arm: ArmConfig, mon: ArmMonitorLike, op: str, outcome: Any
    ) -> ArmMaintenanceResult:
        """Hardware ``MaintenanceOutcome`` -> core ``ArmMaintenanceResult`` (monitor path)."""
        status, detail = self.status_of(arm.id)

        def row(sample: Any) -> ArmMonitorTelemetry | None:
            if sample is None:
                return None
            age = max(0.0, self._clock() - float(getattr(sample, "t_mono", self._clock())))
            return sample_to_telemetry(
                arm.id,
                status,
                detail,
                age,
                sample,
                backstops_match=backstops_match(sample, arm),
                maintenance_busy=False,
            )

        return ArmMaintenanceResult(
            arm_id=arm.id,
            op=str(getattr(outcome, "op", op)),  # type: ignore[arg-type]
            path="monitor",
            ok=bool(outcome.ok),
            detail=str(getattr(outcome, "detail", "") or ""),
            sdk_codes={str(k): int(v) for k, v in dict(getattr(outcome, "sdk_codes", {})).items()},
            warnings=[str(w) for w in (getattr(outcome, "warnings", ()) or ())],
            before=row(getattr(outcome, "before", None)),
            after=row(getattr(outcome, "after", None)),
        )


__all__ = [
    "ArmMonitorLike",
    "CONNECTED_STATUSES",
    "DriverConfigFactory",
    "HardwareStateMonitor",
    "MAINTENANCE_OPS",
    "MAINTENANCE_TIMEOUT_S",
    "MonitorFactory",
    "TCP_COG_MATCH_MM",
    "TCP_LOAD_MATCH_KG",
    "backstops_match",
    "default_driver_cfg_factory",
    "default_monitor_factory",
    "sample_to_telemetry",
]
