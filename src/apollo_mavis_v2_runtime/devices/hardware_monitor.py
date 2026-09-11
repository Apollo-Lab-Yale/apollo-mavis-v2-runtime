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
``clear_errors`` / ``apply_backstops`` / ``home_rail`` to the arm's
``ArmStateMonitor``, which queues it for ITS poll thread (one ``XArmAPI`` is
never driven from two threads; this thread only waits), and returns the core
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

**``home_rail`` (phase-09c; 04-runtime §13.1) - THE ONE op that moves a
mechanical part.** The carriage drives to the track's zero end
(``set_linear_track_back_origin``), operator-triggered from the Hardware tab
and session-less only (the runtime routes it to 409 "end the session first"
while a hardware session exists). Before anything is written the runtime
sweeps a dedicated digital twin (``devices.rail_sweep.RailSweepChecker``) over
the FULL travel at the arm's current posture against the other arm's last
sample: ``dry_run`` returns that ``RailSweepVerdict`` alone (``ok`` iff clear,
zero writes); a blocked sweep returns ``ok=False`` + the verdict (zero writes);
a clear sweep hands ``expected_q = sample.q`` to the hardware monitor, whose
poll thread re-samples and refuses if the joints moved or an error is latched,
then homes, enables and sets the positioning speed and judges from the
registers only. The REST handler waits :data:`HOME_RAIL_TIMEOUT_S` (45 s) for
this op instead of 10 s; while it runs the arm's status reads ``stale`` and
``maintenance_busy`` is true, and ``POST /api/session`` is refused ("rail
homing in progress").

**Phase-09d.** The sweep + refusals are :meth:`HardwareStateMonitor.home_rail_preflight`
(zero writes) and the monitor op :meth:`HardwareStateMonitor.home_rail_execute`,
so ``devices/rail_homing.py`` can decide between them: a posture that is not
sweep-clear gets a planned pre-positioning motion run by a ``RailHomingJob``
(this arm's driver connected alone, the monitor paused meanwhile). The job
registry is attached as :attr:`HardwareStateMonitor.jobs` (``busy(arm_id)`` /
``progress(arm_id)``): ``maintenance_busy`` is true for the job's whole life and
``ArmMonitorTelemetry.maintenance`` carries its ``MaintenanceProgress``.

**``set_collision_sensitivity`` (2026-09-11, operator decision; 04-runtime §13.1).**
ONE write, ``set_collision_sensitivity(level)`` with ``level`` 1..3 (anything else is
``ValueError`` -> 422 here, before the arm monitor sees it), no motion; on THIS path
the arm monitor's poll thread writes, waits for the rich frame and judges by the
READ-BACK (the SDK returns the raw uxbus code: a 1 / 2 / 9 status echo while a fault
is latched is not a failure). The op is also allowed INSIDE a hardware session
(``SessionManager.session_set_collision_sensitivity`` -> the session driver's
monitor thread), unlike ``apply_backstops``. The controller default stays the
config value, re-applied at EVERY driver connect by ``apply_backstops``, so the
runtime remembers the operator's REQUESTED level per arm (:meth:`note_requested_sensitivity`,
set by both paths on success) until the next connect: :meth:`_apply_paused` (a
hardware session or a rail-homing job connects a driver) and a successful
``apply_backstops`` forget it. :func:`backstops_match` then judges the sensitivity
read-back against the requested level instead of the config, and while the monitor
is paused (no fresh sample) the arm's ``ArmMonitorTelemetry.collision_sensitivity``
publishes the requested level so the Cockpit's control shows what the operator
wrote — the UI never shows an optimistic value, the monitor's read-back replaces it
the moment the box is polled again.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from apollo_mavis_v2_core import ArmConfig, WorkcellConfig
from apollo_mavis_v2_core.protocol import (
    ArmMaintenanceResult,
    ArmMonitorTelemetry,
    HardwareMonitorTelemetry,
    MaintenanceProgress,
    RailSweepVerdict,
)

from ..config import HardwareMonitorConfig
from ..errors import MaintenanceUnavailableError
from .rail_sweep import RailSweepChecker, describe_verdict

logger = logging.getLogger(__name__)

CHECK_PERIOD_S = 0.5  # paused() is polled this often
STOP_JOIN_TIMEOUT_S = 5.0
# phase-09b: backstops_match tolerances (contract; the hardware monitor's own settle
# check uses the same 0.05 kg) and the REST wait budget for one maintenance op.
TCP_LOAD_MATCH_KG = 0.05
TCP_COG_MATCH_MM = 10.0
MAINTENANCE_TIMEOUT_S = 10.0
# phase-09c: home_rail = the SDK's 30 s homing wait + enable + speed + an after-sample;
# the caller (REST) waits this long (hardware HOME_RAIL_TIMEOUT_S, D3).
HOME_RAIL_TIMEOUT_S = 45.0
MONITOR_JOIN_TIMEOUT_S = 15.0  # hand-over: wait this long for a poll thread inside the SDK
MAINTENANCE_OPS: tuple[str, ...] = (
    "clear_errors",
    "apply_backstops",
    "recover",
    "home_rail",
    "set_collision_sensitivity",  # 2026-09-11: one write, level 1..3, both paths
)
COLLISION_SENSITIVITY_LEVELS: frozenset[int] = frozenset({1, 2, 3})
# the operator's admissible range (hardware ``backstops.COLLISION_SENSITIVITY_LEVELS``):
# 0 turns detection off, 4 / 5 false-trigger under payload
CONNECTED_STATUSES: frozenset[str] = frozenset({"running", "stale"})  # a maintenance op may run


class ArmMonitorLike(Protocol):
    """Surface of ``apollo_mavis_v2_hardware.ArmStateMonitor`` the runtime relies on
    (also what a test fake must provide)."""

    arm_id: str

    def start(self) -> None: ...

    def stop(self, timeout: float = 2.0) -> None: ...

    def disconnect(self, timeout: float = 2.0) -> bool | None: ...  # False = not yet released

    def join(self, timeout: float | None = None) -> bool: ...  # poll thread exited (phase-09c)

    def snapshot(self) -> Any: ...  # ArmMonitorSample | None

    def maintenance(
        self,
        op: str,
        driver_cfg: Any = None,
        timeout_s: float | None = 10.0,
        *,
        expected_q: Any = None,  # home_rail: the 7 joints the sweep assumed (phase-09c)
        q_tol_rad: float = 0.02,
        level: int | None = None,  # set_collision_sensitivity: the level 1..3 (2026-09-11)
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


class MaintenanceJobsLike(Protocol):
    """Registry of asynchronous maintenance jobs (phase-09d ``RailHomingService``)."""

    def busy(self, arm_id: str) -> bool: ...

    def progress(self, arm_id: str) -> MaintenanceProgress | None: ...


@dataclass
class HomeRailPreflight:
    """Everything :meth:`HardwareStateMonitor.home_rail_preflight` established with
    zero writes: the arm, its monitor + driver config, the sample the sweep used,
    every arm's sample, the twin verdict and the ``before`` telemetry row."""

    arm: ArmConfig
    monitor: ArmMonitorLike
    driver_cfg: Any
    sample: Any
    samples: dict[str, Any]
    verdict: RailSweepVerdict
    before: ArmMonitorTelemetry

    def result(self, *, dry_run: bool, detail: str | None = None) -> ArmMaintenanceResult:
        """The zero-write ``ArmMaintenanceResult`` (dry run or refused sweep)."""
        if detail is None:
            detail = describe_verdict(self.verdict, dry_run=dry_run)
        return ArmMaintenanceResult(
            arm_id=self.arm.id,
            op="home_rail",
            path="monitor",
            ok=bool(self.verdict.clear and dry_run),
            detail=detail,
            sdk_codes={},
            before=self.before,
            after=None,
            rail_sweep=self.verdict,
        )


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


def backstops_match(
    sample: Any, arm: ArmConfig, expected_sensitivity: int | None = None
) -> bool | None:
    """Controller read-back == ``ArmConfig`` backstops (contract tolerances):
    sensitivity equal, ``|d tcp_load| <= 0.05 kg`` and every centre-of-gravity
    component within 10 mm. ``None`` when the sample carries no read-back yet
    (older sample shape, or the monitor never read the rich report frame).
    ``expected_sensitivity`` (2026-09-11) replaces the config value as the
    sensitivity to expect: the level the operator wrote with
    ``set_collision_sensitivity`` (the runtime's per-arm requested level), so
    an obeyed override does not read as "differs from config"."""
    if sample is None:
        return None
    sens = getattr(sample, "collision_sensitivity", None)
    kg = getattr(sample, "tcp_load_kg", None)
    if sens is None or kg is None:
        return None
    expected = arm.collision_sensitivity if expected_sensitivity is None else expected_sensitivity
    if int(sens) != int(expected):
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
    maintenance: MaintenanceProgress | None = None,
) -> ArmMonitorTelemetry:
    """Core ``ArmMonitorTelemetry`` from a monitor's status + its last
    ``ArmMonitorSample`` (``None`` -> data fields at their defaults). The
    phase-09b read-backs are read duck-typed (absent on an older sample ->
    their defaults); ``backstops_match`` / ``maintenance_busy`` are the
    runtime's (:func:`backstops_match`, the monitor's busy flag);
    ``maintenance`` is the phase-09d job progress (``None`` = no job)."""
    if sample is None:
        return ArmMonitorTelemetry(
            arm_id=arm_id,
            status=status,
            detail=detail,
            age_s=age_s,
            maintenance_busy=maintenance_busy,
            maintenance=maintenance,
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
        maintenance=maintenance,
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
        rail_sweep: RailSweepChecker | None = None,
        rail_fallback_m: Mapping[str, float] | None = None,
        check_period_s: float = CHECK_PERIOD_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self.workcell = workcell
        self.paused_fn = paused or (lambda: False)
        self._factory = monitor_factory
        self._driver_cfg_factory = driver_cfg_factory  # None = the hardware package's mapping
        # phase-09c: the home_rail gate (None = no twin scene / sim extra -> home_rail 409)
        self.rail_sweep = rail_sweep
        self.rail_fallback_m: dict[str, float] = dict(rail_fallback_m or {})
        # phase-09d: asynchronous maintenance jobs (RailHomingService), attached by Runtime
        self.jobs: MaintenanceJobsLike | None = None
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
        # 2026-09-11: the collision sensitivity the operator WROTE per arm
        # (set_collision_sensitivity on either path), remembered until the next driver
        # connect re-applies the config value (_apply_paused(True)) or apply_backstops
        # rewrites it. Guarded by _lock; read by arm_telemetry() / the ws telemetry.
        self._requested_sensitivity: dict[str, int] = {}
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
        """Synchronous resume after a hand-over (reconnects every arm) - UNLESS
        the hand-over predicate still holds: a caller that never paused the
        monitor itself (a rail-homing job failing before its connect while a
        hardware ``create()`` owns the boxes) must not put a second SDK client
        on a box the session's drivers hold; the supervisor re-applies the
        predicate every round anyway, this just closes the window."""
        if self._started:
            self._apply_paused(self._safe_paused())

    def join(self, timeout_s: float = MONITOR_JOIN_TIMEOUT_S) -> list[str]:
        """After :meth:`pause`: wait up to ``timeout_s`` PER ARM for the poll
        threads to exit (the SDK client is released for certain only then; a
        ``disconnect()`` may return while the thread is still inside a blocking
        SDK call, up to a whole ``home_rail`` wait). Returns the arm ids whose
        thread is STILL alive - the hardware bring-up refuses on any."""
        alive: list[str] = []
        for arm_id, mon in self._monitors.items():
            join = getattr(mon, "join", None)
            if join is None:
                continue
            try:
                if not join(float(timeout_s)):
                    alive.append(arm_id)
            except Exception:  # noqa: BLE001 - treat a broken join as "still busy"
                logger.exception("arm monitor %r join failed", arm_id)
                alive.append(arm_id)
        return alive

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
                if flag and self._requested_sensitivity:
                    # A hand-over means a driver connects and apply_backstops rewrites
                    # the CONFIG sensitivity: the operator's override is gone with it.
                    logger.info(
                        "hardware monitor paused: forgetting the requested collision "
                        "sensitivity %s (the driver connect re-applies the config value)",
                        dict(self._requested_sensitivity),
                    )
                    self._requested_sensitivity.clear()
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
        """A maintenance op is queued / executing on this arm's monitor (phase-09b)
        or an asynchronous maintenance job owns the arm (phase-09d)."""
        if self.jobs is not None and self.jobs.busy(arm_id):
            return True
        mon = self._monitors.get(arm_id)
        return bool(getattr(mon, "maintenance_busy", False)) if mon is not None else False

    def job_progress(self, arm_id: str) -> MaintenanceProgress | None:
        """Live / lingering progress of the phase-09d job on this arm (``None`` = none)."""
        return self.jobs.progress(arm_id) if self.jobs is not None else None

    # -- the operator's collision-sensitivity override (2026-09-11) ----------------------
    def requested_sensitivity(self, arm_id: str) -> int | None:
        """The level the operator wrote with ``set_collision_sensitivity`` on this arm
        since the last driver connect (either path); ``None`` = the config value /
        unknown. Fills ``ArmTelemetry.collision_sensitivity`` in a hardware session."""
        with self._lock:
            return self._requested_sensitivity.get(arm_id)

    def note_requested_sensitivity(self, arm_id: str, level: int) -> None:
        """Record a SUCCESSFUL ``set_collision_sensitivity`` write (monitor or session
        path). Unknown arm -> ``KeyError``; a level outside 1..3 -> ``ValueError``."""
        if arm_id not in self._arm_cfgs:
            raise KeyError(arm_id)
        if (
            isinstance(level, bool)
            or int(level) != level
            or int(level) not in COLLISION_SENSITIVITY_LEVELS
        ):
            raise ValueError(f"collision sensitivity must be 1, 2 or 3 (got {level!r})")
        with self._lock:
            self._requested_sensitivity[arm_id] = int(level)

    def forget_requested_sensitivity(self, arm_id: str) -> None:
        """The config value is back on the controller (``apply_backstops`` ran)."""
        with self._lock:
            self._requested_sensitivity.pop(arm_id, None)

    def arm_telemetry(self) -> list[ArmMonitorTelemetry]:
        paused = self.paused
        rows: list[ArmMonitorTelemetry] = []
        for arm_id in self.arm_ids:
            mon = self._monitors.get(arm_id)
            status, detail = self.status_of(arm_id)
            sample = mon.snapshot() if mon is not None else None
            requested = self.requested_sensitivity(arm_id)
            expected = requested
            if requested is not None and paused:
                # In a hardware session the box is the driver's: no read-back reaches this
                # sample, so its sensitivity is the PRE-session value. The operator's write
                # (session path) is what the controller holds now: publish it as the row's
                # collision_sensitivity and judge the payload only (the sensitivity term
                # compares the stale read-back with itself).
                expected = getattr(sample, "collision_sensitivity", None)
            row = sample_to_telemetry(
                arm_id,
                status,
                detail,
                mon.age_s if mon is not None else None,
                sample,
                backstops_match=backstops_match(
                    sample, self._arm_cfgs[arm_id], expected_sensitivity=expected
                ),
                maintenance_busy=self.maintenance_busy(arm_id),
                maintenance=self.job_progress(arm_id),
            )
            if requested is not None and paused and row.collision_sensitivity != requested:
                row = row.model_copy(update={"collision_sensitivity": requested})
            rows.append(row)
        return rows

    def telemetry(self) -> HardwareMonitorTelemetry:
        """The ``arms`` part of ``telemetry.hardware_monitor`` (``overlays`` is
        filled by the caller from the ``TwinOverlayRenderer``)."""
        return HardwareMonitorTelemetry(
            enabled=self.enabled, paused=self.paused, arms=self.arm_telemetry()
        )

    # -- maintenance channel (phase-09b/09c; 04-runtime §13.1 monitor path) -------------
    def maintenance(
        self,
        arm_id: str,
        op: str,
        timeout_s: float = MAINTENANCE_TIMEOUT_S,
        *,
        dry_run: bool = False,
        collision_sensitivity: int | None = None,
    ) -> ArmMaintenanceResult:
        """Run one operator-triggered maintenance op on ``arm_id``'s READ-ONLY
        monitor (``path="monitor"``) and wait for its outcome (<= ``timeout_s``).

        ``clear_errors`` = ``clean_error`` + ``clean_warn`` (never
        ``motion_enable``); ``apply_backstops`` = ``backstops.apply_backstops``
        with the arm's ``ArmConfig`` mapped through the hardware package's
        ``XArmDriverConfig`` mapping. ``home_rail`` (phase-09c, THE ONE motion
        op) is gated first by the full-travel twin sweep (module docstring):
        ``dry_run`` -> the verdict alone (``ok`` iff clear, zero writes); a
        blocked sweep -> ``ok=False`` + verdict (zero writes); clear -> the
        hardware monitor homes with ``expected_q = sample.q`` (its poll thread
        re-samples and refuses if the arm moved) and the verdict rides
        ``rail_sweep``. All execute on the arm monitor's poll thread; this
        thread only waits. Refusals (HTTP 409 semantics,
        :class:`MaintenanceUnavailableError`): ``recover`` (needs a hardware
        session), monitor inert / not started / ``off`` / ``paused`` /
        ``connecting`` / ``error`` (the box is not connected), an op already
        running on that arm, or - ``home_rail`` - no twin sweep configured, no
        sample, no linear track, or a latched controller error (the box would
        refuse anyway). Unknown ``arm_id`` -> ``KeyError`` (404), unknown ``op``
        -> ``ValueError`` (422 is FastAPI's job upstream). The result's
        ``before`` / ``after`` rows carry :func:`backstops_match` against the
        config; 200 whether or not ``ok``.

        ``set_collision_sensitivity`` (2026-09-11): ``collision_sensitivity`` is
        the level to write, 1..3 only (else ``ValueError`` -> 422, nothing
        queued); the arm monitor writes it once, waits for the rich frame and
        is ``ok`` iff the read-back equals it (module docstring). On success
        the level is remembered as this arm's requested sensitivity
        (:meth:`note_requested_sensitivity`) and the result's
        ``collision_sensitivity`` carries the read-back; a successful
        ``apply_backstops`` forgets it (the config value is back).
        """
        if op not in MAINTENANCE_OPS:
            raise ValueError(f"unknown maintenance op {op!r}; expected one of {MAINTENANCE_OPS}")
        level: int | None = None
        if op == "set_collision_sensitivity":
            if (
                collision_sensitivity is None
                or isinstance(collision_sensitivity, bool)
                or int(collision_sensitivity) != collision_sensitivity
                or int(collision_sensitivity) not in COLLISION_SENSITIVITY_LEVELS
            ):
                raise ValueError(
                    "set_collision_sensitivity needs collision_sensitivity 1, 2 or 3 "
                    f"(got {collision_sensitivity!r}; 0 turns detection off, 4 / 5 "
                    "false-trigger under payload)"
                )
            level = int(collision_sensitivity)
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
        if op == "home_rail":
            pre = self.home_rail_preflight(arm_id)
            if dry_run or not pre.verdict.clear:
                return pre.result(dry_run=dry_run)
            return self.home_rail_execute(pre, float(timeout_s))
        lock = self._maintenance_locks[arm_id]
        if not lock.acquire(blocking=False):
            raise MaintenanceUnavailableError(f"a maintenance op is already running on {arm_id!r}")
        try:
            if self.maintenance_busy(arm_id):
                raise MaintenanceUnavailableError(
                    f"a maintenance op is already running on {arm_id!r}"
                )
            driver_cfg = self._driver_cfg(arm)
            if op == "set_collision_sensitivity":
                outcome = mon.maintenance(op, driver_cfg, timeout_s=float(timeout_s), level=level)
            else:
                outcome = mon.maintenance(op, driver_cfg, timeout_s=float(timeout_s))
        finally:
            lock.release()
        result = self._result(arm, mon, op, outcome, level=level)
        if result.ok:
            if op == "set_collision_sensitivity" and level is not None:
                self.note_requested_sensitivity(arm_id, level)
            elif op == "apply_backstops":
                self.forget_requested_sensitivity(arm_id)  # the config value is back
        return result

    def home_rail_preflight(self, arm_id: str) -> HomeRailPreflight:
        """``home_rail`` refusals + the twin sweep, ZERO writes (phase-09c/09d).

        Refused (409, :class:`MaintenanceUnavailableError`): the monitor not
        connected to the arm, an op already running on it, ANOTHER arm's homing
        in flight (its monitor publishes nothing during the SDK wait, so its
        sample still shows the pre-homing carriage while the real one travels -
        the sweep would be judged against a wrong posture and two carriages
        would move with unverified geometry), the target's own monitor ``stale``
        (the sweep posture must be the CURRENT one; the hardware monitor
        re-samples before the write as a second line of defence), no sample, no
        linear track, a latched controller error, or no twin. Another arm whose
        monitor is not ``running`` (stale / paused / error) is still posed from
        its last sample, with an ``assumptions`` entry saying so. The sweep
        verdict and the ``before`` row ride the returned :class:`HomeRailPreflight`;
        ``rail_sweep.pre_position`` is left ``None`` for the caller
        (``devices/rail_homing.py``) to fill.
        """
        from ..session.hardware import arm_label  # local: keeps devices free of session imports

        arm = self._arm_cfgs.get(arm_id)
        if arm is None:
            raise KeyError(arm_id)
        mon = self._monitors.get(arm_id)
        status, detail = self.status_of(arm_id)
        if mon is None or not self._started or status not in CONNECTED_STATUSES:
            why = f"monitor {status}" + (f": {detail}" if detail else "")
            raise MaintenanceUnavailableError(
                f"home_rail needs the read-only monitor connected to {arm_id!r} ({why})"
            )
        if self.maintenance_busy(arm_id):
            raise MaintenanceUnavailableError(f"a maintenance op is already running on {arm_id!r}")
        driver_cfg = self._driver_cfg(arm)
        sweep = self.rail_sweep
        if sweep is None:
            raise MaintenanceUnavailableError(
                "home_rail needs the digital twin to gate the sweep (no digital_twin_scene / "
                "[sim] extra)"
            )
        for other_id in self._arm_cfgs:
            if other_id != arm_id and self.maintenance_busy(other_id):
                raise MaintenanceUnavailableError(
                    f"home_rail refused: rail homing in progress on the {arm_label(other_id)} - "
                    "wait for it to finish"
                )
        status, detail = self.status_of(arm_id)
        if status == "stale":
            raise MaintenanceUnavailableError(
                f"home_rail refused: the read-only monitor sample of {arm_id!r} is stale"
                + (f" ({detail})" if detail else "")
                + " - the sweep needs the arm's CURRENT posture; retry when it reads running"
            )
        sample = mon.snapshot()
        if sample is None:
            raise MaintenanceUnavailableError(
                f"home_rail needs a monitor sample of {arm_id!r} (none yet)"
            )
        if not getattr(sample, "rail_present", False):
            raise MaintenanceUnavailableError(
                f"home_rail refused: no linear track detected on {arm_id!r}"
            )
        code = int(getattr(sample, "error_code", 0) or 0)
        if code:
            raise MaintenanceUnavailableError(
                f"home_rail refused: controller error {code} is latched on {arm_id!r} - "
                "clear errors first"
            )
        samples = self.snapshot()
        try:
            verdict = sweep.check(arm_id, samples, self.rail_fallback_m)
        except (KeyError, ValueError, ImportError) as e:
            raise MaintenanceUnavailableError(
                f"home_rail refused: rail sweep unavailable for {arm_id!r} ({e})"
            ) from e
        for other_id in self._arm_cfgs:  # a sampled other arm whose monitor is not live
            if other_id == arm_id or other_id not in samples:
                continue
            o_status, _ = self.status_of(other_id)
            if o_status != "running":
                verdict.assumptions.append(
                    f"{other_id}: monitor {o_status} - posed from its last sample, which may "
                    "not be its current posture"
                )
        before = sample_to_telemetry(
            arm_id,
            status,
            detail,
            mon.age_s,
            sample,
            backstops_match=backstops_match(sample, arm),
            maintenance_busy=False,
        )
        return HomeRailPreflight(arm, mon, driver_cfg, sample, dict(samples), verdict, before)

    def home_rail_execute(self, pre: HomeRailPreflight, timeout_s: float) -> ArmMaintenanceResult:
        """The 09c monitor-path homing after a CLEAR preflight: joints untouched,
        the arm monitor's poll thread re-samples (``expected_q`` = the sweep's
        posture) and homes; judged from the registers. One op per arm at a time
        (per-arm lock + ``maintenance_busy``)."""
        arm, mon, verdict = pre.arm, pre.monitor, pre.verdict
        if not verdict.clear:
            raise MaintenanceUnavailableError(
                f"home_rail on {arm.id!r}: the sweep is not clear - nothing written"
            )
        lock = self._maintenance_locks[arm.id]
        if not lock.acquire(blocking=False):
            raise MaintenanceUnavailableError(f"a maintenance op is already running on {arm.id!r}")
        try:
            if self.maintenance_busy(arm.id):
                raise MaintenanceUnavailableError(
                    f"a maintenance op is already running on {arm.id!r}"
                )
            logger.warning(
                "home_rail on %s: sweep clear (min clearance %s m) - HOMING the linear track "
                "(the carriage drives to the zero end)",
                arm.id,
                f"{verdict.min_clearance_m:.3f}" if verdict.min_clearance_m is not None else "?",
            )
            outcome = mon.maintenance(
                "home_rail",
                pre.driver_cfg,
                timeout_s=float(timeout_s),
                expected_q=tuple(float(v) for v in verdict.q_checked),
            )
        finally:
            lock.release()
        return self._result(arm, mon, "home_rail", outcome, rail_sweep=verdict)

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
        self,
        arm: ArmConfig,
        mon: ArmMonitorLike,
        op: str,
        outcome: Any,
        rail_sweep: Any = None,
        level: int | None = None,
    ) -> ArmMaintenanceResult:
        """Hardware ``MaintenanceOutcome`` -> core ``ArmMaintenanceResult`` (monitor path).

        ``level`` (``set_collision_sensitivity`` only) fills the result's
        ``collision_sensitivity`` with the after-sample's READ-BACK (equal to
        ``level`` when ``ok``), or with ``level`` itself when the op succeeded
        without an after-sample; ``None`` for every other op and a refusal."""
        status, detail = self.status_of(arm.id)
        # backstops_match on the rows: before = against the level in force so far (the
        # operator's earlier override, else the config); after = against the level this
        # op leaves on the box (the written level; the config again after apply_backstops).
        requested = self.requested_sensitivity(arm.id)
        if op == "set_collision_sensitivity":
            after_expected = level
        elif op == "apply_backstops":
            after_expected = None
        else:
            after_expected = requested

        def row(sample: Any, expected: int | None) -> ArmMonitorTelemetry | None:
            if sample is None:
                return None
            age = max(0.0, self._clock() - float(getattr(sample, "t_mono", self._clock())))
            return sample_to_telemetry(
                arm.id,
                status,
                detail,
                age,
                sample,
                backstops_match=backstops_match(sample, arm, expected_sensitivity=expected),
                maintenance_busy=False,
            )

        after_sample = getattr(outcome, "after", None)
        written: int | None = None
        if op == "set_collision_sensitivity" and level is not None:
            readback = getattr(after_sample, "collision_sensitivity", None)
            if readback is not None:
                written = int(readback)
            elif bool(outcome.ok):
                written = level
        return ArmMaintenanceResult(
            arm_id=arm.id,
            op=str(getattr(outcome, "op", op)),  # type: ignore[arg-type]
            path="monitor",
            ok=bool(outcome.ok),
            detail=str(getattr(outcome, "detail", "") or ""),
            sdk_codes={str(k): int(v) for k, v in dict(getattr(outcome, "sdk_codes", {})).items()},
            warnings=[str(w) for w in (getattr(outcome, "warnings", ()) or ())],
            before=row(getattr(outcome, "before", None), requested),
            after=row(after_sample, after_expected),
            rail_sweep=rail_sweep,
            collision_sensitivity=written,
        )


__all__ = [
    "ArmMonitorLike",
    "COLLISION_SENSITIVITY_LEVELS",
    "CONNECTED_STATUSES",
    "DriverConfigFactory",
    "HOME_RAIL_TIMEOUT_S",
    "HardwareStateMonitor",
    "HomeRailPreflight",
    "MaintenanceJobsLike",
    "MAINTENANCE_OPS",
    "MAINTENANCE_TIMEOUT_S",
    "MONITOR_JOIN_TIMEOUT_S",
    "MonitorFactory",
    "TCP_COG_MATCH_MM",
    "TCP_LOAD_MATCH_KG",
    "backstops_match",
    "default_driver_cfg_factory",
    "default_monitor_factory",
    "sample_to_telemetry",
]
