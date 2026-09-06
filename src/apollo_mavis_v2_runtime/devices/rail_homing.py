"""Rail homing with a planned pre-positioning motion (phase-09d; 04-runtime §5 /
§13.1 / §15, 11-safety §4).

``home_rail`` (phase-09c) homes a linear track only when the arm's CURRENT
posture is sweep-clear over the whole travel. Phase-09d turns the blocked case
into a planned maintenance motion instead of a flat refusal. Three pieces:

* :class:`PrePositionPlanner` - given the sweep verdict, tries the candidate
  postures in order (the scene keyframe's 7 joints for the arm - the folded
  factory zero - then the ``<arm>_home`` keyframe): each must be sweep-clear
  itself, reachable by the sweep twin's RRT-Connect from the current posture
  with the rail slot locked at ``rail_fallback_m`` (``RailSweepChecker.plan_path``),
  AND the path must pass the position-agnostic check (``check_path``: every
  densified configuration x all 131 rail positions under the start-posture
  hysteresis). The first success is the :class:`PrePositionPlan`
  (``needed=True, clear=True``); none -> ``clear=False`` with an operator
  suggestion. A clear sweep is ``needed=False`` (the 09c path). ``duration_s``
  is estimated at the hardware executor caps (``session.hardware.ExecutorCaps``
  from the hardware package's ``ServoLimits`` defaults x 0.1 - the workcell's
  driver config forwards no speed field, so those ARE the driver's caps), i.e.
  the per-joint velocity AND the lever-weighted Cartesian step of the servo
  stream, not the host's bare jog slew.
* :class:`RailHomingService` - the REST-facing decision
  (:meth:`RailHomingService.request`): ``dry_run`` -> the verdict incl.
  ``pre_position`` (zero writes); ``needed == False`` -> the synchronous 09c
  monitor-path homing (200, ``status: done``); ``needed and clear`` -> a
  :class:`RailHomingJob` is started (202, ``status: accepted`` + ``job_id``);
  ``needed and not clear`` -> ``status: refused`` (``ok`` False). It is the
  monitor's ``jobs`` registry (``busy`` / ``progress``: ``maintenance_busy`` is
  true for the job's whole life, ``ArmMonitorTelemetry.maintenance`` carries the
  :class:`MaintenanceProgress`) and keeps the last result per arm for
  ``GET /api/hardware/arms/{arm_id}/maintenance/last`` (the 202's ``accepted``
  result while the job runs, its final result afterwards). Mutual exclusion is
  ATOMIC: ``request`` reserves the arm under the service lock before the
  seconds of preflight + planning (``active_arm`` reports the reservation, so a
  second request on either arm, any other op and ``POST /api/session`` - the
  manager's ``maintenance_guard`` - are refused meanwhile) and hands the
  reservation to the job it registers; every other exit releases it. A dry run
  caches its plan per arm; the confirm re-validates THAT plan (``check_path``
  against the fresh samples) and executes it unchanged - a fresh plan only when
  the arm moved, and a fresh plan that differs from the one the operator saw is
  refused ("plan changed - re-confirm").
* :class:`RailHomingJob` - one arm, one thread, the six phases of the contract:
  ``sweeping`` -> ``planning`` (both settled by the request; recorded) ->
  ``connecting`` (under the session manager's lock: monitor paused + joined,
  ONLY this arm's driver connected with ``rail_homing: allow_unhomed`` at speed
  scale 0.1, the other arm frozen in the gate twin at its last sample - 09c D1
  -, the gate unconditional, the ``RailHoldWorkcell`` adapter feeding the twin
  ``rail_fallback_m`` for the unknown carriage and pinning every rail command;
  the measured posture must still match the sweep's within 0.02 rad) ->
  ``positioning`` (``ControlLoop._op_execute_plan`` on a PRIVATE bus, no
  teleop source, wait for the executor; a driver fault aborts) -> ``homing``
  (the loop is stopped first - the driver's servo stream holds the joints and
  no stale 8-dof hold can move the freshly homed carriage - then
  ``driver.home_rail()`` on this thread) -> ``verifying`` (registers homed +
  enabled + no error, ``rail_position_known``, phase ``READY``, raw ``rail_pos_m``
  0.0) -> teardown (D6: mode 0, state 4, brakes -> the arm HOLDS the folded
  posture; no automatic return) -> monitor resumed -> a fresh monitor sample
  awaited -> ``done``. Any failure -> ``failed`` with the same teardown and
  the monitor resumed ONLY when this job paused it (a job that never got to
  connect leaves a session's hand-over alone). ``cancel()`` (``Runtime.stop``)
  aborts before the connect / between executor ticks / before the homing - the
  teardown stops the loop and the drivers, so the arm holds where it is on the
  validated path - and never cuts an SDK homing wait. The position-agnostic
  check is the ONLY safety basis of the motion: the gate twin still guesses the
  carriage during it.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
from apollo_mavis_v2_core import Command
from apollo_mavis_v2_core.protocol import (
    ArmMaintenanceResult,
    MaintenanceProgress,
    PrePositionPlan,
    RailSweepVerdict,
)

from ..bus import RuntimeBus
from ..errors import MaintenanceUnavailableError
from .hardware_monitor import (
    HOME_RAIL_TIMEOUT_S,
    HardwareStateMonitor,
    HomeRailPreflight,
    backstops_match,
    sample_to_telemetry,
)
from .rail_sweep import PLAN_TIMEOUT_S, PathVerdict, RailSweepChecker, describe_verdict

if TYPE_CHECKING:
    from ..config import RuntimeConfig
    from ..session.manager import SessionManager

logger = logging.getLogger(__name__)

JOB_SPEED_SCALE = 0.1  # the maintenance motion always runs at 10 % (contract §3 step 3)
PHASE_ORDER: tuple[str, ...] = (
    "queued",
    "sweeping",
    "planning",
    "connecting",
    "positioning",
    "homing",
    "verifying",
    "done",
)
PHASE_PROGRESS: dict[str, float] = {
    "queued": 0.0,
    "sweeping": 0.05,
    "planning": 0.1,
    "connecting": 0.2,
    "positioning": 0.4,  # + 0.35 * waypoint fraction while executing
    "homing": 0.8,
    "verifying": 0.95,
    "done": 1.0,
    "failed": 1.0,
}
PROGRESS_LINGER_S = 60.0  # a terminal phase stays on telemetry this long
POSITIONING_TIMEOUT_MIN_S = 30.0
POSITIONING_TIMEOUT_FACTOR = 3.0  # x the estimated duration
POSTURE_REACHED_TOL_RAD = 0.05  # measured joints vs the planned target after the motion
POSTURE_SETTLE_S = 3.0
SAMPLE_WAIT_S = 5.0  # a fresh monitor sample after the resume
COMMAND_ACK_S = 5.0
# Runtime.stop: a cancelled job ends within its teardown, a job inside the SDK homing
# wait (<= 30 s) + verify + teardown + resume + sample within this bound
STOP_JOIN_TIMEOUT_S = HOME_RAIL_TIMEOUT_S + 20.0
REFUSAL_HINT = (
    "fold the arm toward the factory zero posture in xArm Studio (joints 2-7 near 0) and retry"
)


class JobFailed(Exception):
    """A phase of the job failed (operator-facing reason in ``str``)."""

    def __init__(self, detail: str, sdk_codes: dict[str, int] | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.sdk_codes = dict(sdk_codes or {})


def _posed_sample(sample: Any, q7: list[float]) -> Any:
    """``sample`` with its joints replaced (the candidate posture's own sweep)."""
    fields = (
        "arm_id",
        "seq",
        "t_mono",
        "rail_present",
        "rail_homed",
        "rail_enabled",
        "rail_pos_m",
        "rail_raw_mm",
        "error_code",
        "warn_code",
    )
    return SimpleNamespace(
        **{name: getattr(sample, name, None) for name in fields}, q=tuple(float(v) for v in q7)
    )


class PrePositionPlanner:
    """Sweep verdict -> :class:`PrePositionPlan` (+ the 8-dof waypoints); module docstring."""

    def __init__(
        self,
        checker: RailSweepChecker,
        rail_fallback_m: dict[str, float],
        *,
        slew_rad_per_tick: float,
        rate_hz: float,
        plan_timeout_s: float = PLAN_TIMEOUT_S,
        cart_step_m: float | None = None,
        lever_arm_m: tuple[float, ...] | None = None,
    ) -> None:
        self.checker = checker
        self.rail_fallback_m = dict(rail_fallback_m)
        # the executor caps at 10 %: per-joint slew (the host jog slew bounded by the
        # driver's max_joint_vel) and the lever-weighted Cartesian step per tick
        self.slew_rad_per_tick = float(slew_rad_per_tick)
        self.cart_step_m = None if cart_step_m is None else float(cart_step_m)
        self.lever_arm_m = None if lever_arm_m is None else tuple(map(float, lever_arm_m))
        self.rate_hz = float(rate_hz)
        self.plan_timeout_s = float(plan_timeout_s)
        self.evaluations = 0  # tests / diagnostics
        self.revalidations = 0

    def duration_s(self, waypoints: list[list[float]]) -> float:
        """Executor-time estimate of ``waypoints`` at the caps (``PrePositionPlan.duration_s``)."""
        from ..session.hardware import plan_duration_s

        return plan_duration_s(
            waypoints,
            self.slew_rad_per_tick,
            self.rate_hz,
            cart_step_m=self.cart_step_m,
            lever_arm_m=self.lever_arm_m,
        )

    def revalidate(
        self, arm_id: str, waypoints: list[list[float]], samples: dict[str, Any]
    ) -> PathVerdict:
        """The position-agnostic check of an EXISTING plan against fresh samples
        (the confirm re-validates the dry run's plan instead of re-planning)."""
        self.revalidations += 1
        return self.checker.check_path(arm_id, waypoints, samples, self.rail_fallback_m)

    def evaluate(
        self, arm_id: str, verdict: RailSweepVerdict, samples: dict[str, Any]
    ) -> tuple[PrePositionPlan, list[list[float]] | None]:
        """``(plan, waypoints | None)``; ``waypoints`` only when ``needed and clear``."""
        self.evaluations += 1
        if verdict.clear:
            return (
                PrePositionPlan(
                    needed=False,
                    source="current",
                    target_q=[float(v) for v in verdict.q_checked],
                    clear=True,
                    detail="current posture is sweep-clear - the rail homes with the joints "
                    "untouched",
                ),
                None,
            )
        q_now = [float(v) for v in verdict.q_checked]
        tried: list[str] = []
        last_path: PathVerdict | None = None
        for source, cand in self.checker.candidate_postures(arm_id):
            if np.max(np.abs(np.asarray(cand) - np.asarray(q_now))) < 1e-6:
                tried.append(f"{source}: is the current posture")
                continue
            cand_samples = dict(samples)
            cand_samples[arm_id] = _posed_sample(samples[arm_id], cand)
            posture = self.checker.check(arm_id, cand_samples, self.rail_fallback_m)
            if not posture.clear:
                pair = " / ".join(posture.first_blocked_pair) or "?"
                blocked_m = posture.first_blocked_m
                at = f"{blocked_m:.3f}" if blocked_m is not None else "?"
                tried.append(f"{source}: posture not sweep-clear (blocked at {at} m, {pair})")
                continue
            plan = self.checker.plan_path(
                arm_id, q_now, cand, samples, self.rail_fallback_m, timeout_s=self.plan_timeout_s
            )
            if not plan.ok or arm_id not in plan.waypoints:
                pair = f" ({' / '.join(plan.failing_pair)})" if plan.failing_pair else ""
                tried.append(f"{source}: no plan ({plan.failure}{pair})")
                continue
            wps = plan.waypoints[arm_id]
            path = self.checker.check_path(arm_id, wps, samples, self.rail_fallback_m)
            last_path = path
            if not path.clear:
                tried.append(f"{source}: {path.detail}")
                continue
            duration = self.duration_s(wps)
            return (
                PrePositionPlan(
                    needed=True,
                    source=source,  # type: ignore[arg-type]
                    target_q=[float(v) for v in cand],
                    waypoints=len(wps),
                    duration_s=round(duration, 1),
                    checked_rail_positions=path.checked_rail_positions,
                    clear=True,
                    detail=(
                        f"the arm first moves along a planned path ({len(wps)} waypoints, "
                        f"~{duration:.0f} s at 10 %) to the {source} posture, which clears the "
                        f"whole rail travel; {path.detail}"
                    ),
                ),
                wps,
            )
        return (
            PrePositionPlan(
                needed=True,
                source="current",
                target_q=[],
                waypoints=0,
                duration_s=0.0,
                checked_rail_positions=(
                    last_path.checked_rail_positions if last_path is not None else 0
                ),
                clear=False,
                detail="no rail-safe pre-positioning path: "
                + "; ".join(tried or ["no candidate posture"])
                + f" - {REFUSAL_HINT}",
            ),
            None,
        )


@dataclass(frozen=True)
class _CachedPlan:
    """A dry run's plan (what the operator confirms) with its verdict."""

    plan: PrePositionPlan
    waypoints: list[list[float]]
    verdict: RailSweepVerdict


@dataclass
class _JobState:
    progress: MaintenanceProgress
    finished_at: float | None = None
    result: ArmMaintenanceResult | None = None
    owns_boxes: bool = False  # the job's driver holds a control box (monitor paused)
    rows: list = field(default_factory=list)


class RailHomingJob(threading.Thread):
    """One arm's planned pre-positioning + homing (module docstring)."""

    def __init__(
        self,
        service: RailHomingService,
        pre: HomeRailPreflight,
        plan: PrePositionPlan,
        waypoints: list[list[float]],
        job_id: str | None = None,
    ) -> None:
        self.service = service
        self.pre = pre
        self.arm_id = pre.arm.id
        self.plan = plan
        self.waypoints = [list(map(float, w)) for w in waypoints]
        self.job_id = job_id or uuid.uuid4().hex
        self.state = _JobState(
            MaintenanceProgress(
                op="home_rail",
                job_id=self.job_id,
                phase="queued",
                detail="accepted - the job thread is starting",
                progress=0.0,
                started_at=time.time(),
            )
        )
        self._lock = threading.Lock()
        self._paused_monitor = False  # THIS job paused the monitor (resume only then)
        self._cancel = threading.Event()  # Runtime.stop: abort at the next safe point
        self.phases_seen: list[str] = ["queued"]  # tests / diagnostics
        self.rig: Any = None  # the HardwareRig once connected (kept after teardown: tests)
        super().__init__(name=f"rail-homing-{self.arm_id}", daemon=True)

    def cancel(self) -> None:
        """Abort at the next safe point (before the connect, between executor
        ticks, before the homing); an SDK homing wait in flight is never cut."""
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def _check_cancel(self, where: str) -> None:
        if self._cancel.is_set():
            raise JobFailed(f"cancelled {where} (runtime stopping) - the arm holds where it is")

    # -- progress ---------------------------------------------------------------------------
    @property
    def progress(self) -> MaintenanceProgress:
        with self._lock:
            return self.state.progress

    @property
    def terminal(self) -> bool:
        return self.progress.phase in ("done", "failed")

    @property
    def owns_boxes(self) -> bool:
        with self._lock:  # one acquisition: ``terminal`` would re-take the (plain) lock
            phase = self.state.progress.phase
            return self.state.owns_boxes and phase not in ("done", "failed")

    def _set(self, phase: str, detail: str, progress: float | None = None) -> None:
        with self._lock:
            self.state.progress = self.state.progress.model_copy(
                update={
                    "phase": phase,
                    "detail": detail,
                    "progress": PHASE_PROGRESS[phase] if progress is None else float(progress),
                }
            )
            if phase != self.phases_seen[-1]:
                self.phases_seen.append(phase)
        logger.info("rail homing job %s (%s): %s - %s", self.job_id, self.arm_id, phase, detail)

    # -- the thread -------------------------------------------------------------------------
    def run(self) -> None:
        outcome_codes: dict[str, int] = {}
        try:
            v = self.pre.verdict
            at = f"{v.first_blocked_m:.3f} m" if v.first_blocked_m is not None else "?"
            pair = " / ".join(v.first_blocked_pair) or "?"
            self._set(
                "sweeping", f"current posture blocked at rail {at} ({pair}) - pre-positioning"
            )
            self._set(
                "planning",
                f"{self.plan.waypoints} waypoints to the {self.plan.source} posture "
                f"(~{self.plan.duration_s:.0f} s at 10 %), clear for every rail position",
            )
            self._set(
                "connecting",
                "pausing the read-only monitor, connecting this arm alone (rail unhomed, "
                "speed 10 %)",
            )
            self._check_cancel("before connecting")
            rig = self._connect()
            self.rig = rig
            try:
                self._verify_start_posture(rig)
                rig.loop.active_arm = None  # no teleop source: the loop only holds / executes
                rig.loop.start()
                self._install_overlay(rig)
                self._set("positioning", f"executing {len(self.waypoints)} waypoints at 10 %")
                self._execute_plan(rig)
                self._check_cancel("before homing")
                self._set(
                    "homing",
                    "carriage driving to the homing end (joints held by the servo stream)",
                )
                # the loop's senders stop first: the driver's own streamer keeps the hold
                # posture, and no 8-dof command can move the carriage once it is homed
                rig.loop.stop()
                driver = rig.inner.arms[self.arm_id]
                outcome = driver.home_rail()
                outcome_codes = dict(getattr(outcome, "sdk_codes", None) or {})
                if not getattr(outcome, "ok", False):
                    raise JobFailed(
                        f"rail homing failed: {getattr(outcome, 'detail', '')}", outcome_codes
                    )
                self._set("verifying", "registers read back: homed, enabled, no error")
                self._verify(driver, outcome)
            finally:
                self._teardown(rig)
            after = self._resume_and_sample()
            detail = (
                f"rail homed after a planned pre-positioning motion ({self.plan.waypoints} "
                f"waypoints to the {self.plan.source} posture): {getattr(outcome, 'detail', '')}"
                " - the arm holds that posture (brakes engaged)"
            )
            self._finish(True, detail, outcome_codes, after)
        except JobFailed as e:
            self._fail(str(e.detail), e.sdk_codes or outcome_codes)
        except BaseException as e:  # noqa: BLE001 - the job must always end in a result
            logger.exception("rail homing job %s (%s) crashed", self.job_id, self.arm_id)
            self._fail(f"{type(e).__name__}: {e}", outcome_codes)

    # -- phases -----------------------------------------------------------------------------
    def _connect(self):
        svc = self.service
        manager = svc.manager
        from ..session.manager import _BringupProgress  # local: avoid an import cycle

        wc = svc.cfg.workcell_config("hardware")
        if wc is None or not wc.digital_twin_scene:
            raise JobFailed("no hardware workcell / digital twin scene configured")
        with manager._lock:  # exclusive with create() / teardown()
            if manager.session_active:
                raise JobFailed(
                    "a session is active or starting - the maintenance motion cannot "
                    "connect; end the session first"
                )
            other = svc.active_arm
            if svc.owns_boxes or other not in (None, self.arm_id):
                raise JobFailed(  # belt and braces: request() reserves the cell atomically
                    f"another rail homing job holds the cell ({other or '?'}) - cannot connect"
                )
            with self._lock:
                self.state.owns_boxes = True  # hand-over predicate true before the pause
                self._paused_monitor = True  # connect_hardware_rig pauses + joins the monitor
            try:
                return manager.connect_hardware_rig(
                    arms=[self.arm_id],
                    wc=wc,
                    twin_scene=wc.digital_twin_scene,
                    samples=self.pre.samples,
                    speed_scale=JOB_SPEED_SCALE,
                    progress=_BringupProgress(self.job_id, None),
                    bus=RuntimeBus(),
                    tracker=False,
                    allow_unhomed_rail=True,
                    rail_hold=True,
                )
            except Exception as e:
                raise JobFailed(f"connect failed: {e}") from e

    def _verify_start_posture(self, rig) -> None:
        from ..session.hardware import START_POSTURE_TOL_RAD

        st = rig.workcell.states()[self.arm_id]
        q_meas = np.asarray(st.q, dtype=np.float64)[:7]
        q_plan = np.asarray(self.pre.verdict.q_checked, dtype=np.float64)
        dev = np.abs(q_meas - q_plan)
        worst = int(np.argmax(dev))
        if dev[worst] > START_POSTURE_TOL_RAD:
            raise JobFailed(
                f"the arm moved since the sweep (joint {worst + 1} differs by "
                f"{dev[worst]:.3f} rad, tolerance {START_POSTURE_TOL_RAD:g} rad) - re-run Home rail"
            )
        if int(st.error_code):
            raise JobFailed(f"controller error {st.error_code} is latched - clear errors first")
        driver = rig.inner.arms[self.arm_id]
        known = getattr(driver, "rail_position_known", None)
        if known:
            logger.info(
                "rail homing job %s: %s reports its track already homed - re-homing anyway",
                self.job_id,
                self.arm_id,
            )

    def _install_overlay(self, rig) -> None:
        overlay = self.service.twin_overlay
        if overlay is None:
            return
        from ..session.hardware import SessionStateProvider

        try:
            overlay.set_state_provider(
                SessionStateProvider(
                    rig.inner,
                    [self.arm_id],
                    {a: self.pre.samples[a] for a in rig.frozen if a in self.pre.samples},
                    rig.loop.gripper_arms,
                )
            )
        except Exception:  # noqa: BLE001 - the overlay must never fail the job
            logger.exception("rail homing job: overlay provider failed")

    def _execute_plan(self, rig) -> None:
        loop = rig.loop
        future = loop.bus.commands.submit(
            Command(
                op="execute_plan",
                args={"waypoints": {self.arm_id: self.waypoints}, "gripper": {}},
                source="internal",
            )
        )
        try:
            ack = future.result(timeout=COMMAND_ACK_S)
        except Exception as e:  # noqa: BLE001 - loop not ticking
            raise JobFailed(f"the control loop did not accept the plan: {e}") from e
        if not ack.ok:
            raise JobFailed(f"the control loop refused the plan: {ack.detail}")
        total = max(1, len(self.waypoints))
        # the executor runs at the rig's caps (the connected driver's servo limits); the
        # request-time estimate used the hardware defaults - take the slower of the two
        duration = max(float(self.plan.duration_s), self._rig_duration_s(rig))
        timeout = max(POSITIONING_TIMEOUT_MIN_S, POSITIONING_TIMEOUT_FACTOR * duration)
        deadline = time.monotonic() + timeout
        clock = self.service.clock
        while True:
            now = time.monotonic()
            if self._cancel.is_set():
                # the teardown that follows stops the loop (no further executor tick) and
                # the drivers: the arm holds at its last command, ON the validated segment
                raise JobFailed(
                    "cancelled during the pre-positioning motion (runtime stopping) - the arm "
                    "holds where it stopped on the validated path"
                )
            if loop.faulted_arms or loop.fault_state is not None:
                faults = loop._arm_fault_details(clock())
                raise JobFailed(
                    "driver fault during the pre-positioning motion: "
                    f"{faults.get(self.arm_id, 'arm stopped')}"
                )
            if not loop.plans.active(self.arm_id):
                state = loop._plan_state.get(self.arm_id)
                if state in ("failed", "cancelled"):
                    raise JobFailed(f"the pre-positioning plan was {state}")
                break
            index = loop.plans._index.get(self.arm_id, 0)
            self._set(
                "positioning",
                f"waypoint {min(index + 1, total)} of {total}",
                PHASE_PROGRESS["positioning"] + 0.35 * min(1.0, index / total),
            )
            if now > deadline:
                report = loop.supervisor.merged_report()
                gate = (
                    f"gate {report.severity} {[' / '.join(p) for p in report.pairs]}"
                    if report.blocked
                    else f"gate {report.severity}"
                )
                q_last = loop._last_cmd.get(self.arm_id)
                held = (
                    f"commanded {np.round(np.asarray(q_last)[:7], 3).tolist()}"
                    if q_last is not None
                    else ""
                )
                raise JobFailed(
                    f"the pre-positioning motion did not finish within {timeout:.0f} s "
                    f"(waypoint {min(index + 1, total)} of {total}; {gate}; {held})"
                )
            time.sleep(0.05)
        # the measured joints must have reached the planned (sweep-clear) posture
        target = np.asarray(self.plan.target_q, dtype=np.float64)
        settle = time.monotonic() + POSTURE_SETTLE_S
        while True:
            q_meas = np.asarray(rig.workcell.states()[self.arm_id].q, dtype=np.float64)[:7]
            dev = np.abs(q_meas - target)
            if float(np.max(dev)) <= POSTURE_REACHED_TOL_RAD:
                return
            if time.monotonic() > settle:
                worst = int(np.argmax(dev))
                raise JobFailed(
                    f"the arm did not reach the planned posture (joint {worst + 1} off by "
                    f"{dev[worst]:.3f} rad) - not homing"
                )
            time.sleep(0.05)

    def _rig_duration_s(self, rig) -> float:
        """The plan's executor time at the connected rig's caps (0.0 when unknown)."""
        from ..session.hardware import plan_duration_s

        jog = rig.loop.cfg.jog
        return plan_duration_s(
            self.waypoints,
            float(jog.slew_rad_per_tick),
            rig.loop.cfg.rate_hz,
            cart_step_m=jog.plan_cart_step_m,
            lever_arm_m=jog.plan_lever_arm_m,
        )

    def _verify(self, driver, outcome) -> None:
        problems: list[str] = []
        if int(getattr(outcome, "on_zero", 0) or 0) != 1:
            problems.append("on_zero != 1")
        if int(getattr(outcome, "is_enabled", 0) or 0) != 1:
            problems.append("track not enabled")
        if int(getattr(outcome, "error", 0) or 0) != 0:
            problems.append(f"linear track error {getattr(outcome, 'error', '?')}")
        if not getattr(driver, "rail_position_known", False):
            problems.append("driver still reports the rail position unknown")
        if str(getattr(driver, "rail_phase", "")) != "READY":
            problems.append(f"rail phase {getattr(driver, 'rail_phase', '?')} (expected READY)")
        try:
            pos = driver.get_state().rail_pos_m
        except Exception as e:  # noqa: BLE001
            problems.append(f"get_state failed: {e}")
            pos = None
        if pos is None or abs(float(pos)) > 1e-3:
            problems.append(f"raw rail position {pos} (expected 0.000 m)")
        if problems:
            raise JobFailed(
                "rail homing verification failed: " + ", ".join(problems),
                dict(getattr(outcome, "sdk_codes", None) or {}),
            )

    def _teardown(self, rig) -> None:
        """D6 hand-back: loop -> drivers (mode 0, state 4, brakes) - the posture is HELD."""
        try:
            self.service.manager.stop_rig(rig.loop, rig.workcell)
        finally:
            overlay = self.service.twin_overlay
            if overlay is not None:
                try:
                    overlay.set_state_provider(None)
                except Exception:  # noqa: BLE001
                    logger.exception("rail homing job: overlay provider reset failed")
            with self._lock:
                self.state.owns_boxes = False

    def _resume_monitor(self, why: str) -> None:
        """Resume the read-only monitor iff THIS job paused it (``_connect``); a job
        that failed before its connect must not touch a session's hand-over."""
        with self._lock:
            paused = self._paused_monitor
            self._paused_monitor = False
        if not paused:
            return
        try:
            self.service.monitor.resume()
        except Exception:  # noqa: BLE001
            logger.exception("rail homing job: monitor resume %s failed", why)

    def _resume_and_sample(self):
        """Monitor back (hand-over predicate is already false), then a fresh sample."""
        monitor = self.service.monitor
        self._resume_monitor("after the homing")
        deadline = time.monotonic() + SAMPLE_WAIT_S
        seq0 = int(getattr(self.pre.sample, "seq", 0) or 0)
        sample = None
        while time.monotonic() < deadline:
            mon = monitor.monitor(self.arm_id)
            sample = mon.snapshot() if mon is not None else None
            if (
                sample is not None
                and int(getattr(sample, "seq", 0) or 0) > seq0
                and getattr(sample, "rail_homed", False)
            ):
                break
            time.sleep(0.05)
        if sample is None:
            return None
        status, detail = monitor.status_of(self.arm_id)
        return sample_to_telemetry(
            self.arm_id,
            status,
            detail,
            None,
            sample,
            backstops_match=backstops_match(sample, self.pre.arm),
            maintenance_busy=False,
        )

    def _finish(self, ok: bool, detail: str, sdk_codes: dict[str, int], after) -> None:
        result = ArmMaintenanceResult(
            arm_id=self.arm_id,
            op="home_rail",
            path="session",
            ok=ok,
            detail=detail,
            sdk_codes={str(k): int(v) for k, v in sdk_codes.items()},
            before=self.pre.before,
            after=after,
            rail_sweep=self.pre.verdict,
            status="done",
            job_id=self.job_id,
        )
        with self._lock:
            self.state.result = result
            self.state.finished_at = time.monotonic()
            self.state.owns_boxes = False
        self.service._record(self.arm_id, result)
        self._set("done" if ok else "failed", detail)

    def _fail(self, reason: str, sdk_codes: dict[str, int]) -> None:
        phase = self.progress.phase
        with self._lock:
            self.state.owns_boxes = False  # predicate false BEFORE the monitor resumes
        self._resume_monitor("after failure")
        detail = f"rail homing job failed during {phase}: {reason}"
        logger.error("rail homing job %s (%s): %s", self.job_id, self.arm_id, detail)
        self._finish(False, detail, sdk_codes, None)


class RailHomingService:
    """REST decision + job registry for ``home_rail`` (module docstring)."""

    def __init__(
        self,
        cfg: RuntimeConfig,
        monitor: HardwareStateMonitor,
        checker: RailSweepChecker | None,
        manager: SessionManager,
        *,
        clock=time.monotonic,
    ) -> None:
        self.cfg = cfg
        self.monitor = monitor
        self.manager = manager
        self.clock = clock
        self.twin_overlay: Any = None  # assigned by Runtime after construction
        self.planner: PrePositionPlanner | None = None
        if checker is not None:
            from ..session.hardware import default_servo_limits, servo_executor_caps

            host_slew = cfg.control.jog.slew_rad_per_tick * JOB_SPEED_SCALE
            servo = default_servo_limits()
            caps = (
                servo_executor_caps(servo, cfg.control.rate_hz, host_slew, scale=JOB_SPEED_SCALE)
                if servo is not None
                else None
            )
            self.planner = PrePositionPlanner(
                checker,
                dict(cfg.twin_overlay.rail_fallback_m),
                slew_rad_per_tick=caps.slew_rad_per_tick if caps is not None else host_slew,
                rate_hz=cfg.control.rate_hz,
                cart_step_m=caps.cart_step_m if caps is not None else None,
                lever_arm_m=caps.lever_arm_m if caps is not None else None,
            )
        self._jobs: dict[str, RailHomingJob] = {}
        self._last: dict[str, ArmMaintenanceResult] = {}
        self._pending: set[str] = set()  # arms with a request being evaluated (reservation)
        self._dry_runs: dict[str, _CachedPlan] = {}  # the plan the operator saw, per arm
        self._lock = threading.Lock()
        monitor.jobs = self
        manager.maintenance_guard = self  # create() refuses any session while active

    # -- registry (MaintenanceJobsLike) --------------------------------------------------
    def job(self, arm_id: str) -> RailHomingJob | None:
        with self._lock:
            return self._jobs.get(arm_id)

    @property
    def active_arm(self) -> str | None:
        """Arm whose job is still running OR whose ``home_rail`` request is being
        evaluated (reserved; ``None`` = none). Every "rail homing in progress"
        refusal (other ops, sessions, a second request) reads this."""
        with self._lock:
            pending = next(iter(self._pending), None)
            jobs = list(self._jobs.items())
        if pending is not None:
            return pending
        for arm_id, job in jobs:
            if not job.terminal:
                return arm_id
        return None

    def cached_plan(self, arm_id: str) -> _CachedPlan | None:
        """The last dry run's plan for ``arm_id`` (tests / diagnostics)."""
        with self._lock:
            return self._dry_runs.get(arm_id)

    def _reserve(self, arm_id: str) -> None:
        """Atomic check-and-reserve: refuse while any job is non-terminal or any
        request is pending, else mark ``arm_id`` pending."""
        from ..session.hardware import arm_label

        with self._lock:
            busy = next(iter(self._pending), None)
            if busy is None:
                for other_id, job in self._jobs.items():
                    if not job.terminal:
                        busy = other_id
                        break
            if busy is not None:
                raise MaintenanceUnavailableError(
                    f"home_rail refused: rail homing in progress on the {arm_label(busy)} - "
                    "wait for it to finish"
                )
            self._pending.add(arm_id)

    def _release(self, arm_id: str) -> None:
        with self._lock:
            self._pending.discard(arm_id)

    @property
    def active(self) -> bool:
        return self.active_arm is not None

    @property
    def owns_boxes(self) -> bool:
        """A job's driver holds a control box (the monitor hand-over predicate)."""
        with self._lock:
            jobs = list(self._jobs.values())
        return any(job.owns_boxes for job in jobs)

    def busy(self, arm_id: str) -> bool:
        job = self.job(arm_id)
        return job is not None and not job.terminal

    def progress(self, arm_id: str) -> MaintenanceProgress | None:
        job = self.job(arm_id)
        if job is None:
            return None
        if job.terminal:
            finished = job.state.finished_at
            if finished is not None and time.monotonic() - finished > PROGRESS_LINGER_S:
                return None
        return job.progress

    def last(self, arm_id: str) -> ArmMaintenanceResult | None:
        with self._lock:
            return self._last.get(arm_id)

    def _record(self, arm_id: str, result: ArmMaintenanceResult) -> None:
        with self._lock:
            self._last[arm_id] = result

    def stop(self, timeout_s: float = STOP_JOIN_TIMEOUT_S) -> None:
        """``Runtime.stop``: cancel every running job at its next safe point (the
        arm holds on the validated path; an SDK homing wait is never cut) and wait
        for its teardown + resume; a job still alive afterwards is logged as
        abandoned (its daemon thread dies with the interpreter)."""
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            if job.is_alive():
                job.cancel()
        for job in jobs:
            if job.is_alive():
                job.join(timeout_s)
                if job.is_alive():
                    logger.error(
                        "rail homing job %s (%s) abandoned after %.0f s in phase %s - the arm "
                        "may be left enabled",
                        job.job_id,
                        job.arm_id,
                        timeout_s,
                        job.progress.phase,
                    )

    # -- the REST decision --------------------------------------------------------------
    def request(self, arm_id: str, *, dry_run: bool = False) -> ArmMaintenanceResult:
        """``POST .../maintenance {op: home_rail}`` (module docstring). Raises
        :class:`MaintenanceUnavailableError` (409) for the preflight refusals and
        while a job runs / a request is being evaluated on any arm. The arm is
        RESERVED for the whole evaluation (``_reserve``); the reservation passes
        to the job on the 202 and is released on every other exit."""
        self._reserve(arm_id)
        started = False
        try:
            result, started = self._request_reserved(arm_id, dry_run=dry_run)
            return result
        finally:
            if not started:
                self._release(arm_id)

    def _request_reserved(self, arm_id: str, *, dry_run: bool) -> tuple[ArmMaintenanceResult, bool]:
        """``(result, job_started)`` with ``arm_id`` reserved by the caller."""
        pre = self.monitor.home_rail_preflight(arm_id)
        if self.planner is None:
            raise MaintenanceUnavailableError(
                "home_rail needs the digital twin to plan the pre-positioning motion"
            )
        plan, waypoints, reused = self._plan_for(arm_id, pre, dry_run=dry_run)
        pre.verdict.pre_position = plan
        if not plan.needed:
            with self._lock:
                self._dry_runs.pop(arm_id, None)
            if dry_run:
                return pre.result(dry_run=True), False
            result = self.monitor.home_rail_execute(pre, HOME_RAIL_TIMEOUT_S)
            self._record(arm_id, result)
            return result, False
        if not plan.clear or waypoints is None:
            detail = (
                f"home_rail refused: {describe_verdict(pre.verdict, dry_run=False)}; {plan.detail}"
            )
            result = pre.result(dry_run=dry_run, detail=detail).model_copy(
                update={"ok": False, "status": "refused"}
            )
            if not dry_run:
                self._record(arm_id, result)
            return result, False
        planned = (
            f"rail sweep blocked at the current posture; {plan.detail}; then the rail homes "
            "and the arm holds that posture"
        )
        if dry_run:  # ok: the op CAN proceed (with the motion) - the UI asks for confirmation
            with self._lock:  # the confirm executes THIS plan (re-validated), not a new one
                self._dry_runs[arm_id] = _CachedPlan(plan, waypoints, pre.verdict)
            return (
                pre.result(dry_run=True, detail=planned + " (dry run, nothing written)").model_copy(
                    update={"ok": True}
                ),
                False,
            )
        job = RailHomingJob(self, pre, plan, waypoints)
        accepted = ArmMaintenanceResult(
            arm_id=arm_id,
            op="home_rail",
            path="session",
            ok=True,
            detail=f"accepted: {planned}",
            sdk_codes={},
            before=pre.before,
            after=None,
            rail_sweep=pre.verdict,
            status="accepted",
            job_id=job.job_id,
        )
        with self._lock:
            existing = self._jobs.get(arm_id)
            if existing is not None and not existing.terminal:  # never overwrite a live job
                raise MaintenanceUnavailableError(
                    f"home_rail refused: rail homing job {existing.job_id} still runs on {arm_id!r}"
                )
            self._jobs[arm_id] = job
            self._last[arm_id] = accepted  # /last answers "accepted" until the job ends
            self._dry_runs.pop(arm_id, None)
            self._pending.discard(arm_id)  # the reservation is now the job itself
        logger.warning(
            "home_rail on %s: job %s accepted - planned pre-positioning (%d waypoints%s) then "
            "HOMING the linear track",
            arm_id,
            job.job_id,
            plan.waypoints,
            ", the dry run's plan re-validated" if reused else "",
        )
        job.start()
        return accepted, True

    def _plan_for(
        self, arm_id: str, pre: HomeRailPreflight, *, dry_run: bool
    ) -> tuple[PrePositionPlan, list[list[float]] | None, bool]:
        """``(plan, waypoints, reused)``: on a confirm with a cached dry run whose
        start posture still matches (``START_POSTURE_TOL_RAD``), the cached plan
        re-validated by ``check_path`` against the fresh samples (``reused``
        True); otherwise a fresh evaluation - refused ("plan changed") when it
        differs from the plan the operator confirmed."""
        from ..session.hardware import START_POSTURE_TOL_RAD

        assert self.planner is not None
        verdict = pre.verdict
        cached = None if dry_run else self.cached_plan(arm_id)
        if cached is not None and not verdict.clear:
            moved = float(
                np.max(np.abs(np.asarray(verdict.q_checked) - np.asarray(cached.verdict.q_checked)))
            )
            if moved <= START_POSTURE_TOL_RAD:
                path = self.planner.revalidate(arm_id, cached.waypoints, pre.samples)
                if path.clear:
                    plan = cached.plan.model_copy(
                        update={"checked_rail_positions": path.checked_rail_positions}
                    )
                    return plan, [list(w) for w in cached.waypoints], True
                logger.warning(
                    "home_rail on %s: the dry run's plan no longer validates (%s) - re-planning",
                    arm_id,
                    path.detail,
                )
        plan, waypoints = self.planner.evaluate(arm_id, verdict, pre.samples)
        if cached is not None and plan.needed and plan.clear and waypoints is not None:
            same_target = len(plan.target_q) == len(cached.plan.target_q) and np.allclose(
                plan.target_q, cached.plan.target_q, atol=1e-6
            )
            if not same_target or plan.waypoints != cached.plan.waypoints:
                detail = (
                    "the pre-positioning plan changed since the dry run (was "
                    f"{cached.plan.waypoints} waypoints to the {cached.plan.source} posture, now "
                    f"{plan.waypoints} to the {plan.source} posture) - open Home rail again to "
                    "re-confirm"
                )
                return plan.model_copy(update={"clear": False, "detail": detail}), None, False
        return plan, waypoints, False


__all__ = [
    "JOB_SPEED_SCALE",
    "PHASE_ORDER",
    "PHASE_PROGRESS",
    "POSTURE_REACHED_TOL_RAD",
    "PROGRESS_LINGER_S",
    "REFUSAL_HINT",
    "STOP_JOIN_TIMEOUT_S",
    "JobFailed",
    "PrePositionPlanner",
    "RailHomingJob",
    "RailHomingService",
]
