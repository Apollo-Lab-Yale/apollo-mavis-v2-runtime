"""ControlLoop — THE single command chokepoint (04-runtime §6; 11-safety §4).

The only code in the stack that resolves per-tick commands and deposits them
for dispatch; every source (teleop, jog, goto/planner waypoints, future
policy/takeover) passes the safety gate inline in the tick. Dispatch itself
happens on the per-arm :class:`ArmSender` threads consuming the ``q_cmd``
slots. 100 Hz, monotonic absolute-deadline pacing, overruns skipped (no
catch-up bursts). ``run_tick(now)`` is callable synchronously for
deterministic tests.

2026-09-08 (not yet in the design docs): ``goto_profile {profile_id}`` walks the
session arms to a CHOSEN saved profile through the reset-to-initial machinery
(:meth:`ControlLoop._op_goto_profile` -> ``SessionManager.request_goto_profile``).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from apollo_mavis_v2_core import (
    ArmState,
    Command,
    CommandResult,
    CommandSource,
    Pose,
    ProfileNotFoundError,
    ProfileStore,
    Twist,
    se3,
)
from apollo_mavis_v2_core.protocol import HELD_CODES, JointTargetArgs, TrackerSettingsArgs

from ..config import ControlConfig
from ..errors import SafetyConfigError
from ..profiles.store import save_from_states, save_initial_overwrite
from ..safety.gate import SafetyGate
from .joint_panel import RATIO_EPS, JogState, PlanExecutor
from .snapshot import StateSnapshot
from .teleop import TargetIntegrator, held_to_twist, twist_to_control_frame
from .tracker_teleop import TRACKER_CLUTCH_CODE, interp_pose

if TYPE_CHECKING:
    from apollo_mavis_v2_core import IKSolver, Pose, WorkcellInterface

    from ..bus import RuntimeBus
    from ..safety.supervisor import SafetySupervisor
    from .tracker_teleop import TrackerTeleop

logger = logging.getLogger(__name__)

RAIL_TRAVEL_M = se3.RAIL_TRAVEL_M
GRIPPER_SEND_EVERY_N_TICKS = 10  # <= 10 Hz (modbus is slow)
PLAN_STATUS_LINGER_TICKS = 100  # keep "done"/"failed" visible ~1 s
# ``plan_cancel_reason`` prefix of the gate-held abort (``HardwareSessionConfig.
# plan_gate_hold_s``, 2026-09-08 evening): the SessionManager matches on it to report
# "the motion was held by the safety gate (<pair> at <mm> mm)" instead of a plain cancel.
GATE_HOLD_PREFIX = "held by the safety gate"
DEVICE_ACTION_LINGER_S = 1.0  # telemetry shows the last device-sourced discrete action this long
# Driver events (hardware ``events.py``) the loop consumes from ``workcell.drain_events()``
# every tick (04-runtime §15; phase-09b), dispatched by CLASS NAME so the runtime never
# imports the optional hardware package: FaultEvent -> the arm stops (sender paused, held),
# ReseedEvent / RecoveredEvent -> re-seed from measured + RECOVERING until every live
# input is released, StudioConflictWarning -> a lingering telemetry warning.
FAULT_EVENT_NAMES: frozenset[str] = frozenset(
    {"FaultEvent", "RecoveredEvent", "ReseedEvent", "StudioConflictWarning"}
)
STUDIO_WARNING_LINGER_S = 5.0  # a StudioConflictWarning stays in arms[*].fault_detail this long
STUDIO_WARNING_DEFAULT = "close UFACTORY Studio live control"
# The Manipulation Arm (arm id ``grip``: xArm Gripper G2 + wrist camera) is the
# default teleop arm on every workcell, hardware and sim (user decision
# 2026-09-04); the Perception Arm (``view``) is reached with Tab / switch_arm.
DEFAULT_ACTIVE_ARM = "grip"
# Nack details of the gate ops outside a policy session (15-online-dagger D3): Space and
# the explicit takeover / handback actions share one string; train_now its own.
TAKEOVER_UNAVAILABLE = "takeover not available in teleop"
NOT_ONLINE_DAGGER = "not an Online DAgger session"
# phase-15 (16-gello §8.2; 2026-09-09 review): the GELLO Cockpit buttons outside a gello
# session - the documented mode refusal, not the dispatcher's "unknown op"
NOT_GELLO = "not a GELLO Manipulation session"


def controller_error_title(code: int) -> str:
    """``"controller error 24: Speed Exceeds Limit"`` from the SDK's ``x_code``
    table via the hardware package (the runtime never imports ``xarm`` itself);
    ``"controller error <code>"`` without the ``[hardware]`` extra; ``""`` for 0."""
    code = int(code)
    if code == 0:
        return ""
    try:
        from apollo_mavis_v2_hardware import controller_error_title as _title  # [hardware]
    except Exception:  # noqa: BLE001 - optional extra
        return f"controller error {code}"
    try:
        return _title(code) or f"controller error {code}"
    except Exception:  # noqa: BLE001 - table lookup is best-effort
        return f"controller error {code}"


def default_active_arm(arms: Sequence[str]) -> str | None:
    """Initial ``active_arm`` of a session: :data:`DEFAULT_ACTIVE_ARM` when the
    session includes it (whatever its position in ``arms``), else the first
    session arm, else ``None`` (no arms)."""
    if DEFAULT_ACTIVE_ARM in arms:
        return DEFAULT_ACTIVE_ARM
    return arms[0] if arms else None


@dataclass(frozen=True)
class HeldSources:
    """The tick's held codes by source, each with its own scale (13-tracker §1.1).

    ``ws``: ``KeysMsg.held`` (keyboard + gamepad) scaled by the WS
    ``InputWatchdog``; ``device``: codes injected by the tracker reader from the
    Vive controller's buttons, scaled ``1.0`` while the sample is fresh /
    ``0.0`` when stale (then the set is empty). ``held`` is the merged set the
    tick acts on; a code held by both sources takes the larger scale, so a
    latched WS deadman (``AWAIT_EMPTY``) never zeroes device-driven motion.
    """

    ws: frozenset[str] = frozenset()
    ws_scale: float = 1.0
    device: frozenset[str] = frozenset()
    device_scale: float = 0.0

    @property
    def held(self) -> frozenset[str]:
        return self.ws | self.device

    def scale_for(self, code: str) -> float:
        """Largest scale among the sources holding ``code``; 0.0 if none does."""
        scale = 0.0
        if code in self.ws:
            scale = max(scale, self.ws_scale)
        if code in self.device:
            scale = max(scale, self.device_scale)
        return scale

    def moving(self, codes: frozenset[str]) -> bool:
        """True when some code in ``codes`` is held by a source with scale > 0."""
        return any(self.scale_for(c) > 0.0 for c in self.held & codes)


class ControlLoop:
    """The 100 Hz mode loop for one session (teleop in phase-05)."""

    def __init__(
        self,
        workcell: WorkcellInterface,
        cfg: ControlConfig,
        bus: RuntimeBus,
        supervisor: SafetySupervisor,
        session_arms: list[str],
        *,
        ik: IKSolver | None = None,
        kin=None,  # SceneKinematics-like: tcp_world(arm,q), base_quat_world(arm)
        planner=None,  # DigitalTwinInterface-like .plan(PlanRequest) -> PlanResult
        profile_store: ProfileStore | None = None,
        workcell_kind: str = "sim",
        recorder=None,  # RecorderThread (collect/dagger): episode ops + status
        gripper_arms: Iterable[str] | None = None,  # None = every session arm
        tracker: TrackerTeleop | None = None,  # clutched Vive-tracker target provider
        clock: Callable[[], float] = time.monotonic,
        plan_gate_hold_s: float = 3.0,  # HardwareSessionConfig.plan_gate_hold_s (sim too)
        speed_scale: float = 0.1,  # SessionSpec.speed_scale: the joint-panel goto's plans are
        #   judged at it (PlanRequest.speed_scale; the default is the slowest speed offered)
    ) -> None:
        # 11-safety §4 item 4 (phase-09c): a hardware loop MUST dispatch through a
        # SafetyGate bound to a live digital twin - the chokepoint refuses anything else.
        if workcell_kind == "hardware" and not (
            isinstance(supervisor.gate, SafetyGate) and supervisor.twin is not None
        ):
            raise SafetyConfigError(
                "hardware sessions require a SafetyGate bound to a live digital twin "
                "(11-safety §4); refusing to build the control loop"
            )
        self.workcell = workcell
        self.cfg = cfg
        self.bus = bus
        self.supervisor = supervisor
        self.session_arms = list(session_arms)
        # Arms that carry a gripper (camera-only arms have none): only these
        # get F/H integration, gripper sends and start_from gripper targets.
        self.gripper_arms: frozenset[str] = frozenset(
            self.session_arms if gripper_arms is None else gripper_arms
        )
        self.ik = ik
        self.kin = kin
        self.planner = planner
        self.profile_store = profile_store
        self.workcell_kind = workcell_kind
        self.recorder = recorder
        self.tracker = tracker
        self._clock = clock
        self.dt = 1.0 / cfg.rate_hz
        self.sources = HeldSources()  # this tick's held codes by source (step 2)

        self.active_arm: str | None = default_active_arm(self.session_arms)
        self.jog = JogState(cfg.jog)
        self.plans = PlanExecutor(cfg.jog)
        self.integrator = TargetIntegrator(cfg.leash.pos_m, cfg.leash.rot_rad)
        self.episode_state: str = "idle"  # phase-07 recorder wires this

        self._last_cmd: dict[str, np.ndarray] = {}
        self._grip_frac: dict[str, float] = {}
        self._states: dict[str, ArmState] = {}
        self._teleop_seeded: set[str] = set()
        # This tick's KEYBOARD twist per arm (world frame), set by ``_teleop_step``'s
        # key branch and consumed after the joint step cap (``_hold_key_target``,
        # 2026-09-09): a capped keyboard tick pulls the integrated target back along
        # the driven axes so it never runs ahead of what the arm can follow.
        self._key_twist: dict[str, Twist] = {}
        # Arm of the current clutch session (clutch PRESS-edge tracking).
        self._clutch_arm: str | None = None
        self._arm_source: dict[str, CommandSource] = {}  # per-arm last resolving source
        self._edge_seq_seen: int | None = None  # controller.edge_seq adopted last tick
        self._device_action: tuple[str, float] | None = None  # (label, show until)
        self._seeded = False
        self._plan_state: dict[str, str] = {}  # arm -> planning|executing|failed
        self._plan_clear_at: dict[str, int] = {}
        self._plan_status: str | None = None  # session-level lifecycle string
        # Why the last running plan was cancelled ("movement key", "jog", "arm switch",
        # "driver fault", ...); None after a plan finished or a new one was loaded. The
        # SessionManager's return-to-start worker reads it (04-runtime §10.5).
        self.plan_cancel_reason: str | None = None
        # An interruptible plan (the return-to-start motion) is cancelled by a jog or an
        # arm switch too, not only by a movement key; start_from plans are not.
        self._plan_interruptible = False
        # Gripper targets an interruptible plan carries are applied on ARRIVAL (never
        # at load: a cancelled return must leave the gripper untouched). Since the
        # 2026-09-08 sequential execution the manager submits the profile's gripper
        # targets with the LAST arm's plan, so a target may name an arm that is not in
        # this plan's waypoints: every target is deferred and applied when the plan is
        # done (``_finish_plan``), whichever arm carried it.
        self._plan_gripper_on_arrival: dict[str, float] = {}
        # Arms whose executor returned the GOAL this tick (2026-09-08 review): the plan
        # is finished only once the gated output equals that goal (``_confirm_plan_
        # arrivals``, after the gate); a goal step the gate holds puts the arm back into
        # the executor, so ``plans.active_arms`` never empties one slew step short.
        self._plan_arriving: dict[str, np.ndarray] = {}
        # Gate-held abort (2026-09-08 evening; ``HardwareSessionConfig.plan_gate_hold_s``):
        # the clock reading at which the gate began holding the PLANNER-sourced command
        # of the running plan with no waypoint progress; None while it progresses. Past
        # ``plan_gate_hold_s`` the plan is cancelled with the blocking pair in the reason
        # (``_plan_gate_watch``). On the real cell (23:16:52) a return sat gate-blocked at
        # 5.2 mm for the whole 30 s budget before anyone learned which pair it was.
        self.plan_gate_hold_s = max(0.0, float(plan_gate_hold_s))
        self._plan_gate_hold_since: float | None = None
        self.speed_scale = min(max(float(speed_scale), 1e-6), 1.0)
        # The browser's control socket dropped since the last tick (Runtime hook).
        self._controller_dropped = False
        # Process-stall detection (04-runtime §6; 2026-09-07): the previous tick's clock.
        self._last_tick_now: float | None = None
        self.process_stalls = 0
        self._plan_status_clear_at: int | None = None
        # Driver fault plumbing (phase-09b; 04-runtime §15). ``_faulted``: arms a
        # FaultEvent stopped (sender paused, held, nothing published); ``_recovering``:
        # arms re-seeded by a ReseedEvent / RecoveredEvent that stay held until every
        # live input is released (clutch re-grip / empty held set); ``_fault_text``:
        # telemetry ``fault_detail`` per arm, kept through RECOVERING, cleared on
        # RUNNING; ``_warnings``: lingering StudioConflictWarning text per arm.
        self._faulted: set[str] = set()
        self._recovering: set[str] = set()
        self._fault_text: dict[str, str] = {}
        self._warnings: dict[str, tuple[str, float]] = {}  # arm -> (text, show until)
        self._fault_state_reported: str | None = None
        self.on_fault_state: Callable[[str | None], None] | None = None
        #   ^ SessionManager hook: "fault" | "recovering" | None (all arms running)
        # SessionManager hook for the `reset_to_initial` key (2026-09-08): the loop
        # validates the request inline (fast), resolves the target profile ONCE and
        # hands both to the manager, which plans the motion on the session twin off
        # the loop thread and returns a short ack detail immediately. Takes the
        # profile so the manager does not re-scan the store from the loop thread.
        # None = no manager attached (unit-test loops).
        self.on_reset_to_initial: Callable[[object], tuple[bool, str]] | None = None
        # Same contract for ``goto_profile`` (2026-09-08): the loop resolves the CHOSEN
        # profile inline and the manager runs the reset-to-initial motion toward it.
        self.on_goto_profile: Callable[[object], tuple[bool, str]] | None = None
        self.fault_events = 0  # FaultEvents consumed (tests / diagnostics)
        self.recoveries = 0  # re-seeds performed after Reseed/RecoveredEvents

        self.tick_count = 0
        self.overrun_count = 0
        self.tick_durations: list[float] = []  # perf harness (bounded)
        # Health line + edge logging (2026-09-07; 04-runtime §14 "Logging"). The
        # loop is the one place that sees every teleop input and every output, so
        # it writes ONE INFO line per ``cfg.health_log_every_s`` summarising the
        # window (tick rate / overruns / clutch / tracker ages / gate / IK slips /
        # servo-stream stats) and an edge line on every transition that changes
        # what the operator's hand does: clutch, controller stream fresh/stale, WS
        # watchdog latch. Counters are cumulative; the line prints window deltas.
        self.ik_slips = 0  # _solve_target residual freezes (target re-anchored)
        self.ik_diverged = 0  # _solve_target divergences (tick held)
        # Ticks on which the uniform step cap (``cfg.dq_max_rad`` + the streamer's
        # lever-weighted Cartesian bound, ``_cap_joint_step``) bound some arm's command
        # (2026-09-09): the axis-purity diagnostic - a saturated tick is one the
        # operator's requested rate exceeded what the arm can execute. On a hardware
        # loop a held keyboard key binds it on nearly every tick (the requested
        # 0.12 m/s exceeds the streamer's lever-weighted capacity in most postures).
        self.clamp_ticks = 0
        # ``_cap_joint_step``'s Cartesian bound, derived from ``cfg.jog`` on first use
        # and whenever the jog config object changes (tests swap ``loop.cfg``).
        self._cart_bound_src: object | None = None
        self._cart_bound_val: tuple[float | None, np.ndarray | None] = (None, None)
        self._health_next: float | None = None
        self._health_durations: list[float] = []
        self._health_prev = {
            "tick": 0,
            "overrun": 0,
            "slip": 0,
            "diverged": 0,
            "tslip": 0,
            "clamp": 0,
        }
        self._clutch_logged = False
        self._device_fresh_logged: bool | None = None
        self._watchdog_logged = False
        self._senders: dict = {}
        self._thread: threading.Thread | None = None
        self._running = False

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        """Seed from measured state, spawn ArmSenders + the loop thread."""
        from .arm_sender import ArmSender

        self._seed_from_measured()
        for arm_id in self.session_arms:
            sender = ArmSender(arm_id, self.workcell.arms[arm_id], self.bus.arm_slot(arm_id))
            sender.start()
            self._senders[arm_id] = sender
        self._running = True
        self._thread = threading.Thread(target=self._run, name="control-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        for sender in self._senders.values():
            sender.stop()
        self._senders.clear()

    def _run(self) -> None:
        next_t = self._clock()
        while self._running:
            t0 = self._clock()
            try:
                self.run_tick(t0)
            except Exception:
                logger.exception("control tick failed")
            dur = self._clock() - t0
            if len(self.tick_durations) < 10000:
                self.tick_durations.append(dur)
            if len(self._health_durations) < 10000:
                self._health_durations.append(dur)
            next_t += self.dt
            lag = self._clock() - next_t
            if lag > self.dt:  # overrun: skip catch-up, no burst commands
                next_t = self._clock()
                self.overrun_count += 1
            elif lag < 0.0:
                time.sleep(-lag)

    # -- seeding -----------------------------------------------------------------
    def _seed_from_measured(self) -> None:
        states = self.workcell.states()
        self._states = states
        for arm_id in self.session_arms:
            st = states[arm_id]
            self._last_cmd[arm_id] = np.array(st.q, dtype=np.float64)
            if arm_id in self.gripper_arms:
                self._grip_frac[arm_id] = float(st.gripper.open_frac)
        self._teleop_seeded.clear()
        self._seeded = True

    def reseed_arm(self, arm_id: str) -> None:
        """Post-recovery re-seed: targets from measured, gate last-safe reset; the
        gripper target follows the MEASURED opening too (a key held through the
        fault integrated nothing, so nothing pre-fault is replayed)."""
        st = self.workcell.arms[arm_id].get_state()
        self._last_cmd[arm_id] = np.array(st.q, dtype=np.float64)
        if arm_id in self.gripper_arms:
            self._grip_frac[arm_id] = float(st.gripper.open_frac)
        self._teleop_seeded.discard(arm_id)
        if self.ik is not None:
            self.ik.reset(arm_id, st.q)
        self.supervisor.reseed(arm_id, st.q)

    # -- the tick (order fixed, 04-runtime §6) -------------------------------------
    def note_controller_disconnect(self) -> None:
        """Runtime hook (any thread): the browser's control socket dropped — an
        interruptible plan (the return-to-start motion) is cancelled on the next tick."""
        self._controller_dropped = True

    def run_tick(self, now: float | None = None) -> StateSnapshot:
        now = self._clock() if now is None else now
        self.tick_count += 1
        if not self._seeded:
            self._seed_from_measured()
        # Process stall (the encoder open / flush holds the GIL for 150-300 ms): the
        # whole process froze, not the browser. Hold every arm this tick (no
        # catch-up step) and credit the silence to the stall so the deadman does not
        # latch a browser that was fresh when the stall began.
        stalled = False
        if self._last_tick_now is not None:
            gap = now - self._last_tick_now
            if gap > self.supervisor.watchdog.timeout_s:
                stalled = True
                self.process_stalls += 1
                self.supervisor.watchdog.on_process_stall(now, self._last_tick_now)
                logger.warning("process stall %.0f ms (tick gap): arms held this tick", gap * 1e3)
        self._last_tick_now = now
        self.bus.commands.drain(self._handle_command)  # 1

        got = self.bus.held_keys.get()  # 2
        held_ws: frozenset[str] = got[0].held if got is not None else frozenset()
        scale = self.supervisor.watchdog.scale(now)  # WS-source scale
        watchdog_tripped = self.supervisor.watchdog.tripped
        if watchdog_tripped != self._watchdog_logged:
            self._watchdog_logged = watchdog_tripped
            if watchdog_tripped:
                dropped = self.jog.clear_all()  # see JogState.clear_all
                logger.warning(
                    "ws input watchdog LATCHED: browser keys ignored until every key is "
                    "released (held %s); device-held codes keep working; jog targets "
                    "dropped (%s)",
                    sorted(held_ws),
                    ", ".join(dropped) or "none",
                )
            else:
                logger.info("ws input watchdog cleared")
        device_codes, device_scale = self._device_inputs(now)
        self.sources = HeldSources(held_ws, scale, device_codes, device_scale)
        held = self.sources.held  # held_eff = held ∪ device_codes (13-tracker §1.1)
        if self.tracker is not None:
            clutch = TRACKER_CLUTCH_CODE in held
            self.tracker.clutch = clutch
            if clutch != self._clutch_logged:
                self._clutch_logged = clutch
                logger.info(
                    "clutch %s (arm %s, source %s)",
                    "ENGAGED" if clutch else "released",
                    self.active_arm,
                    "device" if TRACKER_CLUTCH_CODE in device_codes else "ws",
                )
            if not clutch:
                self._clutch_arm = None  # clutch up -> next press is a rising edge

        states = self.workcell.states()  # 3 (driver caches; never blocks)
        self._states = states
        # 3b (phase-09b, §15): an arm re-seeded on an earlier tick leaves RECOVERING
        # once this tick's inputs (step 2, sampled AFTER the re-seed) hold nothing
        # live (clutch released / keys up or watchdog-latched); then this tick's
        # driver events -> per-arm FAULT / RECOVERING. RECOVERING therefore lasts at
        # least one tick and never ends on inputs sampled before the re-seed.
        self._update_recovering()
        self._drain_driver_events(now)
        self._publish_fault_state()
        self.supervisor.sync(states, now)  # 4
        if self.ik is not None:
            self.ik.sync_passive(states)
        q_meas = {a: states[a].q for a in self.session_arms}

        # A held movement key (from a live source) cancels running plans (04-runtime §7).
        # An INTERRUPTIBLE plan (return-to-start) cancels on the PRESENCE of a movement
        # code from any source, whatever the deadman scale: a key held under a latched
        # deadman is still the operator saying "stop" (safety, 2026-09-07).
        if self.plans.active_arms:
            if self.sources.moving(HELD_CODES):
                self._cancel_plans("movement key")
            elif self._plan_interruptible and (held & HELD_CODES):
                self._cancel_plans("movement key")
            elif self._controller_dropped:
                self._interrupt_plan_for("browser disconnected")
        self._controller_dropped = False

        # 5-7: per-arm action resolution -> commanded q (mode hook, phase-08).
        self._key_twist.clear()
        if stalled:
            resolved = dict.fromkeys(self.session_arms)  # hold: last command, no jump
            source = CommandSource.TELEOP
        else:
            resolved, source = self._resolve_arms(states, held, scale, now)
        if self.tracker is not None:
            self.tracker.end_tick(now)  # not consulted this tick -> anchors cleared
        q_cmd: dict[str, np.ndarray] = {
            arm_id: (q if q is not None else self._last_cmd[arm_id])
            for arm_id, q in resolved.items()
        }

        # Per-tick joint step cap (dq_max, uniform scaling) + rail bound, before the gate.
        clamped = False
        for arm_id, q in q_cmd.items():
            q, capped = self._cap_joint_step(q, self._last_cmd[arm_id])
            if capped:
                clamped = True
                self._hold_key_target(arm_id, q)
            q_cmd[arm_id] = q
        if clamped:
            self.clamp_ticks += 1

        dec = self.supervisor.filter(q_cmd, q_meas, source)  # 8
        self._post_filter(dec, now)
        self._confirm_plan_arrivals(dec)  # a goal the gate held stays in the executor
        if self.plans.active_arms:
            self._plan_gate_watch(dec, now)  # gate-held abort of a running plan

        for arm_id, q in dec.q_out.items():  # 9
            self._last_cmd[arm_id] = np.array(q)
            if arm_id in self._faulted:
                continue  # stopped by a driver fault: nothing dispatched until the re-seed
            self.bus.arm_slot(arm_id).put(np.array(q))

        self._gripper_step(self.sources)
        self._expire_plan_status()

        episode = None
        if self.recorder is not None:
            episode = self.recorder.status()
            self.episode_state = episode.state  # joint_target nack while recording

        snap = StateSnapshot(  # 10
            t_mono=now,
            wallclock_ns=time.time_ns(),
            tick=self.tick_count,
            arms=states,
            q_cmd={a: q.copy() for a, q in self._last_cmd.items()},
            active_arm=self.active_arm,
            gate=self.supervisor.merged_report(),
            clearances=self.supervisor.clearances,
            gripper_frac=dict(self._grip_frac),
            episode=episode,
            watchdog_tripped=watchdog_tripped,
            plan_status=dict(self._plan_state),
            session_extra={
                "plan_status": self._plan_status,
                "kind": self.workcell_kind,
                "arm_faults": self._arm_fault_details(now),  # arm -> fault_detail (§15)
                "arm_recovering": sorted(self._recovering),
                "tracker": (
                    {
                        **self.tracker.telemetry_extra(),
                        "device_action": self._device_action_label(now),
                    }
                    if self.tracker is not None
                    else None
                ),
                **self._session_extra(now),
            },
        )
        self.bus.snapshot.put(snap)
        self._health_log(now, held, source, states)
        return snap

    def _cart_bound(self) -> tuple[float | None, np.ndarray | None]:
        """``(plan_cart_step_m, lever[:7])`` of ``cfg.jog`` - the servo streamer's
        lever-weighted Cartesian step bound a hardware bring-up sets through
        ``apply_executor_caps`` - or ``(None, None)`` when no driver published one
        (sim, fakes). Cached per jog config object."""
        jog = self.cfg.jog
        if jog is not self._cart_bound_src:
            cart = getattr(jog, "plan_cart_step_m", None)
            lever = getattr(jog, "plan_lever_arm_m", None)
            if cart is not None and lever is not None and float(cart) > 0.0:
                self._cart_bound_val = (
                    float(cart),
                    np.asarray(list(lever), dtype=np.float64)[:7],
                )
            else:
                self._cart_bound_val = (None, None)
            self._cart_bound_src = jog
        return self._cart_bound_val

    def _cap_joint_step(self, q: np.ndarray, q_last: np.ndarray) -> tuple[np.ndarray, bool]:
        """Bound this tick's joint step ``q[:7] - q_last[:7]`` to what the arm can
        execute in one tick by UNIFORM scaling (04-runtime §6 "Per-tick joint step
        cap", 2026-09-09): the step is divided by the larger of

        - ``max|dq_j| / cfg.dq_max_rad`` - the per-joint velocity cap (a hardware loop
          lowers ``dq_max`` to the streamer's own ``max_joint_vel / rate_hz``,
          ``apply_teleop_caps``), and
        - ``sum|dq_j| * lever_j / cfg.jog.plan_cart_step_m`` - the streamer's
          lever-weighted Cartesian bound (``ServoLimits.max_cart_step_m`` x
          ``lever_arm_m``, set on a hardware loop by ``apply_executor_caps``; ``None``
          in sim, where the term is skipped),

        when that ratio exceeds 1. The whole step shrinks by one factor, so the
        commanded joint-space direction - and with it the Cartesian direction the IK
        solved for - is preserved and the arm is merely slower. This is the rule
        ``PlanExecutor.step`` applies to planned segments (both bounds), minus its
        rounding to whole steps. Returns ``(q, capped)``.

        WHY BOTH BOUNDS. Until 2026-09-09 this was ``np.clip`` per joint: whenever one
        joint saturated the others kept their full step, the direction bent, and a
        held ``W`` / ``S`` drove the TCP up or down as well (operator report; 6-39 mm
        of vertical drift per 2 s hold on the Manipulation Arm, < 1 mm with the clamp
        inactive). The first fix scaled uniformly against ``dq_max`` alone - correct
        in the sim, whose servo has no streamer, but the real ``_ServoStreamer``
        (apollo_mavis_v2_hardware ``driver.py``) clips per joint at ``max_joint_vel *
        dt``, per joint at the acceleration step, and then scales the step so
        ``sum|dq_j| * lever_j <= max_cart_step_m``. That lever estimate is 5-10x
        conservative for a keyboard step (6-12 mm lever-weighted per tick at 100 %
        against the 4 mm cap), so the streamer executed a third to a half of every
        host step, the command wound up to the leash, and the streamer's OWN per-joint
        clip bent the direction every tick - the mechanism this method had removed,
        one layer down (closed-loop streamer emulation from the initial posture at
        100 %: ``W`` 28 mm off on 109 mm, ``S`` 60 mm + 3.2 deg, ``Q`` 131 mm + 15
        deg). With the Cartesian bound folded in here every streamer clip is inactive
        (it runs at most its acceleration ramp behind the command); the same
        emulation gives < 0.5 mm and < 0.06 deg on every key at both speeds
        (``tools/axis_purity_measure.py --streamer``). The executed TCP rate of a
        held key on the real cell is therefore the streamer's lever-weighted
        capacity, ~0.04-0.10 m/s at 100 %, not the 0.12 m/s the key requests.

        A non-positive ``dq_max`` (a mis-derived cap) HOLDS the joints and the rail
        slot at ``q_last`` and reports the tick as capped - the fail-safe direction of
        the old clip, never an unbounded step. The rail slot is otherwise a separate
        axis with its own controller-side speed and keeps an independent bound
        (unchanged), so a carriage step never slows the joints or vice versa."""
        dq_max = float(self.cfg.dq_max_rad)
        q = np.array(q, dtype=np.float64)
        dq = q[:7] - q_last[:7]
        if dq_max <= 0.0:
            capped = bool(np.any(dq != 0.0))
            q[:7] = q_last[:7]
            if q.shape[0] > 7:
                capped = capped or q[7] != q_last[7]
                q[7] = min(max(float(q_last[7]), 0.0), RAIL_TRAVEL_M)
            return q, capped
        ratio = float(np.max(np.abs(dq))) / dq_max
        cart, lever = self._cart_bound()
        if cart is not None and lever is not None:
            n = min(lever.shape[0], 7)
            ratio = max(ratio, float(np.sum(np.abs(dq[:n]) * lever[:n])) / cart)
        capped = ratio > 1.0 + RATIO_EPS
        if capped:
            q[:7] = q_last[:7] + dq / ratio
        if q.shape[0] > 7:
            q[7] = min(max(q[7], q_last[7] - dq_max), q_last[7] + dq_max)
            q[7] = min(max(q[7], 0.0), RAIL_TRAVEL_M)
        return q, capped

    def _hold_key_target(self, arm_id: str, q: np.ndarray) -> None:
        """A KEYBOARD tick whose joint step the cap scaled: pull the integrated target
        back along the DRIVEN axes to the pose ``q`` reaches, so the target advances
        only as far as the arm can follow this tick (04-runtime §6, 2026-09-09).

        The key is a velocity command with no absolute reference, so a target that
        runs ahead of a joint-capped arm buys nothing: it would wind up to the leash,
        the QP would then chase a 25 mm error every tick (into its own per-joint
        velocity box, which bends the direction again), and the arm would keep going
        for a leash after the key is released. Only the components the operator is
        driving are pulled back - the translation along the commanded direction and
        the rotation about the commanded axis; the off-axis position and the
        undriven orientation stay pinned to the line the seed defined, so the IK keeps
        correcting them instead of ratcheting each tick's residue into the anchor.
        The tracker path is untouched: there the hand pose IS the reference and the
        target catches up within the leash by design. No-op when the arm's target was
        not keyboard-driven this tick (tracker, jog, plan, policy)."""
        tw = self._key_twist.get(arm_id)
        target = self.integrator.get(arm_id)
        if tw is None or target is None or self.kin is None:
            return
        achieved = self.kin.tcp_world(arm_id, q)
        pos = target.position
        quat = target.orientation
        v_norm = float(np.linalg.norm(tw.v))
        if v_norm > 0.0:
            d = tw.v / v_norm
            pos = pos + d * float((achieved.position - pos) @ d)
        w_norm = float(np.linalg.norm(tw.w))
        if w_norm > 0.0:
            u = tw.w / w_norm
            # space-frame rotation carrying the target orientation onto the achieved one
            rel = se3.quat_to_rotvec(se3.quat_mul(achieved.orientation, se3.quat_conj(quat)))
            quat = se3.quat_mul(se3.rotvec_to_quat(u * float(rel @ u)), quat)
        self.integrator.reanchor(arm_id, Pose(pos, quat))

    # -- health line (2026-09-07; 04-runtime §14 "Logging") -----------------------------
    def _health_log(
        self, now: float, held: frozenset[str], source: CommandSource, states: dict[str, ArmState]
    ) -> None:
        """One INFO line per ``cfg.health_log_every_s`` (0 = off). Everything on it
        is already in hand at the end of the tick; the only extra work is a sort
        of this window's tick durations and one ``tick_stats()`` call per
        hardware arm, once a second."""
        every = float(getattr(self.cfg, "health_log_every_s", 0.0) or 0.0)
        if every <= 0.0:
            return
        if self._health_next is None:
            self._health_next = now + every
            self._health_snapshot_counters()
            return
        if now < self._health_next:
            return
        prev = self._health_prev
        window = now - (self._health_next - every)
        ticks = self.tick_count - prev["tick"]
        d = sorted(self._health_durations)
        n = len(d)
        p50 = d[n // 2] if n else 0.0
        p99 = d[min(n - 1, int(n * 0.99))] if n else 0.0
        tslip = self.tracker.slip_count if self.tracker is not None else 0
        lag = self._cmd_lag(states)
        try:
            logger.info(
                "loop: %d ticks/%.1fs (%.0f Hz, tick p50 %.1f ms p99 %.1f ms, +%d overruns) "
                "active=%s src=%s held=%s%s%s%s gate=%s%s ik_slips=+%d ik_diverged=+%d "
                "dq_capped=+%d cmd-meas=%s%s%s",
                ticks,
                window,
                ticks / window if window > 0 else 0.0,
                p50 * 1e3,
                p99 * 1e3,
                self.overrun_count - prev["overrun"],
                self.active_arm,
                getattr(source, "value", source),
                sorted(held) if held else "[]",
                self._tracker_health(now, tslip - prev["tslip"]),
                self._mode_health(now),
                " watchdog=LATCHED" if self._watchdog_logged else "",
                self._gate_health(),
                f" faulted={sorted(self._faulted)}" if self._faulted else "",
                self.ik_slips - prev["slip"],
                self.ik_diverged - prev["diverged"],
                self.clamp_ticks - prev["clamp"],
                lag,
                f" recovering={sorted(self._recovering)}" if self._recovering else "",
                self._servo_health(),
            )
        except Exception:  # noqa: BLE001 - a health line must never break the tick
            logger.exception("health line failed")
        self._health_next = now + every
        self._health_snapshot_counters()

    def _health_snapshot_counters(self) -> None:
        self._health_prev = {
            "tick": self.tick_count,
            "overrun": self.overrun_count,
            "slip": self.ik_slips,
            "diverged": self.ik_diverged,
            "tslip": self.tracker.slip_count if self.tracker is not None else 0,
            "clamp": self.clamp_ticks,
        }
        self._health_durations.clear()

    def _tracker_health(self, now: float, slips: int) -> str:
        """`` tracker=<state> pose_age=<ms> ctl_age=<s> engaged=<arm> leash_slips=+N``:
        the pose path and the button path AGE separately (2026-09-06: poses at
        135 Hz while no button event arrived for minutes), so both ages are
        printed; ``leash_slips`` is the hand travel discarded this window."""
        if self.tracker is None:
            return ""
        got = self.tracker.slot.get()
        if got is None:
            return " tracker=no-sample"
        sample = got[0]
        pose_age = now - sample.pose_rx_mono
        state = (
            "tracking"
            if sample.valid and pose_age <= self.tracker.stale_s
            else ("stale" if sample.valid else "invalid")
        )
        ctl = sample.controller
        ctl_age = f"{now - ctl.rx_mono:.1f}s" if ctl is not None else "none"
        out = f" tracker={state} pose_age={pose_age * 1e3:.0f}ms ctl_age={ctl_age}"
        if self.tracker.engaged_arm is not None:
            out += f" engaged={self.tracker.engaged_arm}"
        if slips:
            out += f" leash_slips=+{slips} (+{self.tracker.slip_pos_total_m:.3f} m total)"
        return out

    def _mode_health(self, now: float) -> str:
        """Mode-loop segment of the health line, printed right after the tracker
        segment (``""`` here; ``GelloLoop`` adds `` gello=<state> age=<ms> lag=<rad>``,
        16-gello §6.4)."""
        return ""

    def _gate_health(self) -> str:
        report = self.supervisor.merged_report()
        if report.severity == "ok":
            return "ok"
        return f"{report.severity} min={report.min_clearance_m:.4f}m pairs={report.pairs[:2]}"

    def _cmd_lag(self, states: dict[str, ArmState]) -> str:
        """Per-arm ``max|q_cmd - q_meas|`` (rad, joints only): how far the last
        command runs ahead of the measured arm. A hardware arm that stops
        following shows a lag that stays put while the hand keeps moving."""
        parts = []
        for arm_id in self.session_arms:
            st = states.get(arm_id)
            q_cmd = self._last_cmd.get(arm_id)
            if st is None or q_cmd is None:
                continue
            n = min(7, len(st.q), len(q_cmd))
            parts.append(f"{arm_id}:{float(np.max(np.abs(q_cmd[:n] - st.q[:n]))):.4f}")
        return "{" + " ".join(parts) + "}" if parts else "n/a"

    def _servo_health(self) -> str:
        """Hardware servo-stream stats per arm (``XArmDriver.tick_stats()``, duck
        typed: the runtime never imports the hardware package). Empty for sim."""
        parts = []
        for arm_id in self.session_arms:
            try:
                arm = self.workcell.arms[arm_id]
            except (KeyError, TypeError):
                continue
            stats_fn = getattr(arm, "tick_stats", None)
            if not callable(stats_fn):
                continue
            try:
                st = stats_fn()
            except Exception:  # noqa: BLE001
                continue
            parts.append(
                f"{arm_id}: {st.ticks} ticks, {st.late_ticks} late, {st.faults} faults, "
                f"p99 {st.p99_s * 1e3:.1f} ms"
            )
        return f" servo={{{'; '.join(parts)}}}" if parts else ""

    # -- mode hooks (overridden by dagger.loop.GatedPolicyExecutor, phase-08) -------
    def _resolve_arms(
        self, states: dict[str, ArmState], held: frozenset[str], scale: float, now: float
    ) -> tuple[dict[str, np.ndarray | None], CommandSource]:
        """Per-arm action resolution; None = hold at last command."""
        out: dict[str, np.ndarray | None] = {}
        source = CommandSource.TELEOP
        for arm_id in self.session_arms:
            q_last = self._last_cmd[arm_id]
            q_next: np.ndarray | None = None
            if self.arm_stopped(arm_id, states[arm_id]):
                q_next = None  # FAULT / RECOVERING: hold; recovery re-seeds (§15)
            elif self.plans.active(arm_id):
                q_next = self._plan_step(arm_id, q_last)
                source = CommandSource.PLANNER
            elif self.jog.active(arm_id):
                q_next = self._jog_step(arm_id, q_last, scale)
                source = CommandSource.JOINT_JOG
                self._note_source(arm_id, CommandSource.JOINT_JOG)
            elif arm_id == self.active_arm:
                self._note_source(arm_id, CommandSource.TELEOP)
                q_next = self._teleop_step(arm_id, states[arm_id], q_last, held, scale, now)
            out[arm_id] = q_next
        return out, source

    def arm_stopped(self, arm_id: str, state: ArmState) -> bool:
        """This arm gets no new command this tick (04-runtime §15): a controller
        error is latched in its state, a driver FaultEvent stopped it, or it was
        re-seeded and waits for the operator to release every input (RECOVERING).
        Shared by every mode loop's ``_resolve_arms`` (teleop here, policy mux in
        ``dagger.loop``)."""
        return state.error_code != 0 or arm_id in self._faulted or arm_id in self._recovering

    # -- driver events: FAULT -> RECOVERING -> RUNNING (phase-09b; 04-runtime §15) ------
    @property
    def fault_state(self) -> str | None:
        """``"fault"`` while any arm is stopped by a driver fault, ``"recovering"``
        while any re-seeded arm waits for the re-grip, else ``None``."""
        if self._faulted:
            return "fault"
        if self._recovering:
            return "recovering"
        return None

    @property
    def faulted_arms(self) -> frozenset[str]:
        return frozenset(self._faulted)

    @property
    def recovering_arms(self) -> frozenset[str]:
        return frozenset(self._recovering)

    def _drain_driver_events(self, now: float) -> None:
        """Consume ``workcell.drain_events()`` (hardware drivers; ``[]`` for sim and
        fakes) and apply the per-arm transitions. Dispatch is by class name
        (:data:`FAULT_EVENT_NAMES`): the runtime never imports the optional
        hardware package's event types."""
        try:
            events = self.workcell.drain_events()
        except Exception:  # noqa: BLE001 - a broken event channel must not stop the tick
            logger.exception("workcell.drain_events failed")
            return
        if not events:
            return
        reseeded: set[str] = set()  # one re-seed per arm per tick (Reseed + Recovered pair)
        for ev in events:
            kind = type(ev).__name__
            arm_id = getattr(ev, "arm_id", None)
            if kind not in FAULT_EVENT_NAMES:
                logger.debug("driver event %s for %s: %r", kind, arm_id, ev)
                continue
            if arm_id not in self.session_arms:
                logger.info("driver event %s for non-session arm %r ignored: %r", kind, arm_id, ev)
                continue
            if kind == "FaultEvent":
                self._on_fault_event(arm_id, ev, now)
            elif kind == "StudioConflictWarning":
                text = str(getattr(ev, "detail", "") or STUDIO_WARNING_DEFAULT)
                self._warnings[arm_id] = (f"warning: {text}", now + STUDIO_WARNING_LINGER_S)
                logger.warning(
                    "%s: %s (controller mode %s state %s)",
                    arm_id,
                    text,
                    getattr(ev, "mode", "?"),
                    getattr(ev, "state", "?"),
                )
            else:  # ReseedEvent / RecoveredEvent
                self._on_recovered_event(arm_id, ev, now, reseeded)

    def _on_fault_event(self, arm_id: str, ev: object, now: float) -> None:
        """The driver stopped ``arm_id`` (controller error, latch, link loss, or the
        start of an operator-requested recovery): pause its sender, hold it, drop
        its plan / jog / teleop seed, release the clutch anchors if it was the
        clutched arm. Siblings are untouched (§15 "other arms hold")."""
        source = str(getattr(ev, "source", "") or "")
        code = int(getattr(ev, "error_code", 0) or 0)
        detail = str(getattr(ev, "detail", "") or "")
        title = controller_error_title(code) if code else ""
        if title and detail:
            text = f"{title} - {detail}"
        elif title:
            text = title
        elif detail:
            text = detail
        elif source == "user":
            text = "operator-requested recovery (re-seed from the measured position)"
        else:
            text = f"driver fault (source {source or '?'}, code {getattr(ev, 'code', '?')})"
        self._fault_text[arm_id] = text
        self._faulted.add(arm_id)
        self._recovering.discard(arm_id)
        self._warnings.pop(arm_id, None)
        self.fault_events += 1
        sender = self._senders.get(arm_id)
        if sender is not None:
            sender.pause()
        if self._plan_interruptible and self.plans.active_arms:
            self._cancel_plans("driver fault")  # a return-to-start stops on EVERY arm
        elif self.plans.active(arm_id) or self._plan_state.get(arm_id) is not None:
            self.plans.cancel(arm_id)
            self._plan_state.pop(arm_id, None)
            self._plan_clear_at.pop(arm_id, None)
            self.plan_cancel_reason = "driver fault"
            if not self.plans.active_arms:
                self._set_plan_status("cancelled", linger=True)
        self.jog.clear(arm_id)
        self._teleop_seeded.discard(arm_id)
        if self.tracker is not None and arm_id == self.active_arm:
            self.tracker.release()  # anchors are meaningless after the arm stopped
            self._clutch_arm = None
        logger.warning("%s: FAULT - %s", arm_id, text)

    def _on_recovered_event(self, arm_id: str, ev: object, now: float, reseeded: set[str]) -> None:
        """ReseedEvent / RecoveredEvent: the driver's streamer holds the MEASURED
        position again -> re-seed our targets from it (once per tick), resume the
        sender, and keep the arm held (RECOVERING) until every live input is
        released so the next clutch press is a true rising edge (zero delta)."""
        if arm_id not in reseeded:
            try:
                self.reseed_arm(arm_id)
            except Exception:  # noqa: BLE001 - keep the arm held rather than crash the tick
                logger.exception("%s: re-seed after recovery failed", arm_id)
                return
            reseeded.add(arm_id)
            self.recoveries += 1
        self._faulted.discard(arm_id)
        self._recovering.add(arm_id)
        sender = self._senders.get(arm_id)
        if sender is not None:
            sender.resume()
        code = int(getattr(ev, "error_code", 0) or 0)
        if type(ev).__name__ == "RecoveredEvent":
            logger.info(
                "%s: recovered%s; waiting for the operator to release every input",
                arm_id,
                f" from {controller_error_title(code)}" if code else "",
            )

    def _update_recovering(self) -> None:
        """RECOVERING -> RUNNING once no code is held by a LIVE source: the device
        clutch must be released (its next press is a rising edge that re-anchors
        at the measured TCP) and WS codes must be up or latched by the watchdog's
        AWAIT_EMPTY, which itself only clears on a fresh EMPTY KeysMsg (§8)."""
        if not self._recovering:
            return
        if any(self.sources.scale_for(c) > 0.0 for c in self.sources.held):
            return
        for arm_id in sorted(self._recovering):
            self._recovering.discard(arm_id)
            self._fault_text.pop(arm_id, None)
            logger.info("%s: RUNNING again (inputs released after recovery)", arm_id)

    def _publish_fault_state(self) -> None:
        cb = self.on_fault_state
        if cb is None:
            return
        state = self.fault_state
        if state == self._fault_state_reported:
            return
        self._fault_state_reported = state
        try:
            cb(state)
        except Exception:  # noqa: BLE001 - a manager bug must not stop the tick
            logger.exception("on_fault_state(%r) failed", state)

    def _arm_fault_details(self, now: float) -> dict[str, str]:
        """Telemetry ``fault_detail`` per arm: the fault text while FAULT /
        RECOVERING, else a not-yet-expired StudioConflictWarning."""
        out = dict(self._fault_text)
        for arm_id, (text, until) in list(self._warnings.items()):
            if now >= until:
                self._warnings.pop(arm_id, None)
            elif arm_id not in out:
                out[arm_id] = text
        return out

    def _note_source(self, arm_id: str, source: CommandSource) -> None:
        """Record the source resolving ``arm_id`` this tick (13-tracker §4
        "Anchor and re-seed rules" (a)): any non-teleop source (plan, jog,
        policy) invalidates the arm's teleop seed, so the next teleop tick
        re-seeds the integrator from the measured TCP and the first clutched
        tick after such motion has zero delta."""
        self._arm_source[arm_id] = source
        if source is not CommandSource.TELEOP:
            self._teleop_seeded.discard(arm_id)

    def _post_filter(self, dec, now: float) -> None:
        """After the safety gate; dagger uses this for block-streak anomaly."""

    def _plan_gate_watch(self, dec, now: float) -> None:
        """Gate-held abort (2026-09-08 evening; ``plan_gate_hold_s``). Runs on the
        control thread while a plan executes, O(active arms) per tick, no allocation
        beyond the cancel itself: the plan is HELD this tick when the gate blocked and
        every executing arm's gated output equals its last command (no waypoint
        progress - ``_last_cmd`` is still the previous tick's output here). The first
        held tick starts the clock, any progressing tick resets it, and once the hold
        has lasted ``plan_gate_hold_s`` the plan is cancelled with the blocking pair
        and distance in ``plan_cancel_reason`` (``GATE_HOLD_PREFIX``) so the manager
        reports it immediately instead of after the return budget. The arms hold where
        they are, exactly as after any other cancel."""
        if not dec.blocked:
            self._plan_gate_hold_since = None
            return
        for arm_id in self.plans.active_arms:
            q_out = dec.q_out.get(arm_id)
            q_last = self._last_cmd.get(arm_id)
            if q_out is None or q_last is None or not np.array_equal(q_out, q_last):
                self._plan_gate_hold_since = None  # this arm still moved: not held
                return
        if self._plan_gate_hold_since is None:
            self._plan_gate_hold_since = now
        held_for = now - self._plan_gate_hold_since
        if held_for < self.plan_gate_hold_s:
            return
        reason = f"{GATE_HOLD_PREFIX}: {self._gate_hold_pairs(dec)}"
        logger.warning(
            "plan held by the safety gate for %.1f s (limit %.1f s): cancelling - %s",
            held_for,
            self.plan_gate_hold_s,
            reason,
        )
        self._cancel_plans(reason)

    def _gate_hold_pairs(self, dec) -> str:
        """``"<body a> / <body b>[, <pair 2>] at <mm> mm"`` for the gate-held abort, from
        this tick's gate report (the merged supervisor report, then the gate's own
        block pairs as fallbacks: the twin's report may say nothing while the gate holds
        inside its hysteresis band or on a stale twin)."""
        report = dec.report
        if not report.pairs:
            merged = self.supervisor.merged_report()
            if merged.pairs:
                report = merged
        pairs = list(report.pairs[:2])
        if pairs:
            names = ", ".join(" / ".join(p) for p in pairs)
            return f"{names} at {report.min_clearance_m * 1e3:.1f} mm"
        fallback = getattr(self.supervisor.gate, "_block_pairs", None)
        if fallback:
            # A hold inside the gate's hysteresis band: the reports carry no pair and
            # their ``min_clearance_m`` is the 1.0 m default (2026-09-08 review: the
            # reason once read "... at 1000.0 mm"). Ask the twin for the real distance
            # at the measured posture; without one, name the pair alone.
            pairs = sorted(fallback)[:2]
            names = ", ".join(" / ".join(p) for p in pairs)
            twin = getattr(self.supervisor.gate, "twin", None)
            if twin is not None and hasattr(twin, "pair_distance"):
                try:
                    dist = min(float(twin.pair_distance(p, None, 0.5)) for p in pairs)
                except Exception:  # noqa: BLE001 - the reason is best-effort text
                    return names
                return f"{names} at {dist * 1e3:.1f} mm"
            return names
        kinds = {ev.kind for ev in report.violations}
        return "stale digital twin" if "stale_twin" in kinds else "no pair reported"

    def _session_extra(self, now: float) -> dict:
        """Extra session_extra entries (dagger/inference telemetry + frames)."""
        return {}

    # -- per-source steps -----------------------------------------------------
    def _teleop_step(
        self,
        arm_id: str,
        state: ArmState,
        q_last: np.ndarray,
        held: frozenset[str],
        scale: float,
        now: float | None = None,
    ) -> np.ndarray | None:
        """Held keys -> twist -> integrate -> IK -> q (04-runtime §6).

        Translate keys act in ``cfg.translate_frame`` (default the operator-fixed
        WORLD frame since 2026-09-08 evening; ``control/teleop.py`` has the
        geometry and the two alternatives, ``camera`` and ``base``) and rotate
        keys about the TCP axes.

        ``held`` is the merged set (WS ∪ device); ``scale`` is the WS watchdog
        scale, which governs the keyboard translate/rotate keys (only WS codes
        carry those). Rail codes are integrated per source at that source's
        scale (``_rail_rate``, 13-tracker §1.1), so a device-held rail code
        moves the rail without a browser and through a WS deadman latch. While
        ``tracker_clutch`` is held by ANY source the tracker supplies the
        target instead (13-tracker §4) at the scale of the source holding the
        clutch (``HeldSources.scale_for``): keyboard translate/rotate keys are
        ignored, rail (here) and gripper (``_gripper_step``) keys keep working.
        A rail-only tick holds the joint posture (``q[:7] = q_last``, so the
        TCP rides the rail; 04-runtime §6 "Rail") and integrates only the rail
        slot; it never seeds the teleop integrator and it INVALIDATES an
        existing seed, because the base slides under the frozen world-frame
        target (13-tracker §4 re-seed rule (d)): the next translate or clutch
        tick re-seeds from the measured TCP instead of stepping up to a leash
        toward the stale target. A rail input held TOGETHER with translate
        keys or a live clutch slides the whole arm too: the integrator target
        and the tracker anchors ride along by the base displacement of this
        tick's rail step (``_ride_rail``) and the IK is seeded with the new
        rail value, so the joints keep tracking the hand instead of folding to
        hold the target in place (2026-09-03; ``control.rail_in_ik`` false =
        the IK never moves the rail on its own either).
        """
        if self.ik is None or self.kin is None:
            return None
        clutch_scale = (
            self.sources.scale_for(TRACKER_CLUTCH_CODE) if self.tracker is not None else 0.0
        )
        has_rail = state.q.shape[0] > 7
        rail_v = self._rail_rate(self.sources) if has_rail else 0.0
        rail_moving = rail_v != 0.0
        if scale <= 0.0 and clutch_scale <= 0.0 and not rail_moving:
            return None  # every live source is latched/stale: hold (anchors clear in end_tick)
        measured_tcp = self.kin.tcp_world(arm_id, state.q)
        rail_new: float | None = None  # this tick's rail slot when a rail code moves it
        d_rail = np.zeros(3)  # base (= TCP at held posture) displacement of that step
        if rail_moving:
            rail_new = min(max(float(q_last[7]) + rail_v * self.dt, 0.0), RAIL_TRAVEL_M)
            q_rail = np.array(q_last, dtype=np.float64)
            q_rail[7] = rail_new
            d_rail = (
                self.kin.tcp_world(arm_id, q_rail).position
                - self.kin.tcp_world(arm_id, q_last).position
            )

        def with_rail(q: np.ndarray) -> np.ndarray:
            q = np.array(q, dtype=np.float64)
            if rail_new is not None:
                q[7] = rail_new
            return q

        if self.tracker is not None and TRACKER_CLUTCH_CODE in held:
            if clutch_scale > 0.0:
                if self._clutch_arm != arm_id:
                    # True clutch PRESS edge (clutch was up last tick; not a brief
                    # stale-sample gap, which keeps the session — 13-tracker §4
                    # "zero delta on engage"): re-anchor to where the arm ACTUALLY
                    # is. The commanded state (_last_cmd / integrator target / IK
                    # warm state) tracks the *commanded* pose; if the arm drifted,
                    # faulted+recovered or was nudged while the clutch was up, that
                    # pose is stale and the engage tick would command a leash-sized
                    # step toward it (the "first-clutch flail"). Snap all three to
                    # measured so A_ee == measured TCP and the first clutched tick
                    # is a no-op. q_last MUST move too, else _solve_target re-seeds
                    # IK from the stale commanded q (ik.solve reseed_threshold) and
                    # undoes this.
                    self._clutch_arm = arm_id
                    q_last = np.array(state.q, dtype=np.float64)
                    self._last_cmd[arm_id] = q_last
                    self._teleop_seeded.discard(arm_id)
                    if self.ik is not None:
                        self.ik.reset(arm_id, state.q)
                self._seed_teleop(arm_id, measured_tcp)
                self._ride_rail(arm_id, d_rail)
                target = self._tracker_target(arm_id, measured_tcp, clutch_scale, now)
            else:
                target = None  # clutch held only by a latched source: hold
            if target is None and not rail_moving:
                return None  # no fresh valid sample: hold-last
        else:
            tw = held_to_twist(held, self.cfg.teleop)  # translate/rotate: WS codes only
            v = tw.v * scale
            w = tw.w * scale
            if np.any(v) or np.any(w):
                self._seed_teleop(arm_id, measured_tcp)
                self._ride_rail(arm_id, d_rail)
                tw_world = twist_to_control_frame(
                    Twist(v=v, w=w),
                    self.kin.base_quat_world(arm_id),
                    measured_tcp.orientation,
                    self.cfg.translate_frame,
                    self._wrist_cam_quat(arm_id, measured_tcp.orientation),
                )
                self._key_twist[arm_id] = tw_world
                target = self.integrator.step(arm_id, tw_world, self.dt, measured_tcp)
            elif rail_moving:
                target = None  # rail-only: joints hold, the rail slot integrates below
            else:
                return None  # nothing held: non-active-style hold (no re-servo)
        if target is None:
            q = np.array(q_last)  # rail-only / tracker unavailable: rail codes still integrate
            # The rail carries the base while the world-frame target is frozen:
            # drop the seed so the next driven tick re-seeds from the measured
            # TCP (otherwise integrator.step / the clutch engage would command a
            # leash-sized step toward the stale target).
            self._teleop_seeded.discard(arm_id)
        else:
            # IK seeded with this tick's rail value: a locked-rail solver adopts
            # it and solves the joints at the new base position.
            q = self._solve_target(arm_id, target, with_rail(q_last), measured_tcp)
            if q is None:
                return None
        if rail_new is not None:
            q[7] = rail_new  # rail codes own the rail slot (ignored w/o rail)
        return q

    def _wrist_cam_quat(self, arm_id: str, tcp_quat: np.ndarray) -> np.ndarray | None:
        """The arm's wrist-camera world orientation for the camera translate frame,
        or None (no camera on this arm / a test fake without the method — the
        ``kin`` seam only ever promised ``tcp_world`` + ``base_quat_world``)."""
        getter = getattr(self.kin, "wrist_cam_quat_world", None)
        return None if getter is None else getter(arm_id, tcp_quat)

    def _ride_rail(self, arm_id: str, d_rail: np.ndarray) -> None:
        """A rail step under a driven tick: slide the world-frame integrator
        target and the tracker anchors by the base displacement so the TCP
        rides the rail while the joints keep tracking the hand / keys
        (04-runtime §6 "Rail"). No-op without a rail step."""
        if not np.any(d_rail):
            return
        prev = self.integrator.get(arm_id)
        if prev is not None:
            self.integrator.reanchor(arm_id, Pose(prev.position + d_rail, prev.orientation))
        if self.tracker is not None:
            self.tracker.translate_anchor(d_rail)

    def _rail_rate(self, sources: HeldSources) -> float:
        """Rail rate (m/s) from the held rail codes, per source at that source's
        scale (mirrors ``_gripper_step``): a device-held rail code moves the
        rail while the WS deadman is latched or no browser is connected, and
        stops within ``stale_s`` when the controller stream dies. The magnitude
        is clamped to ``teleop.rail_mps`` so two sources holding the same
        direction never exceed the configured speed."""
        rail_v = 0.0
        for code in sources.held:
            rv = held_to_twist(frozenset({code}), self.cfg.teleop).rail_v
            if rv != 0.0:
                rail_v += rv * sources.scale_for(code)
        v_max = abs(self.cfg.teleop.rail_mps)
        return min(max(rail_v, -v_max), v_max)

    def _seed_teleop(self, arm_id: str, measured_tcp: Pose) -> None:
        """(Re-)seed the integrated target from the measured TCP the moment
        teleop motion input starts (keys / clutch), not on idle hold ticks: the
        seed is invalidated by ``_note_source`` after plan/jog/policy motion,
        by a rail-only tick (``_teleop_step``, the base slid under the target),
        an arm switch and recovery, so the first driven tick has zero delta."""
        if arm_id not in self._teleop_seeded:
            self.integrator.seed(arm_id, measured_tcp)
            self._teleop_seeded.add(arm_id)

    def _solve_target(
        self, arm_id: str, target: Pose, q_last: np.ndarray, measured_tcp: Pose
    ) -> np.ndarray | None:
        """IK + residual handling shared by the keyboard and tracker paths."""
        result = self.ik.solve(arm_id, target, q_last)
        if result.diverged or not np.isfinite(result.pos_err_m):
            self.ik_diverged += 1
            logger.debug(
                "%s: IK diverged (pos_err %.4f m) - target re-anchored to measured, tick held",
                arm_id,
                result.pos_err_m,
            )
            self.integrator.reanchor(arm_id, measured_tcp)
            if self.tracker is not None:
                self.tracker.slip(target, measured_tcp)
            return None
        over_pos = result.pos_err_m > self.cfg.residual_max_pos_m
        over_rot = result.rot_err_rad > self.cfg.residual_max_rot_rad
        if over_pos or over_rot:
            self.ik_slips += 1
            logger.debug(
                "%s: IK residual over threshold (pos %.4f m, rot %.3f rad) - target frozen "
                "back to the achieved pose%s",
                arm_id,
                result.pos_err_m,
                result.rot_err_rad,
                " (position)"
                if over_pos and not over_rot
                else " (rotation)"
                if over_rot and not over_pos
                else " (both)",
            )
            # Freeze the target back to the achieved pose (glide, don't wind up)
            # COMPONENT-WISE (04-runtime §6): only the component whose residual
            # is over threshold is re-anchored, so a rotation residual never
            # moves the position anchor (the QP trades position for orientation
            # under velocity saturation; slipping both leaked that transient
            # into permanent TCP drift). The tracker anchor slips the same way.
            achieved = self.kin.tcp_world(arm_id, result.q)
            frozen = Pose(
                achieved.position if over_pos else target.position,
                achieved.orientation if over_rot else target.orientation,
            )
            self.integrator.reanchor(arm_id, frozen)
            if self.tracker is not None:
                self.tracker.slip(target, frozen)
        return np.array(result.q)

    def _tracker_target(
        self, arm_id: str, measured_tcp: Pose, scale: float, now: float | None
    ) -> Pose | None:
        """Clutched tracker target: the provider engages/anchors and clamps to
        the leash; the watchdog ``scale`` shrinks the per-tick step toward it
        (not the hand<->arm offset); then the step from the previous commanded
        target is rate-limited to ``target_rate`` (04-runtime §6) so the IK
        never runs into joint-velocity saturation. The rate-limit truncation is
        NOT slipped into the anchor: the target catches up inside the leash,
        and a ``tracker_settings`` change mid catch-up re-anchors to the
        provider's leash-clamped target (rule (c)), not to the rate-limited
        pose handed to IK here (``set_target`` feeds telemetry only)."""
        now = self._clock() if now is None else now
        prev = self.integrator.get(arm_id) or measured_tcp
        clamped = self.tracker.target(arm_id, measured_tcp, prev, now)
        if clamped is None:
            return None
        rate = self.cfg.target_rate
        target = se3.clamp_pose_to_leash(
            interp_pose(prev, clamped, scale), prev, rate.v_mps * self.dt, rate.w_radps * self.dt
        )
        self.integrator.reanchor(arm_id, target)
        self.tracker.set_target(target)
        return target

    def _jog_step(self, arm_id: str, q_last: np.ndarray, scale: float) -> np.ndarray | None:
        if scale <= 0.0:
            return None  # deadman: jog holds too (11-safety §10.1)
        q_next = self.jog.step(arm_id, q_last)
        if q_next is None:
            return None
        return q_last + (q_next - q_last) * scale

    def _device_inputs(self, now: float) -> tuple[frozenset[str], float]:
        """Device-held codes + scale from the newest tracker sample (13-tracker
        §1.1). The controller's own sample stream is their heartbeat: fresh
        (``age <= stale_s``) => the sample's codes at scale 1.0; stale or no
        sample/provider => no codes, scale 0.0. Not covered by the WS watchdog."""
        if self.tracker is None:
            return frozenset(), 0.0
        got = self.tracker.slot.get()
        if got is None:
            return frozenset(), 0.0
        sample = got[0]
        age = now - sample.rx_mono
        fresh = age <= self.tracker.stale_s
        if fresh != self._device_fresh_logged:
            if self._device_fresh_logged is not None or not fresh:  # first fresh: silent
                if fresh:
                    logger.info("controller stream fresh again (sample age %.3f s)", age)
                else:
                    logger.warning(
                        "controller stream STALE (sample age %.3f s > stale_s %.2f): device-held "
                        "codes dropped - clutch, gripper and rail from the controller stop",
                        age,
                        self.tracker.stale_s,
                    )
            self._device_fresh_logged = fresh
        self._device_click_edge(sample, fresh, now)
        if not fresh:
            return frozenset(), 0.0
        return sample.held_codes, 1.0

    def _device_click_edge(self, sample, fresh: bool, now: float) -> None:
        """Fire the discrete device actions bound to controller press edges
        (trackpad click / menu / grip, 13-tracker §1.1): ``controller.edge_seq``
        advanced since the last tick => run every ``sample.click_actions`` entry
        newer than the last seen counter (``switch_arm`` / ``switch_arm_prev``,
        derived per remembered edge, oldest first — two buttons edging inside
        one tick both fire) through the same handler as the WS action (same
        nacks). The first observed counter is adopted silently; edges on a
        stale sample are dropped."""
        ctl = sample.controller
        if ctl is None:
            self._edge_seq_seen = None
            return
        seen, self._edge_seq_seen = self._edge_seq_seen, ctl.edge_seq
        if seen is None or ctl.edge_seq == seen or not fresh:
            return
        for seq, action in sample.click_actions:
            if seq > seen:
                self._fire_device_action(action, now)

    def _fire_device_action(self, name: str, now: float) -> CommandResult:
        res = self._handle_command(Command(op=name, source="internal"))
        label = name if res.ok else f"{name} nacked: {res.detail}"
        self._device_action = (label, now + DEVICE_ACTION_LINGER_S)
        logger.info("device action %s -> %s", name, "ok" if res.ok else f"nack ({res.detail})")
        return res

    def _device_action_label(self, now: float) -> str | None:
        """Last device-sourced discrete action, shown for ``DEVICE_ACTION_LINGER_S``."""
        if self._device_action is None:
            return None
        label, until = self._device_action
        if now >= until:
            self._device_action = None
            return None
        return label

    def _gripper_step(self, sources: HeldSources) -> None:
        """F/H integrate ``open_frac`` per source at that source's scale: a
        device-held gripper code keeps working while the WS deadman is latched
        and stops within ``stale_s`` when the controller stream dies. A FAULTED /
        RECOVERING arm is not commanded at all (§15: nothing integrates, nothing
        is queued for the sender to replay once it is streaming again)."""
        arm_id = self.active_arm
        if arm_id is None or arm_id not in self.gripper_arms:
            return  # gripper keys are ignored on a camera-only arm
        if arm_id in self._faulted or arm_id in self._recovering:
            return  # stopped by a driver fault: the gripper holds too
        grip_v = 0.0
        for code in sources.held:
            gv = held_to_twist(frozenset({code}), self.cfg.teleop).grip_v
            if gv != 0.0:
                grip_v += gv * sources.scale_for(code)
        if grip_v == 0.0:
            return
        frac = self._grip_frac[arm_id] + grip_v * self.dt
        self._grip_frac[arm_id] = min(max(frac, 0.0), 1.0)
        if self.tick_count % GRIPPER_SEND_EVERY_N_TICKS == 0:
            sender = self._senders.get(arm_id)
            if sender is not None:
                sender.put_gripper(self._grip_frac[arm_id])

    # -- plan lifecycle ------------------------------------------------------------
    def _set_plan_status(self, status: str | None, linger: bool = False) -> None:
        self._plan_status = status
        self._plan_status_clear_at = self.tick_count + PLAN_STATUS_LINGER_TICKS if linger else None

    def _plan_step(self, arm_id: str, q_last: np.ndarray) -> np.ndarray | None:
        """One executor step for a plan arm (shared by every mode loop's
        ``_resolve_arms``). When the executor hands back the goal itself (its final
        waypoint) the arm is NOT finished yet: the goal still has to pass the gate this
        tick, so it is parked in ``_plan_arriving`` for :meth:`_confirm_plan_arrivals`.
        Before 2026-09-08 the plan was reported ``done`` here, BEFORE the gate, so a
        final step the gate held left ``_last_cmd`` one slew step short of the goal with
        ``plan_status done`` and ``plan_cancel_reason None``, the deferred gripper was
        applied on an arm that had not arrived and the gate-held watch never ran."""
        q_next = self.plans.step(arm_id, q_last)
        self._note_source(arm_id, CommandSource.PLANNER)
        if q_next is not None and not self.plans.active(arm_id):  # the executor's goal
            self._plan_arriving[arm_id] = np.array(q_next, dtype=np.float64)
        return q_next

    def _confirm_plan_arrivals(self, dec) -> None:
        """After the gate: an arm whose executor returned the goal this tick is done
        only if the gated output IS that goal; otherwise (the gate or the per-tick clamp
        held the last step) the goal goes back into the executor as a one-waypoint plan,
        so the arm stays ``executing``, ``plans.active_arms`` stays non-empty for the
        manager's wait and ``_plan_gate_watch`` sees the hold from the next tick on.

        Exception (2026-09-09): a goal the COMMAND PATH can never reach exactly. The
        per-tick clamp (``_cap_joint_step``) pins the rail slot to ``[0, RAIL_TRAVEL]``,
        so a rail goal a hair outside it - copied verbatim from a measured posture that
        settled a few tenths of a mm past the end stop - is clamped away every tick while
        the gate stays clear. Re-loading it then never finishes: the executor keeps
        returning the goal, the clamp keeps holding the command short, ``q_out != goal``
        forever, and because nothing is BLOCKED ``_plan_gate_watch`` never aborts it - the
        plan hangs ``executing`` and blocks every later reset / Go-to-profile / R. So when
        the gate is not blocking AND the command made no progress toward the goal this
        tick AND the only difference left is the rail slot pinned at the travel end the
        goal lies beyond (:meth:`_clamped_rail_goal`), the arm is at the reachable limit:
        finish it. A genuine gate hold (``dec.blocked``) still re-loads and is aborted by
        the watch if it never clears; a clamp that is merely SPLITTING a long final step
        still progresses and re-loads; a joint goal the command path cannot reach (a
        non-positive ``dq_max`` fail-safe hold) stays ``executing`` for the manager's
        budget to report - it is not an arrival (2026-09-09 review). Pinned by
        ``tests/test_loop_units.py::test_a_rail_goal_a_hair_outside_the_travel_...`` and
        mirrored by the fuzz's headless replay (``tests/test_return_fuzz_mavis_v2.py``)."""
        if not self._plan_arriving:
            return
        for arm_id, goal in list(self._plan_arriving.items()):
            self._plan_arriving.pop(arm_id, None)
            if self._plan_state.get(arm_id) != "executing":
                continue  # cancelled between the step and the gate (a movement key etc.)
            q_out = dec.q_out.get(arm_id)
            if q_out is None:
                self.plans.load(arm_id, [goal])
                continue
            if np.allclose(q_out, goal, atol=1e-9):
                self._finish_plan(arm_id, ok=True)
                continue
            q_last = self._last_cmd.get(arm_id)  # not yet updated for this tick
            progressed = q_last is None or not np.allclose(q_out, q_last, atol=1e-9)
            if not dec.blocked and not progressed and self._clamped_rail_goal(q_out, goal):
                self._finish_plan(arm_id, ok=True)  # at the reachable limit (clamped goal)
            else:
                self.plans.load(arm_id, [goal])

    @staticmethod
    def _clamped_rail_goal(q_out: np.ndarray, goal: np.ndarray) -> bool:
        """True when ``q_out`` differs from ``goal`` ONLY in the rail slot, which sits at
        the travel end (``0`` / ``RAIL_TRAVEL_M``) the goal lies beyond - the one goal the
        per-tick clamp makes unreachable by construction (:meth:`_confirm_plan_arrivals`).
        Shared with the fuzz's headless replay so both judge arrival by the same rule."""
        q_out = np.asarray(q_out, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)
        if q_out.shape[0] <= 7 or goal.shape[0] <= 7:
            return False
        if not np.allclose(q_out[:7], goal[:7], atol=1e-9):
            return False
        rail, want = float(q_out[7]), float(goal[7])
        return (abs(rail) <= 1e-9 and want < 0.0) or (
            abs(rail - RAIL_TRAVEL_M) <= 1e-9 and want > RAIL_TRAVEL_M
        )

    def _finish_plan(self, arm_id: str, ok: bool) -> None:
        if ok:
            self._plan_state.pop(arm_id, None)
            self._plan_clear_at.pop(arm_id, None)
            frac = self._plan_gripper_on_arrival.pop(arm_id, None)
            if frac is not None:  # deferred gripper target of the arriving arm
                self._apply_gripper_target(arm_id, frac)
            if not self.plans.active_arms:
                # the plan is done: deferred targets for arms this plan did not move
                # (the manager rides them on the LAST arm of a sequence) apply now
                for other, frac in list(self._plan_gripper_on_arrival.items()):
                    self._apply_gripper_target(other, frac)
                self._plan_gripper_on_arrival.clear()
                self._set_plan_status("done", linger=True)
                self._plan_interruptible = False
                self._plan_gate_hold_since = None
        else:
            self._plan_state[arm_id] = "failed"
            self._plan_clear_at[arm_id] = self.tick_count + PLAN_STATUS_LINGER_TICKS
            self._set_plan_status("failed", linger=True)

    def _apply_gripper_target(self, arm_id: str, frac: float) -> None:
        """Set + send one gripper target (ignored for a gripperless arm)."""
        if arm_id not in self.gripper_arms:
            return
        self._grip_frac[arm_id] = min(max(float(frac), 0.0), 1.0)
        sender = self._senders.get(arm_id)
        if sender is not None:
            sender.put_gripper(self._grip_frac[arm_id])

    def _cancel_plans(self, reason: str) -> None:
        for arm_id in self.plans.active_arms:
            self._plan_state.pop(arm_id, None)
        self.plans.cancel()
        self._set_plan_status("cancelled", linger=True)
        self.plan_cancel_reason = reason
        self._plan_interruptible = False
        self._plan_gripper_on_arrival.clear()  # a cancelled return leaves the gripper alone
        self._plan_arriving.clear()
        self._plan_gate_hold_since = None
        logger.info("plan cancelled: %s", reason)

    def _op_cancel_plan(self, cmd: Command) -> CommandResult:
        """Internal (SessionManager): cancel every running plan with ``args.reason``
        (return-to-start deadline / teardown) — the arms hold where they are."""
        reason = str(cmd.args.get("reason") or "cancelled")
        if self.plans.active_arms:
            self._cancel_plans(reason)
            return CommandResult(cmd.corr_id, True, "cancelled")
        return CommandResult(cmd.corr_id, True, "no plan")

    def _interrupt_plan_for(self, reason: str) -> None:
        """Operator input during an INTERRUPTIBLE plan (the return-to-start motion,
        04-runtime §10.5) cancels it: the arm holds where it is and the input wins."""
        if self._plan_interruptible and self.plans.active_arms:
            self._cancel_plans(reason)

    def _expire_plan_status(self) -> None:
        if self._plan_status_clear_at is not None and self.tick_count >= self._plan_status_clear_at:
            self._plan_status = None
            self._plan_status_clear_at = None
        for arm_id, at in list(self._plan_clear_at.items()):
            if self.tick_count >= at:
                self._plan_clear_at.pop(arm_id, None)
                if self._plan_state.get(arm_id) == "failed":
                    self._plan_state.pop(arm_id, None)

    # -- command handling (drained at tick boundaries) ------------------------------
    def _handle_command(self, cmd: Command) -> CommandResult:
        handler = getattr(self, f"_op_{cmd.op}", None)
        if handler is None:
            return CommandResult(cmd.corr_id, False, f"unknown op {cmd.op!r}")
        return handler(cmd)

    def _activate_arm(self, cmd: Command, arm_id: str) -> CommandResult:
        self._interrupt_plan_for("arm switch")
        self.active_arm = arm_id
        self._teleop_seeded.discard(self.active_arm)  # reseed target from measured
        if self.tracker is not None:
            self.tracker.release()  # arm switch clears the tracker anchors
        return CommandResult(cmd.corr_id, True, self.active_arm)

    def _switch_arm(self, cmd: Command, step: int) -> CommandResult:
        if not self.session_arms:
            return CommandResult(cmd.corr_id, False, "no session arms")
        i = self.session_arms.index(self.active_arm) if self.active_arm else -step
        return self._activate_arm(cmd, self.session_arms[(i + step) % len(self.session_arms)])

    def _op_switch_arm(self, cmd: Command) -> CommandResult:
        """Server-authoritative Tab / RB cycling; previous arm's target freezes.

        With ``args.arm_id`` (the Cockpit's clickable arm rows, 2026-09-07) the
        switch is EXPLICIT instead: the named arm becomes active whatever the
        session order is, an unknown id is refused, and re-selecting the arm
        that is already active is a no-op — it must NOT release the tracker
        anchors, or a click on the active row would drop a live clutch.
        """
        arm_id = cmd.args.get("arm_id")
        if arm_id is None:
            return self._switch_arm(cmd, +1)
        arm_id = str(arm_id)
        if arm_id not in self.session_arms:
            return CommandResult(cmd.corr_id, False, f"arm {arm_id!r} not in this session")
        if arm_id == self.active_arm:
            return CommandResult(cmd.corr_id, True, arm_id)
        return self._activate_arm(cmd, arm_id)

    def _op_switch_arm_prev(self, cmd: Command) -> CommandResult:
        """KeyZ / LB: previous arm, ``(i - 1) mod n`` (13-tracker §4)."""
        return self._switch_arm(cmd, -1)

    def _op_tracker_settings(self, cmd: Command) -> CommandResult:
        """Mutate the live yaw/scale/rotation/filter settings (echoed in telemetry);
        the provider re-anchors instead of moving the arm when engaged (13-tracker §4)."""
        if self.tracker is None:
            return CommandResult(cmd.corr_id, False, "no tracker provider in this session")
        try:
            args = TrackerSettingsArgs.model_validate(cmd.args)
        except Exception as e:
            return CommandResult(cmd.corr_id, False, f"invalid tracker_settings args: {e}")
        v = self.tracker.settings.update(
            yaw_deg=args.yaw_deg,
            pos_scale=args.pos_scale,
            follow_rotation=args.follow_rotation,
            filter_enabled=args.filter_enabled,
            filter_min_cutoff_hz=args.filter_min_cutoff_hz,
            filter_beta=args.filter_beta,
        )
        return CommandResult(
            cmd.corr_id,
            True,
            f"yaw_deg={v.yaw_deg:g} pos_scale={v.pos_scale:g} "
            f"follow_rotation={str(v.follow_rotation).lower()} "
            f"filter_enabled={str(v.filter_enabled).lower()} "
            f"filter_min_cutoff_hz={v.filter_min_cutoff_hz:g} filter_beta={v.filter_beta:g}",
        )

    def _op_takeover_toggle(self, cmd: Command) -> CommandResult:
        if self.plans.active_arms:
            self._cancel_plans("takeover_toggle")
        return CommandResult(cmd.corr_id, False, TAKEOVER_UNAVAILABLE)

    def _op_takeover(self, cmd: Command) -> CommandResult:
        """Explicit take-over (phase-14; 15-online-dagger D3): a gate op, served by the
        GatedPolicyExecutor of a dagger / inference session; here it nacks exactly like
        Space does (and, like Space, still cancels a running plan — it is an escape)."""
        if self.plans.active_arms:
            self._cancel_plans("takeover")
        return CommandResult(cmd.corr_id, False, TAKEOVER_UNAVAILABLE)

    def _op_handback(self, cmd: Command) -> CommandResult:
        """Explicit hand-back (phase-14; 15-online-dagger D3): no gate here, nack."""
        return CommandResult(cmd.corr_id, False, TAKEOVER_UNAVAILABLE)

    def _op_train_now(self, cmd: Command) -> CommandResult:
        """``train_now`` (phase-14; 15-online-dagger §3) is served by the
        GatedPolicyExecutor of an Online DAgger session only; every other session nacks."""
        return CommandResult(cmd.corr_id, False, NOT_ONLINE_DAGGER)

    def _op_gello_pause(self, cmd: Command) -> CommandResult:
        """``gello_pause`` (phase-15; 16-gello §8.2) is served by the ``GelloLoop`` of a
        GELLO Manipulation session only; every other session nacks with the mode reason
        (the ``train_now`` / ``NOT_ONLINE_DAGGER`` precedent)."""
        return CommandResult(cmd.corr_id, False, NOT_GELLO)

    def _op_gello_resume(self, cmd: Command) -> CommandResult:
        """``gello_resume`` (phase-15; 16-gello §8.2): see :meth:`_op_gello_pause`."""
        return CommandResult(cmd.corr_id, False, NOT_GELLO)

    def _episode_op(self, cmd: Command, op: str) -> CommandResult:
        """Episode ops validate/transition inline (fast); writer work runs on
        the RecorderThread. Invalid transitions ack ``ok=false`` (§10.4)."""
        if self.recorder is None:
            return CommandResult(cmd.corr_id, False, "no recorder in this mode")
        ok, detail = self.recorder.request(op)
        return CommandResult(cmd.corr_id, ok, detail)

    def _op_episode_new(self, cmd: Command) -> CommandResult:
        return self._episode_op(cmd, "new")

    def _op_episode_save(self, cmd: Command) -> CommandResult:
        return self._episode_op(cmd, "save")

    def _op_episode_discard(self, cmd: Command) -> CommandResult:
        return self._episode_op(cmd, "discard")

    def _op_joint_target(self, cmd: Command) -> CommandResult:
        if self.episode_state == "recording":
            return CommandResult(cmd.corr_id, False, "recording")
        try:
            args = JointTargetArgs.model_validate(cmd.args)
        except Exception as e:
            return CommandResult(cmd.corr_id, False, f"invalid joint_target args: {e}")
        arm_id = args.arm_id
        if arm_id not in self.session_arms:
            return CommandResult(cmd.corr_id, False, f"unknown arm {arm_id!r}")
        dof = self.workcell.arms[arm_id].dof
        if len(args.positions) != dof:
            return CommandResult(
                cmd.corr_id, False, f"positions must have length {dof} (full q incl. rail)"
            )
        self._interrupt_plan_for("jog")  # a return-to-start motion yields to the panel
        if self.plans.active(arm_id) or self._plan_state.get(arm_id) == "planning":
            return CommandResult(cmd.corr_id, False, "plan executing")
        target = np.asarray(args.positions, dtype=np.float64)
        if args.mode != "jog" and self._any_plan_in_flight():
            # One plan at a time on the whole loop (2026-09-08 review): a goto for THIS arm
            # while a non-interruptible plan (start_from) walks ANOTHER arm would load two
            # arms into the executor and move them simultaneously through combinations no
            # planner validated - the 23:16:52 incident class - and its own plan would be
            # made with the other arm frozen at a mid-path posture it is about to leave.
            return CommandResult(cmd.corr_id, False, "plan executing")
        if args.mode == "jog":
            # ANY size is accepted (the 0.15 rad `goto_threshold_rad` nack was dropped
            # 2026-09-07). A jog never teleports: `JogState.step` walks from the last
            # COMMANDED q toward the target at `slew_rad_per_tick`, latest-wins, and
            # every intermediate posture goes through the gate like any other command,
            # so a big delta is simply a longer constant-speed move. The threshold only
            # bought the planner's obstacle routing, which the operator does not want on
            # this panel; the price is that a straight joint-space line into an obstacle
            # is HELD by the gate instead of routed around it (`goto` still plans).
            self.jog.set_target(arm_id, target)
            return CommandResult(cmd.corr_id, True, "jog")
        # goto: plan on a worker thread; result returns via the bus.
        if self.planner is None:
            return CommandResult(cmd.corr_id, False, "no planner (twin unavailable)")
        self._plan_state[arm_id] = "planning"
        self._set_plan_status("planning")
        self._spawn_plan_worker({arm_id: [float(x) for x in target]})
        return CommandResult(cmd.corr_id, True, "accepted")

    def _spawn_plan_worker(self, q_goal: dict[str, list[float]]) -> None:
        from apollo_mavis_v2_core import PlanRequest

        if self.supervisor.twin is None and hasattr(self.planner, "sync"):
            # Plain sim (NullGate): nothing else keeps the plan twin's
            # measured context fresh; sync here on the loop thread.
            self.planner.sync(self._states)
        q_start = {a: [float(x) for x in self._states[a].q] for a in q_goal}
        req = PlanRequest(q_start=q_start, q_goal=q_goal, speed_scale=self.speed_scale)
        planner = self.planner

        def work() -> None:
            try:
                result = planner.plan(req)
            except Exception as e:  # planner bug: fail the plan, never the loop
                logger.exception("plan worker failed")
                result = None
                detail = repr(e)
            else:
                detail = ""
            self.bus.commands.submit(
                Command(
                    op="_plan_ready",
                    args={"arms": list(q_goal), "result": result, "detail": detail},
                    source="internal",
                )
            )

        threading.Thread(target=work, name="plan-worker", daemon=True).start()

    def _any_plan_in_flight(self) -> bool:
        """A plan is executing or being planned for ANY arm of this loop."""
        return bool(self.plans.active_arms) or any(
            s in ("planning", "executing") for s in self._plan_state.values()
        )

    def _op__plan_ready(self, cmd: Command) -> CommandResult:
        result = cmd.args.get("result")
        arms = cmd.args.get("arms", [])
        if result is None or not result.ok:
            for arm_id in arms:
                self._finish_plan(arm_id, ok=False)
            return CommandResult(cmd.corr_id, True, "plan failed")
        if self.plans.active_arms:
            # Defence in depth (2026-09-08 review): the goto is refused while a plan runs,
            # so a result can only land here if a plan was loaded AFTER the worker
            # started. Loading it would put two arms into the executor at once; drop it
            # (per-arm ``failed``, the running plan's session status untouched).
            for arm_id in arms:
                self._plan_state[arm_id] = "failed"
                self._plan_clear_at[arm_id] = self.tick_count + PLAN_STATUS_LINGER_TICKS
            logger.warning(
                "plan result for %s dropped: %s already executing", arms, self.plans.active_arms
            )
            return CommandResult(cmd.corr_id, False, "plan executing")
        for arm_id in arms:
            self.plans.load(arm_id, result.waypoints[arm_id])
            self._plan_state[arm_id] = "executing"
        self._set_plan_status("executing")
        return CommandResult(cmd.corr_id, True, "executing")

    def _op_execute_plan(self, cmd: Command) -> CommandResult:
        """Internal: SessionManager hands pre-planned waypoints (start_from §5.2; the
        return-to-start motion passes ``interruptible: True`` so a jog or an arm
        switch cancels it too — a movement key cancels every plan anyway).

        Since 2026-09-08 evening the manager submits a multi-arm plan ONE ARM AT A
        TIME in the planner's ``arm_order`` (that day a two-arm ``reset_to_initial``
        was submitted as one plan and both arms moved simultaneously through
        combinations the sequential planner never validated; the gate held them at
        5.2 mm). The loop is the last line: a command carrying MORE THAN ONE arm's
        waypoints is refused (``one arm per plan``) - the manager bug of that day can
        not recur through this op. An interruptible plan's gripper targets are ALL
        deferred to the plan's arrival - including a target for an arm this command
        does not move, which the manager rides on the last arm of the sequence. A
        command with NO waypoints (a profile that moves no arm but sets a gripper)
        applies its gripper targets at once, interruptible or not: there is no motion
        to arrive, and nothing an operator input could cancel."""
        waypoints: dict[str, list[list[float]]] = cmd.args["waypoints"]
        if len(waypoints) > 1:
            return CommandResult(
                cmd.corr_id,
                False,
                f"one arm per plan (sequential execution), got {sorted(waypoints)}",
            )
        for arm_id in waypoints:
            if arm_id not in self.session_arms:
                return CommandResult(cmd.corr_id, False, f"unknown arm {arm_id!r}")
            if arm_id in self._faulted or arm_id in self._recovering:
                return CommandResult(cmd.corr_id, False, f"arm {arm_id!r} is faulted")
        # One plan at a time (2026-09-08): ``PlanExecutor.load`` would silently replace
        # the running waypoints (and the bookkeeping below would reset the running
        # plan's interruptibility / deferred gripper), and the manager-side "plan
        # executing" blockers run BEFORE a worker plans - so the loop is the last line.
        if self.plans.active_arms or "planning" in self._plan_state.values():
            return CommandResult(cmd.corr_id, False, "plan executing")
        self.plan_cancel_reason = None
        self._plan_gate_hold_since = None
        interruptible = bool(cmd.args.get("interruptible", False))
        self._plan_interruptible = interruptible
        self._plan_gripper_on_arrival.clear()
        for arm_id, wps in waypoints.items():
            self.plans.load(arm_id, wps)
            self._plan_state[arm_id] = "executing"
        moving = bool(self.plans.active_arms)
        for arm_id, frac in cmd.args.get("gripper", {}).items():
            if arm_id not in self.gripper_arms:
                continue
            if interruptible and moving:
                # return-to-start: the profile's gripper is applied on ARRIVAL, so a
                # cancelled return leaves it where the episode ended (also for an arm
                # this command does not move - see the docstring)
                self._plan_gripper_on_arrival[arm_id] = float(frac)
                continue
            self._apply_gripper_target(arm_id, frac)
        if not moving:  # gripper-only: done in this tick
            self._plan_interruptible = False
            self._set_plan_status("done", linger=True)
            return CommandResult(cmd.corr_id, True, "done")
        self._set_plan_status("executing")
        return CommandResult(cmd.corr_id, True, "executing")

    def _op_reset_to_initial(self, cmd: Command) -> CommandResult:
        """``R`` (2026-09-08 operator request): walk the workcell back to the
        designated initial-condition profile for this workcell kind.

        Deliberately a NO-OP with a reason when there is nothing to return to —
        no profile store, or no profile designated as this kind's initial
        condition (``ack.ok == false``, so the UI shows the reason as a toast and
        nothing moves). Refused while an episode records (the motion would
        pollute it) and while another plan runs. The motion itself is planned by
        the SessionManager on the session twin — off the loop thread, gated like
        every other command, and INTERRUPTIBLE: any movement key, a clutch, a jog
        or an arm switch cancels it and the arms hold where they are."""
        if self.episode_state == "recording":
            return CommandResult(cmd.corr_id, False, "recording - save or discard first")
        if self.profile_store is None:
            return CommandResult(cmd.corr_id, False, "no profile store")
        try:
            profile = self.profile_store.initial_for(self.workcell_kind)  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001 - an unreadable store must not kill the loop
            return CommandResult(cmd.corr_id, False, f"profile store unreadable: {e}")
        if profile is None:
            return CommandResult(
                cmd.corr_id,
                False,
                f"no initial condition designated for the {self.workcell_kind} workcell - "
                "save a profile with 'use as initial condition' first",
            )
        if self.plans.active_arms or "planning" in self._plan_state.values():
            return CommandResult(cmd.corr_id, False, "plan executing")
        if self.on_reset_to_initial is None:
            return CommandResult(cmd.corr_id, False, "no session manager attached")
        ok, detail = self.on_reset_to_initial(profile)
        return CommandResult(cmd.corr_id, ok, detail)

    def _op_goto_profile(self, cmd: Command) -> CommandResult:
        """``goto_profile {profile_id}`` (2026-09-08, no key binding): walk the session
        arms to the CHOSEN saved profile — the ``reset_to_initial`` motion with a
        profile the operator picked instead of the designated initial condition.

        Inline validation only (the 100 Hz thread never plans): refused while an
        episode records, for an unknown id, for a profile of another workcell kind,
        for one that covers no arm of this session, and while another plan runs; then
        the SessionManager hook plans it on the session twin off the loop thread and
        runs it through the gated, INTERRUPTIBLE ``execute_plan`` path — operator
        input cancels it and the arms hold where they are. Operator-requested
        motion only; never implicit."""
        if self.episode_state == "recording":
            return CommandResult(cmd.corr_id, False, "recording - save or discard first")
        if self.profile_store is None:
            return CommandResult(cmd.corr_id, False, "no profile store")
        profile_id = str(cmd.args.get("profile_id") or "").strip()
        if not profile_id:
            return CommandResult(cmd.corr_id, False, "profile_id required")
        try:
            profile = self.profile_store.get(profile_id)
        except ProfileNotFoundError:
            return CommandResult(cmd.corr_id, False, f"unknown profile {profile_id!r}")
        except Exception as e:  # noqa: BLE001 - an unreadable store must not kill the loop
            return CommandResult(cmd.corr_id, False, f"profile store unreadable: {e}")
        if profile.workcell_kind != self.workcell_kind:
            return CommandResult(
                cmd.corr_id,
                False,
                f"profile '{profile.name}' is for the {profile.workcell_kind} workcell",
            )
        if not any(a in profile.arms for a in self.session_arms):
            return CommandResult(
                cmd.corr_id, False, f"profile '{profile.name}' covers no arm of this session"
            )
        if self.plans.active_arms or "planning" in self._plan_state.values():
            return CommandResult(cmd.corr_id, False, "plan executing")
        if self.on_goto_profile is None:
            return CommandResult(cmd.corr_id, False, "no session manager attached")
        ok, detail = self.on_goto_profile(profile)
        return CommandResult(cmd.corr_id, ok, detail)

    def _op_save_profile(self, cmd: Command) -> CommandResult:
        if self.profile_store is None:
            return CommandResult(cmd.corr_id, False, "no profile store")
        name = str(cmd.args.get("name", "")).strip()
        if not name:
            return CommandResult(cmd.corr_id, False, "profile name required")
        profile = save_from_states(
            self.profile_store,
            self._states,
            self.session_arms,
            self.workcell_kind,
            name,
            str(cmd.args.get("notes", "")),
        )
        # One Cockpit button since 2026-09-07 (05-ui §8.3): "save current state
        # as profile" carries the initial-condition designation as a switch, so
        # the two paths cannot drift apart into two differently-named snapshots.
        if bool(cmd.args.get("set_initial", False)):
            self.profile_store.set_initial(profile.profile_id)
        return CommandResult(cmd.corr_id, True, profile.profile_id)

    def _op_set_initial_condition(self, cmd: Command) -> CommandResult:
        if self.profile_store is None:
            return CommandResult(cmd.corr_id, False, "no profile store")
        profile_id = cmd.args.get("profile_id")
        if profile_id:
            try:
                self.profile_store.set_initial(str(profile_id))
            except ProfileNotFoundError:
                return CommandResult(cmd.corr_id, False, f"unknown profile {profile_id!r}")
            return CommandResult(cmd.corr_id, True, str(profile_id))
        profile = save_initial_overwrite(
            self.profile_store, self._states, self.session_arms, self.workcell_kind
        )
        return CommandResult(cmd.corr_id, True, profile.profile_id)


__all__ = [
    "ControlLoop",
    "DEFAULT_ACTIVE_ARM",
    "DEVICE_ACTION_LINGER_S",
    "FAULT_EVENT_NAMES",
    "GATE_HOLD_PREFIX",
    "GRIPPER_SEND_EVERY_N_TICKS",
    "HeldSources",
    "NOT_GELLO",
    "NOT_ONLINE_DAGGER",
    "PLAN_STATUS_LINGER_TICKS",
    "STUDIO_WARNING_DEFAULT",
    "STUDIO_WARNING_LINGER_S",
    "TAKEOVER_UNAVAILABLE",
    "controller_error_title",
    "default_active_arm",
]
