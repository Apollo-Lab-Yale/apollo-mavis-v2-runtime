"""ControlLoop — THE single command chokepoint (04-runtime §6; 11-safety §4).

The only code in the stack that resolves per-tick commands and deposits them
for dispatch; every source (teleop, jog, goto/planner waypoints, future
policy/takeover) passes the safety gate inline in the tick. Dispatch itself
happens on the per-arm :class:`ArmSender` threads consuming the ``q_cmd``
slots. 100 Hz, monotonic absolute-deadline pacing, overruns skipped (no
catch-up bursts). ``run_tick(now)`` is callable synchronously for
deterministic tests.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from apollo_xarm7_core import (
    ArmState,
    Command,
    CommandResult,
    CommandSource,
    ProfileNotFoundError,
    ProfileStore,
    se3,
)
from apollo_xarm7_core.protocol import HELD_CODES, JointTargetArgs

from ..config import ControlConfig
from ..profiles.store import save_from_states, save_initial_overwrite
from .joint_panel import JogState, PlanExecutor
from .snapshot import StateSnapshot
from .teleop import TargetIntegrator, held_to_twist, twist_to_control_frame

if TYPE_CHECKING:
    from apollo_xarm7_core import IKSolver, WorkcellInterface

    from ..bus import RuntimeBus
    from ..safety.supervisor import SafetySupervisor

logger = logging.getLogger(__name__)

RAIL_TRAVEL_M = se3.RAIL_TRAVEL_M
GRIPPER_SEND_EVERY_N_TICKS = 10  # <= 10 Hz (modbus is slow)
PLAN_STATUS_LINGER_TICKS = 100  # keep "done"/"failed" visible ~1 s


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
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.workcell = workcell
        self.cfg = cfg
        self.bus = bus
        self.supervisor = supervisor
        self.session_arms = list(session_arms)
        self.ik = ik
        self.kin = kin
        self.planner = planner
        self.profile_store = profile_store
        self.workcell_kind = workcell_kind
        self._clock = clock
        self.dt = 1.0 / cfg.rate_hz

        self.active_arm: str | None = self.session_arms[0] if self.session_arms else None
        self.jog = JogState(cfg.jog)
        self.plans = PlanExecutor(cfg.jog)
        self.integrator = TargetIntegrator(cfg.leash.pos_m, cfg.leash.rot_rad)
        self.episode_state: str = "idle"  # phase-07 recorder wires this

        self._last_cmd: dict[str, np.ndarray] = {}
        self._grip_frac: dict[str, float] = {}
        self._states: dict[str, ArmState] = {}
        self._teleop_seeded: set[str] = set()
        self._seeded = False
        self._plan_state: dict[str, str] = {}  # arm -> planning|executing|failed
        self._plan_clear_at: dict[str, int] = {}
        self._plan_status: str | None = None  # session-level lifecycle string
        self._plan_status_clear_at: int | None = None

        self.tick_count = 0
        self.overrun_count = 0
        self.tick_durations: list[float] = []  # perf harness (bounded)
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
            if len(self.tick_durations) < 10000:
                self.tick_durations.append(self._clock() - t0)
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
            self._grip_frac[arm_id] = float(st.gripper.open_frac)
        self._teleop_seeded.clear()
        self._seeded = True

    def reseed_arm(self, arm_id: str) -> None:
        """Post-recovery re-seed: targets from measured, gate last-safe reset."""
        st = self.workcell.arms[arm_id].get_state()
        self._last_cmd[arm_id] = np.array(st.q, dtype=np.float64)
        self._teleop_seeded.discard(arm_id)
        if self.ik is not None:
            self.ik.reset(arm_id, st.q)
        self.supervisor.reseed(arm_id, st.q)

    # -- the tick (order fixed, 04-runtime §6) -------------------------------------
    def run_tick(self, now: float | None = None) -> StateSnapshot:
        now = self._clock() if now is None else now
        self.tick_count += 1
        if not self._seeded:
            self._seed_from_measured()
        self.bus.commands.drain(self._handle_command)  # 1

        got = self.bus.held_keys.get()  # 2
        held: frozenset[str] = got[0].held if got is not None else frozenset()
        scale = self.supervisor.watchdog.scale(now)
        watchdog_tripped = self.supervisor.watchdog.tripped

        states = self.workcell.states()  # 3 (driver caches; never blocks)
        self._states = states
        self.supervisor.sync(states, now)  # 4
        if self.ik is not None:
            self.ik.sync_passive(states)
        q_meas = {a: states[a].q for a in self.session_arms}

        # A held movement key cancels running plans (04-runtime §7).
        if self.plans.active_arms and scale > 0.0 and (held & HELD_CODES):
            self._cancel_plans("movement key")

        # 5-7: per-arm action resolution -> commanded q.
        q_cmd: dict[str, np.ndarray] = {}
        source = CommandSource.TELEOP
        for arm_id in self.session_arms:
            q_last = self._last_cmd[arm_id]
            q_next: np.ndarray | None = None
            if states[arm_id].error_code != 0:
                q_next = None  # FAULT: hold; recovery re-seeds (§15)
            elif self.plans.active(arm_id):
                q_next = self.plans.step(arm_id, q_last)
                source = CommandSource.PLANNER
                if not self.plans.active(arm_id):  # final waypoint reached
                    self._finish_plan(arm_id, ok=True)
            elif self.jog.active(arm_id):
                q_next = self._jog_step(arm_id, q_last, scale)
                source = CommandSource.JOINT_JOG
            elif arm_id == self.active_arm:
                q_next = self._teleop_step(arm_id, states[arm_id], q_last, held, scale)
            q_cmd[arm_id] = q_next if q_next is not None else q_last

        # Per-tick joint clamp (dq_max) + rail bound, before the gate.
        for arm_id, q in q_cmd.items():
            q_last = self._last_cmd[arm_id]
            q = np.clip(q, q_last - self.cfg.dq_max_rad, q_last + self.cfg.dq_max_rad)
            if q.shape[0] > 7:
                q[7] = min(max(q[7], 0.0), RAIL_TRAVEL_M)
            q_cmd[arm_id] = q

        dec = self.supervisor.filter(q_cmd, q_meas, source)  # 8

        for arm_id, q in dec.q_out.items():  # 9
            self._last_cmd[arm_id] = np.array(q)
            self.bus.arm_slot(arm_id).put(np.array(q))

        self._gripper_step(held, scale)
        self._expire_plan_status()

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
            episode=None,
            watchdog_tripped=watchdog_tripped,
            plan_status=dict(self._plan_state),
            session_extra={"plan_status": self._plan_status, "kind": self.workcell_kind},
        )
        self.bus.snapshot.put(snap)
        return snap

    # -- per-source steps -----------------------------------------------------
    def _teleop_step(
        self,
        arm_id: str,
        state: ArmState,
        q_last: np.ndarray,
        held: frozenset[str],
        scale: float,
    ) -> np.ndarray | None:
        """Held keys -> twist -> integrate -> IK -> q (04-runtime §6)."""
        if self.ik is None or self.kin is None:
            return None
        tw = held_to_twist(held, self.cfg.teleop)
        if scale <= 0.0:
            return None  # deadman latched: hold
        v = tw.v * scale
        w = tw.w * scale
        rail_v = tw.rail_v * scale
        has_rail = state.q.shape[0] > 7
        moving = bool(np.any(v) or np.any(w) or (rail_v and has_rail))
        measured_tcp = self.kin.tcp_world(arm_id, state.q)
        if arm_id not in self._teleop_seeded:
            self.integrator.seed(arm_id, measured_tcp)
            self._teleop_seeded.add(arm_id)
        if not moving:
            return None  # nothing held: non-active-style hold (no re-servo)
        from apollo_xarm7_core import Twist

        tw_world = twist_to_control_frame(
            Twist(v=v, w=w), self.kin.base_quat_world(arm_id), measured_tcp.orientation
        )
        target = self.integrator.step(arm_id, tw_world, self.dt, measured_tcp)
        result = self.ik.solve(arm_id, target, q_last)
        if result.diverged or not np.isfinite(result.pos_err_m):
            self.integrator.reanchor(arm_id, measured_tcp)
            return None
        if (
            result.pos_err_m > self.cfg.residual_max_pos_m
            or result.rot_err_rad > self.cfg.residual_max_rot_rad
        ):
            # Freeze the target back to the achieved pose (glide, don't wind up).
            self.integrator.reanchor(arm_id, self.kin.tcp_world(arm_id, result.q))
        q = np.array(result.q)
        if has_rail:
            # Rail keys integrate the rail slot directly (ignored w/o rail).
            base = q_last[7] if rail_v else q[7]
            q[7] = min(max(base + rail_v * self.dt, 0.0), RAIL_TRAVEL_M)
        return q

    def _jog_step(self, arm_id: str, q_last: np.ndarray, scale: float) -> np.ndarray | None:
        if scale <= 0.0:
            return None  # deadman: jog holds too (11-safety §10.1)
        q_next = self.jog.step(arm_id, q_last)
        if q_next is None:
            return None
        return q_last + (q_next - q_last) * scale

    def _gripper_step(self, held: frozenset[str], scale: float) -> None:
        arm_id = self.active_arm
        if arm_id is None:
            return
        tw = held_to_twist(held, self.cfg.teleop)
        if tw.grip_v == 0.0 or scale <= 0.0:
            return
        frac = self._grip_frac[arm_id] + tw.grip_v * self.dt * scale
        self._grip_frac[arm_id] = min(max(frac, 0.0), 1.0)
        if self.tick_count % GRIPPER_SEND_EVERY_N_TICKS == 0:
            sender = self._senders.get(arm_id)
            if sender is not None:
                sender.put_gripper(self._grip_frac[arm_id])

    # -- plan lifecycle ------------------------------------------------------------
    def _set_plan_status(self, status: str | None, linger: bool = False) -> None:
        self._plan_status = status
        self._plan_status_clear_at = (
            self.tick_count + PLAN_STATUS_LINGER_TICKS if linger else None
        )

    def _finish_plan(self, arm_id: str, ok: bool) -> None:
        if ok:
            self._plan_state.pop(arm_id, None)
            self._plan_clear_at.pop(arm_id, None)
            if not self.plans.active_arms:
                self._set_plan_status("done", linger=True)
        else:
            self._plan_state[arm_id] = "failed"
            self._plan_clear_at[arm_id] = self.tick_count + PLAN_STATUS_LINGER_TICKS
            self._set_plan_status("failed", linger=True)

    def _cancel_plans(self, reason: str) -> None:
        for arm_id in self.plans.active_arms:
            self._plan_state.pop(arm_id, None)
        self.plans.cancel()
        self._set_plan_status("cancelled", linger=True)
        logger.info("plan cancelled: %s", reason)

    def _expire_plan_status(self) -> None:
        if (
            self._plan_status_clear_at is not None
            and self.tick_count >= self._plan_status_clear_at
        ):
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

    def _op_switch_arm(self, cmd: Command) -> CommandResult:
        """Server-authoritative Tab cycling; previous arm's target freezes."""
        if not self.session_arms:
            return CommandResult(cmd.corr_id, False, "no session arms")
        i = self.session_arms.index(self.active_arm) if self.active_arm else -1
        self.active_arm = self.session_arms[(i + 1) % len(self.session_arms)]
        self._teleop_seeded.discard(self.active_arm)  # reseed target from measured
        return CommandResult(cmd.corr_id, True, self.active_arm)

    def _op_takeover_toggle(self, cmd: Command) -> CommandResult:
        if self.plans.active_arms:
            self._cancel_plans("takeover_toggle")
        return CommandResult(cmd.corr_id, False, "takeover not available in teleop")

    def _nack_episode(self, cmd: Command) -> CommandResult:
        return CommandResult(cmd.corr_id, False, "no recorder in teleop")

    _op_episode_new = _nack_episode
    _op_episode_save = _nack_episode
    _op_episode_discard = _nack_episode

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
        if self.plans.active(arm_id) or self._plan_state.get(arm_id) == "planning":
            return CommandResult(cmd.corr_id, False, "plan executing")
        target = np.asarray(args.positions, dtype=np.float64)
        if args.mode == "jog":
            delta = float(np.max(np.abs(target - self._last_cmd[arm_id])))
            if delta > self.cfg.jog.goto_threshold_rad:
                return CommandResult(
                    cmd.corr_id, False,
                    f"delta {delta:.3f} > goto threshold "
                    f"{self.cfg.jog.goto_threshold_rad}; use goto",
                )
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
        from apollo_xarm7_core import PlanRequest

        if self.supervisor.twin is None and hasattr(self.planner, "sync"):
            # Plain sim (NullGate): nothing else keeps the plan twin's
            # measured context fresh; sync here on the loop thread.
            self.planner.sync(self._states)
        q_start = {a: [float(x) for x in self._states[a].q] for a in q_goal}
        req = PlanRequest(q_start=q_start, q_goal=q_goal)
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

    def _op__plan_ready(self, cmd: Command) -> CommandResult:
        result = cmd.args.get("result")
        arms = cmd.args.get("arms", [])
        if result is None or not result.ok:
            for arm_id in arms:
                self._finish_plan(arm_id, ok=False)
            return CommandResult(cmd.corr_id, True, "plan failed")
        for arm_id in arms:
            self.plans.load(arm_id, result.waypoints[arm_id])
            self._plan_state[arm_id] = "executing"
        self._set_plan_status("executing")
        return CommandResult(cmd.corr_id, True, "executing")

    def _op_execute_plan(self, cmd: Command) -> CommandResult:
        """Internal: SessionManager hands pre-planned waypoints (start_from §5.2)."""
        waypoints: dict[str, list[list[float]]] = cmd.args["waypoints"]
        for arm_id, wps in waypoints.items():
            if arm_id not in self.session_arms:
                return CommandResult(cmd.corr_id, False, f"unknown arm {arm_id!r}")
            self.plans.load(arm_id, wps)
            self._plan_state[arm_id] = "executing"
        for arm_id, frac in cmd.args.get("gripper", {}).items():
            if arm_id in self._grip_frac:
                self._grip_frac[arm_id] = min(max(float(frac), 0.0), 1.0)
                sender = self._senders.get(arm_id)
                if sender is not None:
                    sender.put_gripper(self._grip_frac[arm_id])
        self._set_plan_status("executing")
        return CommandResult(cmd.corr_id, True, "executing")

    def _op_save_profile(self, cmd: Command) -> CommandResult:
        if self.profile_store is None:
            return CommandResult(cmd.corr_id, False, "no profile store")
        name = str(cmd.args.get("name", "")).strip()
        if not name:
            return CommandResult(cmd.corr_id, False, "profile name required")
        profile = save_from_states(
            self.profile_store, self._states, self.session_arms,
            self.workcell_kind, name, str(cmd.args.get("notes", "")),
        )
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


__all__ = ["ControlLoop", "GRIPPER_SEND_EVERY_N_TICKS", "PLAN_STATUS_LINGER_TICKS"]
