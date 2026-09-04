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
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from apollo_xarm7_core import (
    ArmState,
    Command,
    CommandResult,
    CommandSource,
    Pose,
    ProfileNotFoundError,
    ProfileStore,
    se3,
)
from apollo_xarm7_core.protocol import HELD_CODES, JointTargetArgs, TrackerSettingsArgs

from ..config import ControlConfig
from ..profiles.store import save_from_states, save_initial_overwrite
from .joint_panel import JogState, PlanExecutor
from .snapshot import StateSnapshot
from .teleop import TargetIntegrator, held_to_twist, twist_to_control_frame
from .tracker_teleop import TRACKER_CLUTCH_CODE, interp_pose

if TYPE_CHECKING:
    from apollo_xarm7_core import IKSolver, Pose, WorkcellInterface

    from ..bus import RuntimeBus
    from ..safety.supervisor import SafetySupervisor
    from .tracker_teleop import TrackerTeleop

logger = logging.getLogger(__name__)

RAIL_TRAVEL_M = se3.RAIL_TRAVEL_M
GRIPPER_SEND_EVERY_N_TICKS = 10  # <= 10 Hz (modbus is slow)
PLAN_STATUS_LINGER_TICKS = 100  # keep "done"/"failed" visible ~1 s
DEVICE_ACTION_LINGER_S = 1.0  # telemetry shows the last device-sourced discrete action this long


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
    ) -> None:
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

        self.active_arm: str | None = self.session_arms[0] if self.session_arms else None
        self.jog = JogState(cfg.jog)
        self.plans = PlanExecutor(cfg.jog)
        self.integrator = TargetIntegrator(cfg.leash.pos_m, cfg.leash.rot_rad)
        self.episode_state: str = "idle"  # phase-07 recorder wires this

        self._last_cmd: dict[str, np.ndarray] = {}
        self._grip_frac: dict[str, float] = {}
        self._states: dict[str, ArmState] = {}
        self._teleop_seeded: set[str] = set()
        # Arm of the current clutch session (clutch PRESS-edge tracking).
        self._clutch_arm: str | None = None
        self._arm_source: dict[str, CommandSource] = {}  # per-arm last resolving source
        self._edge_seq_seen: int | None = None  # controller.edge_seq adopted last tick
        self._device_action: tuple[str, float] | None = None  # (label, show until)
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
            if arm_id in self.gripper_arms:
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
        held_ws: frozenset[str] = got[0].held if got is not None else frozenset()
        scale = self.supervisor.watchdog.scale(now)  # WS-source scale
        watchdog_tripped = self.supervisor.watchdog.tripped
        device_codes, device_scale = self._device_inputs(now)
        self.sources = HeldSources(held_ws, scale, device_codes, device_scale)
        held = self.sources.held  # held_eff = held ∪ device_codes (13-tracker §1.1)
        if self.tracker is not None:
            self.tracker.clutch = TRACKER_CLUTCH_CODE in held
            if TRACKER_CLUTCH_CODE not in held:
                self._clutch_arm = None  # clutch up -> next press is a rising edge

        states = self.workcell.states()  # 3 (driver caches; never blocks)
        self._states = states
        self.supervisor.sync(states, now)  # 4
        if self.ik is not None:
            self.ik.sync_passive(states)
        q_meas = {a: states[a].q for a in self.session_arms}

        # A held movement key (from a live source) cancels running plans (04-runtime §7).
        if self.plans.active_arms and self.sources.moving(HELD_CODES):
            self._cancel_plans("movement key")

        # 5-7: per-arm action resolution -> commanded q (mode hook, phase-08).
        resolved, source = self._resolve_arms(states, held, scale, now)
        if self.tracker is not None:
            self.tracker.end_tick(now)  # not consulted this tick -> anchors cleared
        q_cmd: dict[str, np.ndarray] = {
            arm_id: (q if q is not None else self._last_cmd[arm_id])
            for arm_id, q in resolved.items()
        }

        # Per-tick joint clamp (dq_max) + rail bound, before the gate.
        for arm_id, q in q_cmd.items():
            q_last = self._last_cmd[arm_id]
            q = np.clip(q, q_last - self.cfg.dq_max_rad, q_last + self.cfg.dq_max_rad)
            if q.shape[0] > 7:
                q[7] = min(max(q[7], 0.0), RAIL_TRAVEL_M)
            q_cmd[arm_id] = q

        dec = self.supervisor.filter(q_cmd, q_meas, source)  # 8
        self._post_filter(dec, now)

        for arm_id, q in dec.q_out.items():  # 9
            self._last_cmd[arm_id] = np.array(q)
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
        return snap

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
            if states[arm_id].error_code != 0:
                q_next = None  # FAULT: hold; recovery re-seeds (§15)
            elif self.plans.active(arm_id):
                q_next = self.plans.step(arm_id, q_last)
                source = CommandSource.PLANNER
                self._note_source(arm_id, CommandSource.PLANNER)
                if not self.plans.active(arm_id):  # final waypoint reached
                    self._finish_plan(arm_id, ok=True)
            elif self.jog.active(arm_id):
                q_next = self._jog_step(arm_id, q_last, scale)
                source = CommandSource.JOINT_JOG
                self._note_source(arm_id, CommandSource.JOINT_JOG)
            elif arm_id == self.active_arm:
                self._note_source(arm_id, CommandSource.TELEOP)
                q_next = self._teleop_step(arm_id, states[arm_id], q_last, held, scale, now)
            out[arm_id] = q_next
        return out, source

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
            d_rail = self.kin.tcp_world(arm_id, q_rail).position - self.kin.tcp_world(
                arm_id, q_last
            ).position

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
                from apollo_xarm7_core import Twist

                tw_world = twist_to_control_frame(
                    Twist(v=v, w=w), self.kin.base_quat_world(arm_id), measured_tcp.orientation
                )
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
            self.integrator.reanchor(arm_id, measured_tcp)
            if self.tracker is not None:
                self.tracker.slip(target, measured_tcp)
            return None
        over_pos = result.pos_err_m > self.cfg.residual_max_pos_m
        over_rot = result.rot_err_rad > self.cfg.residual_max_rot_rad
        if over_pos or over_rot:
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
        fresh = now - sample.rx_mono <= self.tracker.stale_s
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
        and stops within ``stale_s`` when the controller stream dies."""
        arm_id = self.active_arm
        if arm_id is None or arm_id not in self.gripper_arms:
            return  # gripper keys are ignored on a camera-only arm
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

    def _switch_arm(self, cmd: Command, step: int) -> CommandResult:
        if not self.session_arms:
            return CommandResult(cmd.corr_id, False, "no session arms")
        i = self.session_arms.index(self.active_arm) if self.active_arm else -step
        self.active_arm = self.session_arms[(i + step) % len(self.session_arms)]
        self._teleop_seeded.discard(self.active_arm)  # reseed target from measured
        if self.tracker is not None:
            self.tracker.release()  # arm switch clears the tracker anchors
        return CommandResult(cmd.corr_id, True, self.active_arm)

    def _op_switch_arm(self, cmd: Command) -> CommandResult:
        """Server-authoritative Tab / RB cycling; previous arm's target freezes."""
        return self._switch_arm(cmd, +1)

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
            yaw_deg=args.yaw_deg, pos_scale=args.pos_scale, follow_rotation=args.follow_rotation,
            filter_enabled=args.filter_enabled, filter_min_cutoff_hz=args.filter_min_cutoff_hz,
            filter_beta=args.filter_beta,
        )
        return CommandResult(
            cmd.corr_id, True,
            f"yaw_deg={v.yaw_deg:g} pos_scale={v.pos_scale:g} "
            f"follow_rotation={str(v.follow_rotation).lower()} "
            f"filter_enabled={str(v.filter_enabled).lower()} "
            f"filter_min_cutoff_hz={v.filter_min_cutoff_hz:g} filter_beta={v.filter_beta:g}",
        )

    def _op_takeover_toggle(self, cmd: Command) -> CommandResult:
        if self.plans.active_arms:
            self._cancel_plans("takeover_toggle")
        return CommandResult(cmd.corr_id, False, "takeover not available in teleop")

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
            if arm_id in self.gripper_arms:
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


__all__ = [
    "ControlLoop",
    "DEVICE_ACTION_LINGER_S",
    "GRIPPER_SEND_EVERY_N_TICKS",
    "HeldSources",
    "PLAN_STATUS_LINGER_TICKS",
]
