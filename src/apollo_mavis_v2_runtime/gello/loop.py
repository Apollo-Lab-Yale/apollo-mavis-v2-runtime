"""``GelloLoop`` — the 100 Hz mode loop of a GELLO Manipulation session (16-gello §6).

A ``ControlLoop`` subclass (the ``GatedPolicyExecutor`` pattern) whose ``_resolve_arms``
is, per arm::

    grip:  stopped -> hold | plan active (or the manager's motion window) -> _plan_step
           | else -> _gello_step
    view:  stopped -> hold | plan active -> _plan_step | viewpoint fresh -> _viewpoint_step
           | else -> hold

* **The Manipulation Arm** (``FOLLOWER_ARM_ID``) follows the leader ONLY while the engage
  machine (:class:`~apollo_mavis_v2_runtime.gello.engage.EngageMachine`) says ``tracking``:
  ``q[:7] = clip(unwrap(sample.q))`` (the ±2π branch fixed at engagement), ``q[7]`` = the
  rail slot integrated from ``_rail_rate(self.sources)`` — ←/→ from every source at that
  source's scale, exactly the rail-only branch of ``_teleop_step`` — and the gripper from
  the leader's trigger (``round(gripper_frac / gripper_quantum) * gripper_quantum`` through
  ``ArmSender.put_gripper`` every ``GRIPPER_SEND_EVERY_N_TICKS``). ``held_to_twist`` is
  never called for the translate / rotate / gripper keys, so they cannot act on the arm
  (D9); ``_gripper_step`` is a no-op. The result passes ``_cap_joint_step`` (uniform
  scaling: a fast leader is followed at the cap, never jumped to) and the gate like every
  other source; the tick reports ``CommandSource.GELLO``. In SIM the loop lowers
  ``dq_max_rad`` to ``gello.max_joint_vel_rad_s / rate_hz * speed_scale`` (0.6 rad/s ->
  0.006 rad/tick at 100 %), the cap the hardware streamer has anyway, so a sim
  proof-of-motion has the hardware's lag and feel - and, BEFORE the base class builds its
  ``PlanExecutor``, lowers ``jog.slew_rad_per_tick`` to the same value (the hardware
  pairing of ``apply_teleop_caps`` / ``apply_executor_caps``; 2026-09-09 review: with the
  executor left at 0.02 rad its waypoint index advanced while the cap held the command
  3.3x short, so every twin-planned motion cut the corner at every waypoint).
* **The Perception Arm** (``VIEW_ARM_ID``) follows the :class:`ViewpointSource` (§7): the
  Online DAgger application path with the Manipulation Arm removed from the layout —
  ``split_action`` over ``arms_meta = [("view", has_rail)]``, ``ActionAnchor.apply_delta``
  (measured ⊕ Δ -> IK, ``SlewLimits``), rail delta, gripper dim ignored (no gripper), then
  the cap and the gate. A NaN three-strike pauses the source until the operator's Resume
  (there is no episode boundary in gello). The source attaches only while the session is
  RUNNING and no planned motion owns the arm (``session_running`` + the motion window).
* **Every hold is "hold the last command"** (the loop's ``None`` resolution re-servos
  ``_last_cmd``), never a move.

Engagement events (§6.1): ``gello_pause`` / ``gello_resume`` (idempotent actions, no key;
``gello_resume`` is NACKED while a planned motion owns the arm - ``planned motion in
progress - Resume after it ends`` - so a Resume clicked mid-return can never pre-arm the
engage rule and defeat §5.3's "paused afterwards"; ``gello_pause`` stays accepted then, it
latches); every ``FaultEvent`` / RECOVERING of the Manipulation Arm forces ``paused`` (the
RECOVERING exit rule needs "no live input" — a non-tracking GELLO is exactly that); ``R`` /
``goto_profile`` force ``paused`` once the base op ACCEPTED (a refused request leaves the
state and the unwrap branch exactly as they were; the manager pauses again, synchronously,
before it plans) and leave it paused afterwards (§5.3). The manager brackets its
twin-planned motions (the launch motion, the returns) with the internal ``gello_motion
{active}`` command so the state reads ``motion`` for the WHOLE sequence — also between two
arms' plans and between the joints and carriage phases — and the engage rule runs once when
the window closes ("engage on arrival"); a plan the executor owns (``plans.active``) is
``motion`` too.

Nacks: ``switch_arm`` / ``switch_arm_prev`` (``GELLO_NO_ARM_SWITCH``), ``takeover*`` /
``handback`` (``GELLO_NO_TAKEOVER``), the episode actions (``GELLO_NO_RECORDING``),
``joint_target`` (``GELLO_NO_JOINT_PANEL``); ``train_now`` keeps the base "not an Online
DAgger session"; ``tracker_settings`` is unaffected (no tracker provider here: D9, the
Vive cannot reach the Manipulation Arm in this mode).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from apollo_mavis_v2_core import ArmState, Command, CommandResult, CommandSource

from ..control.loop import GRIPPER_SEND_EVERY_N_TICKS, RAIL_TRAVEL_M, ControlLoop
from ..dagger.policy_runner import NAN_STRIKES_PER_EPISODE, split_action
from . import FOLLOWER_ARM_ID, VIEW_ARM_ID
from .engage import EngageMachine

if TYPE_CHECKING:
    from ..config import GelloConfig
    from ..dagger.policy_runner import ActionAnchor
    from ..devices.gello import GelloSample
    from .viewpoint import ViewpointSource

logger = logging.getLogger(__name__)

GELLO_NO_ARM_SWITCH = (
    "GELLO drives the Manipulation Arm; the Perception Arm follows the viewpoint node"
)
GELLO_NO_TAKEOVER = "takeover not available in GELLO Manipulation"
GELLO_NO_RECORDING = "GELLO Manipulation records nothing (v1) - no episodes"
GELLO_NO_JOINT_PANEL = (
    "joint panel not available in GELLO Manipulation (the leader drives the Manipulation Arm)"
)
PAUSED_BY_OPERATOR = "paused by operator"
PAUSED_FOR_MOTION = "paused: planned motion requested (press Resume afterwards)"
RESUME_DURING_MOTION = "planned motion in progress - Resume after it ends"


def sim_gello_caps(cfg, gello_cfg: GelloConfig, speed_scale: float):
    """``ControlConfig`` for a SIM gello loop: ``dq_max_rad`` AND ``jog.slew_rad_per_tick``
    both lowered to ``gello.max_joint_vel_rad_s / rate_hz x speed_scale`` (0.006 rad/tick at
    100 %) - the pairing the hardware bring-up applies through ``apply_teleop_caps`` +
    ``apply_executor_caps``. ``jog.rail_m_per_tick`` is left alone (the hardware pairing
    lowers it only when the driver publishes a track speed; SimArm has none). Applied
    BEFORE ``ControlLoop.__init__`` builds the ``PlanExecutor``, so planned segments are
    walked in equal ticks at the cap and every waypoint is reached before the next segment
    starts (2026-09-09 review)."""
    per_tick = float(gello_cfg.max_joint_vel_rad_s) / float(cfg.rate_hz) * float(speed_scale)
    if per_tick <= 0.0:
        return cfg
    dq = min(float(cfg.dq_max_rad), per_tick)
    slew = min(float(cfg.jog.slew_rad_per_tick), per_tick)
    if dq == float(cfg.dq_max_rad) and slew == float(cfg.jog.slew_rad_per_tick):
        return cfg
    return cfg.model_copy(
        update={"dq_max_rad": dq, "jog": cfg.jog.model_copy(update={"slew_rad_per_tick": slew})}
    )


class GelloLoop(ControlLoop):
    """The GELLO Manipulation mode loop (module docstring)."""

    def __init__(
        self,
        *args,
        gello_cfg: GelloConfig,
        reader: Any,  # GelloReader-like: latest() -> GelloSample | None
        engage: EngageMachine | None = None,
        viewpoint: ViewpointSource | None = None,
        anchor: ActionAnchor | None = None,
        launch_pending: bool = False,  # the manager's launch motion window starts open
        **kwargs,
    ) -> None:
        kwargs["tracker"] = None  # D9: the Vive cannot reach the Manipulation Arm in gello
        # SIM: SimArm has no servo slew of its own, so the follower's per-tick cap is the
        # hardware value here too (16-gello §6.2) - and the PlanExecutor's slew with it
        # (``sim_gello_caps``), BEFORE the base class builds the executor; a hardware loop
        # already carries the streamer's caps through apply_teleop_caps / apply_executor_caps
        # and is left alone.
        if kwargs.get("workcell_kind", "sim") != "hardware":
            args = list(args)
            cfg_ix = 1 if len(args) > 1 else None
            base_cfg = args[cfg_ix] if cfg_ix is not None else kwargs["cfg"]
            capped = sim_gello_caps(
                base_cfg, gello_cfg, float(kwargs.get("speed_scale", 0.1))
            )
            if cfg_ix is not None:
                args[cfg_ix] = capped
            else:
                kwargs["cfg"] = capped
        super().__init__(*args, **kwargs)
        self.gello_cfg = gello_cfg
        self.reader = reader
        self.engage = engage or EngageMachine(gello_cfg)
        self.viewpoint = viewpoint
        self.anchor = anchor
        self.follower = FOLLOWER_ARM_ID
        self.view_arm = VIEW_ARM_ID
        if self.follower in self.session_arms:
            self.active_arm = self.follower  # pinned; switch_arm is nacked
        try:
            has_rail = bool(self.workcell.arms[self.view_arm].has_rail)
        except Exception:  # noqa: BLE001 - a session without the Perception Arm (unit loops)
            has_rail = True
        self.arms_meta_view: list[tuple[str, bool]] = [(self.view_arm, has_rail)]
        # the manager's motion window (``gello_motion {active}``): the launch motion / a
        # return owns the Manipulation Arm for the whole sequence, plans or not
        self._motion_hold = bool(launch_pending)
        self._motion_reason = "launch motion" if launch_pending else ""
        # SessionManager hook: the session is RUNNING (viewpoint attach rule, §7)
        self.session_running: Callable[[], bool] = lambda: True
        self._gello_drove = False
        self._view_drove = False
        self._sample: GelloSample | None = None
        self._nan_strikes = 0
        self._vp_detaches = viewpoint.detaches if viewpoint is not None else 0
        self._last_out_t = -1.0
        self.gello_ticks = 0  # ticks the leader was streamed (diagnostics)
        self.gripper_sends = 0

    # -- the mux ------------------------------------------------------------------------------
    def _resolve_arms(
        self, states: dict[str, ArmState], held: frozenset[str], scale: float, now: float
    ) -> tuple[dict[str, np.ndarray | None], CommandSource]:
        self._sample = self.reader.latest() if self.reader is not None else None
        self._gello_drove = False
        self._view_drove = False
        out: dict[str, np.ndarray | None] = {}
        source = CommandSource.TELEOP
        for arm_id in self.session_arms:
            q_last = self._last_cmd[arm_id]
            state = states[arm_id]
            if arm_id == self.follower:
                if self.arm_stopped(arm_id, state):
                    out[arm_id] = None  # FAULT / RECOVERING: hold (04-runtime §15)
                    self._engage_fault(state)
                elif self.plans.active(arm_id):
                    out[arm_id] = self._plan_step(arm_id, q_last)  # finished after the gate
                    source = CommandSource.PLANNER
                    self.engage.step(now, self._sample, state.q, q_last, plan_active=True)
                else:
                    out[arm_id] = self._gello_step(arm_id, state, q_last, now)
            elif arm_id == self.view_arm:
                if self.arm_stopped(arm_id, state):
                    out[arm_id] = None
                    self._poll_viewpoint(now, allow_attach=False)
                elif self.plans.active(arm_id):
                    out[arm_id] = self._plan_step(arm_id, q_last)
                    source = CommandSource.PLANNER
                    self._poll_viewpoint(now, allow_attach=False)
                else:
                    out[arm_id] = self._viewpoint_step(arm_id, state, q_last, now)
            else:
                out[arm_id] = None  # no third arm in this mode
        if self._gello_drove:
            source = CommandSource.GELLO  # a GELLO-driven tick (16-gello §6)
        elif self._view_drove and source is CommandSource.TELEOP:
            source = CommandSource.POLICY
        return out, source

    # -- the Manipulation Arm -----------------------------------------------------------------
    def _engage_fault(self, state: ArmState) -> None:
        """A stopped Manipulation Arm (latched error / FaultEvent / RECOVERING) forces
        ``paused`` once; the engage machine is not stepped while the arm is stopped, so it
        can never read ``tracking`` there (the RECOVERING exit rule, §6.1)."""
        if self.engage.paused:
            return
        if self.follower in self._faulted:
            why = self._fault_text.get(self.follower) or "driver fault"
        elif self.follower in self._recovering:
            why = "recovering - release every input, then Resume"
        else:
            why = f"controller error {state.error_code}"
        self.engage.on_fault(why)

    def _gello_step(
        self, arm_id: str, state: ArmState, q_last: np.ndarray, now: float
    ) -> np.ndarray | None:
        """Leader joints (tracking only) + the rail slot from ←/→ (every state) +
        the trigger gripper (16-gello §6.2). None = hold the last command."""
        sample = self._sample
        st = self.engage.step(
            now, sample, state.q, q_last, plan_active=self._motion_hold, paused_request=None
        )
        has_rail = state.q.shape[0] > 7
        rail_v = self._rail_rate(self.sources) if has_rail else 0.0
        target = self.engage.target(sample) if st == "tracking" else None
        if target is None and rail_v == 0.0:
            return None  # hold: no_leader / out_of_sync / paused / motion, no rail key
        q = np.array(q_last, dtype=np.float64)
        if target is not None:
            q[:7] = target
            self._gello_drove = True
            self.gello_ticks += 1
            self._note_source(arm_id, CommandSource.GELLO)
            self._gello_gripper(arm_id, sample)
        if rail_v != 0.0 and has_rail:
            # the rail-only rule of _teleop_step: the joints ride the carriage, the slot
            # integrates at the source's scale, clamped to the travel
            q[7] = min(max(float(q_last[7]) + rail_v * self.dt, 0.0), RAIL_TRAVEL_M)
        return q

    def _gello_gripper(self, arm_id: str, sample: GelloSample | None) -> None:
        """The leader's trigger -> ``_grip_frac`` quantised to ``gripper_quantum``; sent
        through ``ArmSender.put_gripper`` on the policy path's cadence
        (``GRIPPER_SEND_EVERY_N_TICKS``; the G2 modbus is slow) whenever the quantised
        value differs from the value the arm was LAST COMMANDED WITH - ``_grip_frac`` as
        left by the previous send, by ``_apply_gripper_target`` (a profile's gripper
        applied on arrival) or by ``reseed_arm`` (the measured opening after a recovery).
        2026-09-09 review: comparing against a private "last sent" copy let a profile /
        re-seed change the real gripper while the leader's unchanged trigger was never
        re-sent; ``_grip_frac`` (and telemetry ``gripper_frac``) now always names the value
        in force, read on the cadence ticks only."""
        if sample is None or sample.gripper_frac is None or arm_id not in self.gripper_arms:
            return
        if self.tick_count % GRIPPER_SEND_EVERY_N_TICKS != 0:
            return
        quantum = float(self.gello_cfg.gripper_quantum)
        frac = round(float(sample.gripper_frac) / quantum) * quantum
        frac = min(max(frac, 0.0), 1.0)
        prev = self._grip_frac.get(arm_id)
        if prev is not None and abs(frac - float(prev)) < 1e-9:
            return
        self._grip_frac[arm_id] = frac
        sender = self._senders.get(arm_id)
        if sender is not None:
            sender.put_gripper(frac)
        self.gripper_sends += 1

    def _gripper_step(self, sources) -> None:
        """F/H are ignored in gello mode: the gripper follows the leader's trigger (D9)."""
        return

    # -- the Perception Arm -------------------------------------------------------------------
    def _poll_viewpoint(self, now: float, allow_attach: bool) -> None:
        vp = self.viewpoint
        if vp is None or vp.mode == "hold":
            return
        try:
            vp.poll(now, allow_attach=allow_attach)
        except Exception:  # noqa: BLE001 - the bus never breaks the tick
            logger.exception("gello viewpoint poll failed")
        if vp.detaches != self._vp_detaches:
            # a detached (restarted / re-attached) node starts with a clean NaN count: the
            # source dropped its own pause on detach (2026-09-09 review)
            self._vp_detaches = vp.detaches
            self._nan_strikes = 0

    def _viewpoint_step(
        self, arm_id: str, state: ArmState, q_last: np.ndarray, now: float
    ) -> np.ndarray | None:
        """The view block from the viewpoint node through the Online DAgger application
        path (``ActionAnchor`` + slew), the Manipulation Arm removed from the layout
        (16-gello §6.3). None = hold the GELLO hold posture / the last command."""
        vp = self.viewpoint
        if vp is None or vp.mode == "hold":
            return None
        allow = bool(self.session_running()) and not self._motion_hold
        self._poll_viewpoint(now, allow_attach=allow)
        if not vp.attached or self.anchor is None:
            return None
        out, t_out = vp.latest()
        if out is None:
            return None
        if t_out != self._last_out_t:
            self._last_out_t = t_out
            if not np.all(np.isfinite(out.actions)):
                self._nan_strike(now)
        if not np.all(np.isfinite(out.actions)):
            return None
        stale = vp.staleness_scale(now)
        if stale <= 0.0:
            return None  # > 5 stale periods: hold (12-dagger §6.3)
        block = split_action(out.actions, self.arms_meta_view).get(arm_id)
        if block is None or block.shape[0] < 7:
            return None
        tick_scale = (self.dt / vp.period) * stale
        has_rail = state.q.shape[0] > 7
        rail_d = float(block[7]) * tick_scale if has_rail and block.shape[0] > 7 else None
        q = self.anchor.apply_delta(
            arm_id,
            np.asarray(block[:3], dtype=np.float64) * tick_scale,
            np.asarray(block[3:6], dtype=np.float64) * tick_scale,
            rail_d,
            q_last,
            state.q,
            self.dt,
            now,
        )
        if q is None:
            return None
        self._note_source(arm_id, CommandSource.POLICY)
        self._view_drove = True
        return q

    def _nan_strike(self, now: float) -> None:
        vp = self.viewpoint
        if vp is not None and vp.paused:
            return  # already paused: the count stays at the strike that paused it
        self._nan_strikes += 1
        if self._nan_strikes >= NAN_STRIKES_PER_EPISODE and vp is not None and not vp.paused:
            vp.pause(f"paused after {self._nan_strikes} NaN actions - press Resume")
            logger.warning(
                "gello viewpoint: %d NaN actions - source paused until Resume", self._nan_strikes
            )

    # -- driver faults -> paused ---------------------------------------------------------------
    def _on_fault_event(self, arm_id: str, ev: object, now: float) -> None:
        super()._on_fault_event(arm_id, ev, now)
        if arm_id == self.follower:
            self.engage.on_fault(self._fault_text.get(arm_id) or "driver fault")

    def _on_recovered_event(self, arm_id: str, ev: object, now: float, reseeded: set[str]) -> None:
        super()._on_recovered_event(arm_id, ev, now, reseeded)
        if arm_id == self.follower and not self.engage.paused:
            self.engage.on_fault("recovering - release every input, then Resume")

    def _update_recovering(self) -> None:
        """RECOVERING -> RUNNING needs "no live input" (04-runtime §15): a TRACKING GELLO is
        a live input; a paused / out-of-sync / leaderless one is not (16-gello §6.1)."""
        if not self._recovering:
            return
        if self.follower in self._recovering and self.engage.state == "tracking":
            return
        super()._update_recovering()

    # -- telemetry + health -------------------------------------------------------------------
    def _session_extra(self, now: float) -> dict:
        lag = self.engage.lag_rad()
        vp = self.viewpoint
        return {
            "gello": {
                "state": self.engage.state,
                "state_detail": self.engage.detail,
                "lag_rad": [float(x) for x in lag] if lag is not None else None,
                "max_lag_rad": self.engage.max_lag_rad(),
                "engaged_arm": self.follower if self.engage.state == "tracking" else None,
                "viewpoint": vp.telemetry() if vp is not None else None,
                # the pause LATCH (2026-09-09 review): True while a pause is requested /
                # forced, also inside a motion window where the display still says `motion`
                "paused_latched": bool(self.engage.paused),
            }
        }

    def _mode_health(self, now: float) -> str:
        sample = self._sample
        age = f"{(now - sample.rx_mono) * 1e3:.0f}ms" if sample is not None else "none"
        lag = self.engage.max_lag_rad()
        lag_s = f"{lag:.3f}" if lag is not None else "n/a"
        out = f" gello={self.engage.state} age={age} lag={lag_s}"
        vp = self.viewpoint
        if vp is not None and vp.mode != "hold":
            if vp.paused:
                out += f" viewpoint=paused({self._nan_strikes} NaN)"
            else:
                out += f" viewpoint={'attached' if vp.attached else 'hold'}"
        return out

    # -- commands -----------------------------------------------------------------------------
    def _op_gello_pause(self, cmd: Command) -> CommandResult:
        """``gello_pause`` (Cockpit button; idempotent): the follower holds until Resume."""
        reason = str(cmd.args.get("reason") or PAUSED_BY_OPERATOR)
        if self.engage.paused:
            return CommandResult(cmd.corr_id, True, "already paused")
        self.engage.pause(reason)
        logger.info("gello: %s", reason)
        return CommandResult(cmd.corr_id, True, "paused")

    def _op_gello_resume(self, cmd: Command) -> CommandResult:
        """``gello_resume`` (idempotent): drop the pause latch (the engage rule re-runs on
        the next tick: tracking within tolerance, else out_of_sync) and lift a NaN-paused
        viewpoint source. NACKED while a planned motion owns the Manipulation Arm (the
        manager's window or an executor plan): a Resume then would clear the latch without
        any visible effect and let the follower engage the instant the return ends, against
        §5.3's "paused afterwards" (2026-09-09 review)."""
        if self._motion_hold or self.plans.active(self.follower):
            return CommandResult(cmd.corr_id, False, RESUME_DURING_MOTION)
        vp = self.viewpoint
        if vp is not None and vp.paused:
            self._nan_strikes = 0
            vp.resume()
        if not self.engage.paused:
            return CommandResult(cmd.corr_id, True, "not paused")
        self.engage.resume()
        logger.info("gello: resumed - waiting for the engage rule")
        return CommandResult(cmd.corr_id, True, "resumed")

    def _op_gello_motion(self, cmd: Command) -> CommandResult:
        """Internal (SessionManager): open / close the motion window around a twin-planned
        sequence (the launch motion, a return). Open -> the engage machine reads ``motion``
        whatever the executor does; close -> the engage rule runs once on the next tick."""
        active = bool(cmd.args.get("active", False))
        self._motion_hold = active
        self._motion_reason = str(cmd.args.get("reason") or "planned motion") if active else ""
        return CommandResult(cmd.corr_id, True, "motion" if active else "released")

    def _pause_for_motion(self) -> None:
        """After ``R`` / Go to profile was ACCEPTED (16-gello §5.3): the follower stops
        following now, in the same tick as the ack; the manager's worker pauses again
        (idempotent) before it plans. An earlier pause keeps its own reason. A REFUSED
        request never reaches here, so the state and the unwrap branch stay exactly as they
        were (2026-09-09 review: pausing first and ``resume()``-ing on the nack re-ran the
        engage rule against the MEASURED arm and dropped a lagging follower to
        out_of_sync)."""
        if not self.engage.paused:
            self.engage.pause(PAUSED_FOR_MOTION)

    def _op_reset_to_initial(self, cmd: Command) -> CommandResult:
        res = super()._op_reset_to_initial(cmd)
        if res.ok:
            self._pause_for_motion()
        return res

    def _op_goto_profile(self, cmd: Command) -> CommandResult:
        res = super()._op_goto_profile(cmd)
        if res.ok:
            self._pause_for_motion()
        return res

    def _op_switch_arm(self, cmd: Command) -> CommandResult:
        return CommandResult(cmd.corr_id, False, GELLO_NO_ARM_SWITCH)

    def _op_switch_arm_prev(self, cmd: Command) -> CommandResult:
        return CommandResult(cmd.corr_id, False, GELLO_NO_ARM_SWITCH)

    def _op_takeover_toggle(self, cmd: Command) -> CommandResult:
        if self.plans.active_arms:
            self._cancel_plans("takeover_toggle")  # Space stays an escape
        return CommandResult(cmd.corr_id, False, GELLO_NO_TAKEOVER)

    def _op_takeover(self, cmd: Command) -> CommandResult:
        if self.plans.active_arms:
            self._cancel_plans("takeover")
        return CommandResult(cmd.corr_id, False, GELLO_NO_TAKEOVER)

    def _op_handback(self, cmd: Command) -> CommandResult:
        return CommandResult(cmd.corr_id, False, GELLO_NO_TAKEOVER)

    def _op_episode_new(self, cmd: Command) -> CommandResult:
        return CommandResult(cmd.corr_id, False, GELLO_NO_RECORDING)

    def _op_episode_save(self, cmd: Command) -> CommandResult:
        return CommandResult(cmd.corr_id, False, GELLO_NO_RECORDING)

    def _op_episode_discard(self, cmd: Command) -> CommandResult:
        return CommandResult(cmd.corr_id, False, GELLO_NO_RECORDING)

    def _op_joint_target(self, cmd: Command) -> CommandResult:
        return CommandResult(cmd.corr_id, False, GELLO_NO_JOINT_PANEL)


__all__ = [
    "GELLO_NO_ARM_SWITCH",
    "GELLO_NO_JOINT_PANEL",
    "GELLO_NO_RECORDING",
    "GELLO_NO_TAKEOVER",
    "PAUSED_BY_OPERATOR",
    "PAUSED_FOR_MOTION",
    "RESUME_DURING_MOTION",
    "GelloLoop",
    "sim_gello_caps",
]
