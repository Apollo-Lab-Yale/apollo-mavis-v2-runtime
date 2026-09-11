"""GatedPolicyExecutor + DaggerSession + InferenceSession (12-dagger §3/§6/§10).

The executor is the per-arm policy/human mux — a ``ControlLoop`` subclass, so
policy actions ride the SAME 100 Hz chokepoint, per-tick clamp and twin gate
as every other source. DAgger and inference share it verbatim; the ONLY fork
is ``recorder``: ``DaggerSession`` passes a ``DaggerRecorderThread``,
``InferenceSession`` passes ``recorder=None`` — no dataset object exists in
inference mode, so safety-escape frames structurally cannot be recorded.

Phase-14 (15-online-dagger §3, D3): an Online DAgger session hands the executor its
``OnlineDaggerCoordinator`` (``coordinator=``). The executor then refuses
``episode_new`` with the coordinator's reason (no trainer / waiting for ready /
training in progress / trainer error), serves ``train_now`` (a Cockpit button: ask
the trainer to train on the rollouts saved so far) and fills
``DaggerStatus.online_dagger`` for telemetry. The gate API is session-agnostic:
``takeover`` / ``handback`` are the idempotent siblings of Space's
``takeover_toggle``, and EVERY ``TakeoverGate`` event (Space, the actions,
auto-advance, the episode-boundary reset) is handed to ``on_gate_events`` as an
``events.gate`` payload ``{arm_id, mode, seq, source, episode_id}`` — the manager
wires that to the coordinator (its serial worker) or to the dora publisher's
queue, never to a bus call on the tick. The coordinator never touches the arms.
"""

from __future__ import annotations

import logging
import threading
from collections import deque

import numpy as np
from apollo_mavis_v2_core import ArmState, Command, CommandResult, CommandSource, Pose, se3
from apollo_mavis_v2_core.dagger import ControlMode
from apollo_mavis_v2_core.interfaces.policy import Observation
from apollo_mavis_v2_core.protocol import DaggerStatus, InferenceStatus

from ..control.loop import GRIPPER_SEND_EVERY_N_TICKS, NOT_ONLINE_DAGGER, ControlLoop
from ..recorder.features import arm_action_names
from .policy_runner import (
    BLOCK_STREAK_TICKS,
    NAN_STRIKES_PER_EPISODE,
    ActionAnchor,
    PolicyAnomalyEvent,
    split_action,
)
from .policy_source import PolicySource
from .step import action_space_of, policy_step, staleness_of

logger = logging.getLogger(__name__)

TAKEOVER_ACTIVE = "takeover active"
ALREADY_TAKEN_OVER = "already taken over"  # takeover while HUMAN / TRANSITION (idempotent ack)
POLICY_DRIVING = "policy driving - take over (Space) first"  # R / Go to profile under POLICY
POLICY_ALREADY_DRIVING = "policy already driving"  # handback while POLICY (idempotent ack)
OD_STATUS_EVERY_N = 4  # OnlineDaggerStatus rebuilt at 25 Hz (the telemetry rate), not per tick
EPISODE_OPEN_STATES = ("recording", "saving")  # an episode exists; leaving them is a boundary


class GatedPolicyExecutor(ControlLoop):
    """100 Hz control loop where the policy drives AUTONOMOUS arms and the
    human drives the engaged arm; used by BOTH DaggerSession and
    InferenceSession (12-dagger §3)."""

    def __init__(
        self,
        *args,
        gate,
        runner: PolicySource,  # PolicyRunner (in-process) | ExternalPolicySource (dora)
        anchor: ActionAnchor,
        arms_meta: list[tuple[str, bool]],  # (arm_id, has_rail) in session order
        session_mode: str,  # "dagger" | "inference"
        run_id: str = "",
        reloader=None,  # None in inference (fixed promoted checkpoint, §8)
        trainer_client=None,  # None in inference
        recorder_fps: int = 25,
        version_label: str | None = None,  # inference: fixed promoted policy id
        coordinator=None,  # OnlineDaggerCoordinator (phase-14); None = plain DAgger / inference
        on_gate_events=None,  # callable(list[dict]) fed every TakeoverGate event (events.gate)
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._version_label = version_label
        self.coordinator = coordinator
        self.on_gate_events = on_gate_events
        self._od_status = None  # cached OnlineDaggerStatus (every OD_STATUS_EVERY_N ticks)
        self.gate = gate
        self.runner = runner
        self.anchor = anchor
        self.arms_meta = list(arms_meta)
        self.session_mode = session_mode
        self.run_id = run_id
        self.reloader = reloader
        self.trainer_client = trainer_client
        self._frame_scale = self.cfg.rate_hz / float(recorder_fps)  # tick -> frame units
        self._prev_ep_state: str | None = None
        self._ep_id: str | None = None  # id of the episode the boundary bookkeeping tracks
        self._boundary_lock = threading.Lock()
        self._saved_boundary = False  # set by recorder callback (save path)
        self._boundary_taken = False  # this episode's boundary already ran (no double fire)
        self._ep_ticks = 0
        self._ep_human_ticks = 0
        self._rates = deque(maxlen=10)  # last-10 episode human-frame fractions
        self.episodes_labeled = 0
        self._nan_strikes = 0
        self._episode_dirty = False  # NaN/anomaly seen -> no mark_good
        self._block_streak = 0
        self.anomalies: list[PolicyAnomalyEvent] = []
        self._last_cf: np.ndarray | None = None  # per-frame counterfactual deposit
        self._last_out_t = -1.0
        self._policy_drove = False
        self._cf_warned = False  # one warning when an abs_ee counterfactual cannot be a delta

    # -- episode-boundary plumbing ---------------------------------------------------
    def on_episode_saved(self, index: int, summary, spool_path: str) -> None:
        """Recorder-thread callback: submit to trainer, arm the boundary. The boundary
        is armed even when the trainer submit raises: the episode IS saved, and a boundary
        that never fires would carry a held takeover into the next rollout."""
        try:
            if summary.n_label_frames > 0:
                self.episodes_labeled += 1
            if self.trainer_client is not None:
                self.trainer_client.submit_episode(spool_path, summary)
        finally:
            with self._boundary_lock:
                self._saved_boundary = True

    def _episode_boundary(self, now: float) -> None:
        """save/discard finished: gate reset + staged swap (NEVER mid-episode). The
        reset's ``events.gate`` payload names the episode that just CLOSED (cached when it
        opened): the recorder has already cleared its id by now, whether the boundary runs
        while its state still reads ``saving`` or after the flip to idle."""
        reset_events = self.gate.reset()
        for ev in reset_events:
            self.anchor.on_gate_event(ev)  # re-anchor rule (12-dagger §6 rule 1)
        self._publish_gate(reset_events, episode_id=self._ep_id)
        if self._ep_ticks > 0:
            self._rates.append(self._ep_human_ticks / self._ep_ticks)
        self._ep_ticks = 0
        self._ep_human_ticks = 0
        if self.reloader is not None:
            if not self._episode_dirty and self._nan_strikes == 0:
                self.reloader.mark_good()  # clean episode on current version (§12)
            swapped = self.reloader.maybe_swap(True, self._engaged_mode())
            if swapped is not None:
                logger.info("policy hot-swap -> v%06d at episode boundary", swapped)
        self._nan_strikes = 0
        self._episode_dirty = False
        self.runner.resume()
        # phase-14 (15-online-dagger §6): the boundary spells "episode_boundary"; a handback
        # inside an episode (_op_takeover_toggle / _op_handback) keeps "handback"
        self.runner.drop_and_requery("episode_boundary")

    # -- gate events -> events.gate (15-online-dagger §3) --------------------------------------
    def _publish_gate(self, events, episode_id: str | None = None) -> None:
        """Hand every new ``GateEvent`` to ``on_gate_events`` as the ``events.gate``
        payload ``{arm_id, mode, seq, source, episode_id}`` — the OPEN episode's id unless
        the caller names one (the boundary names the episode that just closed). The hook
        must be cheap and non-blocking (a worker submit / a queue append): this runs on
        the tick."""
        hook = self.on_gate_events
        if hook is None or not events:
            return
        if episode_id is None:
            episode_id = self._open_episode_id()
        payloads = [
            {
                "arm_id": ev.arm_id,
                "mode": ev.mode.value,
                "seq": int(ev.seq),
                "source": ev.source,
                "episode_id": episode_id,
            }
            for ev in events
        ]
        try:
            hook(payloads)
        except Exception:  # noqa: BLE001 - the bus never breaks the tick
            logger.exception("gate event hook failed")

    def _open_episode_id(self) -> str | None:
        rec = self.recorder
        if rec is None:
            return None
        try:
            return rec.open_episode_id
        except Exception:  # noqa: BLE001 - a stub recorder without the property
            return None

    def _check_boundary(self, now: float) -> None:
        """One boundary per episode: the save callback arms it (``_saved_boundary``); a
        discard is the episode LEAVING ``recording`` / ``saving`` without one — whatever
        follows: ``idle``, or ``returning`` while return-to-start drives the arms back
        (D6, dagger default ON), which the pre-phase-14 ``== "idle"`` test missed, so a
        takeover held at discard time stayed engaged into the next rollout."""
        if self.recorder is None:
            return
        self._settle_boundary(self.recorder.status().state, now)

    def _settle_boundary(self, state: str, now: float) -> None:
        """The boundary bookkeeping against the recorder state just read. Runs on the tick
        at the top of ``_resolve_arms`` AND from ``_op_episode_new`` right before a new
        episode opens: with return-to-start off (or skipped) the recorder flips ``saving ->
        idle`` on its own thread and the operator's ``N`` may be drained in the very next
        tick, BEFORE this check reads the recorder — which then already says ``recording``
        (the new episode) and the previous episode's discard boundary would never run
        (its takeover carried into the new rollout as expert frames, a NaN-strike hold
        persisting, no ``policy_reset(episode_boundary)``). Settling first also keeps the
        save boundary consumed in that tick from marking the NEW episode's boundary as
        taken (its later discard would have been skipped)."""
        saved = False
        with self._boundary_lock:
            if self._saved_boundary:
                saved, self._saved_boundary = True, False
        prev = self._prev_ep_state
        # the episode closed: whatever follows (idle / returning) — or, should a caller
        # other than _op_episode_new ever reopen between two reads, `recording` again
        reopened = prev == "saving" and state == "recording"
        closed = prev in EPISODE_OPEN_STATES and (state not in EPISODE_OPEN_STATES or reopened)
        discarded = closed and not saved and not self._boundary_taken
        if saved or discarded:
            self._boundary_taken = True
            self._episode_boundary(now)
        if state == "recording" and prev != "recording":
            self._boundary_taken = False  # a new episode opened (after settling the last)
            self._ep_id = self._open_episode_id()
        self._prev_ep_state = state

    def _engaged_mode(self) -> ControlMode:
        arm = self.gate.engaged_arm()
        return self.gate.mode(arm) if arm is not None else ControlMode.POLICY

    # -- per-tick resolution (the mux) --------------------------------------------
    def _resolve_arms(
        self, states: dict[str, ArmState], held: frozenset[str], scale: float, now: float
    ) -> tuple[dict[str, np.ndarray | None], CommandSource]:
        self._check_boundary(now)
        gate_events = self.gate.tick(now)
        for ev in gate_events:
            self.anchor.on_gate_event(ev)
        self._publish_gate(gate_events)
        engaged = self.gate.engaged_arm()
        policy_on = self._policy_active()
        self._update_counterfactual(now)
        out: dict[str, np.ndarray | None] = {}
        self._policy_drove = False
        for arm_id in self.session_arms:
            q_last = self._last_cmd[arm_id]
            if self.arm_stopped(arm_id, states[arm_id]):
                out[arm_id] = None  # FAULT / RECOVERING: hold (04-runtime §15)
            elif self.plans.active(arm_id):
                out[arm_id] = self._plan_step(arm_id, q_last)  # finished after the gate
            elif arm_id == engaged:
                self._note_source(arm_id, CommandSource.TELEOP)
                out[arm_id] = self._teleop_step(arm_id, states[arm_id], q_last, held, scale, now)
            elif engaged is not None:
                out[arm_id] = None  # frozen: hold; ordinary policy frame (§2)
            elif policy_on and self._drives(arm_id):
                self._note_source(arm_id, CommandSource.POLICY)  # teleop re-seeds after this
                out[arm_id] = self._policy_step(arm_id, states[arm_id], q_last, now)
                self._policy_drove = True
            elif policy_on:
                # v1.3 (14-dora §6.1): the policy does not drive this arm - hold it exactly
                # like a frozen arm (last command re-sent, gripper untouched); the parked
                # Perception Arm of a Manipulation-Arm-only policy
                out[arm_id] = None
            elif self.session_mode == "dagger" and arm_id == self.active_arm:
                # between episodes: plain teleop for scene staging
                self._note_source(arm_id, CommandSource.TELEOP)
                out[arm_id] = self._teleop_step(arm_id, states[arm_id], q_last, held, scale, now)
            else:
                out[arm_id] = None
        if engaged is not None:
            if self._recording() and self.gate.mode(engaged) is ControlMode.HUMAN:
                self._ep_human_ticks += 1  # human-frame fraction numerator
            source = CommandSource.TAKEOVER
        elif self._policy_drove:
            source = CommandSource.POLICY
        else:
            source = CommandSource.TELEOP
        if self._recording():
            self._ep_ticks += 1
        return out, source

    def _policy_active(self) -> bool:
        if self.runner.paused:
            return False
        if self.session_mode == "inference":
            return True
        return self._recording()

    # -- per-arm driving (v1.3, 2026-09-11; 14-dora §6.1) --------------------------------------
    def _driven_arms(self) -> frozenset[str] | None:
        """The source's driven set (``None`` = every arm); read per tick because an external
        spec may change, and defensively because test stubs predate the member."""
        fn = getattr(self.runner, "driven_arms", None)
        if fn is None:
            return None
        try:
            d = fn()
        except Exception:  # noqa: BLE001 - never on the tick
            return None
        return None if d is None else frozenset(d)

    def _drives(self, arm_id: str) -> bool:
        d = self._driven_arms()
        return d is None or arm_id in d

    def _stale(self, now: float, arm_id: str) -> float:
        """This arm's staleness (a source without the per-arm form answers for the cell)."""
        return staleness_of(self.runner, now, arm_id)

    def _action_space(self) -> str:
        """The layout the source's rows follow (``delta_ee`` | ``abs_ee``)."""
        return action_space_of(self.runner, self.anchor)

    def _driven_finite(self, actions: np.ndarray) -> bool:
        """Finite on every block the policy DRIVES (undriven blocks are NaN by contract and
        never a strike)."""
        d = self._driven_arms()
        if d is None:
            return bool(np.all(np.isfinite(actions)))
        for arm_id, block in split_action(actions, self.arms_meta, self._action_space()).items():
            if arm_id in d and not np.all(np.isfinite(block)):
                return False
        return True

    def _recording(self) -> bool:
        return self.recorder is not None and self.episode_state == "recording"

    # -- policy application ---------------------------------------------------------
    def _update_counterfactual(self, now: float) -> None:
        """Latest policy output -> per-frame counterfactual (NaN when absent).

        The ``policy_action`` column keeps the recorded canonical ``delta_ee`` layout in
        dataset-FRAME units. A ``delta_ee`` row is per policy period, so its delta dims
        scale by ``dt / period * frame_scale``; an ``abs_ee`` row is a waypoint one period
        ahead, so its delta against the current command ``FK(q_last)`` is what the policy
        "would move" per period and scales by the SAME factor (no absolute value is ever
        multiplied). Gripper dims pass through. When the conversion is impossible the
        block is NaN and ONE warning is logged."""
        out, t_out = self.runner.latest()
        if out is None:
            self._last_cf = None
            return
        if t_out != self._last_out_t:
            self._last_out_t = t_out
            if not self._driven_finite(out.actions):
                self._nan_strike(now)
        s = float(self.dt / self.runner.period) * self._frame_scale
        space = self._action_space()
        if space == "abs_ee":
            self._last_cf = self._abs_counterfactual(out.actions, s)
            return
        cf = np.asarray(out.actions, dtype=np.float32).copy()
        i = 0
        for arm_id, has_rail in self.arms_meta:
            n = len(arm_action_names(arm_id, has_rail, "delta_ee"))
            cf[i : i + 6] *= s
            if has_rail:
                cf[i + 7] *= s
            i += n
        self._last_cf = cf

    def _abs_counterfactual(self, actions: np.ndarray, s: float) -> np.ndarray:
        """``abs_ee`` rows -> the ``delta_ee``-width counterfactual: per arm the delta from
        ``FK(q_last)`` to the row's TCP (arm_base frame, space-frame rotvec, rail delta),
        scaled per frame; gripper absolute; NaN where the row (or the conversion) is not."""
        parts: list[np.ndarray] = []
        blocks = split_action(actions, self.arms_meta, "abs_ee")
        kin = self.anchor.kin
        for arm_id, has_rail in self.arms_meta:
            n = len(arm_action_names(arm_id, has_rail, "delta_ee"))
            cf = np.full(n, np.nan, dtype=np.float32)
            parts.append(cf)
            block = blocks.get(arm_id)
            if block is None or block.shape[0] < 10 or not np.all(np.isfinite(block)):
                continue  # NaN by contract (undriven / NaN row): no warning
            q_last = self._last_cmd.get(arm_id)
            base_world = getattr(kin, "base_world", None)
            try:
                if q_last is None or base_world is None:
                    raise ValueError("no command or no base_world on the kinematics seam")
                row = np.asarray(block, dtype=np.float64)
                target_w = se3.pose_mul(
                    base_world(arm_id, q_last), Pose(row[:3], se3.rot6d_to_quat(row[3:9]))
                )
                cur = kin.tcp_world(arm_id, q_last)
                base_inv = se3.quat_conj(kin.base_quat_world(arm_id))
                dp_b = se3.quat_rotate(base_inv, target_w.position - cur.position)
                dq_w = se3.quat_mul(target_w.orientation, se3.quat_conj(cur.orientation))
                dr_b = se3.quat_rotate(base_inv, se3.quat_to_rotvec(dq_w))
            except (ValueError, KeyError, AttributeError, TypeError) as exc:
                if not self._cf_warned:
                    self._cf_warned = True
                    logger.warning(
                        "abs_ee counterfactual for %s cannot be converted to a delta (%s); "
                        "the policy_action column records NaN", arm_id, exc,
                    )
                continue
            cf[:3] = dp_b * s
            cf[3:6] = dr_b * s
            cf[6] = row[9]
            if has_rail and row.shape[0] > 10:
                cf[7] = (row[10] - float(q_last[7])) * s
        return np.concatenate(parts)

    def _policy_step(
        self, arm_id: str, state: ArmState, q_last: np.ndarray, now: float
    ) -> np.ndarray | None:
        """One arm's command from the source's newest row (``dagger.step.policy_step``);
        the gripper dim lands in ``_grip_frac`` and is sent every N ticks."""
        return policy_step(
            runner=self.runner,
            anchor=self.anchor,
            arms_meta=self.arms_meta,
            arm_id=arm_id,
            state=state,
            q_last=q_last,
            now=now,
            dt=self.dt,
            on_gripper=self._policy_gripper,
        )

    def _policy_gripper(self, arm_id: str, frac: float) -> None:
        if arm_id not in self.gripper_arms:
            return
        self._grip_frac[arm_id] = frac
        if self.tick_count % GRIPPER_SEND_EVERY_N_TICKS == 0:
            sender = self._senders.get(arm_id)
            if sender is not None:
                sender.put_gripper(self._grip_frac[arm_id])

    def _nan_strike(self, now: float) -> None:
        self._nan_strikes += 1
        self._episode_dirty = True
        if self._nan_strikes >= NAN_STRIKES_PER_EPISODE and not self.runner.paused:
            self.runner.pause()  # all arms hold until the episode boundary
            self._anomaly("nan", None, now, f"{self._nan_strikes} NaN policy actions")
            if self.reloader is not None:
                self.reloader.rollback()

    def _post_filter(self, dec, now: float) -> None:
        """>=30 consecutive gate-blocked AUTONOMOUS policy ticks -> anomaly."""
        blocked = bool(dec.report.blocked) and self._policy_drove
        self._block_streak = self._block_streak + 1 if blocked else 0
        if self._block_streak == BLOCK_STREAK_TICKS:
            self._anomaly(
                "block_streak", None, now, f"{BLOCK_STREAK_TICKS} consecutive gate blocks"
            )

    def _anomaly(self, kind: str, arm_id: str | None, now: float, detail: str) -> None:
        ev = PolicyAnomalyEvent(kind=kind, arm_id=arm_id, t_mono=now, detail=detail)
        self.anomalies.append(ev)
        self._episode_dirty = True
        logger.warning("policy anomaly: %s (%s) — takeover suggested", kind, detail)

    # -- telemetry + recorder deposits ------------------------------------------------
    def _session_extra(self, now: float) -> dict:
        engaged = self.gate.engaged_arm()
        mode = self._engaged_mode()
        mode_int = {"policy": 0, "human": 1, "takeover_transition": 2}[mode.value]
        if self.reloader is not None:
            version = self.reloader.current_version
        elif hasattr(self.runner, "current_version"):
            version = int(self.runner.current_version())  # PolicySource (14-dora §11.1)
        else:
            version = getattr(self.runner.policy.spec, "version", 0)
        # policy_stale (04-runtime §15; phase-12): the newest output is past its
        # staleness window (policy arms decaying to hold) or the external node is gone
        policy_stale = self.runner.staleness_scale(now) < 1.0 or bool(
            getattr(self.runner, "policy_stale", lambda *_: False)(now)
        )
        extra: dict = {
            "dagger_frame": {
                "control_mode": mode_int,
                "action_source": 3 if mode_int != 0 else 0,
                "policy_action": self._last_cf,
                "policy_version": version,
            }
        }
        if self._version_label is not None:
            version_str = self._version_label
        elif self.reloader is None and hasattr(self.runner, "version_label"):
            version_str = str(self.runner.version_label())  # external: "<policy_id>/v000002"
        else:
            version_str = f"{self.run_id}/v{version:06d}" if self.run_id else str(version)
        if self.session_mode == "inference":
            extra["inference"] = InferenceStatus(
                control_mode=mode,
                engaged_arm=engaged,
                policy_version=version_str,
                policy_stale=policy_stale,
            )
            return extra
        trainer = self.trainer_client.status() if self.trainer_client is not None else None
        staged = self.reloader.staged_version() if self.reloader is not None else None
        ep_rate = self._ep_human_ticks / self._ep_ticks if self._ep_ticks else 0.0
        extra["dagger"] = DaggerStatus(
            control_mode=mode,
            engaged_arm=engaged,
            frozen_arms=self.gate.frozen_arms(),
            policy_version=version_str,
            staged_version=(f"{self.run_id}/v{staged:06d}" if staged is not None else None),
            episodes_labeled=self.episodes_labeled,
            takeover_rate_ep=ep_rate,
            takeover_rate_run=(sum(self._rates) / len(self._rates) if self._rates else 0.0),
            new_label_frames=trainer.new_label_frames if trainer is not None else 0,
            trainer=trainer,
            policy_stale=policy_stale,
            online_dagger=self._online_dagger_status(now),
        )
        return extra

    def _online_dagger_status(self, now: float):
        """``DaggerStatus.online_dagger`` (15-online-dagger §5): the coordinator's view,
        rebuilt at the telemetry rate rather than every tick (it is a pydantic model with
        the verbatim trainer status inside)."""
        coordinator = self.coordinator
        if coordinator is None:
            return None
        if self._od_status is None or self.tick_count % OD_STATUS_EVERY_N == 0:
            try:
                self._od_status = coordinator.status(now)
            except Exception:  # noqa: BLE001 - telemetry never kills the tick
                logger.exception("online_dagger status failed")
        return self._od_status

    # -- command handlers ------------------------------------------------------------
    def _profile_motion_refusal(self) -> str | None:
        """`R` / Go to profile while the POLICY drives (04-runtime §10.5, decided
        2026-09-08): refused. The planned motion would pre-empt the rollout arm by arm
        with the runner still producing (ignored) actions and nothing announced; the
        overview's rule that inference never "returns to initial" on its own stands. The
        operator takes over first (Space) - the human is then the driver, the policy's
        arms hold - and the motion is allowed exactly as in teleop; a hand-back after
        it re-queries the policy from the new posture (no jump: the anchor is measured
        + delta). Between DAgger episodes (recorder idle) the policy is not driving and
        nothing is refused here; while an episode RECORDS the base op's own nack
        ("recording - save or discard first") is the actionable one and wins."""
        if self.episode_state == "recording":
            return None
        if self._policy_active() and self.gate.engaged_arm() is None:
            return POLICY_DRIVING
        return None

    def _op_reset_to_initial(self, cmd: Command) -> CommandResult:
        why = self._profile_motion_refusal()
        if why is not None:
            return CommandResult(cmd.corr_id, False, why)
        return super()._op_reset_to_initial(cmd)

    def _op_goto_profile(self, cmd: Command) -> CommandResult:
        why = self._profile_motion_refusal()
        if why is not None:
            return CommandResult(cmd.corr_id, False, why)
        return super()._op_goto_profile(cmd)

    def _op_takeover_toggle(self, cmd: Command) -> CommandResult:
        if self.active_arm is None:
            return CommandResult(cmd.corr_id, False, "no active arm")
        ev = self.gate.on_toggle(self.active_arm, self._clock())
        if ev is None:
            return CommandResult(cmd.corr_id, False, TAKEOVER_ACTIVE)
        self._apply_gate_event(ev, "takeover_toggle")
        return CommandResult(cmd.corr_id, True, ev.mode.value)

    def _op_takeover(self, cmd: Command) -> CommandResult:
        """Explicit take-over of the active arm (15-online-dagger D3): the same transition
        Space makes from POLICY, idempotent — while the arm is already HUMAN / in
        TRANSITION it acks ``"already taken over"`` and moves nothing."""
        if self.active_arm is None:
            return CommandResult(cmd.corr_id, False, "no active arm")
        if self.gate.mode(self.active_arm) is not ControlMode.POLICY:
            return CommandResult(cmd.corr_id, True, ALREADY_TAKEN_OVER)
        ev = self.gate.on_toggle(self.active_arm, self._clock(), "action")
        if ev is None:
            return CommandResult(cmd.corr_id, False, TAKEOVER_ACTIVE)  # another arm engaged
        self._apply_gate_event(ev, "takeover")
        return CommandResult(cmd.corr_id, True, ev.mode.value)

    def _op_handback(self, cmd: Command) -> CommandResult:
        """Hand the engaged arm back to the policy (15-online-dagger D3): the transition
        Space makes from HUMAN / TRANSITION, idempotent — with no arm engaged it acks
        ``"policy already driving"`` and moves nothing."""
        engaged = self.gate.engaged_arm()
        if engaged is None:
            return CommandResult(cmd.corr_id, True, POLICY_ALREADY_DRIVING)
        ev = self.gate.on_toggle(engaged, self._clock(), "action")
        if ev is None:  # cannot happen for the engaged arm; keep the nack honest
            return CommandResult(cmd.corr_id, False, TAKEOVER_ACTIVE)
        self._apply_gate_event(ev, "handback")
        return CommandResult(cmd.corr_id, True, ev.mode.value)

    def _apply_gate_event(self, ev, why: str) -> None:
        """What every operator-made gate transition does besides the mode flip."""
        self.anchor.on_gate_event(ev)
        if ev.mode is ControlMode.TAKEOVER_TRANSITION:  # human in control NOW
            if self.plans.active_arms:
                self._cancel_plans(why)
            self._teleop_seeded.discard(ev.arm_id)  # re-anchor to measured
        else:  # handback/abort: drop pending chunk, fresh query (12-dagger §6)
            self.runner.drop_and_requery()
        self._publish_gate([ev])

    def _op_episode_new(self, cmd: Command) -> CommandResult:
        """Online DAgger (15-online-dagger §3): the coordinator may refuse a new rollout —
        no trainer attached, waiting for the trainer to report ready, training in
        progress, trainer error — with the exact operator-facing reason as the nack
        detail. The recorder's own refusals (returning / busy) follow."""
        coordinator = self.coordinator
        if coordinator is not None:
            why = coordinator.refuse_episode_new()
            if why is not None:
                return CommandResult(cmd.corr_id, False, why)
        if self.recorder is not None:
            # settle the PREVIOUS episode's boundary before the new one opens (see
            # _settle_boundary): commands drain before the tick's own boundary check
            self._settle_boundary(self.recorder.status().state, self._clock())
        return super()._op_episode_new(cmd)

    def _op_train_now(self, cmd: Command) -> CommandResult:
        """**Train now** (``ActionMsg train_now``; 15-online-dagger §3): ask the trainer to
        train on the rollouts saved so far (``events.train_now``). Refused while an
        episode is open and without a fresh trainer status; a nack in every session that
        is not an Online DAgger one. The coordinator's side effects leave through its
        serial worker, never on this thread."""
        coordinator = self.coordinator
        if coordinator is None:
            return CommandResult(cmd.corr_id, False, NOT_ONLINE_DAGGER)
        # read the recorder NOW, not `self.episode_state` (refreshed at the END of the tick):
        # an episode_new handled earlier in this same tick must count as open
        episode_open = (
            self.recorder is not None and self.recorder.status().state in EPISODE_OPEN_STATES
        )
        ok, detail = coordinator.request_train_now(episode_open=episode_open)
        return CommandResult(cmd.corr_id, ok, detail)

    def _op_switch_arm(self, cmd: Command) -> CommandResult:
        if self.gate.engaged_arm() is not None:
            return CommandResult(cmd.corr_id, False, TAKEOVER_ACTIVE)
        return super()._op_switch_arm(cmd)

    def _op_switch_arm_prev(self, cmd: Command) -> CommandResult:
        if self.gate.engaged_arm() is not None:
            return CommandResult(cmd.corr_id, False, TAKEOVER_ACTIVE)
        return super()._op_switch_arm_prev(cmd)


def make_obs_fn(bus, arms_meta: list[tuple[str, bool]], converter, kin):
    """Policy Observation off the latest snapshot (recorder state layout:
    per arm [q1..q7, gripper, rail?, ee pos, ee quat] in the recording frame).
    ``kin`` must be a runner-thread-private SceneKinematics (one MjData per
    thread)."""

    def obs_fn() -> Observation | None:
        got = bus.snapshot.get()
        if got is None:
            return None
        snap = got[0]
        parts: list[float] = []
        images: dict[str, np.ndarray] = {}
        for arm_id, has_rail in arms_meta:
            st = snap.arms[arm_id]
            q = np.asarray(st.q, dtype=np.float64)
            t_w_b = kin.base_world(arm_id, q)
            ee = converter.convert_pose(arm_id, st.ee_pose, t_w_b)
            parts += [float(x) for x in q[:7]]
            parts.append(float(st.gripper.open_frac))
            if has_rail:
                parts.append(float(q[7]))
            parts += [float(x) for x in ee.position]
            parts += [float(x) for x in ee.orientation]
        return Observation(
            state=np.asarray(parts, dtype=np.float32),
            images=images,
            t_mono=snap.t_mono,
            wallclock_ns=snap.wallclock_ns,
        )

    return obs_fn


class _PolicySessionBase:
    """Owns the policy-side thread stack for one session; torn down in reverse."""

    def __init__(
        self,
        executor: GatedPolicyExecutor,
        runner: PolicySource,
        reloader=None,
        trainer_client=None,
    ) -> None:
        self.executor = executor
        self.runner = runner
        self.reloader = reloader
        self.trainer_client = trainer_client

    def start(self) -> None:
        self.runner.start()
        if self.reloader is not None:
            self.reloader.start()

    def stop(self) -> None:
        if self.trainer_client is not None:
            try:
                self.trainer_client.request_stop()
            except Exception:
                logger.exception("trainer stop failed")
        if self.reloader is not None:
            self.reloader.stop()
        self.runner.stop()


class DaggerSession(_PolicySessionBase):
    """recorder = DaggerRecorderThread; trainer + reloader live (12-dagger §1)."""


class InferenceSession(_PolicySessionBase):
    """recorder=None — NO dataset object, NO trainer process. Space is the
    safety escape through the SAME gate; frames have nowhere to be written."""

    def __init__(self, executor: GatedPolicyExecutor, runner: PolicySource) -> None:
        assert executor.recorder is None and executor.trainer_client is None
        super().__init__(executor, runner, reloader=None, trainer_client=None)


__all__ = [
    "ALREADY_TAKEN_OVER",
    "GatedPolicyExecutor",
    "DaggerSession",
    "InferenceSession",
    "make_obs_fn",
    "NOT_ONLINE_DAGGER",
    "POLICY_ALREADY_DRIVING",
    "TAKEOVER_ACTIVE",
]
