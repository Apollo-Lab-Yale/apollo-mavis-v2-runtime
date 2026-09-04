"""GatedPolicyExecutor + DaggerSession + InferenceSession (12-dagger §3/§6/§10).

The executor is the per-arm policy/human mux — a ``ControlLoop`` subclass, so
policy actions ride the SAME 100 Hz chokepoint, per-tick clamp and twin gate
as every other source. DAgger and inference share it verbatim; the ONLY fork
is ``recorder``: ``DaggerSession`` passes a ``DaggerRecorderThread``,
``InferenceSession`` passes ``recorder=None`` — no dataset object exists in
inference mode, so safety-escape frames structurally cannot be recorded.
"""

from __future__ import annotations

import logging
import threading
from collections import deque

import numpy as np
from apollo_mavis_v2_core import ArmState, Command, CommandResult, CommandSource
from apollo_mavis_v2_core.dagger import ControlMode
from apollo_mavis_v2_core.interfaces.policy import Observation
from apollo_mavis_v2_core.protocol import DaggerStatus, InferenceStatus

from ..control.loop import GRIPPER_SEND_EVERY_N_TICKS, ControlLoop
from .policy_runner import (
    BLOCK_STREAK_TICKS,
    NAN_STRIKES_PER_EPISODE,
    ActionAnchor,
    PolicyAnomalyEvent,
    PolicyRunner,
    split_action,
)

logger = logging.getLogger(__name__)

TAKEOVER_ACTIVE = "takeover active"


class GatedPolicyExecutor(ControlLoop):
    """100 Hz control loop where the policy drives AUTONOMOUS arms and the
    human drives the engaged arm; used by BOTH DaggerSession and
    InferenceSession (12-dagger §3)."""

    def __init__(
        self,
        *args,
        gate,
        runner: PolicyRunner,
        anchor: ActionAnchor,
        arms_meta: list[tuple[str, bool]],  # (arm_id, has_rail) in session order
        session_mode: str,  # "dagger" | "inference"
        run_id: str = "",
        reloader=None,  # None in inference (fixed promoted checkpoint, §8)
        trainer_client=None,  # None in inference
        recorder_fps: int = 25,
        version_label: str | None = None,  # inference: fixed promoted policy id
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._version_label = version_label
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
        self._boundary_lock = threading.Lock()
        self._saved_boundary = False  # set by recorder callback (save path)
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

    # -- episode-boundary plumbing ---------------------------------------------------
    def on_episode_saved(self, index: int, summary, spool_path: str) -> None:
        """Recorder-thread callback: submit to trainer, arm the boundary."""
        if summary.n_label_frames > 0:
            self.episodes_labeled += 1
        if self.trainer_client is not None:
            self.trainer_client.submit_episode(spool_path, summary)
        with self._boundary_lock:
            self._saved_boundary = True

    def _episode_boundary(self, now: float) -> None:
        """save/discard finished: gate reset + staged swap (NEVER mid-episode)."""
        self.gate.reset()
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
        self.runner.drop_and_requery()

    def _check_boundary(self, now: float) -> None:
        if self.recorder is None:
            return
        state = self.recorder.status().state
        saved = False
        with self._boundary_lock:
            if self._saved_boundary:
                saved, self._saved_boundary = True, False
        discarded = (
            self._prev_ep_state in ("recording", "saving") and state == "idle" and not saved
        )
        self._prev_ep_state = state
        if saved or discarded:
            self._episode_boundary(now)

    def _engaged_mode(self) -> ControlMode:
        arm = self.gate.engaged_arm()
        return self.gate.mode(arm) if arm is not None else ControlMode.POLICY

    # -- per-tick resolution (the mux) --------------------------------------------
    def _resolve_arms(
        self, states: dict[str, ArmState], held: frozenset[str], scale: float, now: float
    ) -> tuple[dict[str, np.ndarray | None], CommandSource]:
        self._check_boundary(now)
        for ev in self.gate.tick(now):
            self.anchor.on_gate_event(ev)
        engaged = self.gate.engaged_arm()
        policy_on = self._policy_active()
        self._update_counterfactual(now)
        out: dict[str, np.ndarray | None] = {}
        self._policy_drove = False
        for arm_id in self.session_arms:
            q_last = self._last_cmd[arm_id]
            if states[arm_id].error_code != 0:
                out[arm_id] = None
            elif self.plans.active(arm_id):
                q = self.plans.step(arm_id, q_last)
                self._note_source(arm_id, CommandSource.PLANNER)
                if not self.plans.active(arm_id):
                    self._finish_plan(arm_id, ok=True)
                out[arm_id] = q
            elif arm_id == engaged:
                self._note_source(arm_id, CommandSource.TELEOP)
                out[arm_id] = self._teleop_step(arm_id, states[arm_id], q_last, held, scale, now)
            elif engaged is not None:
                out[arm_id] = None  # frozen: hold; ordinary policy frame (§2)
            elif policy_on:
                self._note_source(arm_id, CommandSource.POLICY)  # teleop re-seeds after this
                out[arm_id] = self._policy_step(arm_id, states[arm_id], q_last, now)
                self._policy_drove = True
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

    def _recording(self) -> bool:
        return self.recorder is not None and self.episode_state == "recording"

    # -- policy application ---------------------------------------------------------
    def _update_counterfactual(self, now: float) -> None:
        """Latest policy output -> per-frame counterfactual (NaN when absent)."""
        out, t_out = self.runner.latest()
        if out is None:
            self._last_cf = None
            return
        if t_out != self._last_out_t:
            self._last_out_t = t_out
            if not np.all(np.isfinite(out.actions)):
                self._nan_strike(now)
        cf = np.asarray(out.actions, dtype=np.float32).copy()
        # delta dims scale from policy-period to dataset-frame units; the
        # absolute gripper dim passes through (layout: 6 delta + grip [+ rail]).
        s = float(self.dt / self.runner.period) * self._frame_scale
        i = 0
        for _, has_rail in self.arms_meta:
            n = 8 if has_rail else 7
            cf[i : i + 6] *= s
            if has_rail:
                cf[i + 7] *= s
            i += n
        self._last_cf = cf

    def _policy_step(
        self, arm_id: str, state: ArmState, q_last: np.ndarray, now: float
    ) -> np.ndarray | None:
        out, _ = self.runner.latest()
        if out is None or not np.all(np.isfinite(out.actions)):
            return None  # hold; counterfactual row records NaN (12-dagger §12)
        stale = self.runner.staleness_scale(now)
        if stale <= 0.0:
            return None  # > 5 stale periods: hold
        blocks = split_action(out.actions, self.arms_meta)
        block = blocks.get(arm_id)
        if block is None:
            return None
        tick_scale = (self.dt / self.runner.period) * stale
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
        grip = float(block[6])
        if np.isfinite(grip) and arm_id in self.gripper_arms:
            self._grip_frac[arm_id] = float(np.clip(grip, 0.0, 1.0))
            if self.tick_count % GRIPPER_SEND_EVERY_N_TICKS == 0:
                sender = self._senders.get(arm_id)
                if sender is not None:
                    sender.put_gripper(self._grip_frac[arm_id])
        return q

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
            self._anomaly("block_streak", None, now,
                          f"{BLOCK_STREAK_TICKS} consecutive gate blocks")

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
        version = self.reloader.current_version if self.reloader is not None else (
            getattr(self.runner.policy.spec, "version", 0))
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
        else:
            version_str = f"{self.run_id}/v{version:06d}" if self.run_id else str(version)
        if self.session_mode == "inference":
            extra["inference"] = InferenceStatus(
                control_mode=mode, engaged_arm=engaged, policy_version=version_str,
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
        )
        return extra

    # -- command handlers ------------------------------------------------------------
    def _op_takeover_toggle(self, cmd: Command) -> CommandResult:
        if self.active_arm is None:
            return CommandResult(cmd.corr_id, False, "no active arm")
        now = self._clock()
        ev = self.gate.on_toggle(self.active_arm, now)
        if ev is None:
            return CommandResult(cmd.corr_id, False, TAKEOVER_ACTIVE)
        self.anchor.on_gate_event(ev)
        if ev.mode is ControlMode.TAKEOVER_TRANSITION:  # human in control NOW
            if self.plans.active_arms:
                self._cancel_plans("takeover_toggle")
            self._teleop_seeded.discard(self.active_arm)  # re-anchor to measured
        else:  # handback/abort: drop pending chunk, fresh query (12-dagger §6)
            self.runner.drop_and_requery()
        return CommandResult(cmd.corr_id, True, ev.mode.value)

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

    def __init__(self, executor: GatedPolicyExecutor, runner: PolicyRunner,
                 reloader=None, trainer_client=None) -> None:
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

    def __init__(self, executor: GatedPolicyExecutor, runner: PolicyRunner) -> None:
        assert executor.recorder is None and executor.trainer_client is None
        super().__init__(executor, runner, reloader=None, trainer_client=None)


__all__ = [
    "GatedPolicyExecutor",
    "DaggerSession",
    "InferenceSession",
    "make_obs_fn",
    "TAKEOVER_ACTIVE",
]
