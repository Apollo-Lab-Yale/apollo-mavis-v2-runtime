"""TakeoverGateImpl — Space-toggle takeover state machine (12-dagger §2).

One state machine per participating arm; exactly one arm may be engaged
(HUMAN / TAKEOVER_TRANSITION) at a time. Pure logic, fake-clock friendly:
callers pass ``t_mono`` everywhere. Shared verbatim by DAgger AND inference
sessions (§3) — inference differs only in having no recorder.
"""

from __future__ import annotations

from apollo_xarm7_core.dagger import ControlMode, GateEvent

T_BLEND_MIN_S = 0.2
T_BLEND_MAX_S = 0.5


class TakeoverGateImpl:
    """Implements the core ``TakeoverGate`` Protocol.

    Transition graph (per arm)::

        AUTONOMOUS --toggle--> TAKEOVER_TRANSITION --t>=T_blend--> HUMAN
        TAKEOVER_TRANSITION --toggle--> AUTONOMOUS   # abort
        HUMAN --toggle--> AUTONOMOUS                 # handback
        any --reset()--> AUTONOMOUS                  # episode boundary
    """

    def __init__(self, arm_ids: list[str], t_blend_s: float = 0.3) -> None:
        if not T_BLEND_MIN_S <= t_blend_s <= T_BLEND_MAX_S:
            raise ValueError(
                f"t_blend_s must be in [{T_BLEND_MIN_S}, {T_BLEND_MAX_S}], got {t_blend_s}"
            )
        self.arm_ids = list(arm_ids)
        self.t_blend_s = float(t_blend_s)
        self._mode: dict[str, ControlMode] = dict.fromkeys(arm_ids, ControlMode.POLICY)
        self._transition_t0: dict[str, float] = {}
        self._seq = 0
        self.events: list[GateEvent] = []  # session log; recorder snapshots per episode

    # -- TakeoverGate Protocol -------------------------------------------------
    def mode(self, arm_id: str) -> ControlMode:
        return self._mode[arm_id]

    def engaged_arm(self) -> str | None:
        for arm_id, mode in self._mode.items():
            if mode is not ControlMode.POLICY:
                return arm_id
        return None

    def frozen_arms(self) -> list[str]:
        """Policy arms held while another arm is engaged (§2)."""
        engaged = self.engaged_arm()
        if engaged is None:
            return []
        return [a for a in self.arm_ids if a != engaged]

    def on_toggle(self, arm_id: str, t_mono: float) -> GateEvent | None:
        if arm_id not in self._mode:
            return None
        engaged = self.engaged_arm()
        if engaged is not None and engaged != arm_id:
            return None  # caller Nacks "takeover active"
        mode = self._mode[arm_id]
        if mode is ControlMode.POLICY:
            self._transition_t0[arm_id] = t_mono
            return self._emit(arm_id, ControlMode.TAKEOVER_TRANSITION, t_mono, "keyboard")
        # TRANSITION (abort) or HUMAN (handback) -> AUTONOMOUS
        self._transition_t0.pop(arm_id, None)
        return self._emit(arm_id, ControlMode.POLICY, t_mono, "keyboard")

    def tick(self, t_mono: float) -> list[GateEvent]:
        events: list[GateEvent] = []
        for arm_id, mode in self._mode.items():
            if mode is not ControlMode.TAKEOVER_TRANSITION:
                continue
            t0 = self._transition_t0.get(arm_id)
            if t0 is not None and (t_mono - t0) >= self.t_blend_s:
                self._transition_t0.pop(arm_id, None)
                events.append(self._emit(arm_id, ControlMode.HUMAN, t_mono, "auto_advance"))
        return events

    def reset(self) -> None:
        """Episode boundary: everything back to AUTONOMOUS (no wall clock)."""
        for arm_id, mode in self._mode.items():
            if mode is not ControlMode.POLICY:
                self._emit(arm_id, ControlMode.POLICY, 0.0, "episode_reset")
        self._transition_t0.clear()

    # -- internals ---------------------------------------------------------------
    def _emit(self, arm_id: str, mode: ControlMode, t_mono: float, source: str) -> GateEvent:
        self._mode[arm_id] = mode
        self._seq += 1
        ev = GateEvent(arm_id=arm_id, mode=mode, t_mono=t_mono, seq=self._seq, source=source)
        self.events.append(ev)
        return ev


__all__ = ["TakeoverGateImpl", "T_BLEND_MIN_S", "T_BLEND_MAX_S"]
