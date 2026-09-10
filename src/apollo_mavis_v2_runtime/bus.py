"""Command bus re-exports + named-slot wiring (04-runtime §4; core §15).

Primitives are DEFINED in ``apollo_mavis_v2_core.bus``; this module re-exports
them and wires the named depth-1 slots the runtime threads share.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from apollo_mavis_v2_core import Command, CommandBus, CommandResult, HeldState, LatestSlot

if TYPE_CHECKING:
    import numpy as np

    from .control.snapshot import StateSnapshot
    from .devices.microphone import MicFrame
    from .devices.tracker import TrackerSample


class RuntimeBus:
    """The one command bus + every named latest-value slot (04-runtime §4).

    Slots: ``held_keys`` (WS -> control), ``q_cmd[arm_id]`` (control ->
    senders), ``snapshot`` (control -> telemetry/recorder/twin-sync),
    ``policy_action`` (policy runner -> control, phase-08), ``tracker``
    (TrackerReader -> control/telemetry, 13-tracker §2), ``microphone``
    (MicrophoneReader -> telemetry, phase-11) and ``encoded[stream_id]``
    (encoders -> WS/MJPEG).
    """

    def __init__(self) -> None:
        self.commands = CommandBus()
        self.held_keys: LatestSlot[HeldState] = LatestSlot()
        self.q_cmd: dict[str, LatestSlot[np.ndarray]] = {}
        self.snapshot: LatestSlot[StateSnapshot] = LatestSlot()
        self.policy_action: LatestSlot[Any] = LatestSlot()
        self.tracker: LatestSlot[TrackerSample] = LatestSlot()
        self.microphone: LatestSlot[MicFrame] = LatestSlot()
        self.encoded: dict[str, LatestSlot[bytes]] = {}

    def arm_slot(self, arm_id: str) -> LatestSlot:
        return self.q_cmd.setdefault(arm_id, LatestSlot())

    def encoded_slot(self, stream_id: str) -> LatestSlot[bytes]:
        return self.encoded.setdefault(stream_id, LatestSlot())


__all__ = [
    "Command",
    "CommandBus",
    "CommandResult",
    "LatestSlot",
    "HeldState",
    "RuntimeBus",
]
