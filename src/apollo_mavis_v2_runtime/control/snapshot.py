"""StateSnapshot — published by the control loop every tick (04-runtime §6)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from apollo_mavis_v2_core import ArmState, CollisionReport
from apollo_mavis_v2_core.protocol import EpisodeStatus


@dataclass(frozen=True)
class StateSnapshot:
    """One control-loop tick's published state (consumed at lower rates)."""

    t_mono: float
    wallclock_ns: int
    tick: int
    arms: dict[str, ArmState]  # measured (core schema)
    q_cmd: dict[str, np.ndarray]  # last commanded q (incl. rail)
    active_arm: str | None
    gate: CollisionReport
    clearances: list[tuple[tuple[str, str], float]] = field(default_factory=list)
    gripper_frac: dict[str, float] = field(default_factory=dict)  # commanded open frac
    episode: EpisodeStatus | None = None
    watchdog_tripped: bool = False
    plan_status: dict[str, str] = field(default_factory=dict)  # arm_id -> goto phase
    session_extra: dict[str, object] = field(default_factory=dict)


__all__ = ["StateSnapshot"]
