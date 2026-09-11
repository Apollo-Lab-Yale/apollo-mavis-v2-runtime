"""``ReplayActionSource`` — a recorded action column as a ``PolicySource`` (2026-09-11).

Episode playback with ``source: delta_ee | abs_ee`` (04-runtime §10.8) replays the
``action`` / ``action.abs_ee`` column of a saved episode through the SAME per-tick
path a policy drives: ``dagger/step.policy_step`` + ``ActionAnchor``. This class is the
"policy": it implements :class:`~.policy_source.PolicySource` over an
:class:`~..recorder.playback.EpisodeTrajectory`, so the base ``ControlLoop``'s replay
slot, a ``GatedPolicyExecutor`` in a test rig, and later the dora interface's layout
all consume it unchanged.

Semantics:

* ``period = 1 / fps`` and row ``k`` is current from ``t0 + k * period`` on
  (``latest()`` never skips ahead of the clock and never repeats a row as new: the
  executor's ``dt / period`` scaling of a delta row then reproduces exactly the recorded
  per-frame increment over one period).
* Rows are laid out in the SESSION's arm order with ``arm_action_names`` widths; a
  session arm the episode does not name gets a NaN block (= hold, 14-dora §6.1) and is
  not in :meth:`driven_arms`. An episode arm recorded without a track on a session arm
  that has one gets ``rail.dpos 0`` (delta) / ``rail.pos = rail_hold[arm]`` (abs), so its
  carriage stays put - the rule ``recorder/playback.resample`` applies to the state
  replay.
* ``staleness_scale`` is 1.0 while a row is due and 0.0 once the LAST row has been
  current for a full period: the executor then holds (``policy_step`` returns None) and
  the loop's replay slot finishes. There is no decay - the recording ends, it does not
  go stale.

torch- and dora-free.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence

import numpy as np
from apollo_mavis_v2_core.interfaces.policy import PolicyOutput, PolicySpec

from ..recorder.features import arm_action_names
from ..recorder.playback import ACTION_COLUMNS, EpisodeTrajectory, PlaybackError


class ReplayActionSource:
    """A saved episode's action column, served one row per recorded frame."""

    def __init__(
        self,
        traj: EpisodeTrajectory,
        source: str,
        arms_meta: Sequence[tuple[str, bool]],
        *,
        clock: Callable[[], float] = time.monotonic,
        rail_hold: dict[str, float] | None = None,
    ) -> None:
        if source not in ACTION_COLUMNS:
            raise PlaybackError(f"{source!r} is not an action replay source")
        column = traj.action_column(source)  # PlaybackError when the episode lacks it
        self.traj = traj
        self.source = source
        self.arms_meta = [(str(a), bool(r)) for a, r in arms_meta]
        self._clock = clock
        self.period = 1.0 / float(traj.fps) if traj.fps > 0 else 1.0
        session_arms = [a for a, _ in self.arms_meta]
        driven = [b.arm_id for b in column.blocks if b.arm_id in session_arms]
        if not driven:
            raise PlaybackError("the episode names none of this session's arms")
        self._driven = frozenset(driven)
        # Session layout: each session arm's block by NAME; missing names filled per the
        # rules in the module docstring. Built ONCE - latest() is a row slice on the tick.
        names: list[str] = []
        n = int(column.rows.shape[0])
        rows = np.full((n, 0), np.nan, dtype=np.float32)
        parts: list[np.ndarray] = []
        col_index = {name: i for i, name in enumerate(column.names)}
        for arm_id, has_rail in self.arms_meta:
            block_names = arm_action_names(arm_id, has_rail, source)
            names += block_names
            block = np.full((n, len(block_names)), np.nan, dtype=np.float32)
            if arm_id in self._driven:
                for j, name in enumerate(block_names):
                    i = col_index.get(name)
                    if i is not None:
                        block[:, j] = column.rows[:, i]
                    elif name.endswith("_rail.dpos"):
                        block[:, j] = 0.0  # recorded without a track: the carriage holds
                    elif name.endswith("_rail.pos"):
                        hold = (rail_hold or {}).get(arm_id)
                        if hold is None:
                            raise PlaybackError(
                                f"the episode has no rail column for {arm_id!r} and no "
                                "carriage position to hold was given"
                            )
                        block[:, j] = float(hold)
                    else:  # pragma: no cover - parse_action_names guarantees whole blocks
                        raise PlaybackError(f"the episode's {column.column} lacks {name!r}")
            parts.append(block)
        rows = np.concatenate(parts, axis=1) if parts else rows
        self._rows = rows
        self._n = n
        first = next(a for a in session_arms if a in self._driven)
        self.spec = PolicySpec(
            action_space=source,  # type: ignore[arg-type]
            action_frame=column.frames_map.get(first, f"arm_base:{first}"),
            action_names=names,
            state_names=[],
            camera_keys=[],
            version=0,
        )
        self._t0: float | None = None
        self._paused = False
        self._frame_index = 0

    # -- PolicySource ----------------------------------------------------------------
    def start(self) -> None:
        """Stamp ``t0``: row 0 is current from now on."""
        self._t0 = float(self._clock())
        self._frame_index = 0

    def stop(self) -> None:
        self._t0 = None

    def _k(self, now: float) -> int:
        """The UNCLAMPED row index due at ``now`` (>= ``frames`` once the episode is over)."""
        if self._t0 is None:
            return -1
        return int(math.floor((now - self._t0) / self.period + 1e-9))

    def latest(self) -> tuple[PolicyOutput | None, float]:
        if self._t0 is None:
            return None, 0.0
        k = min(self._n - 1, max(0, self._k(self._clock())))
        self._frame_index = k
        t_row = self._t0 + k * self.period
        out = PolicyOutput(
            actions=self._rows[k],
            version=0,
            t_mono=t_row,
            chunk_remaining=self._n - 1 - k,
        )
        return out, t_row

    def staleness_scale(self, now: float, arm_id: str | None = None) -> float:
        """1.0 while a row is due; 0.0 once the last row has been current for a period."""
        return 0.0 if self.finished_at(now) else 1.0

    def driven_arms(self) -> frozenset[str]:
        return self._driven

    def drop_and_requery(self, reason: str = "handback") -> None:
        """A recording has nothing to re-query; no-op."""

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    @property
    def paused(self) -> bool:
        return self._paused

    def version_label(self) -> str:
        return f"replay/{self.traj.episode_id}"

    def current_version(self) -> int:
        return 0

    def policy_stale(self, now: float) -> bool:
        return False

    # -- progress ------------------------------------------------------------------------
    @property
    def frames(self) -> int:
        return self._n

    @property
    def frame_index(self) -> int:
        """The row :meth:`latest` served last (0 before the first call)."""
        return self._frame_index

    @property
    def started(self) -> bool:
        return self._t0 is not None

    def finished_at(self, now: float) -> bool:
        """True once every row has been current for a full period (the executor holds)."""
        return self._t0 is not None and self._k(now) >= self._n

    @property
    def finished(self) -> bool:
        return self.finished_at(float(self._clock()))


__all__ = ["ReplayActionSource"]
