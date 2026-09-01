"""Label extraction + 50/50 sampling (12-dagger §5/§7).

Labels follow HG-DAgger Eq. 2: only ``control_mode == 1`` (human) frames;
``takeover_transition`` frames are excluded. Data arrives via the recorder's
``trainer_spool/ep_*.parquet`` copies (the live LeRobot shard has no footer);
the trainer never touches the writer API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODE_HUMAN = 1


def read_spool(path: str | Path) -> dict[str, np.ndarray]:
    """Non-video episode rows -> numpy columns (state/action/control_mode)."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    cols = table.to_pydict()
    return {
        "state": np.asarray(cols["observation.state"], dtype=np.float32),
        "action": np.asarray(cols["action"], dtype=np.float32),
        "control_mode": np.asarray(cols["control_mode"], dtype=np.int8),
    }


def label_mask(control_mode: np.ndarray) -> np.ndarray:
    """HG-DAgger Eq. 2: human frames only (transition excluded)."""
    return np.asarray(control_mode) == MODE_HUMAN


@dataclass
class LabelIndex:
    """In-RAM aggregate of supervised pairs + the new-since-checkpoint slice."""

    states: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))
    actions: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))
    episodes: list[int] = field(default_factory=list)
    new_since_checkpoint: int = 0  # trailing rows added since the last checkpoint

    def add_seed(self, states: np.ndarray, actions: np.ndarray) -> int:
        """Seed BC frames: ALL rows are labels (pure demos); never 'new'."""
        self._append(states, actions)
        return int(states.shape[0])

    def add_episode(self, episode_index: int, data: dict[str, np.ndarray]) -> int:
        mask = label_mask(data["control_mode"])
        n = int(np.count_nonzero(mask))
        if n:
            self._append(data["state"][mask], data["action"][mask])
            self.new_since_checkpoint += n
        self.episodes.append(int(episode_index))
        return n

    def _append(self, states: np.ndarray, actions: np.ndarray) -> None:
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if self.states.size == 0:
            self.states, self.actions = states.copy(), actions.copy()
        else:
            self.states = np.concatenate([self.states, states])
            self.actions = np.concatenate([self.actions, actions])

    def mark_checkpoint(self) -> None:
        self.new_since_checkpoint = 0

    @property
    def n_frames(self) -> int:
        return int(self.states.shape[0])


def build_label_index(spool_paths: list[str | Path]) -> LabelIndex:
    idx = LabelIndex()
    for i, p in enumerate(sorted(str(x) for x in spool_paths)):
        idx.add_episode(i, read_spool(p))
    return idx


class FiftyFiftySampler:
    """Each batch: half labels since the last checkpoint, half full aggregate
    (seed BC ∪ all human labels) — lerobot OnlineOfflineMixer, ratio 0.5."""

    def __init__(self, index: LabelIndex, rng: np.random.Generator | None = None) -> None:
        self.index = index
        self.rng = rng or np.random.default_rng(0)

    def next(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        n = self.index.n_frames
        if n == 0:
            raise RuntimeError("no label frames to sample")
        n_new = min(self.index.new_since_checkpoint, n)
        half = batch_size // 2
        agg_idx = self.rng.integers(0, n, size=batch_size - (half if n_new else 0))
        parts = [agg_idx]
        if n_new:
            parts.append(self.rng.integers(n - n_new, n, size=half))
        idx = np.concatenate(parts)
        return self.index.states[idx], self.index.actions[idx]


__all__ = ["read_spool", "label_mask", "LabelIndex", "build_label_index",
           "FiftyFiftySampler", "MODE_HUMAN"]
