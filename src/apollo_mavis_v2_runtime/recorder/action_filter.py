"""Idle-frame filter (10-frames §11.4; 04-runtime §10.5; 2026-09-07 addendum).

The operator's pro-dagger heuristic (``_land_chunk`` / ``_anzu_chunk_first_idle``:
a chunk whose first step moved < 1 mm / 1e-3 / 1 mm is a hesitation chunk unless
the gripper toggles inside it) ported to MAVIS's frame stream:

* every candidate frame is judged against the **LAST KEPT** frame (not its
  neighbour — a neighbour comparison would drop a whole slow motion and put the
  accumulated jump on the next kept frame; against the last kept frame a slow
  motion keeps one frame per accumulated epsilon and the delta action stays
  bounded);
* ``idle`` = for EVERY session arm: Chebyshev |Δxyz| of the commanded TCP <
  ``pos_eps_m``, geodesic angle < ``rot_eps_rad``, |Δgripper| < ``gripper_eps_frac``,
  |Δrail| < ``rail_eps_m``;
* an idle frame is still kept when any adjacent pair of raw captures within
  ±``gripper_context_s`` changes the gripper by more than ``gripper_eps_frac``
  (pro-dagger's gripper-toggle exemption) — hence the look-ahead: decisions run
  ``round(gripper_context_s * fps)`` captures behind the live stream and are
  flushed at save / discard;
* the first frame is always kept; DAgger filters human-controlled frames only
  (``action_source`` teleop / takeover); ``enabled: false`` passes everything.

Kept frames stay contiguous in the dataset (``frame_index`` / ``timestamp``); every
gap is recorded as ``[kept_frame_index, n_skipped_before_it]`` for
``episode.json["filter"]["gaps"]``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Any

import numpy as np
from apollo_mavis_v2_core import se3
from apollo_mavis_v2_core.protocol import ActionFilterConfig

HUMAN_ACTION_SOURCES: frozenset[int] = frozenset({1, 3})  # teleop, takeover (10-frames §7.3)


def rotation_angle(q_a: np.ndarray, q_b: np.ndarray) -> float:
    """Geodesic angle between two unit quaternions (rad)."""
    qa = np.asarray(q_a, dtype=np.float64)
    qb = np.asarray(q_b, dtype=np.float64)
    rel = se3.quat_mul(qb, se3.quat_conj(qa))
    return float(np.linalg.norm(se3.quat_to_rotvec(rel)))


class ActionFilter:
    """Decide keep / skip per capture with a look-ahead buffer (see module doc).

    Captures are duck-typed ``RecorderThread._Capture``: ``cmd_pose_b[arm]``
    (Pose), ``grip_cmd[arm]`` (float), ``rail_cmd[arm]`` (float | None) and an
    optional ``action_source`` (DAgger)."""

    def __init__(
        self, cfg: ActionFilterConfig | None, fps: int, arms: Sequence[str] | None = None
    ) -> None:
        self.cfg = cfg if cfg is not None else ActionFilterConfig(enabled=False)
        self.enabled = bool(self.cfg.enabled)
        self.fps = int(fps)
        self.arms = list(arms) if arms is not None else None
        self.window = int(round(self.cfg.gripper_context_s * self.fps)) if self.enabled else 0
        self._buffer: deque[Any] = deque()  # undecided captures, oldest first
        self._history: deque[Any] = deque(maxlen=max(self.window, 1))  # decided raw captures
        self._last_kept: Any | None = None
        self._kept = 0
        self._pending_skips = 0
        self.frames_seen = 0
        self.frames_skipped = 0
        self.gaps: list[list[int]] = []

    # -- lifecycle --------------------------------------------------------------------------
    def reset(self) -> None:
        self._buffer.clear()
        self._history.clear()
        self._last_kept = None
        self._kept = 0
        self._pending_skips = 0
        self.frames_seen = 0
        self.frames_skipped = 0
        self.gaps = []

    @property
    def buffered(self) -> int:
        """Captures waiting for their look-ahead decision."""
        return len(self._buffer)

    # -- streaming ----------------------------------------------------------------------------
    def push(self, capture: Any) -> list[Any]:
        """Feed one capture; returns the captures decided KEPT (oldest first)."""
        self.frames_seen += 1
        if not self.enabled:
            self._kept += 1
            return [capture]
        self._buffer.append(capture)
        out: list[Any] = []
        while len(self._buffer) > self.window:
            out.extend(self._decide_oldest())
        return out

    def flush(self) -> list[Any]:
        """Decide everything still buffered (save / discard): no more look-ahead."""
        out: list[Any] = []
        while self._buffer:
            out.extend(self._decide_oldest())
        return out

    def summary(self) -> dict[str, Any]:
        """The ``episode.json["filter"]`` block (10-frames §9)."""
        cfg = self.cfg
        return {
            "enabled": self.enabled,
            "params": {
                "pos_eps_m": cfg.pos_eps_m,
                "rot_eps_rad": cfg.rot_eps_rad,
                "gripper_eps_frac": cfg.gripper_eps_frac,
                "rail_eps_m": cfg.rail_eps_m,
                "gripper_context_s": cfg.gripper_context_s,
            },
            "frames_seen": self.frames_seen,
            "frames_skipped": self.frames_skipped,
            "gaps": [list(g) for g in self.gaps],
        }

    # -- decisions ----------------------------------------------------------------------------
    def _decide_oldest(self) -> list[Any]:
        c = self._buffer.popleft()
        keep = (
            self._last_kept is None  # the first frame is always kept
            or not self._filterable(c)
            or not self._idle(c, self._last_kept)
            or self._gripper_change_near(c)
        )
        self._history.append(c)
        if keep:
            if self._pending_skips:
                self.gaps.append([self._kept, self._pending_skips])
                self._pending_skips = 0
            self._last_kept = c
            self._kept += 1
            return [c]
        self.frames_skipped += 1
        self._pending_skips += 1
        return []

    @staticmethod
    def _filterable(c: Any) -> bool:
        source = getattr(c, "action_source", None)
        return source is None or int(source) in HUMAN_ACTION_SOURCES

    def _arm_ids(self, c: Any) -> list[str]:
        return self.arms if self.arms is not None else list(c.cmd_pose_b)

    def _idle(self, c: Any, ref: Any) -> bool:
        cfg = self.cfg
        for arm in self._arm_ids(c):
            pc, pr = c.cmd_pose_b[arm], ref.cmd_pose_b[arm]
            dp = np.abs(np.asarray(pc.position) - np.asarray(pr.position))
            if float(np.max(dp)) >= cfg.pos_eps_m:
                return False
            if rotation_angle(pr.orientation, pc.orientation) >= cfg.rot_eps_rad:
                return False
            dg = abs(float(c.grip_cmd.get(arm, 0.0)) - float(ref.grip_cmd.get(arm, 0.0)))
            if dg >= cfg.gripper_eps_frac:
                return False
            rc, rr = c.rail_cmd.get(arm), ref.rail_cmd.get(arm)
            if rc is not None and rr is not None and abs(float(rc) - float(rr)) >= cfg.rail_eps_m:
                return False
        return True

    def _gripper_change_near(self, c: Any) -> bool:
        """Any adjacent raw pair within ±window (history · c · look-ahead) moved a gripper."""
        eps = self.cfg.gripper_eps_frac
        seq = list(self._history) + [c] + list(self._buffer)
        for a, b in zip(seq[:-1], seq[1:], strict=False):
            for arm in self._arm_ids(c):
                if abs(float(b.grip_cmd.get(arm, 0.0)) - float(a.grip_cmd.get(arm, 0.0))) > eps:
                    return True
        return False


__all__ = ["ActionFilter", "HUMAN_ACTION_SOURCES", "rotation_angle"]
