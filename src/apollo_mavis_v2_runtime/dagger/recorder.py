"""DaggerRecorder — annotation-aware episode recording (12-dagger §4-§5).

Layered on the phase-07 ``RecorderThread``: adds the three DAgger columns
(``control_mode`` int8, ``policy_action`` float32 counterfactual with NaN
rows for unqueried ticks, ``policy_version`` int32), flips
``intervention``/``action_source`` per gate state, snapshots ``gate_events``
+ ``EpisodeSummary`` into the episode sidecar, and spools each saved
episode's non-video rows to ``trainer_spool/ep_<episode_id>.parquet`` (the
trainer reads ONLY the spool — never a video, never a ``LeRobotDataset`` —
12-dagger §7; since 2026-09-07 the spool is keyed by the episode id, the
summary carries both the capture-order ``episode_index`` and ``episode_id``).

Phase-14 (15-online-dagger §4): every DAgger frame also carries ``actor``
(int8, ``{0: novice, 1: expert}``, ``actor = 1 iff control_mode != policy``) —
the operator's readable key next to ``control_mode`` / ``intervention``, NOT a
change of the label rule (labels stay ``control_mode == human``). The summary
gains ``n_expert_frames`` / ``n_novice_frames``; with an ``OnlineDaggerCoordinator``
attached the sidecar gains ``episode.json["online_dagger"]`` and a discard fires
``on_episode_discarded(index, episode_id, reason)``.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from apollo_mavis_v2_core.dagger import EpisodeSummary

from ..recorder.thread import RecorderThread, _Capture

logger = logging.getLogger(__name__)

MODE_POLICY, MODE_HUMAN, MODE_TRANSITION = 0, 1, 2
ACTION_SOURCE_POLICY, ACTION_SOURCE_TAKEOVER = 0, 3
CONTROL_MODE_LABELS = {"0": "policy", "1": "human", "2": "takeover_transition"}
ACTOR_NOVICE, ACTOR_EXPERT = 0, 1  # 15-online-dagger §4: actor = 1 iff control_mode != policy
ACTOR_LABELS = {"0": "novice", "1": "expert"}

SPOOL_COLUMNS = (
    "action",
    "observation.state",
    "control_mode",
    "policy_action",
    "policy_version",
    "intervention",
    "action_source",
    "wallclock_ns",
    "actor",  # phase-14 (appended: readers by name are unaffected)
    "action.abs_ee",  # 2026-09-11 (appended LAST, same reason; 10-frames §6)
)


def dagger_features(features: dict[str, dict], run_id: str) -> dict[str, dict]:
    """Base schema + the three DAgger-only features (12-dagger §4, verbatim) + the
    Online DAgger ``actor`` column (15-online-dagger §4, verbatim). ``apollo_schema`` is
    NOT bumped: additive, present in DAgger repos only."""
    out = dict(features)
    n = features["action"]["shape"][0]
    out["control_mode"] = {
        "dtype": "int8",
        "shape": (1,),
        "names": None,
        "info": {"labels": dict(CONTROL_MODE_LABELS)},
    }
    out["actor"] = {
        "dtype": "int8",
        "shape": (1,),
        "names": None,
        "info": {"labels": dict(ACTOR_LABELS), "derived_from": "control_mode != 0"},
    }
    out["policy_action"] = {
        "dtype": "float32",
        "shape": (n,),
        "names": list(features["action"]["names"]),  # identical layout to `action`
        "info": {"counterfactual": True},
    }
    out["policy_version"] = {
        "dtype": "int32",
        "shape": (1,),
        "names": None,
        "info": {"run_id": run_id},
    }
    return out


@dataclass(frozen=True)
class _DaggerCapture(_Capture):
    """One record tick + the gate/policy annotations at the snapshot instant."""

    control_mode: int = MODE_POLICY
    action_source: int = ACTION_SOURCE_POLICY
    policy_action: np.ndarray | None = None  # per-frame units; None -> NaN row
    policy_version: int = 0

    @property
    def actor(self) -> int:
        return ACTOR_EXPERT if self.control_mode != MODE_POLICY else ACTOR_NOVICE


class DaggerRecorderThread(RecorderThread):
    """RecorderThread + DAgger annotations, spool, summary, gate-event sidecar.

    The control loop deposits per-tick annotations in
    ``snap.session_extra["dagger_frame"]``; captures carry them to frames.
    ``on_episode_saved(index, summary, spool_path)`` fires after each save
    (DaggerSession wires trainer submit + boundary swap off it);
    ``on_episode_discarded(index, episode_id, reason)`` after each discard
    (phase-14: the manager publishes ``events.episode_discarded`` off it).
    ``coordinator`` (an ``OnlineDaggerCoordinator``, phase-14) adds the
    ``online_dagger`` block to ``episode.json``; None for a plain DAgger run.
    """

    def __init__(
        self,
        *args: Any,
        gate=None,
        run_id: str = "",
        dataset_root: Path,
        on_episode_saved=None,
        on_episode_discarded=None,
        coordinator=None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.gate = gate
        self.run_id = run_id
        self.spool_dir = Path(dataset_root) / "trainer_spool"
        self.on_episode_saved = on_episode_saved
        self.on_episode_discarded = on_episode_discarded
        self.coordinator = coordinator
        self._rows: list[dict[str, Any]] = []
        self._gate_events_start = 0
        self._last_summary: EpisodeSummary | None = None

    # -- episode ops -------------------------------------------------------------
    def request(self, op: str) -> tuple[bool, str]:
        ok, detail = super().request(op)
        if ok and op == "new":
            self._rows = []
            if self.gate is not None:
                self._gate_events_start = len(self.gate.events)
        return ok, detail

    def _do_discard(self, reason: str = "") -> None:
        # The base class passes a reason (an all-filtered save discards with one); before
        # phase-14 this override took none and a DAgger save with zero kept frames raised
        # TypeError (fixed on the way, 2026-09-08).
        super()._do_discard(reason)
        self._rows = []

    def _episode_discarded(self, index: int | None, episode_id: str | None, reason: str) -> None:
        self._rows = []
        self._last_summary = None
        if self.on_episode_discarded is not None:
            self.on_episode_discarded(index, episode_id, reason)

    # -- capture/frame assembly ------------------------------------------------------
    def _build_capture(self, snap: Any, images: dict[str, np.ndarray], wallclock: int) -> _Capture:
        base = super()._build_capture(snap, images, wallclock)
        ann = snap.session_extra.get("dagger_frame") or {}
        fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(_Capture)}
        pa = ann.get("policy_action")
        return _DaggerCapture(
            **fields,
            control_mode=int(ann.get("control_mode", MODE_POLICY)),
            action_source=int(ann.get("action_source", ACTION_SOURCE_POLICY)),
            policy_action=None if pa is None else np.asarray(pa, dtype=np.float32),
            policy_version=int(ann.get("policy_version", 0)),
        )

    def _frame_from(self, prev: _Capture, cur: _Capture) -> dict[str, Any]:
        frame = super()._frame_from(prev, cur)
        assert isinstance(prev, _DaggerCapture)
        n = frame["action"].shape[0]
        pa = prev.policy_action
        if pa is None or pa.shape[0] != n:
            pa = np.full(n, np.nan, dtype=np.float32)  # unqueried tick: NaN, never faked
        frame["control_mode"] = np.array([prev.control_mode], dtype=np.int8)
        frame["actor"] = np.array([prev.actor], dtype=np.int8)
        frame["policy_action"] = pa.astype(np.float32)
        frame["policy_version"] = np.array([prev.policy_version], dtype=np.int32)
        frame["intervention"] = np.array([prev.control_mode != MODE_POLICY])
        frame["action_source"] = np.array(
            [ACTION_SOURCE_TAKEOVER if prev.control_mode != MODE_POLICY else ACTION_SOURCE_POLICY],
            dtype=np.int8,
        )
        self._rows.append({k: frame[k] for k in SPOOL_COLUMNS})
        return frame

    # -- save hooks -------------------------------------------------------------------
    def _sidecar_extra(self) -> dict[str, Any]:
        events = []
        if self.gate is not None:
            for ev in list(self.gate.events)[self._gate_events_start :]:
                events.append(
                    {
                        "arm_id": ev.arm_id,
                        "mode": ev.mode.value,
                        "t_mono": ev.t_mono,
                        "seq": ev.seq,
                        "source": ev.source,
                    }
                )
        self._last_summary = self._summarize()
        extra: dict[str, Any] = {
            "gate_events": events,
            "episode_summary": dataclasses.asdict(self._last_summary),
        }
        coordinator = self.coordinator
        if coordinator is not None:  # 15-online-dagger §4: episode.json["online_dagger"]
            extra["online_dagger"] = coordinator.sidecar_block(
                {
                    "novice": self._last_summary.n_novice_frames,
                    "expert": self._last_summary.n_expert_frames,
                }
            )
        return extra

    def _episode_saved(self, index: int, episode_id: str) -> None:
        """After ``recorder.save`` published the directory: write the trainer spool, then
        the hook. A spool failure (pyarrow / disk full) is logged and the hook still fires
        with ``spool_path=""`` — the rollout IS on disk, so ``events.episode_saved`` (with
        ``spool_path: null``), the ``session.json`` row and the executor's save boundary
        must all happen; skipping them would let the boundary masquerade as a discard
        and the next kept rollout reuse this one's ``rollouts_saved``."""
        try:
            spool_path = self._write_spool(episode_id)
        except Exception:  # noqa: BLE001 - the spool is a derived artefact
            logger.exception("trainer spool write failed for episode %s", episode_id)
            spool_path = ""
        summary = self._last_summary or self._summarize()
        summary = dataclasses.replace(summary, episode_index=index, episode_id=episode_id)
        self._rows = []
        self._last_summary = None
        if self.on_episode_saved is not None:
            self.on_episode_saved(index, summary, spool_path)

    def _summarize(self) -> EpisodeSummary:
        modes = np.array([int(r["control_mode"][0]) for r in self._rows], dtype=np.int8)
        engaged = modes != MODE_POLICY
        segments = int(np.count_nonzero(np.diff(np.concatenate(([0], engaged.view(np.int8)))) == 1))
        doubts: list[float] = []
        starts = np.flatnonzero(np.diff(np.concatenate(([0], engaged.view(np.int8)))) == 1)
        for s in starts:
            pa = self._rows[int(s)]["policy_action"]
            doubts.append(float(np.nanvar(pa)) if np.any(np.isfinite(pa)) else 0.0)
        n_expert = int(np.count_nonzero(engaged))  # actor == 1 (transition frames too)
        return EpisodeSummary(
            episode_index=-1,  # patched with the real index at save
            n_frames=len(self._rows),
            n_intervention_frames=n_expert,
            n_label_frames=int(np.count_nonzero(modes == MODE_HUMAN)),
            takeover_segments=segments,
            segment_doubts=doubts,
            success=None,
            n_expert_frames=n_expert,
            n_novice_frames=len(self._rows) - n_expert,
        )

    def _write_spool(self, episode_id: str) -> str:
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.spool_dir.mkdir(parents=True, exist_ok=True)
        path = self.spool_dir / f"ep_{episode_id}.parquet"
        cols: dict[str, Any] = {}
        for key in SPOOL_COLUMNS:
            vals = [r[key] for r in self._rows]
            if vals and isinstance(vals[0], np.ndarray) and vals[0].size > 1:
                cols[key] = pa.array([v.tolist() for v in vals])
            else:
                cols[key] = pa.array([_scalar(v) for v in vals])
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(pa.table(cols), tmp)
        tmp.replace(path)
        return str(path)


def _scalar(v: Any):
    if isinstance(v, np.ndarray):
        return v.reshape(-1)[0].item()
    return v


__all__ = [
    "DaggerRecorderThread",
    "dagger_features",
    "CONTROL_MODE_LABELS",
    "MODE_POLICY",
    "MODE_HUMAN",
    "MODE_TRANSITION",
    "ACTION_SOURCE_POLICY",
    "ACTION_SOURCE_TAKEOVER",
    "ACTOR_NOVICE",
    "ACTOR_EXPERT",
    "ACTOR_LABELS",
    "SPOOL_COLUMNS",
]
