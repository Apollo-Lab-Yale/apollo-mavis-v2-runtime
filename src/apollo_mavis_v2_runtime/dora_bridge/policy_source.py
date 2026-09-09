"""External policy over dora: ``ExternalPolicyHub`` + ``ExternalPolicySource`` (14-dora §5, §6).

``ExternalPolicyHub`` lives for the process (owned by the runtime's dora
wiring): it registers the ``policy_action`` / ``policy_spec`` / ``policy_status``
/ ``policy_trainer_status`` input handlers with the bridge, caches the newest
``PolicySpecAnnounce`` (the 1 Hz heartbeat; ``policy_attached`` = age <=
``dora.policy.spec_stale_s``) and the newest ``TrainerStatusAnnounce`` (phase-14,
15-online-dagger §6: the trainer role's 1 Hz status; ``trainer_capable`` reads the
``online_dagger`` capability off the spec), and forwards actions to the
``ExternalPolicySource`` of the running external session — or drops them
(``dropped_inputs++``) when no such session exists. An Online DAgger session attaches
its ``OnlineDaggerCoordinator`` as the ``trainer_sink``: every trainer status and every
announced policy version is handed to it on the bus thread.

``ExternalPolicySource`` implements ``dagger.policy_source.PolicySource`` for
one session. Everything the bus thread writes is a single reference swap under
a lock; the control thread only reads ``latest()`` / ``staleness_scale()`` —
no dora call ever runs on the tick (§2.5). Rules (§6.2/§6.3):

- validation on receipt: required metadata, ``session_id`` == this session,
  ``action_dim == len(action_names)``, ``chunk_len * action_dim == len(payload)``,
  ``observation_id > watermark``, observation age <= ``max_obs_age_s`` (else
  ``actions_late++``); NaN rows pass through (the executor's 3-strike guard);
- chunks: row 0 is used immediately, one row per ``chunk_dt_s`` thereafter until
  a newer action arrives; ``latest()`` returns the current row rescaled from
  per-``chunk_dt_s`` to per-``period`` units (gripper dims absolute), so the
  executor's ``dt / period`` scaling yields ``dt / chunk_dt_s``;
- ``staleness_scale`` keeps the in-process rule verbatim (``period + 0.05`` then
  linear to 0 over 5 periods; 0.45 s at 15 Hz); the clock is the runtime's;
- ``drop_and_requery()`` publishes ``policy_reset{after_observation_id}`` and
  sets the watermark; ``pause()`` holds (NaN 3-strike) until ``resume()``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
from apollo_mavis_v2_core.interfaces.policy import PolicyOutput, PolicySpec
from apollo_mavis_v2_core.protocol import external as ext
from apollo_mavis_v2_core.protocol.external import (
    PolicyResetMsg,
    PolicySpecAnnounce,
    TrainerStatusAnnounce,
)

from ..config import DoraPolicyConfig
from ..dagger.policy_runner import POLICY_TIMEOUT_MARGIN_S, STALE_DECAY_PERIODS
from . import codec
from .bridge import DoraBridge

logger = logging.getLogger(__name__)

ONLINE_DAGGER_CAPABILITY = "online_dagger"  # PolicySpecAnnounce.capabilities (15-online-dagger §6)


def spec_from_announce(ann: PolicySpecAnnounce) -> PolicySpec:
    s = ann.spec
    return PolicySpec(
        action_space=s.action_space,
        action_frame=s.action_frame,
        action_names=list(s.action_names),
        state_names=list(s.state_names),
        camera_keys=list(s.camera_keys),
        version=int(ann.policy_version),
    )


class ExternalPolicyHub:
    """Process-lifetime cache of the policy node's spec heartbeat + action router."""

    def __init__(
        self, bridge: DoraBridge, cfg: DoraPolicyConfig, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.bridge = bridge
        self.cfg = cfg
        self._clock = clock
        self._lock = threading.Lock()
        self._spec: PolicySpecAnnounce | None = None
        self._spec_t: float = -1e9
        self._status_text: str = ""
        self._source: ExternalPolicySource | None = None
        self.actions_seen = 0
        # phase-14 (15-online-dagger §6): the trainer role's status + the session's coordinator
        self._trainer: TrainerStatusAnnounce | None = None
        self._trainer_t: float = -1e9
        self._trainer_sink: Any = None  # OnlineDaggerCoordinator of the running session
        self.trainer_statuses_seen = 0
        bridge.register_input(ext.IN_POLICY_SPEC, self._on_spec)
        bridge.register_input(ext.IN_POLICY_ACTION, self._on_action)
        bridge.register_input(ext.IN_POLICY_STATUS, self._on_status)
        bridge.register_input(ext.IN_POLICY_TRAINER_STATUS, self._on_trainer_status)
        bridge.on_state_change(self._on_bridge_state)

    # -- spec cache -----------------------------------------------------------------------------
    def spec(self, now: float | None = None) -> PolicySpecAnnounce | None:
        """The cached spec if it is fresh (<= ``spec_stale_s``), else ``None``."""
        now = self._clock() if now is None else now
        with self._lock:
            if self._spec is None or now - self._spec_t > self.cfg.spec_stale_s:
                return None
            return self._spec

    def spec_age_s(self, now: float | None = None) -> float | None:
        now = self._clock() if now is None else now
        with self._lock:
            return None if self._spec is None else max(0.0, now - self._spec_t)

    @property
    def policy_attached(self) -> bool:
        return self.spec() is not None and self.bridge.attached

    def status_text(self) -> str:
        return self._status_text

    # -- trainer role (phase-14; 15-online-dagger §6) --------------------------------------------
    def trainer_capable(self, now: float | None = None) -> bool:
        """The FRESH spec lists the ``online_dagger`` capability (a trainer-capable node)."""
        ann = self.spec(now)
        return ann is not None and ONLINE_DAGGER_CAPABILITY in ann.capabilities

    def trainer_status(self, now: float | None = None) -> TrainerStatusAnnounce | None:
        """The newest ``TrainerStatusAnnounce`` if fresh (<= ``spec_stale_s``), else None."""
        now = self._clock() if now is None else now
        with self._lock:
            if self._trainer is None or now - self._trainer_t > self.cfg.spec_stale_s:
                return None
            return self._trainer

    def trainer_age_s(self, now: float | None = None) -> float | None:
        now = self._clock() if now is None else now
        with self._lock:
            return None if self._trainer is None else max(0.0, now - self._trainer_t)

    def attach_trainer_sink(self, sink: Any) -> None:
        """An Online DAgger session's coordinator: receives ``on_trainer_status(msg, t_recv)``
        and ``on_spec_version(version)`` on the bus thread. The last cached status is
        replayed so a coordinator built after the trainer's first heartbeat sees it."""
        with self._lock:
            self._trainer_sink = sink
            last, t = self._trainer, self._trainer_t
            ann = self._spec
        if sink is not None:
            if ann is not None:
                sink.on_spec_version(int(ann.policy_version))
            if last is not None and self._clock() - t <= self.cfg.spec_stale_s:
                sink.on_trainer_status(last, t)

    def detach_trainer_sink(self, sink: Any = None) -> None:
        with self._lock:
            if sink is None or self._trainer_sink is sink:
                self._trainer_sink = None

    @property
    def trainer_sink(self) -> Any:
        return self._trainer_sink

    # -- session source -------------------------------------------------------------------------
    def attach_source(self, source: ExternalPolicySource) -> None:
        with self._lock:
            self._source = source

    def detach_source(self, source: ExternalPolicySource | None = None) -> None:
        with self._lock:
            if source is None or self._source is source:
                self._source = None

    @property
    def source(self) -> ExternalPolicySource | None:
        return self._source

    # -- handlers (bus thread) -------------------------------------------------------------------
    def _on_bridge_state(self, state: str) -> None:
        if state != "attached":
            with self._lock:
                self._spec_t = -1e9  # a detached bridge holds: the spec must be re-heard
                self._trainer_t = -1e9

    def _on_spec(self, ev: dict[str, Any]) -> None:
        if ev.get("type") != "INPUT":
            return
        try:
            ann = codec.decode_json(ev.get("value"), PolicySpecAnnounce)
        except Exception as exc:  # noqa: BLE001 - malformed JSON is a dropped input
            self.bridge.count_drop(f"policy_spec: {type(exc).__name__}: {str(exc)[:120]}")
            return
        if int(ann.mavis_schema) != ext.MAVIS_SCHEMA:
            self.bridge.count_drop(f"policy_spec: mavis_schema {ann.mavis_schema}")
            return
        with self._lock:
            self._spec = ann
            self._spec_t = self._clock()
            src = self._source
            sink = self._trainer_sink
        if src is not None:
            src.on_spec(ann)
        if sink is not None:
            try:
                sink.on_spec_version(int(ann.policy_version))
            except Exception:  # noqa: BLE001 - the coordinator never breaks the bus thread
                logger.exception("trainer sink on_spec_version failed")

    def _on_trainer_status(self, ev: dict[str, Any]) -> None:
        """``policy_trainer_status`` (15-online-dagger §6): validate, cache, hand to the sink."""
        if ev.get("type") != "INPUT":
            return
        try:
            msg = codec.decode_json(ev.get("value"), TrainerStatusAnnounce)
        except Exception as exc:  # noqa: BLE001 - malformed JSON is a dropped input
            self.bridge.count_drop(f"policy_trainer_status: {type(exc).__name__}: {str(exc)[:120]}")
            return
        if int(msg.mavis_schema) != ext.MAVIS_SCHEMA:
            self.bridge.count_drop(f"policy_trainer_status: mavis_schema {msg.mavis_schema}")
            return
        now = self._clock()
        with self._lock:
            self._trainer = msg
            self._trainer_t = now
            self.trainer_statuses_seen += 1
            sink = self._trainer_sink
        if sink is not None:
            try:
                sink.on_trainer_status(msg, now)
            except Exception:  # noqa: BLE001
                logger.exception("trainer sink on_trainer_status failed")

    def _on_status(self, ev: dict[str, Any]) -> None:
        if ev.get("type") != "INPUT":
            return
        try:
            self._status_text = codec.decode_json_text(ev.get("value"))[:400]
        except Exception:  # noqa: BLE001
            self._status_text = "<unreadable status>"
        logger.info("policy_status: %s", self._status_text)

    def _on_action(self, ev: dict[str, Any]) -> None:
        if ev.get("type") != "INPUT":
            return
        self.actions_seen += 1
        src = self._source
        if src is None:
            self.bridge.count_drop("policy_action without an external session")
            return
        src.on_action(ev, self._clock())


class ExternalPolicySource:
    """``PolicySource`` for one ``policy_source: external`` session (see module doc)."""

    def __init__(
        self,
        hub: ExternalPolicyHub,
        publisher: Any,  # SnapshotPublisher: observation_id / observation_t_mono / publish_event
        *,
        session_id: str,
        spec: PolicySpec,
        policy_id: str,
        arms_meta: Sequence[tuple[str, bool]],
        rate_hz: float,
        chunk_dt_s: float | None,
        cfg: DoraPolicyConfig,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.hub = hub
        self.publisher = publisher
        self.session_id = session_id
        self.spec = spec
        self.policy = SimpleNamespace(spec=spec)  # executor reads runner.policy.spec.version
        self.policy_id = policy_id
        self.arms_meta = list(arms_meta)
        self.period = 1.0 / float(rate_hz)
        self.default_chunk_dt = float(chunk_dt_s) if chunk_dt_s else self.period
        self.cfg = cfg
        self._clock = clock
        self._lock = threading.Lock()
        self._rows: np.ndarray | None = None  # (K, D) per-chunk_dt units
        self._rows_t: float = -1e9  # receive time of the current chunk
        self._chunk_dt: float = self.default_chunk_dt
        self._version: int = int(spec.version)
        self._paused = False
        self._watermark = 0
        self._last_action_t: float | None = None
        self.actions_late = 0
        self.actions_applied = 0
        self.version_changes = 0
        self._started = False
        self._delta_scale = self._build_delta_scale()

    # -- lifecycle ---------------------------------------------------------------------------------
    def start(self) -> None:
        self._started = True
        self.hub.attach_source(self)
        # observation ids restart at 1 for this session (SnapshotPublisher.session_started):
        # the session_start watermark is 0, whatever the publisher counted before
        with self._lock:
            self._rows = None
            self._rows_t = -1e9
            self._watermark = 0
        self._publish_reset("session_start")

    def stop(self) -> None:
        if self._started:
            self._publish_reset("session_stop")
        self.hub.detach_source(self)
        self._started = False

    # -- the PolicySource surface (control thread) ----------------------------------------------
    def latest(self) -> tuple[PolicyOutput | None, float]:
        with self._lock:
            rows, t0, dt, version = self._rows, self._rows_t, self._chunk_dt, self._version
        if rows is None or self._paused:
            return None, -1e9
        now = self._clock()
        k = rows.shape[0]
        idx = 0 if k == 1 else int(min(k - 1, max(0, (now - t0) // dt)))
        active_t = t0 + idx * dt
        row = rows[idx] * self._delta_scale
        return (
            PolicyOutput(
                actions=row.astype(np.float32),
                version=version,
                t_mono=active_t,
                chunk_remaining=k - 1 - idx,
            ),
            active_t,
        )

    def staleness_scale(self, now: float) -> float:
        _, t = self.latest()
        age = now - t
        timeout = self.period + POLICY_TIMEOUT_MARGIN_S
        if age <= timeout:
            return 1.0
        return float(np.clip(1.0 - (age - timeout) / (STALE_DECAY_PERIODS * self.period), 0.0, 1.0))

    def drop_and_requery(self, reason: str = "handback") -> None:
        with self._lock:
            self._rows = None
            self._rows_t = -1e9
            self._watermark = int(getattr(self.publisher, "observation_id", 0))
        self._publish_reset(reason)

    def pause(self) -> None:
        self._paused = True
        with self._lock:
            self._rows = None
            self._rows_t = -1e9

    def resume(self) -> None:
        self._paused = False

    @property
    def paused(self) -> bool:
        return self._paused

    def version_label(self) -> str:
        return f"{self.policy_id}/v{self._version:06d}"

    def current_version(self) -> int:
        return int(self._version)

    # -- telemetry helpers -------------------------------------------------------------------------
    def policy_stale(self, now: float | None = None) -> bool:
        now = self._clock() if now is None else now
        return self.staleness_scale(now) < 1.0 or not self.hub.policy_attached

    def action_age_s(self, now: float | None = None) -> float | None:
        now = self._clock() if now is None else now
        return None if self._last_action_t is None else max(0.0, now - self._last_action_t)

    @property
    def watermark(self) -> int:
        return self._watermark

    # -- inbound (bus thread) ----------------------------------------------------------------------
    def on_spec(self, ann: PolicySpecAnnounce) -> None:
        with self._lock:
            if ann.policy_version != self._version:
                self.version_changes += 1
                self._version = int(ann.policy_version)
                self.policy.spec = self.spec = PolicySpec(
                    action_space=self.spec.action_space,
                    action_frame=self.spec.action_frame,
                    action_names=self.spec.action_names,
                    state_names=self.spec.state_names,
                    camera_keys=self.spec.camera_keys,
                    version=self._version,
                )

    def on_action(self, ev: dict[str, Any], t_recv: float) -> None:
        meta = ev.get("metadata") or {}
        missing = [k for k in ext.POLICY_ACTION_REQUIRED_METADATA if k not in meta]
        if missing:
            self.hub.bridge.count_drop(f"policy_action missing metadata {missing}")
            return
        if str(meta.get(ext.META_SESSION_ID, "")) != self.session_id:
            self.hub.bridge.count_drop(
                f"policy_action session_id {meta.get(ext.META_SESSION_ID)!r} != {self.session_id!r}"
            )
            return
        try:
            oid = int(meta["observation_id"])
            rows = codec.decode_action(ev.get("value"), meta)
            chunk_dt = float(meta.get("chunk_dt_s") or self.default_chunk_dt)
            version = int(meta["policy_version"])
        except (TypeError, ValueError, KeyError) as exc:
            self.hub.bridge.count_drop(f"policy_action malformed: {exc}")
            return
        if rows.shape[1] != len(self.spec.action_names):
            self.hub.bridge.count_drop(
                f"policy_action action_dim {rows.shape[1]} != {len(self.spec.action_names)}"
            )
            return
        if chunk_dt <= 0.0:
            self.hub.bridge.count_drop(f"policy_action chunk_dt_s {chunk_dt} <= 0")
            return
        if oid <= self._watermark:
            self.actions_late += 1
            self.hub.bridge.count_drop(
                f"policy_action observation_id {oid} <= watermark {self._watermark}"
            )
            return
        obs_t = (
            self.publisher.observation_t_mono(oid)
            if hasattr(self.publisher, "observation_t_mono")
            else None
        )
        if obs_t is None:
            self.actions_late += 1
            self.hub.bridge.count_drop(f"policy_action references unknown observation_id {oid}")
            return
        if t_recv - obs_t > self.cfg.max_obs_age_s:
            self.actions_late += 1
            self.hub.bridge.count_drop(
                f"policy_action observation {oid} is {t_recv - obs_t:.2f} s old "
                f"(> {self.cfg.max_obs_age_s})"
            )
            return
        changed = False
        with self._lock:
            if version != self._version:
                self.version_changes += 1
                changed = True
                self._version = version
                self.policy.spec = self.spec = PolicySpec(
                    action_space=self.spec.action_space,
                    action_frame=self.spec.action_frame,
                    action_names=self.spec.action_names,
                    state_names=self.spec.state_names,
                    camera_keys=self.spec.camera_keys,
                    version=version,
                )
            self._rows = rows
            self._rows_t = t_recv
            self._chunk_dt = chunk_dt
            self._last_action_t = t_recv
            self.actions_applied += 1
        if changed:  # phase-14: an action's policy_version is the acting version too (§3)
            sink = self.hub.trainer_sink
            if sink is not None:
                try:
                    sink.on_spec_version(version)
                except Exception:  # noqa: BLE001
                    logger.exception("trainer sink on_spec_version failed")

    # -- helpers -----------------------------------------------------------------------------------
    def _publish_reset(self, reason: str) -> None:
        pub = self.publisher
        after = int(self._watermark)
        msg = PolicyResetMsg(
            reason=reason,
            after_observation_id=after,
            session_id=self.session_id,  # type: ignore[arg-type]
            t_mono=self._clock(),
        )
        bridge = self.hub.bridge
        if bridge.publish_event(
            ext.OUT_POLICY_RESET, lambda: (codec.encode_json(msg), {"reason": reason})
        ):
            if hasattr(pub, "publish_event"):
                pub.publish_event(
                    "reset_watermark",
                    {"reason": reason, "after_observation_id": after},
                    session_id=self.session_id,
                )

    def _build_delta_scale(self) -> np.ndarray:
        """Per-dim factor turning per-``chunk_dt`` deltas into per-``period`` deltas
        (gripper absolute -> 1.0). Recomputed lazily in :meth:`latest` via ``_chunk_dt``."""
        dims: list[float] = []
        for _, has_rail in self.arms_meta:
            dims += [1.0] * 6 + [0.0] + ([1.0] if has_rail else [])
        mask = np.asarray(dims, dtype=np.float64)
        if mask.shape[0] != len(self.spec.action_names):
            mask = np.ones(len(self.spec.action_names))
            mask[6::7] = 0.0  # best effort: gripper every 7th dim
        self._delta_mask = mask
        return np.ones_like(mask)

    @property
    def _delta_scale(self) -> np.ndarray:  # type: ignore[override]
        factor = self.period / self._chunk_dt if self._chunk_dt > 0 else 1.0
        return np.where(self._delta_mask > 0.0, factor, 1.0)

    @_delta_scale.setter
    def _delta_scale(self, value: np.ndarray) -> None:  # set by __init__; derived afterwards
        pass


__all__ = [
    "ONLINE_DAGGER_CAPABILITY",
    "ExternalPolicyHub",
    "ExternalPolicySource",
    "spec_from_announce",
]
