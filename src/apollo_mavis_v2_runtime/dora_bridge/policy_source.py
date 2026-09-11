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
  executor's ``dt / period`` scaling yields ``dt / chunk_dt_s``. An ``abs_ee``
  policy (2026-09-11) announces the 11 / 10-dim ``[x, y, z, r6, gripper, rail?]``
  blocks instead; its rows are waypoints and pass through VERBATIM (the rescale mask
  is all zeros - an absolute value is never multiplied by a chunk factor);
- ``staleness_scale`` keeps the in-process rule verbatim (``period + 0.05`` then
  linear to 0 over 5 periods; 0.45 s at 15 Hz); the clock is the runtime's;
- ``drop_and_requery()`` publishes ``policy_reset{after_observation_id}`` and
  sets the watermark; ``pause()`` holds (NaN 3-strike) until ``resume()``.

Per-arm action streams (v1.3, 2026-09-11; 14-dora §5 / §6.1). The hub also registers
``policy_action_<arm_id>`` for every configured arm; a policy DRIVES the arms its spec
names (``PolicySpecModel.arms``, else every arm whose whole block its ``action_names``
cover - :func:`resolve_driven_arms`). The source keeps ONE chunk slot per driven arm:
a per-arm message fills that arm's slot with its 7 / 8-dim block, the whole-cell
``policy_action`` (still accepted) is split into the driven arms' blocks. ``latest()``
composes the SESSION layout the executor expects - each driven arm's current row,
rescaled with its own ``chunk_dt``, and **NaN for every arm the policy does not drive**
(the executor holds those; they are never a NaN strike). ``staleness_scale(now,
arm_id)`` is per arm (a silent Manipulation Arm stream holds the Manipulation Arm
only); without ``arm_id`` it is the minimum over the driven arms (the telemetry flag).
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections.abc import Callable, Iterable, Sequence
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
from ..recorder.features import arm_action_names
from . import codec
from .bridge import DoraBridge

logger = logging.getLogger(__name__)

ONLINE_DAGGER_CAPABILITY = "online_dagger"  # PolicySpecAnnounce.capabilities (15-online-dagger §6)


class DrivenArmsError(ValueError):
    """The announced spec's ``arms`` / ``action_names`` do not describe a subset of whole
    per-arm blocks of the session layout (the manager turns it into a 409)."""


def arm_blocks(
    arms_meta: Sequence[tuple[str, bool]], action_space: str = "delta_ee"
) -> dict[str, list[str]]:
    """``{arm_id: its action-name block}`` for the announced space, in session order
    (10-frames §6; widths always from ``arm_action_names``)."""
    return {a: arm_action_names(a, bool(r), action_space) for a, r in arms_meta}


def block_rescale_mask(has_rail: bool, action_space: str) -> np.ndarray:
    """Per-dim ``latest()`` rescale mask of one block: ``delta_ee`` deltas (6 + the rail
    delta) rescale per chunk_dt, the gripper is absolute; every ``abs_ee`` dim is absolute
    (all zeros, rows pass verbatim)."""
    if action_space == "delta_ee":
        return np.asarray([1.0] * 6 + [0.0] + ([1.0] if has_rail else []), dtype=np.float64)
    return np.zeros(len(arm_action_names("_", has_rail, action_space)), dtype=np.float64)


def action_block_layout(
    action_names: Sequence[str],
    blocks: dict[str, list[str]],
    driven: Iterable[str],
) -> list[tuple[str, int, int]]:
    """Match ``action_names`` against the whole per-arm blocks of ``driven``, GREEDILY, in
    whatever ORDER the blocks appear (v1.3, 14-dora §6.1): a whole-cell policy may
    concatenate its arms in any order - the runtime maps each block to its arm by name,
    not by position, because a policy announces its spec BEFORE a session exists and cannot
    know the session's arm order (the idle announce is in workcell order; the session
    layout follows ``SessionSpec.arms``). Returns ``[(arm, start, len)]`` in the order the
    blocks appear in ``action_names``. Raises :class:`DrivenArmsError` unless ``action_names``
    is EXACTLY those arms' blocks concatenated - each block contiguous and complete, each
    arm once (arm names are ``<arm>_``-prefixed, so at most one block matches at any offset)."""
    names = list(action_names)
    remaining = set(driven)
    out: list[tuple[str, int, int]] = []
    i = 0
    while i < len(names):
        arm = next((a for a in remaining if names[i : i + len(blocks[a])] == blocks[a]), None)
        if arm is None:
            want = {a: blocks[a] for a in sorted(remaining)}
            raise DrivenArmsError(
                f"external policy action_names {names} are not the per-arm blocks of "
                f"{sorted(remaining)} concatenated (in any order); expected the blocks {want}"
            )
        out.append((arm, i, len(blocks[arm])))
        i += len(blocks[arm])
        remaining.discard(arm)
    if remaining:
        raise DrivenArmsError(
            f"external policy action_names {names} miss the whole block of "
            f"arm(s) {sorted(remaining)}"
        )
    return out


def resolve_driven_arms(
    spec_arms: Sequence[str],
    action_names: Sequence[str],
    arms_meta: Sequence[tuple[str, bool]],
    action_space: str = "delta_ee",
) -> list[str]:
    """The arms a policy drives, in SESSION order (14-dora §6.1 step 2, v1.3).

    ``spec_arms`` non-empty: every entry must be a session arm and ``action_names`` must be
    exactly those arms' blocks concatenated (in ANY order - see :func:`action_block_layout`).
    Empty: the driven arms are inferred - every session arm whose whole block appears in
    ``action_names`` - and ``action_names`` must then be exactly those blocks (any order),
    so a legacy whole-cell spec resolves to every arm. The blocks are those of the ANNOUNCED
    ``action_space`` (``delta_ee`` 8 / 7, ``abs_ee`` 11 / 10). Raises
    :class:`DrivenArmsError` otherwise, with the operator-facing reason.
    """
    blocks = arm_blocks(arms_meta, action_space)
    session_arms = [a for a, _ in arms_meta]
    names = list(action_names)
    if spec_arms:
        unknown = [a for a in spec_arms if a not in blocks]
        if unknown:
            raise DrivenArmsError(
                f"external policy drives unknown arm(s) {unknown} (session arms {session_arms})"
            )
        if len(set(spec_arms)) != len(spec_arms):
            raise DrivenArmsError(f"external policy lists an arm twice: {list(spec_arms)}")
        driven_set = set(spec_arms)
    else:
        have = set(names)
        driven_set = {a for a in session_arms if set(blocks[a]) <= have}
        if not driven_set:
            raise DrivenArmsError(
                f"external policy action_names {names} cover no whole arm block of the session "
                f"layout {[n for a in session_arms for n in blocks[a]]}"
            )
    action_block_layout(names, blocks, driven_set)  # validates the concatenation (any order)
    return [a for a in session_arms if a in driven_set]  # session order


def infer_policy_arms(ann: PolicySpecAnnounce, arm_ids: Sequence[str] = ()) -> list[str]:
    """Best-effort ``ExternalStatus.policy_arms`` for a spec heard OUTSIDE a session (no
    layout to check against): the declared ``arms``, else the ``<arm_id>_`` prefixes of
    its ``action_names`` (configured ``arm_ids`` first, then whatever precedes the last
    ``_`` of a name), in first-seen order."""
    s = ann.spec
    if s.arms:
        return list(s.arms)
    out: list[str] = []
    for n in s.action_names:
        arm = next((a for a in arm_ids if n.startswith(f"{a}_")), None)
        if arm is None:
            arm = n.rsplit("_", 1)[0] if "_" in n else n
        if arm not in out:
            out.append(arm)
    return out


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
        self,
        bridge: DoraBridge,
        cfg: DoraPolicyConfig,
        clock: Callable[[], float] = time.monotonic,
        arm_ids: Sequence[str] = (),
    ) -> None:
        self.bridge = bridge
        self.cfg = cfg
        self._clock = clock
        self.arm_ids = list(arm_ids)  # configured arms -> policy_action_<arm> inputs (v1.3)
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
        for arm in self.arm_ids:  # v1.3: one per-arm action input per configured arm
            bridge.register_input(
                ext.policy_arm_action_input_id(arm), functools.partial(self._on_arm_action, arm)
            )
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

    def policy_arms(self, now: float | None = None) -> list[str]:
        """``ExternalStatus.policy_arms`` (v1.3): the arms the FRESH spec drives (declared, or
        inferred from its action-name prefixes); ``[]`` when no spec is fresh."""
        ann = self.spec(now)
        return [] if ann is None else infer_policy_arms(ann, self.arm_ids)

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

    def _on_arm_action(self, arm_id: str, ev: dict[str, Any]) -> None:
        """``policy_action_<arm_id>`` (v1.3): that arm's block only."""
        if ev.get("type") != "INPUT":
            return
        self.actions_seen += 1
        src = self._source
        if src is None:
            self.bridge.count_drop(f"policy_action_{arm_id} without an external session")
            return
        src.on_action(ev, self._clock(), arm_id=arm_id)


class _ArmSlot:
    """One driven arm's current chunk: ``rows`` (K, n_block) in per-``chunk_dt`` units,
    the receive time the chunk became current and its ``chunk_dt``."""

    __slots__ = ("chunk_dt", "rows", "t0")

    def __init__(self) -> None:
        self.rows: np.ndarray | None = None
        self.t0: float = -1e9
        self.chunk_dt: float = 0.0

    def clear(self) -> None:
        self.rows = None
        self.t0 = -1e9

    def current(self, now: float) -> tuple[np.ndarray | None, float, int]:
        """``(row, active_t, remaining)`` - row 0 at once, one row per ``chunk_dt`` after."""
        rows = self.rows
        if rows is None:
            return None, -1e9, 0
        k = rows.shape[0]
        idx = 0 if k == 1 else int(min(k - 1, max(0, (now - self.t0) // self.chunk_dt)))
        return rows[idx], self.t0 + idx * self.chunk_dt, k - 1 - idx


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
        driven_arms: Sequence[str] | None = None,
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
        # v1.3: the arms this policy drives (session order); every other arm holds
        session_arms = [a for a, _ in self.arms_meta]
        self._driven: list[str] = (
            list(session_arms)
            if driven_arms is None
            else [a for a in session_arms if a in set(driven_arms)]
        )
        self._driven_set = frozenset(self._driven)
        # arm -> block names of the ANNOUNCED space (session layout); per-arm action_dim
        self._blocks = arm_blocks(self.arms_meta, spec.action_space)
        self._block_dim = {a: len(n) for a, n in self._blocks.items()}
        # per-dim scaling mask of a block: delta_ee deltas rescale per chunk_dt, the gripper
        # is absolute (10-frames §6); abs_ee rows are waypoints - nothing rescales
        self._block_mask = {
            a: block_rescale_mask(bool(r), spec.action_space) for a, r in self.arms_meta
        }
        self._slots: dict[str, _ArmSlot] = {a: _ArmSlot() for a in self._driven}
        # whole-cell `action`: the driven blocks as spec.action_names lists them (v1.3 - the
        # policy may concatenate its arms in any order; each block is placed into its arm's
        # slot). Empty when the policy drives nothing (never, once resolve_driven_arms passed).
        self._split: list[tuple[str, int, int]] = (
            action_block_layout(spec.action_names, self._blocks, self._driven_set)
            if self._driven
            else []
        )
        self._version: int = int(spec.version)
        self._paused = False
        self._watermark = 0
        self._last_action_t: float | None = None
        self.actions_late = 0
        self.actions_applied = 0
        self.version_changes = 0
        self._started = False

    # -- lifecycle ---------------------------------------------------------------------------------
    def start(self) -> None:
        self._started = True
        self.hub.attach_source(self)
        # observation ids restart at 1 for this session (SnapshotPublisher.session_started):
        # the session_start watermark is 0, whatever the publisher counted before
        with self._lock:
            self._clear_slots()
            self._watermark = 0
        self._publish_reset("session_start")

    def stop(self) -> None:
        if self._started:
            self._publish_reset("session_stop")
        self.hub.detach_source(self)
        self._started = False

    # -- the PolicySource surface (control thread) ----------------------------------------------
    def driven_arms(self) -> frozenset[str]:
        """The arms this policy drives (v1.3); every other session arm holds."""
        return self._driven_set

    def latest(self) -> tuple[PolicyOutput | None, float]:
        """The SESSION-layout action the executor consumes: each driven arm's current chunk
        row (rescaled from per-``chunk_dt`` to per-``period`` units, gripper absolute), NaN
        for every arm the policy does not drive; ``t`` is the newest row's active time.
        ``(None, -1e9)`` while no driven arm has a chunk (or while paused)."""
        if self._paused:
            return None, -1e9
        now = self._clock()
        with self._lock:
            version = self._version
            parts: list[np.ndarray] = []
            t_newest = -1e9
            remaining = 0
            any_rows = False
            for arm, _ in self.arms_meta:
                n = self._block_dim[arm]
                slot = self._slots.get(arm)
                row = None
                if slot is not None:
                    row, active_t, rem = slot.current(now)
                if row is None:
                    parts.append(np.full(n, np.nan, dtype=np.float64))
                    continue
                any_rows = True
                factor = self.period / slot.chunk_dt if slot.chunk_dt > 0 else 1.0
                scale = np.where(self._block_mask[arm] > 0.0, factor, 1.0)
                parts.append(np.asarray(row, dtype=np.float64) * scale)
                if active_t > t_newest:
                    t_newest, remaining = active_t, rem
        if not any_rows:
            return None, -1e9
        return (
            PolicyOutput(
                actions=np.concatenate(parts).astype(np.float32),
                version=version,
                t_mono=t_newest,
                chunk_remaining=remaining,
            ),
            t_newest,
        )

    def staleness_scale(self, now: float, arm_id: str | None = None) -> float:
        """Per arm (v1.3): 1.0 while that arm's current row is younger than ``period +
        0.05 s``, linear to 0 over 5 periods, 0 with no chunk at all. Without ``arm_id``
        the minimum over the driven arms (the session's ``policy_stale`` flag)."""
        if self._paused:
            return 0.0
        if arm_id is not None:
            return self._stale_one(now, arm_id)
        if not self._driven:
            return 0.0
        return min(self._stale_one(now, a) for a in self._driven)

    def _stale_one(self, now: float, arm_id: str) -> float:
        slot = self._slots.get(arm_id)
        if slot is None:
            return 0.0  # not driven by this policy: the executor holds it before asking
        with self._lock:
            _, t, _ = slot.current(now)
        age = now - t
        timeout = self.period + POLICY_TIMEOUT_MARGIN_S
        if age <= timeout:
            return 1.0
        return float(np.clip(1.0 - (age - timeout) / (STALE_DECAY_PERIODS * self.period), 0.0, 1.0))

    def drop_and_requery(self, reason: str = "handback") -> None:
        with self._lock:
            self._clear_slots()
            self._watermark = int(getattr(self.publisher, "observation_id", 0))
        self._publish_reset(reason)

    def pause(self) -> None:
        self._paused = True
        with self._lock:
            self._clear_slots()

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
                self._set_version(int(ann.policy_version))

    def on_action(self, ev: dict[str, Any], t_recv: float, arm_id: str | None = None) -> None:
        """A ``policy_action`` (whole cell: the driven arms' blocks concatenated in session
        order = the policy's ``action_names``) or, with ``arm_id``, a ``policy_action_<arm>``
        (that arm's block). Validated on the bus thread; a bad message is dropped + counted."""
        what = ext.IN_POLICY_ACTION if arm_id is None else ext.policy_arm_action_input_id(arm_id)
        meta = ev.get("metadata") or {}
        missing = [k for k in ext.POLICY_ACTION_REQUIRED_METADATA if k not in meta]
        if missing:
            self.hub.bridge.count_drop(f"{what} missing metadata {missing}")
            return
        if str(meta.get(ext.META_SESSION_ID, "")) != self.session_id:
            self.hub.bridge.count_drop(
                f"{what} session_id {meta.get(ext.META_SESSION_ID)!r} != {self.session_id!r}"
            )
            return
        if arm_id is not None:
            if arm_id not in self._slots:
                self.hub.bridge.count_drop(
                    f"{what}: arm {arm_id!r} is not driven by this policy (drives {self._driven})"
                )
                return
            claimed = meta.get("arm_id")
            if claimed is not None and str(claimed) != arm_id:
                self.hub.bridge.count_drop(f"{what}: metadata arm_id {claimed!r} != {arm_id!r}")
                return
        try:
            oid = int(meta["observation_id"])
            rows = codec.decode_action(ev.get("value"), meta)
            chunk_dt = float(meta.get("chunk_dt_s") or self.default_chunk_dt)
            version = int(meta["policy_version"])
        except (TypeError, ValueError, KeyError) as exc:
            self.hub.bridge.count_drop(f"{what} malformed: {exc}")
            return
        expected = self._block_dim[arm_id] if arm_id is not None else len(self.spec.action_names)
        if rows.shape[1] != expected:
            self.hub.bridge.count_drop(f"{what} action_dim {rows.shape[1]} != {expected}")
            return
        if chunk_dt <= 0.0:
            self.hub.bridge.count_drop(f"{what} chunk_dt_s {chunk_dt} <= 0")
            return
        if oid <= self._watermark:
            self.actions_late += 1
            self.hub.bridge.count_drop(
                f"{what} observation_id {oid} <= watermark {self._watermark}"
            )
            return
        obs_t = (
            self.publisher.observation_t_mono(oid)
            if hasattr(self.publisher, "observation_t_mono")
            else None
        )
        if obs_t is None:
            self.actions_late += 1
            self.hub.bridge.count_drop(f"{what} references unknown observation_id {oid}")
            return
        if t_recv - obs_t > self.cfg.max_obs_age_s:
            self.actions_late += 1
            self.hub.bridge.count_drop(
                f"{what} observation {oid} is {t_recv - obs_t:.2f} s old "
                f"(> {self.cfg.max_obs_age_s})"
            )
            return
        changed = False
        with self._lock:
            if version != self._version:
                self.version_changes += 1
                changed = True
                self._set_version(version)
            if arm_id is not None:
                self._fill(arm_id, rows, t_recv, chunk_dt)
            else:  # whole cell: place each announced block into its arm's slot (v1.3: any order)
                for arm, start, n in self._split:
                    self._fill(arm, rows[:, start : start + n], t_recv, chunk_dt)
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
    def _fill(self, arm_id: str, rows: np.ndarray, t_recv: float, chunk_dt: float) -> None:
        slot = self._slots[arm_id]
        slot.rows = np.ascontiguousarray(rows, dtype=np.float32)
        slot.t0 = t_recv
        slot.chunk_dt = chunk_dt

    def _clear_slots(self) -> None:
        for slot in self._slots.values():
            slot.clear()

    def _set_version(self, version: int) -> None:
        self._version = version
        self.policy.spec = self.spec = PolicySpec(
            action_space=self.spec.action_space,
            action_frame=self.spec.action_frame,
            action_names=self.spec.action_names,
            state_names=self.spec.state_names,
            camera_keys=self.spec.camera_keys,
            version=version,
        )

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


__all__ = [
    "ONLINE_DAGGER_CAPABILITY",
    "DrivenArmsError",
    "ExternalPolicyHub",
    "ExternalPolicySource",
    "action_block_layout",
    "arm_blocks",
    "block_rescale_mask",
    "infer_policy_arms",
    "resolve_driven_arms",
    "spec_from_announce",
]
