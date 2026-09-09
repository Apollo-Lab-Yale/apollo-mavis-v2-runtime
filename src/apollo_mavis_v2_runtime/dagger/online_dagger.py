"""``OnlineDaggerCoordinator`` — the rollout-level shell of an Online DAgger session
(15-online-dagger §3).

Session-scoped, owned by ``GatedPolicyExecutor`` when ``SessionSpec.online_dagger`` is
set. Pure state + publishing: it owns NO motion, no dora handle and no file descriptor
of its own — the side effects go through two injected callables, ``publish(kind,
payload)`` (``events`` on the dora bus) and ``write_session(doc)`` (``session.json``,
rewritten atomically), which the runtime routes through :class:`SerialWorker` so nothing
here ever performs I/O on the 100 Hz tick. Unit tests inject recording lists instead.

The runtime is algorithm-agnostic (operator decision 2026-09-08 evening): it performs
rollouts, exposes take-over / hand-back, labels every step novice / expert, saves the
kept rollouts and reports what the trainer says. It counts kept rollouts and the
session's actor split — never iterations, hyper-parameters or training artefacts,
which belong to the trainer node in the policy repo.

State (§3): ``phase`` in ``waiting_trainer | rollout | training | error``,
``rollouts_saved``, ``expert_frames_session`` / ``novice_frames_session``, the last
``TrainerStatusAnnounce`` with its receive time, ``policy_version_acting`` and the
``trainer_seen_ready`` latch. Rules:

- start -> ``waiting_trainer`` when ``wait_for_trainer_ready`` else ``rollout``; a trainer
  status echoing THIS ``session_id`` with ``state == "ready"`` latches
  ``trainer_seen_ready`` -> ``rollout``;
- ``training`` (this session) -> ``training`` when ``pause_while_training`` (else the
  phase stays and the pill just shows it); ``ready`` again -> ``rollout``; ``error`` ->
  ``error`` until a non-error status arrives;
- ``episode_new`` refusals: ``"no Online DAgger trainer attached"`` (no fresh status),
  ``"waiting for the trainer to report ready (<detail>)"``, ``"training in progress
  (<detail>)"``, ``"trainer error: <detail>"``;
- a kept rollout -> ``rollouts_saved += 1``, counts added, ``events.episode_saved`` with
  the ``online_dagger`` block and a ``rollouts`` row in ``session.json``; a discard ->
  ``events.episode_discarded`` only (the recorder already removed the temp directory);
- **Train now** -> ``events.train_now``; refused while an episode is open or without a
  fresh trainer status (the trainer may still ignore it);
- every ``TakeoverGate`` event the executor hands over -> ``events.gate``;
- ``policy_version_acting`` follows the ANNOUNCED spec / action version (never the
  trainer's own claim); a change is logged in ``session.json.trainer_log``.

Only a trainer status that echoes this session's id drives a transition: the previous
session's tail (heartbeats in flight, the hub's replayed cache) is ignored outright, a
status with ``session_id: null`` counts as alive and is shown verbatim but moves nothing,
and a status older than the newest seen is dropped. ``session.json`` is the resume
record: a second session with the same name (``resume: true``) continues the counters.
History: this module replaces the v1.0 PRO-DAgger coordinator (iteration machine,
reference-gradient gating, carry logic) superseded on 2026-09-08 before it ever shipped.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from apollo_mavis_v2_core.protocol import (
    OnlineDaggerConfig,
    OnlineDaggerSessionInfo,
    OnlineDaggerStatus,
    SessionSpec,
)
from apollo_mavis_v2_core.protocol.external import OnlineDaggerAnnounce, TrainerStatusAnnounce

logger = logging.getLogger(__name__)

SESSION_JSON = "session.json"
ROLLOUTS_DIR = "rollouts"
PHASES = ("waiting_trainer", "rollout", "training", "error")
TRAINER_LOG_MAX = 200  # session.json.trainer_log keeps the newest rows
Event = tuple[str, dict[str, Any]]

NO_TRAINER = "no Online DAgger trainer attached"
EPISODE_OPEN = "save or discard the episode first"
NO_STATUS_YET = "no trainer status yet"
NOT_SERVING = "the trainer has not picked up this session yet"


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def write_session_json_atomic(path: Path, doc: dict[str, Any]) -> None:
    """``session.json`` writer: tmp + ``os.replace`` (never a half-written file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


class SerialWorker:
    """One daemon thread that runs the coordinator's side effects in submission
    order (the events keep their order, the session file never races itself);
    :meth:`close` drains it."""

    def __init__(self, name: str = "online-dagger-io") -> None:
        self._pool: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=name
        )

    def submit(self, fn: Callable[..., Any], *args: Any) -> None:
        pool = self._pool
        if pool is None:
            self._run(fn, *args)  # closed: run inline (teardown path)
            return
        try:
            pool.submit(self._run, fn, *args)
        except RuntimeError:  # shutting down
            self._run(fn, *args)

    @staticmethod
    def _run(fn: Callable[..., Any], *args: Any) -> None:
        try:
            fn(*args)
        except Exception:  # noqa: BLE001 - a bus / disk hiccup never kills the worker
            logger.exception("Online DAgger side effect %s failed", getattr(fn, "__name__", fn))

    def close(self, wait: bool = True) -> None:
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=wait)


@dataclass(frozen=True)
class OnlineDaggerPaths:
    """The session directory (15-online-dagger §0 item 6): ``<root>/<session_name>/``
    holds ``session.json`` (runtime-owned) and ``rollouts/`` (the episode-directory
    dataset ``online_dagger/<session_name>``). The trainer keeps its own artefacts
    wherever it likes (the skill suggests ``<session_dir>/trainer/``); the runtime never
    creates or reads them."""

    session_dir: Path
    rollouts_dir: Path

    @property
    def session_json(self) -> Path:
        return self.session_dir / SESSION_JSON

    def mkdirs(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.rollouts_dir.mkdir(parents=True, exist_ok=True)


class OnlineDaggerCoordinator:
    """See the module docstring. Every public method is safe from any thread (one
    re-entrant lock); ``publish`` / ``write_session`` are invoked UNDER that lock so
    two threads' transitions cannot interleave their events — they must therefore
    be non-blocking (a :class:`SerialWorker` submit, a list append) and never call
    back into the coordinator."""

    def __init__(
        self,
        cfg: OnlineDaggerConfig,
        *,
        session_id: str,
        paths: OnlineDaggerPaths,
        spec: SessionSpec | None = None,
        task: str | None = None,
        run_id: str = "",
        spec_stale_s: float = 3.0,
        policy_version: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        wallclock: Callable[[], float] = time.time,
        publish: Callable[[str, dict[str, Any]], None] | None = None,
        write_session: Callable[[dict[str, Any]], None] | None = None,
        session_file_hz: float = 1.0,
    ) -> None:
        self.cfg = cfg
        self.session_id = session_id
        self.paths = paths
        self.spec = spec
        self.task = task if task is not None else (spec.task if spec is not None else None)
        self.run_id = run_id
        self.spec_stale_s = float(spec_stale_s)
        self._clock = clock
        self._wallclock = wallclock
        self._publish = publish
        self._write_session = write_session
        self._write_period = 1.0 / session_file_hz if session_file_hz > 0 else 0.0
        self._lock = threading.RLock()

        self.created_at = _iso(wallclock())
        self.phase: str = "waiting_trainer" if cfg.wait_for_trainer_ready else "rollout"
        self.rollouts_saved = 0
        self.expert_frames_session = 0
        self.novice_frames_session = 0
        self.rollouts: list[dict[str, Any]] = []  # session.json rows, capture order
        self.trainer_log: list[dict[str, Any]] = []  # newest TRAINER_LOG_MAX rows
        self.trainer: TrainerStatusAnnounce | None = None
        self._trainer_t: float | None = None
        self._logged_state: str | None = None
        self.trainer_seen_ready = False
        self.policy_version_acting: int | None = policy_version
        self._last_write_t: float = -1e9
        self.started = False
        self.resumed = False
        self.events_published = 0

    # -- announce / sidecar -------------------------------------------------------------------
    @property
    def session_name(self) -> str:
        return self.cfg.session_name

    def announce(self) -> OnlineDaggerAnnounce:
        """``SessionAnnounce.online_dagger`` (15-online-dagger §6)."""
        return OnlineDaggerAnnounce(
            session_name=self.cfg.session_name,
            session_dir=str(self.paths.session_dir),
            rollouts_dir=str(self.paths.rollouts_dir),
        )

    def sidecar_block(self, actor_counts: dict[str, int]) -> dict[str, Any]:
        """``episode.json["online_dagger"]`` of the episode being saved (15-online-dagger
        §4): ``rollouts_saved`` is the count INCLUDING this rollout (the same number the
        ``events.episode_saved`` block carries — the counter moves only on saves, which
        the one recorder thread serialises)."""
        with self._lock:
            return {
                "session_name": self.cfg.session_name,
                "rollouts_saved": self.rollouts_saved + 1,
                "policy_version": self.policy_version_acting,
                "actor_counts": {
                    "novice": int(actor_counts.get("novice", 0)),
                    "expert": int(actor_counts.get("expert", 0)),
                },
            }

    # -- lifecycle -----------------------------------------------------------------------------
    def on_session_start(self) -> list[Event]:
        """Session RUNNING: writes the file. Publishes nothing (the shell has no phase
        event); a trainer status replayed BEFORE this call (the hub replays its cache
        when the sink attaches) keeps the phase it produced."""
        with self._lock:
            self.started = True
            self._emit([], write=True)
            return []

    def close(self) -> None:
        """Session end: one last ``session.json`` (``last_used_at``); no events."""
        with self._lock:
            self._emit([], write=True)

    # -- inbound: trainer status / policy version (bus thread) ------------------------------------
    def on_trainer_status(self, msg: TrainerStatusAnnounce, t_recv: float) -> list[Event]:
        """A ``trainer_status`` (bus thread; also the hub's replay of its cached status
        when the sink attaches). Three guards before the state machine sees it:

        - a status older than the newest one already seen is dropped (the attach
          replay racing a fresh heartbeat), so the trainer view never runs backwards;
        - a status for ANOTHER session (a non-null ``session_id`` that is not ours: the
          previous session's tail) is ignored outright — it is not this session's
          trainer, so it neither refreshes ``trainer_alive`` nor moves anything;
        - a status with no ``session_id`` (the trainer is up, serving nobody yet)
          counts as alive and is shown verbatim, but drives NO transition.

        The phase is a function of the status (plus the ``trainer_seen_ready`` latch):
        ``error`` -> ``error``; ``training`` -> ``training`` when
        ``pause_while_training``; ``ready`` -> ``rollout`` (latches ready); ``idle`` /
        ``preparing`` (and ``training`` without the pause) -> ``rollout`` once ready was
        seen or ``wait_for_trainer_ready`` is off, else ``waiting_trainer``.
        """
        with self._lock:
            t = float(t_recv)
            if self._trainer_t is not None and t < self._trainer_t:
                return []
            if msg.session_id is not None and msg.session_id != self.session_id:
                return []
            self.trainer = msg
            self._trainer_t = t
            if msg.session_id != self.session_id:  # None: attached, not serving us yet
                self._emit([], write=self._write_due())
                return []
            changed = self._log_trainer_state(msg)
            if msg.state == "ready":
                self.trainer_seen_ready = True
            before = self.phase
            self.phase = self._phase_for(msg.state)
            moved = self.phase != before
            if moved:
                logger.info(
                    "Online DAgger %s: %s -> %s (trainer %s%s)",
                    self.cfg.session_name,
                    before,
                    self.phase,
                    msg.state,
                    f": {msg.detail}" if msg.detail else "",
                )
            self._emit([], write=moved or changed or self._write_due())
            return []

    def _phase_for(self, state: str) -> str:
        if state == "error":
            return "error"
        if state == "training" and self.cfg.pause_while_training:
            return "training"
        if state == "ready" or self.trainer_seen_ready or not self.cfg.wait_for_trainer_ready:
            return "rollout"
        return "waiting_trainer"

    def on_spec_version(self, version: int) -> list[Event]:
        """The announced policy version (``policy_spec`` heartbeat or an action's
        metadata) IS the acting version; a change is a ``trainer_log`` row."""
        with self._lock:
            v = int(version)
            old = self.policy_version_acting
            if old == v:
                return []
            self.policy_version_acting = v
            if old is not None:
                self._log(
                    "swapped",
                    v,
                    f"policy v{old} -> v{v}",
                )
                logger.info("Online DAgger %s: policy v%s -> v%s", self.cfg.session_name, old, v)
                self._emit([], write=True)
            return []

    # -- inbound: recorder (recorder thread) -----------------------------------------------------
    def on_episode_saved(
        self, episode_id: str, index: int, summary: Any, spool_path: str
    ) -> list[Event]:
        """A kept rollout: the counters move, a ``rollouts`` row is filed and
        ``events.episode_saved`` carries the ``online_dagger`` block."""
        with self._lock:
            n_exp = int(getattr(summary, "n_expert_frames", 0) or 0)
            n_nov = int(getattr(summary, "n_novice_frames", 0) or 0)
            self.rollouts_saved += 1
            self.expert_frames_session += n_exp
            self.novice_frames_session += n_nov
            counts = {"novice": n_nov, "expert": n_exp}
            # "" = the recorder could not write the spool (logged); the rollout is on disk
            spool = str(spool_path) if spool_path else None
            self.rollouts.append(
                {
                    "episode_id": episode_id,
                    "saved_at": self._now_iso(),
                    "actor_counts": dict(counts),
                    "policy_version": self.policy_version_acting,
                    "spool_path": spool,
                }
            )
            block = {
                "episode_id": episode_id,
                "rollouts_saved": self.rollouts_saved,
                "actor_counts": dict(counts),
                "policy_version": self.policy_version_acting,
                "spool_path": spool,
            }
            payload = {
                "episode_index": int(index),
                "summary": (
                    dataclasses.asdict(summary) if dataclasses.is_dataclass(summary) else summary
                ),
                "dataset_root": str(self.paths.rollouts_dir),
                "spool_path": spool,
                "run_id": self.run_id or None,
                "online_dagger": block,
            }
            events: list[Event] = [("episode_saved", payload)]
            self._emit(events, write=True)
            return events

    def on_episode_discarded(
        self, episode_id: str | None, index: int | None, reason: str = ""
    ) -> list[Event]:
        """A discarded rollout: nothing persisted (the recorder removed the temp
        directory), the counters do not move; a trainer that watched the rollout live
        drops what it landed for that id."""
        with self._lock:
            events: list[Event] = [
                (
                    "episode_discarded",
                    {
                        "episode_index": index,
                        "episode_id": episode_id,
                        "reason": reason or "",
                    },
                )
            ]
            self._emit(events, write=False)
            return events

    # -- inbound: gate (tick thread; the executor hands the payloads over) -------------------------
    def on_gate_events(self, payloads: list[dict[str, Any]]) -> list[Event]:
        """Every ``TakeoverGate`` event of the session -> ``events.gate`` (payload
        ``{arm_id, mode, seq, source, episode_id}`` built by the executor). Publishing is
        a worker submit, so this is safe on the tick."""
        if not payloads:
            return []
        with self._lock:
            events: list[Event] = [("gate", dict(p)) for p in payloads]
            self._emit(events, write=False)
            return events

    # -- operator ------------------------------------------------------------------------------
    def request_train_now(self, episode_open: bool = False) -> tuple[bool, str]:
        """**Train now** (``ActionMsg train_now``; 15-online-dagger §3): ask the trainer to
        train on the rollouts saved so far. Refused while an episode is open and without
        a fresh trainer status (publishing to nobody would lose the request); the
        trainer may still ignore it."""
        with self._lock:
            if episode_open:
                return False, EPISODE_OPEN
            if not self._trainer_alive(None):
                return False, NO_TRAINER
            n = self.rollouts_saved
            events: list[Event] = [
                ("train_now", {"rollouts_saved": n, "requested_by": "operator"})
            ]
            self._emit(events, write=False)
            return True, f"asked the trainer to train ({n} rollout{'s' if n != 1 else ''} saved)"

    def refuse_episode_new(self, now: float | None = None) -> str | None:
        """Why ``episode_new`` is refused right now (15-online-dagger §3), None = allowed."""
        with self._lock:
            return self._refuse_locked(now)

    def _refuse_locked(self, now: float | None = None) -> str | None:
        if not self._trainer_alive(now):
            return NO_TRAINER
        if self.phase == "error":
            tr = self.trainer
            return f"trainer error: {(tr.detail if tr is not None else '') or 'unknown'}"
        if self.phase == "waiting_trainer":
            return f"waiting for the trainer to report ready ({self._trainer_detail()})"
        if self.phase == "training":
            return f"training in progress ({self._trainer_detail()})"
        return None

    def _trainer_detail(self) -> str:
        tr = self.trainer
        if tr is None:
            return NO_STATUS_YET
        if tr.session_id != self.session_id:
            return NOT_SERVING
        if tr.detail:
            return tr.detail
        if tr.state == "training":
            return f"{max(0.0, min(1.0, tr.progress)) * 100:.0f}%"
        return f"trainer {tr.state}"

    # -- freshness -----------------------------------------------------------------------------
    def trainer_age(self, now: float | None = None) -> float | None:
        with self._lock:
            if self._trainer_t is None:
                return None
            now = self._clock() if now is None else now
            return max(0.0, now - self._trainer_t)

    def trainer_alive(self, now: float | None = None) -> bool:
        with self._lock:
            return self._trainer_alive(now)

    def _trainer_alive(self, now: float | None) -> bool:
        if self._trainer_t is None or self.trainer is None:
            return False
        now = self._clock() if now is None else now
        return (now - self._trainer_t) <= self.spec_stale_s

    # -- status ----------------------------------------------------------------------------------
    def status(self, now: float | None = None) -> OnlineDaggerStatus:
        """``DaggerStatus.online_dagger`` (15-online-dagger §5)."""
        with self._lock:
            now = self._clock() if now is None else now
            tr = self.trainer
            detail = self._refuse_locked(now) or (tr.detail if tr is not None else "") or ""
            return OnlineDaggerStatus(
                session_name=self.cfg.session_name,
                phase=self.phase,  # type: ignore[arg-type]
                rollouts_saved=self.rollouts_saved,
                detail=detail,
                trainer_alive=self._trainer_alive(now),
                trainer_age_s=(
                    None if self._trainer_t is None else max(0.0, now - self._trainer_t)
                ),
                trainer=tr,
                policy_version_acting=self.policy_version_acting,
                expert_frames_session=self.expert_frames_session,
                novice_frames_session=self.novice_frames_session,
                session_dir=str(self.paths.session_dir),
            )

    # -- session.json ---------------------------------------------------------------------------
    def to_session_json(self) -> dict[str, Any]:
        with self._lock:
            p = self.paths
            return {
                "session_name": self.cfg.session_name,
                "created_at": self.created_at,
                "session_id": self.session_id,
                "task": self.task,
                "spec": self.spec.model_dump(mode="json") if self.spec is not None else None,
                "paths": {"session_dir": str(p.session_dir), "rollouts": str(p.rollouts_dir)},
                "rollouts": [dict(r) for r in self.rollouts],
                "trainer_log": [dict(r) for r in self.trainer_log[-TRAINER_LOG_MAX:]],
                "current": {
                    "phase": self.phase,
                    "rollouts_saved": self.rollouts_saved,
                    "expert_frames_session": self.expert_frames_session,
                    "novice_frames_session": self.novice_frames_session,
                },
                "last_used_at": self._now_iso(),
            }

    def from_session_json(self, doc: dict[str, Any]) -> None:
        """Resume (15-online-dagger §3): the counters and the rollouts rows continue; the
        phase restarts at the configured start (the new trainer must report ready again
        for THIS session id). Call BEFORE :meth:`on_session_start`."""
        with self._lock:
            self.resumed = True
            self.created_at = str(doc.get("created_at") or self.created_at)
            if self.task is None:
                self.task = doc.get("task")
            self.rollouts = [dict(r) for r in (doc.get("rollouts") or []) if isinstance(r, dict)]
            self.trainer_log = [
                dict(r) for r in (doc.get("trainer_log") or []) if isinstance(r, dict)
            ][-TRAINER_LOG_MAX:]
            cur = doc.get("current") or {}
            self.rollouts_saved = max(
                int(cur.get("rollouts_saved", len(self.rollouts)) or 0), len(self.rollouts)
            )
            self.expert_frames_session = int(cur.get("expert_frames_session", 0) or 0)
            self.novice_frames_session = int(cur.get("novice_frames_session", 0) or 0)
            self.phase = "waiting_trainer" if self.cfg.wait_for_trainer_ready else "rollout"
            self.trainer_seen_ready = False

    @staticmethod
    def read_session_info(path: Path) -> OnlineDaggerSessionInfo | None:
        """One ``session.json`` -> ``GET /api/online_dagger/sessions`` row (None = unreadable)."""
        path = Path(path)
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            logger.warning("online_dagger session file %s skipped: %s", path, e)
            return None
        if not isinstance(doc, dict):
            return None
        cur = doc.get("current") or {}
        spec = doc.get("spec") or {}
        od = (spec.get("online_dagger") or {}) if isinstance(spec, dict) else {}
        rows = doc.get("rollouts") or []
        name = str(doc.get("session_name") or od.get("session_name") or path.parent.name)
        try:
            return OnlineDaggerSessionInfo(
                session_name=name,
                path=str(path.parent),
                created_at=str(doc.get("created_at") or _iso(path.stat().st_mtime)),
                task=doc.get("task") if doc.get("task") is not None else spec.get("task"),
                rollouts=int(cur.get("rollouts_saved", len(rows)) or 0),
                last_used_at=doc.get("last_used_at"),
            )
        except Exception as e:  # noqa: BLE001 - one bad file never 500s the listing
            logger.warning("online_dagger session file %s skipped: %s", path, e)
            return None

    @classmethod
    def scan_sessions(cls, root: Path | None) -> list[OnlineDaggerSessionInfo]:
        """``GET /api/online_dagger/sessions``: every ``<root>/*/session.json``, newest
        ``last_used_at`` first; a missing root lists nothing."""
        if root is None:
            return []
        root = Path(root)
        if not root.is_dir():
            return []
        rows: list[OnlineDaggerSessionInfo] = []
        for path in sorted(root.glob(f"*/{SESSION_JSON}")):
            info = cls.read_session_info(path)
            if info is not None:
                rows.append(info)
        rows.sort(key=lambda r: (r.last_used_at or r.created_at), reverse=True)
        return rows

    # -- trainer log (under the lock) --------------------------------------------------------------
    def _log_trainer_state(self, msg: TrainerStatusAnnounce) -> bool:
        if msg.state == self._logged_state:
            return False
        self._logged_state = msg.state
        self._log(msg.state, int(msg.policy_version), msg.detail)
        return True

    def _log(self, state: str, policy_version: int | None, detail: str) -> None:
        self.trainer_log.append(
            {
                "at": self._now_iso(),
                "state": state,
                "policy_version": policy_version,
                "detail": detail or "",
            }
        )
        if len(self.trainer_log) > TRAINER_LOG_MAX:
            del self.trainer_log[: len(self.trainer_log) - TRAINER_LOG_MAX]

    # -- side effects ----------------------------------------------------------------------------
    def _emit(self, events: list[Event], write: bool) -> None:
        if self._publish is not None:
            for kind, payload in events:
                self._publish(kind, payload)
        self.events_published += len(events)
        if write and self._write_session is not None:
            self._last_write_t = self._clock()
            self._write_session(self.to_session_json())

    def _write_due(self) -> bool:
        return self._clock() - self._last_write_t >= self._write_period

    def _now_iso(self) -> str:
        return _iso(self._wallclock())


__all__ = [
    "EPISODE_OPEN",
    "NO_STATUS_YET",
    "NOT_SERVING",
    "NO_TRAINER",
    "PHASES",
    "ROLLOUTS_DIR",
    "SESSION_JSON",
    "TRAINER_LOG_MAX",
    "OnlineDaggerCoordinator",
    "OnlineDaggerPaths",
    "SerialWorker",
    "write_session_json_atomic",
]
