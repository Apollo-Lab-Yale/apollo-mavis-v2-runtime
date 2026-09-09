"""``fake_policy`` — the runtime's own stand-in for the policy repo (14-dora §6.2, §10).

numpy + pyarrow + dora only (never imports the runtime package at run time, so it
behaves like a foreign node). Attaches as the ``policy`` placeholder and follows
the §6.2 loop: cache ``session`` -> publish ``spec``; on every ``obs_state``
(rate-limited to ``--rate-hz``) publish ``action`` echoing ``observation_id``;
1 Hz ``spec`` heartbeat; ``policy_reset`` drops the pending chunk and ignores
older observations; ``INPUT_CLOSED`` / ``STOP`` -> re-attach (never exit).

Modes: ``echo`` (zeros, K = 1: the RTT instrument), ``scripted`` (small
deterministic sinusoidal deltas), ``nan`` (a NaN row at the ``--nan-at`` act
indices), ``delay`` (``--delay-ms`` before each action), ``chunk`` (K =
``--chunk`` rows), ``collide`` (constant +x delta ``--collide-mps`` m/s towards
an obstacle), ``hold`` (never sends actions: spec only). Test knobs:
``--reset-violate`` (after a ``policy_reset`` keep sending chunks whose
``observation_id <= after_observation_id`` for 2 s), ``--version-bump-after N``
(policy_version += 1 after N actions), ``--stale-obs`` (echo an observation_id
that is ``--stale-obs`` messages old), ``--client`` (metadata ``client``).

**Trainer role** (phase-14; 15-online-dagger §6, §10) — ``FAKE_TRAINER=1`` (or
``--trainer``) plays the policy repo's generic Online DAgger trainer without learning
anything: ``spec.capabilities = ["online_dagger"]``; a ``session`` announce with an
``online_dagger`` block and state ``running`` starts it: ``trainer_status``
``preparing`` (progress 0 -> 1 over ``FAKE_TRAINER_PREPARE_S``, default 0.5 s) then
``ready``; every ``events.episode_saved`` of THIS session counts one kept rollout
(``online_dagger.actor_counts`` are the buffer's expert / novice frames; an
``episode_discarded`` id is remembered and never counted) and after every
``FAKE_TRAINER_EVERY`` (default 2) new rollouts — or on ``events.train_now`` — it
publishes ``training`` (progress over ``FAKE_TRAINER_TRAIN_S``, default 1.0 s; metrics
``loss`` decreasing, ``n_expert_frames``, ``rollouts_trained``; the buffer accumulates
every kept rollout, 15-online-dagger §0 item 7), bumps the acting policy's
``spec.version`` (the ``policy_version`` of every later action too) and publishes
``ready`` with the new version. Statuses heartbeat at 1 Hz while a session is served
and every one echoes the served ``session_id``; the session's end publishes one final
``idle``. ``FAKE_TRAINER_FAIL_AT=k`` publishes ``state: "error"`` at the k-th training
(the next training recovers). Everything slow runs on ONE worker thread; the statuses
it produces are queued and sent on the node thread (``send_output`` and the ``seq``
counter stay single-threaded). It knows no DAgger variant and writes nothing to disk.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import threading
import time
from typing import Any

import numpy as np

from . import node_env_defaults

MAVIS_SCHEMA = 1
NODE_VERSION = "runtime-fake-0.1.0"

# -- trainer role constants (15-online-dagger §6, §10) --------------------------------------------
TRAINER_ID = "runtime-fake/online_dagger"
ONLINE_DAGGER_CAPABILITY = "online_dagger"
RUNNING_STATES = ("running",)
PRE_RUNNING_STATES = ("bringup", "start_from")
ENDING_STATES = ("idle", "teardown")
PROGRESS_SLICES = 10  # progress steps of a prepare / training job


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v is not None and v.strip() else default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None and v.strip() else default


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None or not v.strip():
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fake_policy")
    p.add_argument(
        "--daemon-port", type=int, default=int(os.environ.get("DORA_DAEMON_PORT", "53391"))
    )
    p.add_argument("--node-id", default="policy")
    p.add_argument(
        "--mode",
        default="scripted",
        choices=["echo", "scripted", "nan", "delay", "chunk", "collide", "hold"],
    )
    p.add_argument("--rate-hz", type=float, default=15.0)
    p.add_argument("--policy-id", default="fake-policy")
    p.add_argument("--version", type=int, default=1)
    p.add_argument("--nan-at", default="", help="comma list of act indices returning NaN")
    p.add_argument("--chunk", type=int, default=1)
    p.add_argument("--chunk-dt-s", type=float, default=None)
    p.add_argument("--delay-ms", type=float, default=0.0)
    p.add_argument("--collide-mps", type=float, default=0.15)
    p.add_argument("--collide-axis", default="-z", help="recording-frame axis, e.g. -z, +x")
    p.add_argument("--amplitude-m", type=float, default=0.002)
    p.add_argument("--reset-violate", action="store_true")
    p.add_argument("--version-bump-after", type=int, default=0)
    p.add_argument("--stale-obs", type=int, default=0)
    p.add_argument("--client", default="fake_policy")
    p.add_argument(
        "--action-frame", default=None, help="override spec.action_frame (mismatch tests)"
    )
    p.add_argument("--session-id", default=None, help="override echoed session_id (drop tests)")
    p.add_argument("--duration-s", type=float, default=0.0, help="0 = run until killed")
    p.add_argument("--stats-out", default=None, help="write JSON stats here at exit")
    p.add_argument(
        "--trainer",
        action="store_true",
        default=_env_bool("FAKE_TRAINER"),
        help="Online DAgger trainer role (15-online-dagger §10); also FAKE_TRAINER=1",
    )
    return p


# -- the trainer role -----------------------------------------------------------------------------
def _finite(v: Any) -> float | None:
    """A finite float or None (``TrainerStatusAnnounce`` refuses inf / nan)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


class FakeTrainerRole:
    """See the module docstring ("Trainer role"). ``on_session`` / ``on_event`` / ``pump``
    run on the node thread; the jobs run on ONE worker thread and only push status dicts
    onto ``_out`` (drained by ``pump``) - the ``Node`` is never touched off-thread."""

    def __init__(self, node: FakePolicyNode) -> None:
        self.node = node
        self.prepare_s = _env_float("FAKE_TRAINER_PREPARE_S", 0.5)
        self.train_s = _env_float("FAKE_TRAINER_TRAIN_S", 1.0)
        self.every = max(1, _env_int("FAKE_TRAINER_EVERY", 2))
        self.fail_at = _env_int("FAKE_TRAINER_FAIL_AT", 0)
        self.heartbeat_s = 1.0
        self.progress_min_interval_s = 0.25
        self._lock = threading.RLock()
        self._jobs: queue.Queue[tuple[str, Any] | None] = queue.Queue()
        self._out: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._worker: threading.Thread | None = None
        self._busy = threading.Event()
        self._closed = False
        self._t0 = time.monotonic()
        self._last_publish_t = -1e9
        self.spec_dirty = False  # the acting version changed: the node re-publishes `spec`
        # status (TrainerStatusAnnounce, the generic 10 fields)
        self.state = "idle"
        self.detail = ""
        self.session_id: str | None = None
        self.policy_version = int(node.version)
        self.progress = 0.0
        self.metrics: dict[str, float] = {}
        # the served session + its rollout buffer (accumulates every kept rollout)
        self.ctx: dict[str, Any] | None = None
        self._rollouts: list[dict[str, Any]] = []  # {episode_id, n_expert, n_novice, spool_path}
        self._trained_upto = 0  # len(_rollouts) the last training consumed
        self._discarded: set[str] = set()
        self._trainings = 0
        self._loss = 1.0  # fabricated, decreasing across the session
        self.stats = {
            "sessions": 0,
            "prepares": 0,
            "rollouts_counted": 0,
            "trainings": 0,
            "train_now": 0,
            "discards": 0,
            "swaps": 0,
            "errors": 0,
            "published": 0,
            "ignored_events": 0,
        }

    # -- public (node thread) ------------------------------------------------------------------
    @property
    def active(self) -> bool:
        return self.ctx is not None

    @property
    def busy(self) -> bool:
        return self._busy.is_set() or not self._jobs.empty()

    def on_session(self, ann: dict[str, Any]) -> None:
        try:
            self._on_session(ann)
        except Exception as e:  # noqa: BLE001 - never take the node down
            self._fail(f"on_session: {e!r}")

    def on_event(self, env: dict[str, Any]) -> None:
        try:
            self._on_event(env)
        except Exception as e:  # noqa: BLE001
            self._fail(f"on_event: {e!r}")

    def pump(self, now: float) -> list[str]:
        """The queued ``trainer_status`` JSON texts to send now (+ the 1 Hz heartbeat)."""
        if (self.active or self.busy) and now - self._last_publish_t >= self.heartbeat_s:
            self._publish(force=True)
        out: list[str] = []
        while True:
            try:
                out.append(self._out.get_nowait())
            except queue.Empty:
                break
        return out

    def take_spec_dirty(self) -> bool:
        with self._lock:
            dirty, self.spec_dirty = self.spec_dirty, False
            return dirty

    def close(self) -> None:
        self._closed = True
        w = self._worker
        if w is not None:
            self._jobs.put(None)
            w.join(timeout=2.0)

    # -- announce / events ---------------------------------------------------------------------
    @staticmethod
    def _context_of(ann: dict[str, Any]) -> dict[str, Any] | None:
        """The served session: ``SessionAnnounce.online_dagger`` + ``session_id``."""
        od = ann.get("online_dagger")
        sid = ann.get("session_id")
        if not isinstance(od, dict) or not sid:
            return None
        return {
            "session_id": str(sid),
            "session_name": str(od.get("session_name") or ""),
            "session_dir": str(od.get("session_dir") or ""),
            "rollouts_dir": str(od.get("rollouts_dir") or ""),
        }

    def _on_session(self, ann: dict[str, Any]) -> None:
        ctx = self._context_of(ann)
        state = str(ann.get("state") or "")
        sid = ann.get("session_id")
        if ctx is not None and state in RUNNING_STATES:
            if self.ctx is not None and self.ctx["session_id"] == ctx["session_id"]:
                return  # the 1 Hz heartbeat of the served session
            with self._lock:
                self.ctx = ctx
                self.stats["sessions"] += 1
                self.session_id = ctx["session_id"]
                self.state = "preparing"  # the first status already says what comes next
                self.progress = 0.0
                self.detail = "warming up"
                self.metrics = {}
                self._rollouts = []
                self._trained_upto = 0
                self._discarded = set()
                self._trainings = 0
                self._loss = 1.0
            self._submit("prepare", ctx)
            self._publish(force=True)
            return
        if self.ctx is None:
            return
        same = bool(sid) and str(sid) == self.ctx["session_id"]
        if same and state not in ENDING_STATES and (ctx is not None or state in PRE_RUNNING_STATES):
            return  # a transient state of the served session: keep the context
        with self._lock:
            self.ctx = None
            if not self.busy:
                self._reset_idle()
        self._publish(force=True)

    def _on_event(self, env: dict[str, Any]) -> None:
        kind = str(env.get("kind") or "")
        if kind not in ("episode_saved", "episode_discarded", "train_now"):
            return
        payload = env.get("payload") or {}
        ctx = self.ctx
        if ctx is None or not isinstance(payload, dict):
            self.stats["ignored_events"] += 1
            return
        ev_sid = str(env.get("session_id") or "")
        if ev_sid and ev_sid != ctx["session_id"]:
            self.stats["ignored_events"] += 1
            return
        if kind == "episode_discarded":
            eid = payload.get("episode_id")
            with self._lock:
                if eid:
                    self._discarded.add(str(eid))
                    # a discard never follows a save; should one, the rollout leaves the buffer
                    self._rollouts = [r for r in self._rollouts if r["episode_id"] != str(eid)]
                    self._trained_upto = min(self._trained_upto, len(self._rollouts))
                self.stats["discards"] += 1
            return
        if kind == "episode_saved":
            block = payload.get("online_dagger")
            block = block if isinstance(block, dict) else {}
            summary = payload.get("summary")
            summary = summary if isinstance(summary, dict) else {}
            eid = block.get("episode_id") or summary.get("episode_id")
            with self._lock:
                known = {r["episode_id"] for r in self._rollouts}
                if not eid or str(eid) in self._discarded or str(eid) in known:
                    self.stats["ignored_events"] += 1
                    return
                counts = block.get("actor_counts")
                counts = counts if isinstance(counts, dict) else {}
                n_expert = counts.get("expert", summary.get("n_expert_frames", 0))
                n_novice = counts.get("novice", summary.get("n_novice_frames", 0))
                spool = block.get("spool_path") or payload.get("spool_path") or ""
                self._rollouts.append(
                    {
                        "episode_id": str(eid),
                        "n_expert": int(n_expert or 0),
                        "n_novice": int(n_novice or 0),
                        "spool_path": str(spool),
                    }
                )
                self.stats["rollouts_counted"] += 1
                pending = len(self._rollouts) - self._trained_upto
            if pending >= self.every:
                self._submit(
                    "train",
                    {"session_id": ctx["session_id"], "reason": f"{pending} new rollouts"},
                )
            return
        # train_now: the operator asked; the buffer must hold something to train on
        self.stats["train_now"] += 1
        with self._lock:
            empty = not self._rollouts
            if empty:
                self.detail = "train_now ignored: no rollouts saved yet"
        if empty:
            self._publish(force=True)
            return
        self._submit("train", {"session_id": ctx["session_id"], "reason": "train_now"})

    # -- worker --------------------------------------------------------------------------------
    def _submit(self, kind: str, arg: Any) -> None:
        if self._closed:
            return
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(
                target=self._run_worker, name="fake-trainer", daemon=True
            )
            self._worker.start()
        self._jobs.put((kind, arg))

    def _run_worker(self) -> None:
        while True:
            try:
                job = self._jobs.get(timeout=0.5)
            except queue.Empty:
                if self._closed:
                    return
                continue
            if job is None:
                return
            kind, arg = job
            self._busy.set()
            try:
                if kind == "prepare":
                    self._do_prepare(arg)
                else:
                    self._do_train(arg)
            except Exception as e:  # noqa: BLE001 - errors become status, never exceptions
                self._fail(f"{kind}: {e!r}"[:400])
            finally:
                self._busy.clear()
                self._jobs.task_done()
                self._settle_after_job()

    def _settle_after_job(self) -> None:
        with self._lock:
            if self.ctx is not None or not self._jobs.empty() or self.session_id is None:
                return
            self._reset_idle()
        self._publish(force=True)

    def _reset_idle(self) -> None:
        self.state = "idle"
        self.session_id = None
        self.detail = ""
        self.progress = 0.0
        self.metrics = {}

    def _serving(self, session_id: str) -> bool:
        return self.ctx is not None and self.ctx["session_id"] == session_id

    def _do_prepare(self, ctx: dict[str, Any]) -> None:
        with self._lock:
            if not self._serving(ctx["session_id"]):
                return  # the session ended before we got to it
            self.state = "preparing"
            self.progress = 0.0
            self.detail = "warming up"
        self.stats["prepares"] += 1
        self._publish(force=True)
        for k in range(PROGRESS_SLICES):
            time.sleep(max(0.0, self.prepare_s) / PROGRESS_SLICES)
            with self._lock:
                self.progress = (k + 1) / PROGRESS_SLICES
                self.detail = f"warming up {self.progress * 100:.0f}%"
            self._publish()
        with self._lock:
            if not self._serving(ctx["session_id"]):
                return
            self.state = "ready"
            self.progress = 1.0
            self.detail = f"ready (policy v{self.policy_version}, {len(self._rollouts)} rollouts)"
        self._publish(force=True)

    def _do_train(self, request: dict[str, Any]) -> None:
        sid = str(request["session_id"])
        reason = str(request.get("reason") or "")
        with self._lock:
            if not self._serving(sid):
                return
            rollouts = list(self._rollouts)
            if len(rollouts) <= self._trained_upto and reason != "train_now":
                return  # a queued duplicate: the previous training already took these
            self._trainings += 1
            k = self._trainings
            self._trained_upto = len(rollouts)
            n_expert = sum(int(r["n_expert"]) for r in rollouts)
            self.state = "training"
            self.progress = 0.0
            self.detail = f"training #{k} on {len(rollouts)} rollout(s): {reason}"
            self.metrics = {
                "loss": self._loss,
                "n_expert_frames": float(n_expert),
                "rollouts_trained": float(len(rollouts)),
            }
        self.stats["trainings"] += 1
        self._publish(force=True)
        if self.fail_at and k == int(self.fail_at):
            raise RuntimeError(f"FAKE_TRAINER_FAIL_AT={self.fail_at}")
        loss0 = self._loss
        for s in range(PROGRESS_SLICES):
            time.sleep(max(0.0, self.train_s) / PROGRESS_SLICES)
            frac = (s + 1) / PROGRESS_SLICES
            with self._lock:
                self.progress = frac
                self._loss = loss0 * (0.7 + 0.3 * (1.0 - frac))  # -> 0.7 x per training
                self.metrics["loss"] = self._loss
            self._publish()
        # the swap: the acting policy's version bumps; the node re-publishes `spec` and every
        # later action carries the new policy_version (the runtime's acting version, §3)
        with self._lock:
            if not self._serving(sid):
                return
            self.node.version = int(self.node.version) + 1
            version = int(self.node.version)
            self.spec_dirty = True
            self.stats["swaps"] += 1
            self.policy_version = version
            self.state = "ready"
            self.progress = 1.0
            self.detail = (
                f"trained on {len(rollouts)} rollout(s) ({n_expert} expert frames); "
                f"policy v{version}"
            )
            self.metrics = {
                "loss": self._loss,
                "n_expert_frames": float(n_expert),
                "rollouts_trained": float(len(rollouts)),
            }
        self._publish(force=True)

    # -- status ----------------------------------------------------------------------------------
    def _fail(self, detail: str) -> None:
        with self._lock:
            self.state = "error"
            self.detail = detail
            self.stats["errors"] += 1
        print(f"fake_policy trainer: error {detail}", file=sys.stderr)
        self._publish(force=True)

    def _publish(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_publish_t < self.progress_min_interval_s:
            return
        self._last_publish_t = now
        self._out.put(json.dumps(self.status()))
        self.stats["published"] += 1

    def status(self) -> dict[str, Any]:
        """The ``TrainerStatusAnnounce`` JSON object (core field order, the 10 fields)."""
        with self._lock:
            metrics = {}
            for key, value in self.metrics.items():
                f = _finite(value)
                if f is not None:
                    metrics[str(key)] = f
            return {
                "mavis_schema": MAVIS_SCHEMA,
                "trainer_id": TRAINER_ID,
                "node_version": NODE_VERSION,
                "state": self.state,
                "session_id": self.session_id,
                "policy_version": int(self.policy_version),
                "progress": min(1.0, max(0.0, _finite(self.progress) or 0.0)),
                "metrics": metrics,
                "detail": str(self.detail),
                "uptime_s": max(0.0, time.monotonic() - self._t0),
            }


class FakePolicyNode:
    def __init__(self, args: argparse.Namespace) -> None:
        self.a = args
        self.session: dict | None = None
        self.seq = 0
        self.acts = 0
        self.version = int(args.version)
        self.watermark = 0
        self.violate_until = 0.0
        self.recent_obs: list[int] = []
        self.next_act_t = 0.0
        self.next_spec_t = 0.0
        self.nan_at = {int(x) for x in args.nan_at.split(",") if x.strip()}
        self.t0 = time.monotonic()
        self.stats = {"actions": 0, "specs": 0, "obs": 0, "resets": 0, "reattach": 0}
        # phase-14: the trainer role (FAKE_TRAINER=1 / --trainer); None = a plain policy
        self.trainer: FakeTrainerRole | None = FakeTrainerRole(self) if args.trainer else None
        self.poll_s = 0.1 if self.trainer is not None else 0.5

    # -- messages ---------------------------------------------------------------------------------
    def spec_json(self) -> str:
        s = self.session or {}
        frame = self.a.action_frame
        if frame is None:
            frames = s.get("frames") or {}
            frame = next(iter(frames.values()), "arm_base:grip")
        return json.dumps(
            {
                "mavis_schema": MAVIS_SCHEMA,
                "policy_id": self.a.policy_id,
                "policy_version": self.version,
                "node_version": NODE_VERSION,
                "spec": {
                    "action_space": s.get("action_space") or "delta_ee",
                    "action_frame": frame,
                    "action_names": list(s.get("action_names") or []),
                    "state_names": list(s.get("state_names") or []),
                    "camera_keys": [],
                    "version": self.version,
                },
                "rate_hz": float(self.a.rate_hz),
                "chunk_len": int(self.a.chunk) if self.a.mode == "chunk" else 1,
                "chunk_dt_s": self.a.chunk_dt_s,
                "loader": "custom",
                "device": "cpu",
                "health": "ok",
                "detail": f"mode {self.a.mode}",
                "uptime_s": time.monotonic() - self.t0,
                "acts_total": self.acts,
                # phase-14 (15-online-dagger §6): a trainer-capable node lists "online_dagger"
                "capabilities": [ONLINE_DAGGER_CAPABILITY] if self.trainer is not None else [],
            }
        )

    def act(self, dim: int, k: int) -> np.ndarray:
        rows = np.zeros((k, dim), dtype=np.float32)
        if self.a.mode in ("echo", "hold"):
            pass
        elif self.a.mode == "collide":
            per_step = self.a.collide_mps / float(self.a.rate_hz)
            axis = self.a.collide_axis.strip()
            sign = -1.0 if axis.startswith("-") else 1.0
            idx = "xyz".index(axis[-1].lower())
            rows[:, idx] = sign * per_step  # a constant recording-frame delta every period
        else:
            t = self.acts / float(self.a.rate_hz)
            for r in range(k):
                ph = t + r / float(self.a.rate_hz)
                rows[r, 0] = self.a.amplitude_m * math.sin(2 * math.pi * 0.5 * ph)
                rows[r, 1] = self.a.amplitude_m * math.cos(2 * math.pi * 0.5 * ph)
                if dim > 2:
                    rows[r, 2] = 0.5 * self.a.amplitude_m * math.sin(2 * math.pi * 0.25 * ph)
        # gripper dims (every 7th when the layout is per-arm 6 + grip [+ rail]) stay absolute 1.0
        names = list((self.session or {}).get("action_names") or [])
        for i, n in enumerate(names):
            if n.endswith("gripper.pos") and i < dim:
                rows[:, i] = 1.0
        if self.a.mode == "nan" and self.acts in self.nan_at:
            rows[:] = np.nan
        return rows

    # -- loop -------------------------------------------------------------------------------------
    def run(self) -> int:
        node_env_defaults()
        import signal

        import pyarrow as pa  # noqa: TID251 - node process
        from dora import Node  # noqa: TID251 - node process

        def _term(signum, frame):  # tests SIGTERM the node: leave the stats behind
            self._dump()
            raise SystemExit(0)

        signal.signal(signal.SIGTERM, _term)

        backoff = 1.0
        deadline = self.t0 + self.a.duration_s if self.a.duration_s > 0 else None
        while True:
            try:
                node = Node(self.a.node_id, daemon_port=self.a.daemon_port)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"fake_policy: attach failed ({exc}); retry in {backoff:.0f} s", file=sys.stderr
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 10.0)
                continue
            backoff = 1.0
            rc = self._serve(node, pa, deadline)
            del node
            if rc is not None:
                self._dump()
                return rc
            self.stats["reattach"] += 1
            time.sleep(1.0)

    def _send(self, node, pa, oid: str, arr, meta: dict) -> None:
        self.seq += 1
        base = {
            "mavis_schema": MAVIS_SCHEMA,
            "session_id": self.a.session_id
            if self.a.session_id is not None
            else str((self.session or {}).get("session_id") or ""),
            "client": f"{self.a.client}#{os.getpid()}",  # unique per process instance
            "seq": self.seq,
            "t_mono": time.monotonic(),
            "wallclock_ns": time.time_ns(),
        }
        base.update(meta)
        node.send_output(oid, arr, base)

    def _publish_spec(self, node, pa) -> None:
        if self.session is None:
            return  # the fake learns its layout from the runtime's `session` announce first
        self._send(node, pa, "spec", pa.array([self.spec_json()], type=pa.string()), {})
        self.stats["specs"] += 1

    def _pump_trainer(self, node, pa, now: float) -> None:
        """Node thread: a swapped version re-publishes ``spec`` first, then every queued
        ``trainer_status`` goes out (the worker never touches the ``Node``)."""
        tr = self.trainer
        if tr is None:
            return
        if tr.take_spec_dirty():
            self._publish_spec(node, pa)
            self.next_spec_t = now + 1.0
        for text in tr.pump(now):
            self._send(node, pa, "trainer_status", pa.array([text], type=pa.string()), {})

    def _serve(self, node, pa, deadline: float | None) -> int | None:
        self._publish_spec(node, pa)
        self.next_spec_t = time.monotonic() + 1.0
        while True:
            if deadline is not None and time.monotonic() > deadline:
                return 0
            ev = node.next(timeout=self.poll_s)
            now = time.monotonic()
            if now >= self.next_spec_t:
                self._publish_spec(node, pa)
                self.next_spec_t = now + 1.0
            self._pump_trainer(node, pa, now)
            if ev is None:
                return None  # all senders dropped -> re-attach
            kind = ev.get("type")
            if kind == "ERROR":
                text = str(ev.get("error") or "")
                if "Receiver timed out" in text:
                    continue
                print(f"fake_policy: ERROR {text[:200]}", file=sys.stderr)
                if "daemon channel broken" in text or "fatal" in text:
                    return None
                continue
            if kind == "STOP":
                return None
            if kind == "INPUT_CLOSED":
                continue
            if kind != "INPUT":
                continue
            iid = ev.get("id")
            meta = ev.get("metadata") or {}
            if iid == "session":
                try:
                    self.session = json.loads(ev["value"][0].as_py())
                except Exception:  # noqa: BLE001
                    continue
                self._publish_spec(node, pa)
                self.next_spec_t = now + 1.0
                if self.trainer is not None:
                    self.trainer.on_session(self.session)
                    self._pump_trainer(node, pa, now)
            elif iid == "events":
                if self.trainer is not None:
                    try:
                        env = json.loads(ev["value"][0].as_py())
                    except Exception:  # noqa: BLE001
                        continue
                    self.trainer.on_event(env)
                    self._pump_trainer(node, pa, now)
            elif iid == "policy_reset":
                try:
                    msg = json.loads(ev["value"][0].as_py())
                    self.watermark = int(msg.get("after_observation_id", 0))
                except Exception:  # noqa: BLE001
                    continue
                self.stats["resets"] += 1
                if self.a.reset_violate:
                    self.violate_until = now + 2.0
            elif iid == "obs_state":
                self.stats["obs"] += 1
                self._on_obs(node, pa, meta, now)

    def _on_obs(self, node, pa, meta: dict, now: float) -> None:
        if self.a.mode == "hold" or self.session is None:
            return
        oid = int(meta.get("observation_id", 0))
        self.recent_obs.append(oid)
        del self.recent_obs[:-64]
        # phase-locked rate limit with a quarter-obs tolerance: obs at 30 Hz vs acts at 15 Hz
        # otherwise alias to 10 Hz (an obs landing 0.1 ms before the deadline is skipped)
        if now + 0.25 / 30.0 < self.next_act_t:
            return
        self.next_act_t = now + 1.0 / float(self.a.rate_hz)
        names = list(self.session.get("action_names") or [])
        dim = len(names) or int(meta.get("action_dim", 8))
        k = int(self.a.chunk) if self.a.mode == "chunk" else 1
        if self.a.mode == "delay" and self.a.delay_ms > 0:
            time.sleep(self.a.delay_ms / 1000.0)
        rows = self.act(dim, k)
        echo = oid
        if now < self.violate_until:
            echo = max(1, self.watermark)  # deliberately at/below the watermark
        elif self.a.stale_obs > 0 and len(self.recent_obs) > self.a.stale_obs:
            echo = self.recent_obs[-1 - self.a.stale_obs]
        elif oid <= self.watermark:
            return  # older than the reset watermark: never act on it
        self.acts += 1
        if self.a.version_bump_after and self.acts == self.a.version_bump_after:
            self.version += 1
            self._publish_spec(node, pa)
        chunk_dt = float(self.a.chunk_dt_s) if self.a.chunk_dt_s else 1.0 / float(self.a.rate_hz)
        self._send(
            node,
            pa,
            "action",
            pa.array(rows.reshape(-1), type=pa.float32()),
            {
                "observation_id": int(echo),
                "chunk_len": int(k),
                "action_dim": int(dim),
                "chunk_dt_s": chunk_dt,
                "policy_id": self.a.policy_id,
                "policy_version": int(self.version),
                "compute_ms": float(self.a.delay_ms),
                "finite": bool(np.all(np.isfinite(rows))),
                "image_seq_used": [int(x) for x in (meta.get("image_seq") or [])],
            },
        )
        self.stats["actions"] += 1

    def _dump(self) -> None:
        if self.trainer is not None:
            self.stats["trainer"] = dict(self.trainer.stats)
        if self.a.stats_out:
            with open(self.a.stats_out, "w", encoding="utf-8") as fh:
                json.dump(self.stats, fh)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return FakePolicyNode(args).run()


if __name__ == "__main__":
    sys.exit(main())
