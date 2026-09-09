"""AsyncTrainerClientImpl — spawn/monitor/talk to the trainer process (12-dagger §7).

Transport: ZMQ REQ/REP on ``tcp://127.0.0.1:{port}`` with hard timeouts (REQ
sockets are recreated after every timeout). A worker thread polls status at
1 Hz and drains a submit queue; the control loop never blocks on the trainer.
Death handling (§12): ``proc.poll() != None`` or 3 missed replies -> state
``dead`` -> ONE auto-restart with ``--resume``; a second death stays degraded.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

from apollo_mavis_v2_core.dagger import CheckpointInfo, EpisodeSummary, TrainerStatus
from pydantic import BaseModel

from .trainer.checkpoints import CheckpointStore

logger = logging.getLogger(__name__)

REQ_TIMEOUT_S = 2.0
STATUS_PERIOD_S = 1.0
MISSED_REPLIES_DEAD = 3


class TrainerConfig(BaseModel):
    """The ``--config`` JSON contract between runtime and trainer process."""

    run_id: str
    checkpoints_root: str
    spool_dir: str  # trainer_spool/ep_*.parquet (recorder writes, trainer reads)
    seed_parquet: str | None = None  # seed BC frames (aggregate half)
    seed_bundle: str | None = None  # initial policy weights (v0)
    port: int = 5757
    device: str = "cuda:1"  # CUDA_VISIBLE_DEVICES set by the spawner; GPU 1 on target
    cuda_visible_devices: str | None = "1"
    action_frame: str = ""
    action_space: str = "delta_ee"
    state_dim: int = 0
    action_dim: int = 0
    hidden: int = 64
    min_new_labels: int = 100
    batch_size: int = 64
    lr: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    push_period_s: float = 5.0
    held_out_frames: int = 256
    orphan_timeout_s: float = 120.0


class AsyncTrainerClientImpl:
    """Implements the core ``AsyncTrainerClient`` Protocol."""

    def __init__(
        self,
        cfg: TrainerConfig,
        workdir: Path,
        store: CheckpointStore | None = None,
        stop_grace_s: float = 30.0,
        kill_grace_s: float = 45.0,
        spawn: bool = True,
    ) -> None:
        self.cfg = cfg
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.store = store or CheckpointStore(cfg.checkpoints_root, cfg.run_id)
        self.stop_grace_s = stop_grace_s
        self.kill_grace_s = kill_grace_s
        self._cfg_path = self.workdir / "trainer_config.json"
        self._cfg_path.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
        self.proc: subprocess.Popen | None = None
        self.restarts = 0
        self.degraded = False  # second death: stay degraded, never restart again
        self._status: TrainerStatus | None = None
        self._missed = 0
        self._known_version = 0
        self._queue: queue.Queue[dict] = queue.Queue()
        self._running = False
        self._thread: threading.Thread | None = None
        self._zmq_ctx = None
        if spawn:
            self._spawn(resume=False)
        self._running = True
        self._thread = threading.Thread(target=self._run, name="trainer-client", daemon=True)
        self._thread.start()

    # -- process ---------------------------------------------------------------
    def _spawn(self, resume: bool) -> None:
        cmd = [
            sys.executable,
            "-m",
            "apollo_mavis_v2_runtime.dagger.trainer",
            "--config",
            str(self._cfg_path),
        ]
        if resume:
            cmd.append("--resume")
        env = dict(os.environ)
        if self.cfg.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.cfg.cuda_visible_devices
        self.proc = subprocess.Popen(cmd, env=env)
        self._missed = 0
        logger.info("trainer spawned pid=%s resume=%s", self.proc.pid, resume)

    def _handle_death(self) -> None:
        if self.degraded:
            return
        if self.restarts >= 1:  # never a third silent restart (§12)
            self.degraded = True
            self._status = TrainerStatus(state="dead")
            logger.error("trainer died again; staying degraded")
            return
        self.restarts += 1
        self._status = TrainerStatus(state="dead")
        logger.error("trainer died; auto-restarting once with --resume")
        try:
            self._spawn(resume=True)
        except Exception:
            logger.exception("trainer restart failed")
            self.degraded = True

    # -- worker ---------------------------------------------------------------------
    def _run(self) -> None:
        next_status = 0.0
        while self._running:
            proc = self.proc
            if proc is not None and proc.poll() is not None and not self.degraded:
                self._handle_death()
            try:
                msg = self._queue.get(timeout=0.1)
            except queue.Empty:
                msg = None
            if msg is not None:
                self._request(msg)
            now = time.monotonic()
            if now >= next_status:
                next_status = now + STATUS_PERIOD_S
                reply = self._request({"cmd": "status"})
                if reply is not None and "status" in reply:
                    self._status = TrainerStatus(**reply["status"])
                    self._missed = 0
                else:
                    self._missed += 1
                    if self._missed >= MISSED_REPLIES_DEAD:
                        alive = self.proc is not None and self.proc.poll() is None
                        if not alive:
                            self._handle_death()
                        # else: trainer-busy (long burst); keep waiting (§12)

    def _request(self, msg: dict, timeout_s: float = REQ_TIMEOUT_S) -> dict | None:
        """One REQ/REP exchange; socket per request (safe after timeouts)."""
        import zmq

        if self._zmq_ctx is None:
            self._zmq_ctx = zmq.Context.instance()
        sock = self._zmq_ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
        sock.setsockopt(zmq.SNDTIMEO, int(timeout_s * 1000))
        try:
            sock.connect(f"tcp://127.0.0.1:{self.cfg.port}")
            sock.send_json(msg)
            return sock.recv_json()
        except Exception:
            return None
        finally:
            sock.close()

    # -- AsyncTrainerClient Protocol --------------------------------------------------
    def submit_episode(self, episode_path: str, summary: EpisodeSummary) -> None:
        self._queue.put(
            {"cmd": "submit_episode", "episode_path": episode_path, "summary": asdict(summary)}
        )

    def poll_checkpoint(self) -> CheckpointInfo | None:
        latest = self.store.latest()
        if latest is None or latest <= self._known_version:
            return None
        info = self.store.read_manifest(latest)
        if info is not None and info.sanity_ok:
            self._known_version = latest
            return info
        return None

    def status(self) -> TrainerStatus | None:
        if self.degraded:
            return TrainerStatus(state="dead")
        return self._status

    def notify_rollback(self, version: int) -> None:
        self._queue.put({"cmd": "rollback", "version": int(version)})

    def request_stop(self) -> None:
        """stop -> SIGTERM after grace -> SIGKILL (12-dagger §7)."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        self._request({"cmd": "stop"}, timeout_s=min(REQ_TIMEOUT_S, self.stop_grace_s))
        try:
            proc.wait(timeout=self.stop_grace_s)
            return
        except subprocess.TimeoutExpired:
            proc.terminate()
        try:
            proc.wait(timeout=max(self.kill_grace_s - self.stop_grace_s, 1.0))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)


def summary_from_dict(data: dict) -> EpisodeSummary:
    return EpisodeSummary(
        episode_index=int(data["episode_index"]),
        n_frames=int(data["n_frames"]),
        n_intervention_frames=int(data["n_intervention_frames"]),
        n_label_frames=int(data["n_label_frames"]),
        takeover_segments=int(data["takeover_segments"]),
        segment_doubts=[float(x) for x in data.get("segment_doubts", [])],
        success=data.get("success"),
        episode_id=str(data.get("episode_id", "")),
    )


def load_trainer_config(path: str | Path) -> TrainerConfig:
    return TrainerConfig.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = [
    "AsyncTrainerClientImpl",
    "TrainerConfig",
    "load_trainer_config",
    "summary_from_dict",
    "REQ_TIMEOUT_S",
    "MISSED_REPLIES_DEAD",
]
