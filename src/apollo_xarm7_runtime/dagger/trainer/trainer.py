"""TrainerMain + BCFineTuner — the AsyncTrainer process body (12-dagger §7).

Trigger: episode boundary (submit_episode) AND ``new_label_frames >=
min_new_labels`` -> one burst of ``K = clip(4*new, 200, 1000)`` AdamW steps
(50/50 sampling), sanity gate, versioned checkpoint (throttled to one per
``push_period_s``; ``LATEST`` advances only for sanity_ok). The trainer never
rolls itself back; ``{"cmd": "rollback"}`` reloads the named version.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch
from apollo_xarm7_core.dagger import CheckpointInfo, TrainerStatus

from ..client import TrainerConfig
from ..policies import MLPNet, load_policy_bundle, save_policy_bundle
from .checkpoints import STATE_DICT, CheckpointStore, sha256_file
from .control import ControlEndpoint
from .sampling import FiftyFiftySampler, LabelIndex, read_spool

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BurstStats:
    mean_loss: float
    n_steps: int
    nan_seen: bool


class BCFineTuner:
    """AdamW fine-tune loop (lr 1e-5, batch 64, grad clip 1.0 — §7 sketch)."""

    def __init__(self, net: MLPNet, cfg: TrainerConfig, device: str) -> None:
        self.net = net.to(device).train()
        self.cfg = cfg
        self.device = device
        self.opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_decay)

    def burst(self, sampler: FiftyFiftySampler, k_steps: int) -> BurstStats:
        losses: list[float] = []
        nan_seen = False
        for _ in range(k_steps):
            s, a = sampler.next(self.cfg.batch_size)
            x = torch.as_tensor(s, device=self.device)
            y = torch.as_tensor(a, device=self.device)
            loss = torch.nn.functional.mse_loss(self.net(x), y)
            if not torch.isfinite(loss):
                nan_seen = True
                self.opt.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
            self.opt.step()
            self.opt.zero_grad()
            losses.append(float(loss.detach()))
        mean = float(np.mean(losses)) if losses else float("nan")
        return BurstStats(mean_loss=mean, n_steps=k_steps, nan_seen=nan_seen)


def clip_burst_steps(new_labels: int, lo: int = 200, hi: int = 1000) -> int:
    return int(np.clip(4 * new_labels, lo, hi))


class TrainerMain:
    """Single-threaded run loop: control poll + trigger + checkpoint."""

    def __init__(self, cfg: TrainerConfig, resume: bool = False) -> None:
        self.cfg = cfg
        self.device = cfg.device if torch.cuda.is_available() else "cpu"
        self.store = CheckpointStore(cfg.checkpoints_root, cfg.run_id)
        self.control = ControlEndpoint(cfg.port)
        self.index = LabelIndex()
        self.state = "starting"
        self.steps_total = 0
        self.last_burst_loss: float | None = None
        self.last_ckpt_version: int | None = None
        self.last_ckpt_ts: float | None = None
        self.last_ckpt_wall = 0.0
        self.trained_on_frames = 0
        self.parent_version: int | None = None
        self.new_label_frames = 0
        self._boundary_pending = False
        self._stop = False
        self._last_status_poll = time.monotonic()
        if cfg.seed_bundle:
            self.net, _ = load_policy_bundle(cfg.seed_bundle, self.device)
        else:
            self.net = MLPNet(cfg.state_dim, cfg.action_dim, cfg.hidden)
        self.tuner = BCFineTuner(self.net, cfg, self.device)
        if cfg.seed_parquet:
            data = read_spool(cfg.seed_parquet)
            n = self.index.add_seed(data["state"], data["action"])
            logger.info("seed dataset: %d frames", n)
        versions = self.store.versions()
        self.version = max(versions) if versions else 0
        if versions:
            self.parent_version = self.version  # fine-tuning lineage from the seed
        if resume and self.version > 0:
            self._load_trainer_state(self.version)
        self.state = "idle"

    # -- run loop --------------------------------------------------------------------
    def run(self) -> int:
        logger.info("trainer up: device=%s port=%d run=%s",
                    self.device, self.cfg.port, self.cfg.run_id)
        try:
            while not self._stop:
                req = self.control.poll(timeout_ms=100)
                if req is not None:
                    self.control.reply(self._handle(req))
                if self._boundary_pending and self.new_label_frames >= self.cfg.min_new_labels:
                    self._boundary_pending = False
                    self._train_burst()
                if time.monotonic() - self._last_status_poll > self.cfg.orphan_timeout_s:
                    logger.warning("orphaned (no status poll); exiting")
                    break
        finally:
            self.control.close()
        return 0

    def _handle(self, req: dict) -> dict:
        cmd = req.get("cmd")
        if cmd == "status":
            self._last_status_poll = time.monotonic()
            return {"ok": True, "status": self.status().model_dump()}
        if cmd == "submit_episode":
            path = req.get("episode_path", "")
            try:
                data = read_spool(path)
                n = self.index.add_episode(
                    int(req.get("summary", {}).get("episode_index", -1)), data)
            except Exception as e:
                logger.exception("submit_episode failed")
                return {"ok": False, "detail": repr(e)}
            self.new_label_frames += n
            self._boundary_pending = True  # trigger checks at episode boundary only
            return {"ok": True, "new_label_frames": self.new_label_frames}
        if cmd == "train_now":
            self._train_burst(force=True)
            return {"ok": True}
        if cmd == "rollback":
            v = int(req.get("version", 0))
            ok = self._load_weights_version(v)
            if ok:
                self._load_trainer_state(v)
                self.parent_version = v
            return {"ok": ok}
        if cmd == "stop":
            self._stop = True
            return {"ok": True}
        return {"ok": False, "detail": f"unknown cmd {cmd!r}"}

    def status(self) -> TrainerStatus:
        return TrainerStatus(
            state=self.state,  # type: ignore[arg-type]
            steps_total=self.steps_total,
            last_burst_loss=self.last_burst_loss,
            last_checkpoint_version=self.last_ckpt_version,
            last_checkpoint_ts=self.last_ckpt_ts,
            new_label_frames=self.new_label_frames,
        )

    # -- burst + checkpoint -----------------------------------------------------------
    def _train_burst(self, force: bool = False) -> None:
        if self.index.n_frames == 0:
            return
        consumed = self.new_label_frames
        k = clip_burst_steps(consumed)
        self.state = "training"
        stats = self.tuner.burst(FiftyFiftySampler(self.index), k)
        self.state = "idle"
        self.steps_total += stats.n_steps
        self.last_burst_loss = stats.mean_loss
        self.trained_on_frames += consumed
        self.new_label_frames = 0
        sane = self._sanity_ok(stats)
        self._write_checkpoint(stats, sane)

    def _sanity_ok(self, stats: BurstStats) -> bool:
        if stats.nan_seen or not np.isfinite(stats.mean_loss):
            return False
        n = min(self.cfg.held_out_frames, self.index.n_frames)
        with torch.no_grad():
            x = torch.as_tensor(self.index.states[-n:], device=self.device)
            out = self.net(x).cpu().numpy()
        if not np.all(np.isfinite(out)):
            return False
        acts = self.index.actions
        q01, q99 = np.quantile(acts, 0.01, axis=0), np.quantile(acts, 0.99, axis=0)
        # 1 cm/rad pad floor: near-constant dims (idle rotations, held gripper)
        # must not turn the gate into a zero-width band.
        pad = 3 * np.std(acts, axis=0) + 1e-2
        return bool(np.all(out >= q01 - pad) and np.all(out <= q99 + pad))

    def _write_checkpoint(self, stats: BurstStats, sanity_ok: bool) -> None:
        now = time.monotonic()
        if self.last_ckpt_ts is not None and (now - self.last_ckpt_ts) < self.cfg.push_period_s:
            logger.info("checkpoint throttled (push_period_s)")
            return
        self.version += 1
        d = self.store.version_dir(self.version)
        d.mkdir(parents=True, exist_ok=True)
        save_policy_bundle(str(d / STATE_DICT), self.net, {
            "action_space": self.cfg.action_space,
            "action_frame": self.cfg.action_frame,
        })
        torch.save({"optimizer": self.tuner.opt.state_dict(),
                    "steps_total": self.steps_total,
                    "trained_on_frames": self.trained_on_frames},
                   self.store.trainer_state_path(self.version))
        info = CheckpointInfo(
            run_id=self.cfg.run_id,
            version=self.version,
            path=str(d),
            parent_version=self.parent_version,
            trained_on_frames=self.trained_on_frames,
            trained_on_episodes=list(self.index.episodes),
            action_frame=self.cfg.action_frame,
            action_space=self.cfg.action_space,
            sanity_ok=sanity_ok,
            mean_loss=float(stats.mean_loss) if np.isfinite(stats.mean_loss) else -1.0,
            sha256=sha256_file(d / STATE_DICT),
            created_wallclock_ns=time.time_ns(),
        )
        self.store.write_manifest(info)
        self.parent_version = self.version
        self.last_ckpt_version = self.version
        self.last_ckpt_ts = now
        self.index.mark_checkpoint()
        if sanity_ok:
            self.store.advance_latest(self.version)  # sanity_ok only (§7)
        logger.info("checkpoint v%06d sanity_ok=%s loss=%.6f",
                    self.version, sanity_ok, stats.mean_loss)

    # -- rollback/resume ---------------------------------------------------------------
    def _load_weights_version(self, version: int) -> bool:
        try:
            net, _ = load_policy_bundle(str(self.store.state_dict_path(version)), self.device)
            self.net.load_state_dict(net.state_dict())
            return True
        except Exception:
            logger.exception("rollback load failed (v%06d)", version)
            return False

    def _load_trainer_state(self, version: int) -> None:
        try:
            payload = torch.load(self.store.trainer_state_path(version),
                                 map_location=self.device, weights_only=False)
            self.tuner.opt.load_state_dict(payload["optimizer"])
            self.steps_total = int(payload.get("steps_total", 0))
            self.trained_on_frames = int(payload.get("trained_on_frames", 0))
            self._load_weights_version(version)
        except Exception:
            logger.exception("trainer_state load failed (v%06d)", version)


__all__ = ["TrainerMain", "BCFineTuner", "BurstStats", "clip_burst_steps"]
