"""AsyncTrainer process integration (12-dagger §7/§12; CPU): real spawn over
ZMQ, 100-label trigger, checkpoint layout + sha256, sanity gate on NaN,
boundary-only swap via the reloader, kill -9 -> one --resume restart."""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path

import numpy as np
from apollo_mavis_v2_core.dagger import ControlMode
from helpers import FRAME, free_port, make_net, make_spool, make_summary

from apollo_mavis_v2_runtime.dagger.client import AsyncTrainerClientImpl, TrainerConfig
from apollo_mavis_v2_runtime.dagger.policies import MLPPolicy, save_policy_bundle
from apollo_mavis_v2_runtime.dagger.reloader import PolicyReloaderImpl
from apollo_mavis_v2_runtime.dagger.trainer.checkpoints import sha256_file
from apollo_mavis_v2_runtime.dagger.trainer.trainer import clip_burst_steps


def make_client(tmp_path, run_id="runT", lr=1e-3, push_period_s=1.0):
    bundle = tmp_path / "seed.pt"
    save_policy_bundle(str(bundle), make_net(0),
                       {"action_space": "delta_ee", "action_frame": FRAME})
    cfg = TrainerConfig(
        run_id=run_id, checkpoints_root=str(tmp_path / "ckpts"),
        spool_dir=str(tmp_path / "spool"), seed_bundle=str(bundle),
        port=free_port(), device="cpu", cuda_visible_devices=None,
        action_frame=FRAME, action_space="delta_ee",
        state_dim=16, action_dim=8, min_new_labels=100,
        push_period_s=push_period_s, batch_size=64, lr=lr,
    )
    return AsyncTrainerClientImpl(cfg, workdir=tmp_path / "work",
                                  stop_grace_s=5.0, kill_grace_s=8.0), cfg


def wait(pred, timeout=30.0, dt=0.1):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = pred()
        if v:
            return v
        time.sleep(dt)
    raise TimeoutError("condition not met")


def submit(client, tmp_path, index, n_frames, n_human, **kw):
    path = Path(tmp_path) / "spool" / f"ep_{index:06d}.parquet"
    make_spool(path, n_frames, n_human, seed=index, **kw)
    client.submit_episode(str(path), make_summary(index, n_frames, n_human))


def test_clip_burst_steps_binding():
    assert clip_burst_steps(10) == 200
    assert clip_burst_steps(100) == 400
    assert clip_burst_steps(1000) == 1000


def test_trainer_e2e_trigger_checkpoint_sanity_swap(tmp_path):
    client, cfg = make_client(tmp_path)
    store = client.store
    try:
        wait(lambda: client.status() is not None, 20.0)
        # -- <100 new labels: NO burst -------------------------------------
        submit(client, tmp_path, 0, 80, 60)
        wait(lambda: (s := client.status()) and s.new_label_frames == 60, 20.0)
        time.sleep(1.0)
        assert store.latest() is None  # trigger requires >= 100 labels
        # -- >=100: burst at the episode boundary -> v000001 -----------------
        submit(client, tmp_path, 1, 80, 60)
        wait(lambda: store.latest() == 1, 60.0)
        d = store.version_dir(1)
        assert (d / "state_dict.pt").exists() and (d / "trainer_state.pt").exists()
        manifest = json.loads((d / "manifest.json").read_text())
        assert manifest["sha256"] == sha256_file(d / "state_dict.pt")
        assert manifest["sanity_ok"] is True
        assert manifest["run_id"] == "runT" and manifest["version"] == 1
        assert manifest["action_frame"] == FRAME
        assert manifest["trained_on_frames"] == 120  # label watermark consumed
        assert manifest["trained_on_episodes"] == [0, 1]
        assert np.isfinite(manifest["mean_loss"])
        loss1 = manifest["mean_loss"]
        st = wait(
            lambda: (s := client.status()) is not None
            and s.last_checkpoint_version == 1 and s,
            20.0,
        )
        assert st.new_label_frames == 0  # counter reset by the consumed labels

        # -- reloader: stage + swap ONLY at the boundary ----------------------
        policy = MLPPolicy.from_bundle(cfg.seed_bundle, 0, "cpu")
        reloader = PolicyReloaderImpl(policy, store, FRAME, "delta_ee",
                                      current_version=0)
        reloader.poll()
        assert reloader.staged_version() == 1
        assert reloader.maybe_swap(False, ControlMode.POLICY) is None
        assert reloader.maybe_swap(True, ControlMode.POLICY) == 1
        assert policy.spec.version == 1

        # -- second burst: loss falls on the linear task ----------------------
        time.sleep(cfg.push_period_s)  # clear the checkpoint throttle
        submit(client, tmp_path, 2, 150, 120)
        wait(lambda: store.latest() == 2, 60.0)
        loss2 = json.loads(
            (store.version_dir(2) / "manifest.json").read_text())["mean_loss"]
        assert loss2 < loss1

        # -- NaN-poisoned labels: sanity_ok=false, LATEST stays ---------------
        time.sleep(cfg.push_period_s)
        submit(client, tmp_path, 3, 150, 120, nan_labels=True)
        wait(lambda: store.read_manifest(3) is not None, 60.0)
        m3 = store.read_manifest(3)
        assert m3.sanity_ok is False
        assert store.latest() == 2  # LATEST never advances on a failed gate
        reloader.poll()
        assert reloader.maybe_swap(True, ControlMode.POLICY) == 2
        reloader.poll()
        assert reloader.staged_version() is None  # v3 is never staged/swapped
    finally:
        client.request_stop()
    assert client.proc.poll() == 0  # graceful stop exits 0


def test_trainer_death_one_resume_then_degraded(tmp_path):
    client, cfg = make_client(tmp_path, run_id="runD")
    try:
        wait(lambda: client.status() is not None, 20.0)
        pid1 = client.proc.pid
        os.kill(pid1, signal.SIGKILL)
        wait(lambda: client.restarts == 1, 20.0)  # exactly one auto-restart
        assert client.proc.pid != pid1
        assert "--resume" in client.proc.args
        wait(lambda: (s := client.status()) and s.state != "dead", 20.0)
        assert not client.degraded
        os.kill(client.proc.pid, signal.SIGKILL)  # second death: stay degraded
        wait(lambda: client.degraded, 20.0)
        assert client.restarts == 1  # never a third silent restart
        assert client.status().state == "dead"
        time.sleep(2.0)
        assert client.restarts == 1 and client.degraded
    finally:
        client.request_stop()
