"""Shared phase-08 test helpers: checkpoint bundles, spool files, fakes."""

from __future__ import annotations

import dataclasses
import json
import socket
import time
from pathlib import Path

import numpy as np
from apollo_xarm7_core.dagger import CheckpointInfo

STATE_DIM = 16  # 1 arm + rail: 7 joints + grip + rail + ee pos3 + quat4
ACTION_DIM = 8  # [dx dy dz drx dry drz grip rail.dpos]
FRAME = "arm_base:arm0"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_net(seed: int = 0, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
    import torch

    from apollo_xarm7_runtime.dagger.policies import MLPNet

    torch.manual_seed(seed)
    return MLPNet(state_dim, action_dim, hidden=32)


def write_checkpoint(
    root: Path,
    run_id: str,
    version: int,
    *,
    deploy: bool = False,
    action_frame: str = FRAME,
    action_space: str = "delta_ee",
    sanity_ok: bool = True,
    seed: int = 0,
    net=None,
    advance_latest: bool = True,
) -> CheckpointInfo:
    """A valid bundle + manifest; returns the CheckpointInfo written."""
    from apollo_xarm7_runtime.dagger.policies import save_policy_bundle
    from apollo_xarm7_runtime.dagger.trainer.checkpoints import (
        STATE_DICT,
        CheckpointStore,
        sha256_file,
    )

    if deploy:
        d = Path(root) / run_id / "deploy" / f"v{version:03d}"
    else:
        d = Path(root) / run_id / f"v{version:06d}"
    d.mkdir(parents=True, exist_ok=True)
    net = net if net is not None else make_net(seed)
    save_policy_bundle(str(d / STATE_DICT),
                       net, {"action_space": action_space, "action_frame": action_frame})
    info = CheckpointInfo(
        run_id=run_id, version=version, path=str(d), parent_version=None,
        trained_on_frames=0, trained_on_episodes=[], action_frame=action_frame,
        action_space=action_space, sanity_ok=sanity_ok, mean_loss=0.1,
        sha256=sha256_file(d / STATE_DICT), created_wallclock_ns=time.time_ns(),
    )
    (d / "manifest.json").write_text(json.dumps(dataclasses.asdict(info), indent=2))
    if not deploy and advance_latest and sanity_ok:
        CheckpointStore(root, run_id).advance_latest(version)
    return info


def promote(root: Path, policy_id: str) -> None:
    (Path(root) / "PROMOTED").write_text(policy_id + "\n")


# -- spool fabrication (trainer-side input; 12-dagger §7) --------------------------
LINEAR_W = None  # lazily built (state_dim x action_dim) target map


def target_map(state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM) -> np.ndarray:
    global LINEAR_W
    if LINEAR_W is None or LINEAR_W.shape != (state_dim, action_dim):
        rng = np.random.default_rng(42)
        LINEAR_W = (rng.standard_normal((state_dim, action_dim)) * 0.05).astype(np.float32)
    return LINEAR_W


def make_spool(
    path: Path,
    n_frames: int,
    n_human: int,
    *,
    seed: int = 0,
    nan_labels: bool = False,
    state_dim: int = STATE_DIM,
    action_dim: int = ACTION_DIM,
) -> None:
    """ep_*.parquet with the recorder's SPOOL_COLUMNS; labels follow a known
    linear map (loss must fall under BC); ``nan_labels`` poisons the burst."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    states = rng.standard_normal((n_frames, state_dim)).astype(np.float32)
    actions = (states @ target_map(state_dim, action_dim)).astype(np.float32)
    modes = np.zeros(n_frames, dtype=np.int8)
    modes[:n_human] = 1  # human labels first
    if n_human < n_frames:
        modes[n_human] = 2  # one transition frame (must be excluded)
    if nan_labels:
        actions[: max(n_human, 1)] = np.nan
    pa_cols = {
        "action": pa.array(actions.tolist()),
        "observation.state": pa.array(states.tolist()),
        "control_mode": pa.array(modes.tolist()),
        "policy_action": pa.array(actions.tolist()),
        "policy_version": pa.array([0] * n_frames),
        "intervention": pa.array([bool(m != 0) for m in modes]),
        "action_source": pa.array([3 if m else 0 for m in modes]),
        "wallclock_ns": pa.array(list(range(n_frames))),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(pa_cols), path)


def make_summary(index: int, n_frames: int, n_human: int):
    from apollo_xarm7_core.dagger import EpisodeSummary

    return EpisodeSummary(
        episode_index=index, n_frames=n_frames,
        n_intervention_frames=n_human + 1, n_label_frames=n_human,
        takeover_segments=1, segment_doubts=[0.0], success=None,
    )
