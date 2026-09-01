"""PolicyReloader: staging, boundary-only swap, sha reject, rollback (12-dagger §8)."""

from __future__ import annotations

import dataclasses

import torch
from apollo_xarm7_core.dagger import ControlMode
from helpers import FRAME, make_net, write_checkpoint

from apollo_xarm7_runtime.dagger.policies import MLPPolicy
from apollo_xarm7_runtime.dagger.reloader import PolicyReloaderImpl
from apollo_xarm7_runtime.dagger.trainer.checkpoints import CheckpointStore


def make_rig(tmp_path, on_rollback=None):
    info0 = write_checkpoint(tmp_path, "run1", 0, seed=0, advance_latest=False)
    store = CheckpointStore(tmp_path, "run1")
    policy = MLPPolicy.from_bundle(str(store.state_dict_path(0)), 0, "cpu")
    reloader = PolicyReloaderImpl(policy, store, FRAME, "delta_ee",
                                  current_version=0, on_rollback=on_rollback)
    return store, policy, reloader, info0


def weights(policy) -> torch.Tensor:
    return policy._net.body[0].weight.detach().clone()


def test_poll_stages_newest_and_swaps_only_at_boundary(tmp_path):
    store, policy, reloader, _ = make_rig(tmp_path)
    w0 = weights(policy)
    write_checkpoint(tmp_path, "run1", 1, seed=1)
    reloader.poll()
    assert reloader.staged_version() == 1
    # mid-episode: NEVER swaps
    assert reloader.maybe_swap(False, ControlMode.POLICY) is None
    assert reloader.staged_version() == 1 and torch.equal(weights(policy), w0)
    # newer version supersedes the staged one (drain-latest)
    write_checkpoint(tmp_path, "run1", 2, seed=2)
    reloader.poll()
    assert reloader.staged_version() == 2
    assert reloader.maybe_swap(True, ControlMode.POLICY) == 2
    assert reloader.current_version == 2
    assert not torch.equal(weights(policy), w0)
    assert policy.spec.version == 2
    assert reloader.maybe_swap(True, ControlMode.POLICY) is None  # nothing staged


def test_corrupt_checkpoint_sha_rejected_weights_unchanged(tmp_path):
    store, policy, reloader, _ = make_rig(tmp_path)
    w0 = weights(policy)
    write_checkpoint(tmp_path, "run1", 1, seed=1)
    # truncate state_dict.pt AFTER the manifest sha was computed
    p = store.state_dict_path(1)
    p.write_bytes(p.read_bytes()[: 100])
    reloader.poll()
    assert reloader.staged_version() is None
    assert reloader.rejected and "sha256" in reloader.rejected[-1][1]
    assert reloader.maybe_swap(True, ControlMode.POLICY) is None
    assert torch.equal(weights(policy), w0)


def test_frame_and_space_mismatch_rejected_at_stage(tmp_path):
    store, policy, reloader, _ = make_rig(tmp_path)
    bad = write_checkpoint(tmp_path, "run1", 1, seed=1,
                           action_frame="arm_base:armX")
    reloader.stage(bad)
    assert reloader.staged_version() is None
    assert "action_frame" in reloader.rejected[-1][1]
    bad2 = dataclasses.replace(
        write_checkpoint(tmp_path, "run1", 2, seed=2), action_space="abs_ee")
    reloader.stage(bad2)
    assert reloader.staged_version() is None
    assert "action_space" in reloader.rejected[-1][1]
    not_sane = write_checkpoint(tmp_path, "run1", 3, seed=3, sanity_ok=False)
    reloader.stage(not_sane)
    assert reloader.staged_version() is None


def test_rollback_restores_last_known_good_bit_exact(tmp_path):
    notified = []
    store, policy, reloader, _ = make_rig(tmp_path, on_rollback=notified.append)
    assert store.last_known_good() == 0  # initialized to the seed (§12)
    w0 = weights(policy)
    write_checkpoint(tmp_path, "run1", 1, seed=1)
    reloader.poll()
    assert reloader.maybe_swap(True, ControlMode.POLICY) == 1
    reloader.mark_good()
    assert store.last_known_good() == 1
    w1 = weights(policy)
    write_checkpoint(tmp_path, "run1", 2, seed=2)
    reloader.poll()
    reloader.maybe_swap(True, ControlMode.POLICY)
    assert not torch.equal(weights(policy), w1)
    v = reloader.rollback()  # allowed mid-episode
    assert v == 1 and notified == [1]
    assert torch.equal(weights(policy), w1)  # bit-exact restore
    assert not torch.equal(weights(policy), w0)


def test_load_weights_never_partially_applied(tmp_path):
    store, policy, reloader, _ = make_rig(tmp_path)
    w0 = weights(policy)
    d = store.version_dir(1)
    d.mkdir(parents=True)
    torch.save({"arch": {"state_dim": 2, "action_dim": 2, "hidden": 4},
                "state_dict": make_net(1, 2, 2).state_dict(), "spec": {}},
               store.state_dict_path(1))  # wrong shapes: load must fail cleanly
    try:
        policy.load_weights(str(store.state_dict_path(1)))
    except Exception:
        pass
    assert torch.equal(weights(policy), w0)
