"""DaggerRecorderThread schema + spool + summary (12-dagger §4-§5)."""

from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
import pytest
from apollo_mavis_v2_core import CollisionReport
from helpers import ACTION_DIM
from test_recorder_thread import FakeCam, FakeKin, FakeRecorder, make_state

from apollo_mavis_v2_runtime.bus import RuntimeBus
from apollo_mavis_v2_runtime.control.snapshot import StateSnapshot
from apollo_mavis_v2_runtime.dagger.gate import TakeoverGateImpl
from apollo_mavis_v2_runtime.dagger.recorder import (
    CONTROL_MODE_LABELS,
    DaggerRecorderThread,
    dagger_features,
)
from apollo_mavis_v2_runtime.recorder.features import ArmMeta, build_features
from apollo_mavis_v2_runtime.recorder.frames import RecordingFrameConverter
from apollo_mavis_v2_runtime.recorder.sidecars import SidecarWriter

Q0 = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.10])


def make_snap(tick, q_cmd, wallclock_ns, dagger_frame):
    return StateSnapshot(
        t_mono=wallclock_ns / 1e9, wallclock_ns=wallclock_ns, tick=tick,
        arms={"arm0": make_state(Q0, wallclock_ns)},
        q_cmd={"arm0": np.asarray(q_cmd, dtype=np.float64)},
        active_arm="arm0", gate=CollisionReport.ok(),
        gripper_frac={"arm0": 0.7},
        session_extra={"dagger_frame": dagger_frame},
    )


@pytest.fixture
def rig(tmp_path):
    rec = FakeRecorder()
    bus = RuntimeBus()
    cam = FakeCam()
    gate = TakeoverGateImpl(["arm0"])
    saved = []
    thread = DaggerRecorderThread(
        rec, bus, {"cam0": cam}, [ArmMeta("arm0", True)],
        RecordingFrameConverter({"arm0": "arm_base:arm0"}), FakeKin(),
        fps=25,
        sidecars=SidecarWriter(tmp_path),
        episode_meta_base={"session_id": "s1"},
        gate=gate, run_id="run1", dataset_root=tmp_path,
        on_episode_saved=lambda i, s, p: saved.append((i, s, p)),
    )
    return rec, bus, cam, gate, thread, saved, tmp_path


def feed(thread, bus, cam, tick, now, ann):
    cam.refresh(now)
    bus.snapshot.put(make_snap(tick, Q0, int(now * 1e9), ann))
    thread.run_iteration(now)


def ann(mode, pa=None, version=0):
    return {"control_mode": mode, "action_source": 3 if mode else 0,
            "policy_action": pa, "policy_version": version}


def test_dagger_features_verbatim():
    base = build_features([ArmMeta("arm0", True)], {"arm0": "arm_base:arm0"}, {})
    feats = dagger_features(base, "run1")
    assert feats["control_mode"] == {
        "dtype": "int8", "shape": (1,), "names": None,
        "info": {"labels": {"0": "policy", "1": "human", "2": "takeover_transition"}},
    }
    assert feats["policy_action"]["dtype"] == "float32"
    assert feats["policy_action"]["shape"] == (8,)
    assert feats["policy_action"]["names"] == base["action"]["names"]
    assert feats["policy_action"]["info"] == {"counterfactual": True}
    assert feats["policy_version"] == {
        "dtype": "int32", "shape": (1,), "names": None, "info": {"run_id": "run1"},
    }


def test_frames_carry_modes_nan_rows_and_labels(rig):
    rec, bus, cam, gate, thread, saved, root = rig
    cf = np.arange(ACTION_DIM, dtype=np.float32)
    gate.on_toggle("arm0", 0.0)  # events land in the sidecar snapshot
    assert thread.request("new")[0]
    t = 1.0
    script = (
        [(0, cf, 0)] * 3          # policy frames, counterfactual present
        + [(2, None, 0)] * 2      # transition, unqueried tick -> NaN row
        + [(1, cf, 0)] * 4        # human labels
        + [(0, None, 1)] * 3      # back to policy on a NEW version
    )
    for i, (mode, pa, ver) in enumerate(script):
        t += 0.04
        feed(thread, bus, cam, i + 1, t, ann(mode, pa, ver))
    thread.request("save")
    thread.run_iteration(t + 0.04)
    frames = rec.episodes[0]
    n = len(frames)
    assert n == len(script) - 1  # one pending capture never got its action
    modes = np.array([int(f["control_mode"][0]) for f in frames])
    assert set(modes.tolist()) == {0, 1, 2}  # all three values appear
    for f in frames:
        m = int(f["control_mode"][0])
        assert bool(f["intervention"][0]) == (m != 0)  # transition counts
        assert int(f["action_source"][0]) == (3 if m != 0 else 0)
        assert f["policy_action"].shape == (ACTION_DIM,)
    # NaN exactly where policy_action was None (indices 3,4 and the tail)
    nan_rows = [i for i, f in enumerate(frames)
                if np.all(np.isnan(f["policy_action"]))]
    assert nan_rows == [3, 4, 9, 10]
    versions = [int(f["policy_version"][0]) for f in frames]
    assert versions == [0] * 9 + [1] * 2  # changes only at the deposit switch

    # summary + callback + spool
    assert len(saved) == 1
    idx, summary, spool = saved[0]
    assert idx == 0 and summary.episode_index == 0
    assert summary.n_frames == n
    assert summary.n_label_frames == int((modes == 1).sum())
    assert summary.n_intervention_frames == int((modes != 0).sum())
    assert summary.takeover_segments == 1
    table = pq.read_table(spool)
    assert table.num_rows == n
    spool_modes = np.asarray(table.column("control_mode"))
    assert (spool_modes == modes).all()
    # label rule: HUMAN only, transitions excluded
    labels = spool_modes == 1
    assert labels.sum() == summary.n_label_frames

    # sidecar: gate_events + episode_summary
    ep = json.loads((root / "meta" / "apollo" / "episodes" / "episode_000000.json")
                    .read_text())
    assert ep["episode_summary"]["n_label_frames"] == summary.n_label_frames
    assert ep["gate_events"] == []  # toggle happened BEFORE episode_new
    assert dict(CONTROL_MODE_LABELS)["2"] == "takeover_transition"


def test_gate_events_scoped_to_episode(rig):
    rec, bus, cam, gate, thread, saved, root = rig
    thread.request("new")
    gate.on_toggle("arm0", 1.0)
    gate.tick(1.5)
    t = 1.0
    for i in range(3):
        t += 0.04
        feed(thread, bus, cam, i + 1, t, ann(1, None))
    thread.request("save")
    thread.run_iteration(t + 0.04)
    ep = json.loads((root / "meta" / "apollo" / "episodes" / "episode_000000.json")
                    .read_text())
    assert [e["mode"] for e in ep["gate_events"]] == ["takeover_transition", "human"]
    assert [e["seq"] for e in ep["gate_events"]] == [1, 2]


def test_discard_clears_rows_no_spool(rig):
    rec, bus, cam, gate, thread, saved, root = rig
    thread.request("new")
    t = 1.0
    for i in range(3):
        t += 0.04
        feed(thread, bus, cam, i + 1, t, ann(0, None))
    thread.request("discard")
    thread.run_iteration(t + 0.04)
    assert saved == []
    assert not (root / "trainer_spool").exists()
