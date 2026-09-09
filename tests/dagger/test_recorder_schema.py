"""DaggerRecorderThread schema + spool + summary (12-dagger §4-§5)."""

from __future__ import annotations

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
    ACTOR_LABELS,
    CONTROL_MODE_LABELS,
    SPOOL_COLUMNS,
    DaggerRecorderThread,
    dagger_features,
)
from apollo_mavis_v2_runtime.recorder.features import ArmMeta, build_features
from apollo_mavis_v2_runtime.recorder.frames import RecordingFrameConverter

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
    # phase-14 (15-online-dagger §4, D4): the operator's readable per-step key, verbatim
    assert feats["actor"] == {
        "dtype": "int8", "shape": (1,), "names": None,
        "info": {"labels": {"0": "novice", "1": "expert"}, "derived_from": "control_mode != 0"},
    }
    assert ACTOR_LABELS == {"0": "novice", "1": "expert"}
    assert SPOOL_COLUMNS[-1] == "actor" and SPOOL_COLUMNS[:8] == (
        "action", "observation.state", "control_mode", "policy_action", "policy_version",
        "intervention", "action_source", "wallclock_ns",
    )
    assert "apollo_schema" not in feats.get("actor", {}).get("info", {})  # never bumped


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
        # actor = 1 iff control_mode != policy (transition frames are the expert's too)
        assert f["actor"].dtype == np.int8 and f["actor"].shape == (1,)
        assert int(f["actor"][0]) == (1 if m != 0 else 0)
    actors = np.array([int(f["actor"][0]) for f in frames])
    assert set(actors.tolist()) == {0, 1}
    # NaN exactly where policy_action was None (indices 3,4 and the tail)
    nan_rows = [i for i, f in enumerate(frames)
                if np.all(np.isnan(f["policy_action"]))]
    assert nan_rows == [3, 4, 9, 10]
    versions = [int(f["policy_version"][0]) for f in frames]
    assert versions == [0] * 9 + [1] * 2  # changes only at the deposit switch

    # summary + callback + spool (keyed by the episode id since 2026-09-07, 12-dagger §7)
    assert len(saved) == 1
    idx, summary, spool = saved[0]
    assert idx == 0 and summary.episode_index == 0
    assert spool == str(root / "trainer_spool" / f"ep_{summary.episode_id}.parquet")
    assert summary.episode_id and not summary.episode_id.isdigit()
    assert summary.n_frames == n
    assert summary.n_label_frames == int((modes == 1).sum())
    assert summary.n_intervention_frames == int((modes != 0).sum())
    assert summary.takeover_segments == 1
    # phase-14: the actor split (transition frames count as expert, unlike n_label_frames)
    assert summary.n_expert_frames == int((modes != 0).sum()) == summary.n_intervention_frames
    assert summary.n_novice_frames == int((modes == 0).sum())
    assert summary.n_expert_frames + summary.n_novice_frames == n
    assert summary.n_expert_frames > summary.n_label_frames  # the two transition frames
    table = pq.read_table(spool)
    assert table.num_rows == n
    assert table.column_names == list(SPOOL_COLUMNS)
    spool_modes = np.asarray(table.column("control_mode"))
    assert (spool_modes == modes).all()
    assert (np.asarray(table.column("actor")) == actors).all()
    # label rule: HUMAN only, transitions excluded
    labels = spool_modes == 1
    assert labels.sum() == summary.n_label_frames

    # sidecar payload handed to the recorder: gate_events + episode_summary
    ep = rec.sidecars[0]
    assert ep["episode_summary"]["n_label_frames"] == summary.n_label_frames
    assert ep["episode_summary"]["n_expert_frames"] == summary.n_expert_frames
    assert ep["episode_summary"]["n_novice_frames"] == summary.n_novice_frames
    assert ep["gate_events"] == []  # toggle happened BEFORE episode_new
    assert "online_dagger" not in ep  # plain DAgger run: no coordinator attached
    assert dict(CONTROL_MODE_LABELS)["2"] == "takeover_transition"


def test_spool_write_failure_still_fires_the_saved_hook(rig, monkeypatch):
    """``_write_spool`` raising (pyarrow / disk full) must not skip ``on_episode_saved``:
    the episode directory IS published, so the executor's save boundary, the coordinator's
    ``events.episode_saved`` / ``session.json`` row and the counters all have to happen
    (else the state flip reads as a discard and the next kept rollout reuses the count).
    The hook gets ``spool_path=""``; the rows are dropped like after any save."""
    rec, bus, cam, gate, thread, saved, root = rig

    def boom(episode_id):
        raise OSError("disk full (test)")

    monkeypatch.setattr(thread, "_write_spool", boom)
    assert thread.request("new")[0]
    t = 1.0
    for i in range(4):
        t += 0.04
        feed(thread, bus, cam, i + 1, t, ann(1 if i % 2 else 0, None))
    assert thread.request("save")[0]
    thread.run_iteration(t + 0.04)
    assert len(rec.episodes) == 1 and thread.status().state == "idle"
    [(idx, summary, spool)] = saved
    assert idx == 0 and spool == "" and summary.episode_index == 0
    assert summary.n_frames == 3 and summary.n_expert_frames == 1
    assert thread._rows == [] and not (root / "trainer_spool").exists()
    assert not thread.degraded and thread.request("new")[0]  # recording goes on


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
    ep = rec.sidecars[0]
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


class StubCoordinator:
    """Only what the recorder touches: ``sidecar_block(actor_counts)``."""

    def __init__(self):
        self.calls = []

    def sidecar_block(self, actor_counts):
        self.calls.append(dict(actor_counts))
        return {
            "session_name": "sess1", "rollouts_saved": 3, "policy_version": 7,
            "actor_counts": dict(actor_counts),
        }


def test_sidecar_carries_the_online_dagger_block_when_a_coordinator_is_attached(rig):
    rec, bus, cam, gate, thread, saved, root = rig
    thread.coordinator = StubCoordinator()
    thread.request("new")
    t = 1.0
    script = [0, 0, 1, 1, 1, 2]
    for i, mode in enumerate(script):
        t += 0.04
        feed(thread, bus, cam, i + 1, t, ann(mode, None))
    thread.request("save")
    thread.run_iteration(t + 0.04)
    ep = rec.sidecars[0]
    # 15-online-dagger §4: episode.json["online_dagger"] = {session_name, rollouts_saved,
    # policy_version, actor_counts}; counts = the summary's actor split
    n = len(script) - 1
    counts = {"novice": 2, "expert": n - 2}
    assert ep["online_dagger"] == {
        "session_name": "sess1", "rollouts_saved": 3, "policy_version": 7,
        "actor_counts": counts,
    }
    assert thread.coordinator.calls == [counts]
    assert saved[0][1].n_expert_frames == n - 2 and saved[0][1].n_novice_frames == 2


def test_discard_hook_fires_with_index_id_and_reason(rig):
    rec, bus, cam, gate, thread, saved, root = rig
    discarded = []
    thread.on_episode_discarded = lambda i, eid, why: discarded.append((i, eid, why))
    thread.request("new")
    eid = rec.episode_id
    assert eid
    t = 1.0
    for i in range(3):
        t += 0.04
        feed(thread, bus, cam, i + 1, t, ann(1, None))
    thread.request("discard")
    thread.run_iteration(t + 0.04)
    assert discarded == [(0, eid, "")]
    assert saved == [] and thread._rows == []
    assert thread.status().state == "idle"


def test_teardown_announces_an_open_episode_as_discarded(rig):
    """15-online-dagger §3 / §0 item 5: a rollout still open at session teardown is discarded like
    an operator discard — the hook fires with reason ``"session teardown"`` so the
    coordinator publishes ``events.episode_discarded`` before it closes. An idle
    recorder's shutdown announces nothing."""
    rec, bus, cam, gate, thread, saved, root = rig
    discarded = []
    thread.on_episode_discarded = lambda i, eid, why: discarded.append((i, eid, why))
    thread.request("new")
    eid = rec.episode_id
    assert eid
    t = 1.0
    for i in range(3):
        t += 0.04
        feed(thread, bus, cam, i + 1, t, ann(1, None))
    thread.stop()  # never started: the shutdown runs inline on the caller
    assert discarded == [(0, eid, "session teardown")]
    assert saved == [] and thread._rows == [] and thread.status().state == "idle"
    assert rec.episodes == [] and rec.finalized == 1
    thread.stop()  # idempotent: no second announcement
    assert len(discarded) == 1


def test_save_with_zero_kept_frames_discards_with_a_reason_instead_of_raising(rig):
    """Fixed on the way to 15-online-dagger §4 (2026-09-08): the base class discards an
    all-filtered save with ``reason="empty episode discarded"``; the DAgger override used
    to take no argument and raised TypeError on the recorder thread."""
    rec, bus, cam, gate, thread, saved, root = rig
    discarded = []
    thread.on_episode_discarded = lambda i, eid, why: discarded.append((i, eid, why))

    class DropAll:
        """An idle-frame filter that keeps NOTHING (every capture is an idle frame)."""

        frames_skipped = 0
        buffered = 2  # >= 2: the save is accepted, then flushes to zero frames

        def reset(self):
            pass

        def push(self, capture):
            self.frames_skipped += 1
            return []

        def flush(self):
            return []

        def summary(self):
            return {"frames_skipped": self.frames_skipped}

    thread.filter = DropAll()
    thread.request("new")
    eid = rec.episode_id
    t = 1.0
    for i in range(3):
        t += 0.04
        feed(thread, bus, cam, i + 1, t, ann(0, None))
    ok, detail = thread.request("save")
    assert ok, detail
    thread.run_iteration(t + 0.04)  # would have raised TypeError before the fix
    assert saved == [] and rec.episodes == []
    assert thread.status().state == "idle"
    assert discarded and discarded[0][1] == eid
    assert discarded[0][2] == "empty episode discarded"
    assert thread.status().detail == "empty episode discarded"
    assert not (root / "trainer_spool").exists()
