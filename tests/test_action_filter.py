"""Idle-frame filter (10-frames §11.4; 04-runtime §10.5; 2026-09-07 addendum): judged
against the LAST KEPT frame, the ±gripper_context_s look-ahead, every session arm,
DAgger's human-frames-only rule, the gap ledger, disabled = pass-through, and the
RecorderThread integration (frames_skipped, episode.json.filter, flush at save)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
from apollo_mavis_v2_core import Pose, se3
from apollo_mavis_v2_core.protocol import ActionFilterConfig

from apollo_mavis_v2_runtime.recorder.action_filter import ActionFilter, rotation_angle

FPS = 25
CFG = ActionFilterConfig()  # pro-dagger defaults: 1 mm / 1 mrad / 1 % / 1 mm / 1.6 s
WINDOW = round(1.6 * FPS)  # 40 captures of look-ahead


@dataclass
class Cap:
    """The slice of RecorderThread._Capture the filter reads."""

    cmd_pose_b: dict[str, Pose]
    grip_cmd: dict[str, float]
    rail_cmd: dict[str, float | None] = field(default_factory=dict)
    action_source: int | None = None
    tag: int = 0


def cap(x: float = 0.0, grip: float = 0.5, rail: float | None = 0.1, yaw: float = 0.0,
        arms=("arm0",), tag: int = 0, source: int | None = None, **per_arm) -> Cap:
    poses, grips, rails = {}, {}, {}
    for a in arms:
        ax = per_arm.get(f"{a}_x", x)
        poses[a] = Pose(np.array([ax, 0.0, 0.3]), se3.rotvec_to_quat(np.array([0.0, 0.0, yaw])))
        grips[a] = per_arm.get(f"{a}_grip", grip)
        rails[a] = rail
    return Cap(poses, grips, rails, source, tag)


def run(f: ActionFilter, caps: list[Cap]) -> list[int]:
    kept = []
    for c in caps:
        kept += [k.tag for k in f.push(c)]
    kept += [k.tag for k in f.flush()]
    return kept


def test_rotation_angle():
    q0 = se3.rotvec_to_quat(np.zeros(3))
    q1 = se3.rotvec_to_quat(np.array([0.0, 0.0, 0.002]))
    assert rotation_angle(q0, q1) == pytest.approx(0.002, abs=1e-9)
    assert rotation_angle(q0, q0) == 0.0


def test_idle_frames_are_skipped_and_slow_motion_keeps_one_frame_per_epsilon():
    f = ActionFilter(CFG, FPS)
    # 100 frames: a stationary arm, then 0.1 mm per frame drift for 60 frames
    still = [cap(tag=i) for i in range(40)]
    drift = [cap(x=0.0001 * (i + 1), tag=40 + i) for i in range(60)]
    caps = still + drift
    kept = run(f, caps)
    assert kept[0] == 0  # the first frame is always kept
    assert 1 not in kept and 39 not in kept  # stationary: skipped
    # slow drift: one frame every accumulated 1 mm (10 frames of 0.1 mm), judged vs LAST KEPT
    drift_kept = [t for t in kept if t >= 40]
    assert drift_kept == [49, 59, 69, 79, 89, 99]
    assert f.frames_seen == 100 and f.frames_skipped == 100 - len(kept)
    # one gap before the first drift keep (49 skipped), then 9 between each
    assert f.gaps[0] == [1, 48] and f.gaps[1] == [2, 9]  # 1..48 skipped, then 9 per keep
    s = f.summary()
    assert s["enabled"] is True and s["params"]["pos_eps_m"] == 0.001
    assert s["frames_seen"] == 100 and s["frames_skipped"] == f.frames_skipped
    assert s["gaps"] == f.gaps


def test_rotation_gripper_and_rail_each_break_idleness():
    f = ActionFilter(CFG, FPS)
    caps = [cap(tag=0), cap(tag=1), cap(yaw=0.002, tag=2), cap(yaw=0.002, tag=3),
            cap(yaw=0.002, rail=0.1015, tag=4), cap(yaw=0.002, rail=0.1015, tag=5)]
    assert run(f, caps) == [0, 2, 4]
    # a gripper step is a change (and exempts its neighbourhood, see below)
    f = ActionFilter(ActionFilterConfig(gripper_context_s=0.0), FPS)
    assert run(f, [cap(tag=0), cap(tag=1), cap(grip=0.52, tag=2), cap(grip=0.52, tag=3)]) == [0, 2]


def test_gripper_change_exempts_idle_frames_within_the_context_window():
    """Idle frames up to 1.6 s BEFORE a gripper toggle are kept (look-ahead) and up to
    1.6 s after it (history); idle frames further away are skipped."""
    f = ActionFilter(CFG, FPS)
    n = 200
    caps = [cap(grip=0.5 if i < 100 else 0.9, tag=i) for i in range(n)]  # toggle at 99->100
    kept = run(f, caps)
    # frames 60..139 (within +-40 of the toggle) are all kept; earlier / later idle ones skipped
    assert all(t in kept for t in range(60, 140)), sorted(set(range(60, 140)) - set(kept))
    assert 0 in kept and 30 not in kept and 59 not in kept and 141 not in kept and 199 not in kept
    assert f.gaps and f.gaps[0] == [1, 59]  # 1..59 skipped before kept frame #1 (capture 60)


def test_look_ahead_delays_decisions_by_the_window():
    f = ActionFilter(CFG, FPS)
    out = []
    for i in range(WINDOW):
        out += f.push(cap(tag=i))
    assert out == [] and f.buffered == WINDOW  # nothing decided yet
    out += f.push(cap(tag=WINDOW))
    assert [c.tag for c in out] == [0] and f.buffered == WINDOW
    flushed = f.flush()
    assert f.buffered == 0 and [c.tag for c in flushed] == []  # all idle after the first


def test_multi_arm_idle_needs_every_arm_still():
    f = ActionFilter(CFG, FPS, ["arm0", "arm1"])
    arms = ("arm0", "arm1")
    caps = [cap(arms=arms, tag=0), cap(arms=arms, tag=1), cap(arms=arms, arm1_x=0.002, tag=2),
            cap(arms=arms, arm1_x=0.002, tag=3), cap(arms=arms, arm1_x=0.002, arm0_grip=0.6, tag=4)]
    f2 = ActionFilter(ActionFilterConfig(gripper_context_s=0.0), FPS, ["arm0", "arm1"])
    assert run(f2, caps) == [0, 2, 4]
    assert run(f, caps) == [0, 1, 2, 3, 4]  # the arm0 gripper change at 4 exempts +-1.6 s


def test_dagger_policy_frames_are_never_filtered():
    f = ActionFilter(CFG, FPS)
    caps = [cap(tag=i, source=0) for i in range(5)] + [cap(tag=5 + i, source=1) for i in range(5)] \
        + [cap(tag=10 + i, source=3) for i in range(3)]
    kept = run(f, caps)
    assert kept[:5] == [0, 1, 2, 3, 4]  # policy-driven: kept even though idle
    assert all(t not in kept for t in range(5, 13))  # human (teleop / takeover) idle: skipped
    assert f.frames_skipped == 8


def test_disabled_filter_passes_everything():
    f = ActionFilter(ActionFilterConfig(enabled=False), FPS)
    assert run(f, [cap(tag=i) for i in range(5)]) == [0, 1, 2, 3, 4]
    assert f.frames_skipped == 0 and f.gaps == [] and f.summary()["enabled"] is False
    assert ActionFilter(None, FPS).enabled is False


def test_reset_clears_state():
    f = ActionFilter(CFG, FPS)
    run(f, [cap(tag=i) for i in range(10)])
    f.reset()
    assert (f.frames_seen, f.frames_skipped, f.gaps, f.buffered) == (0, 0, [], 0)


# -- RecorderThread integration -----------------------------------------------------------------
def test_recorder_thread_skips_idle_frames_and_writes_the_filter_block():
    from test_recorder_thread import Q0, FakeCam, FakeKin, FakeRecorder, feed

    from apollo_mavis_v2_runtime.bus import RuntimeBus
    from apollo_mavis_v2_runtime.recorder.features import ArmMeta
    from apollo_mavis_v2_runtime.recorder.frames import RecordingFrameConverter
    from apollo_mavis_v2_runtime.recorder.thread import RecorderThread

    rec, bus, cam = FakeRecorder(), RuntimeBus(), FakeCam()
    thread = RecorderThread(
        rec, bus, {"cam0": cam}, [ArmMeta("arm0", True)],
        RecordingFrameConverter({"arm0": "arm_base:arm0"}), FakeKin(), fps=25,
        episode_meta_base={"session_id": "s1"},
        action_filter=ActionFilterConfig(gripper_context_s=0.2),  # 5-frame look-ahead
    )
    thread.request("new")
    t = 1.0
    q = Q0.copy()
    for i in range(10):  # moving: 2 mm per frame on joint 0 (FakeKin: tcp x = q[0])
        t += 0.04
        q = q + np.array([0.002, 0, 0, 0, 0, 0, 0, 0])
        feed(thread, bus, cam, i + 1, q, t)
    for i in range(20):  # stationary: idle
        t += 0.04
        feed(thread, bus, cam, 11 + i, q, t)
    for i in range(10):  # moving again
        t += 0.04
        q = q + np.array([0.002, 0, 0, 0, 0, 0, 0, 0])
        feed(thread, bus, cam, 31 + i, q, t)
    st = thread.status()
    assert st.state == "recording" and st.frames_skipped >= 15  # the stationary stretch
    assert thread.request("save") == (True, "saving")
    thread.run_iteration(t + 0.04)
    assert thread.status().state == "idle"
    payload = rec.sidecars[0]
    filt = payload["filter"]
    assert filt["enabled"] is True and filt["frames_seen"] == 40
    assert filt["frames_skipped"] == st.frames_skipped + 0 or filt["frames_skipped"] >= 15
    assert len(filt["gaps"]) == 1 and filt["gaps"][0][1] == filt["frames_skipped"]
    frames = rec.episodes[0]
    assert len(frames) == 40 - filt["frames_skipped"] - 1  # trailing kept capture has no action
    # the delta action of the frame before the gap spans the WHOLE gap (last kept -> next kept)
    dx = [float(f["action"][0]) for f in frames]
    assert max(dx) == pytest.approx(0.002, abs=1e-6)  # every kept-to-kept step is one motion step


def test_recorder_thread_save_flushes_a_short_episode_inside_the_look_ahead():
    """An episode shorter than the look-ahead window is entirely buffered when save
    arrives: the flush must still write it (not nack 'empty episode')."""
    from test_recorder_thread import Q0, FakeCam, FakeKin, FakeRecorder, feed

    from apollo_mavis_v2_runtime.bus import RuntimeBus
    from apollo_mavis_v2_runtime.recorder.features import ArmMeta
    from apollo_mavis_v2_runtime.recorder.frames import RecordingFrameConverter
    from apollo_mavis_v2_runtime.recorder.thread import RecorderThread

    rec, bus, cam = FakeRecorder(), RuntimeBus(), FakeCam()
    thread = RecorderThread(
        rec, bus, {"cam0": cam}, [ArmMeta("arm0", True)],
        RecordingFrameConverter({"arm0": "arm_base:arm0"}), FakeKin(), fps=25,
        action_filter=ActionFilterConfig(),  # 40-frame look-ahead
    )
    thread.request("new")
    t, q = 1.0, Q0.copy()
    for i in range(8):
        t += 0.04
        q = q + np.array([0.002, 0, 0, 0, 0, 0, 0, 0])
        feed(thread, bus, cam, i + 1, q, t)
    assert thread.status().frames == 0 and thread.filter.buffered == 8
    assert thread.request("save") == (True, "saving")
    thread.run_iteration(t + 0.04)
    assert len(rec.episodes) == 1 and len(rec.episodes[0]) == 7
    # an all-idle short episode saves as a discard (nothing to write)
    thread.request("new")
    for i in range(6):
        t += 0.04
        feed(thread, bus, cam, 20 + i, q, t)
    assert thread.request("save") == (True, "saving")
    thread.run_iteration(t + 0.04)
    st = thread.status()
    assert st.state == "idle" and len(rec.episodes) == 1 and "empty" in st.detail
