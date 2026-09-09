"""RecorderThread unit behaviour with a fake recorder/cameras/kinematics:
pending-frame action alignment, camera-age drops, discard/save semantics,
save retry + degrade, status transitions, the sidecar payload handed to
``save(sidecar, audio)`` (04-runtime §10; 10-frames §3, §11.6)."""

from __future__ import annotations

import numpy as np
import pytest
from apollo_mavis_v2_core import CameraFrame, CollisionReport, GripperState, Pose, se3
from apollo_mavis_v2_core.state import ArmState

from apollo_mavis_v2_runtime.bus import RuntimeBus
from apollo_mavis_v2_runtime.control.snapshot import StateSnapshot
from apollo_mavis_v2_runtime.recorder.features import ArmMeta
from apollo_mavis_v2_runtime.recorder.frames import RecordingFrameConverter
from apollo_mavis_v2_runtime.recorder.thread import RecorderThread


class FakeRecorder:
    """In-memory ``EpisodeDirRecorder`` double (the 10-frames §11 API: ``start`` mints
    an id, ``save(sidecar, audio)`` returns ``(ordinal, episode_id)``); scriptable
    save failures; remembers every sidecar + audio sink it was handed."""

    def __init__(self, fail_saves: int = 0) -> None:
        self.episodes: list[list[dict]] = []
        self.sidecars: list[dict] = []
        self.audio_sinks: list[object] = []
        self.buffer: list[dict] = []
        self.recording = False
        self.finalized = 0
        self.fail_saves = fail_saves
        self.episodes_saved = 0
        self.total_frames = 0
        self.repo_id = "apollo/fake"
        self.episode_id: str | None = None
        self._minted = 0

    def start(self, meta):
        self.buffer = []
        self.recording = True
        self._minted += 1
        self.episode_id = f"20260907T000000.{self._minted:03d}Z-{self._minted:06x}"

    def add_frame(self, frame):
        self.buffer.append(frame)

    def save(self, sidecar, audio=None):
        if self.fail_saves > 0:
            self.fail_saves -= 1
            raise RuntimeError("scripted save failure")
        self.episodes.append(self.buffer)
        self.sidecars.append(dict(sidecar))
        self.audio_sinks.append(audio)
        self.total_frames += len(self.buffer)
        self.buffer = []
        self.recording = False
        self.episodes_saved += 1
        eid, self.episode_id = self.episode_id, None
        return self.episodes_saved - 1, eid

    def discard(self):
        self.buffer = []
        self.recording = False
        self.episode_id = None

    def finalize(self):
        self.finalized += 1


class FakeCam:
    def __init__(self, camera_id="cam0", shape=(48, 64, 3)):
        self.camera_id = camera_id
        self.shape = shape
        self.t_mono = 0.0
        self.seq = 0

    def refresh(self, t):
        self.t_mono = t
        self.seq += 1

    def latest(self):
        rgb = np.full(self.shape, self.seq % 255, dtype=np.uint8)
        return CameraFrame(self.camera_id, rgb, self.t_mono, int(self.t_mono * 1e9), self.seq)


class FakeKin:
    """tcp_base = (q[0:3], rotvec q[3:6]); base_world = rail translation."""

    def base_world(self, arm_id, q):
        d = float(q[7]) if q.shape[0] > 7 else 0.0
        return Pose(np.array([0.0, d, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))

    def tcp_base(self, arm_id, q):
        return Pose(np.array(q[:3], dtype=float), se3.rotvec_to_quat(q[3:6]))


def make_state(q, wallclock_ns, ee=None):
    q = np.asarray(q, dtype=np.float64)
    return ArmState(
        arm_id="arm0", q=q, dq=np.zeros_like(q),
        ee_pose=ee or Pose(np.array([0.4, 0.0, 0.3]), np.array([0.0, 1.0, 0.0, 0.0])),
        gripper=GripperState(open_frac=0.5), rail_pos_m=float(q[7]),
        error_code=0, warn_code=0, mode=1, state=0, stale=False,
        t_mono=wallclock_ns / 1e9, wallclock_ns=wallclock_ns,
    )


def make_snap(tick, q_meas, q_cmd, wallclock_ns, grip_cmd=0.7):
    return StateSnapshot(
        t_mono=wallclock_ns / 1e9, wallclock_ns=wallclock_ns, tick=tick,
        arms={"arm0": make_state(q_meas, wallclock_ns)},
        q_cmd={"arm0": np.asarray(q_cmd, dtype=np.float64)},
        active_arm="arm0", gate=CollisionReport.ok(),
        gripper_frac={"arm0": grip_cmd},
    )


@pytest.fixture
def rig(tmp_path):
    rec = FakeRecorder()
    bus = RuntimeBus()
    cam = FakeCam()
    thread = RecorderThread(
        rec, bus, {"cam0": cam}, [ArmMeta("arm0", True)],
        RecordingFrameConverter({"arm0": "arm_base:arm0"}), FakeKin(),
        fps=25,
        episode_meta_base={"session_id": "s1", "frames": {"arm0": "arm_base:arm0"}},
        extrinsics_fn=lambda states: {"cam0": {"T_W_C": "stub"}},
    )
    return rec, bus, cam, thread


Q0 = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.10])


def feed(thread, bus, cam, tick, q_cmd, now, grip=0.7):
    cam.refresh(now)
    bus.snapshot.put(make_snap(tick, Q0, q_cmd, int(now * 1e9), grip))
    thread.run_iteration(now)


def test_transition_matrix(rig):
    rec, bus, cam, thread = rig
    assert thread.status().state == "idle"
    assert thread.request("save") == (False, "idle")
    assert thread.request("discard") == (False, "idle")
    ok, detail = thread.request("new")
    assert ok and thread.status().state == "recording"
    ok, detail = thread.request("new")  # invalid while recording
    assert not ok and detail == "recording"


def test_pending_frame_action_alignment(rig):
    rec, bus, cam, thread = rig
    thread.request("new")
    q1 = Q0.copy()
    q2 = q1 + np.array([0.002, 0, 0, 0, 0, 0.01, 0, 0.001])
    q3 = q2 + np.array([0, 0.004, 0, 0, 0, 0, 0, 0])
    feed(thread, bus, cam, 1, q1, 1.00, grip=0.7)
    feed(thread, bus, cam, 2, q2, 1.04, grip=0.8)
    feed(thread, bus, cam, 3, q3, 1.08)
    assert thread.status().frames == 2  # 3 captures -> 2 committed frames
    f1, f2 = rec.buffer
    # action[k] = executed commanded delta k -> k+1 (leading convention §3.2)
    assert np.allclose(f1["action"], [0.002, 0, 0, 0, 0, 0.01, 0.7, 0.001], atol=1e-9)
    assert np.allclose(f2["action"], [0, 0.004, 0, 0, 0, 0, 0.8, 0.0], atol=1e-9)
    # obs at frame k: measured state, 16 dims, ee verbatim (arm_base frame)
    s = f1["observation.state"]
    assert s.shape == (16,) and s.dtype == np.float32
    assert np.allclose(s[:7], Q0[:7], atol=1e-6)
    assert np.allclose(s[7:9], [0.5, 0.10], atol=1e-6)  # gripper, rail
    assert np.allclose(s[9:12], [0.4, 0.0, 0.3], atol=1e-6)
    assert np.allclose(s[12:], [0.0, 1.0, 0.0, 0.0], atol=1e-6)
    # fixed columns
    assert f1["intervention"][0] == np.False_
    assert f1["action_source"][0] == 1 and f1["action_source"].dtype == np.int8
    assert f1["wallclock_ns"].dtype == np.int64
    assert f1["wallclock_ns"][0] < f2["wallclock_ns"][0]  # strictly increasing
    assert f1["observation.images.cam0"].shape == (48, 64, 3)
    assert "task" not in f1  # FakeRecorder sees the raw frame (task added later)


def test_camera_age_drop_counts_and_skips(rig):
    rec, bus, cam, thread = rig
    thread.request("new")
    feed(thread, bus, cam, 1, Q0, 1.00)
    feed(thread, bus, cam, 2, Q0, 1.04)
    assert thread.status().frames == 1
    # camera stops producing: age > 2/fps = 0.08 s
    bus.snapshot.put(make_snap(3, Q0, Q0, int(1.20e9)))
    thread.run_iteration(1.20)  # cam.t_mono still 1.04 -> stale
    assert thread.status().frames == 1
    assert thread.frames_dropped_total == 1
    # camera recovers; pending chain resumes
    feed(thread, bus, cam, 4, Q0, 1.24)
    assert thread.status().frames == 2


def test_duplicate_tick_and_stale_wallclock_not_recorded(rig):
    rec, bus, cam, thread = rig
    thread.request("new")
    feed(thread, bus, cam, 1, Q0, 1.00)
    cam.refresh(1.04)
    thread.run_iteration(1.04)  # same snapshot tick -> no new capture
    assert thread.status().frames == 0
    assert rec.buffer == []


def test_discard_semantics(rig):
    rec, bus, cam, thread = rig
    thread.request("new")
    for i, t in enumerate((1.0, 1.04, 1.08)):
        feed(thread, bus, cam, i + 1, Q0, t)
    ok, _ = thread.request("discard")
    assert ok and thread.status().state == "saving"  # transient
    thread.run_iteration(1.12)
    assert thread.status().state == "idle"
    assert rec.episodes == [] and rec.buffer == [] and not rec.recording
    assert rec.sidecars == []  # nothing is written for a discarded episode
    assert thread.open_episode_id is None


def test_save_hands_the_sidecar_to_the_recorder_and_drops_trailing_pending(rig):
    rec, bus, cam, thread = rig
    thread.request("new")
    open_id = thread.open_episode_id
    assert open_id == rec.episode_id and open_id is not None
    for i, t in enumerate((1.0, 1.04, 1.08, 1.12)):
        feed(thread, bus, cam, i + 1, Q0 + i * 0.001, t)
    assert thread.request("save") == (True, "saving")
    assert thread.status().state == "saving"
    assert thread.open_episode_id == open_id  # still open while saving (delete -> 409)
    thread.run_iteration(1.16)
    st = thread.status()
    assert st.state == "idle" and st.index == 0 and st.total_episodes == 1
    assert st.total_frames == 3 and st.repo_id == "apollo/fake"
    assert len(rec.episodes) == 1 and len(rec.episodes[0]) == 3  # trailing pending dropped
    assert thread.last_saved_id == open_id and thread.open_episode_id is None
    # the sidecar payload is assembled BEFORE save (10-frames §11.6 step 3)
    payload = rec.sidecars[0]
    assert payload["session_id"] == "s1"
    assert payload["frames"] == {"arm0": "arm_base:arm0"}
    assert payload["frames_dropped"] == 0 and payload["success"] is None
    assert payload["extrinsics"] == {"cam0": {"T_W_C": "stub"}}
    assert rec.audio_sinks == [None]


def test_on_episode_done_fires_before_the_flip_to_idle(rig):
    """The manager's return-to-start hook must see ``saving`` (never a spurious
    ``idle`` frame): ``set_returning(True)`` inside the hook makes the very next
    status read ``returning``."""
    rec, bus, cam, thread = rig
    seen = []

    def hook(outcome, index):
        seen.append((outcome, index, thread.status().state))
        thread.set_returning(True, "returning to profile 'ready'")

    thread.on_episode_done = hook
    thread.request("new")
    for i, t in enumerate((1.0, 1.04, 1.08)):
        feed(thread, bus, cam, i + 1, Q0, t)
    thread.request("save")
    thread.run_iteration(1.12)
    assert seen == [("saved", 0, "saving")]
    st = thread.status()
    assert st.state == "returning" and st.detail == "returning to profile 'ready'"
    assert thread.request("new") == (False, "returning to the initial configuration")
    thread.set_returning(False, "return cancelled: movement key")
    st = thread.status()
    assert st.state == "idle" and st.detail == "return cancelled: movement key"
    assert thread.request("new")[0] is True
    assert thread.status().detail == ""  # a new episode clears the note
    # discard fires the hook too (idle -> returning -> idle path)
    seen.clear()
    thread.request("discard")
    thread.run_iteration(1.20)
    assert seen == [("discarded", None, "saving")]
    assert thread.status().state == "returning"


def test_empty_episode_save_nacked(rig):
    rec, bus, cam, thread = rig
    thread.request("new")
    assert thread.request("save") == (False, "empty episode")
    feed(thread, bus, cam, 1, Q0, 1.0)  # one capture = still zero committed frames
    assert thread.request("save") == (False, "empty episode")


def test_save_retry_once_then_degrade(tmp_path):
    bus = RuntimeBus()
    cam = FakeCam()
    # one failure: retry succeeds, buffer kept
    rec = FakeRecorder(fail_saves=1)
    thread = RecorderThread(
        rec, bus, {"cam0": cam}, [ArmMeta("arm0", True)],
        RecordingFrameConverter({"arm0": "arm_base:arm0"}), FakeKin(), fps=25,
    )
    thread.request("new")
    for i, t in enumerate((1.0, 1.04, 1.08)):
        feed(thread, bus, cam, i + 1, Q0, t)
    thread.request("save")
    thread.run_iteration(1.12)
    assert len(rec.episodes) == 1 and not thread.degraded
    # two failures: degrade (recording off), dataset still finalizable
    rec2 = FakeRecorder(fail_saves=2)
    thread2 = RecorderThread(
        rec2, bus, {"cam0": cam}, [ArmMeta("arm0", True)],
        RecordingFrameConverter({"arm0": "arm_base:arm0"}), FakeKin(), fps=25,
    )
    discarded, done = [], []
    thread2._episode_discarded = lambda i, eid, why: discarded.append((i, eid, why))
    thread2.on_episode_done = lambda outcome, index: done.append(outcome)
    thread2.request("new")
    eid2 = rec2.episode_id
    for i, t in enumerate((2.0, 2.04, 2.08)):
        feed(thread2, bus, cam, i + 1, Q0, t)
    thread2.request("save")
    thread2.run_iteration(2.12)
    assert thread2.degraded and thread2.status().state == "idle"
    assert rec2.buffer  # buffer KEPT (04-runtime §15): never recorder.discard() here
    # ... but the episode is over and says so (an Online DAgger trainer that watched the
    # rollout live is told to drop it: events.episode_discarded rides this hook); no
    # return-to-start motion is started on a failure path
    assert discarded == [(0, eid2, "save failed twice - recording degraded (buffer kept)")]
    assert done == []
    assert thread2.request("new")[0] is False
    thread2.stop()
    assert rec2.finalized == 1  # guard still finalizes
    assert len(discarded) == 1  # the shutdown found the recorder idle: no second announcement


def test_stop_discards_open_buffer_and_finalizes_once(rig):
    rec, bus, cam, thread = rig
    thread.request("new")
    for i, t in enumerate((1.0, 1.04)):
        feed(thread, bus, cam, i + 1, Q0, t)
    thread.stop()
    assert rec.episodes == [] and not rec.recording
    assert rec.finalized == 1
    thread.stop()  # idempotent
    assert rec.finalized == 1


def test_stop_defers_the_shutdown_while_the_recorder_thread_is_still_busy(rig, monkeypatch):
    """A save that outlives the 30 s join (a long finish_episode) must not be raced by
    a concurrent _shutdown: stop() logs and leaves it to the thread's finally."""
    rec, bus, cam, thread = rig

    class Busy:
        def join(self, timeout=None):
            return None

        def is_alive(self):
            return True

    thread._thread = Busy()
    thread._running = True
    thread.stop()
    assert rec.finalized == 0 and thread._thread is not None  # deferred, not run here
    thread._thread = None
    thread.stop()  # the thread is gone: the shutdown runs once
    assert rec.finalized == 1
