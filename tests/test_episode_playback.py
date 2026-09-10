"""Episode playback (2026-09-10; operator request, 04-runtime §10.8, 05-ui §8.1 item 7).

The Welcome page's Datasets panel gets a **Playback** button per episode, opening a
dialog with "return to this episode's initial state" and "play back the whole episode",
the second gated on the first. This file covers the runtime half of the first action and
the reader both share:

* the ``observation.state`` column parser, which is what keeps an episode recorded with
  another arm set from being mis-sliced in silence;
* ``load_trajectory``'s refusals, each of which the REST layer turns into a 404 / 409
  with the same text the operator reads;
* the transient profile the goto is built on — the reason the motion inherits the twin
  planner, the gate, the two separately planned phases and one-arm-at-a-time execution
  for free instead of growing a second motion path;
* the REST contract end to end over ``create_app`` + a sim workcell.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from conftest import make_runtime_config
from starlette.testclient import TestClient

from apollo_mavis_v2_runtime.recorder.features import arm_state_names
from apollo_mavis_v2_runtime.recorder.playback import (
    ExecutorCaps,
    PlaybackError,
    load_trajectory,
    parse_state_names,
    resample,
)
from apollo_mavis_v2_runtime.runtime import Runtime
from apollo_mavis_v2_runtime.server.app import create_app

FPS = 25.0


# -- the column parser ----------------------------------------------------------------
def test_parses_the_real_two_arm_layout():
    """The layout ``recorder/features`` writes for this cell: joints, gripper, rail, then
    the measured TCP pose, per arm, Manipulation Arm first."""
    names = arm_state_names("grip", True) + arm_state_names("view", True)
    grip, view = parse_state_names(names)
    assert (grip.arm_id, view.arm_id) == ("grip", "view")
    assert grip.joints == (0, 1, 2, 3, 4, 5, 6)
    assert (grip.gripper, grip.rail) == (7, 8)
    # The TCP pose dims sit between the two arms' blocks and are skipped, so the second
    # arm's joints are NOT at index 9 — reading them positionally would be wrong.
    assert view.joints[0] > 9
    assert view.rail == view.joints[-1] + 2


def test_an_arm_without_a_track_reads_back_with_no_rail():
    names = arm_state_names("grip", True) + arm_state_names("view", False)
    grip, view = parse_state_names(names)
    assert grip.rail is not None
    assert view.rail is None  # "keep the carriage": there is nothing to command


def test_a_single_arm_episode_reads_back():
    (only,) = parse_state_names(arm_state_names("arm0", False))
    assert only.arm_id == "arm0" and only.rail is None and only.gripper is not None


def test_a_layout_missing_a_joint_is_refused_not_guessed():
    names = [n for n in arm_state_names("grip", True) if not n.endswith("joint4.pos")]
    with pytest.raises(PlaybackError, match="joint4.pos"):
        parse_state_names(names)


def test_a_layout_with_no_joints_at_all_is_refused():
    with pytest.raises(PlaybackError, match="names no arm joints"):
        parse_state_names(["grip_ee.x", "grip_ee.y"])


# -- a synthetic episode on disk -------------------------------------------------------
def write_episode(
    root: Path,
    episode_id: str,
    *,
    arms: dict[str, bool],  # arm_id -> has_rail
    frames: int = 4,
    manifest_names: bool = True,
) -> Path:
    """One episode directory in the 10-frames §11 layout, with a real parquet."""
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    names: list[str] = []
    for arm_id, has_rail in arms.items():
        names += arm_state_names(arm_id, has_rail)
    # Row k = every dim at k/100, so a frame's values are recognisable per index.
    rows = [[(k + 1) / 100.0 + i / 1000.0 for i in range(len(names))] for k in range(frames)]

    ds_root = root
    (ds_root / "episodes" / episode_id).mkdir(parents=True, exist_ok=True)
    directory = ds_root / "episodes" / episode_id
    features = {"observation.state": {"dtype": "float32", "names": names}}
    if manifest_names:
        (ds_root / "manifest.json").write_text(
            json.dumps({"apollo_dataset_layout": 1, "fps": FPS, "features": features})
        )
    else:
        (ds_root / "manifest.json").write_text(json.dumps({"apollo_dataset_layout": 1, "fps": FPS}))
    (directory / "episode.json").write_text(
        json.dumps({"length": frames, "fps": FPS, "tasks": ["t"], "duration_s": frames / FPS})
    )
    table = pa.table({"observation.state": pa.array(rows, type=pa.list_(pa.float32()))})
    pq.write_table(table, directory / "frames.parquet")
    return directory


def test_load_trajectory_reads_the_initial_state(tmp_path):
    write_episode(tmp_path, "20260910T000000.000Z-aaaaaa", arms={"grip": True, "view": True})
    traj = load_trajectory("bc/x", tmp_path, "20260910T000000.000Z-aaaaaa")
    assert traj.frames == 4
    assert traj.fps == FPS
    assert traj.duration_s == pytest.approx(4 / FPS)
    assert traj.arm_ids == ["grip", "view"]
    initial = traj.initial_state()
    assert set(initial) == {"grip", "view"}
    # Frame 0's dims are 0.01 + i/1000, so the joints of the FIRST arm are the first 7.
    assert initial["grip"].q == pytest.approx([0.01 + i / 1000 for i in range(7)])
    assert initial["grip"].rail_pos_m == pytest.approx(0.01 + 8 / 1000)
    assert initial["grip"].gripper_open_frac == pytest.approx(0.01 + 7 / 1000)
    # ... and the LAST frame is a different posture (state_at walks the whole episode).
    assert traj.state_at(3)["grip"].q[0] == pytest.approx(0.04)


def test_the_episode_json_features_are_the_fallback_when_a_manifest_lacks_names(tmp_path):
    """An episode directory copied out of its dataset must still read."""
    directory = write_episode(
        tmp_path, "20260910T000000.000Z-bbbbbb", arms={"grip": True}, manifest_names=False
    )
    meta = json.loads((directory / "episode.json").read_text())
    meta["features"] = {"observation.state": {"names": arm_state_names("grip", True)}}
    (directory / "episode.json").write_text(json.dumps(meta))
    traj = load_trajectory("bc/x", tmp_path, "20260910T000000.000Z-bbbbbb")
    assert traj.arm_ids == ["grip"]


def test_no_column_names_anywhere_is_refused_with_a_reason(tmp_path):
    write_episode(
        tmp_path, "20260910T000000.000Z-cccccc", arms={"grip": True}, manifest_names=False
    )
    with pytest.raises(PlaybackError, match="no observation.state column names"):
        load_trajectory("bc/x", tmp_path, "20260910T000000.000Z-cccccc")


def test_an_unknown_episode_is_a_not_found(tmp_path):
    write_episode(tmp_path, "20260910T000000.000Z-dddddd", arms={"grip": True})
    with pytest.raises(PlaybackError) as e:
        load_trajectory("bc/x", tmp_path, "20260910T000000.000Z-999999")
    assert e.value.not_found is True


def test_a_missing_parquet_is_refused(tmp_path):
    directory = write_episode(tmp_path, "20260910T000000.000Z-eeeeee", arms={"grip": True})
    (directory / "frames.parquet").unlink()
    with pytest.raises(PlaybackError, match="has no frames.parquet"):
        load_trajectory("bc/x", tmp_path, "20260910T000000.000Z-eeeeee")


def test_a_corrupt_parquet_is_a_refusal_not_a_crash(tmp_path):
    directory = write_episode(tmp_path, "20260910T000000.000Z-ffffff", arms={"grip": True})
    (directory / "frames.parquet").write_bytes(b"not a parquet file")
    with pytest.raises(PlaybackError, match="is unreadable"):
        load_trajectory("bc/x", tmp_path, "20260910T000000.000Z-ffffff")


def test_a_row_that_does_not_match_the_names_is_refused(tmp_path):
    """A width mismatch means the columns cannot be trusted — and a mis-sliced row is a
    posture command, so guessing is not an option."""
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    directory = write_episode(tmp_path, "20260910T000000.000Z-a0a0a0", arms={"grip": True})
    short = pa.table({"observation.state": pa.array([[0.0, 1.0]], type=pa.list_(pa.float32()))})
    pq.write_table(short, directory / "frames.parquet")
    with pytest.raises(PlaybackError, match="but the manifest names"):
        load_trajectory("bc/x", tmp_path, "20260910T000000.000Z-a0a0a0")


# -- the REST contract -----------------------------------------------------------------
SPEC = {
    "mode": "teleop",
    "kind": "sim",
    "arms": ["arm0"],
    "frames": {"arm0": "arm_base:arm0"},
    "sim_scene": "single_rail",
}


@pytest.fixture()
def client(tmp_path):
    cfg = make_runtime_config(tmp_path)
    rt = Runtime(cfg)
    # The generic namespace root of the test config: datasets_root/<ns>/<name>.
    ds_root = cfg.datasets_root / "bc" / "x"
    write_episode(ds_root, "20260910T000000.000Z-a1b2c3", arms={"arm0": True}, frames=3)
    with TestClient(create_app(rt)) as c:
        yield c
        c.delete("/api/session")


PLAYBACK = "/api/datasets/bc/x/episodes/20260910T000000.000Z-a1b2c3/playback"


def test_playback_info_is_session_less_and_says_why_it_cannot_run(client):
    """The dialog must open and explain itself BEFORE anything moves — the operator
    should never have to interpret a bare 409."""
    got = client.get(PLAYBACK)
    assert got.status_code == 200, got.text
    body = got.json()
    assert body["frames"] == 3 and body["fps"] == FPS
    assert body["duration_s"] == pytest.approx(3 / FPS)
    assert [a["arm_id"] for a in body["arms"]] == ["arm0"]
    assert body["playable"] is False
    assert "Start a session first" in body["reason"]


def test_playback_info_is_playable_inside_a_matching_session(client):
    assert client.post("/api/session", json=SPEC).status_code == 200
    body = client.get(PLAYBACK).json()
    assert body["playable"] is True and body["reason"] == ""


def test_playback_info_404s_an_unknown_episode_and_dataset(client):
    assert (
        client.get("/api/datasets/bc/x/episodes/20260910T000000.000Z-999999/playback").status_code
        == 404
    )
    assert (
        client.get(
            "/api/datasets/bc/nope/episodes/20260910T000000.000Z-a1b2c3/playback"
        ).status_code
        == 404
    )


def test_goto_initial_without_a_session_is_a_refusal_not_an_error(client):
    """Like ``return_home``: an operational refusal is a 200 with ``ok: false`` and a
    ``detail`` the dialog shows, so a thrown error in the UI really is a transport fault."""
    r = client.post(
        "/api/session/playback",
        json={
            "repo_id": "bc/x",
            "episode_id": "20260910T000000.000Z-a1b2c3",
            "action": "goto_initial",
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is False
    assert r.json()["detail"] == "no active session"


def test_goto_initial_walks_the_arms_to_the_first_frame(client):
    """The whole point: after this the arms ARE at frame 0, which is what unlocks the
    Playback button. Runs the real twin planner, gate and executor in sim."""
    assert client.post("/api/session", json=SPEC).status_code == 200
    body = client.get(PLAYBACK).json()
    target = next(a for a in body["arms"] if a["arm_id"] == "arm0")
    r = client.post(
        "/api/session/playback",
        json={
            "repo_id": "bc/x",
            "episode_id": "20260910T000000.000Z-a1b2c3",
            "action": "goto_initial",
        },
    )
    assert r.status_code == 200, r.text
    result = r.json()
    assert result["ok"] is True, result
    assert result["status"] in ("done", "skipped"), result
    if result["status"] == "done":
        with client.websocket_connect("/ws/telemetry") as ws:
            arms = ws.receive_json()["arms"]
        row = next(a for a in arms if a["arm_id"] == "arm0")
        assert row["q"] == pytest.approx(target["q"], abs=2e-3)
        assert row["rail_pos_m"] == pytest.approx(target["rail_pos_m"], abs=2e-3)


def test_goto_initial_refuses_an_episode_whose_arms_the_session_does_not_drive(client, tmp_path):
    """An episode recorded with another arm set cannot be placed at all — refused by
    name, never silently applied to whichever arms happen to be there."""
    cfg_root = client.app.state.runtime.cfg.datasets_root / "bc" / "other"
    write_episode(cfg_root, "20260910T000000.000Z-b2b2b2", arms={"grip": True}, frames=2)
    assert client.post("/api/session", json=SPEC).status_code == 200
    other = "/api/datasets/bc/other/episodes/20260910T000000.000Z-b2b2b2/playback"
    info = client.get(other).json()
    assert info["playable"] is False
    assert "does not drive" in info["reason"]
    r = client.post(
        "/api/session/playback",
        json={
            "repo_id": "bc/other",
            "episode_id": "20260910T000000.000Z-b2b2b2",
            "action": "goto_initial",
        },
    )
    assert r.json()["ok"] is False and "does not drive" in r.json()["detail"]


def test_the_request_body_cannot_escape_the_dataset_roots(client):
    """``repo_id`` / ``episode_id`` arrive in a JSON body and are joined onto a
    filesystem root, so the model validates them (422), not just the path routes."""
    for body in (
        {"repo_id": "../../etc", "episode_id": "x", "action": "goto_initial"},
        {"repo_id": "bc/x", "episode_id": "../../../etc/passwd", "action": "goto_initial"},
        {"repo_id": "bc/x/y", "episode_id": "x", "action": "goto_initial"},
        {"repo_id": "bc/x", "episode_id": "..", "action": "goto_initial"},
    ):
        assert client.post("/api/session/playback", json=body).status_code == 422, body


def test_the_transient_profile_is_never_stored(client):
    """The goto is given a ``StateProfile`` only to reuse the return-to-initial path; it
    must not appear in the operator's profile list."""
    assert client.post("/api/session", json=SPEC).status_code == 200
    before = client.get("/api/profiles").json()
    client.post(
        "/api/session/playback",
        json={
            "repo_id": "bc/x",
            "episode_id": "20260910T000000.000Z-a1b2c3",
            "action": "goto_initial",
        },
    )
    assert client.get("/api/profiles").json() == before


def test_the_initial_profile_covers_only_the_sessions_arms_and_clamps_the_gripper(tmp_path):
    """Unit-level: the transient profile's shape. ``ArmPosture`` clamps ``[0, 1]``, so a
    recording whose gripper column is slightly out of range must not raise."""
    cfg = make_runtime_config(tmp_path)
    rt = Runtime(cfg)
    ds_root = cfg.datasets_root / "bc" / "two"
    write_episode(ds_root, "20260910T000000.000Z-c3c3c3", arms={"arm0": True, "ghost": True})
    traj = load_trajectory("bc/two", ds_root, "20260910T000000.000Z-c3c3c3")

    class FakeSession:
        spec = type("S", (), {"kind": "sim", "arms": ["arm0"]})()

    profile = rt.manager._episode_initial_profile(FakeSession(), traj)  # noqa: SLF001
    assert set(profile.arms) == {"arm0"}, "an arm the session does not drive is dropped"
    assert profile.profile_id == "" and "Transient" in profile.notes
    assert 0.0 <= profile.arms["arm0"].gripper_open_frac <= 1.0
    assert all(math.isfinite(v) for v in profile.arms["arm0"].q)


# -- the resampler: what makes a multi-arm replay safe ---------------------------------
# The loop's ONE op that moves several arms at once is playback_path, and what earns it
# that right is this function plus the twin verification: the arms must stay in lockstep
# on exactly the trajectory that was recorded.
SIM_CAPS = ExecutorCaps(slew_rad_per_tick=0.02, rail_m_per_tick=0.002)


def make_traj(tmp_path, arms, frames=5, name="20260910T000000.000Z-r00001"):
    write_episode(tmp_path, name, arms=arms, frames=frames)
    return load_trajectory("bc/x", tmp_path, name)


def test_every_arm_gets_the_same_number_of_waypoints(tmp_path):
    """Different lengths would desynchronise the arms tick by tick, i.e. silently replay a
    path that was never recorded — the loop refuses such a command outright."""
    traj = make_traj(tmp_path, {"grip": True, "view": True})
    plan = resample(traj, SIM_CAPS, loop_hz=100.0)
    lengths = {a: len(w) for a, w in plan.waypoints.items()}
    assert len(set(lengths.values())) == 1, lengths
    assert set(plan.waypoints) == {"grip", "view"}


def test_real_time_when_the_caps_allow_it(tmp_path):
    """25 fps on a 100 Hz loop = 4 waypoints per recorded frame, so the replay takes as
    long as the recording did. The fixture's rail moves 10 mm per frame, well past the
    2 mm/tick cap, so this is checked on a trackless arm where only the joints bind."""
    traj = make_traj(tmp_path, {"grip": False}, frames=5)
    plan = resample(traj, SIM_CAPS, loop_hz=100.0)
    assert plan.length == 1 + 4 * 4  # first frame + 4 intervals x 4 ticks
    assert plan.slowdown == pytest.approx(1.0)
    assert plan.length / 100.0 == pytest.approx(traj.frames / traj.fps, abs=0.05)


def test_a_fast_carriage_alone_is_enough_to_slow_the_replay(tmp_path):
    """The rail is the cap that binds first on this cell (2 mm/tick vs 0.02 rad): a
    recording whose carriage moved faster than the track's positioning cap is replayed
    slower, and the arms stay in step because the SAME factor applies to both."""
    traj = make_traj(tmp_path, {"grip": True}, frames=5)  # 10 mm per frame
    plan = resample(traj, SIM_CAPS, loop_hz=100.0)
    assert plan.slowdown > 1.2
    # ceil(0.01 / 0.002) = 5 ticks per interval, one or two more where float32 rounding
    # pushes the recorded delta a hair over 10 mm.
    assert 1 + 4 * 5 <= plan.length <= 1 + 4 * 6


def test_a_recording_faster_than_the_caps_is_slowed_UNIFORMLY(tmp_path):
    """Recorded at full speed, replayed on a 10 %-speed session: the replay must slow
    down, and by the SAME factor for both arms — a per-arm scale would change their
    relative timing, which is the one thing the twin never validated."""
    traj = make_traj(tmp_path, {"grip": True, "view": True})
    slow = ExecutorCaps(slew_rad_per_tick=0.0005, rail_m_per_tick=0.0002)
    plan = resample(traj, slow, loop_hz=100.0)
    assert plan.slowdown > 1.5
    assert len(set(len(w) for w in plan.waypoints.values())) == 1


def test_no_segment_exceeds_a_single_executor_tick(tmp_path):
    """The property the lockstep rests on: the executor walks one waypoint per tick only
    while every segment is within its caps, otherwise it subdivides on its own and the
    arms drift apart."""
    traj = make_traj(tmp_path, {"grip": True, "view": True})
    for caps in (SIM_CAPS, ExecutorCaps(slew_rad_per_tick=0.0005, rail_m_per_tick=0.0002)):
        plan = resample(traj, caps, loop_hz=100.0)
        for arm, wps in plan.waypoints.items():
            for k in range(len(wps) - 1):
                assert caps.ticks_for(wps[k], wps[k + 1]) == 1, (arm, k)


def test_the_hardware_cartesian_bound_is_respected_too(tmp_path):
    """On hardware the executor also bounds the lever-weighted Cartesian step; a
    resampling that ignored it would be subdivided by the executor and desynchronise."""
    traj = make_traj(tmp_path, {"grip": True})
    hw = ExecutorCaps(
        slew_rad_per_tick=0.02,
        rail_m_per_tick=0.002,
        cart_step_m=0.0004,  # 10 % of the cell's 4 mm
        lever_arm_m=(0.9, 0.8, 0.7, 0.6, 0.4, 0.2, 0.1),
    )
    plan = resample(traj, hw, loop_hz=100.0)
    for wps in plan.waypoints.values():
        for k in range(len(wps) - 1):
            assert hw.ticks_for(wps[k], wps[k + 1]) == 1


def test_the_waypoints_start_at_frame_0_and_end_at_the_last_frame(tmp_path):
    traj = make_traj(tmp_path, {"grip": True}, frames=4)
    plan = resample(traj, SIM_CAPS, loop_hz=100.0)
    first, last = traj.state_at(0)["grip"], traj.state_at(3)["grip"]
    assert plan.waypoints["grip"][0] == pytest.approx([*first.q, first.rail_pos_m])
    assert plan.waypoints["grip"][-1] == pytest.approx([*last.q, last.rail_pos_m])


def test_the_gripper_track_follows_the_waypoints_one_for_one(tmp_path):
    """The recorded opening belongs to the frame the arm is AT, so it is a track, not a
    single target applied on arrival — for a manipulation episode the grip IS the task."""
    traj = make_traj(tmp_path, {"grip": True}, frames=4)
    plan = resample(traj, SIM_CAPS, loop_hz=100.0)
    assert len(plan.gripper["grip"]) == plan.length
    assert all(0.0 <= v <= 1.0 for v in plan.gripper["grip"])
    assert plan.gripper["grip"][0] == pytest.approx(traj.state_at(0)["grip"].gripper_open_frac)


def test_an_arm_with_no_rail_column_holds_its_carriage(tmp_path):
    """`rail_hold` supplies the live carriage position so the slot is commanded to stay
    put, rather than being dropped (which would leave the arm's rail uncommanded)."""
    traj = make_traj(tmp_path, {"grip": False})
    plan = resample(traj, SIM_CAPS, loop_hz=100.0, rail_hold={"grip": 0.42})
    assert all(w[7] == pytest.approx(0.42) for w in plan.waypoints["grip"])


def test_resampling_only_the_sessions_arms(tmp_path):
    traj = make_traj(tmp_path, {"grip": True, "view": True})
    plan = resample(traj, SIM_CAPS, loop_hz=100.0, arms=["grip"])
    assert set(plan.waypoints) == {"grip"}
    with pytest.raises(PlaybackError, match="names none of this session's arms"):
        resample(traj, SIM_CAPS, loop_hz=100.0, arms=["nobody"])


def test_a_single_frame_episode_is_one_waypoint(tmp_path):
    traj = make_traj(tmp_path, {"grip": True}, frames=1)
    plan = resample(traj, SIM_CAPS, loop_hz=100.0)
    assert plan.length == 1


def test_play_refuses_when_the_arms_are_not_at_the_first_frame(client):
    """The dialog's rule enforced server-side: a drifted cell is news, not something to
    quietly walk away from."""
    assert client.post("/api/session", json=SPEC).status_code == 200
    r = client.post(
        "/api/session/playback",
        json={
            "repo_id": "bc/x",
            "episode_id": "20260910T000000.000Z-a1b2c3",
            "action": "play",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is False and body["status"] == "refused"
    assert "Return to the initial state" in body["detail"]


def test_stop_is_idempotent_and_says_nothing_is_playing(client):
    assert client.post("/api/session", json=SPEC).status_code == 200
    for _ in range(2):
        r = client.post(
            "/api/session/playback",
            json={"repo_id": "bc/x", "episode_id": "20260910T000000.000Z-a1b2c3", "action": "stop"},
        )
        assert r.status_code == 200
        assert r.json()["ok"] is True and "nothing is playing" in r.json()["detail"]


def test_play_then_the_arms_walk_the_recorded_trajectory(client):
    """End to end in sim: place the arms at frame 0, replay, and land on the LAST frame."""
    assert client.post("/api/session", json=SPEC).status_code == 200
    body = {"repo_id": "bc/x", "episode_id": "20260910T000000.000Z-a1b2c3"}
    first = client.post("/api/session/playback", json={**body, "action": "goto_initial"}).json()
    assert first["ok"] is True, first
    played = client.post("/api/session/playback", json={**body, "action": "play"}).json()
    assert played["ok"] is True, played
    assert played["status"] == "done"
    assert "replayed 3 frames" in played["detail"]
    traj = load_trajectory(
        "bc/x",
        client.app.state.runtime.cfg.datasets_root / "bc" / "x",
        "20260910T000000.000Z-a1b2c3",
    )
    last = traj.state_at(traj.frames - 1)["arm0"]
    with client.websocket_connect("/ws/telemetry") as ws:
        arms = ws.receive_json()["arms"]
    row = next(a for a in arms if a["arm_id"] == "arm0")
    assert row["q"] == pytest.approx(last.q, abs=3e-3)
    assert row["rail_pos_m"] == pytest.approx(last.rail_pos_m, abs=3e-3)


# -- the crash of 2026-09-10: verification must not touch the loop's twin data ---------
# A real hardware playback died at waypoint 244 with
#   mujoco.FatalError: collisionTask: collision function returned 0 contacts for geom
#   pair (22, 24), expected at most -75 from mj_maxContact
# because `_verify_playback` wrote qpos and ran mj_collision on the SESSION TWIN's own
# MjData from the REST thread while the 100 Hz loop was gating ticks through
# `twin.check()` on the same data. The negative budget is MuJoCo's contact bookkeeping
# already corrupted. Both tests below fail if the private-MjData fix is undone.
def _plan_for(tmp_path, arms=("grip", "view")):
    from apollo_mavis_v2_runtime.recorder.playback import resample

    traj = make_traj(
        tmp_path, dict.fromkeys(arms, True), frames=6, name="20260910T010000.000Z-c0ffee"
    )
    return resample(traj, SIM_CAPS, loop_hz=100.0, arms=list(arms))


def _fake_session(twin):
    import types

    return types.SimpleNamespace(
        twin=twin,
        spec=types.SimpleNamespace(kind="hardware", arms=["grip", "view"]),
        loop=types.SimpleNamespace(cfg=types.SimpleNamespace(rate_hz=100.0)),
    )


def test_verification_never_mutates_the_twins_own_data(tmp_path):
    """The invariant: the twin's `data` belongs to the control loop. Verification reads
    it once (a snapshot for the non-arm dofs) and writes only its OWN private copy."""
    pytest.importorskip("mujoco")
    sim = pytest.importorskip("apollo_mavis_v2_sim")
    import numpy as np

    twin = sim.DigitalTwin(sim.REGISTRY.build("mavis_v2"), inflation_m=0.025)
    before = np.array(twin.data.qpos)
    rt = Runtime(make_runtime_config(tmp_path))
    reason = rt.manager._verify_playback(_fake_session(twin), _plan_for(tmp_path))  # noqa: SLF001
    assert isinstance(reason, str)  # "" or a refusal — either is fine here
    assert np.array_equal(twin.data.qpos, before), "verification wrote the loop's qpos"


def test_verification_survives_a_loop_gating_the_same_twin(tmp_path):
    """The actual crash, reproduced: verify on the REST thread while another thread does
    what the 100 Hz gate does. Before the fix this raised mujoco.FatalError."""
    pytest.importorskip("mujoco")
    sim = pytest.importorskip("apollo_mavis_v2_sim")
    import threading

    import numpy as np

    twin = sim.DigitalTwin(sim.REGISTRY.build("mavis_v2"), inflation_m=0.025)
    plan = _plan_for(tmp_path)
    rt = Runtime(make_runtime_config(tmp_path))
    q_meas = {a: np.array(twin.data.qpos[twin.addr[a].qpos_adr]) for a in ("grip", "view")}
    stop = threading.Event()
    errors: list[BaseException] = []

    def gate_ticks() -> None:
        while not stop.is_set():
            try:
                twin.check(q_meas)  # exactly what SafetyGate.filter does every tick
            except BaseException as e:  # noqa: BLE001 - record it, do not hide it
                errors.append(e)
                return

    ticker = threading.Thread(target=gate_ticks, name="fake-gate", daemon=True)
    ticker.start()
    try:
        for _ in range(5):  # several passes: the race needs a few thousand checks
            reason = rt.manager._verify_playback(_fake_session(twin), plan)  # noqa: SLF001
            assert "could not check waypoint" not in reason, reason
    finally:
        stop.set()
        ticker.join(timeout=5.0)
    assert errors == [], f"the gate thread's twin was corrupted: {errors[0]!r}"
