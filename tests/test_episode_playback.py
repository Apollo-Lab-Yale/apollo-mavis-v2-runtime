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
* the REST contract end to end over ``create_app`` + a sim workcell;
* (2026-09-11) the ACTION replay sources ``delta_ee`` / ``abs_ee``: the action-column
  reader, the ``ReplayActionSource`` the loop's replay slot consumes, the refusals
  (hardware, policy sessions, a missing column) and two sim replays of an FK-consistent
  synthetic episode that must land on the last recorded frame.
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from apollo_mavis_v2_core import se3
from conftest import make_runtime_config
from starlette.testclient import TestClient

from apollo_mavis_v2_runtime.recorder.features import arm_action_names, arm_state_names
from apollo_mavis_v2_runtime.recorder.playback import (
    ExecutorCaps,
    PlaybackError,
    load_trajectory,
    parse_action_names,
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
ACTION_SOURCES = ("delta_ee", "abs_ee")


def _action_feature(arms: dict[str, bool], space: str, names: list[str]) -> dict:
    """The manifest feature block ``recorder/features.build_features`` writes."""
    return {
        "dtype": "float32",
        "shape": [len(names)],
        "names": names,
        "info": {
            "apollo_schema": 1,
            "action_space": space,
            "frames": {a: f"arm_base:{a}" for a in arms},
        },
    }


def write_episode(
    root: Path,
    episode_id: str,
    *,
    arms: dict[str, bool],  # arm_id -> has_rail
    frames: int = 4,
    manifest_names: bool = True,
    actions: tuple[str, ...] = ACTION_SOURCES,
    state_rows: list[list[float]] | None = None,
    action_rows: dict[str, np.ndarray] | None = None,
) -> Path:
    """One episode directory in the 10-frames §11 layout, with a real parquet.

    ``observation.state`` always; the ``action`` (delta_ee) and ``action.abs_ee`` columns
    for the sources in ``actions`` (both by default, like every episode the recorder
    writes since 2026-09-11) - synthetic but well-formed rows unless ``action_rows``
    supplies FK-consistent ones (:func:`write_fk_episode`).
    """
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    names: list[str] = []
    for arm_id, has_rail in arms.items():
        names += arm_state_names(arm_id, has_rail)
    # Row k = every dim at k/100, so a frame's values are recognisable per index.
    rows = state_rows or [
        [(k + 1) / 100.0 + i / 1000.0 for i in range(len(names))] for k in range(frames)
    ]
    frames = len(rows)

    ds_root = root
    (ds_root / "episodes" / episode_id).mkdir(parents=True, exist_ok=True)
    directory = ds_root / "episodes" / episode_id
    features: dict = {"observation.state": {"dtype": "float32", "names": names}}
    columns = {"observation.state": pa.array(rows, type=pa.list_(pa.float32()))}
    for space in actions:
        column = "action" if space == "delta_ee" else "action.abs_ee"
        anames = [n for a, r in arms.items() for n in arm_action_names(a, r, space)]
        if action_rows is not None and space in action_rows:
            mat = np.asarray(action_rows[space], dtype=np.float32)
        else:
            mat = np.zeros((frames, len(anames)), dtype=np.float32)
            if space == "abs_ee":  # a well-formed absolute row: identity r6, gripper 1
                off = 0
                for arm_id, has_rail in arms.items():
                    mat[:, off : off + 3] = [0.3, 0.0, 0.4]
                    mat[:, off + 3 : off + 9] = [1, 0, 0, 0, 1, 0]
                    mat[:, off + 9] = 1.0
                    off += len(arm_action_names(arm_id, has_rail, space))
        assert mat.shape == (frames, len(anames))
        features[column] = _action_feature(arms, space, anames)
        columns[column] = pa.FixedSizeListArray.from_arrays(
            pa.array(mat.reshape(-1), type=pa.float32()), mat.shape[1]
        )
    if manifest_names:
        (ds_root / "manifest.json").write_text(
            json.dumps({"apollo_dataset_layout": 1, "fps": FPS, "features": features})
        )
    else:
        (ds_root / "manifest.json").write_text(json.dumps({"apollo_dataset_layout": 1, "fps": FPS}))
    (directory / "episode.json").write_text(
        json.dumps({"length": frames, "fps": FPS, "tasks": ["t"], "duration_s": frames / FPS})
    )
    pq.write_table(pa.table(columns), directory / "frames.parquet")
    return directory


def write_fk_episode(
    root: Path,
    episode_id: str,
    *,
    q_path: list[list[float]],  # per frame: 7 joints + rail (arm0 of ``single_rail``)
    gripper: float = 0.8,
    scene_id: str = "single_rail",
    arm_id: str = "arm0",
) -> Path:
    """An episode whose action columns AGREE with its states through the twin's FK - what
    the recorder writes for a commanded trajectory the arm followed exactly: row k of
    ``action`` is the TCP delta (arm_base frame, 10-frames §3.2) from the commanded pose at
    k to the one at k+1, row k of ``action.abs_ee`` is the commanded pose at k+1 (r6 rotation),
    the gripper is absolute, the rail a delta / an absolute position; the last row holds."""
    sim = pytest.importorskip("apollo_mavis_v2_sim")
    from apollo_mavis_v2_runtime.recorder.kinematics import RecorderKinematics

    kin = RecorderKinematics(sim.REGISTRY.build(scene_id))
    q_path = [np.asarray(q, dtype=np.float64) for q in q_path]
    poses = [kin.tcp_base(arm_id, q) for q in q_path]
    n = len(q_path)
    state_rows: list[list[float]] = []
    delta = np.zeros((n, len(arm_action_names(arm_id, True, "delta_ee"))), dtype=np.float32)
    absr = np.zeros((n, len(arm_action_names(arm_id, True, "abs_ee"))), dtype=np.float32)
    for k in range(n):
        q, pose = q_path[k], poses[k]
        state_rows.append(
            [*q[:7], gripper, q[7], *pose.position, *pose.orientation]  # arm_state_names order
        )
        nxt = min(n - 1, k + 1)
        a, b = poses[k], poses[nxt]
        dp = b.position - a.position
        dr = se3.quat_to_rotvec(se3.quat_mul(b.orientation, se3.quat_conj(a.orientation)))
        delta[k] = [*dp, *dr, gripper, q_path[nxt][7] - q[7]]
        absr[k] = [*b.position, *se3.quat_to_rot6d(b.orientation), gripper, q_path[nxt][7]]
    return write_episode(
        root,
        episode_id,
        arms={arm_id: True},
        state_rows=state_rows,
        action_rows={"delta_ee": delta, "abs_ee": absr},
    )


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
    # 2026-09-11: both action columns read back, sliced per arm by the manifest names.
    assert traj.sources == ["state", "delta_ee", "abs_ee"]
    assert traj.action_space == "delta_ee"
    assert traj.action_rows("delta_ee").shape == (4, 16)
    assert traj.action_rows("abs_ee").shape == (4, 22)
    assert traj.action_names("abs_ee")[11] == "view_ee.x"  # the second arm's block
    assert traj.action_block("abs_ee", "view", 2).shape == (11,)
    assert traj.action_block("abs_ee", "view", 2)[3:9].tolist() == [1, 0, 0, 0, 1, 0]


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
    # 2026-09-11: the sources this episode offers, so the dialog can enable its control
    assert body["sources"] == ["state", "delta_ee", "abs_ee"]
    assert body["action_space"] == "delta_ee"


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


# -- the action sources (2026-09-11) ----------------------------------------------------
# ``delta_ee`` / ``abs_ee`` replay the recorded action column through the executor path a
# policy drives (dagger/step.policy_step inside the plain ControlLoop's replay slot). The
# reader slices the columns by the manifest's names with the arm_action_names grammar; the
# ReplayActionSource lays them out in the SESSION's arm order, one row per recorded period.
def test_parse_action_names_groups_whole_blocks_per_arm():
    names = arm_action_names("grip", True, "abs_ee") + arm_action_names("view", False, "abs_ee")
    grip, view = parse_action_names(names, "abs_ee")
    assert (grip.arm_id, grip.start, grip.stop, grip.has_rail) == ("grip", 0, 11, True)
    assert (view.arm_id, view.start, view.stop, view.has_rail) == ("view", 11, 21, False)
    (only,) = parse_action_names(arm_action_names("arm0", True, "delta_ee"), "delta_ee")
    assert only.width == 8


def test_parse_action_names_refuses_a_partial_or_foreign_block():
    names = arm_action_names("grip", True, "delta_ee")
    with pytest.raises(PlaybackError, match="not the delta_ee layout"):
        parse_action_names(names[:-2], "delta_ee")  # gripper + rail missing
    with pytest.raises(PlaybackError, match="not a abs_ee dim"):
        parse_action_names(names, "abs_ee")  # delta names read as abs
    with pytest.raises(PlaybackError, match="two separate blocks"):
        parse_action_names(
            arm_action_names("a", False, "delta_ee")
            + arm_action_names("b", False, "delta_ee")
            + arm_action_names("a", False, "delta_ee"),
            "delta_ee",
        )


def test_an_episode_without_the_abs_column_offers_state_and_delta_only(tmp_path):
    """A pre-backfill episode: ``abs_ee`` is not offered, and asking for it names the
    backfill tool rather than failing the whole load (the state replay never depends on
    an action column)."""
    write_episode(
        tmp_path, "20260911T000000.000Z-0b0f11", arms={"grip": True}, actions=("delta_ee",)
    )
    traj = load_trajectory("bc/x", tmp_path, "20260911T000000.000Z-0b0f11")
    assert traj.sources == ["state", "delta_ee"]
    with pytest.raises(PlaybackError, match="backfill_abs_ee"):
        traj.action_column("abs_ee")
    write_episode(tmp_path, "20260911T000000.000Z-0ac750", arms={"grip": True}, actions=())
    traj = load_trajectory("bc/x", tmp_path, "20260911T000000.000Z-0ac750")
    assert traj.sources == ["state"] and traj.action_space is None


def test_a_malformed_action_column_drops_that_source_only(tmp_path):
    """The manifest names 8 delta dims but the column is wider: ``delta_ee`` is withheld
    with the reason, ``state`` and ``abs_ee`` still read."""
    directory = write_episode(tmp_path, "20260911T000000.000Z-bada57", arms={"grip": True})
    meta = json.loads((tmp_path / "manifest.json").read_text())
    meta["features"]["action"]["names"] = arm_action_names("grip", False, "delta_ee")  # 7 != 8
    (tmp_path / "manifest.json").write_text(json.dumps(meta))
    traj = load_trajectory("bc/x", directory.parent.parent, "20260911T000000.000Z-bada57")
    assert traj.sources == ["state", "abs_ee"]
    with pytest.raises(PlaybackError, match="manifest names 7 dims"):
        traj.action_rows("delta_ee")


class _Clock:
    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _source(tmp_path, source, arms_meta, *, rail_hold=None, frames=4):
    from apollo_mavis_v2_runtime.dagger.replay_source import ReplayActionSource

    write_episode(tmp_path, "20260911T000000.000Z-50e5ce", arms={"grip": True}, frames=frames)
    traj = load_trajectory("bc/x", tmp_path, "20260911T000000.000Z-50e5ce")
    clock = _Clock()
    return ReplayActionSource(traj, source, arms_meta, clock=clock, rail_hold=rail_hold), clock


def test_replay_source_lays_rows_out_in_the_session_order_with_nan_for_undriven_arms(tmp_path):
    """The episode names ``grip`` only; the session drives ``view`` then ``grip``. The row
    is in SESSION order, ``view``'s block is NaN (= hold, 14-dora §6.1), and only ``grip``
    is driven. ``spec`` is the layout a PolicySource announces."""
    src, clock = _source(tmp_path, "abs_ee", [("view", True), ("grip", True)])
    assert src.driven_arms() == frozenset({"grip"})
    assert src.spec.action_space == "abs_ee" and src.spec.action_frame == "arm_base:grip"
    assert src.spec.action_names == arm_action_names("view", True, "abs_ee") + arm_action_names(
        "grip", True, "abs_ee"
    )
    assert src.period == pytest.approx(1 / FPS)
    assert src.latest() == (None, 0.0)  # not started: nothing is current
    src.start()
    out, t_row = src.latest()
    assert t_row == clock.t and out.t_mono == clock.t and out.chunk_remaining == 3
    assert np.isnan(out.actions[:11]).all()
    assert out.actions[11:14].tolist() == pytest.approx([0.3, 0.0, 0.4])
    assert out.actions[14:20].tolist() == [1, 0, 0, 0, 1, 0]
    assert src.version_label() == "replay/20260911T000000.000Z-50e5ce"
    assert src.current_version() == 0 and not src.paused


def test_replay_source_serves_one_row_per_period_then_goes_stale(tmp_path):
    """Row k is current from ``t0 + k / fps``; the clock never skips a row ahead of time and
    the last row stays current for one period, after which the source reports 0.0 (the
    executor holds) and ``finished``."""
    src, clock = _source(tmp_path, "delta_ee", [("grip", True)], frames=4)
    src.start()
    period = 1 / FPS
    for k in range(4):
        clock.t = 100.0 + k * period + 0.3 * period
        out, t_row = src.latest()
        assert out.chunk_remaining == 3 - k and src.frame_index == k
        assert t_row == pytest.approx(100.0 + k * period)
        assert src.staleness_scale(clock.t) == 1.0 and not src.finished
    clock.t = 100.0 + 4 * period - 1e-4  # the last row, still within its period
    assert src.staleness_scale(clock.t) == 1.0
    out, _ = src.latest()
    assert out.chunk_remaining == 0
    clock.t = 100.0 + 4 * period + 1e-4  # the last row has been due for a period
    assert src.staleness_scale(clock.t, "grip") == 0.0 and src.finished
    src.stop()
    assert src.latest() == (None, 0.0)


def test_replay_source_fills_a_missing_rail_column(tmp_path):
    """An episode recorded without a track on a session arm that has one: the delta rows
    get ``rail.dpos 0`` and the abs rows ``rail.pos = rail_hold`` (the carriage stays put,
    the rule the state replay follows); abs without a hold value is refused."""
    from apollo_mavis_v2_runtime.dagger.replay_source import ReplayActionSource

    write_episode(tmp_path, "20260911T000000.000Z-0a1100", arms={"grip": False}, frames=2)
    traj = load_trajectory("bc/x", tmp_path, "20260911T000000.000Z-0a1100")
    delta = ReplayActionSource(traj, "delta_ee", [("grip", True)], clock=_Clock())
    delta.start()
    assert delta.latest()[0].actions[7] == 0.0
    absr = ReplayActionSource(
        traj, "abs_ee", [("grip", True)], clock=_Clock(), rail_hold={"grip": 0.42}
    )
    absr.start()
    assert absr.latest()[0].actions[10] == pytest.approx(0.42)
    with pytest.raises(PlaybackError, match="no rail column"):
        ReplayActionSource(traj, "abs_ee", [("grip", True)], clock=_Clock())
    with pytest.raises(PlaybackError, match="names none of this session's arms"):
        ReplayActionSource(traj, "delta_ee", [("view", True)], clock=_Clock())
    with pytest.raises(PlaybackError, match="not an action replay source"):
        ReplayActionSource(traj, "state", [("grip", True)], clock=_Clock())


def test_action_replay_is_refused_on_hardware_and_in_policy_sessions(tmp_path):
    """The per-tick gate is the only check an action replay gets, and its fidelity is
    what it measures - so the real arms do not get it until the operator says so; a
    dagger / inference loop drives its own policy and has no replay branch."""
    import types

    cfg = make_runtime_config(tmp_path)
    rt = Runtime(cfg)
    ds_root = cfg.datasets_root / "bc" / "x"
    write_episode(
        ds_root, "20260911T000000.000Z-0a0a0a", arms={"arm0": True}, actions=("delta_ee",)
    )
    traj = load_trajectory("bc/x", ds_root, "20260911T000000.000Z-0a0a0a")
    loop = types.SimpleNamespace(ik=object(), kin=object())

    def session(kind="sim", mode="teleop"):
        return types.SimpleNamespace(spec=types.SimpleNamespace(kind=kind, mode=mode), loop=loop)

    refusal = rt.manager._action_replay_refusal  # noqa: SLF001
    assert refusal(session("hardware"), traj, "delta_ee") == "action replay is admitted in sim only"
    assert "teleop / collect sessions only" in refusal(session(mode="inference"), traj, "abs_ee")
    assert "backfill_abs_ee" in refusal(session(), traj, "abs_ee")  # the column is missing
    assert "unknown playback source" in refusal(session(), traj, "joint")
    assert refusal(session(), traj, "delta_ee") == ""
    assert "no IK" in refusal(
        types.SimpleNamespace(spec=session().spec, loop=types.SimpleNamespace(ik=None, kin=None)),
        traj,
        "delta_ee",
    )


def _fk_path(frames: int, *, dq: float = 0.01, drail: float = 0.002) -> list[list[float]]:
    """A gentle, well-conditioned joint path for ``single_rail``'s arm0: elbow bent, joints
    2 and 4 sweeping ``dq`` per frame, the carriage ``drail`` per frame."""
    q0 = np.array([0.0, -0.5, 0.0, 0.8, 0.0, 1.3, 0.0, 0.3])
    path = []
    for k in range(frames):
        q = q0.copy()
        q[1] += dq * k
        q[3] += dq * k
        q[7] += drail * k
        path.append(q.tolist())
    return path


FK_EPISODE = "20260911T000000.000Z-f00001"


def _fk_client_episode(client, frames=12, **kw) -> tuple[str, str]:
    root = client.app.state.runtime.cfg.datasets_root / "bc" / "fk"
    write_fk_episode(root, FK_EPISODE, q_path=_fk_path(frames, **kw))
    return "bc/fk", FK_EPISODE


@pytest.mark.parametrize("source", ["delta_ee", "abs_ee"])
def test_play_replays_the_action_column_through_the_executor_and_lands_on_the_last_frame(
    client, source
):
    """End to end in sim: place the arm at frame 0, replay the ACTION column through the
    policy's executor path, and end within tolerance of the last recorded state - the
    residual the outcome reports IS the executor-fidelity metric."""
    repo_id, episode_id = _fk_client_episode(client)
    assert client.post("/api/session", json=SPEC).status_code == 200
    info = client.get(f"/api/datasets/bc/fk/episodes/{episode_id}/playback").json()
    assert info["playable"] is True and info["sources"] == ["state", "delta_ee", "abs_ee"]
    body = {"repo_id": repo_id, "episode_id": episode_id}
    first = client.post("/api/session/playback", json={**body, "action": "goto_initial"}).json()
    assert first["ok"] is True, first
    played = client.post(
        "/api/session/playback", json={**body, "action": "play", "source": source}
    ).json()
    assert played["ok"] is True, played
    assert played["status"] == "done"
    assert f"replayed 12 {source} rows" in played["detail"]
    assert "TCP" in played["detail"] and "mrad" in played["detail"]  # the residual report
    traj = load_trajectory(
        repo_id, client.app.state.runtime.cfg.datasets_root / "bc" / "fk", episode_id
    )
    last = traj.state_at(traj.frames - 1)["arm0"]
    with client.websocket_connect("/ws/telemetry") as ws:
        arms = ws.receive_json()["arms"]
    row = next(a for a in arms if a["arm_id"] == "arm0")
    assert row["q"] == pytest.approx(last.q, abs=PLAYBACK_JOINT_TOL_RAD)
    assert row["rail_pos_m"] == pytest.approx(last.rail_pos_m, abs=3e-3)
    # the telemetry's plan status lingers "done" like a finished plan's
    telemetry = client.get("/api/session").json()
    assert telemetry["state"] == "running"


#: How far the IK-driven replay may end from the recorded joints. The executor tracks the
#: TCP, and a 7-DOF arm has a null space, so this is looser than the state replay's 3 mrad.
PLAYBACK_JOINT_TOL_RAD = 0.02


def test_play_refuses_an_action_source_the_episode_does_not_carry(client, tmp_path):
    root = client.app.state.runtime.cfg.datasets_root / "bc" / "old"
    write_episode(root, "20260911T000000.000Z-01d001", arms={"arm0": True}, actions=("delta_ee",))
    assert client.post("/api/session", json=SPEC).status_code == 200
    info = client.get("/api/datasets/bc/old/episodes/20260911T000000.000Z-01d001/playback").json()
    assert info["sources"] == ["state", "delta_ee"]
    r = client.post(
        "/api/session/playback",
        json={
            "repo_id": "bc/old",
            "episode_id": "20260911T000000.000Z-01d001",
            "action": "play",
            "source": "abs_ee",
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is False and r.json()["status"] == "refused"
    assert "backfill_abs_ee" in r.json()["detail"]
    # an unknown source is a 422 from the model, not a refusal
    r = client.post(
        "/api/session/playback",
        json={
            "repo_id": "bc/old",
            "episode_id": "20260911T000000.000Z-01d001",
            "action": "play",
            "source": "joint",
        },
    )
    assert r.status_code == 422


def test_stop_cancels_a_running_action_replay(client):
    """The Welcome page has no control socket: ``stop`` is its only cancel. A 4 s replay is
    stopped from another thread; the synchronous ``play`` answers ``cancelled`` with the
    operator's reason and the loop is free again."""
    repo_id, episode_id = _fk_client_episode(client, frames=100, dq=0.001, drail=0.0002)
    assert client.post("/api/session", json=SPEC).status_code == 200
    body = {"repo_id": repo_id, "episode_id": episode_id}
    first = client.post("/api/session/playback", json={**body, "action": "goto_initial"}).json()
    assert first["ok"] is True, first
    result: dict = {}

    def play():
        result["r"] = client.post(
            "/api/session/playback", json={**body, "action": "play", "source": "delta_ee"}
        ).json()

    worker = threading.Thread(target=play, daemon=True)
    worker.start()
    loop = client.app.state.runtime.manager.session.loop
    t0 = time.monotonic()
    while not loop.motion_active and time.monotonic() - t0 < 5.0:
        time.sleep(0.02)
    assert loop.motion_active, "the replay never started"
    time.sleep(0.3)
    stopped = client.post("/api/session/playback", json={**body, "action": "stop"}).json()
    assert stopped["ok"] is True and "the arms hold" in stopped["detail"]
    worker.join(timeout=10.0)
    assert not worker.is_alive()
    played = result["r"]
    assert played["ok"] is False and played["status"] == "cancelled", played
    assert played["detail"] == "playback stopped by the operator"
    assert not loop.motion_active
    again = client.post("/api/session/playback", json={**body, "action": "stop"}).json()
    assert "nothing is playing" in again["detail"]
