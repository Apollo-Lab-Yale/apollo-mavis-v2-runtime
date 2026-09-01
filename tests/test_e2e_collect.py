"""Sim-backed collect-mode e2e over a REAL uvicorn server (phase-07 验收):
collect session -> WS episode_new -> teleop drive -> episode_save -> record
again -> episode_discard -> teardown -> reload with LeRobotDataset and assert
schema (10-frames §6-§9), sidecars, and Ack semantics."""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
from conftest import LiveServer, make_runtime_config
from test_e2e_teleop import Ctl, Tele

TASK = "pick cube e2e"
REPO_ID = "apollo/xarm7_pick_cube_e2e_1arm_dee-base"
SPEC = {
    "mode": "collect",
    "kind": "sim",
    "arms": ["arm0"],
    "frames": {"arm0": "arm_base:arm0"},
    "sim_scene": "guardrail_env",  # 1 arm (rail) + 1 camera (cam_front)
    "task": TASK,
}


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    srv = LiveServer(make_runtime_config(tmp_path_factory.mktemp("rt"), scene="guardrail_env"))
    yield srv
    srv.stop()


@pytest.fixture(scope="module")
def api(server):
    with httpx.Client(base_url=server.http, timeout=120.0) as client:
        yield client


@pytest.fixture(scope="module")
def dataset_root(server, api) -> Path:
    """Runs the WHOLE collect session once; downstream tests assert on disk."""
    r = api.post("/api/session", json=SPEC)
    assert r.status_code == 200, r.text
    assert r.json()["mode"] == "collect"
    for _ in range(200):
        if api.get("/api/session").json()["state"] == "running":
            break
        time.sleep(0.05)
    ctl, tele = Ctl(server), Tele(server)
    loop = server.runtime.manager.session.loop
    try:
        # -- Ack semantics before any episode ---------------------------------
        assert ctl.action("episode_save")["ok"] is False  # idle: nothing to save
        assert ctl.action("episode_discard")["ok"] is False

        # -- episode 0: record ~5 s of teleop, then save -----------------------
        ack = ctl.action("episode_new")
        assert ack["ok"] is True, ack
        bad = ctl.action("episode_new")  # invalid transition while recording
        assert bad["ok"] is False and bad["detail"] == "recording"
        jt = ctl.action(
            "joint_target",
            {"arm_id": "arm0", "positions": [0.0] * 8, "mode": "jog"},
        )
        assert jt["ok"] is False and jt["detail"] == "recording"  # never source 2/4

        ticks0, t0 = loop.tick_count, time.monotonic()
        ctl.hold(["KeyW"], 2.5)
        mid = tele.latest()
        ctl.hold(["KeyS"], 2.5)
        rate = (loop.tick_count - ticks0) / (time.monotonic() - t0)
        assert 97.0 <= rate <= 103.0, f"control loop {rate:.1f} Hz during recording"
        assert mid["episode"]["state"] == "recording"
        assert mid["episode"]["frames"] > 20
        assert mid["episode"]["duration_s"] == pytest.approx(
            mid["episode"]["frames"] / 25.0
        )

        ack = ctl.action("episode_save")
        assert ack["ok"] is True and ack["detail"] == "saving"
        t_save = time.monotonic()
        while tele.latest()["episode"]["state"] != "idle":
            assert time.monotonic() - t_save < 5.0
        save_s = time.monotonic() - t_save
        assert save_s < 1.0, f"save_episode took {save_s:.2f}s (streaming encoding)"

        # -- episode 1: record briefly, then discard ---------------------------
        assert ctl.action("episode_new")["ok"] is True
        ctl.hold(["KeyW"], 1.5)
        assert tele.latest()["episode"]["state"] == "recording"
        assert ctl.action("episode_discard")["ok"] is True
        t_disc = time.monotonic()
        while tele.latest()["episode"]["state"] != "idle":
            assert time.monotonic() - t_disc < 5.0
    finally:
        ctl.close()
        tele.close()
        api.delete("/api/session")  # TEARDOWN -> recorder finalize
    return server.runtime.cfg.datasets_root / REPO_ID


def test_dataset_reloads_with_lerobot(dataset_root):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(REPO_ID, root=dataset_root)
    assert ds.meta.info.codebase_version == "v3.0"
    assert ds.num_episodes == 1  # only the saved episode; discard left no trace
    assert ds.fps == 25
    assert 25 * 3 <= ds.num_frames <= 25 * 6  # ~5 s at 25 fps
    assert ds.meta.info.robot_type == "xarm7_1arm_rail_mujoco"
    # read one frame back through the reader (video decode included)
    row = ds[0]
    assert row["task"] == TASK
    img = np.asarray(row["observation.images.cam_front"])
    assert img.size == 480 * 640 * 3


def test_feature_schema_matches_10frames(dataset_root):
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    feats = info["features"]
    assert feats["action"]["names"] == [
        "arm0_ee.dx", "arm0_ee.dy", "arm0_ee.dz",
        "arm0_ee.drx", "arm0_ee.dry", "arm0_ee.drz",
        "arm0_gripper.pos", "arm0_rail.dpos",
    ]
    assert feats["action"]["info"] == {
        "apollo_schema": 1,
        "action_space": "delta_ee",
        "frames": {"arm0": "arm_base:arm0"},
        "rail": {"axis": "y", "travel_m": 0.65, "arms": ["arm0"]},
    }
    assert feats["observation.state"]["info"]["frames"] == {"arm0": "arm_base:arm0"}
    assert len(feats["observation.state"]["names"]) == 16
    assert all(n.startswith("arm0_") for n in feats["observation.state"]["names"])
    assert feats["observation.images.cam_front"]["dtype"] == "video"
    assert feats["intervention"]["dtype"] == "bool"
    assert feats["action_source"]["info"]["labels"] == {
        "0": "policy", "1": "teleop", "2": "joint_jog", "3": "takeover", "4": "planner",
    }
    assert feats["wallclock_ns"]["dtype"] == "int64"


def _col(table, name) -> np.ndarray:
    """Column -> (n,) or (n, d) array; robust to scalar vs list storage."""
    values = table.column(name).to_numpy(zero_copy_only=False)
    return np.stack([np.atleast_1d(np.asarray(v)) for v in values]).squeeze()


def test_parquet_columns(dataset_root):
    import pyarrow.parquet as pq

    files = sorted(dataset_root.glob("data/**/*.parquet"))
    assert files
    table = pq.read_table(files[0])
    n = table.num_rows
    intervention = _col(table, "intervention")
    assert intervention.shape == (n,) and not intervention.any()  # all False
    src = _col(table, "action_source")
    assert (src == 1).all()  # all teleop; 2 (joint_jog) / 4 (planner) never appear
    wall = _col(table, "wallclock_ns")
    assert (np.diff(wall.astype(np.int64)) > 0).all()  # strictly increasing
    action = _col(table, "action")
    assert action.shape == (n, 8)
    assert np.abs(action[:, 0]).max() > 1e-4  # the W/S drive left real ee.dx deltas
    state = _col(table, "observation.state")
    assert state.shape == (n, 16)
    quat_norm = np.linalg.norm(state[:, 12:16], axis=1)
    assert np.allclose(quat_norm, 1.0, atol=1e-3)  # ee quat dims sane


def test_video_decodes_with_matching_frame_count(dataset_root):
    import av

    videos = sorted(dataset_root.glob("videos/observation.images.cam_front/**/*.mp4"))
    assert videos
    import pyarrow.parquet as pq

    n_rows = sum(pq.read_table(f).num_rows for f in dataset_root.glob("data/**/*.parquet"))
    decoded = 0
    for path in videos:
        with av.open(str(path)) as container:
            decoded += sum(1 for _ in container.decode(video=0))
    assert decoded == n_rows


def test_meta_apollo_sidecars(dataset_root):
    base = dataset_root / "meta" / "apollo"
    sessions = list(base.glob("session_*.json"))
    assert len(sessions) == 1
    session = json.loads(sessions[0].read_text())
    assert session["mode"] == "collect"
    assert session["spec"]["task"] == TASK
    assert session["workcell"]["kind"] == "sim" and session["workcell"]["rail"] == {
        "arm0": True
    }
    assert session["software"]["lerobot"] is not None
    assert "geom_inflation_m" in session["safety"]

    episodes = sorted(base.glob("episodes/*.json"))
    assert [p.name for p in episodes] == ["episode_000000.json"]  # discard: no sidecar
    ep = json.loads(episodes[0].read_text())
    assert ep["frames"] == {"arm0": "arm_base:arm0"}
    assert ep["start_from"] == "keep_current"
    # live cameras occasionally miss the 2/fps age budget; a handful of
    # counted drops is legitimate — bad frames are dropped, never written.
    assert 0 <= ep["frames_dropped"] < 25
    assert ep["arm_bases"]["arm0"]["has_rail"] is True
    ext = ep["extrinsics"]["cam_front"]
    assert len(ext["T_W_C"]["orientation_wxyz"]) == 4
    assert ext["extrinsics_frame"] == "world"

    scenes = list(base.glob("scenes/*.xml"))
    assert len(scenes) == 1
    assert scenes[0].stem == ep["scene_xml_sha256"][:16]
    import mujoco

    model = mujoco.MjModel.from_xml_string(scenes[0].read_text())
    assert model.nq > 0  # snapshot recompiles

    state = json.loads((dataset_root / "recorder_state.json").read_text())
    assert state == {"repo_id": REPO_ID, "episodes_saved": 1, "finalized": True}
