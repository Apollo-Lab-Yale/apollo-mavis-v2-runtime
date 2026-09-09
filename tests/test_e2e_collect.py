"""Sim-backed collect-mode e2e over a REAL uvicorn server (phase-13 验收; 04-runtime
§16 3(b); 10-frames §11): collect session into a NAMED dataset -> WS episode_new ->
KEYBOARD teleop (``KeysMsg`` KeyW / KeyS, the clutch key inert without a tracker, F
closes the gripper, Tab cycles) -> episode_save -> record again -> episode_discard ->
teardown -> exactly one ``episodes/<id>/`` directory; then the LeRobot v3 export over
REST (202 + telemetry progress) re-opened with ``LeRobotDataset``; a resumed session
appends a second directory; ``DELETE`` removes one directory without touching the
other; the resume refusal matrix (fps / robot_type / legacy / exists / unknown)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
from conftest import LiveServer, make_runtime_config
from test_e2e_teleop import PulsingCtl, Tele

from apollo_mavis_v2_runtime.config import MicrophoneConfig

TASK = "pick cube e2e"
REPO_ID = "apollo/pick_cube"
SPEC = {
    "mode": "collect",
    "kind": "sim",
    "arms": ["arm0"],
    "frames": {"arm0": "arm_base:arm0"},
    "sim_scene": "guardrail_env",  # 1 arm (rail) + 1 camera (cam_front)
    "task": TASK,
    "dataset": "pick_cube",  # bare name -> apollo/pick_cube
    "dataset_resume": False,
    # the return-to-start motion has its own e2e (test_return_to_start.py); this one
    # records without a profile, so the default-on flag must be unticked (else 409)
    "return_to_start": False,
}


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    cfg = make_runtime_config(tmp_path_factory.mktemp("rt"), scene="guardrail_env")
    # a FAKE microphone: the only way audio.wav is exercised in sim (04-runtime §10.5)
    cfg = cfg.model_copy(update={"microphone": MicrophoneConfig(enabled=True, backend="fake")})
    srv = LiveServer(cfg)
    yield srv
    srv.stop()


@pytest.fixture(scope="module")
def api(server):
    with httpx.Client(base_url=server.http, timeout=120.0) as client:
        yield client


def wait_state(tele: Tele, state: str, timeout: float = 5.0) -> dict:
    t0 = time.monotonic()
    while True:
        msg = tele.latest()
        if msg["episode"]["state"] == state:
            return msg
        assert time.monotonic() - t0 < timeout, (state, msg["episode"])


def wait_running(api) -> None:
    for _ in range(200):
        if api.get("/api/session").json()["state"] == "running":
            return
        time.sleep(0.05)
    raise AssertionError("session never reached running")


@pytest.fixture(scope="module")
def dataset_root(server, api) -> Path:
    """Runs the WHOLE first collect session once; downstream tests assert on disk."""
    r = api.post("/api/session", json=SPEC)
    assert r.status_code == 200, r.text
    assert r.json()["mode"] == "collect"
    wait_running(api)
    ctl, tele = PulsingCtl(server), Tele(server)
    loop = server.runtime.manager.session.loop
    root = server.runtime.cfg.datasets_root / REPO_ID
    try:
        # -- the dataset exists (manifest) from bring-up on and is in use ------------
        rows = api.get("/api/datasets").json()
        assert [d["repo_id"] for d in rows] == [REPO_ID]
        assert rows[0]["in_use"] is True and rows[0]["total_episodes"] == 0
        assert rows[0]["layout"] == "episode_dirs" and rows[0]["export"]["state"] == "none"
        assert (root / "manifest.json").exists() and not (root / "meta" / "info.json").exists()
        assert api.post(f"/api/datasets/{REPO_ID}/export", json={}).status_code == 409  # in use
        assert api.delete(f"/api/datasets/{REPO_ID}").status_code == 409
        assert api.post("/api/session", json=SPEC).status_code == 409  # a session exists

        # -- Ack semantics before any episode ---------------------------------
        assert ctl.action("episode_save")["ok"] is False  # idle: nothing to save
        assert ctl.action("episode_discard")["ok"] is False

        # -- keyboard is a full teleop interface (overview §5) --------------------
        x0 = tele.ee_x()
        ctl.hold(["KeyC"], 0.5)  # the clutch key is inert without a tracker
        assert abs(tele.ee_x() - x0) < 2e-3
        ctl.hold(["KeyW"], 1.0)  # ... and W still drives afterwards
        assert tele.ee_x() - x0 > 0.05
        g0 = tele.latest()["arms"][0]["gripper_open_frac"]
        ctl.hold(["KeyF"], 0.5)  # gripper close
        assert tele.latest()["arms"][0]["gripper_open_frac"] < g0 - 0.2
        ctl.keys([])
        assert ctl.action("switch_arm")["ok"] is True  # Tab (one arm: stays arm0)
        assert tele.latest()["active_arm"] == "arm0"

        # -- episode 0: record ~5 s of KEYBOARD teleop, then save -----------------------
        ack = ctl.action("episode_new")
        assert ack["ok"] is True, ack
        bad = ctl.action("episode_new")  # invalid transition while recording
        assert bad["ok"] is False and bad["detail"] == "recording"
        jt = ctl.action(
            "joint_target",
            {"arm_id": "arm0", "positions": [0.0] * 8, "mode": "jog"},
        )
        assert jt["ok"] is False and jt["detail"] == "recording"  # never source 2/4
        eps = api.get(f"/api/datasets/{REPO_ID}/episodes").json()
        assert len(eps) == 1 and eps[0]["open"] is True  # the open episode is listed
        open_id = eps[0]["episode_id"]
        r = api.delete(f"/api/datasets/{REPO_ID}/episodes/{open_id}")
        assert r.status_code == 409 and "being recorded" in r.json()["detail"]

        time.sleep(0.6)  # let the recorder open the encoder (prepare at N) before any key
        # -- idle-frame filter acceptance: drive 2 s, release 3 s, drive 2 s ----------------
        # -x then +x back: +x saturates guardrail_env's reach at x = 0.575 m after ~1.4 s
        # (measured 2026-09-08), and a saturated command IS idle to the filter.
        ticks0, t0 = loop.tick_count, time.monotonic()
        ctl.hold(["KeyS"], 2.0)
        time.sleep(3.0)  # hesitation: these frames must NOT be recorded
        mid = tele.latest()
        ctl.hold(["KeyW"], 2.0)
        rate = (loop.tick_count - ticks0) / (time.monotonic() - t0)
        # 95 Hz floor (was 97): the fake microphone + two encoder threads share the GIL
        assert 95.0 <= rate <= 103.0, f"control loop {rate:.1f} Hz during recording"
        print(f"REPORT collect e2e: control loop {rate:.1f} Hz during recording")
        assert mid["episode"]["state"] == "recording"
        assert mid["episode"]["frames"] > 20
        assert mid["episode"]["frames_skipped"] > 10  # the pause is being skipped live
        assert mid["episode"]["duration_s"] == pytest.approx(mid["episode"]["frames"] / 25.0)
        assert mid["episode"]["repo_id"] == REPO_ID and mid["episode"]["total_episodes"] == 0

        ack = ctl.action("episode_save")
        assert ack["ok"] is True and ack["detail"] == "saving"
        t_save = time.monotonic()
        idle = wait_state(tele, "idle")
        save_s = time.monotonic() - t_save
        assert save_s < 2.0, f"save took {save_s:.2f}s (one rename, streaming encoder)"
        print(f"REPORT collect e2e: save -> idle {save_s:.2f}s")
        assert idle["episode"]["total_episodes"] == 1 and idle["episode"]["index"] == 0
        assert abs(idle["episode"]["total_frames"] - mid["episode"]["frames"]) <= 200
        eps = api.get(f"/api/datasets/{REPO_ID}/episodes").json()
        assert [e["episode_id"] for e in eps] == [open_id] and eps[0]["open"] is False
        assert eps[0]["audio"] is True and eps[0]["export_ok"] is True

        # -- episode 1: gripper-only motion is kept by the filter, then discard -------------
        assert ctl.action("episode_new")["ok"] is True
        time.sleep(0.6)
        ctl.keys([])
        ctl.hold(["KeyF"], 1.0)  # the arm is still; only the gripper closes
        ctl.keys([])
        time.sleep(1.9)  # > the 1.6 s look-ahead: the gripper frames are decided
        st = tele.latest()["episode"]
        assert st["state"] == "recording" and st["frames"] >= 20, st
        assert st["frames_skipped"] <= 3, st  # gripper motion exempts its whole context
        assert ctl.action("episode_discard")["ok"] is True
        wait_state(tele, "idle")
        assert api.get(f"/api/datasets/{REPO_ID}").json()["total_episodes"] == 1
    finally:
        ctl.close()
        tele.close()
        api.delete("/api/session")  # TEARDOWN -> recorder finalize
    return root


def table_rows(d: Path) -> int:
    import pyarrow.parquet as pq

    return pq.read_table(d / "frames.parquet").num_rows


def test_episode_directory_layout(dataset_root):
    root = dataset_root
    dirs = sorted(p for p in (root / "episodes").iterdir())
    assert len(dirs) == 1 and not dirs[0].name.startswith(".")  # discard: nothing; no .tmp-*
    d = dirs[0]
    assert {p.name for p in d.iterdir()} == {"episode.json", "frames.parquet", "video", "audio.wav"}
    ep = json.loads((d / "episode.json").read_text())
    assert ep["episode_id"] == d.name and ep["episode_index"] is None
    import pyarrow.parquet as pq

    table = pq.read_table(d / "frames.parquet")
    assert table.num_rows == ep["length"] > 25 * 3, (ep["length"], ep["filter"])
    assert ep["duration_s"] == pytest.approx(ep["length"] / 25) and ep["fps"] == 25
    import av

    with av.open(str(d / "video" / "cam_front.mp4")) as c:
        assert sum(1 for _ in c.decode(video=0)) == ep["length"]
    assert ep["video"]["cam_front"]["frames"] == ep["length"]
    assert ep["video"]["cam_front"]["codec"] in ("h264", "hevc", "av1")
    assert set(ep["stats"]) >= {"action", "observation.state", "observation.images.cam_front"}
    assert ep["export_ok"] is True and ep["tasks"] == [TASK]
    assert ep["frames"] == {"arm0": "arm_base:arm0"} and ep["start_from"] == "keep_current"
    # idle-frame filter (2026-09-07 addendum): W 2 s / idle 3 s / W 2 s
    filt = ep["filter"]
    assert filt["enabled"] is True and filt["params"]["pos_eps_m"] == 0.001
    assert filt["params"]["gripper_context_s"] == 1.6
    assert 6 * 25 <= filt["frames_seen"] <= 8.5 * 25, filt  # ~7 s of captures
    # two gaps and nothing else: the 0.6 s the test waits after episode_new (frame 0 is
    # always kept, then ~15 idle captures) and the 3 s hesitation between the two drives
    assert len(filt["gaps"]) == 2, filt["gaps"]
    (first_kept, prepare_idle), (kept_before_gap, skipped) = filt["gaps"]
    assert first_kept == 1 and 5 <= prepare_idle <= 25, filt["gaps"]
    assert 3 * 25 - 10 <= skipped <= 3 * 25 + 10, filt  # the hesitation, minus a few
    assert prepare_idle + skipped == filt["frames_skipped"]
    assert 25 <= kept_before_gap <= 3 * 25 + 5  # the gap sits right after the first 2 s drive
    assert filt["frames_seen"] - filt["frames_skipped"] - 1 == ep["length"]
    assert filt["frames_seen"] - filt["frames_skipped"] - 1 == table_rows(d)
    assert ep["return_profile_id"] is None  # return_to_start unticked
    assert 0 <= ep["frames_dropped"] < 25
    ext = ep["extrinsics"]["cam_front"]
    assert len(ext["T_W_C"]["orientation_wxyz"]) == 4 and ext["extrinsics_frame"] == "world"
    # audio (fake microphone): a WAV whose duration matches the episode
    audio = ep["audio"]
    assert audio["path"] == "audio.wav" and audio["sample_rate"] == 48000
    # the WAV is the continuous WALL-CLOCK recording (10-frames §11.4: audio maps onto the
    # frames' wallclock_ns, gaps included), so it outlasts length / fps by the skipped frames
    assert audio["duration_s"] >= ep["duration_s"]
    assert audio["duration_s"] == pytest.approx(filt["frames_seen"] / 25, abs=0.7)
    assert "wallclock_ns" in table.column_names  # the alignment column survives the filter
    import wave

    with wave.open(str(d / "audio.wav"), "rb") as w:
        assert w.getnframes() == audio["samples"] and w.getnchannels() == 1
    # dataset-level files
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["episodes"] == 1 and manifest["frames"] == ep["length"]
    assert manifest["repo_id"] == REPO_ID and manifest["fps"] == 25
    assert manifest["robot_type"] == "xarm7_1arm_rail_mujoco"
    sessions = list((root / "sessions").glob("session_*.json"))
    assert len(sessions) == 1
    session = json.loads(sessions[0].read_text())
    assert session["mode"] == "collect" and session["spec"]["dataset"] == "pick_cube"
    assert session["spec"]["return_to_start"] is False
    assert session["workcell"]["kind"] == "sim" and session["workcell"]["rail"] == {"arm0": True}
    scenes = list((root / "scenes").glob("*.xml"))
    assert len(scenes) == 1 and scenes[0].stem == ep["scene_xml_sha256"][:16]
    import mujoco

    assert mujoco.MjModel.from_xml_string(scenes[0].read_text()).nq > 0
    assert not (root / "meta").exists() and not (root / "recorder_state.json").exists()


def test_parquet_columns(dataset_root):
    import pyarrow.parquet as pq

    d = next(p for p in (dataset_root / "episodes").iterdir())
    table = pq.read_table(d / "frames.parquet")
    n = table.num_rows
    assert table.column("frame_index").to_pylist() == list(range(n))
    assert np.allclose(table.column("timestamp").to_numpy(), np.arange(n) / 25.0)
    assert table.column("task").to_pylist() == [TASK] * n
    assert not any(table.column("intervention").to_pylist())
    assert set(table.column("action_source").to_pylist()) == {1}  # teleop only
    wall = np.asarray(table.column("wallclock_ns").to_pylist(), dtype=np.int64)
    assert (np.diff(wall) > 0).all()
    action = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
    assert action.shape == (n, 8) and np.abs(action[:, 0]).max() > 1e-4  # W/S left ee.dx
    state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
    assert state.shape == (n, 16)
    assert np.allclose(np.linalg.norm(state[:, 12:16], axis=1), 1.0, atol=1e-3)


def test_export_over_rest_and_lerobot_reload(server, api, dataset_root):
    root = dataset_root
    r = api.post(f"/api/datasets/{REPO_ID}/export", json={"format": "lerobot_v3"})
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["repo_id"] == REPO_ID and body["format"] == "lerobot_v3" and body["started_at"]
    tele = Tele(server)
    phases: dict[str, float] = {}
    try:
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            msg = tele.latest()
            block = msg.get("datasets")
            if block and block.get("export"):
                ex = block["export"]
                phases.setdefault(ex["phase"], time.monotonic())
                if ex["phase"] in ("done", "failed"):
                    break
            time.sleep(0.02)
    finally:
        tele.close()
    assert ex["phase"] == "done", ex
    assert ex["repo_id"] == REPO_ID and "1 episodes" in ex["detail"]
    assert server.runtime.manager.dataset_store.wait_export(60.0)
    # videos + data + meta < 2 s (remux, no decode) from the job's own phase clock;
    # the validating phase (lerobot import) is excluded
    started = server.runtime.manager.dataset_store._export_progress.phase_started
    assert started["validating"] - started["videos"] < 2.0, started
    print(f"REPORT export: videos+data+meta {started['validating'] - started['videos']:.2f}s")
    out = root / "exports" / "lerobot_v3"
    assert (out / "meta" / "apollo" / "episode_map.json").exists()
    info = api.get(f"/api/datasets/{REPO_ID}").json()
    assert info["export"]["state"] == "fresh" and info["export"]["episodes"] == 1
    assert info["export"]["path"] == "exports/lerobot_v3"
    assert api.post(f"/api/datasets/{REPO_ID}/export", json={}).status_code == 202  # again is fine
    assert server.runtime.manager.dataset_store.wait_export(60.0)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(REPO_ID, root=out)
    assert ds.meta.info.codebase_version == "v3.0" and ds.num_episodes == 1 and ds.fps == 25
    assert ds.meta.info.robot_type == "xarm7_1arm_rail_mujoco"
    first, last = ds[0], ds[len(ds) - 1]
    assert first["task"] == TASK and int(last["frame_index"]) == len(ds) - 1
    assert np.asarray(first["observation.images.cam_front"]).size == 480 * 640 * 3
    feats = ds.meta.info.features
    assert feats["action"]["info"]["apollo_schema"] == 1
    assert feats["action"]["names"][:3] == ["arm0_ee.dx", "arm0_ee.dy", "arm0_ee.dz"]
    import pyarrow.parquet as pq

    table = pq.read_table(next(out.glob("data/**/*.parquet")))
    assert set(table.column("action_source").to_pylist()) == {1}
    wall = np.asarray(table.column("wallclock_ns").to_pylist(), dtype=np.int64)
    assert (np.diff(wall) > 0).all()
    assert "task" not in table.column_names  # dropped: features + five bookkeeping columns only


def test_resume_appends_delete_removes_one_directory(server, api, dataset_root):
    root = dataset_root
    first = next(p for p in (root / "episodes").iterdir())
    mtimes = {p: p.stat().st_mtime_ns for p in first.rglob("*")}
    spec = dict(SPEC, dataset_resume=True, action_filter={"enabled": False})
    r = api.post("/api/session", json=spec)
    assert r.status_code == 200, r.text
    wait_running(api)
    ctl, tele = PulsingCtl(server), Tele(server)
    try:
        assert tele.latest()["episode"]["total_episodes"] == 1  # resumed: counters continue
        assert ctl.action("episode_new")["ok"]
        time.sleep(0.6)
        ctl.hold(["KeyW"], 1.0)
        ctl.keys([])
        time.sleep(1.0)  # idle frames: recorded, the filter is OFF
        st = tele.latest()["episode"]
        assert st["frames_skipped"] == 0 and st["frames"] >= 40, st
        assert ctl.action("episode_save")["ok"]
        idle = wait_state(tele, "idle")
        assert idle["episode"]["total_episodes"] == 2 and idle["episode"]["index"] == 1
        # deleting the PREVIOUS episode during the session is allowed (10-frames §11.7)
        eps = api.get(f"/api/datasets/{REPO_ID}/episodes").json()
        assert [e["index"] for e in eps] == [0, 1] and eps[0]["episode_id"] == first.name
    finally:
        ctl.close()
        tele.close()
        api.delete("/api/session")
    dirs = sorted(p.name for p in (root / "episodes").iterdir())
    assert len(dirs) == 2 and dirs[0] == first.name
    second = root / "episodes" / dirs[1]
    second_ep = json.loads((second / "episode.json").read_text())
    assert second_ep["filter"]["enabled"] is False and second_ep["filter"]["frames_skipped"] == 0
    assert second_ep["filter"]["gaps"] == []
    second_mtimes = {p: p.stat().st_mtime_ns for p in second.rglob("*")}
    assert {p: p.stat().st_mtime_ns for p in first.rglob("*")} == mtimes  # resume rewrote nothing
    assert api.get(f"/api/datasets/{REPO_ID}").json()["export"]["state"] == "stale"

    r = api.delete(f"/api/datasets/{REPO_ID}/episodes/{first.name}")
    assert r.status_code == 204
    assert not first.exists() and second.exists()
    assert {p: p.stat().st_mtime_ns for p in second.rglob("*")} == second_mtimes  # untouched
    info = api.get(f"/api/datasets/{REPO_ID}").json()
    assert info["total_episodes"] == 1 and info["export"]["state"] == "stale"
    assert api.delete(f"/api/datasets/{REPO_ID}/episodes/{first.name}").status_code == 404
    # re-export: one episode
    assert api.post(f"/api/datasets/{REPO_ID}/export", json={}).status_code == 202
    assert server.runtime.manager.dataset_store.wait_export(60.0)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(REPO_ID, root=root / "exports" / "lerobot_v3")
    assert ds.num_episodes == 1
    emap = json.loads((root / "exports/lerobot_v3/meta/apollo/episode_map.json").read_text())
    assert emap["episodes"] == [{"episode_index": 0, "episode_id": second.name}]
    assert api.get(f"/api/datasets/{REPO_ID}").json()["export"]["state"] == "fresh"


def test_resume_refusal_matrix(server, api, dataset_root):
    root = dataset_root

    def refused(body: dict, needle: str) -> None:
        r = api.post("/api/session", json=body)
        assert r.status_code == 409, r.text
        assert needle in r.json()["detail"], (needle, r.json()["detail"])

    refused(SPEC, "already exists")  # dataset_resume false on an existing dataset
    refused(dict(SPEC, dataset="never_recorded", dataset_resume=True), "unknown dataset")
    manifest_path = root / "manifest.json"
    original = manifest_path.read_text()
    try:
        m = json.loads(original)
        m["fps"] = 30
        manifest_path.write_text(json.dumps(m))
        refused(dict(SPEC, dataset_resume=True), "cannot be continued")
        m = json.loads(original)
        m["robot_type"] = "xarm7_1arm_rail"  # a hardware recording
        manifest_path.write_text(json.dumps(m))
        refused(dict(SPEC, dataset_resume=True), "cannot be continued")
    finally:
        manifest_path.write_text(original)
    legacy = server.runtime.cfg.datasets_root / "apollo" / "legacy_v3"
    (legacy / "meta").mkdir(parents=True, exist_ok=True)
    (legacy / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v3.0", "fps": 25, "total_episodes": 1, "total_frames": 10,
        "robot_type": "xarm7_1arm_rail_mujoco", "features": {},
    }))
    refused(dict(SPEC, dataset="legacy_v3", dataset_resume=True), "read-only")
    refused(dict(SPEC, dataset="legacy_v3", dataset_resume=False), "read-only")
    rows = {d["repo_id"]: d for d in api.get("/api/datasets").json()}
    assert rows["apollo/legacy_v3"]["layout"] == "lerobot_v3"
    assert api.delete("/api/datasets/apollo/legacy_v3").status_code == 409  # read-only
    assert api.post("/api/datasets/apollo/legacy_v3/export", json={}).status_code == 409
    assert api.get("/api/session").status_code == 404  # nothing was started by a refusal
    # the whole dataset can go once nothing uses it
    assert api.delete(f"/api/datasets/{REPO_ID}").status_code == 204
    assert not root.exists() and api.get(f"/api/datasets/{REPO_ID}").status_code == 404
