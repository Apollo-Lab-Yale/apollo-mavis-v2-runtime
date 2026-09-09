"""WS/REST contract tests over create_app + a sim workcell (04-runtime §13)."""

from __future__ import annotations

import pytest
from conftest import make_runtime_config
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from apollo_mavis_v2_runtime.runtime import Runtime
from apollo_mavis_v2_runtime.server.app import create_app

SPEC = {
    "mode": "teleop",
    "kind": "sim",
    "arms": ["arm0"],
    "frames": {"arm0": "arm_base:arm0"},
    "sim_scene": "single_rail",
}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    cfg = make_runtime_config(tmp_path_factory.mktemp("rt"))
    app = create_app(Runtime(cfg))
    with TestClient(app) as c:
        yield c
        c.delete("/api/session")


@pytest.fixture()
def session(client):
    r = client.post("/api/session", json=SPEC)
    assert r.status_code == 200, r.text
    yield r.json()
    client.delete("/api/session")


def test_health_and_epoch(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and len(body["epoch"]) == 32


def test_keymap_matches_core(client):
    from apollo_mavis_v2_core.protocol import KEYMAP

    served = client.get("/api/keymap").json()
    assert served == [e.model_dump() for e in KEYMAP]
    # 13-tracker §3: + KeyC tracker_clutch, KeyZ switch_arm_prev; 2026-09-08: + KeyR
    assert len(served) == 24
    codes = [row["code"] for row in served]
    assert len(set(codes)) == 24  # phase-13: unique across the whole table (no keyboard flag)
    assert all("keyboard" not in row for row in served)
    by_code = {row["code"]: row for row in served}
    assert by_code["KeyC"]["gamepad"] == "RT" and by_code["KeyZ"]["gamepad"] == "LB"
    assert by_code["KeyW"]["kind"] == "held" and by_code["KeyS"]["action"] == "translate_x_neg"
    assert {c: by_code[c]["action"] for c in ("KeyN", "Enter", "Backspace")} == {
        "KeyN": "episode_new", "Enter": "episode_save", "Backspace": "episode_discard",
    }
    # 2026-09-08: the return-to-initial key, a plain discrete session row.
    assert by_code["KeyR"] == {
        "code": "KeyR", "action": "reset_to_initial", "kind": "discrete",
        "label": "return to the initial condition", "group": "session",
        "requires_rail": False, "gamepad": None,
    }


def test_collect_without_dataset_name_still_refuses_a_legacy_or_exporting_task_repo(client):
    """``dataset: None`` -> the phase-07 task-derived repo id; a legacy v3 tree or a
    running export there is 409 exactly like a named dataset (review item 4)."""
    import json

    from apollo_mavis_v2_runtime.recorder.features import ArmMeta, build_repo_id

    rt = client.app.state.runtime
    repo_id = build_repo_id("legacy task", [ArmMeta("arm0", True)], {"arm0": "arm_base:arm0"})
    legacy = rt.cfg.datasets_root / repo_id
    (legacy / "meta").mkdir(parents=True)
    (legacy / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v3.0", "fps": 25}))
    body = dict(SPEC, mode="collect", task="legacy task", return_to_start=False)
    r = client.post("/api/session", json=body)
    assert r.status_code == 409 and "read-only" in r.json()["detail"], r.text
    assert client.get("/api/session").status_code == 404
    store = rt.manager.dataset_store
    real = type(store).exporting
    try:
        type(store).exporting = property(lambda self: repo_id)  # an export of THAT repo runs
        r = client.post("/api/session", json=body)
        assert r.status_code == 409 and "being exported" in r.json()["detail"], r.text
    finally:
        type(store).exporting = real
    import shutil

    shutil.rmtree(legacy)


def test_datasets_routes_pre_session(client):
    """/api/datasets* (04-runtime §13.1): an empty root lists nothing; unknown ids are
    404; the episode-id path pattern admits the 10-frames §11.3 grammar only."""
    assert client.get("/api/datasets").json() == []
    assert client.get("/api/datasets/apollo/nope").status_code == 404
    assert client.get("/api/datasets/apollo/nope/episodes").status_code == 404
    assert client.delete("/api/datasets/apollo/nope").status_code == 404
    r = client.delete("/api/datasets/apollo/nope/episodes/20260907T141203.512Z-3f9a1c")
    assert r.status_code == 404
    assert client.delete("/api/datasets/apollo/nope/episodes/..%2Fx").status_code in (404, 422)
    r = client.post("/api/datasets/apollo/nope/export", json={"format": "lerobot_v3"})
    assert r.status_code == 404
    r = client.post("/api/datasets/apollo/nope/export", json={"format": "aloha_hdf5"})
    assert r.status_code == 422
    # the deprecated alias answers with nulls without a collect session
    assert client.get("/api/episodes").json() == {
        "repo_id": None, "total_episodes": 0, "total_frames": 0,
    }


def test_scenes_listing(client):
    """Phase-11: the UI/API list only the lab scene (registry ``hidden`` filter)
    and label it by its display title; hidden scenes stay buildable by id
    (sessions and tests keep using ``single_rail``)."""
    from apollo_mavis_v2_sim import REGISTRY
    from apollo_mavis_v2_sim.scenes.descriptor import SceneMeta

    scenes = client.get("/api/scenes", params={"kind": "sim"}).json()
    ids = {s["scene_id"] for s in scenes}
    assert "mavis_v2" in ids
    row = next(s for s in scenes if s["scene_id"] == "mavis_v2")
    assert row["num_arms"] == 2 and row["rail_flags"] == [True, True]
    assert {"view", "grip"} <= set(REGISTRY.meta("mavis_v2").arm_ids)
    meta = REGISTRY.meta("mavis_v2")
    if "hidden" in getattr(SceneMeta, "__dataclass_fields__", {}):  # sim phase-11 landed
        assert ids == {"mavis_v2"}, ids
        assert row["label"] == meta.title == "APOLLO MAVIS V2 Digital Twin"
        twin = client.get("/api/scenes", params={"kind": "twin"}).json()
        assert {s["scene_id"] for s in twin} == {"mavis_v2"}
    else:  # older registry: full listing, description label
        assert "single_rail" in ids
        assert row["label"] == meta.description
    # Hidden scenes are still addressable by id (sessions validate via REGISTRY.meta).
    assert REGISTRY.meta("single_rail").n_arms == 1
    assert client.get("/api/scenes", params={"kind": "nope"}).status_code == 422


def test_workcell_and_cameras_pre_session(client):
    ws = client.get("/api/workcell").json()
    assert ws["kind"] == "sim" and ws["available_kinds"] == ["sim"]
    assert ws["hardware_ready"] is False  # no hardware workcell configured
    arm = ws["arms"][0]
    assert arm["has_rail"] and len(arm["joint_limits"]) == 8
    assert arm["joint_limits"][7] == [0.0, 0.65]
    assert arm["reachable"] == "unknown" and arm["ip"] is None  # sim rows are not probed
    cams = client.get("/api/cameras").json()
    assert {c["camera_id"] for c in cams} == {"cam_front", "arm0_wrist_cam"}
    assert all(c["live"] for c in cams)
    # ?kind= selects the workcell described; sim == legacy rows here.
    assert client.get("/api/workcell", params={"kind": "sim"}).json()["arms"] == ws["arms"]
    hw = client.get("/api/workcell", params={"kind": "hardware"}).json()
    assert hw["kind"] == "hardware" and hw["arms"] == [] and hw["cameras"] == []
    assert hw["available_kinds"] == ["sim"] and hw["hardware_ready"] is False
    assert client.get("/api/workcell", params={"kind": "nope"}).status_code == 422
    # No microphone configured -> empty list, no telemetry block.
    assert client.get("/api/microphones").json() == []


def test_session_404_then_lifecycle(client):
    assert client.get("/api/session").status_code == 404
    r = client.post("/api/session", json=SPEC)
    assert r.status_code == 200
    info = r.json()
    assert info["mode"] == "teleop" and "sim" in info["streams"]
    assert client.post("/api/session", json=SPEC).status_code == 409  # singleton
    got = client.get("/api/session").json()
    assert got["session_id"] == info["session_id"]
    assert client.delete("/api/session").status_code == 204
    assert client.delete("/api/session").status_code == 204  # idempotent
    assert client.get("/api/session").status_code == 404


def test_post_session_409_matrix(client):
    bad_kind = dict(SPEC, kind="hardware", digital_twin_scene="single_rail")
    assert client.post("/api/session", json=bad_kind).status_code == 409
    bad_mode = dict(SPEC, mode="dagger", task="t")  # collect lands in phase-07
    assert client.post("/api/session", json=bad_mode).status_code == 409
    bad_arms = dict(SPEC, arms=["arm9"], frames={})
    assert client.post("/api/session", json=bad_arms).status_code == 409
    bad_scene = dict(SPEC, sim_scene="not_a_scene")
    assert client.post("/api/session", json=bad_scene).status_code == 409
    bad_profile = dict(SPEC, start_from="profile:doesnotexist")
    assert client.post("/api/session", json=bad_profile).status_code == 409
    invalid = dict(SPEC, start_from="gohome")  # fails core regex -> 422
    assert client.post("/api/session", json=invalid).status_code == 422


def test_ws_control_hello_roles_and_observer_nack(client, session):
    with client.websocket_connect("/ws/control") as ws1:
        hello1 = ws1.receive_json()
        assert hello1["t"] == "hello" and hello1["role"] == "controller"
        assert hello1["epoch"] and hello1["session_id"] == session["session_id"]
        with client.websocket_connect("/ws/control") as ws2:
            hello2 = ws2.receive_json()
            assert hello2["role"] == "observer"  # never close-1008
            ws2.send_json({"t": "action", "name": "switch_arm"})
            ack = ws2.receive_json()
            assert ack == {"t": "ack", "name": "switch_arm", "ok": False,
                           "detail": "observer"}
        ws1.send_json({"t": "action", "name": "switch_arm"})
        ack = ws1.receive_json()
        assert ack["ok"] is True


def test_ws_control_no_session_nack(client):
    with client.websocket_connect("/ws/control") as ws:
        assert ws.receive_json()["session_id"] is None
        ws.send_json({"t": "action", "name": "switch_arm"})
        ack = ws.receive_json()
        assert not ack["ok"] and ack["detail"] == "no session"


def test_ws_control_invalid_args_nack(client, session):
    with client.websocket_connect("/ws/control") as ws:
        ws.receive_json()
        ws.send_json({"t": "action", "name": "joint_target", "args": {"nope": 1}})
        ack = ws.receive_json()
        assert not ack["ok"] and "invalid args" in ack["detail"]


def test_ws_video_unknown_and_reserved_pre_session(client):
    with client.websocket_connect("/ws/video/never-existed") as ws:
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_bytes()
        assert exc.value.code == 1008
    with client.websocket_connect("/ws/video/sim") as ws:  # session-only id
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_bytes()
        assert exc.value.code == 1008


def test_telemetry_pre_session_idle(client):
    with client.websocket_connect("/ws/telemetry") as ws:
        msg = ws.receive_json()
        assert msg["t"] == "telemetry"
        assert msg["session"]["state"] == "idle"
        assert msg["arms"] == [] and msg["collision"]["severity"] == "ok"
        # Tracker block is populated pre-session (device fields; backend none here).
        trk = msg["tracker"]
        assert trk["backend"] == "none" and trk["status"] == "no_backend"
        assert trk["clutch"] is False and trk["engaged_arm"] is None
        assert trk["settings"] == {
            "yaw_deg": 0.0, "pos_scale": 1.0, "follow_rotation": True,
            "filter_enabled": True, "filter_min_cutoff_hz": 1.0, "filter_beta": 5.0,
        }
        assert trk["pose_filtered"] is None and trk["device_action"] is None
        assert msg["microphone"] is None  # microphone.enabled false in this config
        # Phase-09a: the hardware_monitor block is always present; no hardware
        # workcell here -> the valid inert block (core HardwareMonitorTelemetry()).
        assert msg["hardware_monitor"] == {
            "enabled": False, "paused": False, "arms": [], "overlays": [],
        }
        assert list(msg)[-6:] == [
            "session", "tracker", "microphone", "external", "hardware_monitor", "datasets",
        ]
        assert msg["external"]["state"] == "disabled"  # dora.enabled false (phase-12)
        assert msg["datasets"] is None  # no export ran in this process


# -- Online DAgger (phase-14; 15-online-dagger §3, §7, §9) --------------------------------------
OD_SPEC = {
    "mode": "dagger",
    "kind": "sim",
    "arms": ["arm0"],
    "frames": {"arm0": "arm_base:arm0"},
    "sim_scene": "single_rail",
    "task": "pick",
    "policy_source": "external",
    "return_to_start": False,
    "online_dagger": {"session_name": "s1"},
}


def test_online_dagger_skill_routes(client):
    import io
    import tarfile

    from apollo_mavis_v2_runtime.online_dagger import SKILL_NAME, skill_markdown

    r = client.get("/api/online_dagger/skill")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/markdown")
    md = r.text
    assert md == skill_markdown() and md.startswith("---\nname: mavis-online-dagger-trainer\n")
    r = client.get("/api/online_dagger/skill.tgz")
    assert r.status_code == 200 and r.headers["content-type"] == "application/gzip"
    assert r.headers["content-disposition"] == f'attachment; filename="{SKILL_NAME}.tgz"'
    with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tar:
        assert tar.getnames() == [
            f"{SKILL_NAME}/SKILL.md",
            f"{SKILL_NAME}/references/contract.md",
            f"{SKILL_NAME}/references/pro-dagger-example.md",
        ]
        assert tar.extractfile(f"{SKILL_NAME}/SKILL.md").read().decode("utf-8") == md
    assert client.get("/api/online_dagger/sessions").json() == []  # no online_dagger root yet
    # the v1.0 prefix is gone, not aliased
    assert client.get("/api/pro_dagger/skill").status_code == 404
    assert client.get("/api/pro_dagger/sessions").status_code == 404


def test_online_dagger_skill_dir_override_and_404(client, tmp_path):
    rt = client.app.state.runtime
    real = rt.cfg.online_dagger.skill_dir
    try:
        object.__setattr__(rt.cfg.online_dagger, "skill_dir", tmp_path / "missing")
        r = client.get("/api/online_dagger/skill")
        assert r.status_code == 404 and "Online DAgger skill not found" in r.json()["detail"]
        assert client.get("/api/online_dagger/skill.tgz").status_code == 404
        alt = tmp_path / "alt"
        alt.mkdir()
        (alt / "SKILL.md").write_text("---\nname: alt\n---\n", encoding="utf-8")
        object.__setattr__(rt.cfg.online_dagger, "skill_dir", alt)
        assert client.get("/api/online_dagger/skill").text == "---\nname: alt\n---\n"
    finally:
        object.__setattr__(rt.cfg.online_dagger, "skill_dir", real)


def test_online_dagger_sessions_lists_session_json_rows_newest_first(client):
    import shutil

    from apollo_mavis_v2_runtime.dagger.online_dagger import write_session_json_atomic

    rt = client.app.state.runtime
    root = rt.manager.online_dagger_root()
    assert root == rt.cfg.datasets_root / "online_dagger"  # no mapped namespace in this config
    docs = {
        "old": {
            "session_name": "old", "created_at": "2026-09-08T10:00:00.000Z", "task": "pick",
            "rollouts": [{"episode_id": "a"}, {"episode_id": "b"}, {"episode_id": "c"}],
            "current": {"phase": "rollout", "rollouts_saved": 3},
            "last_used_at": "2026-09-08T11:00:00.000Z",
        },
        "new": {
            "session_name": "new", "created_at": "2026-09-08T12:00:00.000Z",
            "spec": {"task": "place", "online_dagger": {"session_name": "new"}},
            "current": {"phase": "waiting_trainer", "rollouts_saved": 0},
            "last_used_at": "2026-09-08T12:30:00.000Z",
        },
    }
    try:
        for name, doc in docs.items():
            write_session_json_atomic(root / name / "session.json", doc)
        (root / "junk").mkdir()
        (root / "junk" / "session.json").write_text("nope")
        rows = client.get("/api/online_dagger/sessions").json()
        assert [r["session_name"] for r in rows] == ["new", "old"]
        assert rows[1] == {
            "session_name": "old", "path": str(root / "old"),
            "created_at": "2026-09-08T10:00:00.000Z", "task": "pick", "rollouts": 3,
            "last_used_at": "2026-09-08T11:00:00.000Z",
        }
        assert rows[0]["task"] == "place" and rows[0]["rollouts"] == 0
        assert set(rows[0]) == {
            "session_name", "path", "created_at", "task", "rollouts", "last_used_at",
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_online_dagger_post_409_matrix(client, tmp_path):
    """Every Online DAgger refusal (15-online-dagger §3/§7) is evaluated BEFORE any side
    effect: the session-directory rule, then the trainer (bridge, spec heartbeat,
    ``online_dagger`` capability). There is NO offline-dataset check any more."""
    import shutil
    from types import SimpleNamespace

    from dora_bridge.hubfakes import announce, make_bridge_and_hub, spec_event

    rt = client.app.state.runtime
    od_root = rt.manager.online_dagger_root()

    def refused(body, text):
        r = client.post("/api/session", json=body)
        assert r.status_code == 409, r.text
        assert text in r.json()["detail"], r.text
        assert client.get("/api/session").status_code == 404

    real_dora = rt.manager.dora
    try:
        # the session-directory rule
        (od_root / "s1").mkdir(parents=True)
        refused(
            OD_SPEC, "Online DAgger session 's1' already exists - resume it or pick another name"
        )
        resume = {**OD_SPEC, "online_dagger": {"session_name": "s2", "resume": True}}
        refused(resume, "Online DAgger session 's2' not found")
        (od_root / "s1" / "session.json").write_text("{not json", encoding="utf-8")
        resume1 = {**OD_SPEC, "online_dagger": {"session_name": "s1", "resume": True}}
        refused(
            resume1, "Online DAgger session 's1': session.json is unreadable - fix or remove it"
        )
        assert (od_root / "s1" / "session.json").read_text(encoding="utf-8") == "{not json"
        (od_root / "s1" / "session.json").write_text("[]", encoding="utf-8")
        refused(resume1, "session.json is unreadable")
        fresh = {**OD_SPEC, "online_dagger": {"session_name": "s3"}}
        # the trainer: no bridge, no spec heartbeat, no capability
        refused(fresh, "no external policy attached (dora bridge is not attached)")
        clock, bridge, hub, _node = make_bridge_and_hub(tmp_path / "dora")
        rt.manager.dora = SimpleNamespace(enabled=False, bridge=bridge, policy_hub=hub)
        refused(fresh, "no external policy attached (no policy_spec heartbeat within 3 s)")
        hub._on_spec(spec_event(announce()))
        refused(
            fresh,
            "no Online DAgger trainer attached (the policy node does not report the "
            "online_dagger capability)",
        )
        assert not (od_root / "s3").exists()  # refused before any side effect
        assert not (od_root / "s2").exists()
        # the core validators still 422 a malformed block
        bad = {**OD_SPEC, "dataset": "x"}
        assert client.post("/api/session", json=bad).status_code == 422
        bad = {**OD_SPEC, "policy_source": "checkpoint"}
        assert client.post("/api/session", json=bad).status_code == 422
        bad = {**OD_SPEC, "online_dagger": {"session_name": "a/b"}}
        assert client.post("/api/session", json=bad).status_code == 422
        # no hyper-parameters / dataset ride the block (extra="forbid")
        bad = {**OD_SPEC, "online_dagger": {"session_name": "s3", "offline_dataset": "demo"}}
        assert client.post("/api/session", json=bad).status_code == 422
        bad = {**OD_SPEC, "online_dagger": {"session_name": "s3", "rollouts_per_iteration": 5}}
        assert client.post("/api/session", json=bad).status_code == 422
    finally:
        rt.manager.dora = real_dora
        shutil.rmtree(od_root, ignore_errors=True)


def test_gate_ops_and_train_now_are_nacked_outside_their_sessions(client, session):
    assert session["online_dagger"] is None  # SessionInfo echo (phase-14)
    with client.websocket_connect("/ws/control") as ws:
        ws.receive_json()
        ws.send_json({"t": "action", "name": "train_now"})
        assert ws.receive_json() == {
            "t": "ack", "name": "train_now", "ok": False,
            "detail": "not an Online DAgger session",
        }
        for name in ("takeover", "handback", "takeover_toggle"):
            ws.send_json({"t": "action", "name": name})
            assert ws.receive_json() == {
                "t": "ack", "name": name, "ok": False,
                "detail": "takeover not available in teleop",
            }
        ws.send_json({"t": "action", "name": "train_now", "args": {"x": 1}})
        ack = ws.receive_json()
        assert not ack["ok"] and "invalid args" in ack["detail"]
        ws.send_json({"t": "action", "name": "takeover", "args": {"x": 1}})
        assert not ws.receive_json()["ok"]
