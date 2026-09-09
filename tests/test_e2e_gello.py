"""Full-stack sim e2e of GELLO Manipulation (16-gello §5 / §6; phase-15) over a REAL uvicorn
server on the KITCHEN twin (``mavis_v2_kitchen``) with the full safety stack (``safety_debug``
gate + twin) and the FAKE leader (``gello.backend: fake``, moved with
``srv.runtime.gello.fake_set``):

* ``POST /api/gello/preview``: ``clear`` + a PNG of ``cam_kitchen`` at the synced posture,
  ``collision`` with the ``fridge_body`` pair (and a PNG) with the leader inside the fridge
  shell, ``joint_limit``; ``kind: hardware`` is 409 (no hardware workcell here);
* ``POST /api/session {mode: gello}`` with the leader inside the fridge shell is 409 with the
  colliding pair and leaves NO session behind;
* a synced launch reaches ``tracking`` (the fake's default posture = the keyframe of the
  Manipulation Arm); ``fake_set`` moves the follower at the 0.6 rad/s cap (never a jump);
  ``ArrowLeft`` moves ONLY the rail slot; ``switch_arm`` is nacked; ``gello_pause`` then
  ``fake_set`` leaves the follower still and ``gello_resume`` re-engages; ``R`` forces
  ``paused``; ``SessionInfo.gello`` echoes the block.
"""

from __future__ import annotations

import base64
import math
import time

import httpx
import numpy as np
import pytest
from apollo_mavis_v2_core import ArmPosture, StateProfile, WorkcellConfig
from conftest import ControlConfig, DatasetsConfig, LiveServer, RuntimeConfig, VideoConfig
from test_e2e_teleop import PulsingCtl, Tele
from test_gello_session import FRIDGE_HIT, HOLD, KITCHEN

from apollo_mavis_v2_runtime.config import GelloConfig, MicrophoneConfig
from apollo_mavis_v2_runtime.devices.gello import FAKE_DEFAULT_Q
from apollo_mavis_v2_runtime.gello.loop import GELLO_NO_ARM_SWITCH, GelloLoop

pytestmark = pytest.mark.egl

PI = math.pi
SPEC = {
    "mode": "gello",
    "kind": "sim",
    "arms": ["view", "grip"],
    "frames": {"view": "arm_base:view", "grip": "arm_base:grip"},
    "sim_scene": KITCHEN,
    "gello": {},
}
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _config(tmp_path) -> RuntimeConfig:
    wc = {
        "kind": "sim",
        "sim_scene": "mavis_v2",  # the Welcome previews; the GELLO card launches the kitchen
        "arms": [
            {"id": "view", "base_in_world": {}, "gripper": "none", "microphone": True},
            {"id": "grip", "base_in_world": {}},
        ],
        "cameras": [],
        "safety": {"safety_debug": True},
    }
    return RuntimeConfig(
        workcells={"sim": WorkcellConfig.model_validate(wc)},
        profiles_dir=tmp_path / "profiles",
        datasets_root=tmp_path / "datasets",
        datasets=DatasetsConfig(default_namespace="apollo", namespaces={}),
        checkpoints_root=tmp_path / "ckpts",
        calibration_dir=tmp_path / "calibration",
        video=VideoConfig(preview_fps=15, session_fps=30),
        control=ControlConfig(translate_frame="base"),
        microphone=MicrophoneConfig(enabled=False),
        gello=GelloConfig(backend="fake", calibration_path=tmp_path / "gello_calibration.json"),
    )


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    srv = LiveServer(_config(tmp_path_factory.mktemp("rt")))
    store = srv.runtime.profile_store
    initial = store.save(
        StateProfile(
            name="initial",
            workcell_kind="sim",
            arms={
                "grip": ArmPosture(q=list(FAKE_DEFAULT_Q), rail_pos_m=None, gripper_open_frac=1.0),
                "view": ArmPosture(q=list(HOLD), rail_pos_m=None),
            },
        )
    )
    store.set_initial(initial.profile_id)
    with httpx.Client(base_url=srv.http, timeout=30) as api:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if api.get("/api/gello").json()["status"] == "connected":
                break
            time.sleep(0.05)
    yield srv
    srv.stop()


@pytest.fixture()
def api(server):
    with httpx.Client(base_url=server.http, timeout=120.0) as client:
        yield client


def _leader(server, q, gripper=1.0) -> None:
    server.runtime.gello.fake_set(np.asarray(q, dtype=np.float64), gripper)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        s = server.runtime.gello.latest()
        if s is not None and np.allclose(s.q, q):
            return
        time.sleep(0.01)
    raise AssertionError("fake leader did not take the posture")


def _wait_running(api, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if api.get("/api/session").json()["state"] == "running":
            return
        time.sleep(0.05)
    raise AssertionError("session never reached running")


def _wait_gello(tele: Tele, pred, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        msg = tele.latest()
        if pred(msg["gello"]):
            return msg
        if time.monotonic() > deadline:
            raise AssertionError(f"gello block never satisfied the predicate: {msg['gello']}")


def _arm(msg: dict, arm_id: str) -> dict:
    return next(a for a in msg["arms"] if a["arm_id"] == arm_id)


def test_preview_clear_then_collision_with_png_then_joint_limit(server, api):
    r = api.post("/api/gello/preview", json={"kind": "sim"})
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["status"] == "clear" and res["ok"] and res["detail"] == "clear"
    assert res["camera"] == "cam_kitchen" and res["pairs"] == []
    assert res["q_goal"]["grip"] == pytest.approx([*FAKE_DEFAULT_Q, 0.65])
    assert res["q_goal"]["view"] == pytest.approx([*HOLD, 0.0])
    assert res["leader_q"] == pytest.approx(list(FAKE_DEFAULT_Q))
    png = base64.b64decode(res["image_png_b64"])
    assert png.startswith(PNG_MAGIC) and len(png) > 2000
    # the fridge shell: collision, the pair, a (tinted) PNG; the sheet keeps polling
    _leader(server, FRIDGE_HIT)
    try:
        t0 = time.monotonic()
        r = api.post("/api/gello/preview", json={"kind": "sim"})
        dt = time.monotonic() - t0
        assert r.status_code == 200, r.text
        res = r.json()
        assert res["status"] == "collision" and not res["ok"]
        assert res["pairs"][0]["a"] == "fridge_body" and res["pairs"][0]["b"] == "grip_right_finger"
        assert res["pairs"][0]["dist_m"] < -0.07
        assert res["detail"].startswith("collides: fridge_body / grip_right_finger at -7")
        assert base64.b64decode(res["image_png_b64"]).startswith(PNG_MAGIC)
        assert dt < 2.0  # a cached twin: no scene build per poll
        # joint limit
        over = list(FAKE_DEFAULT_Q)
        over[1] = 2.3
        _leader(server, over)
        res = api.post("/api/gello/preview", json={"kind": "sim"}).json()
        assert res["status"] == "joint_limit" and not res["ok"]
        assert "joint 2 = 2.30 rad" in res["detail"]
        assert res["image_png_b64"] is not None  # rendered at the clipped posture
    finally:
        _leader(server, FAKE_DEFAULT_Q)
    r = api.post("/api/gello/preview", json={"kind": "hardware"})
    assert r.status_code == 409 and "no hardware workcell is configured" in r.json()["detail"]
    assert api.post("/api/gello/preview", json={"kind": "x"}).status_code == 422
    r = api.post("/api/gello/preview", json={"kind": "sim", "scene": "single_rail"})
    assert r.status_code == 200 and r.json()["status"] == "scene_error"


def test_launch_inside_the_fridge_shell_is_409_with_the_pair_and_no_session(server, api):
    _leader(server, FRIDGE_HIT)
    try:
        r = api.post("/api/session", json=SPEC)
        assert r.status_code == 409, r.text
        detail = r.json()["detail"]
        assert detail.startswith("GELLO posture collides: fridge_body / grip_right_finger at -7")
        assert detail.endswith(" mm - move GELLO and retry")
        assert api.get("/api/session").status_code == 404
        # 422s from core: a dataset / a profile start / a foreign mode with a gello block
        assert api.post("/api/session", json={**SPEC, "dataset": "x"}).status_code == 422
        assert api.post("/api/session", json={**SPEC, "start_from": "profile:x"}).status_code == 422
        assert api.post("/api/session", json={**SPEC, "mode": "teleop"}).status_code == 422
    finally:
        _leader(server, FAKE_DEFAULT_Q)


def test_synced_launch_tracks_follows_at_the_cap_rail_pause_resume_and_R(server, api):
    _leader(server, FAKE_DEFAULT_Q)
    r = api.post("/api/session", json=SPEC)
    assert r.status_code == 200, r.text
    info = r.json()
    assert info["mode"] == "gello" and info["gello"] == {"viewpoint": "auto"}
    assert info["policy_source"] == "checkpoint"
    _wait_running(api)
    session = server.runtime.manager.session
    assert isinstance(session.loop, GelloLoop) and session.loop.tracker is None
    assert session.loop.cfg.dq_max_rad == pytest.approx(0.006)  # 0.6 rad/s in sim too
    assert session.twin.allowed.allows_labels("grip_left_finger", "fridge_door_handle")
    ctl = PulsingCtl(server)
    tele = Tele(server)
    try:
        msg = _wait_gello(tele, lambda g: g["state"] == "tracking")
        g = msg["gello"]
        assert g["engaged_arm"] == "grip" and g["backend"] == "fake"
        assert g["viewpoint"] == {
            "mode": "auto",
            "attached": False,
            "policy_id": None,
            "detail": "no dora bridge (dora.enabled is false) - holding the GELLO posture",
            "paused": False,  # 2026-09-09 review: the NaN pause is on the wire
        }
        assert g["paused_latched"] is False
        assert msg["active_arm"] == "grip" and msg["session"]["state"] == "running"
        assert api.get("/api/session").json()["fault_detail"] == ""
        assert _arm(msg, "view")["q"] == pytest.approx(HOLD, abs=2e-3)
        # -- the follower streams the leader at the cap ------------------------------------------
        # joint 2 goes NEGATIVE: at +0.23 rad the gripper finger reaches the arm's own carriage
        # (grip_rail_base) and the kitchen twin's gate holds the command there (checked on the
        # twin, 2026-09-09) - which is the gate doing its job, not the follower's cap
        target = np.array(FAKE_DEFAULT_Q)
        target[1] -= 0.3
        t0 = time.monotonic()
        _leader(server, target)
        samples: list[tuple[float, float]] = []
        while time.monotonic() - t0 < 1.5:
            m = tele.latest()
            samples.append((time.monotonic() - t0, _arm(m, "grip")["q"][1]))
        q2 = [q for _, q in samples]
        assert q2[-1] == pytest.approx(-0.3, abs=0.01)
        # never a jump: successive telemetry samples (~40 ms apart) move <= 0.6 rad/s x dt + slack
        for (ta, qa), (tb, qb) in zip(samples, samples[1:], strict=False):
            assert qa - qb <= 0.6 * (tb - ta) + 0.03, (ta, qa, tb, qb)
        assert q2[0] - q2[-1] > 0.2  # ... and it did get there
        m = tele.latest()
        assert _arm(m, "grip")["q"][0] == pytest.approx(PI, abs=1e-3)  # other joints untouched
        assert _arm(m, "grip")["rail_pos_m"] == pytest.approx(0.65, abs=2e-3)
        assert m["gello"]["max_lag_rad"] < 0.02
        # -- ArrowLeft moves ONLY the rail slot of the Manipulation Arm ----------------------------
        # (the keyframe carriage sits at 0.65 = the operator's RIGHT end: ArrowRight has nowhere
        # to go, ArrowLeft drives it toward the zero end)
        rail0 = _arm(m, "grip")["rail_pos_m"]
        ctl.hold(["ArrowLeft"], 1.0)
        time.sleep(0.4)
        m = tele.latest()
        assert rail0 - _arm(m, "grip")["rail_pos_m"] > 0.05
        assert _arm(m, "grip")["q"] == pytest.approx(list(target), abs=5e-3)
        assert _arm(m, "view")["rail_pos_m"] == pytest.approx(0.0, abs=2e-3)
        assert m["gello"]["state"] == "tracking"
        # translate keys do nothing to the Manipulation Arm
        ctl.hold(["KeyW"], 0.5)
        time.sleep(0.3)
        m = tele.latest()
        assert _arm(m, "grip")["q"] == pytest.approx(list(target), abs=5e-3)
        # -- nacks ------------------------------------------------------------------------------
        ack = ctl.action("switch_arm")
        assert not ack["ok"] and ack["detail"] == GELLO_NO_ARM_SWITCH
        assert not ctl.action("episode_new")["ok"]
        assert not ctl.action("takeover")["ok"]
        # -- pause: the leader moves, the follower does not; resume re-engages -------------------
        ack = ctl.action("gello_pause")
        assert ack["ok"] and ack["detail"] == "paused"
        _wait_gello(tele, lambda g: g["state"] == "paused")
        _leader(server, FAKE_DEFAULT_Q)
        time.sleep(0.6)
        m = tele.latest()
        assert _arm(m, "grip")["q"][1] == pytest.approx(-0.3, abs=0.01)  # still
        assert m["gello"]["state"] == "paused" and m["gello"]["max_lag_rad"] > 0.25
        assert m["gello"]["engaged_arm"] is None
        ack = ctl.action("gello_resume")
        assert ack["ok"] and ack["detail"] == "resumed"
        # 0.3 rad away = beyond the 0.10 engage tolerance: out_of_sync until the leader comes back
        _wait_gello(tele, lambda g: g["state"] == "out_of_sync")
        _leader(server, target)
        _wait_gello(tele, lambda g: g["state"] == "tracking")
        _leader(server, FAKE_DEFAULT_Q)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            m = tele.latest()
            if abs(_arm(m, "grip")["q"][1]) < 0.01:
                break
        assert abs(_arm(m, "grip")["q"][1]) < 0.01  # followed back
        # -- R forces paused (the initial condition = this posture -> the motion is skipped) -------
        ack = ctl.action("reset_to_initial")
        assert ack["ok"], ack
        m = _wait_gello(tele, lambda g: g["state"] == "paused")
        detail = m["gello"]["state_detail"]
        assert "planned motion" in detail or "Return" in detail
        time.sleep(0.5)
        assert tele.latest()["gello"]["state"] == "paused"  # sticky until Resume
        assert ctl.action("gello_resume")["ok"]
        _wait_gello(tele, lambda g: g["state"] == "tracking")
    finally:
        ctl.close()
        tele.close()
        api.delete("/api/session")
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and api.get("/api/session").status_code != 404:
        time.sleep(0.05)
    assert api.get("/api/session").status_code == 404
    # session-less again: the preview still answers
    assert api.post("/api/gello/preview", json={"kind": "sim"}).json()["status"] == "clear"
