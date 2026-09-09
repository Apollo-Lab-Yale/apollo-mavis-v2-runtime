"""13-tracker §4 e2e: the ``fake`` tracker backend on ``mavis_v2`` through a
REAL uvicorn server — clutch via ``KeyC`` moves the EE, telemetry carries
``tracker.*`` pre-session and in-session, settings/switch_arm_prev ack, no
ERROR logs. §1.1: a scripted controller state on the fake backend drives the EE,
the gripper and the rail without any KeysMsg, even while the WS deadman is
latched; the menu button switches arms."""

from __future__ import annotations

import json
import logging
import math
import time

import httpx
import pytest
from conftest import LiveServer, make_runtime_config
from websockets.sync.client import connect as ws_connect

from apollo_mavis_v2_runtime.config import TrackerConfig
from apollo_mavis_v2_runtime.devices.tracker import ControllerState

SPEC = {
    "mode": "teleop",
    "kind": "sim",
    "arms": ["view", "grip"],
    "frames": {"view": "arm_base:view", "grip": "arm_base:grip"},
    "sim_scene": "mavis_v2",
}


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    cfg = make_runtime_config(
        tmp_path_factory.mktemp("rt"), scene="mavis_v2", tracker=TrackerConfig(backend="fake")
    )
    srv = LiveServer(cfg)
    yield srv
    srv.stop()


@pytest.fixture(scope="module")
def api(server):
    with httpx.Client(base_url=server.http, timeout=30.0) as client:
        yield client


class Ctl:
    def __init__(self, server):
        self.sock = ws_connect(f"{server.ws}/ws/control")
        self.hello = json.loads(self.sock.recv(timeout=5))
        self.seq = 0

    def keys(self, held: list[str]) -> None:
        self.seq += 1
        self.sock.send(json.dumps({"t": "keys", "seq": self.seq, "ts": time.time(), "held": held}))

    def action(self, name: str, args: dict | None = None) -> dict:
        self.sock.send(json.dumps({"t": "action", "name": name, "args": args or {}}))
        ack = json.loads(self.sock.recv(timeout=10))
        assert ack["t"] == "ack" and ack["name"] == name, ack
        return ack

    def close(self) -> None:
        self.sock.close()


class Tele:
    def __init__(self, server):
        self.sock = ws_connect(f"{server.ws}/ws/telemetry")

    def latest(self) -> dict:
        msg = json.loads(self.sock.recv(timeout=5))
        while True:
            try:
                msg = json.loads(self.sock.recv(timeout=0.001))
            except TimeoutError:
                return msg

    def close(self) -> None:
        self.sock.close()


FILTER_DEFAULTS = {"filter_enabled": True, "filter_min_cutoff_hz": 1.0, "filter_beta": 5.0}


def _dist(a, b) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


@pytest.fixture()
def session(api):
    r = api.post("/api/session", json=SPEC)
    assert r.status_code == 200, r.text
    for _ in range(200):
        if api.get("/api/session").json()["state"] == "running":
            break
        time.sleep(0.05)
    yield r.json()
    api.delete("/api/session")


def test_tracker_telemetry_pre_session_device_fields(server, api):
    assert api.get("/api/session").status_code == 404
    tele = Tele(server)
    try:
        deadline = time.monotonic() + 5.0
        trk = tele.latest()["tracker"]
        while trk["status"] != "tracking" and time.monotonic() < deadline:
            trk = tele.latest()["tracker"]
        assert trk["backend"] == "fake" and trk["status"] == "tracking", trk
        assert trk["object_name"] == "WM0" and trk["seq"] > 0 and trk["rate_hz"] > 50
        assert trk["age_s"] is not None and trk["age_s"] < 0.2
        assert trk["pose_raw"] is not None and trk["pose_world"] is not None
        assert _dist(trk["pose_raw"]["position"], trk["pose_world"]["position"]) < 1e-9  # yaw 0
        assert trk["clutch"] is False and trk["engaged_arm"] is None
        assert trk["anchor_tcp"] is None and trk["target_tcp"] is None
        assert trk["settings"] == FILTER_DEFAULTS | {
            "yaw_deg": 0.0, "pos_scale": 1.0, "follow_rotation": True
        }
        assert trk["pose_filtered"] is None and trk["device_action"] is None  # no session
    finally:
        tele.close()


def test_clutch_keyc_moves_ee_then_release_holds(server, api, session, caplog):
    caplog.set_level(logging.INFO)
    ctl = Ctl(server)
    tele = Tele(server)
    try:
        assert ctl.hello["role"] == "controller"
        msg = tele.latest()
        assert msg["active_arm"] == "grip"  # the Manipulation Arm is the default teleop arm
        p0 = msg["arms"][1]["ee_pose"]["position"]  # telemetry arms follow the spec: [view, grip]
        engaged_frames = 0
        end = time.monotonic() + 2.0
        while time.monotonic() < end:  # KeysMsg + 25 Hz heartbeat with the clutch held
            ctl.keys(["KeyC"])
            time.sleep(0.04)
            trk = tele.latest()["tracker"]
            if trk["engaged_arm"] == "grip":
                engaged_frames += 1
                assert trk["clutch"] is True
                assert trk["anchor_tcp"] is not None and trk["target_tcp"] is not None
                assert trk["status"] == "tracking"
        assert engaged_frames > 20, engaged_frames
        msg = _wait_ee_moved(tele, p0, 1, tick=lambda: ctl.keys(["KeyC"]))
        trk = msg["tracker"]
        # The target never leads the anchored hand by more than the hand moved.
        assert _dist(trk["target_tcp"]["position"], trk["anchor_tcp"]["position"]) > 0.01
        ctl.keys([])  # release: hold-last, anchors cleared
        time.sleep(0.4)
        msg = tele.latest()
        trk = msg["tracker"]
        assert trk["clutch"] is False and trk["engaged_arm"] is None
        assert trk["anchor_tcp"] is None and trk["target_tcp"] is None
        p2 = msg["arms"][1]["ee_pose"]["position"]
        time.sleep(0.5)
        p3 = tele.latest()["arms"][1]["ee_pose"]["position"]
        assert _dist(p2, p3) < 2e-3  # stopped while the fake tracker keeps circling
        # The non-active (view) arm never moved.
        assert _dist(msg["arms"][0]["ee_pose"]["position"],
                     tele.latest()["arms"][0]["ee_pose"]["position"]) < 2e-3
    finally:
        ctl.close()
        tele.close()
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors == [], [r.getMessage() for r in errors]


def test_tracker_settings_and_switch_arm_prev_over_ws(server, api, session):
    ctl = Ctl(server)
    tele = Tele(server)
    try:
        ack = ctl.action("tracker_settings", {"pos_scale": 2.0, "yaw_deg": 30.0})
        assert ack["ok"], ack
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            s = tele.latest()["tracker"]["settings"]
            if s["pos_scale"] == 2.0:
                break
        assert s == FILTER_DEFAULTS | {"yaw_deg": 30.0, "pos_scale": 2.0, "follow_rotation": True}
        trk = tele.latest()["tracker"]  # pose_world now yaw-rotated vs pose_raw
        assert _dist(trk["pose_raw"]["position"], trk["pose_world"]["position"]) > 0.01
        ack = ctl.action("tracker_settings", {"pos_scale": 9.0})
        assert not ack["ok"] and "invalid args" in ack["detail"]
        ack = ctl.action("tracker_settings", {"pos_scale": 1.0, "yaw_deg": 0.0})
        assert ack["ok"]
        ack = ctl.action("switch_arm_prev")
        assert ack["ok"] and ack["detail"] == "view"  # (i - 1) from the default grip
        ack = ctl.action("switch_arm_prev")
        assert ack["ok"] and ack["detail"] == "grip"  # wraps
        ack = ctl.action("switch_arm_prev", {"nope": 1})
        assert not ack["ok"] and "invalid args" in ack["detail"]
    finally:
        ctl.close()
        tele.close()


# -- 13-tracker §1.1: controller buttons as device-held codes -----------------------------------
def _script_controller(server):
    """Scripted controller state for the fake backend (polled every fake tick)."""
    script = {"state": None}
    server.runtime.tracker.controller_provider = lambda: script["state"]
    return script


def _pad(x: float = 0.0, y: float = 0.0, trigger: bool = False, click: bool = True):
    """Raw scripted controller state (the reader classifies the click at its edge)."""
    return ControllerState(
        trigger=1.0 if trigger else 0.0, trigger_pressed=trigger,
        trackpad_touch=True, trackpad_click=click, trackpad_x=x, trackpad_y=y,
    )


def _wait_ee_moved(tele, p0, i: int, min_m: float = 0.03, timeout_s: float = 8.0, tick=None):
    """Latest frame once arm ``i``'s EE has moved ``min_m`` away from ``p0``.

    The fake hand circles at ~0.047 m/s, but the sim arm follows it through IK
    at a rate that depends on the posture and on which way the circle happens
    to be going when the clutch engages — over the same 2 s window the EE
    covered 24 mm on one run and 35 mm on the next, so a fixed sleep is a flaky
    way to ask "did the clutch move the arm". Wait for the distance instead: the
    hand keeps circling, so it only grows. ``tick`` is called between frames for
    the keyboard path, whose WS deadman needs a KeysMsg every < 0.2 s.
    """
    deadline = time.monotonic() + timeout_s
    msg = tele.latest()
    while _dist(p0, msg["arms"][i]["ee_pose"]["position"]) <= min_m:
        assert time.monotonic() < deadline, (
            f"EE moved {_dist(p0, msg['arms'][i]['ee_pose']['position']):.4f} m "
            f"in {timeout_s} s, wanted > {min_m}"
        )
        if tick is not None:
            tick()
        msg = tele.latest()
    return msg


def _wait_gripper(tele, i: int, pred, timeout_s: float = 4.0) -> float:
    """Arm ``i``'s ``gripper_open_frac`` once it satisfies ``pred`` (asserts).

    A held gripper code commands a RATE (1.2/s in sim), so how far the jaw has
    travelled after a fixed sleep depends on how many ticks the loop got — on a
    loaded machine 0.6 s bought 0.37 of travel where the test wanted 0.48. Wait
    for the value, not for the clock.
    """
    deadline = time.monotonic() + timeout_s
    g = tele.latest()["arms"][i]["gripper_open_frac"]
    while not pred(g):
        assert time.monotonic() < deadline, f"gripper_open_frac stuck at {g:.3f}"
        g = tele.latest()["arms"][i]["gripper_open_frac"]
    return g


def _wait_tracker(tele, pred, timeout_s=3.0):
    deadline = time.monotonic() + timeout_s
    trk = tele.latest()["tracker"]
    while not pred(trk) and time.monotonic() < deadline:
        trk = tele.latest()["tracker"]
    return trk


def _wait_active(tele, arm: str, timeout_s: float = 3.0) -> dict:
    """Latest telemetry frame once ``active_arm`` reads ``arm`` (asserts)."""
    deadline = time.monotonic() + timeout_s
    msg = tele.latest()
    while msg["active_arm"] != arm and time.monotonic() < deadline:
        msg = tele.latest()
    assert msg["active_arm"] == arm, msg["active_arm"]
    return msg


def test_controller_trigger_moves_ee_without_any_keysmsg(server, api, session, caplog):
    caplog.set_level(logging.INFO)
    script = _script_controller(server)
    tele = Tele(server)
    try:
        trk = _wait_tracker(tele, lambda t: t["status"] == "tracking")
        assert trk["controller"] is None and trk["device_held"] == []
        p0 = tele.latest()["arms"][1]["ee_pose"]["position"]  # grip: the default active arm
        script["state"] = ControllerState(trigger=1.0, trigger_pressed=True)
        trk = _wait_tracker(tele, lambda t: t["engaged_arm"] == "grip")
        assert trk["clutch"] is True and trk["engaged_arm"] == "grip", trk
        assert trk["device_held"] == ["KeyC"]
        c = trk["controller"]
        assert c["trigger"] == 1.0 and c["trigger_pressed"] is True
        assert c["trackpad_click"] is False and c["grip"] is False
        # No /ws/control client ever connected: the device-held KeyC alone drives it.
        msg = _wait_ee_moved(tele, p0, 1)
        assert msg["controller_connected"] is False
        assert msg["tracker"]["anchor_tcp"] is not None and msg["tracker"]["target_tcp"]
        script["state"] = ControllerState(trigger=0.0)  # trigger released: hold-last
        trk = _wait_tracker(tele, lambda t: t["device_held"] == [] and not t["clutch"])
        assert trk["engaged_arm"] is None and trk["anchor_tcp"] is None
        assert trk["controller"]["trigger_pressed"] is False
        time.sleep(0.4)  # let the sim arm settle on the frozen command
        p2 = tele.latest()["arms"][1]["ee_pose"]["position"]
        time.sleep(0.5)
        assert _dist(p2, tele.latest()["arms"][1]["ee_pose"]["position"]) < 2e-3
        # Trackpad click left / right / inside the deadzone -> ArrowLeft / ArrowRight /
        # nothing; each click is classified at its press edge, so release in between.
        # The rail codes drive the rail of the active (grip) arm with no /ws/control
        # client. Left first: the grip rail rests at the +q end of its travel.
        r0 = tele.latest()["arms"][1]["rail_pos_m"]
        script["state"] = _pad(x=-0.9)
        trk = _wait_tracker(tele, lambda t: t["device_held"] == ["ArrowLeft"])
        assert trk["device_held"] == ["ArrowLeft"] and trk["controller"]["trackpad_click"]
        assert trk["controller"]["trackpad_x"] == -0.9
        time.sleep(0.5)
        r1 = tele.latest()["arms"][1]["rail_pos_m"]
        assert r1 < r0 - 0.02, (r0, r1)  # 0.10 m/s while held
        script["state"] = _pad(x=-0.9, click=False)
        _wait_tracker(tele, lambda t: t["device_held"] == [])
        time.sleep(0.3)
        r2 = tele.latest()["arms"][1]["rail_pos_m"]
        script["state"] = _pad(x=0.9)
        trk = _wait_tracker(tele, lambda t: t["device_held"] == ["ArrowRight"])
        assert trk["device_held"] == ["ArrowRight"] and trk["controller"]["trackpad_x"] == 0.9
        time.sleep(0.5)
        assert tele.latest()["arms"][1]["rail_pos_m"] > r2 + 0.02
        script["state"] = _pad(x=0.9, click=False)
        _wait_tracker(tele, lambda t: t["device_held"] == [])
        script["state"] = _pad(x=0.1, y=0.1)
        time.sleep(0.3)
        trk = tele.latest()["tracker"]
        assert trk["device_held"] == [] and trk["controller"]["trackpad_click"] is True
        assert trk["controller"]["trackpad_x"] == pytest.approx(0.1)
        assert trk["device_action"] is None
    finally:
        script["state"] = None
        server.runtime.tracker.controller_provider = None
        tele.close()
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors == [], [r.getMessage() for r in errors]


def test_controller_codes_survive_ws_deadman_latch(server, api, session):
    script = _script_controller(server)
    ctl = Ctl(server)
    tele = Tele(server)
    try:
        # grip (the arm with a gripper) is the default active arm: no switch needed.
        ctl.keys(["KeyW"])  # one KeysMsg, then silence: deadman trips and latches
        watchdog = server.runtime.manager.session.supervisor.watchdog
        deadline = time.monotonic() + 2.0
        while watchdog.state.value != "await_empty" and time.monotonic() < deadline:
            time.sleep(0.02)
        assert watchdog.state.value == "await_empty"
        msg = tele.latest()
        assert msg["active_arm"] == "grip"
        p0 = msg["arms"][1]["ee_pose"]["position"]
        g0 = msg["arms"][1]["gripper_open_frac"]
        script["state"] = _pad(y=-1.0, trigger=True)  # trigger click + trackpad down
        trk = _wait_tracker(tele, lambda t: t["engaged_arm"] == "grip")
        assert sorted(trk["device_held"]) == ["KeyC", "KeyF"]
        time.sleep(1.5)  # long enough for the gripper, which closes at 1.2/s while held
        msg = _wait_ee_moved(tele, p0, 1, min_m=0.02)  # the EE rate depends on the posture
        assert watchdog.state.value == "await_empty"  # WS still latched throughout
        assert msg["arms"][1]["gripper_open_frac"] < g0 - 0.5
        script["state"] = None
        _wait_tracker(tele, lambda t: t["controller"] is None and t["device_held"] == [])
    finally:
        script["state"] = None
        server.runtime.tracker.controller_provider = None
        ctl.close()
        tele.close()


# -- 13-tracker §1.1 remap: trackpad up/down gripper, left/right rail, menu arm switch --------
def test_trackpad_click_gripper_only_on_grip_arm_and_menu_switches_arm(server, api, session):
    script = _script_controller(server)
    ctl = Ctl(server)
    tele = Tele(server)
    try:
        assert tele.latest()["active_arm"] == "grip"  # default: the Manipulation Arm
        assert ctl.action("switch_arm")["detail"] == "view"
        msg = _wait_active(tele, "view")  # camera-only arm: gripper codes are ignored
        script["state"] = _pad(click=False)  # pad touched, not clicked: controller adopted
        _wait_tracker(tele, lambda t: t["controller"] is not None)
        g_grip0 = msg["arms"][1]["gripper_open_frac"]
        script["state"] = _pad(y=-0.9)  # down -> gripper_close (held)
        trk = _wait_tracker(tele, lambda t: t["device_held"] == ["KeyF"])
        assert trk["device_action"] is None
        time.sleep(0.6)
        msg = tele.latest()
        assert msg["active_arm"] == "view"
        assert msg["arms"][1]["gripper_open_frac"] == pytest.approx(g_grip0, abs=0.01)  # no cmd
        script["state"] = _pad(y=-0.9, click=False)
        _wait_tracker(tele, lambda t: t["device_held"] == [])
        # Menu press: switch_arm fires once on the press edge -> grip arm, latched ~1 s.
        script["state"] = ControllerState(menu=True)
        trk = _wait_tracker(tele, lambda t: t["device_action"] == "switch_arm")
        assert trk["device_held"] == [] and trk["controller"]["menu"] is True
        time.sleep(0.5)
        msg = tele.latest()
        assert msg["active_arm"] == "grip" and msg["tracker"]["device_action"] == "switch_arm"
        script["state"] = ControllerState()  # menu released
        trk = _wait_tracker(tele, lambda t: t["device_action"] is None, timeout_s=2.5)
        assert trk["device_action"] is None and tele.latest()["active_arm"] == "grip"
        # Now the gripper codes act: down closes ...
        g0 = tele.latest()["arms"][1]["gripper_open_frac"]
        script["state"] = _pad(y=-0.9)
        _wait_tracker(tele, lambda t: t["device_held"] == ["KeyF"])
        g1 = _wait_gripper(tele, 1, lambda g: g < g0 - 0.4)  # 1.2/s while held
        script["state"] = _pad(y=-0.9, click=False)
        _wait_tracker(tele, lambda t: t["device_held"] == [])
        # ... and up (y dominant even with some x) opens again.
        script["state"] = _pad(x=0.4, y=0.9)
        _wait_tracker(tele, lambda t: t["device_held"] == ["KeyH"])
        # The sim gripper opens slower than it closes; the direction is the point.
        _wait_gripper(tele, 1, lambda g: g > g1 + 0.1)
        script["state"] = _pad(y=0.9, click=False)
        _wait_tracker(tele, lambda t: t["device_held"] == [])
        # Menu again -> switch_arm wraps back to view (arm_prev has no controller binding).
        script["state"] = ControllerState(menu=True)
        trk = _wait_tracker(tele, lambda t: t["device_action"] == "switch_arm")
        time.sleep(0.2)
        assert tele.latest()["active_arm"] == "view"
        script["state"] = None
        _wait_tracker(tele, lambda t: t["controller"] is None)
    finally:
        script["state"] = None
        server.runtime.tracker.controller_provider = None
        ctl.close()
        tele.close()


def test_filter_settings_echoed_and_pose_filtered_in_session(server, api, session):
    ctl = Ctl(server)
    tele = Tele(server)
    try:
        trk = _wait_tracker(tele, lambda t: t["pose_filtered"] is not None)
        assert trk["settings"] == FILTER_DEFAULTS | {
            "yaw_deg": 0.0, "pos_scale": 1.0, "follow_rotation": True
        }
        # Free-running filter while the clutch is up: filtered pose trails the moving fake.
        assert _dist(trk["pose_filtered"]["position"], trk["pose_world"]["position"]) < 0.05
        ack = ctl.action(
            "tracker_settings",
            {"filter_min_cutoff_hz": 5.0, "filter_beta": 0.5, "filter_enabled": False},
        )
        assert ack["ok"] and "filter_min_cutoff_hz=5" in ack["detail"], ack
        trk = _wait_tracker(tele, lambda t: t["settings"]["filter_enabled"] is False)
        assert trk["settings"]["filter_min_cutoff_hz"] == 5.0
        assert trk["settings"]["filter_beta"] == 0.5 and trk["settings"]["pos_scale"] == 1.0
        trk = tele.latest()["tracker"]  # passthrough: filtered == world pose
        assert _dist(trk["pose_filtered"]["position"], trk["pose_world"]["position"]) < 2e-3
        ack = ctl.action("tracker_settings", {"filter_min_cutoff_hz": 0.0})
        assert not ack["ok"] and "invalid args" in ack["detail"]
        ack = ctl.action(
            "tracker_settings",
            {"filter_min_cutoff_hz": 1.0, "filter_beta": 5.0, "filter_enabled": True},
        )
        assert ack["ok"]
        _wait_tracker(tele, lambda t: t["settings"] == FILTER_DEFAULTS | {
            "yaw_deg": 0.0, "pos_scale": 1.0, "follow_rotation": True
        })
    finally:
        ctl.close()
        tele.close()
