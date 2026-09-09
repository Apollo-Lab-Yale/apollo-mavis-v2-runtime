"""Return-to-start after save / discard (D6, DEFAULT ON; 04-runtime §10.5) over a real
sim server: 409 with neither a start_from nor an initial-condition profile;
``return_to_start: false`` -> no motion; ``start_from: profile`` -> saving -> returning
-> idle back to the profile (every joint < 0.02 rad), episode_new nacked meanwhile,
discard returns too, a KeyW cancels (detail "cancelled", arm holds); with only an
initial-condition profile the arm returns to it."""

from __future__ import annotations

import time

import httpx
import pytest
from conftest import LiveServer, make_runtime_config
from test_e2e_teleop import Ctl, PulsingCtl, Tele

TASK = "return to start e2e"
BASE = {
    "kind": "sim",
    "arms": ["arm0"],
    "frames": {"arm0": "arm_base:arm0"},
    "sim_scene": "guardrail_env",
}
NEEDS_PROFILE = "return_to_start needs a start_from profile or an initial-condition profile"


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    srv = LiveServer(make_runtime_config(tmp_path_factory.mktemp("rt"), scene="guardrail_env"))
    yield srv
    srv.stop()


@pytest.fixture(scope="module")
def api(server):
    with httpx.Client(base_url=server.http, timeout=120.0) as client:
        yield client


def collect(dataset: str, **over) -> dict:
    return {**BASE, "mode": "collect", "task": TASK, "dataset": dataset, **over}


def wait_running(api) -> None:
    for _ in range(400):
        st = api.get("/api/session").json()
        if st["state"] == "running":
            return
        time.sleep(0.05)
    raise AssertionError("session never reached running")


def settle(tele: Tele, api, timeout: float = 30.0) -> None:
    """Wait for START_FROM to finish (running + no plan executing)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        msg = tele.latest()
        session = msg.get("session") or {}
        if session.get("state") == "running" and session.get("plan_status") not in (
            "planning", "executing",
        ):
            time.sleep(0.3)
            return
        time.sleep(0.05)
    raise AssertionError("start_from never settled")


def watch_until_idle(tele: Tele, timeout: float = 20.0) -> tuple[list[str], dict]:
    """Every distinct episode state seen until idle, plus the idle frame."""
    seen: list[str] = []
    t0 = time.monotonic()
    while True:
        msg = tele.latest()
        st = msg["episode"]["state"]
        if not seen or seen[-1] != st:
            seen.append(st)
        if st == "idle" and len(seen) > 1:
            return seen, msg
        assert time.monotonic() - t0 < timeout, seen


def max_joint_err(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) for x, y in zip(a[:7], b[:7], strict=False))


def drive(ctl: Ctl, keys: list[str], duration_s: float) -> None:
    """Hold ``keys`` for ``duration_s`` (the PulsingCtl heartbeat carries them at 25 Hz)."""
    ctl.hold(keys, duration_s)


def new_episode(ctl: Ctl) -> None:
    """episode_new, then let the recorder's first frame open the video encoder before any
    key is held: the NVENC session start stalls the process for a few hundred ms, which
    would trip the 0.2 s WS deadman under a held key and silently ignore the hold."""
    assert ctl.action("episode_new")["ok"]
    time.sleep(0.8)


def test_409_without_any_return_profile(api):
    r = api.post("/api/session", json=collect("rts_none"))  # default return_to_start: True
    assert r.status_code == 409, r.text
    assert NEEDS_PROFILE in r.json()["detail"] and "untick" in r.json()["detail"]
    assert api.get("/api/session").status_code == 404
    assert api.get("/api/datasets").json() == []  # refused before anything was written


def test_unticked_session_never_moves_after_save(server, api):
    r = api.post("/api/session", json=collect("rts_off", return_to_start=False))
    assert r.status_code == 200, r.text
    wait_running(api)
    ctl, tele = PulsingCtl(server), Tele(server)
    try:
        new_episode(ctl)
        drive(ctl, ["KeyW"], 1.0)
        time.sleep(1.0)  # let the sim servo settle before taking the reference
        q_before = tele.q_full()
        assert ctl.action("episode_save")["ok"]
        seen, idle = watch_until_idle(tele)
        assert "returning" not in seen and idle["episode"]["detail"] == ""
        time.sleep(0.8)
        assert max_joint_err(tele.q_full(), q_before) < 5e-3  # nothing moved (settle noise only)
    finally:
        ctl.close()
        tele.close()
        api.delete("/api/session")


@pytest.fixture(scope="module")
def profiles(server, api) -> dict:
    """A teleop session makes two postures: 'ready' (a plain profile) and the
    designated initial condition, at different places."""
    r = api.post("/api/session", json={**BASE, "mode": "teleop"})
    assert r.status_code == 200, r.text
    wait_running(api)
    ctl, tele = PulsingCtl(server), Tele(server)
    try:
        drive(ctl, ["KeyE", "KeyA"], 1.2)  # up + left: away from the pedestal
        time.sleep(0.5)
        q_ready = tele.q_full()
        ack = ctl.action("save_profile", {"name": "ready"})
        assert ack["ok"], ack
        ready_id = ack["detail"]
        drive(ctl, ["KeyD", "KeyQ"], 1.0)  # right + down (back toward the start)
        time.sleep(0.5)
        q_initial = tele.q_full()
        ack = ctl.action("set_initial_condition")
        assert ack["ok"], ack
        initial_id = ack["detail"]
        assert max_joint_err(q_ready, q_initial) > 0.05  # the two postures differ
    finally:
        ctl.close()
        tele.close()
        api.delete("/api/session")
    rows = {p["profile_id"]: p for p in api.get("/api/profiles").json()}
    assert rows[initial_id]["is_initial_condition"] and not rows[ready_id]["is_initial_condition"]
    return {"ready": ready_id, "q_ready": q_ready, "initial": initial_id, "q_initial": q_initial}


def test_start_from_profile_returns_after_save_discard_and_cancels_on_input(server, api, profiles):
    spec = collect("rts_ready", start_from=f"profile:{profiles['ready']}")  # default flag ON
    r = api.post("/api/session", json=spec)
    assert r.status_code == 200, r.text
    ctl, tele = PulsingCtl(server), Tele(server)
    try:
        settle(tele, api)
        q_ready = profiles["q_ready"]
        assert max_joint_err(tele.q_full(), q_ready) < 0.02  # start_from put us there

        # -- save: saving -> returning -> idle, back at the profile ------------------
        new_episode(ctl)
        drive(ctl, ["KeyE", "KeyA"], 1.5)
        time.sleep(0.3)
        assert max_joint_err(tele.q_full(), q_ready) > 0.05  # we moved away
        assert ctl.action("episode_save")["ok"]
        nacked = None
        seen: list[str] = []
        t0 = time.monotonic()
        while True:
            msg = tele.latest()
            st = msg["episode"]["state"]
            if not seen or seen[-1] != st:
                seen.append(st)
            if st == "returning" and nacked is None:
                nacked = ctl.action("episode_new")  # refused while returning
                assert msg["episode"]["detail"].startswith("returning to profile 'ready'")
            if st == "idle" and len(seen) > 1:
                break
            assert time.monotonic() - t0 < 30.0, seen
        # the first frame after the ack may still read 'recording' (or even 'idle' + a note):
        # strip everything before the first 'saving' and assert the exact suffix
        assert "saving" in seen, seen
        assert seen[seen.index("saving"):] == ["saving", "returning", "idle"], seen
        assert nacked is not None and nacked["ok"] is False
        assert nacked["detail"] == "returning to the initial configuration"
        time.sleep(0.4)
        assert max_joint_err(tele.q_full(), q_ready) < 0.02
        assert msg["episode"]["detail"] == "" and msg["episode"]["total_episodes"] == 1
        new_episode(ctl)  # idle again: allowed

        # -- discard: idle -> returning -> idle -------------------------------------
        drive(ctl, ["KeyE"], 1.2)
        assert ctl.action("episode_discard")["ok"]
        seen, msg = watch_until_idle(tele)
        assert "returning" in seen, seen
        time.sleep(0.4)
        assert max_joint_err(tele.q_full(), q_ready) < 0.02
        assert msg["episode"]["total_episodes"] == 1  # discarded: not saved

        # -- cancel: a movement key during the return holds the arm where it is ------
        new_episode(ctl)
        drive(ctl, ["KeyE", "KeyA"], 1.5)
        time.sleep(0.2)
        assert ctl.action("episode_save")["ok"]
        t0 = time.monotonic()
        while tele.latest()["episode"]["state"] != "returning":
            assert time.monotonic() - t0 < 20.0
        drive(ctl, ["KeyE"], 0.15)  # one keyboard input -> cancel
        # the cancel is immediate (the arm just stops), so idle may already be on the wire
        t0 = time.monotonic()
        while (msg := tele.latest())["episode"]["state"] != "idle":
            assert time.monotonic() - t0 < 20.0, msg["episode"]
        assert "cancelled" in msg["episode"]["detail"], msg["episode"]
        assert msg["episode"]["detail"] == "return cancelled: movement key"  # the loop's reason
        time.sleep(1.0)  # settle
        q_stop = tele.q_full()
        d0 = max_joint_err(q_stop, q_ready)
        assert d0 > 0.02  # did NOT complete the return
        time.sleep(1.0)
        d1 = max_joint_err(tele.q_full(), q_ready)
        assert d1 >= d0 - 5e-3  # the return did NOT continue: the distance never shrinks
        assert max_joint_err(tele.q_full(), q_stop) < 5e-3  # ... the arm holds
        assert not server.runtime.manager.session.loop.plans.active_arms
        assert msg["episode"]["total_episodes"] == 2
    finally:
        ctl.close()
        tele.close()
        api.delete("/api/session")
    import json

    root = server.runtime.cfg.datasets_root / "apollo" / "rts_ready"
    eps = sorted((root / "episodes").glob("*/episode.json"))
    assert len(eps) == 2
    assert json.loads(eps[0].read_text())["return_profile_id"] == profiles["ready"]


def test_initial_condition_profile_is_the_fallback_return_target(server, api, profiles):
    r = api.post("/api/session", json=collect("rts_initial"))  # keep_current, flag ON
    assert r.status_code == 200, r.text
    wait_running(api)
    ctl, tele = PulsingCtl(server), Tele(server)
    try:
        q_initial = profiles["q_initial"]
        assert max_joint_err(tele.q_full(), q_initial) > 0.02  # not there yet
        new_episode(ctl)
        drive(ctl, ["KeyE"], 1.0)
        assert ctl.action("episode_save")["ok"]
        seen, msg = watch_until_idle(tele)
        assert "returning" in seen, seen
        time.sleep(0.4)
        assert max_joint_err(tele.q_full(), q_initial) < 0.02
    finally:
        ctl.close()
        tele.close()
        api.delete("/api/session")
    import json

    root = server.runtime.cfg.datasets_root / "apollo" / "rts_initial"
    ep = json.loads(next((root / "episodes").glob("*/episode.json")).read_text())
    assert ep["return_profile_id"] == profiles["initial"]
    assert ep["initial_condition_profile_id"] == profiles["initial"]
