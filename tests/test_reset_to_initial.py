"""Return to the initial condition — the ``R`` key and the Cockpit's exit
(04-runtime §6/§10.5; 2026-09-08 operator request).

Unit half: ``ControlLoop._op_reset_to_initial`` is a NO-OP with a reason when there
is nothing to return to, refuses while an episode records or a plan runs, and
otherwise hands the motion to the SessionManager hook without blocking the loop.

E2E half over a real sim server: ``POST /api/session/return_home`` answers
``skipped`` (ok) with no initial condition designated, walks the arm back to the
designated one, runs the CARRIAGE as a SECOND phase after the joints, is cancelled
by a movement key, and refuses while an episode is open. The ``R`` key fires the
same motion over /ws/control.
"""

from __future__ import annotations

import time

import httpx
import pytest
from apollo_mavis_v2_core import ArmPosture, Command, StateProfile
from conftest import LiveServer, make_runtime_config
from test_e2e_teleop import PulsingCtl, Tele

# -- unit: the loop's inline validation ------------------------------------------------


def _ack(loop, bus, **args):
    fut = bus.commands.submit(Command(op="reset_to_initial", args=args, source="ws"))
    loop.run_tick()
    return fut.result(timeout=1.0)


def test_no_initial_condition_is_a_noop_with_a_reason(fake_loop):
    cell, bus, loop = fake_loop
    loop.run_tick()
    called: list[object] = []
    loop.on_reset_to_initial = lambda profile: (called.append(profile), (True, "moving"))[1]
    ack = _ack(loop, bus)
    assert ack.ok is False
    assert "no initial condition designated for the sim workcell" in ack.detail
    assert called == []  # nothing was asked to move
    assert not loop.plans.active_arms


def test_hands_the_motion_to_the_manager_hook(fake_loop):
    cell, bus, loop = fake_loop
    loop.run_tick()
    profile = loop.profile_store.save(
        StateProfile(
            name="home",
            workcell_kind="sim",
            arms={"arm0": ArmPosture(q=[0.0] * 7, rail_pos_m=0.1)},
        )
    )
    loop.profile_store.set_initial(profile.profile_id)
    calls: list[object] = []

    def hook(target):
        calls.append(target)
        return True, "returning to 'home'"

    loop.on_reset_to_initial = hook
    ack = _ack(loop, bus)
    assert (ack.ok, ack.detail) == (True, "returning to 'home'")
    # The loop resolves the profile ONCE and hands it over, so the manager never
    # re-scans the store from the 100 Hz thread.
    assert [p.profile_id for p in calls] == [profile.profile_id]


def test_refused_while_recording_and_without_a_manager(fake_loop):
    cell, bus, loop = fake_loop
    loop.run_tick()
    profile = loop.profile_store.save(
        StateProfile(name="home", workcell_kind="sim", arms={"arm0": ArmPosture(q=[0.0] * 7)})
    )
    loop.profile_store.set_initial(profile.profile_id)
    loop.on_reset_to_initial = lambda profile: (True, "moving")
    loop.episode_state = "recording"
    ack = _ack(loop, bus)
    assert ack.ok is False and "recording" in ack.detail
    loop.episode_state = "idle"
    # No manager attached (a bare unit loop): refused rather than silently ignored.
    loop.on_reset_to_initial = None
    ack = _ack(loop, bus)
    assert ack.ok is False and "no session manager" in ack.detail


# -- e2e over a real sim server -------------------------------------------------------

BASE = {
    "kind": "sim",
    "arms": ["arm0"],
    "frames": {"arm0": "arm_base:arm0"},
    "sim_scene": "guardrail_env",
}


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    srv = LiveServer(make_runtime_config(tmp_path_factory.mktemp("rt"), scene="guardrail_env"))
    yield srv
    srv.stop()


@pytest.fixture(scope="module")
def api(server):
    with httpx.Client(base_url=server.http, timeout=200.0) as client:
        yield client


def wait_running(api) -> None:
    for _ in range(400):
        if api.get("/api/session").json()["state"] == "running":
            return
        time.sleep(0.05)
    raise AssertionError("session never reached running")


def max_joint_err(a, b) -> float:
    return max(abs(x - y) for x, y in zip(a[:7], b[:7], strict=False))


def return_home(api) -> dict:
    r = api.post("/api/session/return_home")
    assert r.status_code == 200, r.text
    return r.json()


def test_without_a_session_it_refuses(api):
    assert api.get("/api/session").status_code == 404
    res = return_home(api)
    assert res == {
        "ok": False,
        "status": "refused",
        "detail": "no active session",
        "arms": [],
        "profile_id": None,
    }


def test_skipped_when_no_initial_condition_is_designated(server, api):
    assert api.post("/api/session", json={**BASE, "mode": "teleop"}).status_code == 200
    wait_running(api)
    try:
        res = return_home(api)
        assert res["ok"] is True and res["status"] == "skipped"
        assert "no initial condition designated for the sim workcell" in res["detail"]
        assert res["profile_id"] is None
    finally:
        api.delete("/api/session")


def test_returns_the_joints_then_the_carriage_and_is_cancelled_by_a_key(server, api):
    assert api.post("/api/session", json={**BASE, "mode": "teleop"}).status_code == 200
    wait_running(api)
    ctl, tele = PulsingCtl(server), Tele(server)
    try:
        # Designate the START posture (joints AND carriage) as the initial condition.
        ack = ctl.action("save_profile", {"name": "home", "set_initial": True})
        assert ack["ok"], ack
        q_home = tele.q_full()
        assert api.get("/api/profiles").json()[0]["is_initial_condition"] is True

        # Already there -> "skipped", still a success and no motion.
        res = return_home(api)
        assert (res["ok"], res["status"]) == (True, "skipped")
        assert res["detail"] == "already at the initial condition"

        # Move BOTH the joints and the carriage away, then walk back.
        ctl.hold(["KeyE", "ArrowRight"], 1.5)
        time.sleep(0.5)
        q_away = tele.q_full()
        assert max_joint_err(q_away, q_home) > 0.05
        assert abs(q_away[7] - q_home[7]) > 0.02  # the rail moved too
        res = return_home(api)
        assert res == {
            "ok": True,
            "status": "done",
            "detail": "",
            "arms": ["arm0"],
            "profile_id": api.get("/api/profiles").json()[0]["profile_id"],
        }
        time.sleep(0.5)
        q_back = tele.q_full()
        assert max_joint_err(q_back, q_home) < 0.02
        assert abs(q_back[7] - q_home[7]) < 2e-3  # the carriage phase ran as well

        # A movement key during the motion cancels it: the arm holds part-way. Move
        # far enough that the walk back lasts long enough to interrupt.
        ctl.hold(["KeyE", "ArrowRight"], 3.0)
        time.sleep(0.4)

        import threading

        out: dict = {}

        def call() -> None:
            out["res"] = return_home(api)

        worker = threading.Thread(target=call, daemon=True)
        worker.start()
        deadline = time.monotonic() + 30.0  # wait for the plan to be EXECUTING
        while (tele.latest().get("session") or {}).get("plan_status") != "executing":
            assert time.monotonic() < deadline, "the return never started executing"
            time.sleep(0.02)
        ctl.hold(["KeyQ"], 0.2)  # one keyboard input -> cancel
        worker.join(timeout=120)
        assert not worker.is_alive()
        res = out["res"]
        assert res["ok"] is False and res["status"] == "cancelled", res
        assert "cancelled" in res["detail"] and "hold where they are" in res["detail"]
        time.sleep(1.0)
        q_stop = tele.q_full()
        assert max_joint_err(q_stop, q_home) > 0.02  # did NOT complete
        time.sleep(0.7)
        assert max_joint_err(tele.q_full(), q_stop) < 5e-3  # ... and holds
        assert not server.runtime.manager.session.loop.plans.active_arms
    finally:
        ctl.close()
        tele.close()
        api.delete("/api/session")


def test_the_r_key_fires_the_same_motion_and_is_refused_while_recording(server, api):
    spec = {**BASE, "mode": "collect", "task": "reset key e2e", "dataset": "reset_key",
            "return_to_start": False}
    assert api.post("/api/session", json=spec).status_code == 200, "collect session"
    wait_running(api)
    ctl, tele = PulsingCtl(server), Tele(server)
    try:
        assert ctl.action("save_profile", {"name": "home2", "set_initial": True})["ok"]
        q_home = tele.q_full()
        ctl.hold(["KeyE"], 1.2)
        time.sleep(0.4)
        assert max_joint_err(tele.q_full(), q_home) > 0.05

        # The key: an immediate ack that only says the motion STARTED.
        ack = ctl.action("reset_to_initial")
        assert ack["ok"] is True and ack["detail"] == "returning to 'home2'", ack
        deadline = time.monotonic() + 60.0
        while max_joint_err(tele.q_full(), q_home) > 0.02:
            assert time.monotonic() < deadline, "the R key never brought the arm back"
            time.sleep(0.05)

        # While an episode is open BOTH entry points refuse and nothing moves.
        assert ctl.action("episode_new")["ok"]
        time.sleep(0.8)  # let the encoder open before holding a key
        q_rec = tele.q_full()
        ack = ctl.action("reset_to_initial")
        assert ack["ok"] is False and "recording" in ack["detail"], ack
        res = return_home(api)
        assert res["ok"] is False and res["status"] == "refused"
        assert "still recording" in res["detail"], res
        assert max_joint_err(tele.q_full(), q_rec) < 5e-3
        assert ctl.action("episode_discard")["ok"]
    finally:
        ctl.close()
        tele.close()
        api.delete("/api/session")


def test_a_profile_covering_no_session_arm_is_skipped(server, api):
    store = None
    assert api.post("/api/session", json={**BASE, "mode": "teleop"}).status_code == 200
    wait_running(api)
    try:
        store = server.runtime.manager.profile_store
        other = store.save(
            StateProfile(
                name="other cell",
                workcell_kind="sim",
                arms={"nosucharm": ArmPosture(q=[0.0] * 7)},
            )
        )
        store.set_initial(other.profile_id)
        res = return_home(api)
        assert res["ok"] is True and res["status"] == "skipped"
        assert "covers no arm of this session" in res["detail"]
        assert res["arms"] == []
    finally:
        api.delete("/api/session")
        if store is not None:  # leave the module store as the other tests found it
            for p in store.list():
                if p.is_initial_condition and "other" in p.name:
                    store._path(p.profile_id).unlink()


# -- the seeded default posture (2026-09-08 operator numbers) --------------------------


def test_seeded_default_posture_matches_the_operator_degrees(tmp_path):
    """``profiles.seed_initial`` writes exactly the degrees the operator gave, in rad,
    with the carriages left unset, and designates one initial condition per kind."""
    import math

    from apollo_mavis_v2_core import ProfileStore

    from apollo_mavis_v2_runtime.profiles.seed_initial import (
        DEFAULT_POSTURE_DEG,
        PROFILE_NAME,
        seed,
    )

    assert DEFAULT_POSTURE_DEG == {
        "grip": [-180.0, -12.0, -20.0, 30.0, -5.0, 35.0, -8.9],
        "view": [0.0, 0.8, 0.0, 28.9, 0.0, 28.2, 0.0],
    }
    store = ProfileStore(tmp_path / "profiles")
    for kind in ("hardware", "sim"):
        profile = seed(store, kind)
        assert profile.is_initial_condition and profile.workcell_kind == kind
        assert set(profile.arms) == {"grip", "view"}
        for arm_id, deg in DEFAULT_POSTURE_DEG.items():
            posture = profile.arms[arm_id]
            assert posture.q == pytest.approx([math.radians(d) for d in deg])
            assert posture.rail_pos_m is None  # keep the carriage
            assert posture.gripper_open_frac == 1.0
        assert store.initial_for(kind).profile_id == profile.profile_id  # type: ignore[union-attr]
    # Re-running rewrites the same two files instead of accumulating.
    before = {p.profile_id for p in store.list()}
    for kind in ("hardware", "sim"):
        seed(store, kind)
    assert {p.profile_id for p in store.list()} == before
    assert len([p for p in store.list() if p.name == PROFILE_NAME]) == 2


def test_the_seeded_posture_is_collision_free_and_plannable_in_the_twin():
    """The whole point of a default posture: the twin must accept it (else the R key
    and the exit return would always end in the "adjust it from Studio" dialog).
    Checked at both carriage ends and with the microphone body ON (hardware twin)."""
    import math

    mujoco = pytest.importorskip("mujoco")  # noqa: F841
    sim = pytest.importorskip("apollo_mavis_v2_sim")
    from apollo_mavis_v2_core import PlanRequest
    from apollo_mavis_v2_sim.scenes.descriptor import SceneOverrides

    from apollo_mavis_v2_runtime.profiles.seed_initial import DEFAULT_POSTURE_DEG

    q = {a: [math.radians(d) for d in deg] for a, deg in DEFAULT_POSTURE_DEG.items()}
    for mic in (False, True):
        scene = sim.REGISTRY.build(
            "mavis_v2", SceneOverrides(microphones={"view": mic}) if mic else None
        )
        twin = sim.DigitalTwin(scene)
        for rails in ({"grip": 0.65, "view": 0.0}, {"grip": 0.0, "view": 0.65}):
            report = twin.check({a: [*q[a], rails[a]] for a in q})
            assert not report.blocked, (mic, rails, report.pairs)
        # ... and reachable from the cell's start state by the same planner the
        # return uses.
        kf = {a: list(scene.model.key_qpos[0][twin.addr[a].qpos_adr]) for a in q}
        result = twin.plan(
            PlanRequest(
                q_start=kf,
                q_goal={a: [*q[a], float(kf[a][7])] for a in q},
            )
        )
        assert result.ok, (mic, result.failure, result.failing_pair)
