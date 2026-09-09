"""Sim-backed e2e over a REAL uvicorn server: teleop, watchdog, joint panel,
profiles, start_from (验收标准 items)."""

from __future__ import annotations

import json
import threading
import time

import httpx
import pytest
from conftest import LiveServer, make_runtime_config
from websockets.sync.client import connect as ws_connect

SPEC = {
    "mode": "teleop",
    "kind": "sim",
    "arms": ["arm0"],
    "frames": {"arm0": "arm_base:arm0"},
    "sim_scene": "single_rail",
}


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    srv = LiveServer(make_runtime_config(tmp_path_factory.mktemp("rt")))
    yield srv
    srv.stop()


@pytest.fixture(scope="module")
def api(server):
    with httpx.Client(base_url=server.http, timeout=30.0) as client:
        yield client


class Ctl:
    """Controller WS helper: seq bookkeeping + heartbeats + action acks."""

    def __init__(self, server):
        self.sock = ws_connect(f"{server.ws}/ws/control")
        self.hello = json.loads(self.sock.recv(timeout=5))
        self.seq = 0

    def keys(self, held: list[str]) -> None:
        self.seq += 1
        self.sock.send(json.dumps(
            {"t": "keys", "seq": self.seq, "ts": time.time(), "held": held}
        ))

    def hold(self, held: list[str], duration_s: float) -> None:
        """Heartbeat a held set at 25 Hz for duration_s."""
        end = time.monotonic() + duration_s
        while time.monotonic() < end:
            self.keys(held)
            time.sleep(0.04)

    def action(self, name: str, args: dict | None = None) -> dict:
        self.sock.send(json.dumps({"t": "action", "name": name, "args": args or {}}))
        ack = json.loads(self.sock.recv(timeout=10))
        assert ack["t"] == "ack" and ack["name"] == name, ack
        return ack

    def close(self) -> None:
        self.sock.close()


class PulsingCtl(Ctl):
    """``Ctl`` + the Cockpit's 25 Hz heartbeat: the real UI streams its (possibly
    empty) held set continuously from the hello on, so the runtime's deadman never
    trips between two key presses and a return-to-start is never skipped as
    "browser input latched". One lock serialises the socket between the heartbeat
    thread and the test's actions."""

    def __init__(self, server) -> None:
        super().__init__(server)
        self._lock = threading.Lock()
        self._held: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._beat, name="ctl-heartbeat", daemon=True)
        self._thread.start()

    def _beat(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                try:
                    super().keys(list(self._held))
                except Exception:  # noqa: BLE001 - socket closed by the test
                    return
            self._stop.wait(0.04)

    def keys(self, held: list[str]) -> None:  # type: ignore[override]
        with self._lock:
            self._held = list(held)
            super().keys(list(held))

    def hold(self, held: list[str], duration_s: float) -> None:  # type: ignore[override]
        self.keys(held)
        time.sleep(duration_s)
        self.keys([])

    def action(self, name: str, args: dict | None = None) -> dict:  # type: ignore[override]
        with self._lock:
            return super().action(name, args)

    def close(self) -> None:  # type: ignore[override]
        self._stop.set()
        self._thread.join(timeout=1.0)
        super().close()


class Tele:
    def __init__(self, server):
        self.sock = ws_connect(f"{server.ws}/ws/telemetry")

    def latest(self) -> dict:
        """The first frame PRODUCED after this call (``ts`` is the in-process server's
        monotonic clock): a drain-with-tiny-timeout returned stale backlog frames under
        GIL contention (2026-09-07), which read as "the arm never moved"."""
        t0 = time.monotonic()
        deadline = t0 + 5.0
        while True:
            msg = json.loads(self.sock.recv(timeout=max(0.01, deadline - time.monotonic())))
            if msg.get("ts", 0.0) >= t0 or time.monotonic() > deadline:
                return msg

    def ee_x(self) -> float:
        return self.latest()["arms"][0]["ee_pose"]["position"][0]

    def q_full(self) -> list[float]:
        arm = self.latest()["arms"][0]
        return list(arm["q"]) + [arm["rail_pos_m"]]

    def close(self) -> None:
        self.sock.close()


@pytest.fixture()
def session(api):
    r = api.post("/api/session", json=SPEC)
    assert r.status_code == 200, r.text
    for _ in range(100):  # wait past START_FROM
        if api.get("/api/session").json()["state"] == "running":
            break
        time.sleep(0.05)
    yield r.json()
    api.delete("/api/session")


def test_teleop_keyw_moves_ee_then_stops(server, api, session):
    ctl = Ctl(server)
    tele = Tele(server)
    try:
        assert ctl.hello["role"] == "controller" and ctl.hello["epoch"]
        x0 = tele.ee_x()
        samples = [x0]
        end = time.monotonic() + 2.0
        while time.monotonic() < end:  # KeysMsg + 25 Hz heartbeat
            ctl.keys(["KeyW"])
            time.sleep(0.04)
            samples.append(tele.ee_x())
        assert samples[-1] - samples[0] > 0.10  # ~0.12 m/s for 2 s
        diffs = [b - a for a, b in zip(samples, samples[1:], strict=False)]
        # Monotonic increase, modulo sub-mm physics-servo jitter.
        assert all(d > -2e-3 for d in diffs), min(diffs)
        ctl.keys([])  # release
        time.sleep(0.3)
        x1 = tele.ee_x()
        time.sleep(0.5)
        assert abs(tele.ee_x() - x1) < 2e-3  # stopped
    finally:
        ctl.close()
        tele.close()


def test_telemetry_rate_in_band(server, session):
    sock = ws_connect(f"{server.ws}/ws/telemetry")
    try:
        sock.recv(timeout=5)
        n = 0
        t0 = time.monotonic()
        while time.monotonic() - t0 < 2.0:
            sock.recv(timeout=1)
            n += 1
        rate = n / (time.monotonic() - t0)
        assert 20 <= rate <= 30, rate
    finally:
        sock.close()


def test_watchdog_deadman_ramp_and_empty_held_resume(server, api, session):
    ctl = Ctl(server)
    tele = Tele(server)
    try:
        ctl.hold(["KeyW"], 0.5)
        moving_x = tele.ee_x()
        # Heartbeats STOP with the key still held: deadman at 0.2 s, ramp 0.1 s.
        time.sleep(0.5)
        x_after_trip = tele.ee_x()
        time.sleep(0.5)
        x_later = tele.ee_x()
        assert abs(x_later - x_after_trip) < 2e-3  # fully stopped
        assert x_after_trip - moving_x < 0.12 * 0.35  # stop within ~0.3 s budget
        # Heartbeats resume but held stays non-empty: NO motion (AWAIT_EMPTY).
        ctl.hold(["KeyW"], 0.6)
        assert abs(tele.ee_x() - x_later) < 2e-3
        # One empty held set, then press again: motion resumes.
        ctl.keys([])
        time.sleep(0.05)
        ctl.hold(["KeyW"], 0.6)
        assert tele.ee_x() - x_later > 0.03
        ctl.keys([])
    finally:
        ctl.close()
        tele.close()


def test_joint_panel_jog_goto_and_cancel(server, api, session):
    ctl = Ctl(server)
    tele = Tele(server)
    try:
        q = tele.q_full()
        # jog: small delta, slew-limited to target.
        target = list(q)
        target[3] += 0.10
        ack = ctl.action("joint_target",
                         {"arm_id": "arm0", "positions": target, "mode": "jog"})
        assert ack["ok"], ack
        time.sleep(0.8)
        assert abs(tele.q_full()[3] - target[3]) < 0.02
        # A jog of ANY size is accepted since 2026-09-07 (the 0.15 rad `goto_threshold_rad`
        # nack is gone): `JogState.step` walks from the last commanded q at
        # `jog.slew_rad_per_tick`, so a big delta is simply a longer constant-speed move and
        # every intermediate posture still goes through the gate. The per-tick rate is pinned
        # in test_loop_units.py; here only the ack and the arrival matter (no timing window).
        # Joint 6, so the goto leg below still has joint 2 to itself.
        lo5, hi5 = api.get("/api/workcell").json()["arms"][0]["joint_limits"][5]
        big_jog = list(tele.q_full())
        big_jog[5] = min(hi5 - 0.05, max(lo5 + 0.05, big_jog[5] + 0.4))
        assert abs(big_jog[5] - tele.q_full()[5]) > 0.3  # the point: far past the old threshold
        ack = ctl.action("joint_target",
                         {"arm_id": "arm0", "positions": big_jog, "mode": "jog"})
        assert ack["ok"] and ack["detail"] == "jog", ack
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and abs(tele.q_full()[5] - big_jog[5]) > 0.02:
            time.sleep(0.02)
        assert abs(tele.q_full()[5] - big_jog[5]) < 0.02  # walked all the way, no nack, no jump
        # Same-size delta on joint 2 as a goto: accepted; runs via the twin planner.
        big = list(tele.q_full())
        big[1] += 0.4
        ack = ctl.action("joint_target",
                         {"arm_id": "arm0", "positions": big, "mode": "goto"})
        assert ack["ok"] and ack["detail"] == "accepted"
        seen = set()
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            msg = tele.latest()
            status = (msg.get("session") or {}).get("plan_status")
            if status:
                seen.add(status)
            if status == "done":
                break
            time.sleep(0.02)
        assert "done" in seen, seen
        assert seen & {"planning", "executing"}, seen
        time.sleep(0.5)  # "done" = final waypoint COMMANDED; let the servo settle
        assert abs(tele.q_full()[1] - big[1]) < 0.02
        # goto again, then hold a movement key mid-plan: decelerating cancel.
        far = list(tele.q_full())
        far[1] -= 0.4
        far[7] = min(0.6, far[7] + 0.3)  # long rail leg: 2 mm/tick
        ack = ctl.action("joint_target",
                         {"arm_id": "arm0", "positions": far, "mode": "goto"})
        assert ack["ok"]
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if (tele.latest().get("session") or {}).get("plan_status") == "executing":
                break
        ctl.hold(["KeyW"], 0.3)
        ctl.keys([])
        msg = tele.latest()
        assert (msg.get("session") or {}).get("plan_status") in ("cancelled", None)
        assert abs(tele.q_full()[7] - far[7]) > 0.05  # stopped short of the goal
    finally:
        ctl.close()
        tele.close()


def test_profile_flows_and_start_from(server, api):
    r = api.post("/api/session", json=SPEC)
    assert r.status_code == 200, r.text
    ctl = Ctl(server)
    tele = Tele(server)
    try:
        # Drive somewhere non-home, save it as a profile.
        ctl.hold(["KeyW", "KeyE"], 1.0)
        ctl.keys([])
        time.sleep(0.3)
        q_saved = tele.q_full()
        ack = ctl.action("save_profile", {"name": "wide", "notes": "e2e"})
        assert ack["ok"] and ack["detail"]
        profile_id = ack["detail"]
        # set_initial_condition with NO args: saves "initial" and designates it.
        ack1 = ctl.action("set_initial_condition")
        assert ack1["ok"]
        initial_id = ack1["detail"]
        ack2 = ctl.action("set_initial_condition")  # repeat OVERWRITES same profile
        assert ack2["ok"] and ack2["detail"] == initial_id
        rows = api.get("/api/profiles").json()
        initials = [p for p in rows if p["is_initial_condition"]]
        assert [p["profile_id"] for p in initials] == [initial_id]
        assert all(p["workcell_kind"] == "sim" for p in rows)  # 2026-09-07: kind on the wire
        assert {p["name"] for p in rows} >= {"wide", "initial"}
        # Deleting the designated initial profile is refused.
        assert api.delete(f"/api/profiles/{initial_id}").status_code == 409
        # Move away from the saved posture, then relaunch from the profile.
        ctl.keys([])
        ctl.hold(["KeyS", "KeyQ"], 1.0)
        ctl.keys([])
    finally:
        ctl.close()
        tele.close()
    assert api.delete("/api/session").status_code == 204
    spec = dict(SPEC, start_from=f"profile:{profile_id}")
    r = api.post("/api/session", json=spec)
    assert r.status_code == 200, r.text
    tele = Tele(server)
    try:
        deadline = time.monotonic() + 30.0
        blocked = False
        seen_plan = False  # the plan must have been OBSERVED executing / done before we judge
        while time.monotonic() < deadline:
            msg = tele.latest()
            blocked |= msg["collision"]["severity"] == "blocked"
            session_block = msg.get("session") or {}
            status = session_block.get("plan_status")
            progress = session_block.get("start_from_progress")
            if status in ("executing", "done") or progress is not None:
                seen_plan = True
            if (
                seen_plan
                and session_block.get("state") == "running"
                and status not in ("planning", "executing")
            ):
                break
            time.sleep(0.02)
        assert seen_plan, "start_from plan never showed up on telemetry"
        time.sleep(0.5)
        q_now = tele.q_full()
        err = max(abs(a - b) for a, b in zip(q_now, q_saved, strict=False))
        assert err < 0.03, (q_now, q_saved)  # arm reached the profile posture
        assert not blocked  # gate never blocked during the planned motion
    finally:
        tele.close()
        api.delete("/api/session")
