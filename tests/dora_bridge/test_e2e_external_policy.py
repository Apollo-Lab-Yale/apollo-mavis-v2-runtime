"""Tier 4 (14-dora §10): external policy sessions over a REAL server + REAL dora control plane,
with ``nodes/fake_policy.py`` playing the policy repo (``guardrail_env`` sim scene, safety_debug).

Covers the 验收标准 "外部策略" block: the 409 / 422 matrix, RUNNING <= 1 s, hold-on-stale after
``kill -9`` (0.45 s + 1 tick at 15 Hz), the reset watermark, late actions, the twin gate blocking
a collision course with ``CollisionEvent.source == "policy"``, Space takeover / handback, and
external DAgger (zero trainer, recorder on, ``events.episode_saved``, ``policy_version`` column,
mid-episode version change).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import httpx
import numpy as np
import pytest
from conftest import LiveServer
from test_e2e_teleop import Ctl, Tele

from dora_bridge.harness import (
    dora_runtime_config,
    node_env,
    requires_dora,
    wait_until,
)

pytestmark = [pytest.mark.dora, pytest.mark.egl, requires_dora]

MOD = "apollo_mavis_v2_runtime.dora_bridge.nodes"
SPEC = {
    "mode": "inference",
    "kind": "sim",
    "arms": ["arm0"],
    "frames": {"arm0": "arm_base:arm0"},
    "sim_scene": "guardrail_env",
    "policy_source": "external",
}
RATE_HZ = 15.0
HOLD_S = 1.0 / RATE_HZ + 0.05 + 5.0 / RATE_HZ  # 0.45 s (12-dagger §6.3)


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    cfg = dora_runtime_config(
        tmp_path_factory.mktemp("rt"), scene="guardrail_env", arms=("arm0",), safety_debug=True
    )
    srv = LiveServer(cfg)
    with httpx.Client(base_url=srv.http, timeout=30) as api:
        wait_until(lambda: api.get("/api/dora").json()["state"] == "attached", 10.0, "attached")
    yield srv, cfg
    bridge = srv.runtime.dora.bridge
    plane_pids = list(bridge.plane.child_pids()) if bridge.plane is not None else []
    srv.stop()
    time.sleep(0.3)
    alive = [p for p in plane_pids if os.path.exists(f"/proc/{p}")]
    assert not alive, f"runtime's dora children survived stop: {alive}"


@pytest.fixture(scope="module")
def api(server):
    with httpx.Client(base_url=server[0].http, timeout=60.0) as client:
        yield client


def external(api: httpx.Client, srv) -> dict:
    tele = Tele(srv)
    try:
        return tele.latest()["external"]
    finally:
        tele.close()


class FakePolicy:
    """A ``fake_policy`` subprocess bound to the server's private daemon."""

    def __init__(self, server, *args: str, wait_spec: bool = True) -> None:
        srv, cfg = server
        self.srv = srv
        info = httpx.get(f"{srv.http}/api/dora", timeout=10).json()
        # a real policy knows its training frame; the fake is told (the session's frames only
        # reach the node through the `session` announce AFTER the session exists)
        if "--action-frame" not in args:
            args = (*args, "--action-frame", SPEC["frames"]["arm0"])
        args = (*args, "--stats-out", str(cfg.dora.var_dir / "fake_stats.json"))
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                f"{MOD}.fake_policy",
                "--daemon-port",
                str(info["daemon_port"]),
                "--rate-hz",
                str(RATE_HZ),
                *args,
            ],
            env=node_env(info["bind_host"], info["zenoh_port"]),
            stdout=open(cfg.dora.var_dir / "fake_policy.log", "ab"),  # noqa: SIM115
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        if wait_spec:
            wait_until(lambda: external_ok(srv), 10.0, "fake policy spec cached")

    def kill9(self) -> None:
        os.killpg(self.proc.pid, signal.SIGKILL)
        self.proc.wait(timeout=5)

    def close(self) -> None:
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)


def external_ok(srv) -> bool:
    with httpx.Client(base_url=srv.http, timeout=10) as api:
        return external(api, srv)["policy_attached"] is True


def wait_running(api: httpx.Client, timeout: float = 20.0) -> None:
    wait_until(lambda: api.get("/api/session").json()["state"] == "running", timeout, "running")


def q_of(tele: Tele) -> np.ndarray:
    return np.asarray(tele.q_full(), dtype=np.float64)


def q_cmd(srv, arm: str = "arm0") -> np.ndarray:
    """The COMMANDED joints (the sim's measured joints keep settling after a hold)."""
    return np.array(srv.runtime.bus.snapshot.get()[0].q_cmd[arm], dtype=np.float64)


def test_409_422_matrix_then_running_fast(server, api):
    srv, cfg = server
    r = api.post("/api/session", json=SPEC)  # no spec ever received
    assert r.status_code == 409 and "no external policy attached" in r.json()["detail"], r.text
    r = api.post("/api/session", json={**SPEC, "policy": "xyz"})
    assert r.status_code == 422  # checkpoint + external at once (core validator)
    fake = FakePolicy(server, "--mode", "hold", "--action-frame", "world")
    try:
        r = api.post("/api/session", json=SPEC)
        assert r.status_code == 409 and "frame mismatch" in r.json()["detail"], r.text
    finally:
        fake.close()
    # spec heartbeat gone for > 3 s -> 409 again
    wait_until(lambda: not external_ok(srv), 6.0, "spec stale")
    r = api.post("/api/session", json=SPEC)
    assert r.status_code == 409 and "no external policy attached" in r.json()["detail"]
    fake = FakePolicy(server, "--mode", "scripted", "--amplitude-m", "0.02")
    try:
        t0 = time.monotonic()
        r = api.post("/api/session", json=SPEC)
        assert r.status_code == 200, r.text
        assert time.monotonic() - t0 <= 1.0
        assert r.json()["policy_source"] == "external"
        wait_running(api)
        tele = Tele(srv)
        try:
            msg = tele.latest()
            assert msg["inference"]["policy_version"] == "fake-policy/v000001"
            wait_until(
                lambda: tele.latest()["external"]["action_age_s"] is not None, 5.0, "first action"
            )
            msg = tele.latest()
            assert msg["inference"]["policy_stale"] is False
            assert (
                msg["external"]["policy_attached"] and msg["external"]["policy_id"] == "fake-policy"
            )
            assert msg["session"]["trainer_alive"] is None
            session = srv.runtime.manager.session
            assert type(session.loop.runner).__name__ == "ExternalPolicySource"
            assert (
                session.policy_session.trainer_client is None
                and session.policy_session.reloader is None
            )
            q0 = q_of(tele)
            wait_until(
                lambda: float(np.max(np.abs(q_of(tele) - q0))) > 1e-3, 5.0, "policy moves the arm"
            )
            # -- hold on stale: kill -9 the node -> q_cmd frozen within 0.45 s + 1 tick ------------
            fake.kill9()
            t_kill = time.monotonic()
            last_change = t_kill
            bus = srv.runtime.bus  # q_cmd = the commanded target (the measured sim joints
            prev = np.array(bus.snapshot.get()[0].q_cmd["arm0"])  # keep settling afterwards)
            while time.monotonic() - t_kill < 1.5:
                q = np.array(bus.snapshot.get()[0].q_cmd["arm0"])
                if float(np.max(np.abs(q - prev))) > 1e-9:
                    last_change = time.monotonic()
                prev = q
                time.sleep(0.005)
            assert last_change - t_kill <= HOLD_S + 0.01 + 0.04, (
                f"moved until {last_change - t_kill:.3f}s"
            )
            msg = tele.latest()
            assert msg["inference"]["policy_stale"] is True
            age1 = msg["external"]["action_age_s"]
            time.sleep(0.3)
            msg2 = tele.latest()
            assert msg2["external"]["action_age_s"] > age1  # monotonic growth
            wait_until(
                lambda: tele.latest()["external"]["policy_attached"] is False, 4.0, "spec stale"
            )
            # -- late actions are dropped and counted; a fresh node resumes motion ---------
            late = FakePolicy(
                server, "--mode", "scripted", "--amplitude-m", "0.02", "--stale-obs", "40"
            )
            try:
                ext0 = tele.latest()["external"]
                wait_until(
                    lambda: tele.latest()["external"]["actions_late"] > ext0["actions_late"],
                    8.0,
                    "late actions counted",
                )
                # the first ~40 echoes of the late fake were fresh (its history was short) and
                # moved the arm; once every action is late the policy decays to hold again
                wait_until(  # past the 0.45 s decay: total hold
                    lambda: (tele.latest()["external"]["action_age_s"] or 0.0) > HOLD_S + 0.1,
                    5.0,
                    "hold",
                )
                q_hold = q_cmd(srv)
                time.sleep(0.5)
                assert float(np.max(np.abs(q_cmd(srv) - q_hold))) < 1e-9  # nothing applied
            finally:
                late.kill9()
            fresh = FakePolicy(server, "--mode", "scripted", "--amplitude-m", "0.02")
            try:
                q1 = q_of(tele)
                wait_until(
                    lambda: float(np.max(np.abs(q_of(tele) - q1))) > 1e-3, 6.0, "motion resumes"
                )
            finally:
                fresh.close()
        finally:
            tele.close()
    finally:
        fake.close()
        assert api.delete("/api/session").status_code == 204


def start_events_observer(info: dict, out, seconds: float) -> subprocess.Popen:
    """An `observer` node that appends every `events` message to a JSON file."""
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"""
import json, time
from dora import Node
node = Node("observer", daemon_port={info["daemon_port"]})
out = []
t_end = time.monotonic() + {seconds}
while time.monotonic() < t_end:
    ev = node.next(timeout=1.0)
    if ev is None or ev.get("type") == "STOP": break
    if ev.get("type") == "INPUT" and ev["id"] == "events":
        out.append(json.loads(ev["value"][0].as_py()))
        json.dump(out, open({json.dumps(str(out))}, "w"))
""",
        ],
        env=node_env(info["bind_host"], info["zenoh_port"]),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def test_collision_course_is_blocked_with_policy_source(server, api, tmp_path):
    srv, cfg = server
    fake = FakePolicy(server, "--mode", "collide", "--collide-mps", "0.25")
    events_out = tmp_path / "events.json"
    observer = start_events_observer(api.get("/api/dora").json(), events_out, 60)
    ctl = None
    try:
        # guardrail_env: the table lies below the arm (test_e2e_safety_debug descends into it
        # with KeyQ = -z); the fake's constant -z delta (its default axis) reaches it in seconds
        r = api.post("/api/session", json=SPEC)
        assert r.status_code == 200, r.text
        wait_running(api)
        tele = Tele(srv)
        try:
            seen = []
            t0 = time.monotonic()
            while time.monotonic() - t0 < 25.0:
                msg = tele.latest()
                seen.append(msg["collision"]["severity"])
                if msg["collision"]["blocked"]:
                    break
            assert "blocked" in seen, seen[-20:]
            # the gate's own CollisionEvent (with the command source) rides `events`
            wait_until(
                lambda: (
                    events_out.exists()
                    and any(e["kind"] == "collision" for e in json.load(open(events_out)))
                ),
                10.0,
                "collision event on the bus",
            )
            coll = [e for e in json.load(open(events_out)) if e["kind"] == "collision"]
            blocked = [e for e in coll if e["payload"]["kind"] in ("blocked", "penetration")]
            assert blocked, coll
            assert all(e["payload"]["source"] == "policy" for e in blocked), [
                (e["payload"]["source"], e["payload"]["pairs"]) for e in blocked
            ]
            assert all(e["session_id"] == srv.runtime.manager.session.session_id for e in coll)
            # blocked = held: the measured config sits still while the fake keeps pushing
            q0 = q_of(tele)
            time.sleep(0.5)
            assert float(np.max(np.abs(q_of(tele) - q0))) < 5e-3
            # Space takeover / handback go through the same gate as in-process policies
            ctl = Ctl(srv)
            ack = ctl.action("takeover_toggle")
            assert ack["ok"] and ack["detail"] == "takeover_transition"
            wait_until(lambda: tele.latest()["inference"]["control_mode"] == "human", 3.0, "human")
            ack = ctl.action("takeover_toggle")
            assert ack["ok"] and ack["detail"] == "policy"
        finally:
            tele.close()
    finally:
        if ctl is not None:
            ctl.close()
        fake.close()
        os.killpg(observer.pid, signal.SIGTERM)
        api.delete("/api/session")


def test_reset_watermark_drops_chunks_at_or_below_after_observation_id(server, api):
    srv, cfg = server
    fake = FakePolicy(
        server, "--mode", "chunk", "--chunk", "4", "--amplitude-m", "0.02", "--reset-violate"
    )
    ctl = None
    try:
        r = api.post("/api/session", json=SPEC)
        assert r.status_code == 200, r.text
        wait_running(api)
        tele, ctl = Tele(srv), Ctl(srv)
        try:
            q0 = q_of(tele)
            wait_until(lambda: float(np.max(np.abs(q_of(tele) - q0))) > 1e-3, 5.0, "chunks drive")
            # takeover + handback -> policy_reset{handback, after_observation_id = newest}; the fake
            # then sends chunks with observation_id <= watermark for 2 s: all dropped, arm still
            assert ctl.action("takeover_toggle")["detail"] == "takeover_transition"
            time.sleep(0.5)
            src = srv.runtime.manager.session.loop.runner
            late0 = src.actions_late
            assert ctl.action("takeover_toggle")["detail"] == "policy"
            time.sleep(0.4)  # slew window of the handback
            q_hold = q_cmd(srv)
            t0 = time.monotonic()
            while time.monotonic() - t0 < 1.2:
                assert float(np.max(np.abs(q_cmd(srv) - q_hold))) < 1e-9, (
                    "a pre-watermark chunk moved the arm"
                )
                time.sleep(0.01)
            assert src.actions_late > late0
            assert src.watermark > 0
            ev = tele.latest()["external"]
            assert ev["actions_late"] == src.actions_late
        finally:
            tele.close()
    finally:
        if ctl is not None:
            ctl.close()
        fake.close()
        api.delete("/api/session")


def test_external_dagger_two_episodes(server, api, tmp_path):
    srv, cfg = server
    # return_to_start False: D6 makes it default ON for dagger (409 without a profile)
    spec = {**SPEC, "mode": "dagger", "task": "external dagger e2e", "return_to_start": False}
    fake = FakePolicy(
        server, "--mode", "scripted", "--amplitude-m", "0.01", "--version-bump-after", "65"
    )
    events_out = tmp_path / "events.json"
    info = api.get("/api/dora").json()
    observer = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"""
import json, time
from dora import Node
node = Node("observer", daemon_port={info["daemon_port"]})
out = []
t_end = time.monotonic() + 40
while time.monotonic() < t_end:
    ev = node.next(timeout=1.0)
    if ev is None or ev.get("type") == "STOP": break
    if ev.get("type") == "INPUT" and ev["id"] == "events":
        out.append(json.loads(ev["value"][0].as_py()))
        json.dump(out, open({json.dumps(str(events_out))}, "w"))
""",
        ],
        env=node_env(info["bind_host"], info["zenoh_port"]),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    ctl = None
    try:
        r = api.post("/api/session", json=spec)
        assert r.status_code == 200, r.text
        wait_running(api)
        session = srv.runtime.manager.session
        loop = session.loop
        assert session.policy_session.trainer_client is None
        ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
        assert "dagger.trainer" not in ps  # zero trainer processes
        tele, ctl = Tele(srv), Ctl(srv)
        try:
            assert tele.latest()["dagger"]["policy_version"] == "fake-policy/v000001"
            assert tele.latest()["session"]["trainer_alive"] is None
            for _ in range(2):
                assert ctl.action("episode_new")["ok"]
                time.sleep(2.5)  # ~37 policy acts per episode at 15 Hz -> the bump lands in ep 1
                assert ctl.action("episode_save")["ok"]
                wait_until(lambda: tele.latest()["episode"]["state"] == "idle", 15.0, "saved")
            wait_until(
                lambda: (
                    events_out.exists()
                    and sum(1 for e in json.load(open(events_out)) if e["kind"] == "episode_saved")
                    >= 2
                ),
                15.0,
                "two episode_saved events",
            )
            events = json.load(open(events_out))
            saved = [e for e in events if e["kind"] == "episode_saved"]
            import dataclasses

            from apollo_mavis_v2_core.dagger import EpisodeSummary

            names = {f.name for f in dataclasses.fields(EpisodeSummary)}
            for e in saved:
                assert set(e["payload"]["summary"]) == names
                assert os.path.exists(e["payload"]["spool_path"])
                assert e["payload"]["dataset_root"] and e["payload"]["run_id"] == loop.run_id
                assert e["session_id"] == session.session_id
            changed = [e for e in events if e["kind"] == "policy_version_changed"]
            fake.close()  # SIGTERM -> the fake dumps its stats
            stats = (
                (cfg.dora.var_dir / "fake_stats.json").read_text()
                if (cfg.dora.var_dir / "fake_stats.json").exists()
                else "<no stats>"
            )
            assert len(changed) == 1 and changed[0]["payload"]["to"] == 2, (
                [e["kind"] for e in events],
                stats,
                loop.runner.current_version(),
            )
            assert tele.latest()["external"]["version_changes_mid_episode"] == 1
            assert tele.latest()["dagger"]["policy_version"] == "fake-policy/v000002"
            root = saved[0]["payload"]["dataset_root"]
        finally:
            tele.close()
        api.delete("/api/session")
        # the dataset's policy_version column == the action metadata versions (1 then 2).
        # One directory per episode (10-frames §11): episodes/<episode_id>/frames.parquet;
        # ids are capture-time stamps, so sorted == recording order (LeRobot v3 is an export).
        import pyarrow.parquet as pq

        files = sorted(__import__("pathlib").Path(root).glob("episodes/*/frames.parquet"))
        assert len(files) == 2, files
        vers = [
            np.asarray(
                [
                    v[0] if isinstance(v, list) else v
                    for v in pq.read_table(f).column("policy_version").to_pylist()
                ]
            ).astype(int)
            for f in files
        ]
        ver = np.concatenate(vers)
        assert set(np.unique(ver).tolist()) <= {1, 2} and 1 in ver and 2 in ver
        assert set(np.unique(vers[0]).tolist()) == {1}  # episode 0 entirely v1
        assert 2 in vers[1]  # the bump landed in episode 1
    finally:
        if ctl is not None:
            ctl.close()
        fake.close()
        os.killpg(observer.pid, signal.SIGTERM)
        api.delete("/api/session")
