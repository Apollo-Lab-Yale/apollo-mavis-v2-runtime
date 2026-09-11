"""Tier 4 (14-dora §10; v1.3, 2026-09-11): per-arm action streams on the two-arm ``mavis_v2``
sim cell over a REAL server + REAL dora control plane, ``nodes/fake_policy.py`` playing the
policy repo.

Covers: a Manipulation-Arm-only policy (``--arms grip``) drives ``grip`` through
``action_grip`` while the Perception Arm HOLDS exactly (the "fancy tripod"); the whole-cell
``action`` of a two-arm policy still works now that the frame check is per arm (before v1.3
every two-arm external session 409'd ``policy/dataset frame mismatch``); the 409 for a policy
that names an arm the session does not have; ``ExternalStatus.policy_arms`` before launch;
and the microphone reaching the ``policy`` placeholder (``mic_<id>`` is a policy input).
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
from test_e2e_teleop import Tele

from dora_bridge.harness import dora_runtime_config, node_env, requires_dora, wait_until

pytestmark = [pytest.mark.dora, pytest.mark.egl, requires_dora]

MOD = "apollo_mavis_v2_runtime.dora_bridge.nodes.fake_policy"
SPEC = {
    "mode": "inference",
    "kind": "sim",
    "arms": ["grip", "view"],  # the recorded datasets' block order (grip first)
    "frames": {"grip": "arm_base:grip", "view": "arm_base:view"},
    "sim_scene": "mavis_v2",
    "policy_source": "external",
}
RATE_HZ = 15.0


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    cfg = dora_runtime_config(
        tmp_path_factory.mktemp("rt"), scene="mavis_v2", arms=("view", "grip"), mic=True
    )
    srv = LiveServer(cfg)
    with httpx.Client(base_url=srv.http, timeout=30) as api:
        wait_until(lambda: api.get("/api/dora").json()["state"] == "attached", 10.0, "attached")
    yield srv, cfg
    srv.stop()


@pytest.fixture(scope="module")
def api(server):
    with httpx.Client(base_url=server[0].http, timeout=60.0) as client:
        yield client


class Fake:
    def __init__(self, server, *args: str) -> None:
        srv, cfg = server
        self.srv = srv
        self.stats_path = cfg.dora.var_dir / f"fake_stats_{int(time.time() * 1e3)}.json"
        info = httpx.get(f"{srv.http}/api/dora", timeout=10).json()
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                MOD,
                "--daemon-port",
                str(info["daemon_port"]),
                "--rate-hz",
                str(RATE_HZ),
                "--stats-out",
                str(self.stats_path),
                *args,
            ],
            env=node_env(info["bind_host"], info["zenoh_port"]),
            stdout=open(cfg.dora.var_dir / "fake_policy.log", "ab"),  # noqa: SIM115
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        wait_until(lambda: self.external()["policy_attached"] is True, 10.0, "spec cached")

    def external(self) -> dict:
        tele = Tele(self.srv)
        try:
            return tele.latest()["external"]
        finally:
            tele.close()

    def close(self) -> dict:
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
        wait_until(self.stats_path.is_file, 5.0, "fake stats dumped")
        return json.loads(self.stats_path.read_text())


def wait_running(api: httpx.Client, timeout: float = 20.0) -> None:
    wait_until(lambda: api.get("/api/session").json()["state"] == "running", timeout, "running")


def q_cmd(srv, arm: str) -> np.ndarray:
    return np.array(srv.runtime.bus.snapshot.get()[0].q_cmd[arm], dtype=np.float64)


def wait_spec_gone(srv) -> None:
    def gone() -> bool:
        tele = Tele(srv)
        try:
            return tele.latest()["external"]["policy_attached"] is False
        finally:
            tele.close()

    wait_until(gone, 6.0, "spec stale")


def test_manipulation_arm_only_policy_drives_grip_and_holds_view(server, api):
    srv, _cfg = server
    fake = Fake(server, "--mode", "scripted", "--amplitude-m", "0.02", "--arms", "grip")
    try:
        ext_status = fake.external()
        assert ext_status["policy_arms"] == ["grip"]  # visible BEFORE the session (the launcher)
        r = api.post("/api/session", json=SPEC)
        assert r.status_code == 200, r.text
        wait_running(api)
        session = srv.runtime.manager.session
        assert session.loop.runner.driven_arms() == frozenset({"grip"})
        view0, grip0 = q_cmd(srv, "view"), q_cmd(srv, "grip")
        tele = Tele(srv)
        try:
            wait_until(
                lambda: tele.latest()["external"]["action_age_s"] is not None, 5.0, "first action"
            )
            wait_until(
                lambda: float(np.max(np.abs(q_cmd(srv, "grip") - grip0))) > 1e-3,
                5.0,
                "the Manipulation Arm moves",
            )
            msg = tele.latest()
            assert msg["inference"]["policy_stale"] is False
            assert msg["external"]["policy_arms"] == ["grip"]
            # the Perception Arm never receives a command change: a hold, tick after tick
            t_end = time.monotonic() + 2.0
            while time.monotonic() < t_end:
                np.testing.assert_allclose(q_cmd(srv, "view"), view0, atol=1e-9)
                time.sleep(0.01)
            assert float(np.max(np.abs(q_cmd(srv, "grip") - grip0))) > 1e-3
            assert msg["external"]["dropped_inputs"] == 0 and msg["external"]["actions_late"] == 0
        finally:
            tele.close()
        assert api.delete("/api/session").status_code == 204
    finally:
        stats = fake.close()
    assert stats["per_arm_actions"].get("grip", 0) > 0 and "view" not in stats["per_arm_actions"]
    assert stats["mic_blocks"] > 0  # mic_<id> reaches the policy placeholder (v1.3)
    wait_spec_gone(srv)


def test_whole_cell_policy_on_two_arms_moves_both_now_that_frames_are_per_arm(server, api):
    srv, _cfg = server
    fake = Fake(server, "--mode", "scripted", "--amplitude-m", "0.02")
    try:
        # before a session the fake copies the IDLE announce (workcell order: view, grip)
        assert set(fake.external()["policy_arms"]) == {"grip", "view"}
        r = api.post("/api/session", json=SPEC)
        assert r.status_code == 200, r.text  # before v1.3: 409 policy/dataset frame mismatch
        wait_running(api)
        assert srv.runtime.manager.session.loop.runner.driven_arms() == frozenset(
            {"grip", "view"}
        )
        view0, grip0 = q_cmd(srv, "view"), q_cmd(srv, "grip")
        wait_until(
            lambda: float(np.max(np.abs(q_cmd(srv, "grip") - grip0))) > 1e-3
            and float(np.max(np.abs(q_cmd(srv, "view") - view0))) > 1e-3,
            8.0,
            "both arms move",
        )
        assert api.delete("/api/session").status_code == 204
    finally:
        stats = fake.close()
    assert stats["actions"] > 0 and stats["per_arm_actions"] == {}
    wait_spec_gone(srv)


def test_abs_ee_manipulation_arm_policy_holds_its_observed_tcp_and_wiggles(server, api):
    """2026-09-11: an external ``abs_ee`` policy (11-dim ``[x, y, z, r6, gripper, rail]``
    grip block) is accepted; the fake HOLDS the observed TCP + a 2 cm x/y wiggle, so the
    Manipulation Arm moves (deadline-interpolated waypoints, never a jump past the per-tick
    cap) while the Perception Arm holds; the counterfactual / telemetry stay sane."""
    srv, _cfg = server
    fake = Fake(
        server, "--mode", "scripted", "--amplitude-m", "0.02", "--arms", "grip",
        "--action-space", "abs_ee",
    )
    try:
        assert fake.external()["policy_arms"] == ["grip"]
        r = api.post("/api/session", json=SPEC)
        assert r.status_code == 200, r.text  # abs_ee is no longer a 409
        wait_running(api)
        session = srv.runtime.manager.session
        assert session.loop.runner.spec.action_space == "abs_ee"
        assert session.loop.runner.driven_arms() == frozenset({"grip"})
        view0, grip0 = q_cmd(srv, "view"), q_cmd(srv, "grip")
        tele = Tele(srv)
        try:
            wait_until(
                lambda: tele.latest()["external"]["action_age_s"] is not None, 5.0, "first action"
            )
            wait_until(
                lambda: float(np.max(np.abs(q_cmd(srv, "grip") - grip0))) > 1e-3,
                5.0,
                "the Manipulation Arm follows the abs waypoints",
            )
            dq_max = session.loop.cfg.dq_max_rad
            last = q_cmd(srv, "grip")
            t_end = time.monotonic() + 2.0
            while time.monotonic() < t_end:  # never a jump; the Perception Arm never moves
                cur = q_cmd(srv, "grip")
                assert float(np.max(np.abs(cur[:7] - last[:7]))) <= dq_max + 1e-9
                last = cur
                np.testing.assert_allclose(q_cmd(srv, "view"), view0, atol=1e-9)
                time.sleep(0.01)
            msg = tele.latest()
            assert msg["inference"]["policy_stale"] is False
            assert msg["external"]["dropped_inputs"] == 0
        finally:
            tele.close()
        assert api.delete("/api/session").status_code == 204
    finally:
        stats = fake.close()
    assert stats["per_arm_actions"].get("grip", 0) > 0
    wait_spec_gone(srv)


def test_policy_naming_an_unknown_arm_is_refused_at_launch(server, api):
    srv, _cfg = server
    fake = Fake(server, "--mode", "hold", "--arms", "nope")
    try:
        # read THIS fake's spec, not a residual one from a prior test in the module fixture
        wait_until(lambda: fake.external()["policy_arms"] == ["nope"], 6.0, "the nope spec")
        r = api.post("/api/session", json=SPEC)
        assert r.status_code == 409 and "unknown arm" in r.json()["detail"], r.text
        assert api.get("/api/session").status_code == 404
    finally:
        fake.close()
    wait_spec_gone(srv)
