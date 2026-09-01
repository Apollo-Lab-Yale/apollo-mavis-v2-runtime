"""Inference-mode e2e over a real server (12-dagger §3): promoted-checkpoint
gating (409), policy driving, Space = safety escape through the twin gate
(safety_debug), episode keys nacked, ZERO dataset files / trainer processes."""

from __future__ import annotations

import os
import time

import httpx
import pytest
from conftest import LiveServer, make_runtime_config
from helpers import make_net, promote, write_checkpoint
from test_e2e_teleop import Ctl, Tele

SPEC = {
    "mode": "inference",
    "kind": "sim",
    "arms": ["arm0"],
    "frames": {"arm0": "arm_base:arm0"},
    "sim_scene": "guardrail_env",
}


def tiny_net():
    net = make_net(0)
    net.body[-1].weight.data *= 0.01  # near-hold deltas: stable random policy
    net.body[-1].bias.data.zero_()
    return net


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    cfg = make_runtime_config(
        tmp_path_factory.mktemp("rt"), scene="guardrail_env", safety_debug=True
    )
    srv = LiveServer(cfg)
    yield srv
    srv.stop()


@pytest.fixture(scope="module")
def api(server):
    with httpx.Client(base_url=server.http, timeout=120.0) as client:
        yield client


def test_no_promoted_deploy_checkpoint_409(server, api):
    r = api.post("/api/session", json=SPEC)
    assert r.status_code == 409
    assert "promoted" in r.json()["detail"]


def test_online_checkpoint_alone_still_409(server, api):
    write_checkpoint(server.runtime.cfg.checkpoints_root, "seed0", 1, net=tiny_net())
    r = api.post("/api/session", json=SPEC)
    assert r.status_code == 409  # inference loads ONLY promoted deploy ckpts
    r = api.post("/api/session", json={**SPEC, "policy": "seed0/v000001"})
    assert r.status_code == 409


def test_frame_mismatch_409(server, api):
    root = server.runtime.cfg.checkpoints_root
    write_checkpoint(root, "seedW", 1, deploy=True, net=tiny_net(),
                     action_frame="world")
    promote(root, "seedW/deploy/v001")
    r = api.post("/api/session", json=SPEC)
    assert r.status_code == 409 and "frame mismatch" in r.json()["detail"]


def test_inference_session_e2e(server, api):
    root = server.runtime.cfg.checkpoints_root
    write_checkpoint(root, "seed0", 1, deploy=True, net=tiny_net())
    promote(root, "seed0/deploy/v001")
    pol = api.get("/api/policies").json()
    assert any(p["policy_id"] == "seed0/deploy/v001" and p["promoted"] for p in pol)
    assert api.get("/api/workcell").json()["policies_available"] is True

    r = api.post("/api/session", json=SPEC)  # policy=None -> the promoted ckpt
    assert r.status_code == 200, r.text
    for _ in range(200):
        if api.get("/api/session").json()["state"] == "running":
            break
        time.sleep(0.05)
    ctl, tele = Ctl(server), Tele(server)
    session = server.runtime.manager.session
    try:
        # gate telemetry rides `inference`; dagger block is absent (§11)
        msg = tele.latest()
        assert msg["dagger"] is None and msg["episode"] is None
        assert msg["inference"]["control_mode"] == "policy"
        assert msg["inference"]["policy_version"] == "seed0/deploy/v001"
        assert msg["collision"] is not None  # twin gate live (safety_debug)

        # structural no-recording guarantees
        assert session.recorder_thread is None
        assert session.policy_session.trainer_client is None
        assert session.policy_session.reloader is None

        for op in ("episode_new", "episode_save", "episode_discard"):
            assert ctl.action(op)["ok"] is False  # no episodes in inference

        # Space = safety escape through the SAME gate
        ack = ctl.action("takeover_toggle")
        assert ack["ok"] is True and ack["detail"] == "takeover_transition"
        t0 = time.monotonic()
        while tele.latest()["inference"]["control_mode"] != "human":
            assert time.monotonic() - t0 < 2.0
        assert ctl.action("switch_arm")["ok"] is False  # takeover active
        x0 = tele.ee_x()
        ctl.hold(["KeyW"], 1.0)  # human steers to safety; twin-gated like policy
        deadline = time.monotonic() + 3.0  # poll past any stale WS frames
        while tele.ee_x() - x0 <= 0.03:
            assert time.monotonic() < deadline, "human twist did not move the arm"
        ack = ctl.action("takeover_toggle")  # handback permitted
        assert ack["ok"] is True and ack["detail"] == "policy"
    finally:
        ctl.close()
        tele.close()
        r = api.delete("/api/session")  # "Terminate session" path
        assert r.status_code == 204

    # zero dataset files for the whole session (semantic invariant, §3)
    ds_root = server.runtime.cfg.datasets_root
    assert not ds_root.exists() or list(os.scandir(ds_root)) == []
