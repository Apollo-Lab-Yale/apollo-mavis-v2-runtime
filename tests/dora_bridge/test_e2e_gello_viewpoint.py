"""Tier 4 (14-dora §10; 16-gello §7 / §12.3): the GELLO viewpoint node over a REAL server + REAL
dora control plane on the kitchen twin, with ``nodes/fake_policy.py`` playing the viewpoint
node (``--mode scripted``, NO ``--action-frame``: the node must derive its frame from the
announce's ``external_arms``) and the FAKE leader synced. The spec lists the arms in the UI's
order (Manipulation Arm first, ``["grip", "view"]``) - 2026-09-09 review: a node taking the
FIRST announced arm's frame answered ``arm_base:grip`` and was ignored in every UI-launched
session; the earlier version of this test hid it with a view-first spec and an explicit frame.

Covers: the ``session`` announce of a gello session carries ``external_arms == ["view"]`` and a
VIEW-ONLY ``action_names`` / ``state_names`` layout; the fake node derives its spec (frame
included) from that announce, the runtime attaches it (``viewpoint: auto``) once the launch
motion is over and ``telemetry.gello.viewpoint`` says so; the Perception Arm's ``q_cmd`` moves
while the Manipulation Arm's joints stay on the leader; ``kill -9`` of the node -> the
Perception Arm holds within 0.45 s + 1 tick and the source detaches when the spec heartbeat
goes stale.
"""

from __future__ import annotations

import time

import httpx
import numpy as np
import pytest
from conftest import LiveServer
from test_e2e_teleop import Tele

from apollo_mavis_v2_runtime.config import GelloConfig
from apollo_mavis_v2_runtime.devices.gello import FAKE_DEFAULT_Q
from apollo_mavis_v2_runtime.gello.viewpoint import view_action_names, view_state_names
from apollo_mavis_v2_runtime.recorder.features import arm_state_names
from dora_bridge.harness import dora_runtime_config, requires_dora, wait_until
from dora_bridge.test_e2e_external_policy import FakePolicy, external_ok, wait_running

pytestmark = [pytest.mark.dora, pytest.mark.egl, requires_dora]

KITCHEN = "mavis_v2_kitchen"
SPEC = {  # the UI's order: the Manipulation Arm first (buildGelloSpec / orderArms)
    "mode": "gello",
    "kind": "sim",
    "arms": ["grip", "view"],
    "frames": {"grip": "arm_base:grip", "view": "arm_base:view"},
    "sim_scene": KITCHEN,
    "gello": {"viewpoint": "auto"},
}
RATE_HZ = 15.0
HOLD_S = 1.0 / RATE_HZ + 0.05 + 5.0 / RATE_HZ  # 0.45 s (12-dagger §6.3)


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rt")
    cfg = dora_runtime_config(tmp, scene="mavis_v2", arms=("view", "grip")).model_copy(
        update={"gello": GelloConfig(backend="fake", calibration_path=tmp / "cal.json")}
    )
    srv = LiveServer(cfg)
    with httpx.Client(base_url=srv.http, timeout=30) as api:
        wait_until(lambda: api.get("/api/dora").json()["state"] == "attached", 10.0, "attached")
        wait_until(lambda: api.get("/api/gello").json()["status"] == "connected", 10.0, "leader")
    yield srv, cfg
    bridge = srv.runtime.dora.bridge
    plane_pids = list(bridge.plane.child_pids()) if bridge.plane is not None else []
    srv.stop()
    time.sleep(0.3)
    import os

    alive = [p for p in plane_pids if os.path.exists(f"/proc/{p}")]
    assert not alive, f"runtime's dora children survived stop: {alive}"


@pytest.fixture(scope="module")
def api(server):
    with httpx.Client(base_url=server[0].http, timeout=60.0) as client:
        yield client


def q_cmd(srv, arm: str) -> np.ndarray:
    return np.array(srv.runtime.bus.snapshot.get()[0].q_cmd[arm], dtype=np.float64)


def test_viewpoint_node_drives_the_perception_arm_only(server, api):
    srv, cfg = server
    fake = FakePolicy(  # no --action-frame: the node derives it from external_arms (14-dora §5)
        server, "--mode", "scripted", "--amplitude-m", "0.02", frame_from_announce=True
    )
    try:
        assert external_ok(srv)  # the idle announce taught it the TWO-arm layout so far
        r = api.post("/api/session", json=SPEC)
        assert r.status_code == 200, r.text
        assert r.json()["gello"] == {"viewpoint": "auto"}
        wait_running(api)
        # -- the announce: external_arms + the view-only layout (16-gello D5) ---------------------
        facts = srv.runtime.dora.publisher.facts
        assert facts.external_arms == ["view"]
        assert facts.action_names == view_action_names(True)
        assert facts.state_names == view_state_names(True)
        assert facts.arm_ids == ["grip", "view"]  # obs_state still carries both arms ...
        assert facts.obs_state_names == [  # ... and its metadata names that full vector
            n for a in ("grip", "view") for n in arm_state_names(a, True)
        ]
        ann = srv.runtime.dora.publisher._announce("running")
        assert ann.external_arms == ["view"] and ann.action_names == view_action_names(True)
        assert ann.frames == {"grip": "arm_base:grip", "view": "arm_base:view"}
        assert list(ann.model_dump())[-1] == "external_arms"
        # -- the node re-publishes its spec from the announce -> compatible -> attached ---------
        tele = Tele(srv)
        try:
            try:
                wait_until(
                    lambda: tele.latest()["gello"]["viewpoint"]["attached"] is True,
                    15.0,
                    "viewpoint attached",
                )
            except TimeoutError as e:  # say WHY the runtime ignored the node
                hub_spec = srv.runtime.dora.policy_hub.spec()
                raise AssertionError(
                    f"{e}; viewpoint={tele.latest()['gello']['viewpoint']}; "
                    f"hub spec={hub_spec.spec if hub_spec is not None else None}"
                ) from e
            m = tele.latest()
            assert m["gello"]["viewpoint"]["policy_id"] == "fake-policy"
            assert m["gello"]["viewpoint"]["paused"] is False
            assert m["gello"]["state"] == "tracking"  # the synced leader
            assert m["gello"]["paused_latched"] is False
            # the node derived arm_base:view from external_arms (not the first arm's frame)
            spec_ann = srv.runtime.dora.policy_hub.spec()
            assert spec_ann is not None and spec_ann.spec.action_frame == "arm_base:view"
            assert m["external"]["policy_attached"] is True
            grip0 = q_cmd(srv, "grip")
            view0 = q_cmd(srv, "view")
            wait_until(
                lambda: float(np.max(np.abs(q_cmd(srv, "view")[:7] - view0[:7]))) > 1e-3,
                10.0,
                "the viewpoint node moves the Perception Arm",
            )
            wait_until(
                lambda: tele.latest()["external"]["action_age_s"] is not None, 5.0, "actions"
            )
            # the Manipulation Arm's joints stay on the leader (nothing on the bus moves it)
            assert np.allclose(q_cmd(srv, "grip")[:7], grip0[:7], atol=1e-6)
            assert np.allclose(grip0[:7], FAKE_DEFAULT_Q, atol=1e-6)
            assert tele.latest()["gello"]["state"] == "tracking"
            # -- kill -9: the Perception Arm holds within 0.45 s + 1 tick ------------------------
            fake.kill9()
            t_kill = time.monotonic()
            last_change = t_kill
            prev = q_cmd(srv, "view")
            while time.monotonic() - t_kill < 1.5:
                q = q_cmd(srv, "view")
                if float(np.max(np.abs(q - prev))) > 1e-9:
                    last_change = time.monotonic()
                prev = q
                time.sleep(0.005)
            assert last_change - t_kill <= HOLD_S + 0.01 + 0.04, last_change - t_kill
            # the spec heartbeat goes stale (3 s): the source detaches, the arm holds
            wait_until(
                lambda: tele.latest()["gello"]["viewpoint"]["attached"] is False,
                8.0,
                "viewpoint detached on a stale spec",
            )
            assert "waiting for a viewpoint node" in tele.latest()["gello"]["viewpoint"]["detail"]
            assert tele.latest()["gello"]["state"] == "tracking"  # the follower is unaffected
        finally:
            tele.close()
    finally:
        fake.close()
        api.delete("/api/session")
    wait_until(lambda: api.get("/api/session").status_code == 404, 10.0, "torn down")
    idle = srv.runtime.dora.publisher._announce("idle")
    assert idle.external_arms == [] and idle.session_id is None
