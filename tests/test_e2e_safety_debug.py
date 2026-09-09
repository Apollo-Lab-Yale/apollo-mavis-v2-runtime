"""safety_debug e2e: the full hardware safety stack over the sim cell.

Replays the guardrail ``env_table_descend`` through the WS control path and
asserts the gate blocks BEFORE contact, telemetry ``collision.severity`` walks
``warn -> blocked``, and blocked commands never reach the workcell
(hold-last-safe). This is the runtime side of the 11-safety §5 CI regression;
``guardrail_check --all`` is wired in test_guardrail_ci.py.
"""

from __future__ import annotations

import json
import time

import httpx
import numpy as np
import pytest
from conftest import ControlConfig, DatasetsConfig, LiveServer, RuntimeConfig, VideoConfig
from websockets.sync.client import connect as ws_connect

SPEC = {
    "mode": "teleop",
    "kind": "sim",
    "arms": ["arm0"],
    "frames": {"arm0": "arm_base:arm0"},
    "sim_scene": "guardrail_env",
}


def _debug_config(tmp_path) -> RuntimeConfig:
    wc = {
        "kind": "sim",
        "sim_scene": "guardrail_env",
        "arms": [{"id": "arm0", "base_in_world": {}}],
        "cameras": [],
        "safety": {"safety_debug": True, "geom_inflation_m": 0.008},
    }
    return RuntimeConfig(
        workcells={"sim": wc},
        profiles_dir=tmp_path / "profiles",
        # no mapped dataset roots: the startup sweep must never touch the operator's ~/data
        datasets=DatasetsConfig(default_namespace="apollo", namespaces={}),
        video=VideoConfig(preview_fps=15, session_fps=30),
        # "KeyQ descends" is the whole replay: pin the pre-2026-09-08 base frame
        # (the runtime default is "world" since 2026-09-08 evening; "camera", where
        # the keys follow the tool, was that morning's). See the note in
        # conftest.make_runtime_config.
        control=ControlConfig(translate_frame="base"),
    )


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    srv = LiveServer(_debug_config(tmp_path_factory.mktemp("rt")))
    yield srv
    srv.stop()


def test_gate_blocks_before_contact_warn_then_blocked(server):
    with httpx.Client(base_url=server.http, timeout=30.0) as api:
        assert api.post("/api/session", json=SPEC).status_code == 200
        for _ in range(100):
            if api.get("/api/session").json()["state"] == "running":
                break
            time.sleep(0.05)
        ctl = ws_connect(f"{server.ws}/ws/control")
        tele = ws_connect(f"{server.ws}/ws/telemetry")
        try:
            json.loads(ctl.recv(timeout=5))  # hello
            severities: list[str] = []
            seq = 0
            end = time.monotonic() + 5.0
            while time.monotonic() < end:
                seq += 1
                # KeyQ = translate z-neg; base frame + arm0 base identity -> descend.
                ctl.send(json.dumps(
                    {"t": "keys", "seq": seq, "ts": time.time(), "held": ["KeyQ"]}
                ))
                time.sleep(0.04)
                try:
                    while True:
                        msg = json.loads(tele.recv(timeout=0.001))
                        sev = msg["collision"]["severity"]
                        if not severities or severities[-1] != sev:
                            severities.append(sev)
                except TimeoutError:
                    pass
                if severities[-1:] == ["blocked"] and "warn" in severities:
                    break
        finally:
            ctl.close()
            tele.close()
        # ok -> warn -> blocked, in order.
        assert "blocked" in severities, severities
        assert "warn" in severities, severities
        assert severities.index("warn") < severities.index("blocked"), severities

        # Blocked commands are held: measured arm config sits still under the
        # sustained descent twist (hold-last-safe; no real contact).
        runtime = server.runtime
        session = runtime.manager.session
        data = session.workcell._data
        assert int((data.contact.dist[: data.ncon] <= 0).sum()) == 0
        q0 = np.array(session.workcell.states()["arm0"].q)
        seq2 = seq
        # Re-open a controller to keep the deadman fresh and keep pushing.
        ctl = ws_connect(f"{server.ws}/ws/control")
        try:
            json.loads(ctl.recv(timeout=5))
            for _ in range(60):
                seq2 += 1
                ctl.send(json.dumps(
                    {"t": "keys", "seq": seq2, "ts": time.time(), "held": ["KeyQ"]}
                ))
                time.sleep(0.02)
            q1 = np.array(session.workcell.states()["arm0"].q)
            assert float(np.max(np.abs(q1 - q0))) < 5e-3  # held in place
            assert int((data.contact.dist[: data.ncon] <= 0).sum()) == 0
        finally:
            ctl.close()
        api.delete("/api/session")
