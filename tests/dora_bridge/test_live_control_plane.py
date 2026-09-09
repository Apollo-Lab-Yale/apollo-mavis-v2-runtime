"""Tier 3 (14-dora §10): the runtime against a REAL private dora control plane.

Marked ``dora``: skipped without the ``[dora]`` extra + CLI. Every test builds a
``LiveServer`` (real uvicorn) from :func:`dora_runtime_config` (sim ``mavis_v2``,
free private ports, ``var_dir`` in ``tmp_path``) and tears it down; the
coordinator / daemon it spawns are reaped by ``Runtime.stop()`` and asserted
gone with ``pgrep -x dora``. Port discipline: see ``harness.py``.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import httpx
import pytest
from conftest import LiveServer
from websockets.sync.client import connect as ws_connect

from dora_bridge.harness import (
    dora_cli,
    dora_pids,
    dora_runtime_config,
    node_env,
    requires_dora,
    wait_until,
)

pytestmark = [pytest.mark.dora, requires_dora]

MOD = "apollo_mavis_v2_runtime.dora_bridge.nodes"


def get_dora(api: httpx.Client) -> dict:
    return api.get("/api/dora").json()


def wait_state(api: httpx.Client, state: str, timeout: float) -> tuple[dict, float]:
    t0 = time.monotonic()

    def probe():
        i = get_dora(api)
        return i if i["state"] == state else None

    info = wait_until(probe, timeout, f"dora state {state}", dt=0.05)
    return info, time.monotonic() - t0


def telemetry_external(srv: LiveServer) -> dict:
    with ws_connect(f"{srv.ws}/ws/telemetry") as ws:
        return json.loads(ws.recv(timeout=5))["external"]


def listening(pid: int) -> list[str]:
    """``ss -ltnp`` local addresses of ``pid`` (TCP listeners)."""
    out = subprocess.run(["ss", "-ltnpH"], capture_output=True, text=True).stdout
    return [line.split()[3] for line in out.splitlines() if f"pid={pid}," in line]


def udp_sockets(pids: set[int]) -> list[str]:
    out = subprocess.run(["ss", "-lunpH"], capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if any(f"pid={p}," in line for p in pids)]


@pytest.fixture
def server(tmp_path):
    cfg = dora_runtime_config(tmp_path, mic=True)
    srv = LiveServer(cfg)
    before = set(dora_pids())
    yield srv, cfg, before
    srv.stop()
    time.sleep(0.3)
    leaked = set(dora_pids()) - before
    assert not leaked, f"dora processes left behind: {leaked}"


def test_attach_facts_processes_ports_and_teardown(server, tmp_path):
    srv, cfg, _ = server
    with httpx.Client(base_url=srv.http, timeout=30) as api:
        info, dt = wait_state(api, "attached", 5.0)
        # dora 1.0.1 Node() monkeypatches logging.basicConfig (a `<string>` wrapper that
        # injects handlers=); the bridge restores the stdlib function (§16.2)
        import logging as _logging

        assert _logging.basicConfig.__code__.co_filename.endswith("logging/__init__.py"), (
            _logging.basicConfig.__code__.co_filename
        )
        assert dt <= 5.0
        assert info["dataflow_id"] and info["placeholders"] == ["policy", "viewer", "observer"]
        assert "control_plane" not in info and "token" not in json.dumps(info)
        assert info["bind_host"] == "127.0.0.1" and info["auth"] is False
        assert info["zenoh_connect"] == f"tcp/127.0.0.1:{cfg.dora.zenoh_port}"
        assert info["dataflow_yaml"] == str(tmp_path / "dora" / "mavis_v2.dora.yml")
        # two dora children on OUR ports, none on the machine-global defaults
        plane = srv.runtime.dora.bridge.plane
        pids = plane.child_pids()
        assert len(pids) == 2
        ps = subprocess.run(
            ["ps", "-o", "pid=,args=", "-p", ",".join(map(str, pids))],
            capture_output=True,
            text=True,
        ).stdout
        assert "coordinator" in ps and "daemon" in ps
        listens = sorted(a for p in pids for a in listening(p))
        ports = {int(a.rsplit(":", 1)[1]) for a in listens}
        assert {cfg.dora.coordinator_port, cfg.dora.daemon_port, cfg.dora.zenoh_port} <= ports
        assert not ({6013, 53291} & ports)
        assert all(a.startswith("127.0.0.1:") for a in listens), listens  # §9 loopback only
        runtime_listens = listening(os.getpid())
        assert all(a.startswith("127.0.0.1:") for a in runtime_listens), runtime_listens
        assert udp_sockets(set(pids) | {os.getpid()}) == []  # no UDP, no 224.0.0.224 membership
        ext = telemetry_external(srv)
        assert ext["state"] == "attached" and ext["enabled"] is True
        assert ext["idle_reader"] in ("running", "stale")
        # var_dir holds the rendered YAML + dora's out/; the repo work tree gets no out/
        assert (tmp_path / "dora" / "out").is_dir() and (
            tmp_path / "dora" / "mavis_v2.dora.yml"
        ).is_file()
        repo = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        assert not os.path.exists(os.path.join(repo, "out"))
        # node_config() agreed with the rendered YAML (detail is empty when it matches)
        assert info["detail"] == ""


def test_dora_stop_start_and_daemon_kill_reattach(server):
    srv, cfg, _ = server
    plane = srv.runtime.dora.bridge.plane
    with httpx.Client(base_url=srv.http, timeout=30) as api:
        wait_state(api, "attached", 5.0)
        cli = dora_cli()
        coord = [
            "--coordinator-addr",
            "127.0.0.1",
            "--coordinator-port",
            str(cfg.dora.coordinator_port),
        ]
        t0 = time.monotonic()
        subprocess.run(
            [cli, "stop", "--name", "mavis_v2", "--grace-duration", "1s", *coord],
            capture_output=True,
            text=True,
            timeout=30,
            env=plane._env(),
        )
        _, dt = wait_state(api, "detached", 1.5 + (time.monotonic() - t0))
        assert dt <= 1.5, f"detached after {dt:.2f}s"
        # the bridge re-runs bring-up itself (re-`dora start`); <= 10 s
        info, dt2 = wait_state(api, "attached", 10.0)
        assert info["reattach_count"] == 1
        # kill -9 the daemon: tick + probe silent > 1 s -> detached <= 2 s
        daemon_pid = plane.daemon.pid
        os.kill(daemon_pid, signal.SIGKILL)
        t1 = time.monotonic()
        _, dt3 = wait_state(api, "detached", 2.5)
        assert dt3 <= 2.0, f"detached after {dt3:.2f}s"
        info, _ = wait_state(api, "attached", 15.0)
        assert info["reattach_count"] == 2
        assert plane.daemon.pid != daemon_pid and plane.running  # control plane rebuilt
        assert telemetry_external(srv)["reattach_count"] == 2
        assert time.monotonic() - t1 < 15.0


def test_second_runtime_instance_is_unavailable(server, tmp_path):
    srv, cfg, _ = server
    with httpx.Client(base_url=srv.http, timeout=30) as api:
        wait_state(api, "attached", 5.0)
    from apollo_mavis_v2_runtime.config import DoraConfig
    from apollo_mavis_v2_runtime.dora_bridge.bridge import DoraBridge

    # a second bridge on the SAME var_dir (its own control plane ports would collide too:
    # give it free ports so the lock is the thing under test)
    from dora_bridge.harness import free_port

    cfg2 = DoraConfig(
        enabled=True,
        var_dir=cfg.dora.var_dir,
        coordinator_port=free_port(),
        daemon_port=free_port(),
        zenoh_port=free_port(),
        attach_retry_s=(0.2, 0.5),
    )
    b2 = DoraBridge(cfg2, epoch="second")
    b2.set_outputs([], [], None)
    b2.start()
    try:
        wait_until(
            lambda: b2.state == "unavailable" and "another runtime" in b2.detail,
            10.0,
            "second instance refused",
        )
    finally:
        b2.stop()


def test_control_plane_port_busy_is_unavailable_then_recovers(tmp_path):
    import socket

    cfg = dora_runtime_config(tmp_path)
    blocker = socket.socket()
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", cfg.dora.coordinator_port))
    blocker.listen(1)
    t0 = time.monotonic()
    srv = LiveServer(cfg)
    up = time.monotonic() - t0
    before = set(dora_pids())
    try:
        with httpx.Client(base_url=srv.http, timeout=30) as api:
            t1 = time.monotonic()
            assert api.get("/api/health").status_code == 200
            assert time.monotonic() - t1 < 1.0  # serving never blocks on the bridge

            def unavailable():
                i = get_dora(api)
                failed = i["state"] == "unavailable" and i["detail"] != "starting control plane"
                return i if failed else None

            info = wait_until(unavailable, 10.0, "unavailable")
            assert "coordinator" in info["detail"] or "status" in info["detail"], info["detail"]
            blocker.close()
            _, dt = wait_state(api, "attached", 12.0)
            assert dt <= 10.0, f"attached {dt:.1f}s after the port was freed"
    finally:
        srv.stop()
        time.sleep(0.3)
        assert not (set(dora_pids()) - before)
    # a plain start (no blocker) must be about as fast: the bridge never blocks serving
    cfg2 = dora_runtime_config(tmp_path / "b")
    t2 = time.monotonic()
    srv2 = LiveServer(cfg2)
    up2 = time.monotonic() - t2
    srv2.stop()
    assert abs(up - up2) <= 1.0, (up, up2)


def test_version_mismatch_disables_the_bridge(tmp_path, monkeypatch):
    from apollo_mavis_v2_runtime.dora_bridge import bridge as bridge_mod

    monkeypatch.setattr(
        bridge_mod,
        "check_versions",
        lambda: "dora version mismatch: python dora 1.0.1 vs CLI 1.0.9",
    )
    cfg = dora_runtime_config(tmp_path)
    before = set(dora_pids())
    srv = LiveServer(cfg)
    try:
        with httpx.Client(base_url=srv.http, timeout=30) as api:
            assert api.get("/api/health").status_code == 200
            info = get_dora(api)
            assert info["state"] == "disabled" and "version" in info["detail"]
            assert telemetry_external(srv)["state"] == "disabled"
            assert set(dora_pids()) == before  # nothing spawned
    finally:
        srv.stop()


def test_stdout_diagnostics_are_redirected(tmp_path):
    """<= 1 dora diagnostic line per second on the runtime's stdout while two cameras
    (+ depth) publish (14-dora §8). Measured in a subprocess so fd 1 is really ours."""
    cfg_path = tmp_path / "cfg.json"
    child = f"""
import json, os, sys, time
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, {json.dumps(os.path.dirname(os.path.dirname(__file__)))})
from conftest import LiveServer
from dora_bridge.harness import dora_runtime_config
import httpx
import pathlib
srv = LiveServer(dora_runtime_config(pathlib.Path({json.dumps(str(tmp_path))})))
with httpx.Client(base_url=srv.http, timeout=30) as api:
    for _ in range(100):
        if api.get("/api/dora").json()["state"] == "attached":
            break
        time.sleep(0.1)
    time.sleep({os.environ.get("P12_STDOUT_WINDOW_S", "20")})
    print("PYTHON_PRINT_STILL_VISIBLE", flush=True)
srv.stop()
"""
    cfg_path.write_text(child)
    res = subprocess.run(
        [sys.executable, str(cfg_path)], capture_output=True, text=True, timeout=180
    )
    lines = res.stdout.splitlines()
    diag = [ln for ln in lines if ln.startswith("{") and '"level"' in ln]
    window = float(os.environ.get("P12_STDOUT_WINDOW_S", "20"))
    assert "PYTHON_PRINT_STILL_VISIBLE" in lines, (res.returncode, res.stderr[-2500:])
    assert len(diag) <= window, f"{len(diag)} dora diagnostic lines on stdout in {window:.0f} s"
    log = tmp_path / "dora" / "node-stdout.log"
    assert log.is_file() and log.stat().st_size > 0  # the flood went to the file instead


def test_no_session_publishing_and_probe_nodes(server):
    """Fixed viewpoint before any session (14-dora §7) + the other helper nodes attach."""
    srv, cfg, _ = server
    with httpx.Client(base_url=srv.http, timeout=30) as api:
        info, _ = wait_state(api, "attached", 5.0)
        env = node_env(info["bind_host"], info["zenoh_port"])
        out = cfg.dora.var_dir / "viewer.json"
        t0 = time.monotonic()
        res = subprocess.run(
            [
                sys.executable,
                "-m",
                f"{MOD}.viewer_probe",
                "--daemon-port",
                str(info["daemon_port"]),
                "--camera",
                "view_wrist_cam,grip_wrist_cam",
                "--duration-s",
                "5",
                "--warmup-s",
                "1",
                "--out",
                str(out),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert res.returncode == 0, res.stderr[-800:]
        r = json.loads(out.read_text())
        assert r["errors"] == []
        cam = r["streams"]["cam_view_wrist_cam"]
        assert cam["n"] > 0 and abs(cam["rate_hz"] - 15.0) <= 1.0, cam
        assert r["pose_missing"] == [] and r["pose_ok_frames"] > 0
        lp = r["last_pose"]
        assert len(lp["camera_pose_world"]) == 7 and len(lp["tcp_pose_world"]) == 7
        assert len(lp["q"]) == 8 and lp["pose_source"] == "idle" and lp["session_id"] == ""
        assert r["streams"]["cam_view_wrist_cam_depth"]["n"] > 0  # sim depth sibling
        assert r["streams"]["cam_grip_wrist_cam_depth"]["n"] == 0  # not in depth_cameras
        arm = r["streams"]["arm_state"]
        assert (
            abs(arm["rate_hz"] - cfg.dora.publish.idle_state_hz)
            <= 0.1 * cfg.dora.publish.idle_state_hz
        )
        assert r["arm_state"]["source"] == "idle" and r["arm_state"]["session_id"] == ""
        assert r["arm_state"]["tick"] == -1 and r["arm_state"]["n_values"] == 64
        assert r["sessions"][-1] == {"session_id": None, "state": "idle"}
        assert time.monotonic() - t0 < 30
        # env helper prints the same facts
        res = subprocess.run(
            [sys.executable, "-m", f"{MOD}.env", "--url", srv.http],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert res.returncode == 0
        assert f"export DORA_ZENOH_CONNECT=tcp/127.0.0.1:{cfg.dora.zenoh_port}" in res.stdout
        assert f"export DORA_DAEMON_PORT={info['daemon_port']}" in res.stdout
        assert "DORA_AUTH_TOKEN" not in res.stdout  # loopback: auth off
