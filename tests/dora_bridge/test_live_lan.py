"""Tier 3, LAN (14-dora §9 v0.3): the control plane bound to this host's LAN interface with
auth on, a second daemon ``--machine-id remote`` on the same host, the ``viewer_remote``
placeholder, join idempotency, the 401 for a token-less daemon, and the remote daemon's death.

Skips unless the configured interface (``P12_LAN_IFACE``, default ``wlp38s0``) has an IPv4
address outside the control-box subnets. Ports are free private ones; the daemons this test
starts are killed in ``finally`` and asserted gone.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

import httpx
import pytest
from conftest import LiveServer

from apollo_mavis_v2_runtime.dora_bridge import netaddr
from dora_bridge.harness import (
    dora_cli,
    dora_runtime_config,
    free_port,
    node_env,
    requires_dora,
    wait_until,
)

pytestmark = [pytest.mark.dora, requires_dora]

IFACE = os.environ.get("P12_LAN_IFACE", "wlp38s0")
ARM_IPS = ("192.168.1.201", "192.168.2.219")
MOD = "apollo_mavis_v2_runtime.dora_bridge.nodes"


def lan_ip() -> str:
    ip = netaddr.interface_ipv4(IFACE)
    if ip is None:
        pytest.skip(f"interface {IFACE} has no IPv4 address")
    try:
        netaddr.vet_bind_ip(ip, ARM_IPS)
    except netaddr.BindHostError as exc:
        pytest.skip(str(exc))
    return ip


def listening(pids: set[int]) -> dict[int, list[str]]:
    out = subprocess.run(["ss", "-ltnpH"], capture_output=True, text=True).stdout
    res: dict[int, list[str]] = {p: [] for p in pids}
    for line in out.splitlines():
        for p in pids:
            if f"pid={p}," in line:
                res[p].append(line.split()[3])
    return res


def udp(pids: set[int]) -> list[str]:
    out = subprocess.run(["ss", "-lunpH"], capture_output=True, text=True).stdout
    return [ln for ln in out.splitlines() if any(f"pid={p}," in ln for p in pids)]


def spawn_daemon(
    cli, machine_id, bind_ip, coord_port, q2, z2, *, token: str | None, log, home=None
):
    """A second daemon on this host. ``cwd`` is the log's directory: a dora daemon writes
    ``out/dora-daemon-<machine>.txt`` into its cwd, and pytest's cwd is the repo root
    (``test_attach_facts_processes_ports_and_teardown`` asserts it stays clean)."""
    env = dict(os.environ)
    env.pop("DORA_COORDINATOR_ADDR", None)
    env.pop("DORA_COORDINATOR_PORT", None)
    env["RUST_LOG"] = "info"
    if token:
        env["DORA_AUTH_TOKEN"] = token
    else:
        env.pop("DORA_AUTH_TOKEN", None)
    if home is not None:
        env["HOME"] = str(home)  # no ~/.config/dora/.dora-token to fall back on
    return subprocess.Popen(
        [
            cli,
            "daemon",
            "--machine-id",
            machine_id,
            "--coordinator-addr",
            bind_ip,
            "--coordinator-port",
            str(coord_port),
            "--local-listen-port",
            str(q2),
            "--zenoh-no-multicast",
            "--zenoh-listen",
            f"{bind_ip}:{z2}",
        ],
        env=env,
        cwd=str(pathlib.Path(log).parent),
        stdout=open(log, "ab"),
        stderr=subprocess.STDOUT,
        start_new_session=True,  # noqa: SIM115
    )


def kill(p: subprocess.Popen) -> None:
    import signal

    if p.poll() is None:
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait(timeout=5)


def test_lan_bind_auth_remote_daemon_and_join(tmp_path):
    ip = lan_ip()
    cli = dora_cli()
    cfg = dora_runtime_config(
        tmp_path,
        bind_host=IFACE,
        machines=[{"id": "remote", "placeholders": ["viewer"]}],
        rescan_s=2.0,
    )
    assert cfg.dora.auth_effective is True  # non-loopback -> auth on by default
    srv = LiveServer(cfg)
    daemons: list[subprocess.Popen] = []
    try:
        with httpx.Client(base_url=srv.http, timeout=60) as api:
            info = wait_until(
                lambda: (
                    (i := api.get("/api/dora").json()) and (i if i["state"] == "attached" else None)
                ),
                15.0,
                "attached",
            )
            assert (
                info["bind_host"] == ip and info["coordinator_addr"] == ip and info["auth"] is True
            )
            assert info["zenoh_connect"] == f"tcp/{ip}:{cfg.dora.zenoh_port}"
            assert "token" not in json.dumps(info).lower()
            assert info["machines"] == [
                {
                    "id": "remote",
                    "registered": False,
                    "joined": False,
                    "placeholders": [],
                    "detail": "",
                }
            ]
            token_path = cfg.dora.var_dir / ".dora-token"
            assert token_path.is_file() and len(token_path.read_text().strip()) >= 32
            token = token_path.read_text().strip()
            # -- socket hygiene: coordinator + zenoh on the LAN ip only; node ports on loopback ----
            plane = srv.runtime.dora.bridge.plane
            pids = set(plane.child_pids()) | {os.getpid()}
            listens = listening(pids)
            flat = [a for v in listens.values() for a in v]
            assert (
                f"{ip}:{cfg.dora.coordinator_port}" in flat
                and f"{ip}:{cfg.dora.zenoh_port}" in flat
            )
            assert f"127.0.0.1:{cfg.dora.daemon_port}" in flat
            for a in flat:
                host = a.rsplit(":", 1)[0]
                assert host in (ip, "127.0.0.1"), a
                assert not host.startswith(("192.168.1.", "192.168.2.")), a
            assert udp(pids) == [], udp(pids)  # no UDP, no 224.0.0.224
            # -- a token-less daemon is refused (401) ----------------------------------------------
            bad_home = tmp_path / "nohome"
            bad_home.mkdir()
            bad = spawn_daemon(
                cli,
                "x",
                ip,
                cfg.dora.coordinator_port,
                free_port(),
                free_port(),
                token=None,
                log=tmp_path / "bad_daemon.log",
                home=bad_home,
            )
            daemons.append(bad)
            wait_until(
                lambda: "401" in (tmp_path / "bad_daemon.log").read_text(errors="replace"),
                15.0,
                "401 in the token-less daemon log",
            )
            kill(bad)
            # -- the remote daemon registers: REST-visible within rescan_s, NO restart (§16.1) -----
            q2, z2 = free_port(), free_port()
            remote = spawn_daemon(
                cli,
                "remote",
                ip,
                cfg.dora.coordinator_port,
                q2,
                z2,
                token=token,
                log=tmp_path / "remote_daemon.log",
            )
            daemons.append(remote)
            t0 = time.monotonic()
            info = wait_until(
                lambda: (
                    (i := api.get("/api/dora").json())
                    and (i if i["machines"][0]["registered"] else None)
                ),
                cfg.dora.rescan_s + 5.0,
                "remote registered",
            )
            print(f"\nremote registered after {time.monotonic() - t0:.1f}s")
            assert info["dataflow_restarts"] == 0 and info["machines"][0]["joined"] is False
            # -- explicit join: the dataflow restarts with viewer_remote and WAITS for the remote
            #    consumer (dora 1.0.1 start barrier); the viewer attaches inside the window --------
            r = api.post("/api/dora/machines/remote/join")
            assert r.status_code == 202, r.text
            assert api.post("/api/dora/machines/nope/join").status_code == 404
            window = float(os.environ.get("P12_LAN_WINDOW_S", "10"))
            out = tmp_path / "viewer_remote.json"
            env = node_env(ip, z2)  # the REMOTE daemon's zenoh endpoint
            time.sleep(1.5)  # the restart is in flight: the new dataflow exists, barrier closed
            viewer = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    f"{MOD}.viewer_probe",
                    "--daemon-port",
                    str(q2),
                    "--node-id",
                    "viewer_remote",
                    "--camera",
                    "view_wrist_cam",
                    "--duration-s",
                    str(window),
                    "--warmup-s",
                    "1.0",
                    "--first-timeout-s",
                    "30",
                    "--dump-frames",
                    "--out",
                    str(out),
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            t1 = time.monotonic()
            info = wait_until(
                lambda: (
                    (i := api.get("/api/dora").json())
                    and (i if i["machines"][0]["joined"] and i["state"] == "attached" else None)
                ),
                cfg.dora.join_attach_timeout_s + 10.0,
                "joined + re-attached",
            )
            print(f"joined + attached {time.monotonic() - t1:.1f}s after the viewer started")
            assert info["dataflow_restarts"] == 1
            assert info["machines"][0]["placeholders"] == ["viewer_remote"]
            assert info["machines"][0]["detail"] == "joined"
            yml = (cfg.dora.var_dir / "mavis_v2.dora.yml").read_text()
            assert "  - id: viewer_remote" in yml and "deploy: {machine: remote}" in yml
            # join is idempotent while joined: 202, no further restart
            r = api.post("/api/dora/machines/remote/join")
            assert r.status_code == 202 and r.json()["machines"][0]["joined"] is True
            time.sleep(cfg.dora.rescan_s + 1.0)
            assert api.get("/api/dora").json()["dataflow_restarts"] == 1
            # -- viewer_remote receives frames over the cross-daemon TCP path ---------------
            bridge = srv.runtime.dora.bridge
            ow0 = dict(bridge.slot_overwrites)
            vout, verr = viewer.communicate(timeout=window + 60)
            ow = {
                k: v - ow0.get(k, 0)
                for k, v in bridge.slot_overwrites.items()
                if v != ow0.get(k, 0)
            }
            assert viewer.returncode == 0, verr[-1500:]
            rep = json.loads(out.read_text())
            cam = rep["streams"]["cam_view_wrist_cam"]
            print("remote viewer:", json.dumps({k: v for k, v in cam.items() if k != "gap_list"}))
            # attribution of any gap: a runtime-side drop shows as a slot overwrite; a
            # consumer-side one as dora's "Discarding event ... queue size limit" on the
            # viewer's stdout (the local + remote viewers gapped at the same seq until the
            # probe pre-paid pyarrow's first to_numpy(), 2026-09-08)
            print("remote gaps:", cam["gaps"], cam.get("gap_list"), "runtime slot overwrites:", ow)
            (tmp_path / "viewer_remote.stdout").write_text(vout)
            import re

            for ln in vout.splitlines():
                if "Discarding" in ln:
                    m = re.search(r'"timestamp":"([^"]+)".*input `([^`]+)`', ln)
                    print(
                        "  discard", m.group(1)[11:23] if m else "?", m.group(2) if m else ln[:80]
                    )
            gl = cam.get("gap_list") or []
            if gl and rep.get("frames"):
                import datetime as _dt

                fr = {f["seq"]: f for f in rep["frames"] if "seq" in f}
                for a, b in gl:
                    fa, fb = fr.get(a), fr.get(b)
                    if fa and fb:
                        wa, wb = (
                            _dt.datetime.fromtimestamp(f["wall"], _dt.UTC).strftime("%H:%M:%S.%f")[
                                :12
                            ]
                            if "wall" in f
                            else "?"
                            for f in (fa, fb)
                        )
                        print(
                            f"  gap {a}->{b}: t {fa['t']:.3f} -> {fb['t']:.3f} (wall {wa} -> {wb})"
                        )
            print(
                "viewer stdout discards:",
                vout.count("Discarding"),
                "| tail:",
                vout[-300:].replace("\n", " "),
            )
            assert rep["errors"] == [] and cam["n"] > 0, cam
            assert ow == {}, ow
            assert cam["lat_ms"]["p99"] <= 10.0, cam["lat_ms"]
            assert rep["pose_missing"] == []
            assert cam["gaps"] == 0, cam
            # -- the remote daemon dies: NO restart (§16.1 "remote daemon loss") - the machine is
            #    marked lost (registered/joined false), its placeholders stay in the running
            #    dataflow, local streams are unaffected ------------------------------------------

            kill(remote)
            local_out = tmp_path / "viewer_local.json"
            res = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    f"{MOD}.viewer_probe",
                    "--daemon-port",
                    str(cfg.dora.daemon_port),
                    "--camera",
                    "view_wrist_cam",
                    "--duration-s",
                    "4",
                    "--warmup-s",
                    "0.5",
                    "--out",
                    str(local_out),
                ],
                env=node_env(ip, cfg.dora.zenoh_port),
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert res.returncode == 0, res.stderr[-800:]
            lcam = json.loads(local_out.read_text())["streams"]["cam_view_wrist_cam"]
            assert lcam["n"] > 0 and lcam["gaps"] == 0, lcam
            print(
                "local viewer after the remote death:",
                json.dumps({k: lcam[k] for k in ("n", "gaps", "rate_hz")}),
            )
            # the coordinator stops answering the CLI for ~50 s after the loss (dora 1.0.1,
            # 14-dora §16.1 "remote daemon loss"; the bridge backs its rescan off to 10 s while
            # `dora doctor` fails), so `registered` lags that long - measured 47.6 s
            t_dead = time.monotonic()
            wait_until(
                lambda: api.get("/api/dora").json()["machines"][0]["registered"] is False,
                90.0,
                "remote unregistered",
            )
            print(f"remote unregistered {time.monotonic() - t_dead:.1f}s after the kill")
            info = api.get("/api/dora").json()
            m = info["machines"][0]
            assert m["joined"] is False and "unregistered" in m["detail"], m
            assert info["dataflow_restarts"] == 1 and info["state"] == "attached", info
            assert plane.dataflow_running() is True
            assert "  - id: viewer_remote" in (cfg.dora.var_dir / "mavis_v2.dora.yml").read_text()
            # ... it comes back: registered again, still no restart, then an explicit re-join
            #    re-deploys viewer_remote on the new daemon (restart 2)
            q3, z3 = free_port(), free_port()
            remote2 = spawn_daemon(
                cli,
                "remote",
                ip,
                cfg.dora.coordinator_port,
                q3,
                z3,
                token=token,
                log=tmp_path / "remote_daemon2.log",
            )
            daemons.append(remote2)
            t_back = time.monotonic()
            wait_until(  # the coordinator's ~50 s 429 window may fall here instead (§16.1)
                lambda: api.get("/api/dora").json()["machines"][0]["registered"] is True,
                90.0,
                "re-registered",
            )
            print(f"re-registered {time.monotonic() - t_back:.1f}s after the new daemon started")
            time.sleep(cfg.dora.rescan_s + 1.0)
            info = api.get("/api/dora").json()
            assert info["dataflow_restarts"] == 1 and info["machines"][0]["joined"] is False
            r = api.post("/api/dora/machines/remote/join")
            assert r.status_code == 202, r.text
            time.sleep(1.5)
            out2 = tmp_path / "viewer_remote2.json"
            viewer2 = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    f"{MOD}.viewer_probe",
                    "--daemon-port",
                    str(q3),
                    "--node-id",
                    "viewer_remote",
                    "--camera",
                    "view_wrist_cam",
                    "--duration-s",
                    "4",
                    "--warmup-s",
                    "1.0",
                    "--first-timeout-s",
                    "30",
                    "--out",
                    str(out2),
                ],
                env=node_env(ip, z3),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            info = wait_until(
                lambda: (
                    (i := api.get("/api/dora").json())
                    and (i if i["machines"][0]["joined"] and i["state"] == "attached" else None)
                ),
                cfg.dora.join_attach_timeout_s + 10.0,
                "re-joined + attached",
            )
            assert info["dataflow_restarts"] == 2 and info["machines"][0]["detail"] == "joined"
            _, verr2 = viewer2.communicate(timeout=60)
            assert viewer2.returncode == 0, verr2[-800:]
            cam2 = json.loads(out2.read_text())["streams"]["cam_view_wrist_cam"]
            assert cam2["n"] > 0 and cam2["gaps"] == 0, cam2
            print(
                "re-joined remote viewer:",
                json.dumps({k: cam2[k] for k in ("n", "gaps", "rate_hz")}),
            )
    finally:
        for d in daemons:
            kill(d)
        srv.stop()
        time.sleep(0.3)
        for d in daemons:
            assert d.poll() is not None
