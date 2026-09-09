"""Tier 3 continued: microphone stream, idle arm reader over the FakeSDK hardware workcell,
CPU budget of the two bridge threads, and the two-hop RTT instrument (14-dora §10, §12)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time

import httpx
import pytest
from conftest import LiveServer

from dora_bridge.harness import (
    dora_pids,
    dora_runtime_config,
    load_fake_xarm_api,
    node_env,
    requires_dora,
    wait_until,
)

pytestmark = [pytest.mark.dora, requires_dora]

MOD = "apollo_mavis_v2_runtime.dora_bridge.nodes"
CLK_TCK = os.sysconf("SC_CLK_TCK")


def thread_cpu_seconds(name_prefixes: tuple[str, ...]) -> dict[str, float]:
    """utime+stime of this process's threads whose name starts with a prefix (/proc, no psutil)."""
    out: dict[str, float] = {}
    for t in threading.enumerate():
        tid = getattr(t, "native_id", None)
        if tid is None or not t.name.startswith(name_prefixes):
            continue
        try:
            with open(f"/proc/self/task/{tid}/stat") as fh:
                fields = fh.read().rsplit(")", 1)[1].split()
        except OSError:
            continue
        out[t.name] = (int(fields[11]) + int(fields[12])) / CLK_TCK
    return out


def attached(api: httpx.Client, timeout: float = 5.0) -> dict:
    def probe():
        i = api.get("/api/dora").json()
        return i if i["state"] == "attached" else None

    return wait_until(probe, timeout, "attached")


def test_microphone_stream_and_bridge_cpu_budget(tmp_path):
    cfg = dora_runtime_config(tmp_path, mic=True)
    srv = LiveServer(cfg)
    before = set(dora_pids())
    try:
        with httpx.Client(base_url=srv.http, timeout=30) as api:
            info = attached(api)
            env = node_env(info["bind_host"], info["zenoh_port"])
            window = float(os.environ.get("P12_MIC_WINDOW_S", "10"))
            probe = f"""
import json, os, sys, time
os.environ.setdefault("RUST_LOG", "error")
from dora import Node
node = Node("observer", daemon_port={info["daemon_port"]})
seqs, rms, n_other, t_first = [], [], 0, None
while True:
    ev = node.next(timeout=2.0)
    now = time.monotonic()
    if t_first is not None and now - t_first > {window}: break
    if ev is None or ev["type"] == "STOP": break
    if ev["type"] != "INPUT": continue
    if t_first is None: t_first = now
    if ev["id"] == "mic_mic_view":
        m = ev["metadata"]; v = ev["value"]
        seqs.append(int(m["block_seq"])); rms.append(float(m["rms_dbfs"]))
        assert len(v) == 1920 and str(v.type) == "float", (len(v), v.type)
        assert m["sample_rate"] == 48000 and m["channels"] == 1 and m["sample_type"] == "f32"
    elif ev["id"] == "telemetry" and len(seqs) % 25 == 0 and rms:
        # parse 1 in 25 telemetry frames (a lean consumer): same-frame rms agreement
        tele = json.loads(ev["value"][0].as_py())
        if tele["microphone"]:
            # same-frame agreement: newest mic frame rms == telemetry's when the seqs match
            if tele["microphone"]["seq"] == seqs[-1]:
                assert abs(tele["microphone"]["rms_dbfs"] - rms[-1]) < 1e-6
    else:
        n_other += 1
gaps = sum(1 for a, b in zip(seqs, seqs[1:]) if b != a + 1)
rate = (len(seqs) - 1) / (time.monotonic() - t_first)
print(json.dumps({{"n": len(seqs), "gaps": gaps, "rate_hz": rate, "other": n_other}}))
"""
            # CPU of the bridge threads over the same window
            cpu0 = thread_cpu_seconds(("dora-bus", "dora-publisher"))
            t0 = time.monotonic()
            res = subprocess.run(
                [sys.executable, "-c", probe],
                env=env,
                capture_output=True,
                text=True,
                timeout=window + 60,
            )
            elapsed = time.monotonic() - t0
            cpu1 = thread_cpu_seconds(("dora-bus", "dora-publisher"))
            assert res.returncode == 0, res.stderr[-1500:]
            r = json.loads([ln for ln in res.stdout.splitlines() if ln.startswith('{"n"')][-1])
            overwrites = dict(srv.runtime.dora.bridge.slot_overwrites)
            assert r["gaps"] == 0 and abs(r["rate_hz"] - cfg.telemetry_hz) <= 1.0, (r, overwrites)
            assert r["n"] >= window * cfg.telemetry_hz * 0.9
            busy = sum(cpu1.get(k, 0.0) - cpu0.get(k, 0.0) for k in cpu1)
            pct = 100.0 * busy / elapsed
            print(f"\nbridge threads CPU: {pct:.1f} % of one core over {elapsed:.1f} s ({cpu1})")
            assert pct <= 10.0, f"dora-bus + dora-publisher used {pct:.1f} % of a core"
    finally:
        srv.stop()
        time.sleep(0.3)
        assert not (set(dora_pids()) - before)


def test_rtt_two_hops_through_fake_policy_echo(tmp_path, procs):
    """obs_state (30 Hz) -> fake_policy --mode echo -> policy_action, >= 1000 round trips
    (14-dora §12: p50 <= 1.5 ms, p99 <= 5 ms, max <= 60 ms, 0 lost). The rtt_probe plays
    the runtime's role on the runtime's own private dataflow (the runtime is not attached:
    dynamic node ids are not exclusive, so the probe must own `mavis_runtime` alone)."""
    cfg = dora_runtime_config(tmp_path)
    cfg.dora.enabled = True
    from apollo_mavis_v2_runtime.dora_bridge.control_plane import DoraControlPlane
    from apollo_mavis_v2_runtime.dora_bridge.dataflow import render_dataflow

    plane = DoraControlPlane(cfg.dora, cfg.dora.var_dir)
    cfg.dora.var_dir.mkdir(parents=True, exist_ok=True)
    plane.start()
    before = set(dora_pids()) - set(plane.child_pids())
    try:
        yml = cfg.dora.var_dir / "rtt.dora.yml"
        yml.write_text(render_dataflow(cfg.dora, [], None, sys.executable))
        plane.validate(yml)
        plane.start_dataflow(yml)
        env = node_env(plane.bind_ip, cfg.dora.zenoh_port)
        fake = procs.spawn(
            [
                sys.executable,
                "-m",
                f"{MOD}.fake_policy",
                "--daemon-port",
                str(cfg.dora.daemon_port),
                "--mode",
                "echo",
                "--rate-hz",
                "1000",
                "--policy-id",
                "echo",
            ],
            env=env,
            log=cfg.dora.var_dir / "fake_policy.log",
        )
        count = int(os.environ.get("P12_RTT_COUNT", "1000"))
        out = cfg.dora.var_dir / "rtt.json"
        res = subprocess.run(
            [
                sys.executable,
                "-m",
                f"{MOD}.rtt_probe",
                "--daemon-port",
                str(cfg.dora.daemon_port),
                "--count",
                str(count),
                "--hz",
                "30",
                "--out",
                str(out),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=count / 30 + 60,
        )
        assert res.returncode == 0, res.stderr[-1500:]
        r = json.loads(out.read_text())
        print("\nrtt:", json.dumps(r))
        assert r["spec_seen"] and r["lost"] == 0 and r["replies"] == count
        assert (
            r["rtt_ms"]["p50"] <= 1.5 and r["rtt_ms"]["p99"] <= 5.0 and r["rtt_ms"]["max"] <= 60.0
        )
        assert fake.poll() is None
    finally:
        plane.shutdown()
        time.sleep(0.3)
        assert not (set(dora_pids()) - before)


def test_idle_arm_reader_over_fake_hardware(tmp_path):
    """IdleArmReader (driver source) against the FakeSDK hardware workcell: zero writes between
    sessions, paused before BRINGUP / resumed after TEARDOWN, stale + frozen on a dead report
    stream, reconnect afterwards (14-dora §4.2, §8)."""
    pytest.importorskip("apollo_mavis_v2_hardware")
    FakeXArmAPI = load_fake_xarm_api()
    from apollo_mavis_v2_core import WorkcellConfig

    from apollo_mavis_v2_runtime.config import DoraPublishConfig, RuntimeConfig

    cfg = dora_runtime_config(tmp_path, publish=DoraPublishConfig(idle_source="driver"))
    hw = WorkcellConfig.model_validate(
        {
            "kind": "hardware",
            "digital_twin_scene": "mavis_v2",
            "arms": [
                {"id": "view", "ip": "192.168.2.219", "base_in_world": {}, "gripper": "none"},
                {"id": "grip", "ip": "192.168.1.201", "base_in_world": {}, "gripper": "xarm_g2"},
            ],
            "cameras": [],
            "safety": {"enabled": True},
        }
    )
    cfg = RuntimeConfig(**{**cfg.model_dump(), "workcells": {**cfg.workcells, "hardware": hw}})
    cfg.hardware_monitor.enabled = False  # the driver source is the thing under test
    apis: dict[str, object] = {}

    def api_factory(ip, **kw):
        api = FakeXArmAPI(
            ip, auto_report_hz=100.0, has_rail=True, rail_homed=True, rail_enabled=True, **kw
        )
        apis[ip] = api
        return api

    from apollo_mavis_v2_runtime.runtime import Runtime

    rt = Runtime(cfg)
    rt.manager.driver_api_factory = api_factory
    rt.start()
    try:
        wait_until(lambda: rt.dora.bridge.attached, 10.0, "attached")
        reader = rt.dora.idle_reader
        assert reader is not None and type(reader.source).__name__ == "DriverIdleSource"
        wait_until(lambda: len(apis) == 2 and reader.snapshots > 3, 10.0, "both boxes read")
        writes = {
            "clean_error",
            "clean_warn",
            "motion_enable",
            "set_mode",
            "set_state",
            "set_servo_angle_j",
            "set_gripper_enable",
            "set_gripper_mode",
            "set_gripper_speed",
            "set_gripper_position",
            "set_gripper_g2_position",
            "set_linear_track_enable",
            "set_linear_track_speed",
            "set_linear_track_back_origin",
            "set_linear_track_pos",
            "set_tcp_load",
            "set_collision_sensitivity",
            "save_conf",
        }
        for api in apis.values():
            assert not (set(api.call_names()) & writes), set(api.call_names()) & writes
            assert api.ctor_args.get("report_type") == "real"
        got = rt.bus.snapshot.get()[0]
        assert (
            got.tick == -1
            and set(got.arms) == {"view", "grip"}
            and not any(s.stale for s in got.arms.values())
        )
        # pause / resume around a session: the manager calls these
        rt.dora.before_bringup()
        assert (
            reader.paused
            and all(not d for d in reader.source.drivers.values())
            or not reader.source.drivers
        )
        for api in apis.values():
            assert "disconnect" in api.call_names()
        n = reader.snapshots
        time.sleep(0.3)
        assert reader.snapshots == n
        t0 = time.monotonic()
        rt.dora.after_teardown()
        wait_until(lambda: reader.snapshots > n, 1.0, "resumed within 1 s")
        assert time.monotonic() - t0 <= 1.0
        # dead report stream on one box -> that arm stale + q frozen; the other keeps flowing
        view_api = apis["192.168.2.219"]
        view_api.stop_fakes()
        wait_until(lambda: rt.bus.snapshot.get()[0].arms["view"].stale, 3.0, "view stale")
        snap = rt.bus.snapshot.get()[0]
        assert not snap.arms["grip"].stale
        q_frozen = snap.arms["view"].q.copy()
        time.sleep(0.3)
        assert (rt.bus.snapshot.get()[0].arms["view"].q == q_frozen).all()
        assert reader.status == "stale"
        for api in apis.values():
            assert not (set(api.call_names()) & writes)
    finally:
        rt.stop()
    assert all("disconnect" in api.call_names() for api in apis.values())
