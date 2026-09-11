"""DoraBridge state machine over an in-memory node (14-dora §2.4, §8, §10 tier 2)."""

from __future__ import annotations

import time

import pytest
from apollo_mavis_v2_core.protocol import external as ext

from apollo_mavis_v2_runtime.config import DoraConfig, DoraMachineConfig
from apollo_mavis_v2_runtime.dora_bridge import codec
from apollo_mavis_v2_runtime.dora_bridge.bridge import PREWARM_KEY, DoraBridge
from dora_bridge.fake_node import FakeControlPlane, FakeNode

pa = pytest.importorskip("pyarrow")


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def make_bridge(tmp_path, *, cfg=None, plane_kwargs=None, node=None, versions=lambda: None):
    cfg = cfg or DoraConfig(
        enabled=True,
        var_dir=tmp_path / "dora",
        bus_poll_s=0.001,
        attach_retry_s=(0.05, 0.2),
        rescan_s=0.5,
    )
    holder = {}
    clock = Clock()

    def cp_factory(c, var_dir, arm_ips):
        holder["plane"] = FakeControlPlane(c, var_dir, arm_ips, **(plane_kwargs or {}))
        return holder["plane"]

    # v1.3: the rendered dataflow carries a policy_action_<arm> input per configured arm and
    # the bridge's attach check expects exactly those rows (FakeNode mirrors them)
    arm_ids = ["view", "grip"]
    fake_node = node or FakeNode(
        inputs=[*ext.RUNTIME_INPUTS, *(ext.policy_arm_action_input_id(a) for a in arm_ids)]
    )
    holder["node"] = fake_node
    bridge = DoraBridge(
        cfg,
        epoch="ep-test",
        control_plane_factory=cp_factory,
        node_factory=lambda nid, port: fake_node,
        versions_ok=versions,
        clock=clock,
        barrier_probe=lambda timeout: holder["plane"].barrier_open,
    )
    bridge.set_outputs(["view_wrist_cam"], ["view_wrist_cam"], "mic_view", arm_ids)
    holder["clock"] = clock
    return bridge, holder


def wait(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.005)
    return False


def test_disabled_paths_never_start_a_thread(tmp_path):
    b, _ = make_bridge(tmp_path, cfg=DoraConfig(enabled=False, var_dir=tmp_path / "d"))
    b.start()
    assert b.state == "disabled" and "enabled" in b.detail and b._thread is None
    assert b.try_publish("arm_state", lambda: (pa.array([1.0]), {})) is False
    b2, _ = make_bridge(
        tmp_path, versions=lambda: "dora version mismatch: python 1.0.1 vs CLI 1.0.2"
    )
    b2.start()
    assert b2.state == "disabled" and "version" in b2.detail
    b3, _ = make_bridge(
        tmp_path, cfg=DoraConfig(enabled=True, bind_host="0.0.0.0", var_dir=tmp_path / "d3")
    )
    b3.start()
    assert b3.state == "disabled" and "0.0.0.0" in b3.detail
    cfg4 = DoraConfig(enabled=True, bind_host="192.168.1.11", var_dir=tmp_path / "d4")
    b4 = DoraBridge(
        cfg4,
        epoch="e",
        arm_ips=("192.168.1.201", "192.168.2.219"),
        control_plane_factory=lambda c, d, ips: FakeControlPlane(c, d, ips),
        versions_ok=lambda: None,
    )
    b4.start()
    assert b4.state == "disabled" and "control-box subnet" in b4.detail
    for x in (b, b2, b3, b4):
        x.stop()
        assert x.state == "closed"


def test_attach_prewarms_cameras_and_stamps_common_metadata(tmp_path):
    b, h = make_bridge(tmp_path)
    b.start()
    try:
        assert wait(lambda: b.state == "attached"), b.detail
        node: FakeNode = h["node"]
        plane: FakeControlPlane = h["plane"]
        assert plane.starts == 1 and plane.dataflow_starts == 1 and plane.validated
        assert (tmp_path / "dora" / "mavis_v2.dora.yml").is_file()
        assert (tmp_path / "dora" / "mavis_runtime.lock").is_file()
        ids = node.sent_ids()
        assert ids[:2] == ["cam_view_wrist_cam", "cam_view_wrist_cam_depth"]  # one dummy each
        assert all(m.get(PREWARM_KEY) for _, _, m in node.sent[:2])
        assert b.try_publish_now(
            "arm_state", pa.array([1.0, 2.0]), {"tick": 5, "layout": ["a", "b"]}
        )
        assert wait(lambda: node.sent_for("arm_state"))
        _, meta = node.sent_for("arm_state")[0]
        assert meta[ext.META_SCHEMA] == 1 and meta[ext.META_EPOCH] == "ep-test"
        assert meta[ext.META_SESSION_ID] == "" and meta[ext.META_SEQ] == 1
        assert isinstance(meta[ext.META_T_MONO], float) and isinstance(
            meta[ext.META_WALLCLOCK_NS], int
        )
        assert meta["tick"] == 5 and meta["layout"] == ["a", "b"]
        # depth-1 slot: two quick publishes -> the newest wins, seq keeps counting
        b.try_publish_now("telemetry", pa.array(["a"]))
        b.try_publish_now("telemetry", pa.array(["b"]))
        assert wait(lambda: node.sent_for("telemetry"))
        time.sleep(0.05)
        tele = node.sent_for("telemetry")
        assert len(tele) >= 1 and tele[-1][0].to_pylist() == ["b"]
        info = b.info()
        assert info.state == "attached" and info.placeholders == ["policy", "viewer", "observer"]
        assert (
            info.dataflow_id == "df-1" and info.zenoh_connect == f"tcp/127.0.0.1:{b.cfg.zenoh_port}"
        )
        st = b.status()
        assert st.enabled and st.state == "attached" and st.reattach_count == 0
        assert st.publish_hz.get("arm_state") is not None
    finally:
        b.stop()
    assert (
        h["plane"].downs == 1 and h["plane"].reaps >= 1 and not (tmp_path / "dora" / "out").exists()
    )


def test_unavailable_control_plane_retries_with_backoff(tmp_path):
    b, h = make_bridge(tmp_path, plane_kwargs={"start_fails": 2})
    b.start()
    try:
        assert wait(lambda: h["plane"].starts >= 1)
        assert b.state == "unavailable" and "port busy" in b.detail
        assert wait(lambda: b.state == "attached", 5.0), b.detail
        assert h["plane"].starts == 3
    finally:
        b.stop()


def test_stop_error_none_and_pulse_silence_detach_then_reattach(tmp_path):
    b, h = make_bridge(tmp_path)
    b.start()
    try:
        node: FakeNode = h["node"]
        assert wait(lambda: b.state == "attached")
        closed = []
        b.register_input(ext.IN_POLICY_SPEC, lambda ev: closed.append(ev["type"]))
        node.push({"type": "STOP", "id": "MANUAL"})
        assert wait(lambda: b.state == "detached")
        assert b.reattach_count == 1 and "INPUT_CLOSED" in closed  # handlers hear the detach
        assert b.try_publish_now("arm_state", pa.array([0.0])) is False  # no side effect
        assert wait(lambda: b.state == "attached", 3.0)
        node.push({"type": "ERROR", "error": "Timeout event stream error: Receiver timed out"})
        time.sleep(0.1)
        assert b.state == "attached"  # idle timeout is not a fault
        node.push({"type": "ERROR", "error": "fatal event stream error: daemon channel broken"})
        assert wait(lambda: b.state == "detached")
        assert wait(lambda: b.state == "attached", 3.0)
        # pulse watchdog: tick + probe silent > 1 s on the bridge's clock
        node.push_input(ext.IN_TICK)
        time.sleep(0.05)
        h["clock"].t += 1.5
        assert wait(lambda: b.state == "detached", 2.0)
        assert b.reattach_count == 3
        assert wait(lambda: b.state == "attached", 3.0)
        node.fail_send = True
        b.try_publish_now("arm_state", pa.array([0.0]))
        assert wait(lambda: b.state == "detached", 2.0)
        node.fail_send = False
    finally:
        b.stop()


def test_inbound_validation_and_dispatch(tmp_path):
    b, h = make_bridge(tmp_path)
    b.start()
    try:
        node: FakeNode = h["node"]
        assert wait(lambda: b.state == "attached")
        got = []
        b.register_input(ext.IN_POLICY_ACTION, lambda ev: got.append(ev))
        node.push_input(
            ext.IN_POLICY_ACTION, pa.array([0.0]), {"mavis_schema": 2, "seq": 1, "client": "c"}
        )
        node.push_input(
            ext.IN_POLICY_ACTION, pa.array([0.0]), {"mavis_schema": 1, "seq": 5, "client": "c"}
        )
        node.push_input(
            ext.IN_POLICY_ACTION, pa.array([0.0]), {"mavis_schema": 1, "seq": 5, "client": "c"}
        )
        node.push_input(
            ext.IN_POLICY_ACTION, pa.array([0.0]), {"mavis_schema": 1, "seq": 6, "client": "c"}
        )
        node.push_input(
            ext.IN_POLICY_ACTION, pa.array([0.0]), {"mavis_schema": 1, "seq": 1, "client": "other"}
        )
        node.push_input("weights_reload", pa.array([0.0]), {})
        assert wait(lambda: len(got) == 3 and b.dropped_inputs == 3)
        assert [g["metadata"]["seq"] for g in got] == [5, 6, 1]
        # a restarted client begins at seq 1 again: accepted, counter reset
        node.push_input(
            ext.IN_POLICY_ACTION, pa.array([0.0]), {"mavis_schema": 1, "seq": 1, "client": "c"}
        )
        node.push_input(
            ext.IN_POLICY_ACTION, pa.array([0.0]), {"mavis_schema": 1, "seq": 2, "client": "c"}
        )
        assert wait(lambda: len(got) == 5 and b.dropped_inputs == 3)
        node.push({"type": "INPUT_CLOSED", "id": ext.IN_POLICY_ACTION})
        assert wait(lambda: any(g["type"] == "INPUT_CLOSED" for g in got))
        # v1.3: a per-arm input is just another registered id (its own seq counter per
        # client); an unregistered per-arm id is dropped like any unknown input
        per_arm = []
        b.register_input(ext.policy_arm_action_input_id("grip"), lambda ev: per_arm.append(ev))
        dropped = b.dropped_inputs
        node.push_input(
            ext.policy_arm_action_input_id("grip"),
            pa.array([0.0] * 8),
            {"mavis_schema": 1, "seq": 1, "client": "c"},
        )
        node.push_input(
            ext.policy_arm_action_input_id("nope"),
            pa.array([0.0] * 8),
            {"mavis_schema": 1, "seq": 1, "client": "c"},
        )
        assert wait(lambda: len(per_arm) == 1 and b.dropped_inputs == dropped + 1)
        assert per_arm[0]["id"] == "policy_action_grip"
        assert b.detail == "" or "inputs" not in b.detail  # the attach check knows the per-arm rows
    finally:
        b.stop()


def test_join_protocol_renders_placeholders_and_rescan_drops_a_vanished_daemon(tmp_path):
    cfg = DoraConfig(
        enabled=True,
        var_dir=tmp_path / "dora",
        bus_poll_s=0.001,
        rescan_s=0.2,
        attach_retry_s=(0.05, 0.2),
        join_attach_timeout_s=1.0,
        machines=[DoraMachineConfig(id="remote", placeholders=["viewer"])],
    )
    b, h = make_bridge(tmp_path, cfg=cfg)
    b.start()
    try:
        plane: FakeControlPlane = h["plane"]
        assert wait(lambda: b.state == "attached")
        info = b.info()
        assert info.machines[0].id == "remote" and info.machines[0].registered is False
        assert "viewer_remote" not in (tmp_path / "dora" / "mavis_v2.dora.yml").read_text()
        assert b.request_join("nope") is False
        # join before the daemon is registered: refused with a detail, no restart
        assert b.request_join("remote") is True
        assert wait(lambda: "join refused" in b.machines_info()[0].detail, 3.0)
        assert b.dataflow_restarts == 0
        # the daemon registers: REST-visible within a rescan, but NO restart (§16.1)
        plane.machines.add("remote")
        end_t = time.monotonic() + 3.0
        while not b.machines_info()[0].registered and time.monotonic() < end_t:
            h["clock"].t += 1.0  # the periodic rescan runs on the bridge's (fake) clock
            time.sleep(0.02)
        assert b.machines_info()[0].registered is True and b.dataflow_restarts == 0
        assert b.machines_info()[0].joined is False
        # explicit join: restart with viewer_remote, barrier open -> attached again
        assert b.request_join("remote") is True
        assert wait(lambda: b.dataflow_restarts == 1, 3.0)
        assert wait(lambda: b.state == "attached", 3.0)
        text = (tmp_path / "dora" / "mavis_v2.dora.yml").read_text()
        assert "viewer_remote" in text and "observer_remote" not in text
        m = b.machines_info()[0]
        assert m.joined is True and m.placeholders == ["viewer_remote"] and m.detail == "joined"
        assert plane.stops >= 1 and plane.dataflow_starts == 2
        assert b.request_join("remote") is True  # idempotent while joined
        time.sleep(0.3)
        assert b.dataflow_restarts == 1
        # the remote daemon dies: marked LOST (registered/joined false, detail), but NO restart -
        # its placeholders stay in the running dataflow until the next restart (§16.1)
        plane.machines.clear()
        end_t = time.monotonic() + 3.0
        while b.machines_info()[0].registered and time.monotonic() < end_t:
            h["clock"].t += 1.0
            time.sleep(0.02)
        m = b.machines_info()[0]
        assert m.registered is False and m.joined is False and "unregistered" in m.detail
        assert b.dataflow_restarts == 1 and b.state == "attached"
        assert "viewer_remote" in (tmp_path / "dora" / "mavis_v2.dora.yml").read_text()
        assert b.request_join("remote") is True  # a join while lost is NOT the idempotent no-op
        assert wait(lambda: "join refused" in b.machines_info()[0].detail, 3.0)
        assert b.dataflow_restarts == 1
        # ... it comes back and re-joins: that restart re-deploys viewer_remote on the new daemon
        plane.machines.add("remote")
        end_t = time.monotonic() + 3.0
        while not b.machines_info()[0].registered and time.monotonic() < end_t:
            h["clock"].t += 1.0
            time.sleep(0.02)
        assert b.machines_info()[0].registered is True and b.dataflow_restarts == 1
        starts_before = plane.dataflow_starts
        assert b.request_join("remote") is True
        assert wait(lambda: b.dataflow_restarts == 2, 3.0)
        assert wait(lambda: b.state == "attached", 3.0)
        m = b.machines_info()[0]
        assert m.joined is True and m.detail == "joined", m
        # exactly ONE start: the re-attach inside the join must not render the machine out again
        assert plane.dataflow_starts == starts_before + 1, (plane.dataflow_starts, starts_before)
        assert "viewer_remote" in (tmp_path / "dora" / "mavis_v2.dora.yml").read_text()
        # lost again, then a restart for another reason (a detach) renders it out
        plane.machines.clear()
        end_t = time.monotonic() + 3.0
        while b.machines_info()[0].registered and time.monotonic() < end_t:
            h["clock"].t += 1.0
            time.sleep(0.02)
        assert b.machines_info()[0].joined is False and b.dataflow_restarts == 2
        n_re = b.reattach_count
        h["node"].push({"type": "STOP", "id": "MANUAL"})  # detach -> the re-attach renders it out
        assert wait(lambda: b.state == "attached" and b.reattach_count == n_re + 1, 5.0)
        assert "viewer_remote" not in (tmp_path / "dora" / "mavis_v2.dora.yml").read_text()
        assert "dropped" in b.machines_info()[0].detail
        # a join whose consumer never attaches (barrier closed) expires and rolls back
        plane.machines.add("remote")
        plane.barrier_open = False
        h["clock"].t += 1.0
        assert wait(lambda: b.machines_info()[0].registered, 3.0)
        assert b.request_join("remote") is True
        end_t = time.monotonic() + 5.0
        while "timed out" not in b.machines_info()[0].detail and time.monotonic() < end_t:
            h["clock"].t += 0.6  # the barrier wait is on the bridge's clock
            time.sleep(0.02)
        assert "timed out" in b.machines_info()[0].detail
        assert wait(lambda: b.state == "attached", 3.0)
        assert b.machines_info()[0].joined is False
        assert "viewer_remote" not in (tmp_path / "dora" / "mavis_v2.dora.yml").read_text()
    finally:
        b.stop()


def test_second_instance_is_refused_by_the_lock(tmp_path):
    b1, h1 = make_bridge(tmp_path)
    b1.start()
    try:
        assert wait(lambda: b1.state == "attached")
        b2, h2 = make_bridge(tmp_path, node=FakeNode())
        b2.start()
        try:
            assert wait(lambda: b2.state == "unavailable" and "another runtime" in b2.detail, 3.0)
            assert not h2["node"].sent
        finally:
            b2.stop()
    finally:
        b1.stop()


def test_event_fifo_keeps_order_and_json_payloads(tmp_path):
    b, h = make_bridge(tmp_path)
    b.start()
    try:
        node: FakeNode = h["node"]
        assert wait(lambda: b.state == "attached")
        for i in range(5):
            b.publish_event(
                "events", (lambda i=i: (codec.encode_json({"i": i}), {"kind": "collision"}))
            )
        assert wait(lambda: len(node.sent_for("events")) == 5)
        assert [codec.decode_json_text(v) for v, _ in node.sent_for("events")] == [
            '{"i": 0}',
            '{"i": 1}',
            '{"i": 2}',
            '{"i": 3}',
            '{"i": 4}',
        ]
        assert [m["seq"] for _, m in node.sent_for("events")] == [1, 2, 3, 4, 5]
    finally:
        b.stop()
