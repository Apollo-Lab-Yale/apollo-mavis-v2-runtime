"""HardwareProbe (phase-11): TCP connect-and-close reachability per configured arm."""

from __future__ import annotations

import socket
import threading
import time

from conftest import AcceptingListener

from apollo_mavis_v2_runtime.config import HardwareProbeConfig
from apollo_mavis_v2_runtime.devices.hardware_probe import (
    HardwareProbe,
    default_probe_fn,
    local_tcp_probe,
)


def _wait(pred, timeout_s: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


def _listening_socket() -> tuple[socket.socket, int]:
    """Bare listener (nobody accepts): fine for tests that open <= a handful of
    connections; the polling test below uses ``AcceptingListener`` instead."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    return srv, srv.getsockname()[1]


def _closed_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_local_tcp_probe_open_refused_and_writes_nothing():
    srv, port = _listening_socket()
    received = []

    def accept():
        conn, _ = srv.accept()
        conn.settimeout(1.0)
        try:
            received.append(conn.recv(64))  # b"" on a clean close, never data
        except OSError:
            received.append(b"")
        conn.close()

    t = threading.Thread(target=accept, daemon=True)
    t.start()
    try:
        assert local_tcp_probe("127.0.0.1", port, 1.0) == "open"
        t.join(timeout=2.0)
        assert received == [b""]  # connect-and-close: no bytes on the wire
    finally:
        srv.close()
    assert local_tcp_probe("127.0.0.1", _closed_port(), 1.0) == "refused"


def test_local_tcp_probe_timeout_and_no_route_are_unreachable(monkeypatch):
    """Networks differ in how they answer TEST-NET addresses (some routers RST),
    so the timeout / no-route paths are driven through the socket seam."""
    import socket as sock_mod

    def timeout(*a, **k):
        raise TimeoutError("timed out")

    monkeypatch.setattr(sock_mod, "create_connection", timeout)
    assert local_tcp_probe("192.0.2.1", 502, 0.2) == "unreachable"

    def no_route(*a, **k):
        raise OSError(101, "Network is unreachable")

    monkeypatch.setattr(sock_mod, "create_connection", no_route)
    assert local_tcp_probe("192.0.2.1", 502, 0.2) == "unreachable"

    def refused(*a, **k):
        raise ConnectionRefusedError(111, "Connection refused")

    monkeypatch.setattr(sock_mod, "create_connection", refused)
    assert local_tcp_probe("192.0.2.1", 502, 0.2) == "refused"


def test_default_probe_fn_prefers_hardware_tcp_probe_when_importable():
    fn = default_probe_fn()
    try:
        from apollo_mavis_v2_hardware.netsetup.probe import tcp_probe
    except Exception:  # noqa: BLE001 - [hardware] extra absent
        assert fn is local_tcp_probe
    else:
        assert fn is tcp_probe
    srv, port = _listening_socket()
    try:
        assert fn("127.0.0.1", port, 1.0) == "open"
    finally:
        srv.close()


def test_probe_once_snapshot_and_hardware_ready():
    srv, port = _listening_socket()
    try:
        cfg = HardwareProbeConfig(period_s=0.05, timeout_s=0.5, port=port)
        probe = HardwareProbe({"grip": "127.0.0.1", "view": "127.0.0.1"}, cfg)
        assert probe.snapshot() == {"grip": "unknown", "view": "unknown"}
        assert probe.reachable("grip") == "unknown" and probe.hardware_ready is False
        assert probe.probe_once() == {"grip": "open", "view": "open"}
        assert probe.hardware_ready is True and probe.updated_at is not None
        assert probe.reachable("nope") == "unknown"
    finally:
        srv.close()
    # Same arms against a now-closed port: refused, not ready.
    assert probe.probe_once() == {"grip": "refused", "view": "refused"}
    assert probe.hardware_ready is False


def test_thread_polls_until_stopped_and_pauses_during_hardware_session():
    # Unbounded number of rounds -> the listener must drain its accept queue.
    listener = AcceptingListener()
    paused = {"on": False}
    calls = {"n": 0}

    def counting_probe(ip: str, p: int, timeout: float) -> str:
        calls["n"] += 1
        return local_tcp_probe(ip, p, timeout)

    try:
        cfg = HardwareProbeConfig(period_s=0.05, timeout_s=0.5, port=listener.port)
        probe = HardwareProbe(
            {"grip": "127.0.0.1"}, cfg, paused=lambda: paused["on"], probe_fn=counting_probe
        )
        probe.start()
        assert _wait(lambda: probe.rounds >= 3)
        assert probe.snapshot() == {"grip": "open"} and probe.hardware_ready
        paused["on"] = True
        time.sleep(0.15)
        rounds = probe.rounds
        time.sleep(0.2)
        assert probe.rounds == rounds  # frozen while paused
        assert probe.snapshot() == {"grip": "open"}  # last snapshot kept
        paused["on"] = False
        assert _wait(lambda: probe.rounds > rounds)
        probe.stop()
        assert probe._thread is None
        n = probe.rounds
        time.sleep(0.15)
        assert probe.rounds == n  # no rounds after stop
        assert calls["n"] == n  # one connect per round (single arm) ...
        assert _wait(lambda: listener.accepted == n)  # ... each drained by the acceptor
    finally:
        listener.close()


def test_probe_without_arms_or_disabled_is_a_no_op():
    probe = HardwareProbe({}, HardwareProbeConfig())
    probe.start()
    assert probe._thread is None and probe.snapshot() == {} and probe.hardware_ready is False
    assert probe.probe_once() == {}
    off = HardwareProbe({"grip": "127.0.0.1"}, HardwareProbeConfig(enabled=False))
    off.start()
    assert off._thread is None and off.reachable("grip") == "unknown"
    none_ip = HardwareProbe({"grip": None}, HardwareProbeConfig())
    assert none_ip.arms == {} and not none_ip.enabled


def test_probe_fn_exceptions_and_garbage_are_unreachable():
    def bad(ip, port, timeout):
        if ip == "boom":
            raise RuntimeError("socket exploded")
        return "weird"

    probe = HardwareProbe({"a": "boom", "b": "x"}, HardwareProbeConfig(), probe_fn=bad)
    assert probe.probe_once() == {"a": "unreachable", "b": "unreachable"}
