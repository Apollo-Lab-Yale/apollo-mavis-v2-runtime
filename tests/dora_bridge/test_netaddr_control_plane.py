"""bind_host resolution / vetting (14-dora §9) and control-plane helpers (§2.2)."""

from __future__ import annotations

import pytest

from apollo_mavis_v2_runtime.config import DoraConfig
from apollo_mavis_v2_runtime.dora_bridge import netaddr
from apollo_mavis_v2_runtime.dora_bridge.control_plane import (
    check_versions,
    parse_registered_machines,
)
from dora_bridge.harness import dora_available

ARM_IPS = ("192.168.1.201", "192.168.2.219")


def test_bind_host_literals_and_interface_names():
    assert netaddr.resolve_bind_host("127.0.0.1") == "127.0.0.1"
    assert netaddr.resolve_bind_host("localhost") == "127.0.0.1"
    assert netaddr.resolve_bind_host("lo") == "127.0.0.1"  # interface name -> its IPv4
    assert netaddr.resolve_bind_host(" 10.1.2.3 ") == "10.1.2.3"
    with pytest.raises(netaddr.BindHostError, match="neither an IPv4"):
        netaddr.resolve_bind_host("no_such_iface_xyz")
    assert netaddr.is_loopback("127.0.0.1") and not netaddr.is_loopback("192.168.0.88")


def test_bind_ip_vetting_refuses_wildcard_and_control_box_subnets():
    with pytest.raises(netaddr.BindHostError, match="0.0.0.0"):
        netaddr.vet_bind_ip("0.0.0.0", ARM_IPS)
    for bad in ("192.168.1.11", "192.168.2.12", "192.168.1.201"):
        with pytest.raises(netaddr.BindHostError, match="control-box subnet"):
            netaddr.vet_bind_ip(bad, ARM_IPS)
    netaddr.vet_bind_ip("192.168.0.88", ARM_IPS)  # the lab Wi-Fi: fine
    netaddr.vet_bind_ip("127.0.0.1", ARM_IPS)
    assert [str(n) for n in netaddr.control_box_subnets(ARM_IPS)] == [
        "192.168.1.0/24",
        "192.168.2.0/24",
    ]
    assert netaddr.resolve_and_vet("lo", ARM_IPS) == "127.0.0.1"


def test_dora_config_defaults_and_guards():
    cfg = DoraConfig()
    assert cfg.enabled is False and cfg.bind_host == "127.0.0.1" and cfg.machine_id == "lab"
    assert (cfg.coordinator_port, cfg.daemon_port, cfg.zenoh_port) == (6113, 53391, 7447)
    assert cfg.auth_effective is False
    assert DoraConfig(bind_host="wlp38s0").auth_effective is True
    assert DoraConfig(bind_host="192.168.0.88", auth=False).auth_effective is False
    assert (
        cfg.publish.state_hz == 50 and cfg.publish.idle_state_hz == 10 and cfg.publish.obs_hz == 30
    )
    assert cfg.policy.spec_stale_s == 3.0 and cfg.policy.max_obs_age_s == 0.5
    with pytest.raises(ValueError, match="6013"):
        DoraConfig(coordinator_port=6013)
    with pytest.raises(ValueError, match="differ"):
        DoraConfig(coordinator_port=7000, daemon_port=7000)
    with pytest.raises(ValueError, match="unique"):
        DoraConfig(machines=[{"id": "a"}, {"id": "a"}])
    with pytest.raises(ValueError, match="local machine_id"):
        DoraConfig(machines=[{"id": "lab"}])
    with pytest.raises(ValueError):
        DoraConfig(machines=[{"id": "bad-id"}])  # ids are [a-zA-Z0-9_]


def test_parse_registered_machines_from_dora_doctor_output():
    text = """  PASS  Coordinator: reachable at 127.0.0.1:56109
  PASS  Daemon: connected
  PASS  Connected machines: 2
    lab-01a07edd-8bc1-77c0-8f16-9af655bdc025 (heartbeat: 4994ms ago)
    remote_gpu-01a07edd-c060-7623-8621-245f345d9233 (heartbeat: 1483ms ago)
  PASS  Dataflows: 0 running, 0 failed, 0 total
"""
    assert parse_registered_machines(text) == {"lab", "remote_gpu"}
    assert parse_registered_machines("PASS Connected machines: 0\n") == set()


def test_check_versions_agrees_when_the_extra_is_installed():
    if not dora_available():
        detail = check_versions()
        assert detail is not None and "dora" in detail
    else:
        assert check_versions() is None
