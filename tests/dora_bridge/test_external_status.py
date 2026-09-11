"""``DoraWiring.external_status`` (14-dora §13; phase-14 15-online-dagger §6/§8): the FRESH
spec's ``capabilities`` and the newest ``trainer_status`` ride ``telemetry.external`` so
the launcher can gate "Start Online DAgger" and show the trainer pill BEFORE a session
exists; both fall back ([] / None) when the spec goes stale or the node detaches."""

from __future__ import annotations

import pytest
from apollo_mavis_v2_core.protocol.external import ExternalStatus

from apollo_mavis_v2_runtime.dora_bridge.wiring import DoraWiring
from dora_bridge.hubfakes import (
    announce,
    make_bridge_and_hub,
    spec_event,
    trainer_event,
    trainer_status,
)

pytest.importorskip("pyarrow")


def wiring_over(bridge, hub) -> DoraWiring:
    """A ``DoraWiring`` reduced to what ``external_status`` reads (no Runtime)."""
    w = DoraWiring.__new__(DoraWiring)
    w.bridge = bridge
    w.policy_hub = hub
    w.publisher = None
    w.idle_reader = None
    return w


def test_external_status_carries_capabilities_and_trainer_status(tmp_path):
    clock, bridge, hub, _node = make_bridge_and_hub(tmp_path / "d")
    w = wiring_over(bridge, hub)
    st = w.external_status(clock())
    assert isinstance(st, ExternalStatus)
    assert st.policy_attached is False and st.capabilities == [] and st.trainer_status is None
    # appended LAST (additive wire change)
    assert list(ExternalStatus.model_fields)[-3:] == [
        "capabilities",
        "trainer_status",
        "policy_arms",  # v1.3 (2026-09-11): the arms the fresh spec drives
    ]
    assert st.policy_arms == []
    hub._on_spec(spec_event(announce(version=2, capabilities=["online_dagger", "x"])))
    msg = trainer_status("preparing", progress=0.5, metrics={"loss": 0.25}, detail="pool 3/6")
    hub._on_trainer_status(trainer_event(msg, client="node", seq=7))
    st = w.external_status(clock())
    assert st.policy_attached is True and st.policy_version == 2
    assert st.capabilities == ["online_dagger", "x"] and st.trainer_status == msg
    dumped = st.model_dump()
    assert dumped["trainer_status"] == {
        "mavis_schema": 1,
        "trainer_id": "repo/online_dagger",
        "node_version": "0.1",
        "state": "preparing",
        "session_id": None,
        "policy_version": 0,
        "progress": 0.5,
        "metrics": {"loss": 0.25},
        "detail": "pool 3/6",
        "uptime_s": 0.0,
    }
    again = ExternalStatus.model_validate_json(st.model_dump_json())
    assert again.trainer_status == msg and again.capabilities == ["online_dagger", "x"]
    # a node without the capability: the pill shows the trainer status, the gate stays shut
    hub._on_spec(spec_event(announce(version=2)))
    st = w.external_status(clock())
    assert st.capabilities == [] and st.trainer_status == msg
    # stale (> spec_stale_s): both fall back
    hub._on_spec(spec_event(announce(version=2, capabilities=["online_dagger"])))
    clock.t += 3.1
    st = w.external_status(clock())
    assert st.policy_attached is False and st.capabilities == [] and st.trainer_status is None
    assert st.spec_age_s == pytest.approx(3.1)
    # fresh again, then a detach forgets both at once
    hub._on_spec(spec_event(announce(version=3, capabilities=["online_dagger"])))
    hub._on_trainer_status(trainer_event(msg))
    assert w.external_status(clock()).trainer_status == msg
    bridge._set_state("detached", "daemon gone")
    st = w.external_status(clock())
    assert st.state == "detached" and st.capabilities == [] and st.trainer_status is None
    bridge._set_state("attached", "")
    assert w.external_status(clock()).trainer_status is None  # must be re-heard
    hub._on_trainer_status(trainer_event(msg))
    assert w.external_status(clock()).trainer_status == msg
