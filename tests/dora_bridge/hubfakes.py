"""Shared fakes for the external-policy hub (14-dora §5/§6; phase-14 15-online-dagger §6):
a ``DoraBridge`` over :class:`FakeNode` / :class:`FakeControlPlane` that is "attached"
without a thread (the tests call the input handlers directly), an
:class:`ExternalPolicyHub` on an injectable clock, and builders for the two JSON
heartbeats the policy node sends (``policy_spec`` / ``policy_trainer_status``)."""

from __future__ import annotations

from apollo_mavis_v2_core.protocol import external as ext
from apollo_mavis_v2_core.protocol.external import (
    PolicySpecAnnounce,
    PolicySpecModel,
    TrainerStatusAnnounce,
)

from apollo_mavis_v2_runtime.config import DoraConfig, DoraPolicyConfig
from apollo_mavis_v2_runtime.dora_bridge import codec
from apollo_mavis_v2_runtime.dora_bridge.bridge import DoraBridge
from apollo_mavis_v2_runtime.dora_bridge.policy_source import ExternalPolicyHub
from dora_bridge.fake_node import FakeControlPlane, FakeNode

NAMES = [
    "arm0_ee.dx",
    "arm0_ee.dy",
    "arm0_ee.dz",
    "arm0_ee.drx",
    "arm0_ee.dry",
    "arm0_ee.drz",
    "arm0_gripper.pos",
    "arm0_rail.dpos",
]


class Clock:
    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def make_bridge_and_hub(var_dir, clock=None):
    """``(clock, bridge, hub, node)``: an attached bridge with no bus thread. ``clock``
    defaults to a frozen :class:`Clock` (unit tests step it); pass ``time.monotonic`` when
    the hub feeds a coordinator that runs on the real clock (the runtime's arrangement)."""
    clock = clock or Clock()
    cfg = DoraConfig(enabled=True, var_dir=var_dir)
    node = FakeNode()
    bridge = DoraBridge(
        cfg,
        epoch="e",
        control_plane_factory=lambda c, d, ips: FakeControlPlane(c, d, ips),
        node_factory=lambda *a: node,
        versions_ok=lambda: None,
        clock=clock,
    )
    bridge.state = "attached"
    bridge.node = node
    hub = ExternalPolicyHub(bridge, DoraPolicyConfig(), clock=clock)
    return clock, bridge, hub, node


def announce(
    version: int = 1,
    frame: str = "arm_base:arm0",
    names=NAMES,
    rate: float = 15.0,
    capabilities: list[str] | None = None,
) -> PolicySpecAnnounce:
    return PolicySpecAnnounce(
        policy_id="fake",
        policy_version=version,
        node_version="t",
        rate_hz=rate,
        spec=PolicySpecModel(
            action_space="delta_ee",
            action_frame=frame,
            action_names=list(names),
            state_names=[f"arm0_joint{i}.pos" for i in range(1, 4)],
        ),
        capabilities=list(capabilities or []),
    )


def spec_event(ann: PolicySpecAnnounce) -> dict:
    return {
        "type": "INPUT",
        "id": ext.IN_POLICY_SPEC,
        "value": codec.encode_json(ann),
        "metadata": {},
    }


def trainer_status(
    state: str = "preparing", *, progress: float = 0.5, **kw
) -> TrainerStatusAnnounce:
    """The generic trainer status (15-online-dagger §6): state + progress + free metrics."""
    return TrainerStatusAnnounce(
        trainer_id="repo/online_dagger",
        node_version="0.1",
        state=state,  # type: ignore[arg-type]
        progress=progress,
        **kw,
    )


def trainer_event(msg: TrainerStatusAnnounce, **meta) -> dict:
    return {
        "type": "INPUT",
        "id": ext.IN_POLICY_TRAINER_STATUS,
        "value": codec.encode_json(msg),
        "metadata": dict(meta),
    }


__all__ = [
    "NAMES",
    "Clock",
    "announce",
    "make_bridge_and_hub",
    "spec_event",
    "trainer_event",
    "trainer_status",
]
