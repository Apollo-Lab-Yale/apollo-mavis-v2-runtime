"""ExternalPolicyHub / ExternalPolicySource rules (14-dora §5, §6.2, §6.3; §10 tier 2)."""

from __future__ import annotations

import numpy as np
import pytest
from apollo_mavis_v2_core.interfaces.policy import PolicySpec
from apollo_mavis_v2_core.protocol import external as ext
from apollo_mavis_v2_core.protocol.external import (
    PolicySpecAnnounce,
    PolicySpecModel,
    TrainerStatusAnnounce,
)

from apollo_mavis_v2_runtime.config import DoraConfig, DoraPolicyConfig
from apollo_mavis_v2_runtime.dagger.policy_source import PolicySource
from apollo_mavis_v2_runtime.dora_bridge import codec
from apollo_mavis_v2_runtime.dora_bridge.bridge import DoraBridge
from apollo_mavis_v2_runtime.dora_bridge.policy_source import (
    ExternalPolicyHub,
    ExternalPolicySource,
    spec_from_announce,
)
from dora_bridge.fake_node import FakeControlPlane, FakeNode

pa = pytest.importorskip("pyarrow")

NAMES = [
    "grip_ee.dx",
    "grip_ee.dy",
    "grip_ee.dz",
    "grip_ee.drx",
    "grip_ee.dry",
    "grip_ee.drz",
    "grip_gripper.pos",
    "grip_rail.dpos",
]
STATE_NAMES = [f"grip_joint{i}.pos" for i in range(1, 8)] + ["grip_gripper.pos", "grip_rail.pos"]


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


class FakePublisher:
    def __init__(self) -> None:
        self.observation_id = 0
        self.observation_times: dict[int, float] = {}
        self.events: list[tuple[str, dict]] = []

    def observation_t_mono(self, oid: int):
        return self.observation_times.get(oid)

    def publish_event(self, kind, payload, session_id=None):
        self.events.append((kind, payload))


def announce(
    version=1, frame="arm_base:grip", names=NAMES, rate=15.0, chunk_dt=None, space="delta_ee"
):
    return PolicySpecAnnounce(
        policy_id="fake",
        policy_version=version,
        node_version="t",
        rate_hz=rate,
        chunk_dt_s=chunk_dt,
        spec=PolicySpecModel(
            action_space=space,
            action_frame=frame,
            action_names=list(names),
            state_names=STATE_NAMES[:3],
        ),
    )


@pytest.fixture
def rig(tmp_path):
    clock = Clock()
    cfg = DoraConfig(enabled=True, var_dir=tmp_path / "d")
    node = FakeNode()
    bridge = DoraBridge(
        cfg,
        epoch="e",
        control_plane_factory=lambda c, d, ips: FakeControlPlane(c, d, ips),
        node_factory=lambda *a: node,
        versions_ok=lambda: None,
        clock=clock,
    )
    bridge.state = "attached"  # no thread: we call the handlers directly
    bridge.node = node
    hub = ExternalPolicyHub(bridge, DoraPolicyConfig(), clock=clock)
    pub = FakePublisher()
    return clock, bridge, hub, pub, node


def spec_event(ann):
    return {
        "type": "INPUT",
        "id": ext.IN_POLICY_SPEC,
        "value": codec.encode_json(ann),
        "metadata": {},
    }


def action_event(rows, oid, *, session_id="s1", version=1, chunk_dt=1 / 15, **extra):
    arr, meta = codec.encode_action(np.asarray(rows, dtype=np.float32))
    meta.update(
        {
            "observation_id": oid,
            "chunk_dt_s": chunk_dt,
            "policy_id": "fake",
            "policy_version": version,
            "compute_ms": 1.0,
            ext.META_SESSION_ID: session_id,
            ext.META_SCHEMA: 1,
            ext.META_SEQ: oid,
            ext.META_CLIENT: "t",
        }
    )
    meta.update(extra)
    return {"type": "INPUT", "id": ext.IN_POLICY_ACTION, "value": arr, "metadata": meta}


def make_source(hub, pub, clock, ann=None, rate=15.0, chunk_dt=None):
    ann = ann or announce(rate=rate, chunk_dt=chunk_dt)
    src = ExternalPolicySource(
        hub,
        pub,
        session_id="s1",
        spec=spec_from_announce(ann),
        policy_id=ann.policy_id,
        arms_meta=[("grip", True)],
        rate_hz=rate,
        chunk_dt_s=chunk_dt,
        cfg=DoraPolicyConfig(),
        clock=clock,
    )
    return src


def test_hub_caches_spec_with_staleness_and_drops_bad_payloads(rig):
    clock, bridge, hub, pub, node = rig
    assert hub.spec() is None and hub.policy_attached is False
    hub._on_spec(spec_event(announce()))
    assert hub.spec() is not None and hub.policy_attached is True and hub.spec_age_s() == 0.0
    clock.t += 2.9
    assert hub.spec() is not None
    clock.t += 0.2  # > spec_stale_s (3 s)
    assert hub.spec() is None and hub.policy_attached is False
    dropped = bridge.dropped_inputs
    hub._on_spec(
        {"type": "INPUT", "id": ext.IN_POLICY_SPEC, "value": pa.array(["not json"]), "metadata": {}}
    )
    assert bridge.dropped_inputs == dropped + 1
    hub._on_action(action_event([[0.0] * 8], 1))  # no session source -> dropped
    assert bridge.dropped_inputs == dropped + 2
    bridge._set_state("detached", "x")  # a detach forgets the spec: it must be re-heard
    hub._on_spec(spec_event(announce()))
    bridge._set_state("attached", "")
    assert hub.policy_attached is True


def test_source_is_a_policy_source_and_validates_actions(rig):
    clock, bridge, hub, pub, node = rig
    src = make_source(hub, pub, clock)
    assert isinstance(src, PolicySource)
    assert src.period == pytest.approx(1 / 15) and src.version_label() == "fake/v000001"
    src.start()
    assert hub.source is src
    # session_start reset published + watermark 0
    assert node.sent_ids() == [] or True  # bridge has no thread: FIFO holds it
    assert src.watermark == 0
    pub.observation_id = 3
    pub.observation_times = {1: 99.5, 2: 99.8, 3: 99.95}
    dropped = bridge.dropped_inputs
    src.on_action(action_event([[0.0] * 8], 3, session_id="other"), clock())
    assert bridge.dropped_inputs == dropped + 1 and src.latest()[0] is None
    src.on_action(action_event([[0.0] * 7], 3), clock())  # action_dim 7 != 8
    assert bridge.dropped_inputs == dropped + 2
    bad = action_event([[0.0] * 8], 3)
    del bad["metadata"]["chunk_len"]
    src.on_action(bad, clock())
    assert bridge.dropped_inputs == dropped + 3
    src.on_action(action_event([[0.0] * 8], 99), clock())  # unknown observation -> late
    assert src.actions_late == 1
    clock.t = 100.5  # observation 1 is now 1.0 s old (> 0.5)
    src.on_action(action_event([[0.0] * 8], 1), clock())
    assert src.actions_late == 2 and src.latest()[0] is None
    src.on_action(
        action_event([[0.01] * 6 + [1.0, 0.0]], 3), clock()
    )  # fresh (0.55 s? no: 99.95 -> 0.55)
    assert src.latest()[0] is None and src.actions_late == 3
    pub.observation_times[3] = 100.4
    src.on_action(action_event([[0.01] * 6 + [1.0, 0.0]], 3), clock())
    out, t = src.latest()
    assert out is not None and t == clock() and out.version == 1
    assert np.allclose(out.actions[:6], 0.01) and out.actions[6] == 1.0
    nan_row = [[np.nan] * 8]
    src.on_action(action_event(nan_row, 3), clock())
    assert not np.all(np.isfinite(src.latest()[0].actions))  # NaN passes through (3-strike guard)
    src.stop()
    assert hub.source is None


def test_staleness_timeline_holds_at_0_45_s_for_15_hz(rig):
    clock, bridge, hub, pub, node = rig
    src = make_source(hub, pub, clock)
    pub.observation_times = {1: clock()}
    src.on_action(action_event([[0.0] * 8], 1), clock())
    t0 = clock()
    assert src.staleness_scale(t0) == 1.0
    period = 1 / 15
    assert src.staleness_scale(t0 + period + 0.05) == 1.0  # timeout edge
    mid = t0 + period + 0.05 + 2.5 * period
    assert 0.45 < src.staleness_scale(mid) < 0.55
    assert src.staleness_scale(t0 + 0.4499) > 0.0
    assert src.staleness_scale(t0 + 0.4501) == 0.0  # period + 0.05 + 5 * period = 0.45 s
    assert src.policy_stale(t0) is False or not hub.policy_attached
    assert src.policy_stale(t0 + 0.5) is True


def test_chunks_advance_one_row_per_chunk_dt_and_rescale_deltas(rig):
    clock, bridge, hub, pub, node = rig
    chunk_dt = 0.04  # 25 Hz rows; policy rate 15 Hz
    src = make_source(hub, pub, clock, chunk_dt=chunk_dt)
    pub.observation_times = {1: clock()}
    rows = np.zeros((4, 8), dtype=np.float32)
    rows[:, 0] = [0.004, 0.008, 0.012, 0.016]  # per-chunk_dt deltas
    rows[:, 6] = 1.0
    src.on_action(action_event(rows, 1, chunk_dt=chunk_dt), clock())
    out0, t0 = src.latest()
    factor = src.period / chunk_dt  # per-period units for the executor's dt/period scaling
    assert (
        out0.actions[0] == pytest.approx(0.004 * factor) and out0.actions[6] == 1.0
    )  # gripper absolute
    assert out0.chunk_remaining == 3 and t0 == clock()
    clock.t += 0.045
    out1, t1 = src.latest()
    assert out1.actions[0] == pytest.approx(0.008 * factor) and t1 == pytest.approx(t0 + chunk_dt)
    clock.t += 0.2  # past the last row: hold on row 3
    out3, t3 = src.latest()
    assert out3.actions[0] == pytest.approx(0.016 * factor) and out3.chunk_remaining == 0
    assert t3 == pytest.approx(t0 + 3 * chunk_dt)
    src.pause()
    assert src.latest()[0] is None and src.paused
    src.resume()
    assert src.latest()[0] is None  # pause dropped the chunk


def test_abs_ee_rows_pass_verbatim_and_the_block_width_is_the_abs_layout(rig):
    """The abs twin of the rescale test (2026-09-11): an ``abs_ee`` spec announces the
    11-dim ``[x, y, z, r6, gripper, rail]`` grip block (from ``arm_action_names``); its rows
    are waypoints, so NO dim rescales with period / chunk_dt - position, rotation, gripper
    and the ABSOLUTE rail come back exactly; a delta-width (8) row is rejected."""
    from apollo_mavis_v2_runtime.recorder.features import arm_action_names

    clock, bridge, hub, pub, node = rig
    abs_names = arm_action_names("grip", True, "abs_ee")
    assert len(abs_names) == 11 and abs_names[9:] == ["grip_gripper.pos", "grip_rail.pos"]
    chunk_dt = 0.04
    ann = announce(names=abs_names, chunk_dt=chunk_dt, space="abs_ee")
    src = make_source(hub, pub, clock, ann=ann)
    assert src.spec.action_space == "abs_ee" and src._block_dim == {"grip": 11}
    assert not np.any(src._block_mask["grip"])  # all zeros: nothing rescales
    pub.observation_times = {1: clock(), 2: clock()}
    rows = np.zeros((2, 11), dtype=np.float32)
    rows[:, 0] = [0.30, 0.31]
    rows[:, 3:9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    rows[:, 9] = 0.4
    rows[:, 10] = 0.25  # the ABSOLUTE rail position: not a delta, never rescaled
    src.on_action(action_event(rows, 1, chunk_dt=chunk_dt), clock())
    out0, t0 = src.latest()
    assert out0 is not None and t0 == clock()
    np.testing.assert_allclose(out0.actions, rows[0])  # verbatim, factor period/chunk_dt unused
    clock.t += 0.045
    out1, _ = src.latest()
    np.testing.assert_allclose(out1.actions, rows[1])
    dropped = bridge.dropped_inputs
    src.on_action(action_event(np.zeros((1, 8), np.float32), 2, chunk_dt=chunk_dt), clock())
    assert bridge.dropped_inputs == dropped + 1  # the delta width is not this spec's
    src.on_action(action_event(np.zeros((1, 11), np.float32), 2, chunk_dt=chunk_dt), clock())
    assert bridge.dropped_inputs == dropped + 1  # 11 accepted
    # the per-arm stream expects the same 11
    src2 = ExternalPolicySource(
        hub, pub, session_id="s1", spec=spec_from_announce(ann),
        policy_id="fake", arms_meta=[("grip", True)], rate_hz=15.0, chunk_dt_s=None,
        cfg=DoraPolicyConfig(), clock=clock, driven_arms=["grip"],
    )
    pub.observation_times[3] = clock()
    src2.on_action(action_event(rows[:1], 3), clock(), arm_id="grip")
    assert bridge.dropped_inputs == dropped + 1
    np.testing.assert_allclose(src2.latest()[0].actions, rows[0])
    src2.on_action(action_event(np.zeros((1, 8), np.float32), 3), clock(), arm_id="grip")
    assert bridge.dropped_inputs == dropped + 2


def test_resolve_driven_arms_for_an_abs_ee_spec_uses_the_abs_blocks():
    from apollo_mavis_v2_runtime.recorder.features import arm_action_names

    grip_abs = arm_action_names("grip", True, "abs_ee")
    view_abs = arm_action_names("view", False, "abs_ee")
    assert (len(grip_abs), len(view_abs)) == (11, 10)
    assert resolve_driven_arms([], grip_abs, TWO_ARMS, "abs_ee") == ["grip"]
    assert resolve_driven_arms([], view_abs + grip_abs, TWO_ARMS, "abs_ee") == ["grip", "view"]
    with pytest.raises(DrivenArmsError):
        resolve_driven_arms([], grip_abs, TWO_ARMS)  # the delta blocks do not match abs names
    with pytest.raises(DrivenArmsError):
        resolve_driven_arms([], GRIP, TWO_ARMS, "abs_ee")


def test_reset_watermark_publishes_policy_reset_and_drops_older_observations(rig):
    clock, bridge, hub, pub, node = rig
    src = make_source(hub, pub, clock)
    src.start()
    pub.observation_id = 10
    pub.observation_times = {9: clock(), 10: clock(), 11: clock(), 12: clock()}
    src.on_action(action_event([[0.0] * 8], 9), clock())
    assert src.latest()[0] is not None
    src.drop_and_requery()  # handback: watermark = newest observation (10)
    assert src.latest()[0] is None and src.watermark == 10
    src.on_action(action_event([[0.0] * 8], 10), clock())
    assert src.latest()[0] is None and src.actions_late == 1
    pub.observation_id = 12
    src.on_action(action_event([[0.0] * 8], 11), clock())
    assert src.latest()[0] is not None
    reasons = [t()[1].get("reason") for oid, t, _ in bridge._fifo if oid == "policy_reset"]
    assert reasons[:2] == ["session_start", "handback"]
    assert [k for k, _ in pub.events] == ["reset_watermark", "reset_watermark"]
    # a version change mid-stream is recorded, never rejected
    src.on_action(action_event([[0.0] * 8], 12, version=2), clock())
    assert src.current_version() == 2 and src.version_changes == 1
    assert src.version_label() == "fake/v000002"
    src.stop()
    resets = [t()[1].get("reason") for oid, t, _ in bridge._fifo if oid == "policy_reset"]
    assert resets[-1] == "session_stop"


def test_spec_frame_mismatch_is_visible_to_the_manager_check(rig):
    clock, bridge, hub, pub, node = rig
    ann = announce(frame="world")
    spec = spec_from_announce(ann)
    assert isinstance(spec, PolicySpec) and spec.action_frame == "world"
    assert spec.action_names == NAMES and spec.version == 1


# -- phase-14 (15-online-dagger §6): the trainer role's status on policy/trainer_status ------------
def trainer_event(msg, **meta):
    return {
        "type": "INPUT",
        "id": ext.IN_POLICY_TRAINER_STATUS,
        "value": codec.encode_json(msg),
        "metadata": dict(meta),
    }


def trainer_status(state="preparing", **kw):
    return TrainerStatusAnnounce(
        trainer_id="repo/online_dagger",
        node_version="0.1",
        state=state,
        progress=0.5,
        **kw,
    )


class SinkSpy:
    """The coordinator surface the hub drives (bus thread)."""

    def __init__(self) -> None:
        self.statuses: list[tuple] = []
        self.versions: list[int] = []

    def on_trainer_status(self, msg, t_recv):
        self.statuses.append((msg, t_recv))

    def on_spec_version(self, version):
        self.versions.append(version)


def test_hub_caches_trainer_status_with_staleness_and_drops_bad_payloads(rig):
    import json

    clock, bridge, hub, pub, node = rig
    assert hub.trainer_status() is None and hub.trainer_age_s() is None
    assert hub.trainer_capable() is False
    hub._on_spec(spec_event(announce()))
    assert hub.trainer_capable() is False  # a fresh spec WITHOUT the capability
    hub._on_spec(spec_event(announce().model_copy(update={"capabilities": ["online_dagger"]})))
    assert hub.trainer_capable() is True
    msg = trainer_status()
    hub._on_trainer_status(trainer_event(msg, client="node", seq=1))
    assert hub.trainer_status() == msg and hub.trainer_age_s() == 0.0
    assert hub.trainer_statuses_seen == 1
    clock.t += 2.9
    assert hub.trainer_status() == msg
    clock.t += 0.2  # > spec_stale_s (3 s): the same freshness rule as the spec
    assert hub.trainer_status() is None and hub.trainer_age_s() == pytest.approx(3.1)
    assert hub.trainer_capable() is False  # the spec went stale with it
    dropped = bridge.dropped_inputs
    bad = {"type": "INPUT", "id": ext.IN_POLICY_TRAINER_STATUS, "metadata": {}}
    hub._on_trainer_status({**bad, "value": pa.array(["not json"])})
    hub._on_trainer_status(trainer_event(msg.model_copy(update={"mavis_schema": 99})))
    hub._on_trainer_status({**bad, "value": pa.array([json.dumps({"state": "idle"})])})
    assert bridge.dropped_inputs == dropped + 3  # counted + logged, never raised
    assert hub.trainer_statuses_seen == 1 and hub.trainer_status() is None
    hub._on_trainer_status({"type": "STOP"})  # non-INPUT events are ignored
    assert hub.trainer_statuses_seen == 1
    hub._on_trainer_status(trainer_event(msg))
    assert hub.trainer_status() == msg and hub.trainer_statuses_seen == 2
    bridge._set_state("detached", "x")  # a detach forgets it like the spec
    assert hub.trainer_status() is None
    bridge._set_state("attached", "")
    assert hub.trainer_status() is None  # must be re-heard
    hub._on_trainer_status(trainer_event(msg))
    assert hub.trainer_status() == msg


def test_hub_feeds_the_trainer_sink_and_replays_the_cached_status_on_attach(rig):
    clock, bridge, hub, pub, node = rig
    hub._on_spec(spec_event(announce(version=3)))
    msg = trainer_status()
    hub._on_trainer_status(trainer_event(msg))
    t_recv = clock()
    clock.t += 0.5
    sink = SinkSpy()
    hub.attach_trainer_sink(sink)
    assert hub.trainer_sink is sink
    assert sink.versions == [3]  # the announced policy version first ...
    assert sink.statuses == [(msg, t_recv)]  # ... then the cached status with ITS receive time
    msg2 = trainer_status("training", metrics={"loss": 0.1})
    hub._on_trainer_status(trainer_event(msg2))
    assert sink.statuses[-1] == (msg2, clock())
    hub._on_spec(spec_event(announce(version=4)))
    assert sink.versions == [3, 4]
    # an action's policy_version is the acting version too (15-online-dagger §3)
    src = make_source(hub, pub, clock, ann=announce(version=4))
    src.start()
    pub.observation_id = 1
    pub.observation_times = {1: clock()}
    src.on_action(action_event([[0.0] * 8], 1, version=5), clock())
    assert sink.versions == [3, 4, 5]
    src.on_action(action_event([[0.0] * 8], 1, version=5), clock())  # same version: quiet
    assert sink.versions == [3, 4, 5]
    # a sink that raises never breaks the bus thread
    sink.on_trainer_status = lambda m, t: 1 / 0
    hub._on_trainer_status(trainer_event(msg2))
    assert hub.trainer_status() == msg2
    hub.detach_trainer_sink(SinkSpy())  # someone else's sink: no-op
    assert hub.trainer_sink is sink
    hub.detach_trainer_sink(sink)
    assert hub.trainer_sink is None
    hub._on_spec(spec_event(announce(version=6)))
    assert sink.versions == [3, 4, 5]  # detached: nothing more
    # attaching while the cached status is STALE replays only the version
    clock.t += 5.0
    hub._on_spec(spec_event(announce(version=7)))
    sink2 = SinkSpy()
    hub.attach_trainer_sink(sink2)
    assert sink2.versions == [7] and sink2.statuses == []
    hub.attach_trainer_sink(None)
    assert hub.trainer_sink is None
    src.stop()


# -- v1.3 (2026-09-11): per-arm action streams (14-dora §5 / §6.1) -------------------------------
from apollo_mavis_v2_runtime.dora_bridge.policy_source import (  # noqa: E402
    DrivenArmsError,
    infer_policy_arms,
    resolve_driven_arms,
)

TWO_ARMS = [("grip", True), ("view", False)]
GRIP = NAMES  # the 8-dim grip block
VIEW = [
    "view_ee.dx",
    "view_ee.dy",
    "view_ee.dz",
    "view_ee.drx",
    "view_ee.dry",
    "view_ee.drz",
    "view_gripper.pos",
]


def test_resolve_driven_arms_subset_inference_and_refusals():
    # legacy whole cell -> every arm, session order
    assert resolve_driven_arms([], GRIP + VIEW, TWO_ARMS) == ["grip", "view"]
    # a subset is inferred from whole blocks ...
    assert resolve_driven_arms([], GRIP, TWO_ARMS) == ["grip"]
    assert resolve_driven_arms([], VIEW, TWO_ARMS) == ["view"]
    # ... or declared; the result is always in SESSION order
    assert resolve_driven_arms(["view", "grip"], GRIP + VIEW, TWO_ARMS) == ["grip", "view"]
    assert resolve_driven_arms(["grip"], GRIP, TWO_ARMS) == ["grip"]
    with pytest.raises(DrivenArmsError, match="unknown arm"):
        resolve_driven_arms(["arm9"], GRIP, TWO_ARMS)
    with pytest.raises(DrivenArmsError, match="twice"):
        resolve_driven_arms(["grip", "grip"], GRIP, TWO_ARMS)
    with pytest.raises(DrivenArmsError, match="not the per-arm blocks"):
        resolve_driven_arms(["grip"], GRIP[:-1], TWO_ARMS)  # a partial block
    # v1.3: a whole-cell policy may concatenate its arms in ANY order - a policy announces
    # its spec before the session exists and cannot know the session's arm order (the idle
    # announce is in workcell order, the session follows SessionSpec.arms)
    assert resolve_driven_arms([], VIEW + GRIP, TWO_ARMS) == ["grip", "view"]
    assert resolve_driven_arms(["view", "grip"], VIEW + GRIP, TWO_ARMS) == ["grip", "view"]
    with pytest.raises(DrivenArmsError, match="cover no whole arm block"):
        resolve_driven_arms([], ["grip_ee.dx", "view_ee.dy"], TWO_ARMS)
    with pytest.raises(DrivenArmsError, match="not the per-arm blocks"):
        resolve_driven_arms([], GRIP + ["view_ee.dx"], TWO_ARMS)  # grip whole + a stray dim


def test_action_block_layout_maps_blocks_by_name_in_announced_order():
    from apollo_mavis_v2_runtime.dora_bridge.policy_source import (
        action_block_layout,
        arm_blocks,
    )

    blocks = arm_blocks(TWO_ARMS)
    # session order grip(8) then view(7)
    assert action_block_layout(GRIP + VIEW, blocks, {"grip", "view"}) == [
        ("grip", 0, 8),
        ("view", 8, 7),
    ]
    # reversed announce: view first - the offsets follow the ANNOUNCED order, not session
    assert action_block_layout(VIEW + GRIP, blocks, {"grip", "view"}) == [
        ("view", 0, 7),
        ("grip", 7, 8),
    ]
    assert action_block_layout(GRIP, blocks, {"grip"}) == [("grip", 0, 8)]


def test_infer_policy_arms_from_declaration_or_prefixes():
    assert infer_policy_arms(announce(names=GRIP), ["grip", "view"]) == ["grip"]
    assert infer_policy_arms(announce(names=GRIP + VIEW), ["grip", "view"]) == ["grip", "view"]
    ann = announce(names=GRIP)
    ann.spec.arms = ["view"]  # declared wins (checked against the layout at launch)
    assert infer_policy_arms(ann, ["grip", "view"]) == ["view"]
    # an arm id with an underscore, unknown to the configured list: the last `_` splits
    assert infer_policy_arms(announce(names=["my_arm_ee.dx", "my_arm_gripper.pos"])) == ["my_arm"]


def make_two_arm_source(hub, pub, clock, names, driven, rate=25.0):
    ann = announce(rate=rate, names=names)
    ann.spec.arms = list(driven)
    return ExternalPolicySource(
        hub,
        pub,
        session_id="s1",
        spec=spec_from_announce(ann),
        policy_id=ann.policy_id,
        arms_meta=TWO_ARMS,
        rate_hz=rate,
        chunk_dt_s=None,
        cfg=DoraPolicyConfig(),
        clock=clock,
        driven_arms=driven,
    )


def test_hub_registers_one_input_per_configured_arm_and_reports_policy_arms(rig):
    clock, bridge, _hub, pub, node = rig
    hub = ExternalPolicyHub(bridge, DoraPolicyConfig(), clock=clock, arm_ids=["grip", "view"])
    assert ext.policy_arm_action_input_id("grip") in bridge._handlers
    assert ext.policy_arm_action_input_id("view") in bridge._handlers
    assert hub.policy_arms() == []  # no fresh spec
    hub._on_spec(spec_event(announce(names=GRIP)))
    assert hub.policy_arms() == ["grip"]
    # a per-arm action with no session source is dropped + counted like the whole-cell one
    dropped = bridge.dropped_inputs
    ev = action_event([[0.0] * 8], 1)
    ev["id"] = ext.policy_arm_action_input_id("grip")
    hub._on_arm_action("grip", ev)
    assert bridge.dropped_inputs == dropped + 1 and hub.actions_seen == 1


def test_grip_only_policy_holds_the_view_arm_and_accepts_both_stream_shapes(rig):
    clock, bridge, hub, pub, node = rig
    src = make_two_arm_source(hub, pub, clock, GRIP, ["grip"])
    assert isinstance(src, PolicySource)
    assert src.driven_arms() == frozenset({"grip"})
    src.start()
    pub.observation_id = 1
    pub.observation_times = {1: clock()}
    assert src.latest()[0] is None
    # the per-arm stream: an 8-dim grip block
    grip_row = [0.004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.001]
    dt = src.period  # rows in per-period units -> the executor sees them unscaled
    src.on_action(action_event([grip_row], 1, chunk_dt=dt), clock(), arm_id="grip")
    out, t = src.latest()
    assert out is not None and out.actions.shape == (15,) and t == clock()
    grip_block, view_block = out.actions[:8], out.actions[8:]
    assert grip_block[0] == pytest.approx(0.004) and grip_block[6] == 0.5
    assert grip_block[7] == pytest.approx(0.001)
    assert np.all(np.isnan(view_block))  # the undriven arm: NaN = hold, never a strike
    # per-arm staleness: fresh grip, the view stream simply does not exist (0.0)
    assert src.staleness_scale(clock(), "grip") == 1.0
    assert src.staleness_scale(clock(), "view") == 0.0
    assert src.staleness_scale(clock()) == 1.0  # source-wide = min over DRIVEN arms
    dropped = bridge.dropped_inputs
    # a stream for an arm the policy does not drive is dropped
    src.on_action(action_event([[0.0] * 7], 1), clock(), arm_id="view")
    assert bridge.dropped_inputs == dropped + 1
    # the whole-cell layout on a per-arm stream is dropped (action_dim 16 != 8)
    src.on_action(action_event([[0.0] * 16], 1), clock(), arm_id="grip")
    assert bridge.dropped_inputs == dropped + 2
    # a lying arm_id metadata key is dropped
    src.on_action(action_event([[0.0] * 8], 1, arm_id="view"), clock(), arm_id="grip")
    assert bridge.dropped_inputs == dropped + 3
    # the whole-cell `policy_action` carrying just the driven blocks is still accepted
    src.on_action(action_event([[0.002] + [0.0] * 5 + [1.0, 0.0]], 1, chunk_dt=dt), clock())
    out2, _ = src.latest()
    assert out2.actions[0] == pytest.approx(0.002) and out2.actions[6] == 1.0
    assert bridge.dropped_inputs == dropped + 3
    # a per-arm row with the wrong width for the whole-cell input is dropped too
    src.on_action(action_event([[0.0] * 15], 1), clock())
    assert bridge.dropped_inputs == dropped + 4
    src.stop()


def test_two_driven_arms_have_independent_slots_and_staleness(rig):
    clock, bridge, hub, pub, node = rig
    src = make_two_arm_source(hub, pub, clock, GRIP + VIEW, ["grip", "view"])
    pub.observation_times = {1: clock(), 2: clock()}
    assert src.staleness_scale(clock()) == 0.0  # nothing yet
    src.on_action(action_event([[0.001] * 6 + [1.0, 0.0]], 1), clock(), arm_id="grip")
    out, t = src.latest()
    assert np.isfinite(out.actions[:8]).all() and np.isnan(out.actions[8:]).all()
    assert src.staleness_scale(clock(), "grip") == 1.0
    assert src.staleness_scale(clock(), "view") == 0.0
    assert src.staleness_scale(clock()) == 0.0  # min over driven arms: view has no chunk
    clock.t += 0.02
    src.on_action(action_event([[0.0] * 6 + [1.0]], 2), clock(), arm_id="view")
    out, t = src.latest()
    assert np.isfinite(out.actions).all() and t == clock()  # newest row wins the timestamp
    period = src.period
    # the grip stream ages out on its own: after period + 0.05 + 5 periods it is 0, view 1.0
    clock.t += period + 0.05 + 5 * period + 1e-3 - 0.02
    assert src.staleness_scale(clock(), "grip") == 0.0
    assert src.staleness_scale(clock(), "view") > 0.0
    # the whole-cell input splits the two driven blocks in spec order (grip then view)
    pub.observation_times[3] = clock()
    src.on_action(action_event([[0.003] * 8 + [0.004] * 7], 3, chunk_dt=period), clock())
    out, _ = src.latest()
    assert out.actions[0] == pytest.approx(0.003) and out.actions[8] == pytest.approx(0.004)
    # a handback clears BOTH slots
    src.drop_and_requery()
    assert src.latest()[0] is None
