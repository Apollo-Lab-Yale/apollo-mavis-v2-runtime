"""TakeoverGate state-machine determinism (12-dagger §2/§13-1; fake clock)."""

from __future__ import annotations

import pytest
from apollo_xarm7_core.dagger import ControlMode

from apollo_xarm7_runtime.dagger.gate import TakeoverGateImpl

ARMS = ["arm0", "arm1"]


def make_gate(t_blend=0.3):
    return TakeoverGateImpl(ARMS, t_blend)


def test_toggle_transition_auto_advance_human():
    g = make_gate(0.3)
    ev = g.on_toggle("arm0", 1.0)
    assert ev is not None and ev.mode is ControlMode.TAKEOVER_TRANSITION
    assert ev.source == "keyboard" and ev.seq == 1
    assert g.mode("arm0") is ControlMode.TAKEOVER_TRANSITION
    assert g.tick(1.0 + 0.29) == []  # not yet
    evs = g.tick(1.0 + 0.30)
    assert [e.mode for e in evs] == [ControlMode.HUMAN]
    assert evs[0].source == "auto_advance" and evs[0].seq == 2
    assert g.mode("arm0") is ControlMode.HUMAN
    assert g.tick(2.0) == []  # HUMAN is stable


def test_abort_during_transition_and_handback():
    g = make_gate()
    g.on_toggle("arm0", 0.0)
    ev = g.on_toggle("arm0", 0.1)  # abort mid-TRANSITION
    assert ev.mode is ControlMode.POLICY
    assert g.tick(1.0) == []  # no pending auto-advance survives the abort
    g.on_toggle("arm0", 2.0)
    g.tick(2.301)
    assert g.mode("arm0") is ControlMode.HUMAN
    ev = g.on_toggle("arm0", 3.0)  # explicit handback
    assert ev.mode is ControlMode.POLICY and g.engaged_arm() is None


@pytest.mark.parametrize("t_blend", [0.2, 0.3, 0.5])
@pytest.mark.parametrize("fps", [25, 30])
def test_transition_frame_counts_exact(t_blend, fps):
    """TRANSITION frame count at the dataset rate is exactly ceil-ish of
    t_blend*fps (0.3 s @ 25 fps ≈ 7-8 frames)."""
    g = make_gate(t_blend)
    dt = 1.0 / fps
    g.on_toggle("arm0", 0.0)
    frames = 0
    t = 0.0
    for _ in range(fps * 2):
        t += dt
        g.tick(t)
        if g.mode("arm0") is ControlMode.TAKEOVER_TRANSITION:
            frames += 1
        else:
            break
    expected = int(t_blend / dt)  # ticks strictly before the advance instant
    assert frames in (expected - 1, expected, expected + 1)
    if t_blend == 0.3 and fps == 25:
        assert 7 <= frames <= 8


def test_gate_events_exactly_one_per_switch_seq_monotonic():
    g = make_gate()
    g.on_toggle("arm0", 0.0)
    for i in range(50):
        g.tick(0.01 * (i + 1))  # ticks past T_blend emit exactly ONE advance
    g.on_toggle("arm0", 1.0)
    seqs = [e.seq for e in g.events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    assert [e.mode for e in g.events] == [
        ControlMode.TAKEOVER_TRANSITION, ControlMode.HUMAN, ControlMode.POLICY,
    ]


def test_single_engaged_arm_other_toggles_rejected():
    g = make_gate()
    assert g.on_toggle("arm0", 0.0) is not None
    assert g.on_toggle("arm1", 0.1) is None  # caller Nacks "takeover active"
    g.tick(0.5)  # arm0 -> HUMAN
    assert g.on_toggle("arm1", 0.6) is None
    assert g.frozen_arms() == ["arm1"]
    assert g.mode("arm1") is ControlMode.POLICY  # frozen frames stay policy
    assert g.engaged_arm() == "arm0"


def test_reset_returns_all_to_policy_with_episode_reset_events():
    g = make_gate()
    g.on_toggle("arm0", 0.0)
    g.tick(0.5)
    n = len(g.events)
    g.reset()
    assert g.mode("arm0") is ControlMode.POLICY and g.engaged_arm() is None
    new = g.events[n:]
    assert len(new) == 1 and new[0].source == "episode_reset"
    g.reset()  # idempotent: no extra events
    assert len(g.events) == n + 1


def test_t_blend_validation():
    with pytest.raises(ValueError):
        TakeoverGateImpl(ARMS, 0.1)
    with pytest.raises(ValueError):
        TakeoverGateImpl(ARMS, 0.6)


def test_unknown_arm_toggle_rejected():
    g = make_gate()
    assert g.on_toggle("nope", 0.0) is None
