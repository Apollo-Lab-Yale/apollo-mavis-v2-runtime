"""InputWatchdog state machine — exhaustive transitions (11-safety §10.1)."""

from __future__ import annotations

from apollo_xarm7_core import HeldState

from apollo_xarm7_runtime.safety.watchdog import InputWatchdog, WatchdogState


def keys(seq: int, t: float, held: list[str] | None = None) -> HeldState:
    return HeldState(held=frozenset(held or []), seq=seq, rx_mono=t)


def test_fresh_input_full_scale():
    wd = InputWatchdog(0.2, 0.1)
    wd.on_keys(keys(1, 0.0, ["KeyW"]))
    assert wd.scale(0.1) == 1.0
    assert wd.state is WatchdogState.OK


def test_deadman_trips_after_timeout_and_ramps():
    wd = InputWatchdog(0.2, 0.1)
    wd.on_keys(keys(1, 0.0, ["KeyW"]))
    assert wd.scale(0.19) == 1.0
    s = wd.scale(0.25)  # 0.05 into the 0.1 s ramp
    assert 0.4 < s < 0.6
    assert wd.state is WatchdogState.TRIPPED
    assert wd.scale(0.31) == 0.0  # ramp complete at 0.2 + 0.1
    assert wd.state is WatchdogState.AWAIT_EMPTY


def test_resumed_heartbeat_with_held_keys_does_not_resume():
    wd = InputWatchdog(0.2, 0.1)
    wd.on_keys(keys(1, 0.0, ["KeyW"]))
    assert wd.scale(0.5) == 0.0  # long past ramp
    wd.on_keys(keys(2, 0.5, ["KeyW"]))  # heartbeat resumes, key still down
    assert wd.scale(0.51) == 0.0
    assert wd.state is WatchdogState.AWAIT_EMPTY


def test_empty_held_set_resumes():
    wd = InputWatchdog(0.2, 0.1)
    wd.on_keys(keys(1, 0.0, ["KeyW"]))
    wd.scale(0.5)  # latch
    wd.on_keys(keys(2, 0.5, []))  # all keys up
    assert wd.state is WatchdogState.OK
    wd.on_keys(keys(3, 0.55, ["KeyW"]))
    assert wd.scale(0.6) == 1.0


def test_empty_set_mid_ramp_resumes_directly():
    wd = InputWatchdog(0.2, 0.1)
    wd.on_keys(keys(1, 0.0, ["KeyW"]))
    assert 0.0 < wd.scale(0.25) < 1.0  # TRIPPED, mid-ramp
    wd.on_keys(keys(2, 0.26, []))
    assert wd.state is WatchdogState.OK
    assert wd.scale(0.27) == 1.0


def test_seq_regression_rejected():
    wd = InputWatchdog(0.2, 0.1)
    assert wd.on_keys(keys(5, 0.0, ["KeyW"]))
    assert not wd.on_keys(keys(5, 0.1, []))  # equal seq dropped
    assert not wd.on_keys(keys(4, 0.1, []))  # older seq dropped
    assert wd.state is WatchdogState.OK  # empty set never accepted


def test_disconnect_latches_and_new_controller_resumes():
    wd = InputWatchdog(0.2, 0.1)
    wd.on_keys(keys(9, 0.0, ["KeyW"]))
    wd.on_disconnect()
    assert wd.state is WatchdogState.AWAIT_EMPTY
    assert wd.scale(0.01) == 0.0
    # New controller restarts its own seq space; empty set resumes.
    assert wd.on_keys(keys(1, 0.02, []))
    assert wd.state is WatchdogState.OK


def test_recovery_forces_all_keys_up():
    wd = InputWatchdog(0.2, 0.1)
    wd.on_keys(keys(1, 0.0, ["KeyW"]))
    wd.on_recovery()
    assert wd.scale(0.01) == 0.0
    wd.on_keys(keys(2, 0.02, []))
    assert wd.state is WatchdogState.OK
