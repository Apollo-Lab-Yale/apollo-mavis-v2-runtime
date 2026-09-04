"""mavis_v2 camera-only arm (``view``: gripper none) through the real
SessionManager: /api/workcell reports gripper presence per arm, and F/H on
the camera-only arm never reaches ``SimArm.command_gripper`` (which raises)."""

from __future__ import annotations

import logging
import time

import pytest
from apollo_mavis_v2_core.protocol import SessionSpec
from conftest import make_runtime_config

from apollo_mavis_v2_runtime.runtime import Runtime
from apollo_mavis_v2_runtime.session.types import SessionState

SPEC = SessionSpec(
    mode="teleop", kind="sim", arms=["view", "grip"],
    frames={"view": "arm_base:view", "grip": "arm_base:grip"}, sim_scene="mavis_v2",
)


@pytest.fixture()
def runtime(tmp_path):
    rt = Runtime(make_runtime_config(tmp_path, scene="mavis_v2"))
    rt.manager.start_previews()
    yield rt
    rt.stop()  # teardown() + stop_previews()


def _gripper_by_arm(manager) -> dict[str, str]:
    return {a.arm_id: a.gripper for a in manager.workcell_status().arms}


def _create_running(runtime) -> None:
    runtime.manager.create(SPEC)  # keep_current: RUNNING without motion
    for _ in range(200):
        if runtime.manager.state == SessionState.RUNNING:
            return
        time.sleep(0.02)
    raise AssertionError(f"session state {runtime.manager.state}")


def _hold(runtime, held: list[str], duration_s: float, seq: int) -> int:
    """Controller KeysMsg + 25 Hz heartbeat (deadman never trips)."""
    end = time.monotonic() + duration_s
    while time.monotonic() < end:
        seq += 1
        runtime.on_keys(seq, held)
        time.sleep(0.04)
    seq += 1
    runtime.on_keys(seq, [])
    return seq


def test_workcell_status_reports_gripper_presence(runtime):
    m = runtime.manager
    expected = {"view": "none", "grip": "xarm"}
    assert _gripper_by_arm(m) == expected  # pre-session: preview scene
    _create_running(runtime)
    assert _gripper_by_arm(m) == expected  # in-session: workcell scene
    assert not any(a.gripper_force_capable for a in m.workcell_status().arms)


def test_gripper_keys_on_camera_only_arm_send_nothing(runtime, caplog):
    _create_running(runtime)
    loop = runtime.manager.session.loop
    assert loop.active_arm == "view" and loop.gripper_arms == {"grip"}

    seq = _hold(runtime, ["KeyF"], 0.4, 0)  # close on the camera-only arm
    assert loop.tick_count > 20
    assert "view" not in loop._grip_frac
    assert loop._senders["view"]._last_grip is None  # nothing forwarded

    # The gripper arm still works: Tab, then close.
    runtime.submit_action("switch_arm", {}).result(timeout=5.0)
    assert loop.active_arm == "grip"
    frac0 = loop._grip_frac["grip"]
    _hold(runtime, ["KeyF"], 0.3, seq)
    assert loop._grip_frac["grip"] < frac0
    assert loop._senders["grip"]._last_grip is not None

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors == [], [r.getMessage() for r in errors]
