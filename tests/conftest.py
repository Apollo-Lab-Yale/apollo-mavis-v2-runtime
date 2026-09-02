"""Shared fixtures. Sets MUJOCO_GL=egl before any mujoco GL init."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")  # noqa: E402 - must precede mujoco GL init

import numpy as np
import pytest
from apollo_xarm7_core.testing import FakeArm, FakeWorkcell

from apollo_xarm7_runtime.bus import RuntimeBus
from apollo_xarm7_runtime.config import ControlConfig, RuntimeConfig, TrackerConfig, VideoConfig
from apollo_xarm7_runtime.control.loop import ControlLoop
from apollo_xarm7_runtime.safety.gate import NullGate
from apollo_xarm7_runtime.safety.supervisor import SafetySupervisor
from apollo_xarm7_runtime.safety.watchdog import InputWatchdog

SIM_WORKCELL_YAML = {
    "kind": "sim",
    "sim_scene": "single_rail",
    "arms": [{"id": "arm0", "base_in_world": {}}],
    "cameras": [],
}


def make_runtime_config(
    tmp_path, scene: str = "single_rail", *, tracker: TrackerConfig | None = None, **safety
) -> RuntimeConfig:
    wc = dict(SIM_WORKCELL_YAML, sim_scene=scene)
    if safety:
        wc["safety"] = safety
    return RuntimeConfig(
        workcells={"sim": wc},
        profiles_dir=tmp_path / "profiles",
        datasets_root=tmp_path / "datasets",
        checkpoints_root=tmp_path / "ckpts",
        video=VideoConfig(preview_fps=15, session_fps=30),
        tracker=tracker or TrackerConfig(),
    )


@pytest.fixture
def fake_loop(tmp_path):
    """ControlLoop over FakeWorkcell with a NullGate; deterministic ticks."""
    from apollo_xarm7_core import ProfileStore

    cell = FakeWorkcell(
        {"arm0": FakeArm("arm0", has_rail=True), "arm1": FakeArm("arm1")}
    )
    cell.start()
    bus = RuntimeBus()
    supervisor = SafetySupervisor(NullGate(), InputWatchdog())
    loop = ControlLoop(
        cell,
        ControlConfig(),
        bus,
        supervisor,
        ["arm0", "arm1"],
        profile_store=ProfileStore(tmp_path / "profiles"),
        workcell_kind="sim",
    )
    return cell, bus, loop


def run_ticks(loop: ControlLoop, cell: FakeWorkcell, n: int, t0: float = 0.0) -> float:
    """Advance loop + fake workcell n deterministic ticks; returns end time."""
    t = t0
    for _ in range(n):
        t += loop.dt
        loop.run_tick(t)
        cell.step(loop.dt)
    return t


def q_of(cell: FakeWorkcell, arm_id: str) -> np.ndarray:
    return cell.arms[arm_id].get_state().q


class LiveServer:
    """Real uvicorn (own thread, random port, deflate OFF) for e2e tests."""

    def __init__(self, cfg: RuntimeConfig) -> None:
        import threading
        import time as _time

        import uvicorn

        from apollo_xarm7_runtime.runtime import Runtime
        from apollo_xarm7_runtime.server.app import create_app

        self.runtime = Runtime(cfg)
        self._server = uvicorn.Server(
            uvicorn.Config(
                create_app(self.runtime),
                host="127.0.0.1",
                port=0,
                ws_per_message_deflate=False,  # binding (04-runtime §13.5)
                log_level="warning",
            )
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = _time.monotonic() + 15.0
        while not self._server.started:
            if _time.monotonic() > deadline:
                raise RuntimeError("uvicorn failed to start")
            _time.sleep(0.02)
        self.port = self._server.servers[0].sockets[0].getsockname()[1]
        self.http = f"http://127.0.0.1:{self.port}"
        self.ws = f"ws://127.0.0.1:{self.port}"

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10.0)
