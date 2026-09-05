"""Shared fixtures. Sets MUJOCO_GL=egl before any mujoco GL init."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")  # noqa: E402 - must precede mujoco GL init

import socket
import threading
import time
from dataclasses import dataclass, field, replace

import numpy as np
import pytest
from apollo_mavis_v2_core.testing import FakeArm, FakeWorkcell

from apollo_mavis_v2_runtime.bus import RuntimeBus
from apollo_mavis_v2_runtime.config import ControlConfig, RuntimeConfig, TrackerConfig, VideoConfig
from apollo_mavis_v2_runtime.control.loop import ControlLoop
from apollo_mavis_v2_runtime.safety.gate import NullGate
from apollo_mavis_v2_runtime.safety.supervisor import SafetySupervisor
from apollo_mavis_v2_runtime.safety.watchdog import InputWatchdog

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
        calibration_dir=tmp_path / "calibration",  # never read ~/apollo/calibration in tests
        video=VideoConfig(preview_fps=15, session_fps=30),
        tracker=tracker or TrackerConfig(),
    )


@pytest.fixture
def fake_loop(tmp_path):
    """ControlLoop over FakeWorkcell with a NullGate; deterministic ticks."""
    from apollo_mavis_v2_core import ProfileStore

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


class AcceptingListener:
    """Loopback TCP listener that accepts and immediately closes every connection
    on a daemon thread. Use it wherever a poller (``HardwareProbe`` at 10 Hz x 2
    arms in the API tests) hits a "control box" for longer than a few rounds: a
    bare ``listen(N)`` socket holds only N+1 un-accepted connections, after which
    the kernel drops SYNs and every probe times out -> ``unreachable``."""

    def __init__(self, backlog: int = 8) -> None:
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(backlog)
        self._srv.settimeout(0.2)  # so the acceptor notices close() promptly
        self.port: int = self._srv.getsockname()[1]
        self.accepted = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._drain, name="test-acceptor", daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._srv.accept()
            except TimeoutError:
                continue
            except OSError:  # listener closed
                return
            self.accepted += 1
            conn.close()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._srv.close()


class LiveServer:
    """Real uvicorn (own thread, random port, deflate OFF) for e2e tests."""

    def __init__(self, cfg: RuntimeConfig) -> None:
        import threading
        import time as _time

        import uvicorn

        from apollo_mavis_v2_runtime.runtime import Runtime
        from apollo_mavis_v2_runtime.server.app import create_app

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


# -- phase-09a fakes: read-only arm monitor --------------------------------------------------
@dataclass(frozen=True)
class FakeMonitorSample:
    """Duck-typed stand-in for ``apollo_mavis_v2_hardware.ArmMonitorSample`` (same
    field names; the runtime never imports the hardware dataclass)."""

    arm_id: str
    seq: int = 1
    t_mono: float = 0.0
    q: tuple[float, ...] = (3.141592653589793, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    tcp_pose: tuple[float, ...] = (0.2075, 0.0, 0.1125, 3.141592653589793, 0.0, 0.0)
    error_code: int = 0
    warn_code: int = 0
    state: int | None = 4
    mode: int | None = 0
    rail_present: bool | None = True
    rail_homed: bool | None = False
    rail_enabled: bool | None = False
    rail_pos_m: float | None = None
    rail_raw_mm: float | None = 0.0
    gripper_open_frac: float | None = None
    gripper_raw: float | None = None
    # phase-09b safety read-backs (rich report frame; None / () until read)
    collision_sensitivity: int | None = None
    tcp_load_kg: float | None = None
    tcp_load_cog_mm: tuple[float, ...] = ()


@dataclass(frozen=True)
class FakeMaintenanceOutcome:
    """Duck-typed ``apollo_mavis_v2_hardware.MaintenanceOutcome`` (same field names)."""

    arm_id: str
    op: str
    ok: bool
    detail: str = ""
    sdk_codes: dict[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    before: object | None = None
    after: object | None = None


# backstops.BACKSTOP_SDK_METHODS order (the reduced-mode pair only with a boundary);
# tests/test_maintenance_api.py pins this against the hardware package when importable.
BACKSTOP_SEQUENCE: tuple[str, ...] = (
    "set_tcp_load",
    "set_gravity_direction",
    "set_collision_sensitivity",
    "set_self_collision_detection",
    "set_collision_tool_model",
    "set_collision_rebound",
)


def fake_backstop_sequence(driver_cfg) -> tuple[str, ...]:
    seq = list(BACKSTOP_SEQUENCE)
    if getattr(driver_cfg, "reduced_tcp_boundary_mm", None) is not None:
        seq[-1:-1] = ["set_reduced_tcp_boundary", "set_reduced_mode"]
    return tuple(seq)


@dataclass
class FakeArmMonitor:
    """``ArmStateMonitor`` surface with a call log; ``status`` / ``sample`` are set
    by the test. ``start()`` -> status ``running`` (unless the test pinned one),
    ``disconnect()`` -> ``paused``, ``stop()`` -> ``off``.

    ``maintenance()`` (phase-09b) mirrors the hardware monitor's contract: refusals
    for ``recover`` / a missing driver config / a disconnected monitor;
    ``clear_errors`` writes exactly ``clean_error`` + ``clean_warn`` and zeroes the
    codes of the newest sample; ``apply_backstops`` writes the ``backstops.py``
    sequence and echoes the driver config into the read-back fields. A test can
    pin ``scripted_outcome`` (returned verbatim) or ``maintenance_busy``."""

    arm_id: str
    ip: str
    gripper: str = "none"
    expect_rail: bool = True
    poll_hz: float = 10.0
    stale_s: float = 0.5
    reconnect_s: float = 2.0
    calls: list[str] = field(default_factory=list)
    sample: object | None = None
    forced_status: str | None = None  # e.g. "stale" / "error" pinned by a test
    detail_text: str = ""
    maintenance_busy: bool = False
    maintenance_calls: list[tuple[str, object, float]] = field(default_factory=list)
    scripted_outcome: object | None = None
    _status: str = "off"

    def start(self) -> None:
        self.calls.append("start")
        self._status = "running"

    def stop(self, timeout: float = 2.0) -> None:
        self.calls.append("stop")
        self._status = "off"

    def disconnect(self, timeout: float = 2.0) -> None:
        self.calls.append("disconnect")
        self._status = "paused"

    def snapshot(self):
        return self.sample

    def maintenance(self, op: str, driver_cfg=None, timeout_s: float = 10.0):
        self.maintenance_calls.append((op, driver_cfg, timeout_s))
        if op not in ("clear_errors", "apply_backstops", "recover"):
            raise ValueError(f"unknown maintenance op {op!r}")
        if op == "recover":
            return FakeMaintenanceOutcome(self.arm_id, op, False, "recover needs a session")
        if op == "apply_backstops" and driver_cfg is None:
            return FakeMaintenanceOutcome(
                self.arm_id, op, False, "apply_backstops needs the arm's driver config"
            )
        if self._status != "running":
            return FakeMaintenanceOutcome(
                self.arm_id, op, False, f"not connected to {self.ip} (monitor {self._status})"
            )
        if self.scripted_outcome is not None:
            return self.scripted_outcome
        before = self.sample
        if op == "clear_errors":
            codes = {"clean_error": 0, "clean_warn": 0}
            err = getattr(before, "error_code", 0) if before is not None else 0
            detail = (
                f"cleared controller error {err}"
                if err
                else "no controller error or warning was latched; clean_error + clean_warn sent"
            )
            after = (
                replace(before, error_code=0, warn_code=0, seq=before.seq + 2)
                if before is not None
                else None
            )
        else:
            codes = dict.fromkeys(fake_backstop_sequence(driver_cfg), 0)
            cog = tuple(float(v) for v in driver_cfg.tcp_load_cog_mm)
            detail = (
                f"safety settings applied: sensitivity {driver_cfg.collision_sensitivity}, "
                f"payload {driver_cfg.tcp_load_kg:.2f} kg at "
                f"({', '.join(f'{v:g}' for v in cog)}) mm"
            )
            after = (
                replace(
                    before,
                    seq=before.seq + 2,
                    collision_sensitivity=int(driver_cfg.collision_sensitivity),
                    tcp_load_kg=float(driver_cfg.tcp_load_kg),
                    tcp_load_cog_mm=cog,
                )
                if before is not None
                else None
            )
        if after is not None:
            self.sample = after  # the monitor's newest sample reflects the op
        return FakeMaintenanceOutcome(self.arm_id, op, True, detail, codes, (), before, after)

    @property
    def status(self) -> str:
        if self.forced_status is not None and self._status == "running":
            return self.forced_status
        return self._status

    @property
    def detail(self) -> str:
        return self.detail_text

    @property
    def age_s(self) -> float | None:
        return None if self.sample is None else max(0.0, time.monotonic() - self.sample.t_mono)

    @property
    def connected(self) -> bool:
        return self._status == "running"


class FakeMonitorFactory:
    """``monitor_factory`` seam: records every constructed :class:`FakeArmMonitor`."""

    def __init__(self, samples: dict[str, object] | None = None) -> None:
        self.monitors: dict[str, FakeArmMonitor] = {}
        self.samples = samples or {}

    def __call__(self, arm_id: str, ip: str, **kw) -> FakeArmMonitor:
        mon = FakeArmMonitor(arm_id, ip, **kw)
        mon.sample = self.samples.get(arm_id)
        self.monitors[arm_id] = mon
        return mon
