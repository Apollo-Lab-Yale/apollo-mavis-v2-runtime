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
from apollo_mavis_v2_runtime.config import (
    ControlConfig,
    DatasetsConfig,
    HardwareSessionConfig,
    RuntimeConfig,
    TrackerConfig,
    VideoConfig,
)
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
        # Dataset roots (2026-09-08; 15-online-dagger §7 / D5): the runtime default maps
        # bc_demo / online_dagger to ~/data; tests pin the pre-D5 layout (bare names ->
        # apollo/<name> under the tmp datasets_root, no mapped roots) so the e2e
        # expectations hold AND no test ever lists or sweeps the operator's ~/data.
        datasets=DatasetsConfig(default_namespace="apollo", namespaces={}),
        checkpoints_root=tmp_path / "ckpts",
        calibration_dir=tmp_path / "calibration",  # never read ~/apollo/calibration in tests
        video=VideoConfig(preview_fps=15, session_fps=30),
        # Keyboard translate frame: the e2e suites here assert displacement along
        # their test scenes' arm-base axes to test the WS / watchdog / gate plumbing,
        # not the frame, so they stay pinned to the pre-2026-09-08 "base" frame. The
        # RUNTIME default is "world" (operator-fixed; decision of 2026-09-08 evening —
        # that morning's default "camera" pointed W along the tool axis, straight DOWN
        # at every test scene's folded start posture, where the gate legitimately
        # stops it). The frame itself is covered by tests/test_teleop_math.py and
        # tests/test_camera_frame.py.
        # `orphan_session_grace_s=0` DISABLES the orphaned-session watch for every test
        # (2026-09-09; session/orphan.py, 04-runtime §13.2). The suites here drive sessions
        # over REST and the bus without ever opening a controller /ws/control socket, which
        # is exactly what the watch reads as abandoned — with the shipped 30 s default a
        # long e2e test would have its session torn down under its assertions. The watch
        # itself is covered by tests/test_orphan_session.py with an injected clock.
        control=ControlConfig(translate_frame="base", orphan_session_grace_s=0.0),
        tracker=tracker or TrackerConfig(),
        # tests run against fakes only: arm the (fake) hardware paths so home_rail jobs and
        # hardware sessions can be exercised; the real-SDK path stays behind the conftest
        # guard below, and test_hardware_session pins the unarmed refusal explicitly.
        hardware_session=HardwareSessionConfig(armed=True),
    )


@pytest.fixture(autouse=True)
def _never_touch_real_hardware(monkeypatch):
    """Safety net for EVERY runtime test (phase-09d): the lab machine can reach both
    control boxes, so a code path that builds the REAL ``HardwareWorkcell`` with the
    real SDK (no ``driver_api_factory`` fake) must fail loudly instead of connecting,
    enabling or moving an arm. Tests that want the real workcell classes set
    ``manager.driver_api_factory`` to a ``FakeXArmAPI`` factory first."""
    from apollo_mavis_v2_runtime.errors import SessionError
    from apollo_mavis_v2_runtime.session.manager import SessionManager

    real = SessionManager._default_workcell_factory

    def guarded(self, session_cfg, driver_factory):
        if self.driver_api_factory is None:
            raise SessionError(
                "TEST GUARD: refusing to build a real HardwareWorkcell against the real xArm "
                "SDK (set manager.workcell_factory or manager.driver_api_factory)"
            )
        return real(self, session_cfg, driver_factory)

    monkeypatch.setattr(SessionManager, "_default_workcell_factory", guarded)


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


_DISPATCHED: dict[int, dict[str, float]] = {}  # id(loop) -> arm -> put_mono last sent


def dispatch_commands(loop: ControlLoop, cell: FakeWorkcell) -> None:
    """What the ``ArmSender`` threads do in a live loop, synchronously: hand each
    arm's newest ``q_cmd`` slot value to the fake arm's ``command_joints`` (once per
    put - a stale value is never re-sent, exactly like ``wait_fresh``; a FAULTED arm's
    sender is paused and dispatches nothing). Without this the fakes' MEASURED q never
    follows the command in the hand-built sessions, and since 2026-09-08 the manager's
    sequential execution waits for measured arrival (``SessionManager._await_arrival``)."""
    sent = _DISPATCHED.setdefault(id(loop), {})
    for arm_id in loop.session_arms:
        got = loop.bus.arm_slot(arm_id).get()
        if got is None or arm_id in loop.faulted_arms:
            continue
        q, put_mono = got
        if sent.get(arm_id) == put_mono:
            continue
        sent[arm_id] = put_mono
        arm = cell.arms.get(arm_id)
        if arm is not None:
            arm.command_joints(np.asarray(q, dtype=np.float64))


def run_ticks(loop: ControlLoop, cell: FakeWorkcell, n: int, t0: float = 0.0) -> float:
    """Advance loop + fake workcell n deterministic ticks (the loop's output dispatched
    to the fake arms in between, see :func:`dispatch_commands`); returns end time."""
    t = t0
    for _ in range(n):
        t += loop.dt
        loop.run_tick(t)
        dispatch_commands(loop, cell)
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
    sequence and echoes the driver config into the read-back fields;
    ``home_rail`` (phase-09c) refuses without ``expected_q`` / on a posture
    mismatch (``q_tol_rad``) / with an error latched, else writes exactly the
    hardware ``MAINTENANCE_SDK_METHODS["home_rail"]`` set and the newest sample
    reads homed + enabled at 0.000 m (``homing_calls`` counts them). A test can
    pin ``scripted_outcome`` (returned verbatim) or ``maintenance_busy``; ``join``
    returns ``join_result`` (False = the poll thread is still inside the SDK)."""

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
    join_result: bool = True  # phase-09c: False = poll thread still inside the SDK
    homing_calls: list[tuple[float, ...]] = field(default_factory=list)  # expected_q per homing
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

    def join(self, timeout: float | None = None) -> bool:
        self.calls.append("join")
        return self.join_result

    def snapshot(self):
        return self.sample

    def maintenance(
        self,
        op: str,
        driver_cfg=None,
        timeout_s: float | None = 10.0,
        *,
        expected_q=None,
        q_tol_rad: float = 0.02,
    ):
        self.maintenance_calls.append((op, driver_cfg, timeout_s))
        if op not in ("clear_errors", "apply_backstops", "recover", "home_rail"):
            raise ValueError(f"unknown maintenance op {op!r}")
        if op == "recover":
            return FakeMaintenanceOutcome(self.arm_id, op, False, "recover needs a session")
        if op == "apply_backstops" and driver_cfg is None:
            return FakeMaintenanceOutcome(
                self.arm_id, op, False, "apply_backstops needs the arm's driver config"
            )
        if op == "home_rail":
            if driver_cfg is None:
                return FakeMaintenanceOutcome(
                    self.arm_id, op, False, "home_rail needs the arm's driver config (rail speed)"
                )
            if expected_q is None or len(expected_q) != 7:
                return FakeMaintenanceOutcome(
                    self.arm_id,
                    op,
                    False,
                    "home_rail needs expected_q: the 7 joint angles the rail sweep was checked at",
                )
        if self._status != "running":
            return FakeMaintenanceOutcome(
                self.arm_id, op, False, f"not connected to {self.ip} (monitor {self._status})"
            )
        if self.scripted_outcome is not None:
            return self.scripted_outcome
        before = self.sample
        if op == "home_rail":
            return self._home_rail(before, driver_cfg, tuple(expected_q), q_tol_rad)
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

    def _home_rail(self, before, driver_cfg, expected_q, q_tol_rad):
        """Mirror of the hardware monitor's ``home_rail`` branch (zero writes on a
        refusal; judged from the after-sample registers)."""
        op = "home_rail"
        if before is None:
            return FakeMaintenanceOutcome(self.arm_id, op, False, "home_rail refused: no sample")
        if before.error_code:
            return FakeMaintenanceOutcome(
                self.arm_id,
                op,
                False,
                f"home_rail refused: controller error {before.error_code} is latched",
                before=before,
            )
        worst = max(range(7), key=lambda i: abs(before.q[i] - expected_q[i]))
        dev = abs(before.q[worst] - expected_q[worst])
        if dev > q_tol_rad:
            return FakeMaintenanceOutcome(
                self.arm_id,
                op,
                False,
                f"home_rail refused: the arm moved since the sweep (joint {worst + 1} differs "
                f"by {dev:.3f} rad, tolerance {q_tol_rad:g} rad); re-run the sweep",
                before=before,
            )
        self.homing_calls.append(tuple(expected_q))
        speed = int(getattr(driver_cfg, "rail_speed_mm_s", 50))
        codes = {
            "set_linear_track_back_origin": 0,
            "set_linear_track_enable": 0,
            "set_linear_track_speed": 0,
        }
        after = replace(
            before,
            seq=before.seq + 2,
            rail_present=True,
            rail_homed=True,
            rail_enabled=True,
            rail_pos_m=0.0,
            rail_raw_mm=0.0,
        )
        self.sample = after
        detail = (
            "rail homed: carriage at 0.000 m (register 0 mm), track enabled, "
            f"positioning speed {speed} mm/s"
        )
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
