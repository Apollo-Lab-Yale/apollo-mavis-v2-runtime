"""Helpers for the dora tests (14-dora §10); ``conftest.py`` exposes the fixtures.

Tier 1/2 tests (codec, dataflow, bridge over :class:`FakeNode`) need nothing.
Tier 3/4 tests are marked ``dora`` and SKIP unless the ``[dora]`` extra AND the
``dora`` CLI are present; they start a PRIVATE control plane on free ports.
PORT DISCIPLINE: never 6013 / 53291 / 7447, never ``dora up`` / ``dora down``
without our own ``--coordinator-port``, never ``pkill -f dora``; every process
this module spawns is reaped in the fixture's teardown.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

RESERVED_PORTS = {6013, 53291, 7447}


def free_port() -> int:
    while True:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        if port not in RESERVED_PORTS:
            return port


def dora_available() -> bool:
    try:
        import dora  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return dora_cli() is not None


def dora_cli() -> str | None:
    venv = Path(sys.executable).parent / "dora"
    if venv.is_file():
        return str(venv)
    return shutil.which("dora")


requires_dora = pytest.mark.skipif(not dora_available(), reason="[dora] extra / CLI missing")


def wait_until(pred, timeout: float, what: str = "", dt: float = 0.05):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = pred()
        if v:
            return v
        time.sleep(dt)
    raise TimeoutError(f"timed out after {timeout}s: {what}")


def dora_pids() -> list[int]:
    """PIDs of every ``dora`` process (exact name match; never -f)."""
    r = subprocess.run(["pgrep", "-x", "dora"], capture_output=True, text=True)
    return [int(p) for p in r.stdout.split()] if r.returncode == 0 else []


class ProcessSet:
    """Child processes a test started; killed (TERM -> KILL, whole session) on close."""

    def __init__(self) -> None:
        self.procs: list[subprocess.Popen] = []

    def spawn(
        self,
        cmd: list[str],
        *,
        cwd: Path | None = None,
        env: dict | None = None,
        log: Path | None = None,
    ) -> subprocess.Popen:
        out = open(log, "ab") if log is not None else subprocess.DEVNULL  # noqa: SIM115
        p = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        if log is not None:
            out.close()
        self.procs.append(p)
        return p

    def kill(self, p: subprocess.Popen, sig=None) -> None:
        import signal as _signal

        sig = sig or _signal.SIGTERM
        try:
            os.killpg(p.pid, sig)
        except ProcessLookupError:
            pass

    def close(self) -> None:
        import signal as _signal

        for p in self.procs:
            if p.poll() is None:
                self.kill(p, _signal.SIGTERM)
        deadline = time.monotonic() + 3.0
        for p in self.procs:
            while p.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            if p.poll() is None:
                self.kill(p, _signal.SIGKILL)
                try:
                    p.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    pass
        self.procs.clear()


def node_env(bind_ip: str, zenoh_port: int, extra: dict | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "DORA_ZENOH_CONNECT": f"tcp/{bind_ip}:{zenoh_port}",
            "DORA_ZENOH_MULTICAST": "off",
            "DORA_ZENOH_LISTEN": "tcp/127.0.0.1:0",
            "RUST_LOG": "error",
        }
    )
    env.pop("DORA_COORDINATOR_ADDR", None)
    env.pop("DORA_COORDINATOR_PORT", None)
    if extra:
        env.update(extra)
    return env


def dora_runtime_config(
    tmp_path: Path,
    *,
    scene: str = "mavis_v2",
    bind_host: str = "127.0.0.1",
    machines=(),
    auth=None,
    mic: bool = False,
    safety_debug: bool = False,
    arms=("view", "grip"),
    **dora_overrides,
):
    """A sim RuntimeConfig with the dora bridge ON, private free ports, var_dir in tmp."""
    from apollo_mavis_v2_runtime.config import (
        DatasetsConfig,
        DoraConfig,
        DoraMachineConfig,
        HardwareSessionConfig,
        MicrophoneConfig,
        RuntimeConfig,
        VideoConfig,
    )

    wc = {
        "kind": "sim",
        "sim_scene": scene,
        "arms": [
            {"id": a, "base_in_world": {}, **({"gripper": "none"} if a == "view" else {})}
            for a in arms
        ],
        "cameras": [],
        "safety": {"safety_debug": safety_debug},
    }
    dora = DoraConfig(
        enabled=True,
        bind_host=bind_host,
        machines=[DoraMachineConfig(**m) if isinstance(m, dict) else m for m in machines],
        auth=auth,
        coordinator_port=free_port(),
        daemon_port=free_port(),
        zenoh_port=free_port(),
        var_dir=tmp_path / "dora",
        **dora_overrides,
    )
    return RuntimeConfig(
        workcells={"sim": wc},
        profiles_dir=tmp_path / "profiles",
        datasets_root=tmp_path / "datasets",
        # no mapped namespace roots: never list / sweep the operator's ~/data from a test
        datasets=DatasetsConfig(default_namespace="apollo", namespaces={}),
        checkpoints_root=tmp_path / "ckpts",
        calibration_dir=tmp_path / "calibration",
        video=VideoConfig(preview_fps=15, session_fps=30),
        hardware_session=HardwareSessionConfig(armed=True),
        microphone=MicrophoneConfig(enabled=mic, backend="fake" if mic else "none"),
        dora=dora,
    )


HW_TESTS = Path(__file__).resolve().parents[3] / "apollo-mavis-v2-hardware" / "tests"


def load_fake_xarm_api():
    """The hardware package's ``FakeXArmAPI`` (``tests/fakes``, not in the wheel), loaded as a
    private package like ``tests/test_hardware_session.py`` does; skips when not checked out."""
    import importlib
    import importlib.util

    path = HW_TESTS / "fakes"
    if not (path / "fake_xarm_api.py").exists():
        pytest.skip("apollo-mavis-v2-hardware/tests/fakes not checked out next to the runtime")
    name = "hw_test_fakes"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, path / "__init__.py", submodule_search_locations=[str(path)]
        )
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[name] = pkg
        spec.loader.exec_module(pkg)
    return importlib.import_module(f"{name}.fake_xarm_api").FakeXArmAPI
