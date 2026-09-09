"""The runtime-owned PRIVATE dora control plane (14-dora §2.2, §9).

``DoraControlPlane`` spawns ``dora coordinator`` + ``dora daemon`` as children
(own session via ``setsid``, cwd = ``var_dir``, private ports), waits for
``dora status``, ``dora start``s the rendered dataflow, re-reads the set of
registered daemons (``dora doctor`` is the ONE 1.0.1 CLI surface that lists
them — ``dora status --format json`` reports a single ``daemon.status`` and
``dora list`` only dataflows, verified 2026-09-07), and tears everything down
in the documented order (``dora stop`` -> ``dora down --coordinator-port
<own>`` -> SIGTERM/SIGKILL of the two PIDs it spawned; ``dora down`` reports
success even when it leaves processes behind). It never runs ``dora down`` /
``destroy`` against a port it did not spawn.

Every CLI call carries ``--coordinator-addr <bind_ip> --coordinator-port
<P>``: with ``--interface <LAN IP>`` the coordinator does NOT listen on
loopback. With ``auth`` on, the token the coordinator wrote to
``<var_dir>/.dora-token`` (its cwd) is handed to the daemon and to every CLI
call through ``DORA_AUTH_TOKEN`` — never through ``GET /api/dora``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import DoraConfig
from .netaddr import BindHostError, is_loopback, resolve_and_vet

logger = logging.getLogger(__name__)

STATUS_WAIT_S = 5.0  # coordinator + daemon must answer `dora status` within this
STATUS_POLL_S = 0.1
STOP_GRACE = "2s"
CLI_TIMEOUT_S = 15.0
REAP_TERM_WAIT_S = 3.0
TOKEN_FILE = ".dora-token"
_MACHINE_RE = re.compile(
    r"^\s*(?P<id>.+?)-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\s+\(heartbeat",
    re.IGNORECASE,
)
_DATAFLOW_RE = re.compile(r"dataflow started:\s*([0-9a-f-]{36})", re.IGNORECASE)


class ControlPlaneError(RuntimeError):
    """Bring-up / dataflow failure; ``str(exc)`` is the ``ExternalStatus.detail``."""


def dora_cli_path() -> str | None:
    """The ``dora`` binary of THIS interpreter's venv (never a stray PATH one)."""
    venv_bin = Path(sys.executable).parent / "dora"
    if venv_bin.is_file() and os.access(venv_bin, os.X_OK):
        return str(venv_bin)
    return shutil.which("dora")


def python_dora_version() -> str | None:
    try:
        import dora  # noqa: TID251 - the sanctioned lazy import site
    except Exception:  # noqa: BLE001 - missing extra / broken wheel
        return None
    return str(getattr(dora, "__version__", "") or "")


def cli_dora_version(cli: str | None) -> str | None:
    if not cli:
        return None
    try:
        out = subprocess.run([cli, "--version"], capture_output=True, text=True, timeout=10.0)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (out.stdout or out.stderr or "").strip()
    m = re.search(r"(\d+\.\d+\.\d+\S*)", text)
    return m.group(1) if m else (text or None)


def check_versions() -> str | None:
    """``None`` when the python package and the CLI agree; else the ``disabled`` detail."""
    py = python_dora_version()
    if py is None:
        return "dora python package not importable (install the runtime [dora] extra)"
    cli = dora_cli_path()
    if cli is None:
        return "dora CLI not found in the runtime venv (install the runtime [dora] extra)"
    cv = cli_dora_version(cli)
    if cv is None:
        return f"dora CLI at {cli} did not report a version"
    if cv != py:
        return f"dora version mismatch: python dora {py} vs CLI {cv} (upgrade both together)"
    return None


def parse_registered_machines(doctor_text: str) -> set[str]:
    """Machine ids from ``dora doctor``'s ``Connected machines`` block (ids are
    ``<machine_id>-<uuid7>``; the uuid is stripped)."""
    out: set[str] = set()
    for line in doctor_text.splitlines():
        m = _MACHINE_RE.match(line)
        if m:
            out.add(m.group("id"))
    return out


@dataclass
class ControlPlaneFacts:
    """What foreign clients need (feeds ``GET /api/dora``)."""

    bind_ip: str
    coordinator_port: int
    daemon_port: int
    zenoh_port: int
    machine_id: str
    auth: bool

    @property
    def zenoh_connect(self) -> str:
        return f"tcp/{self.bind_ip}:{self.zenoh_port}"

    def node_env(self) -> dict[str, str]:
        """Environment a dynamic node on THIS host sets before ``Node(...)`` (14-dora §9)."""
        return {
            "DORA_ZENOH_CONNECT": self.zenoh_connect,
            "DORA_ZENOH_MULTICAST": "off",
            "DORA_ZENOH_LISTEN": "tcp/127.0.0.1:0",
        }


@dataclass
class DoraControlPlane:
    """Spawn / query / tear down the private coordinator + daemon pair."""

    cfg: DoraConfig
    var_dir: Path
    arm_ips: tuple[str | None, ...] = ()
    cli: str | None = field(default_factory=dora_cli_path)
    bind_ip: str | None = None
    coordinator: subprocess.Popen | None = None
    daemon: subprocess.Popen | None = None
    dataflow_id: str | None = None
    token: str | None = None
    started_at: float | None = None

    # -- facts ------------------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return (
            self.coordinator is not None
            and self.coordinator.poll() is None
            and self.daemon is not None
            and self.daemon.poll() is None
        )

    @property
    def auth(self) -> bool:
        return self.cfg.auth_effective

    def facts(self) -> ControlPlaneFacts:
        ip = self.bind_ip or self.resolve_bind_ip()
        return ControlPlaneFacts(
            bind_ip=ip,
            coordinator_port=self.cfg.coordinator_port,
            daemon_port=self.cfg.daemon_port,
            zenoh_port=self.cfg.zenoh_port,
            machine_id=self.cfg.machine_id,
            auth=self.auth,
        )

    def resolve_bind_ip(self) -> str:
        """Resolve + vet ``cfg.bind_host`` NOW (re-run before every dataflow restart:
        the lab Wi-Fi address is DHCP). Raises :class:`BindHostError`."""
        ip = resolve_and_vet(self.cfg.bind_host, self.arm_ips)
        self.bind_ip = ip
        return ip

    # -- CLI plumbing --------------------------------------------------------------------------
    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("RUST_LOG", "error")
        if self.token:
            env["DORA_AUTH_TOKEN"] = self.token
        # never let a stray DORA_COORDINATOR_* from the shell redirect our calls
        env.pop("DORA_COORDINATOR_ADDR", None)
        env.pop("DORA_COORDINATOR_PORT", None)
        return env

    def _coord_args(self) -> list[str]:
        assert self.bind_ip is not None
        return [
            "--coordinator-addr",
            self.bind_ip,
            "--coordinator-port",
            str(self.cfg.coordinator_port),
        ]

    def cli_run(
        self, *args: str, timeout: float = CLI_TIMEOUT_S, coordinator: bool = True
    ) -> subprocess.CompletedProcess:
        if not self.cli:
            raise ControlPlaneError("dora CLI missing")
        cmd = [self.cli, *args]
        if coordinator:
            cmd += self._coord_args()
        t0 = time.monotonic()
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.var_dir),
                env=self._env(),
            )
        except subprocess.TimeoutExpired:
            logger.warning("dora %s timed out after %.0f s", " ".join(args[:2]), timeout)
            raise
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "dora %s -> rc %d in %.0f ms%s",
                " ".join(args[:2]),
                res.returncode,
                (time.monotonic() - t0) * 1e3,
                f" ({(res.stdout + res.stderr).strip()[-160:]!r})" if res.returncode else "",
            )
        return res

    # -- bring-up (§2.2 steps 3-4) -----------------------------------------------------------------
    def start(self) -> None:
        """Spawn coordinator + daemon and wait for ``dora status`` (<= 5 s)."""
        if not self.cli:
            raise ControlPlaneError("dora CLI missing")
        ip = self.resolve_bind_ip()
        self.var_dir.mkdir(parents=True, exist_ok=True)
        self.token = None
        token_path = self.var_dir / TOKEN_FILE
        if self.auth and token_path.exists():
            token_path.unlink()  # the new coordinator writes a fresh one
        coord_cmd = [
            self.cli,
            "coordinator",
            "--interface",
            ip,
            "--port",
            str(self.cfg.coordinator_port),
            "--store",
            "memory",
            "--quiet",
        ]
        if self.auth:
            coord_cmd.append("--auth")
        self.coordinator = self._spawn(coord_cmd, "coordinator.log")
        if self.auth:
            self.token = self._await_token(token_path)
        daemon_cmd = [
            self.cli,
            "daemon",
            "--machine-id",
            self.cfg.machine_id,
            "--coordinator-addr",
            ip,
            "--coordinator-port",
            str(self.cfg.coordinator_port),
            "--local-listen-port",
            str(self.cfg.daemon_port),
            "--zenoh-no-multicast",
            "--zenoh-listen",
            f"{ip}:{self.cfg.zenoh_port}",
            "--quiet",
        ]
        self.daemon = self._spawn(daemon_cmd, "daemon.log")
        self._await_status()
        self.started_at = time.monotonic()
        logger.info(
            "dora control plane up: coordinator %s:%d daemon %s (node port 127.0.0.1:%d, "
            "zenoh %s:%d)%s",
            ip,
            self.cfg.coordinator_port,
            self.cfg.machine_id,
            self.cfg.daemon_port,
            ip,
            self.cfg.zenoh_port,
            f" auth ON (token in {token_path})" if self.auth else "",
        )

    def _spawn(self, cmd: list[str], log_name: str) -> subprocess.Popen:
        log_path = self.var_dir / log_name
        try:
            log = open(log_path, "ab")  # noqa: SIM115 - handed to the child
        except OSError as exc:
            raise ControlPlaneError(f"cannot open {log_path}: {exc}") from exc
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.var_dir),
                env=self._env(),
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            log.close()
            raise ControlPlaneError(f"cannot spawn {cmd[1]}: {exc}") from exc
        finally:
            pass
        log.close()  # the child holds its own descriptor
        return proc

    def _await_token(self, path: Path, timeout: float = STATUS_WAIT_S) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.is_file():
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    return text
            if self.coordinator is not None and self.coordinator.poll() is not None:
                break
            time.sleep(STATUS_POLL_S)
        self.reap()
        raise ControlPlaneError(f"coordinator did not write {path} within {timeout:.0f} s")

    def _await_status(self, timeout: float = STATUS_WAIT_S) -> None:
        deadline = time.monotonic() + timeout
        last = ""
        while time.monotonic() < deadline:
            for proc, what in ((self.coordinator, "coordinator"), (self.daemon, "daemon")):
                if proc is not None and proc.poll() is not None:
                    self.reap()
                    raise ControlPlaneError(
                        f"dora {what} exited with code {proc.returncode} (port busy? see "
                        f"{self.var_dir / (what + '.log')})"
                    )
            try:
                res = self.cli_run("status", "--format", "json", timeout=5.0)
            except (subprocess.TimeoutExpired, ControlPlaneError) as exc:
                last = str(exc)
                time.sleep(STATUS_POLL_S)
                continue
            if res.returncode == 0 and '"running"' in res.stdout and '"daemon"' in res.stdout:
                daemon_ok = re.search(r'"daemon"\s*:\s*\{\s*"status"\s*:\s*"running"', res.stdout)
                if daemon_ok:
                    return
            last = (res.stdout + res.stderr).strip()[-300:]
            time.sleep(STATUS_POLL_S)
        self.reap()
        raise ControlPlaneError(f"dora status not healthy within {timeout:.0f} s: {last}")

    # -- dataflow ----------------------------------------------------------------------------------
    def validate(self, yaml_path: Path) -> None:
        res = self.cli_run("validate", "--strict-types", str(yaml_path), coordinator=False)
        if res.returncode != 0:
            raise ControlPlaneError(
                f"dora validate failed: {(res.stdout + res.stderr).strip()[-400:]}"
            )

    def start_dataflow(self, yaml_path: Path) -> str:
        """``dora start <yaml> --name <dataflow_name> --detach`` -> dataflow id."""
        res = self.cli_run(
            "start", str(yaml_path), "--name", self.cfg.dataflow_name, "--detach", timeout=60.0
        )
        text = res.stdout + res.stderr
        m = _DATAFLOW_RE.search(text)
        if res.returncode != 0 or not m:
            raise ControlPlaneError(f"dora start failed: {text.strip()[-400:]}")
        self.dataflow_id = m.group(1)
        return self.dataflow_id

    def stop_dataflow(self) -> None:
        if self.dataflow_id is None or not self.running:
            self.dataflow_id = None
            return
        try:
            self.cli_run(
                "stop",
                "--name",
                self.cfg.dataflow_name,
                "--grace-duration",
                STOP_GRACE,
                timeout=20.0,
            )
        except (subprocess.TimeoutExpired, ControlPlaneError, OSError):
            logger.warning("dora stop failed", exc_info=True)
        self.dataflow_id = None

    def dataflow_running(self) -> bool | None:
        """``dora list`` says the named dataflow is Running (None = unknown)."""
        try:
            res = self.cli_run("list", timeout=5.0)
        except (subprocess.TimeoutExpired, ControlPlaneError, OSError):
            return None
        if res.returncode != 0:
            return None
        for line in res.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[1] == self.cfg.dataflow_name:
                return parts[2] == "Running"
        return False

    def node_status(self, node_id: str) -> str | None:
        """STATUS of ``node_id`` in the CURRENT dataflow (``dora node list --dataflow <uuid>
        --format json``; ``None`` = unknown / failed / no dataflow). Filtered by UUID: after an
        unclean stop (a node on a dead machine) the previous same-named dataflow can keep
        listing its nodes, and its Running ``probe`` once fooled the join's barrier check."""
        if self.dataflow_id is None:
            return None
        try:
            res = self.cli_run(
                "node", "list", "--dataflow", self.dataflow_id, "--format", "json", timeout=10.0
            )
        except (subprocess.TimeoutExpired, ControlPlaneError, OSError):
            return None
        if res.returncode != 0:
            return None
        for line in res.stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if str(row.get("node") or row.get("id") or row.get("node_id")) == node_id:
                st = row.get("status")
                return str(st) if st is not None else None
        return None

    # -- registered machines (§2.2 step 6) ---------------------------------------------------------
    def registered_machines(self) -> set[str] | None:
        """Machine ids of the daemons registered at our coordinator, via ``dora doctor``
        (``None`` when the CLI call failed)."""
        try:
            res = self.cli_run("doctor", timeout=10.0)
        except (subprocess.TimeoutExpired, ControlPlaneError, OSError) as exc:
            logger.warning("dora doctor failed: %s", str(exc)[:400])
            return None
        if res.returncode != 0 and "Connected machines" not in res.stdout:
            logger.warning(
                "dora doctor rc=%s without a machine list; stdout tail %r stderr tail %r",
                res.returncode,
                res.stdout[-300:],
                res.stderr[-300:],
            )
            return None
        return parse_registered_machines(res.stdout)

    # -- teardown (§2.2 "Shutdown") ----------------------------------------------------------------
    def shutdown(self) -> None:
        self.stop_dataflow()
        if self.running and self.bind_ip is not None:
            try:
                self.cli_run("down", timeout=20.0)  # ONLY ever against our own port
            except (subprocess.TimeoutExpired, ControlPlaneError, OSError):
                logger.warning("dora down failed", exc_info=True)
        self.reap()

    def reap(self) -> None:
        """SIGTERM -> SIGKILL the two PIDs we spawned (and their process groups)."""
        procs = [p for p in (self.daemon, self.coordinator) if p is not None]
        for p in procs:
            if p.poll() is None:
                _signal_group(p, signal.SIGTERM)
        deadline = time.monotonic() + REAP_TERM_WAIT_S
        for p in procs:
            while p.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            if p.poll() is None:
                _signal_group(p, signal.SIGKILL)
                try:
                    p.wait(timeout=2.0)
                except subprocess.TimeoutExpired:  # pragma: no cover - kernel refused
                    logger.error("dora child pid %d survived SIGKILL", p.pid)
        self.daemon = None
        self.coordinator = None
        self.dataflow_id = None

    def child_pids(self) -> list[int]:
        return [
            p.pid for p in (self.coordinator, self.daemon) if p is not None and p.poll() is None
        ]

    def info(self) -> dict[str, Any]:
        return {
            "bind_ip": self.bind_ip,
            "running": self.running,
            "pids": self.child_pids(),
            "dataflow_id": self.dataflow_id,
            "auth": self.auth,
        }


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)  # start_new_session=True -> pgid == pid
    except ProcessLookupError:
        pass
    except PermissionError:  # pragma: no cover
        proc.send_signal(sig)


__all__ = [
    "BindHostError",
    "ControlPlaneError",
    "ControlPlaneFacts",
    "DoraControlPlane",
    "check_versions",
    "cli_dora_version",
    "dora_cli_path",
    "is_loopback",
    "parse_registered_machines",
    "python_dora_version",
]
