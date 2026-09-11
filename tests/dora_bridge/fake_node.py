"""In-memory ``dora.Node`` duck type for bridge unit tests (14-dora §10 tier 2).

Implements exactly what ``DoraBridge`` touches: ``try_recv`` / ``next`` /
``send_output`` / ``node_config`` / ``dataflow_id``. Tests script inbound
events with :meth:`push` and read outbound sends from :attr:`sent`.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any

from apollo_mavis_v2_core.protocol import external as ext


class FakeNode:
    def __init__(
        self,
        node_id: str = ext.EXTERNAL_NODE_ID,
        daemon_port: int = 0,
        outputs: list[str] | None = None,
        inputs: list[str] | None = None,  # default: the fixed RUNTIME_INPUTS (no per-arm rows)
    ) -> None:
        self.node_id = node_id
        self.daemon_port = daemon_port
        self._events: deque[dict[str, Any]] = deque()
        self._lock = threading.Lock()
        self.sent: list[tuple[str, Any, dict[str, Any]]] = []
        self.fail_send = False
        self._outputs = outputs
        self._inputs = list(inputs) if inputs is not None else list(ext.RUNTIME_INPUTS)
        self.closed = False

    # -- test side ------------------------------------------------------------------------------
    def push(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._events.append(event)

    def push_input(
        self, input_id: str, value: Any = None, metadata: dict[str, Any] | None = None
    ) -> None:
        self.push(
            {"type": "INPUT", "id": input_id, "value": value, "metadata": dict(metadata or {})}
        )

    def sent_ids(self) -> list[str]:
        return [s[0] for s in self.sent]

    def sent_for(self, output_id: str) -> list[tuple[Any, dict[str, Any]]]:
        return [(v, m) for oid, v, m in self.sent if oid == output_id]

    # -- node surface -----------------------------------------------------------------------------
    def try_recv(self) -> dict[str, Any] | None:
        with self._lock:
            return self._events.popleft() if self._events else None

    def next(self, timeout: float = 0.0) -> dict[str, Any] | None:
        ev = self.try_recv()
        if ev is None:
            return {"type": "ERROR", "error": "Timeout event stream error: Receiver timed out"}
        return ev

    def send_output(
        self, output_id: str, data: Any, metadata: dict[str, Any] | None = None
    ) -> None:
        if self.fail_send:
            raise RuntimeError("fatal event stream error: daemon channel broken")
        self.sent.append((output_id, data, dict(metadata or {})))

    def node_config(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "inputs": dict.fromkeys(self._inputs, {}),
            "outputs": list(self._outputs) if self._outputs is not None else None,
        }

    def dataflow_id(self) -> str:
        return "fake-dataflow"


class FakeControlPlane:
    """``DoraControlPlane`` surface with scripted outcomes."""

    def __init__(
        self, cfg, var_dir, arm_ips=(), *, start_fails: int = 0, machines: set[str] | None = None
    ) -> None:
        self.cfg = cfg
        self.var_dir = var_dir
        self.arm_ips = tuple(arm_ips)
        self.bind_ip: str | None = None
        self.running = False
        self.dataflow_id: str | None = None
        self.start_fails = start_fails
        self.machines: set[str] = set(machines or ())
        self.starts = 0
        self.dataflow_starts = 0
        self.stops = 0
        self.reaps = 0
        self.downs = 0
        self.validated: list[str] = []
        self.token = None

    def resolve_bind_ip(self) -> str:
        from apollo_mavis_v2_runtime.dora_bridge.netaddr import resolve_and_vet

        self.bind_ip = resolve_and_vet(self.cfg.bind_host, self.arm_ips)
        return self.bind_ip

    def facts(self):
        from apollo_mavis_v2_runtime.dora_bridge.control_plane import ControlPlaneFacts

        return ControlPlaneFacts(
            bind_ip=self.bind_ip or self.resolve_bind_ip(),
            coordinator_port=self.cfg.coordinator_port,
            daemon_port=self.cfg.daemon_port,
            zenoh_port=self.cfg.zenoh_port,
            machine_id=self.cfg.machine_id,
            auth=self.cfg.auth_effective,
        )

    def start(self) -> None:
        from apollo_mavis_v2_runtime.dora_bridge.control_plane import ControlPlaneError

        self.starts += 1
        if self.start_fails > 0:
            self.start_fails -= 1
            raise ControlPlaneError("dora coordinator exited with code 1 (port busy)")
        self.resolve_bind_ip()
        self.running = True

    def validate(self, yaml_path) -> None:
        self.validated.append(str(yaml_path))

    def start_dataflow(self, yaml_path) -> str:
        self.dataflow_starts += 1
        self.dataflow_id = f"df-{self.dataflow_starts}"
        return self.dataflow_id

    def stop_dataflow(self) -> None:
        self.stops += 1
        self.dataflow_id = None

    def dataflow_running(self):
        return self.dataflow_id is not None

    def registered_machines(self):
        return set(self.machines)

    barrier_open = True  # the remote consumer "attached": the probe node reads Running

    def node_status(self, node_id: str):
        return "Running" if self.barrier_open else "Unknown"

    def shutdown(self) -> None:
        self.stop_dataflow()
        self.downs += 1
        self.reap()

    def reap(self) -> None:
        self.reaps += 1
        self.running = False
        self.dataflow_id = None

    def child_pids(self):
        return []

    def info(self):
        return {"running": self.running}


__all__ = ["FakeControlPlane", "FakeNode"]
