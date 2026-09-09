"""``DoraBridge`` — the runtime's dynamic node ``mavis_runtime`` (14-dora §2.4, §2.5, §8).

State machine ``disabled | unavailable | attached | detached | closed`` and ONE
thread, ``dora-bus``, that owns the ``dora.Node`` handle: every
``send_output`` / ``try_recv`` happens there (the 1.0.1 node methods take an
internal ``try_lock``; re-entry from a second thread while the GIL is released
panics). Producers never block: :meth:`try_publish` drops a payload thunk into a
depth-1 slot per topic (images, state, telemetry) or a bounded FIFO
(``events``, ``policy_reset``) and wakes the bus; the bus drains slots
newest-first, then ``try_recv()`` until empty, then waits <= ``bus_poll_s``.

Bring-up runs on the bus thread too (§2.2 steps 1-5, all best-effort):
``check_versions`` -> resolve/vet ``bind_host`` -> spawn the private control
plane -> render + validate + ``dora start`` -> set the node env
(``DORA_ZENOH_CONNECT`` / ``_MULTICAST=off`` / ``_LISTEN=tcp/127.0.0.1:0``) ->
``fcntl.flock(<var_dir>/mavis_runtime.lock)`` -> ``Node(node_id, daemon_port)``
-> ``node_config()`` check -> ``attached`` -> one dummy frame per camera
output (SHM pre-warm). Detach triggers: ``STOP``, ``ERROR`` containing
``daemon channel broken`` / ``fatal``, ``None`` from ``next()``, or ``tick`` +
``probe_heartbeat`` both silent > 1 s. ``ERROR`` containing ``Receiver timed
out`` is the normal idle return of ``next(timeout)`` in 1.0.1. Re-attach backs
off 1 -> 10 s and re-runs steps 3-5 (a dead control plane is reaped and
respawned). Every ``dora.rescan_s`` (and on ``request_join``) the registered
daemon set is re-read; a change for a configured remote machine re-renders the
YAML and restarts the dataflow (``dataflow_restarts += 1``).

The control thread never calls anything here (§2.5).
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apollo_mavis_v2_core.protocol import external as ext
from apollo_mavis_v2_core.protocol.external import DoraInfo, DoraMachineInfo, ExternalStatus

from ..config import DoraConfig
from . import codec
from .control_plane import (
    BindHostError,
    ControlPlaneError,
    DoraControlPlane,
    check_versions,
)
from .dataflow import placeholders_for, render_dataflow
from .stdout_guard import StdoutGuard

logger = logging.getLogger(__name__)

PULSE_SILENCE_S = 1.0  # tick (10 Hz) + probe_heartbeat (1 Hz) both silent -> detached
LOCK_FILE = "mavis_runtime.lock"
YAML_NAME = "mavis_v2.dora.yml"
EVENT_FIFO = 64
LOST_BACKOFF_S = 10.0  # next rescan after a failed `dora doctor` (coordinator busy)
PREWARM_KEY = "prewarm"
_FATAL_MARKERS = ("daemon channel broken", "fatal")
_IDLE_MARKERS = ("Receiver timed out",)
HZ_WINDOW_S = 2.0

Thunk = Callable[[], tuple[Any, dict[str, Any]]]  # -> (pa.Array, extra metadata)
InputHandler = Callable[[dict[str, Any]], None]  # receives the raw dora event


def _import_dora() -> Any:
    """The sanctioned lazy import; ``RUST_LOG=error`` must precede it (§8)."""
    os.environ.setdefault("RUST_LOG", "error")
    import dora  # noqa: TID251 - the ONLY dora import site of the runtime

    return dora


@dataclass
class _Slot:
    thunk: Thunk | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class _RateMeter:
    """Messages per second per topic over a sliding window (``publish_hz``)."""

    def __init__(self, window_s: float = HZ_WINDOW_S) -> None:
        self.window = window_s
        self._stamps: dict[str, deque[float]] = {}

    def mark(self, topic: str, now: float) -> None:
        d = self._stamps.setdefault(topic, deque())
        d.append(now)
        while d and now - d[0] > self.window:
            d.popleft()

    def rates(self, now: float) -> dict[str, float]:
        out: dict[str, float] = {}
        for topic, d in self._stamps.items():
            while d and now - d[0] > self.window:
                d.popleft()
            if len(d) >= 2:
                out[topic] = round((len(d) - 1) / max(1e-6, d[-1] - d[0]), 2)
            elif d:
                out[topic] = 0.0
        return out


class DoraBridge:
    """Process-lifetime owner of the ``mavis_runtime`` node (see module docstring).

    Test seams: ``control_plane_factory(cfg, var_dir, arm_ips) -> DoraControlPlane``-like,
    ``node_factory(node_id, daemon_port) -> Node``-like (``try_recv`` / ``next`` /
    ``send_output`` / ``node_config`` / ``dataflow_id``), ``versions_ok() -> str | None``.
    """

    def __init__(
        self,
        cfg: DoraConfig,
        *,
        epoch: str,
        arm_ips: tuple[str | None, ...] = (),
        session_id: Callable[[], str] = lambda: "",
        control_plane_factory: Callable[..., Any] | None = None,
        node_factory: Callable[..., Any] | None = None,
        versions_ok: Callable[[], str | None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        barrier_probe: Callable[[float], bool] | None = None,
    ) -> None:
        self.cfg = cfg
        self.epoch = epoch
        self.var_dir: Path = Path(cfg.var_dir)
        self.arm_ips = tuple(arm_ips)
        self._session_id = session_id
        self._cp_factory = control_plane_factory or (
            lambda c, d, ips: DoraControlPlane(c, d, tuple(ips))
        )
        self._node_factory = node_factory
        self._versions_ok = versions_ok  # None -> control_plane.check_versions at start()
        self._clock = clock
        # (timeout_s) -> True once dora's multi-machine start barrier is open; the default
        # attaches the lab-side `canary` dynamic node in a SUBPROCESS (nodes/canary.py)
        self._barrier_probe = barrier_probe or self._subprocess_canary

        self.state: ext.ExternalState = "disabled"
        self.detail: str = "dora.enabled is false"
        self.plane: Any = None
        self.node: Any = None
        self.reattach_count = 0
        self.dataflow_restarts = 0
        self.dropped_inputs = 0
        self.attached_since: float | None = None
        self.publish_seq: dict[str, int] = {}
        self.camera_outputs: list[str] = []  # cam_<id> outputs to pre-warm
        self.depth_outputs: list[str] = []
        self.mic_output: str | None = None
        self.camera_ids: list[str] = []  # publishable camera ids (config filter applied)
        self.yaml_path = self.var_dir / YAML_NAME
        self._registered: set[str] = set()  # daemons seen by the last rescan (configured ids)
        self._rendered_machines: tuple[str, ...] = ()  # machines whose placeholders are live
        # rendered machines whose daemon vanished: their placeholders STAY in the running
        # dataflow (the daemon drops frames to a dead peer; nothing stalls) until the next restart
        # for another reason - a stop/start right after the loss made the dora 1.0.1 coordinator
        # answer 429 to every CLI call for ~50 s (§16.1 "remote daemon loss")
        self._lost_machines: set[str] = set()
        self._machine_detail: dict[str, str] = {}  # join outcome per machine (REST detail)
        self._join_pending: deque[str] = deque()
        self._rescan_at = 0.0
        self._join_requested = threading.Event()
        # registered-daemon rescan (§2.2 step 6) runs `dora doctor` on ITS OWN thread: a
        # subprocess spawn stalls for tens of ms and the bus must never skip a camera frame
        self._rescan_thread: threading.Thread | None = None
        self._rescan_result: set[str] | None = None
        self._rescan_lock = threading.Lock()

        self._cv = threading.Condition()
        self._slots: dict[str, _Slot] = {}
        self._fifo: deque[tuple[str, Thunk, dict[str, Any]]] = deque(maxlen=EVENT_FIFO)
        self._handlers: dict[str, InputHandler] = {}
        self._closed_inputs: set[str] = set()
        self._on_state: list[Callable[[str], None]] = []
        self._last_pulse: float | None = None
        self._lock_fd: int | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._meter = _RateMeter()
        self._stdout = StdoutGuard(self.var_dir)
        # per-topic cost accounting (seconds spent building payloads / in send_output)
        self.cost: dict[str, list[float]] = {}  # topic -> [build_s, send_s, n, max_build, max_send]
        self.slot_overwrites: dict[str, int] = {}  # topic -> payloads superseded before a send
        self._client_seq: dict[str, int] = {}
        self.dora_mod: Any = None

    # -- wiring -----------------------------------------------------------------------------
    def set_outputs(
        self, camera_ids: list[str], depth_camera_ids: list[str], mic_id: str | None
    ) -> None:
        """Which optional outputs the rendered dataflow declares (before ``start``)."""
        pub = self.cfg.publish
        cams = (
            list(camera_ids)
            if pub.cameras == "all"
            else [c for c in camera_ids if c in pub.cameras]
        )
        self.camera_ids = cams
        self.camera_outputs = [ext.camera_output_id(c) for c in cams]
        self.depth_outputs = [ext.depth_output_id(c) for c in cams if c in depth_camera_ids]
        self.mic_output = ext.mic_output_id(mic_id) if (mic_id and pub.audio) else None

    def register_input(self, input_id: str, handler: InputHandler) -> None:
        self._handlers[input_id] = handler

    def on_state_change(self, cb: Callable[[str], None]) -> None:
        self._on_state.append(cb)

    # -- lifecycle -----------------------------------------------------------------------------
    def start(self) -> None:
        if not self.cfg.enabled:
            self._set_state("disabled", "dora.enabled is false")
            return
        why = (self._versions_ok or check_versions)()
        if why is not None:
            self._set_state("disabled", why)
            logger.warning("dora bridge disabled: %s", why)
            return
        try:
            self.var_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._set_state("disabled", f"cannot create dora var_dir {self.var_dir}: {exc}")
            return
        try:
            probe = self._cp_factory(self.cfg, self.var_dir, self.arm_ips)
            probe.resolve_bind_ip()
        except BindHostError as exc:
            self._set_state("disabled", str(exc))
            logger.warning("dora bridge disabled: %s", exc)
            return
        self.plane = probe
        self._set_state("unavailable", "starting control plane")
        self._running = True
        self._thread = threading.Thread(target=self._run, name="dora-bus", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        with self._cv:
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=30.0)
            self._thread = None

        self._detach_node(quiet=True)
        if self.plane is not None:
            try:
                self.plane.shutdown()
            except Exception:  # noqa: BLE001
                logger.exception("dora control plane shutdown failed")
        self._release_lock()
        self._set_state("closed", "runtime stopped")

    # -- producers (any thread) -------------------------------------------------------------------
    def try_publish(
        self, output_id: str, thunk: Thunk, metadata: dict[str, Any] | None = None
    ) -> bool:
        """Depth-1 slot per topic. ``thunk()`` runs on the bus thread and returns
        ``(pa.Array, extra_metadata)``. Returns False (no side effect) unless attached."""
        if self.state != "attached":
            return False
        with self._cv:
            slot = self._slots.get(output_id)
            if slot is None:
                slot = self._slots[output_id] = _Slot()
            if slot.thunk is not None:  # the previous payload was never sent: depth-1 drop
                self.slot_overwrites[output_id] = self.slot_overwrites.get(output_id, 0) + 1
            slot.thunk = thunk
            slot.meta = dict(metadata or {})
            self._cv.notify()
        return True

    def try_publish_now(
        self, output_id: str, array: Any, metadata: dict[str, Any] | None = None
    ) -> bool:
        """Convenience for ready payloads (small vectors / JSON)."""
        return self.try_publish(output_id, lambda: (array, {}), metadata)

    def publish_event(
        self, output_id: str, thunk: Thunk, metadata: dict[str, Any] | None = None
    ) -> bool:
        """Bounded FIFO (order kept) for ``events`` / ``policy_reset``."""
        if self.state != "attached":
            return False
        with self._cv:
            self._fifo.append((output_id, thunk, dict(metadata or {})))
            self._cv.notify()
        return True

    def request_join(self, machine_id: str) -> bool:
        """``POST /api/dora/machines/{id}/join`` (14-dora §16.1 join protocol): render this
        machine's placeholders, restart the dataflow and wait up to
        ``join_attach_timeout_s`` for the remote consumer to attach. Idempotent while the
        machine is already joined. False = not in ``dora.machines``."""
        if machine_id not in {m.id for m in self.cfg.machines}:
            return False
        if machine_id in self._rendered_machines and machine_id not in self._lost_machines:
            return True  # already live: nothing to restart
        with self._cv:
            if machine_id not in self._join_pending:
                self._join_pending.append(machine_id)
            self._cv.notify()
        return True

    def count_drop(self, why: str = "") -> None:
        self.dropped_inputs += 1
        if self.dropped_inputs <= 20 or self.dropped_inputs % 100 == 0:
            logger.warning("dora input dropped (%d): %s", self.dropped_inputs, why)

    # -- status ------------------------------------------------------------------------------------
    def status(self) -> ExternalStatus:
        now = self._clock()
        return ExternalStatus(
            enabled=self.cfg.enabled,
            state=self.state,
            detail=self.detail,
            node_id=self.cfg.node_id,
            dataflow_id=self.plane.dataflow_id if self.plane is not None else None,
            reattach_count=self.reattach_count,
            dataflow_restarts=self.dataflow_restarts,
            publish_hz=self._meter.rates(now) if self.state == "attached" else {},
            dropped_inputs=self.dropped_inputs,
        )

    def machines_info(self) -> list[DoraMachineInfo]:
        rendered = placeholders_for(self.cfg, self._rendered_machines)
        return [
            DoraMachineInfo(
                id=m.id,
                registered=m.id in self._registered,
                joined=m.id in self._rendered_machines and m.id not in self._lost_machines,
                placeholders=rendered.get(m.id, []),
                detail=self._machine_detail.get(m.id, ""),
            )
            for m in self.cfg.machines
        ]

    def info(self) -> DoraInfo:
        plane = self.plane
        bind_ip = (plane.bind_ip if plane is not None else None) or self.cfg.bind_host
        return DoraInfo(
            enabled=self.cfg.enabled,
            state=self.state,
            detail=self.detail,
            bind_host=bind_ip,
            machine_id=self.cfg.machine_id,
            auth=self.cfg.auth_effective,
            coordinator_addr=bind_ip,
            coordinator_port=self.cfg.coordinator_port,
            daemon_port=self.cfg.daemon_port,
            zenoh_port=self.cfg.zenoh_port,
            zenoh_connect=f"tcp/{bind_ip}:{self.cfg.zenoh_port}",
            dataflow_name=self.cfg.dataflow_name,
            dataflow_id=plane.dataflow_id if plane is not None else None,
            node_id=self.cfg.node_id,
            placeholders=list(ext.PLACEHOLDERS),
            machines=self.machines_info(),
            dataflow_restarts=self.dataflow_restarts,
            reattach_count=self.reattach_count,
            dataflow_yaml=str(self.yaml_path) if self.cfg.enabled else None,
        )

    @property
    def attached(self) -> bool:
        return self.state == "attached"

    # -- bus thread --------------------------------------------------------------------------------
    def _run(self) -> None:
        backoff = self.cfg.attach_retry_s[0]
        while self._running:
            if self.state != "attached":
                ok = self._attach()
                if not ok:
                    self._sleep(backoff)
                    backoff = min(backoff * 2.0, self.cfg.attach_retry_s[1])
                    continue
                backoff = self.cfg.attach_retry_s[0]
            try:
                self._pump()
            except Exception:  # noqa: BLE001 - the bus must never die
                logger.exception("dora bus iteration failed")
                self._detach("bus exception")
            if self.state == "detached":
                # §2.4: a detach waits the first backoff step before re-running steps 3-5
                # (consumers observe `detached`; a daemon that is dying gets a moment to die)
                self._sleep(backoff)

    def _sleep(self, seconds: float) -> None:
        with self._cv:
            if self._running:
                self._cv.wait(timeout=seconds)

    def _pump(self) -> None:
        """One attached iteration: drain outbound, drain inbound, watchdog, rescan, wait."""
        now = self._clock()
        self._drain_outbound(now)
        self._drain_inbound()
        if self.state != "attached":
            return
        now = self._clock()
        if self._last_pulse is not None and now - self._last_pulse > PULSE_SILENCE_S:
            self._detach(f"tick + probe_heartbeat silent for {now - self._last_pulse:.1f} s")
            return
        with self._cv:
            join = self._join_pending.popleft() if self._join_pending else None
        if join is not None:
            self._join(join)
            return  # state changed (detached -> re-attach, or attached again)
        if now >= self._rescan_at:
            self._rescan_at = now + self.cfg.rescan_s
            self._start_rescan()
        with self._rescan_lock:
            result, self._rescan_result = self._rescan_result, None
        if result is not None:
            self._apply_rescan(result)
            if self.state != "attached":
                return
        with self._cv:
            if not (self._fifo or any(s.thunk is not None for s in self._slots.values())):
                self._cv.wait(timeout=self.cfg.bus_poll_s)

    # -- attach / detach ---------------------------------------------------------------------------
    def _attach(self) -> bool:
        plane = self.plane
        try:
            if not plane.running:
                plane.reap()
                plane.start()
            running = plane.dataflow_running()
            lost = [m for m in self._rendered_machines if m in self._lost_machines]
            if running and lost:
                # re-attaching to a dataflow that still deploys placeholders on a vanished
                # machine is unverified against dora's start barrier; we are detached anyway,
                # so nothing local is lost by rendering it out first
                logger.info("dora re-attach: dropping the placeholders of vanished %s first", lost)
                plane.stop_dataflow()
                running = False
            if plane.dataflow_id is None or running is False:
                # a fresh dataflow never carries remote placeholders: dora 1.0.1 blocks every
                # node of a multi-machine dataflow until each remote dynamic placeholder is
                # attached (§16.1) - remotes re-join explicitly after any restart
                for mid in self._rendered_machines:
                    self._machine_detail[mid] = "re-join required: the dataflow was restarted"
                self._start_dataflow(rendered=())
            if not self._acquire_lock():
                self._set_state(
                    "unavailable", "another runtime holds the node id (mavis_runtime.lock)"
                )
                return False
            facts = plane.facts()
            os.environ.update(facts.node_env())
            if self.cfg.log.quiet_node_diagnostics:
                self._stdout.redirect()  # dora's stdout WARN flood -> node-stdout.log (§8)
            dora = self.dora_mod or _import_dora()
            self.dora_mod = dora
            factory = self._node_factory or (lambda nid, port: dora.Node(nid, daemon_port=port))
            t0 = self._clock()
            basic_config = logging.basicConfig
            self.node = factory(self.cfg.node_id, self.cfg.daemon_port)
            if logging.basicConfig is not basic_config:
                # dora 1.0.1 `Node()` execs a `<string>` wrapper into the stdlib `logging`
                # module that calls basicConfig with `handlers=[...]`; from then on ANY
                # `logging.basicConfig(stream=... | filename=...)` in this process raises
                # ValueError ("... should not be specified together with 'handlers'"). The
                # runtime's own entry point and any library configuring logging lazily would
                # hit it - put the stdlib function back (§16.2).
                logging.basicConfig = basic_config
                logger.info("dora Node() replaced logging.basicConfig; restored the stdlib one")
            problems = self._check_node_config()
            self._closed_inputs.clear()
            self._client_seq.clear()
            self._last_pulse = self._clock()
            self.attached_since = self._clock()
            self._rescan_at = self._clock() + self.cfg.rescan_s
            codec.warm_up()  # first-call pyarrow / numpy set-up (312 ms once) off the session
            self._prewarm()  # before consumers see "attached": the first SHM send is slow
            if self.node is None:  # a prewarm send failed -> _detach already ran
                return False
            self._set_state("attached", problems or "")
            logger.info(
                "dora node %s attached in %.0f ms (dataflow %s, daemon port %d)%s",
                self.cfg.node_id,
                (self._clock() - t0) * 1e3,
                plane.dataflow_id,
                self.cfg.daemon_port,
                f" - {problems}" if problems else "",
            )
            return True
        except (ControlPlaneError, BindHostError, OSError) as exc:
            self._set_state("unavailable", str(exc))
            logger.warning("dora bridge unavailable: %s", exc)
            return False
        except Exception as exc:  # noqa: BLE001 - dora raises bare exceptions
            self._set_state("unavailable", f"{type(exc).__name__}: {exc}")
            logger.warning("dora attach failed: %s: %s", type(exc).__name__, exc)
            self._detach_node(quiet=True)
            return False

    def _start_dataflow(self, *, rendered: tuple[str, ...]) -> None:
        plane = self.plane
        plane.resolve_bind_ip()  # DHCP: re-resolve before every (re)start
        configured = [m.id for m in self.cfg.machines]
        rendered = tuple(m for m in configured if m in rendered)
        text = render_dataflow(
            self.cfg,
            self.camera_ids,
            self._mic_id(),
            sys.executable,
            rendered,
            depth_camera_ids=[
                c for c in self.camera_ids if ext.depth_output_id(c) in self.depth_outputs
            ],
        )
        self.yaml_path.write_text(text, encoding="utf-8")
        plane.validate(self.yaml_path)
        plane.start_dataflow(self.yaml_path)
        for mid in self._lost_machines - set(rendered):
            self._machine_detail[mid] = (
                "placeholders dropped at a dataflow restart (its daemon had vanished) - "
                "POST join again when it is back"
            )
        self._lost_machines &= set(rendered)
        self._rendered_machines = rendered
        self.publish_seq.clear()
        logger.info(
            "dora dataflow %s started: %s (remote placeholders for %s)",
            self.cfg.dataflow_name,
            plane.dataflow_id,
            list(rendered) or "no machine",
        )

    def _mic_id(self) -> str | None:
        if self.mic_output is None:
            return None
        return self.mic_output[len(ext.MIC_OUTPUT_PREFIX) :]

    def _check_node_config(self) -> str:
        try:
            cfg = self.node.node_config()
        except Exception as exc:  # noqa: BLE001
            return f"node_config() unavailable: {exc}"
        inputs = cfg.get("inputs") if isinstance(cfg, dict) else None
        outputs = cfg.get("outputs") if isinstance(cfg, dict) else None
        problems = []
        if isinstance(inputs, dict) and set(inputs) != set(ext.RUNTIME_INPUTS):
            problems.append(f"inputs {sorted(inputs)} != {sorted(ext.RUNTIME_INPUTS)}")
        expected_out = (
            set(ext.RUNTIME_FIXED_OUTPUTS) | set(self.camera_outputs) | set(self.depth_outputs)
        )
        if self.mic_output:
            expected_out.add(self.mic_output)
        if isinstance(outputs, (list, set, tuple)) and not expected_out <= set(outputs):
            problems.append(f"outputs missing {sorted(expected_out - set(outputs))}")
        return "; ".join(problems)

    def _prewarm(self) -> None:
        """One dummy frame per camera output: the first SHM send costs 10-12 ms (§2.5)."""
        import numpy as np

        for oid in self.camera_outputs:
            arr, meta = codec.encode_rgb8(np.zeros((480, 640, 3), dtype=np.uint8))
            meta[PREWARM_KEY] = True
            self._send(oid, arr, meta, self._clock())
        for oid in self.depth_outputs:
            arr, meta = codec.encode_mono16(np.zeros((480, 640), dtype=np.uint16))
            meta[PREWARM_KEY] = True
            self._send(oid, arr, meta, self._clock())

    def _detach(self, why: str) -> None:
        if self.state == "attached":
            self.reattach_count += 1
            logger.warning("dora node detached: %s", why)
        self._detach_node(quiet=True)
        self._release_lock()
        self._set_state("detached", why)
        self._notify_closed_all()

    def _detach_node(self, *, quiet: bool) -> None:
        node, self.node = self.node, None
        self._stdout.restore()
        if node is not None:
            t0 = self._clock()
            try:
                del node
            except Exception:  # noqa: BLE001
                if not quiet:
                    raise
            logger.debug("dora node dropped in %.0f ms", (self._clock() - t0) * 1e3)
        with self._cv:
            for slot in self._slots.values():
                slot.thunk = None
            self._fifo.clear()

    def _notify_closed_all(self) -> None:
        for input_id, handler in self._handlers.items():
            try:
                handler({"type": "INPUT_CLOSED", "id": input_id})
            except Exception:  # noqa: BLE001
                logger.exception("dora input handler failed on detach notice")

    # -- outbound ----------------------------------------------------------------------------------
    def _drain_outbound(self, now: float) -> None:
        with self._cv:
            pending = [
                (oid, s.thunk, s.meta) for oid, s in self._slots.items() if s.thunk is not None
            ]
            for s in self._slots.values():
                s.thunk = None
            fifo = list(self._fifo)
            self._fifo.clear()
        for oid, thunk, meta in pending + fifo:
            t_build = time.perf_counter()
            try:
                arr, extra = thunk()
            except Exception:  # noqa: BLE001
                logger.exception("dora payload build failed for %s", oid)
                continue
            merged = dict(extra)
            merged.update(meta)
            t_send = time.perf_counter()
            self._send(oid, arr, merged, now)
            t_done = time.perf_counter()
            c = self.cost.get(oid)
            if c is None:
                c = self.cost[oid] = [0.0, 0.0, 0, 0.0, 0.0]  # + max build, max send
            c[0] += t_send - t_build
            c[1] += t_done - t_send
            c[2] += 1
            c[3] = max(c[3], t_send - t_build)
            c[4] = max(c[4], t_done - t_send)
            if self.state != "attached":
                return

    def _send(self, output_id: str, array: Any, meta: dict[str, Any], now: float) -> None:
        seq = self.publish_seq.get(output_id, 0) + 1
        self.publish_seq[output_id] = seq
        full = {
            ext.META_SCHEMA: ext.MAVIS_SCHEMA,
            ext.META_EPOCH: self.epoch,
            ext.META_SESSION_ID: self._session_id() or "",
            ext.META_SEQ: seq,
            ext.META_T_MONO: float(meta.pop(ext.META_T_MONO, now)),
            ext.META_WALLCLOCK_NS: int(meta.pop(ext.META_WALLCLOCK_NS, time.time_ns())),
            "send_t_mono": self._clock(),  # publish instant (t_mono may be the capture time)
        }
        full.update(codec.clean_metadata(meta))
        try:
            self.node.send_output(output_id, array, full)
        except Exception as exc:  # noqa: BLE001 - dora raises bare exceptions
            self._detach(f"send_output({output_id}) failed: {type(exc).__name__}: {exc}")
            return
        self._meter.mark(output_id, now)

    # -- inbound -----------------------------------------------------------------------------------
    def _drain_inbound(self) -> None:
        node = self.node
        for _ in range(256):  # bounded per pump
            if node is None or self.state != "attached":
                return
            try:
                ev = node.try_recv()
            except Exception as exc:  # noqa: BLE001
                self._detach(f"try_recv failed: {type(exc).__name__}: {exc}")
                return
            if ev is None:
                return
            self._handle_event(ev)

    def _handle_event(self, ev: dict[str, Any]) -> None:
        kind = ev.get("type")
        if kind == "INPUT":
            iid = str(ev.get("id"))
            if iid in (ext.IN_TICK, ext.IN_PROBE_HEARTBEAT):
                self._last_pulse = self._clock()
                return
            handler = self._handlers.get(iid)
            if handler is None:
                self.count_drop(f"unknown input {iid!r}")
                return
            if not self._common_inbound_ok(iid, ev):
                return
            try:
                handler(ev)
            except Exception:  # noqa: BLE001
                logger.exception("dora input handler %s failed", iid)
                self.count_drop(f"handler {iid} raised")
            return
        if kind == "INPUT_CLOSED":
            iid = str(ev.get("id"))
            self._closed_inputs.add(iid)
            handler = self._handlers.get(iid)
            if handler is not None:
                try:
                    handler(ev)
                except Exception:  # noqa: BLE001
                    logger.exception("dora INPUT_CLOSED handler %s failed", iid)
            return
        if kind == "STOP":
            self._detach(f"STOP ({ev.get('id') or ev.get('cause') or 'dataflow stopped'})")
            return
        if kind == "ERROR":
            text = str(ev.get("error") or ev.get("value") or "")
            if any(m in text for m in _IDLE_MARKERS):
                return
            if any(m in text for m in _FATAL_MARKERS):
                self._detach(f"ERROR: {text.splitlines()[0][:160]}")
                return
            logger.warning("dora ERROR event: %s", text[:200])
            return

    def _common_inbound_ok(self, iid: str, ev: dict[str, Any]) -> bool:
        """§3.2 checks every command input shares: schema major + per-client seq."""
        meta = ev.get("metadata") or {}
        schema = meta.get(ext.META_SCHEMA)
        if schema is not None:
            try:
                if int(schema) != ext.MAVIS_SCHEMA:
                    self.count_drop(f"{iid}: mavis_schema {schema} != {ext.MAVIS_SCHEMA}")
                    return False
            except (TypeError, ValueError):
                self.count_drop(f"{iid}: bad mavis_schema {schema!r}")
                return False
        client = str(meta.get(ext.META_CLIENT, ""))
        seq = meta.get(ext.META_SEQ)
        if seq is not None:
            try:
                seq_i = int(seq)
            except (TypeError, ValueError):
                self.count_drop(f"{iid}: bad seq {seq!r}")
                return False
            key = f"{iid}:{client}"
            last = self._client_seq.get(key)
            # seq 1 = a (re)started client: its counter begins again (a restarted policy node
            # keeps its `client` name); anything else must be strictly monotonic
            if last is not None and seq_i <= last and seq_i != 1:
                self.count_drop(f"{iid}: non-monotonic seq {seq_i} <= {last} from {client!r}")
                return False
            self._client_seq[key] = seq_i
        return True

    # -- rescan (§2.2 step 6) ----------------------------------------------------------------------
    def _start_rescan(self) -> None:
        """Kick `dora doctor` on the helper thread (skipped while one is still running)."""
        if not self.cfg.machines:
            return  # nothing to discover: no remote machines configured
        if self._rescan_thread is not None and self._rescan_thread.is_alive():
            logger.info("dora rescan skipped: the previous `dora doctor` is still running")
            return
        plane = self.plane

        def work() -> None:
            try:
                registered = plane.registered_machines()
            except Exception:  # noqa: BLE001
                logger.warning("dora rescan (`dora doctor`) raised", exc_info=True)
                registered = None
            if registered is None:
                logger.info(
                    "dora rescan: `dora doctor` failed or timed out (coordinator busy?); keeping "
                    "the last set and backing off"
                )
                self._rescan_at = max(self._rescan_at, self._clock() + LOST_BACKOFF_S)
            else:
                with self._rescan_lock:
                    self._rescan_result = set(registered)
                with self._cv:
                    self._cv.notify()

        self._rescan_thread = threading.Thread(target=work, name="dora-rescan", daemon=True)
        self._rescan_thread.start()

    def _apply_rescan(self, registered: set[str]) -> None:
        """Rescan result: ``registered`` is REST-visible; a JOINED machine whose daemon is
        gone is marked LOST (``joined`` false, detail) but its placeholders stay in the running
        dataflow: the lab daemon drops frames to a dead peer without stalling, whereas a
        stop/start right after the loss wedged the coordinator (429 for ~50 s, §16.1). They go
        at the next restart for any other reason (a join, a re-attach). Registration alone
        never adds placeholders - that is the explicit join (§16.1)."""
        configured = [m.id for m in self.cfg.machines]
        fresh = {m for m in registered if m in configured}
        if fresh != self._registered:
            logger.info(
                "dora registered remote daemons: %s -> %s", sorted(self._registered), sorted(fresh)
            )
        self._registered = fresh
        gone = tuple(
            m
            for m in self._rendered_machines
            if m not in self._registered and m not in self._lost_machines
        )
        for mid in gone:
            self._lost_machines.add(mid)
            self._machine_detail[mid] = (
                "daemon unregistered: its placeholders stay in the running dataflow (frames to "
                "it are dropped) until the next restart - POST join again when it is back"
            )
            logger.info("dora remote daemon %s vanished: marked lost, no restart", mid)

    def _join(self, machine_id: str) -> None:
        """Join protocol: restart with the machine's placeholders, then wait for dora's
        multi-machine start barrier (the ``probe`` node only gets spawned once every remote
        dynamic placeholder has been attached); expired -> restart without it again."""
        if machine_id in self._rendered_machines and machine_id not in self._lost_machines:
            return
        if machine_id not in self._registered:
            fresh = self.plane.registered_machines()
            if fresh is not None:
                self._registered = {m for m in fresh if m in {c.id for c in self.cfg.machines}}
            if machine_id not in self._registered:
                self._machine_detail[machine_id] = (
                    "join refused: no registered daemon for this machine id (start "
                    f"`dora daemon --machine-id {machine_id}` against this coordinator first)"
                )
                return
        wanted = tuple(sorted((set(self._rendered_machines) - self._lost_machines) | {machine_id}))
        self._lost_machines.discard(machine_id)  # the join re-deploys it on its NEW daemon
        self._machine_detail[machine_id] = (
            f"joining: attach viewer_{machine_id} / observer_{machine_id} within "
            f"{self.cfg.join_attach_timeout_s:.0f} s"
        )
        ok = self._restart_dataflow(wanted, barrier_timeout=self.cfg.join_attach_timeout_s)
        if ok:
            self._machine_detail[machine_id] = "joined"
        else:
            logger.warning("dora join of %s expired: publishing without it", machine_id)
            self._machine_detail[machine_id] = (
                "join timed out: no consumer attached within "
                f"{self.cfg.join_attach_timeout_s:.0f} s - start your viewer/observer node, "
                "then POST join again"
            )
            self._restart_dataflow(
                tuple(m for m in wanted if m != machine_id), barrier_timeout=None
            )

    def _restart_dataflow(
        self, rendered: tuple[str, ...], *, barrier_timeout: float | None
    ) -> bool:
        """Stop + re-render + start the dataflow with ``rendered`` remote machines, then
        re-attach. With remotes present wait for the start barrier first (``probe`` Running),
        up to ``barrier_timeout``; returns False when it expired (nothing attached yet)."""
        plane = self.plane
        self._detach_node(quiet=True)
        self._release_lock()
        self._notify_closed_all()
        try:
            plane.stop_dataflow()
            self._start_dataflow(rendered=rendered)
            self.dataflow_restarts += 1
        except (ControlPlaneError, BindHostError) as exc:
            self._set_state("detached", f"dataflow restart failed: {exc}")
            return False
        self._set_state("detached", "dataflow restarted")
        if rendered and barrier_timeout is not None:
            # NEVER attach mavis_runtime while the barrier may be closed: a blocked Node()
            # holds the GIL and freezes the whole runtime (§16.1). The canary subprocess pays
            # for that experiment instead.
            if not self._barrier_probe(barrier_timeout):
                return False  # barrier still closed: the remote never attached
        return self._attach()

    def _subprocess_canary(self, timeout_s: float) -> bool:
        """Attach ``canary`` (a lab-side dynamic node of the rendered dataflow) in a child
        process; it exits 0 the moment dora lets it through. Timeout -> killed -> False."""
        import subprocess

        facts = self.plane.facts()
        env = dict(os.environ)
        env.update(facts.node_env())
        env["RUST_LOG"] = "error"
        cmd = [
            sys.executable,
            "-m",
            "apollo_mavis_v2_runtime.dora_bridge.nodes.canary",
            "--daemon-port",
            str(self.cfg.daemon_port),
        ]
        t0 = time.monotonic()
        try:
            res = subprocess.run(
                cmd, capture_output=True, text=True, timeout=max(1.0, timeout_s), env=env
            )
        except subprocess.TimeoutExpired:
            logger.info("dora start barrier still closed after %.0f s (canary killed)", timeout_s)
            return False
        if res.returncode != 0:
            logger.warning(
                "dora barrier canary exited %d: %s", res.returncode, res.stderr.strip()[-300:]
            )
            return False
        logger.info("dora start barrier open after %.1f s (canary attached)", time.monotonic() - t0)
        return True

    # -- lock --------------------------------------------------------------------------------------
    def _acquire_lock(self) -> bool:
        if self._lock_fd is not None:
            return True
        import fcntl

        path = self.var_dir / LOCK_FILE
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()} {self.epoch}\n".encode())
        self._lock_fd = fd
        return True

    def _release_lock(self) -> None:
        fd, self._lock_fd = self._lock_fd, None
        if fd is not None:
            try:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(fd)

    # -- state -------------------------------------------------------------------------------------
    def _set_state(self, state: ext.ExternalState, detail: str) -> None:
        changed = state != self.state
        self.state = state
        self.detail = detail
        if changed:
            for cb in self._on_state:
                try:
                    cb(state)
                except Exception:  # noqa: BLE001
                    logger.exception("dora state callback failed")


__all__ = ["DoraBridge", "PREWARM_KEY", "PULSE_SILENCE_S", "LOCK_FILE", "YAML_NAME"]
