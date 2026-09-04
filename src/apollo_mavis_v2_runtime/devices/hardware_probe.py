"""Hardware reachability probe (phase-11; 04-runtime §13.1 ``reachable``).

``Runtime`` owns one :class:`HardwareProbe` when a ``hardware`` workcell is
configured. A daemon thread does a TCP connect-and-close on every configured
arm's ``ip:port`` (xArm control port 502 -- 30001-30003 are the report
streams and are never probed) every ``period_s`` and publishes a
``{arm_id: reachable}`` snapshot: ``open`` = control box up, ``refused`` =
host up but the control service not listening yet (box booting), ``unreachable``
= no route / timeout, ``unknown`` = not probed yet. Nothing is ever written on
the socket. ``hardware_ready`` is True iff every configured arm is ``open``.

The probe reuses ``apollo_mavis_v2_hardware.netsetup.probe.tcp_probe`` when the
``[hardware]`` extra is importable and falls back to an identical local socket
implementation otherwise. It is paused (snapshot frozen) while a hardware
session runs -- the driver's report stream is the truth then.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from ..config import HardwareProbeConfig

logger = logging.getLogger(__name__)

Reachable = Literal["open", "refused", "unreachable", "unknown"]
ProbeFn = Callable[[str, int, float], str]

STOP_JOIN_TIMEOUT_S = 5.0


def local_tcp_probe(ip: str, port: int, timeout: float) -> str:
    """Connect-and-close probe; writes nothing (mirror of hardware ``tcp_probe``)."""
    try:
        socket.create_connection((ip, port), timeout=timeout).close()
        return "open"
    except ConnectionRefusedError:
        return "refused"
    except OSError:
        return "unreachable"


def default_probe_fn() -> ProbeFn:
    """Hardware package's ``tcp_probe`` when importable, else :func:`local_tcp_probe`."""
    try:
        from apollo_mavis_v2_hardware.netsetup.probe import tcp_probe
    except Exception:  # noqa: BLE001 - [hardware] extra absent or broken
        return local_tcp_probe
    return tcp_probe


class HardwareProbe:
    """Periodic per-arm reachability snapshot (see module docstring).

    ``arms`` maps arm id -> ip; an empty map (no hardware workcell) or
    ``cfg.enabled == False`` makes :meth:`start` a no-op and every arm
    ``unknown``. ``paused()`` is polled before each round (True = skip).
    """

    def __init__(
        self,
        arms: dict[str, str | None],
        cfg: HardwareProbeConfig,
        *,
        paused: Callable[[], bool] | None = None,
        probe_fn: ProbeFn | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.arms: dict[str, str] = {a: ip for a, ip in arms.items() if ip}
        self.cfg = cfg
        self.paused = paused or (lambda: False)
        self._probe_fn = probe_fn
        self._clock = clock
        self._lock = threading.Lock()
        self._snapshot: dict[str, Reachable] = dict.fromkeys(self.arms, "unknown")
        self._updated_at: float | None = None
        self.rounds = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle -----------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return bool(self.arms) and self.cfg.enabled

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="hardware-probe", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = STOP_JOIN_TIMEOUT_S) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("hardware probe thread did not stop within %.1f s", timeout)
            else:
                self._thread = None

    # -- probing -----------------------------------------------------------------------------
    def probe_once(self) -> dict[str, Reachable]:
        """Probe every arm concurrently (each bounded by ``timeout_s``) and
        publish the snapshot; returns it."""
        if not self.arms:
            return {}
        fn = self._probe_fn or default_probe_fn()
        self._probe_fn = fn
        port, timeout = self.cfg.port, self.cfg.timeout_s

        def one(item: tuple[str, str]) -> tuple[str, Reachable]:
            arm_id, ip = item
            try:
                result = fn(ip, port, timeout)
            except Exception as e:  # noqa: BLE001 - a probe must never kill the thread
                logger.debug("probe %s (%s) failed: %r", arm_id, ip, e)
                result = "unreachable"
            if result not in ("open", "refused", "unreachable"):
                result = "unreachable"
            return arm_id, result  # type: ignore[return-value]

        with ThreadPoolExecutor(max_workers=len(self.arms)) as pool:
            results = dict(pool.map(one, list(self.arms.items())))
        with self._lock:
            self._snapshot = results
            self._updated_at = self._clock()
            self.rounds += 1
        return dict(results)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                if not self.paused():
                    self.probe_once()
            except Exception:  # last line of defence
                logger.exception("hardware probe round failed")
            self._stop.wait(self.cfg.period_s)

    # -- snapshot ----------------------------------------------------------------------------
    def snapshot(self) -> dict[str, Reachable]:
        with self._lock:
            return dict(self._snapshot)

    def reachable(self, arm_id: str) -> Reachable:
        with self._lock:
            return self._snapshot.get(arm_id, "unknown")

    @property
    def updated_at(self) -> float | None:
        with self._lock:
            return self._updated_at

    @property
    def hardware_ready(self) -> bool:
        """Every configured hardware arm is ``open`` (False with no arms)."""
        with self._lock:
            return bool(self._snapshot) and all(v == "open" for v in self._snapshot.values())


__all__ = ["HardwareProbe", "Reachable", "default_probe_fn", "local_tcp_probe"]
