"""ZMQ REP control endpoint for the trainer process (12-dagger §7).

Commands: status / submit_episode / train_now / rollback / stop. The REP
socket is polled with a timeout from the trainer main loop — the trainer is
single-threaded on purpose (bursts block replies; the client treats missed
replies as trainer-busy up to its crash threshold).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ControlEndpoint:
    """REP socket on tcp://127.0.0.1:{port}; poll() -> request dict | None."""

    def __init__(self, port: int) -> None:
        import zmq

        self._zmq = zmq
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.REP)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind(f"tcp://127.0.0.1:{port}")
        self._pending = False  # REP state: a recv'd request awaits its reply

    def poll(self, timeout_ms: int = 100) -> dict | None:
        if self._pending:
            raise RuntimeError("previous request not yet replied")
        if not self.sock.poll(timeout_ms, self._zmq.POLLIN):
            return None
        try:
            msg = self.sock.recv_json()
        except Exception:
            logger.exception("control recv failed")
            return None
        self._pending = True
        return msg if isinstance(msg, dict) else {"cmd": "invalid"}

    def reply(self, payload: dict) -> None:
        if not self._pending:
            return
        try:
            self.sock.send_json(payload)
        except Exception:
            logger.exception("control reply failed")
        finally:
            self._pending = False

    def close(self) -> None:
        self.sock.close()


__all__ = ["ControlEndpoint"]
