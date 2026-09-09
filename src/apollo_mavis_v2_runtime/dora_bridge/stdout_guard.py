"""Redirect the process's fd 1 while a dora node is alive (14-dora §8 "Log flood").

dora 1.0.1's node API prints JSON WARN diagnostics to **stdout** — several per
>= ~600 KB output (``dora-rs/dora#2742``: "zenoh direct path", "entering zenoh
put", "entering report_output_sent"), plus zenoh SHM watchdog priority warnings
and per-``queue_size`` discards. Measured 2026-09-07 on the lab host with two
640x480 cameras at 15 Hz: ``RUST_LOG=error`` (set before ``import dora``) does
NOT silence them — the node API's tracing subscriber emits WARN regardless — so
the bridge ``dup2``s fd 1 to ``<var_dir>/node-stdout.log`` for the node's
lifetime and restores it at detach. Python's own ``sys.stdout`` is re-opened on
a duplicate of the original fd first, so the runtime's prints still reach the
terminal; uvicorn / logging write to stderr and are unaffected.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

NODE_STDOUT_LOG = "node-stdout.log"
MAX_BYTES = 20 * 1024 * 1024  # rotate: rename to .1 when the file exceeds this


class StdoutGuard:
    """``redirect()`` / ``restore()`` pair; idempotent; never raises."""

    def __init__(self, var_dir: Path) -> None:
        self.path = Path(var_dir) / NODE_STDOUT_LOG
        self._saved_fd: int | None = None
        self._log_fd: int | None = None
        self._orig_stdout = None

    @property
    def active(self) -> bool:
        return self._saved_fd is not None

    def redirect(self) -> bool:
        if self.active:
            return True
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists() and self.path.stat().st_size > MAX_BYTES:
                self.path.replace(self.path.with_suffix(".log.1"))
            log_fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            saved = os.dup(1)
            try:
                sys.stdout.flush()
            except Exception:  # noqa: BLE001
                pass
            os.dup2(log_fd, 1)
            # Python-level prints keep going to the real terminal / launcher capture.
            self._orig_stdout = sys.stdout
            sys.stdout = os.fdopen(saved, "w", buffering=1, closefd=False)
            self._saved_fd = saved
            self._log_fd = log_fd
            return True
        except OSError as exc:
            logger.warning("node stdout redirect failed: %s", exc)
            return False

    def restore(self) -> None:
        if not self.active:
            return
        saved, self._saved_fd = self._saved_fd, None
        log_fd, self._log_fd = self._log_fd, None
        try:
            try:
                sys.stdout.flush()
            except Exception:  # noqa: BLE001
                pass
            os.dup2(saved, 1)
            if self._orig_stdout is not None:
                sys.stdout = self._orig_stdout
                self._orig_stdout = None
        finally:
            for fd in (saved, log_fd):
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass


__all__ = ["MAX_BYTES", "NODE_STDOUT_LOG", "StdoutGuard"]
