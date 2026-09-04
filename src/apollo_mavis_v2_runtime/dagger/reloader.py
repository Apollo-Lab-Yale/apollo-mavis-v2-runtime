"""PolicyReloaderImpl — the ONLY code that changes live policy weights (12-dagger §8).

1 Hz watcher thread reads ``LATEST``; a newer ``sanity_ok`` version is sha256-
verified + frame/space-matched, then staged (drain-latest: only the newest
staged version survives). ``maybe_swap`` applies staged weights strictly at
episode boundaries under the policy lock; ``rollback`` reloads
``LAST_KNOWN_GOOD`` immediately (mid-episode allowed); ``mark_good`` advances
the pointer after clean episodes.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from apollo_mavis_v2_core.dagger import CheckpointInfo, ControlMode

from .trainer.checkpoints import CheckpointStore

logger = logging.getLogger(__name__)


class PolicyReloaderImpl:
    """Implements the core ``PolicyReloader`` Protocol."""

    def __init__(
        self,
        policy,  # core Policy (+ set_version); load_weights under the lock
        store: CheckpointStore,
        session_action_frame: str,
        session_action_space: str,
        policy_lock: threading.Lock | None = None,
        current_version: int = 0,
        on_rollback: Callable[[int], None] | None = None,  # notify trainer
    ) -> None:
        self.policy = policy
        self.store = store
        self.session_action_frame = session_action_frame
        self.session_action_space = session_action_space
        self.lock = policy_lock or threading.Lock()
        self.current_version = int(current_version)
        self._on_rollback = on_rollback
        self._staged: CheckpointInfo | None = None
        self._staged_lock = threading.Lock()
        self.rejected: list[tuple[int, str]] = []  # (version, reason) telemetry log
        self._thread: threading.Thread | None = None
        self._running = False
        # LAST_KNOWN_GOOD always exists: initialize to the session seed (§12).
        if self.store.last_known_good() is None:
            self.store.set_last_known_good(self.current_version)

    # -- 1 Hz watcher ------------------------------------------------------------
    def start(self, period_s: float = 1.0) -> None:
        self._running = True

        def run() -> None:
            while self._running:
                try:
                    self.poll()
                except Exception:
                    logger.exception("reloader poll failed")
                time.sleep(period_s)

        self._thread = threading.Thread(target=run, name="policy-reloader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def poll(self) -> None:
        """Read LATEST; verify + stage a newer sanity_ok version."""
        latest = self.store.latest()
        if latest is None or latest <= self.current_version:
            return
        with self._staged_lock:
            if self._staged is not None and self._staged.version >= latest:
                return
        info = self.store.read_manifest(latest)
        if info is None:
            self._reject(latest, "unreadable manifest")
            return
        self.stage(info)

    # -- PolicyReloader Protocol ---------------------------------------------------
    def stage(self, ckpt: CheckpointInfo) -> None:
        if not ckpt.sanity_ok:
            self._reject(ckpt.version, "sanity_ok=false")
            return
        if ckpt.action_frame != self.session_action_frame:
            self._reject(ckpt.version, f"action_frame {ckpt.action_frame!r} != session")
            return
        if ckpt.action_space != self.session_action_space:
            self._reject(ckpt.version, f"action_space {ckpt.action_space!r} != session")
            return
        if not self.store.verify(ckpt):
            self._reject(ckpt.version, "sha256 mismatch")
            return
        with self._staged_lock:
            if self._staged is None or ckpt.version > self._staged.version:
                self._staged = ckpt  # drain-latest: stale staged versions drop

    def staged_version(self) -> int | None:
        with self._staged_lock:
            return self._staged.version if self._staged is not None else None

    def maybe_swap(self, at_episode_boundary: bool, current_mode: ControlMode) -> int | None:
        if not at_episode_boundary:
            return None
        with self._staged_lock:
            ckpt = self._staged
            if ckpt is None or ckpt.version <= self.current_version:
                self._staged = None
                return None
            self._staged = None
        if not self._load(ckpt.version, "swap"):
            return None
        return ckpt.version

    def rollback(self) -> int:
        """Load LAST_KNOWN_GOOD NOW (current weights are the emergency)."""
        target = self.store.last_known_good()
        if target is None:
            logger.error("rollback requested but LAST_KNOWN_GOOD missing")
            return self.current_version
        if target != self.current_version:
            self._load(target, "rollback")
        if self._on_rollback is not None:
            try:
                self._on_rollback(target)
            except Exception:
                logger.exception("trainer rollback notify failed")
        return self.current_version

    def mark_good(self) -> None:
        self.store.set_last_known_good(self.current_version)

    # -- internals -------------------------------------------------------------------
    def _load(self, version: int, why: str) -> bool:
        path = self.store.state_dict_path(version)
        try:
            with self.lock:
                self.policy.load_weights(str(path))  # scratch-copy inside impl
                if hasattr(self.policy, "set_version"):
                    self.policy.set_version(version)
        except Exception as e:
            self._reject(version, f"{why} load failed: {e!r}")
            return False
        self.current_version = version
        logger.info("policy %s -> v%06d", why, version)
        return True

    def _reject(self, version: int, reason: str) -> None:
        if not self.rejected or self.rejected[-1] != (version, reason):
            self.rejected.append((version, reason))
            logger.warning("checkpoint v%06d rejected: %s", version, reason)


__all__ = ["PolicyReloaderImpl"]
