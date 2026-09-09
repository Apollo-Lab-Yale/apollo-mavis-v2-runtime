"""``PolicySource`` — the one seam ``GatedPolicyExecutor`` talks to (14-dora §11.1).

Two implementations: the in-process :class:`~.policy_runner.PolicyRunner`
(torch policy on its own thread, GPU 0) and the runtime's
``dora_bridge.policy_source.ExternalPolicySource`` (fed by ``policy_action``
messages from the independent policy node). The executor never learns which
one it holds: it reads ``latest()`` / ``staleness_scale()`` / ``period`` and
drives ``drop_and_requery()`` / ``pause()`` / ``resume()``. This module is
dora-free on purpose (the dagger package must import without the extra).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from apollo_mavis_v2_core.interfaces.policy import PolicyOutput, PolicySpec


@runtime_checkable
class PolicySource(Protocol):
    """Latest-wins policy output for the 100 Hz mux (12-dagger §3/§6)."""

    period: float  # seconds between policy outputs (staleness + per-tick scaling)
    spec: PolicySpec  # layout the actions follow

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def latest(self) -> tuple[PolicyOutput | None, float]:
        """``(output, t_recv)``; ``t_recv`` is the runtime clock time the output became
        current (for chunked sources: the time the current row became active)."""
        ...

    def staleness_scale(self, now: float) -> float:
        """1.0 fresh; linear decay to 0 over 5 periods past ``period + 0.05 s``."""
        ...

    def drop_and_requery(self, reason: str = "handback") -> None:
        """Handback / episode boundary: forget pending output, fresh query. ``reason``
        is the ``PolicyResetReason`` an external source publishes (phase-14:
        ``"episode_boundary"`` at every boundary, ``"handback"`` inside an episode)."""
        ...

    def pause(self) -> None: ...

    def resume(self) -> None: ...

    @property
    def paused(self) -> bool: ...

    def version_label(self) -> str:
        """Telemetry string (``"{run_id}/v{n:06d}"`` / ``"{policy_id}/v{n:06d}"``)."""
        ...

    def current_version(self) -> int:
        """The version recorded into every DAgger frame (``policy_version`` column)."""
        ...


__all__ = ["PolicySource"]
