"""Runtime-local typed errors (04-runtime; 11-safety §4)."""

from __future__ import annotations

from apollo_mavis_v2_core import ApolloError, ConfigError


class RuntimeError_(ApolloError):
    """Base for runtime-local errors."""


class SafetyConfigError(ConfigError):
    """Hardware session without a live twin-backed SafetyGate (11-safety §4)."""


class SessionError(RuntimeError_):
    """Session lifecycle conflict; maps to HTTP 409."""


class SessionNotFoundError(RuntimeError_):
    """No active session; maps to HTTP 404."""


class MaintenanceUnavailableError(RuntimeError_):
    """An arm maintenance op (phase-09b) cannot run on the current path: the
    monitor is off / paused / not connected, the op is not available inside
    or outside a hardware session, or another op is busy on that arm; maps to
    HTTP 409."""


class GelloUnavailableError(RuntimeError_):
    """A session-less GELLO op (``POST /api/gello/calibrate``, phase-15) cannot run:
    a session exists / is starting, the leader has no fresh valid sample, or the
    Manipulation Arm's current joints are unknown for the requested kind; maps to
    HTTP 409."""


__all__ = [
    "RuntimeError_",
    "SafetyConfigError",
    "SessionError",
    "SessionNotFoundError",
    "MaintenanceUnavailableError",
    "GelloUnavailableError",
]
