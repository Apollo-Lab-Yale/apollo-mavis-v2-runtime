"""Session state machine types (04-runtime §5).

``SessionSpec``/``SessionInfo`` are DEFINED in ``core.protocol.session`` —
runtime imports them and never redefines wire shapes.
"""

from __future__ import annotations

from enum import Enum

from apollo_xarm7_core.protocol import SessionInfo, SessionSpec  # re-export


class SessionState(str, Enum):
    IDLE = "idle"
    BRINGUP = "bringup"
    START_FROM = "start_from"
    RUNNING = "running"
    RECOVERING = "recovering"
    FAULT = "fault"
    TEARDOWN = "teardown"


__all__ = ["SessionState", "SessionSpec", "SessionInfo"]
