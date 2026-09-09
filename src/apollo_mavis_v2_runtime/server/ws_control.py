"""/ws/control — single-writer control channel (04-runtime §13.2).

First connection = controller; later ones are accepted read-only as
observers (never close-1008). Handlers only shuttle JSON and update shared
state — NO kinematics/motion work here (binding). Watchdog feeds on server
receive time; client ``ts`` is a latency metric only.
"""

from __future__ import annotations

import asyncio
import logging

from apollo_mavis_v2_core.protocol import (
    AckMsg,
    ActionMsg,
    HelloMsg,
    KeysMsg,
    parse_client_msg,
    validate_action_args,
)
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


async def endpoint(websocket: WebSocket) -> None:
    runtime = websocket.app.state.runtime
    await websocket.accept()
    is_controller = not runtime.controller_connected
    if is_controller:
        runtime.controller_connected = True
    role = "controller" if is_controller else "observer"
    session = runtime.manager.session
    await websocket.send_text(
        HelloMsg(
            epoch=runtime.epoch,
            session_id=session.session_id if session else None,
            role=role,
        ).model_dump_json()
    )
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = parse_client_msg(raw)
            except Exception:
                logger.warning("unparseable control message dropped: %.120s", raw)
                continue
            if isinstance(msg, KeysMsg):
                if is_controller:
                    runtime.on_keys(msg.seq, msg.held)  # stale seq dropped inside
                continue  # observer keys are silently ignored
            assert isinstance(msg, ActionMsg)
            await websocket.send_text(
                (await _handle_action(runtime, msg, is_controller)).model_dump_json()
            )
    except WebSocketDisconnect:
        pass
    finally:
        if is_controller:
            runtime.on_controller_disconnect()  # zero-twist path + drop held state


async def _handle_action(runtime, msg: ActionMsg, is_controller: bool) -> AckMsg:
    if not is_controller:
        return AckMsg(name=msg.name, ok=False, detail="observer")
    try:
        validate_action_args(msg)
    except Exception as e:
        return AckMsg(name=msg.name, ok=False, detail=f"invalid args: {e}")
    if runtime.manager.session is None:
        return AckMsg(name=msg.name, ok=False, detail="no session")
    future = runtime.submit_action(msg.name, msg.args)
    try:
        result = await asyncio.wait_for(asyncio.wrap_future(future), timeout=5.0)
    except asyncio.TimeoutError:
        future.cancel()
        return AckMsg(name=msg.name, ok=False, detail="timeout")
    return AckMsg(name=msg.name, ok=result.ok, detail=result.detail)


__all__ = ["endpoint"]
