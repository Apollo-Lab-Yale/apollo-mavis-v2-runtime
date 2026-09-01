"""/ws/telemetry — 25 Hz broadcast, latest-wins per client (04-runtime §13.3)."""

from __future__ import annotations

import asyncio
import time

from apollo_xarm7_core import CollisionReport
from apollo_xarm7_core.protocol import (
    ArmTelemetry,
    ClearanceItem,
    PoseMsg,
    SessionTelemetry,
    TelemetryMsg,
)
from fastapi import WebSocket, WebSocketDisconnect


def build_telemetry(runtime, seq: int) -> TelemetryMsg:
    """One frame from the latest StateSnapshot + session manager state."""
    got = runtime.bus.snapshot.get()
    session = runtime.manager.session
    snap = got[0] if got is not None and session is not None else None
    arms: list[ArmTelemetry] = []
    if snap is not None:
        for arm_id, st in snap.arms.items():
            arms.append(
                ArmTelemetry(
                    arm_id=arm_id,
                    connected=True,
                    q=[float(x) for x in st.q[:7]],
                    rail_pos_m=st.rail_pos_m,
                    ee_pose=PoseMsg(
                        position=tuple(float(x) for x in st.ee_pose.position),
                        orientation=tuple(float(x) for x in st.ee_pose.orientation),
                    ),
                    gripper_open_frac=float(st.gripper.open_frac),
                    error_code=st.error_code,
                    warn_code=st.warn_code,
                    stale=st.stale,
                    goto=snap.plan_status.get(arm_id),
                )
            )
    return TelemetryMsg(
        seq=seq,
        ts=time.monotonic(),
        epoch=runtime.epoch,
        active_arm=snap.active_arm if snap is not None else None,
        controller_connected=runtime.controller_connected,
        arms=arms,
        collision=snap.gate if snap is not None else CollisionReport.ok(),
        clearances=[
            ClearanceItem(pair=pair, dist_m=dist)
            for pair, dist in (snap.clearances if snap is not None else [])
        ],
        episode=snap.episode if snap is not None else None,
        dagger=None,
        inference=None,
        session=SessionTelemetry(
            state=runtime.manager.state.value,
            start_from_progress=session.start_from_progress if session else None,
            plan_status=(
                snap.session_extra.get("plan_status") if snap is not None else None
            ),
        ),
    )


async def endpoint(websocket: WebSocket) -> None:
    runtime = websocket.app.state.runtime
    await websocket.accept()
    period = 1.0 / runtime.cfg.telemetry_hz
    seq = 0
    try:
        while True:
            t0 = time.monotonic()
            seq += 1
            # A slow consumer only delays its own next frame (latest-wins:
            # each frame is built fresh from the newest snapshot).
            await websocket.send_text(build_telemetry(runtime, seq).model_dump_json())
            await asyncio.sleep(max(0.0, period - (time.monotonic() - t0)))
    except WebSocketDisconnect:
        pass


__all__ = ["endpoint", "build_telemetry"]
