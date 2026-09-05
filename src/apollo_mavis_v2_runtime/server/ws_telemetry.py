"""/ws/telemetry — 25 Hz broadcast, latest-wins per client (04-runtime §13.3)."""

from __future__ import annotations

import asyncio
import time

from apollo_mavis_v2_core import CollisionReport, Pose
from apollo_mavis_v2_core.protocol import (
    ArmTelemetry,
    ClearanceItem,
    ControllerTelemetry,
    HardwareMonitorTelemetry,
    MicrophoneTelemetry,
    PoseMsg,
    SessionTelemetry,
    TelemetryMsg,
    TrackerSettingsMsg,
    TrackerTelemetry,
)
from fastapi import WebSocket, WebSocketDisconnect

from ..control.tracker_teleop import align_pose
from ..devices.microphone import to_telemetry
from ..devices.tracker import ControllerState


def _controller_msg(state: ControllerState | None) -> ControllerTelemetry | None:
    if state is None:
        return None
    return ControllerTelemetry(
        trigger=state.trigger,
        trigger_pressed=state.trigger_pressed,
        trackpad_touch=state.trackpad_touch,
        trackpad_click=state.trackpad_click,
        trackpad_x=state.trackpad_x,
        trackpad_y=state.trackpad_y,
        grip=state.grip,
        menu=state.menu,
        system=state.system,
    )


def _pose_msg(pose: Pose | None) -> PoseMsg | None:
    if pose is None:
        return None
    return PoseMsg(
        position=tuple(float(x) for x in pose.position),
        orientation=tuple(float(x) for x in pose.orientation),
    )


def build_tracker_telemetry(runtime, snap, now: float) -> TrackerTelemetry:
    """Device fields (incl. the raw controller state and the device-held codes,
    13-tracker §1.1) from the Runtime-owned reader (pre-session too); clutch/
    anchor/target/filtered pose/last device action from
    ``session_extra["tracker"]`` (13-tracker §3.5/§4); settings incl. the live
    filter fields from the Runtime-owned ``TrackerSettings``; the calibration
    wizard snapshot from the Runtime-owned ``TrackerCalibration`` (phase-10)."""
    dev = runtime.tracker.status(now)
    settings = runtime.tracker_settings.get()
    extra = (snap.session_extra.get("tracker") if snap is not None else None) or {}
    return TrackerTelemetry(
        backend=dev.backend,
        status=dev.status,
        detail=dev.detail,
        object_name=dev.object_name,
        seq=dev.seq,
        rate_hz=dev.rate_hz,
        age_s=dev.age_s,
        pose_raw=_pose_msg(dev.pose_raw),
        pose_world=_pose_msg(
            align_pose(dev.pose_raw, settings.yaw_deg) if dev.pose_raw is not None else None
        ),
        pose_filtered=_pose_msg(extra.get("pose_filtered")),
        clutch=bool(extra.get("clutch", False)),
        engaged_arm=extra.get("engaged_arm"),
        anchor_tcp=_pose_msg(extra.get("anchor_tcp")),
        target_tcp=_pose_msg(extra.get("target_tcp")),
        settings=TrackerSettingsMsg(
            yaw_deg=settings.yaw_deg,
            pos_scale=settings.pos_scale,
            follow_rotation=settings.follow_rotation,
            filter_enabled=settings.filter_enabled,
            filter_min_cutoff_hz=settings.filter_min_cutoff_hz,
            filter_beta=settings.filter_beta,
        ),
        controller=_controller_msg(dev.controller),
        device_held=sorted(dev.device_held),
        device_action=extra.get("device_action"),
        charging=dev.charging,
        calibration=runtime.tracker_calibration.status(),
    )


def build_microphone_telemetry(runtime, now: float) -> MicrophoneTelemetry | None:
    """Microphone block (phase-11): one frame per tick from the Runtime-owned
    reader, pre-session too; ``None`` when no microphone is configured. The
    same device status feeds ``GET /api/microphones``."""
    if not runtime.cfg.microphone.enabled:
        return None
    return to_telemetry(runtime.microphone.status(now))


def build_hardware_monitor_telemetry(runtime) -> HardwareMonitorTelemetry:
    """``hardware_monitor`` block (phase-09a): the read-only monitor's per-arm
    status/sample rows + the twin overlays' per-stream rows, pre-session and
    session-less. Always present; ``enabled: false`` with empty ``arms`` when
    no hardware workcell is configured (or the hardware extra is missing)."""
    block = runtime.hardware_monitor.telemetry()
    overlay = getattr(runtime, "twin_overlay", None)
    if overlay is not None:
        block.overlays = overlay.telemetry()
    return block


def build_telemetry(runtime, seq: int) -> TelemetryMsg:
    """One frame from the latest StateSnapshot + session manager state."""
    got = runtime.bus.snapshot.get()
    session = runtime.manager.session
    snap = got[0] if got is not None and session is not None else None
    now = time.monotonic()
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
        ts=now,
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
        dagger=(dagger := snap.session_extra.get("dagger") if snap is not None else None),
        inference=snap.session_extra.get("inference") if snap is not None else None,
        session=SessionTelemetry(
            state=runtime.manager.state.value,
            start_from_progress=session.start_from_progress if session else None,
            plan_status=(
                snap.session_extra.get("plan_status") if snap is not None else None
            ),
            trainer_alive=(
                dagger.trainer.state != "dead"
                if dagger is not None and dagger.trainer is not None
                else None
            ),
        ),
        tracker=build_tracker_telemetry(runtime, snap, now),
        microphone=build_microphone_telemetry(runtime, now),
        hardware_monitor=build_hardware_monitor_telemetry(runtime),
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


__all__ = [
    "endpoint",
    "build_telemetry",
    "build_tracker_telemetry",
    "build_microphone_telemetry",
    "build_hardware_monitor_telemetry",
]
