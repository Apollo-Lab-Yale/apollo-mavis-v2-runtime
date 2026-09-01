"""RecorderThread — the single owner of the LeRobot writer (04-runtime §10.1).

Paced at ``RecorderConfig.fps`` (20-30 band) off the ``snapshot`` LatestSlot;
the 100 Hz control loop is never recorded and never back-pressured — episode
ops arrive through a depth-1 pending-op slot, frames flow one-way.

Delta actions: the recorded ``delta_ee`` action of dataset frame k is the
executed (post-gate) commanded-TCP increment from frame k to frame k+1 in
the arm's recording frame — so ``apply(action[k], obs[k].ee) ≈ obs[k+1].ee``
(10-frames §3.2). This needs one frame of lookahead: each capture is held
pending until the next one fixes its action ("executed action aligned with
obs"). Camera staleness (age > 2/fps) drops the dataset frame and counts
``frames_dropped`` — never a bad frame.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from apollo_xarm7_core import ArmState, Pose, se3
from apollo_xarm7_core.interfaces.recorder import EpisodeRecorder
from apollo_xarm7_core.protocol import EpisodeStatus

from .features import ArmMeta
from .frames import RecordingFrameConverter

if TYPE_CHECKING:
    from apollo_xarm7_core.interfaces.camera import CameraInterface

    from ..bus import RuntimeBus
    from .sidecars import SidecarWriter

logger = logging.getLogger(__name__)

ACTION_SOURCE_TELEOP = 1  # collect mode records teleop frames only (§7.3)


@dataclass(frozen=True)
class _Capture:
    """One sampled record tick, held until the next tick fixes its action."""

    wallclock_ns: int
    tick: int
    state_vec: np.ndarray  # float32, ee dims already in the recording frame
    images: dict[str, np.ndarray]
    cmd_pose_b: dict[str, Pose]  # commanded TCP in arm_base (post-gate FK)
    base_quat_w: dict[str, np.ndarray]
    rail_cmd: dict[str, float | None]
    grip_cmd: dict[str, float]


class RecorderThread:
    """Owns the ``EpisodeRecorder``; state machine idle -> recording -> saving."""

    def __init__(
        self,
        recorder: EpisodeRecorder,
        bus: RuntimeBus,
        cameras: dict[str, CameraInterface],
        arms: list[ArmMeta],
        converter: RecordingFrameConverter,
        kin: Any,  # base_world(arm, q) / tcp_base(arm, q) -> Pose
        fps: int,
        sidecars: SidecarWriter | None = None,
        episode_meta_base: dict[str, Any] | None = None,
        extrinsics_fn: Callable[[dict[str, ArmState]], dict[str, Any]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.recorder = recorder
        self.bus = bus
        self.cameras = dict(cameras)
        self.arms = list(arms)
        self.converter = converter
        self.kin = kin
        self.fps = int(fps)
        self.dt = 1.0 / self.fps
        self.max_frame_age_s = 2.0 / self.fps
        self.sidecars = sidecars
        self.episode_meta_base = dict(episode_meta_base or {})
        self.extrinsics_fn = extrinsics_fn
        self._clock = clock

        self._lock = threading.Lock()
        self._state: str = "idle"  # idle | recording | saving
        self._pending_op: str | None = None  # depth-1: "save" | "discard"
        self._episode_index: int | None = None
        self._last_saved_index: int | None = None
        self._frames = 0
        self._frames_dropped = 0
        self.frames_dropped_total = 0
        self.degraded = False

        self._pending: _Capture | None = None
        self._need_extrinsics = False
        self._episode_extrinsics: dict[str, Any] = {}
        self._last_snap_tick = -1
        self._last_wallclock = -1

        self._thread: threading.Thread | None = None
        self._running = False
        self._shutdown_done = False

    # -- episode ops (called from the control-loop thread; never writer ops) -------
    def request(self, op: str) -> tuple[bool, str]:
        """Validate + transition; writer work happens on the recorder thread."""
        with self._lock:
            if self.degraded:
                return False, "recorder degraded (save failed twice)"
            if self._pending_op is not None:
                return False, f"busy ({self._pending_op})"
            if op == "new":
                if self._state != "idle":
                    return False, self._state
                try:
                    self.recorder.start({})  # opens our buffer; no dataset op
                except Exception as e:
                    return False, f"start failed: {e!r}"
                self._episode_index = getattr(self.recorder, "episodes_saved", 0)
                self._frames = 0
                self._frames_dropped = 0
                self._pending = None
                self._need_extrinsics = True
                self._episode_extrinsics = {}
                self._state = "recording"
                return True, f"episode {self._episode_index}"
            if op == "save":
                if self._state != "recording":
                    return False, self._state
                if self._frames == 0:  # a lone pending capture has no action yet
                    return False, "empty episode"
                self._state = "saving"
                self._pending_op = "save"
                return True, "saving"
            if op == "discard":
                if self._state != "recording":
                    return False, self._state
                self._state = "saving"  # transient; buttons disabled during clear
                self._pending_op = "discard"
                return True, "discarding"
            return False, f"unknown episode op {op!r}"

    def status(self) -> EpisodeStatus:
        with self._lock:
            return EpisodeStatus(
                state=self._state,  # type: ignore[arg-type]
                index=self._episode_index if self._state != "idle" else self._last_saved_index,
                frames=self._frames,
                duration_s=self._frames / self.fps,
            )

    # -- lifecycle ------------------------------------------------------------------
    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, name="recorder", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Teardown path: discard any open buffer, finalize exactly once."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=30.0)
            self._thread = None
        self._shutdown()  # no-op if the run loop's finally already did it

    def _run(self) -> None:
        next_t = self._clock()
        try:
            while self._running:
                self.run_iteration(self._clock())
                next_t += self.dt
                lag = self._clock() - next_t
                if lag > self.dt:  # overrun: skip catch-up (no burst frames)
                    next_t = self._clock()
                elif lag < 0.0:
                    time.sleep(-lag)
        except Exception:
            logger.exception("recorder thread crashed; finalizing dataset")
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        """VideoEncodingManager-style guard: TEARDOWN / exception / stop all
        funnel here; discard-then-finalize, exactly once (04-runtime §10.4)."""
        with self._lock:
            if self._shutdown_done:
                return
            self._shutdown_done = True
            self._state = "idle"
            self._pending_op = None
        try:
            self.recorder.discard()
        except Exception:
            logger.exception("discard during recorder shutdown failed")
        for attempt in (1, 2):  # finalize failures are retried once (§10.4)
            try:
                self.recorder.finalize()
                break
            except Exception:
                logger.exception("recorder finalize failed (attempt %d)", attempt)

    # -- one paced iteration (public for deterministic tests) -------------------------
    def run_iteration(self, now: float) -> None:
        with self._lock:
            op = self._pending_op
            recording = self._state == "recording"
        if op == "save":
            self._do_save()
        elif op == "discard":
            self._do_discard()
        elif recording:
            self._capture(now)

    def _do_save(self) -> None:
        index: int | None = None
        try:
            index = self.recorder.save()
        except Exception:
            logger.exception("save_episode failed; buffer kept, retrying once (§15)")
            try:
                index = self.recorder.save()
            except Exception:
                logger.exception("save_episode failed twice; recording degraded")
                with self._lock:
                    self.degraded = True
                    self._pending_op = None
                    self._state = "idle"
                return
        if self.sidecars is not None:
            try:
                payload = dict(self.episode_meta_base)
                payload["extrinsics"] = dict(self._episode_extrinsics)
                payload["frames_dropped"] = self._frames_dropped
                payload["success"] = None
                payload.update(self._sidecar_extra())
                self.sidecars.write_episode(index, payload)
            except Exception:
                logger.exception("episode sidecar write failed")
        with self._lock:
            self._pending_op = None
            self._state = "idle"
            self._last_saved_index = index
            self._episode_index = None
            self._pending = None
        try:
            self._episode_saved(index)
        except Exception:
            logger.exception("episode-saved hook failed")

    # -- subclass hooks (DaggerRecorderThread, phase-08) ---------------------------
    def _sidecar_extra(self) -> dict[str, Any]:
        return {}

    def _episode_saved(self, index: int) -> None:
        pass

    def _do_discard(self) -> None:
        try:
            self.recorder.discard()  # cancels the streaming encoder; free
        except Exception:
            logger.exception("clear_episode_buffer failed")
        with self._lock:
            self._pending_op = None
            self._state = "idle"
            self._episode_index = None
            self._frames = 0
            self._pending = None

    # -- frame capture -----------------------------------------------------------
    def _capture(self, now: float) -> None:
        got = self.bus.snapshot.get()
        if got is None:
            return
        snap = got[0]
        if snap.tick == self._last_snap_tick:
            return  # control loop has not ticked; never record duplicates
        wallclock = max(st.wallclock_ns for st in snap.arms.values())
        if wallclock <= self._last_wallclock:
            return  # stale driver reports; keep wallclock_ns strictly increasing
        images: dict[str, np.ndarray] = {}
        for cam_id, cam in self.cameras.items():
            frame = cam.latest()
            if frame is None or (now - frame.t_mono) > self.max_frame_age_s:
                with self._lock:
                    self._frames_dropped += 1
                    self.frames_dropped_total += 1
                return  # drop the whole dataset frame; never write a bad one
            images[cam_id] = frame.rgb
        if self._need_extrinsics and self.extrinsics_fn is not None:
            try:
                self._episode_extrinsics = self.extrinsics_fn(snap.arms)
            except Exception:
                logger.exception("extrinsics snapshot failed")
                self._episode_extrinsics = {}
            self._need_extrinsics = False

        capture = self._build_capture(snap, images, wallclock)
        if self._pending is not None:
            try:
                self.recorder.add_frame(self._frame_from(self._pending, capture))
                with self._lock:
                    self._frames += 1
            except Exception:
                logger.exception("add_frame failed; frame dropped")
                with self._lock:
                    self._frames_dropped += 1
                    self.frames_dropped_total += 1
        self._pending = capture
        self._last_snap_tick = snap.tick
        self._last_wallclock = wallclock

    def _build_capture(
        self, snap: Any, images: dict[str, np.ndarray], wallclock: int
    ) -> _Capture:
        state_parts: list[float] = []
        cmd_pose_b: dict[str, Pose] = {}
        base_quat_w: dict[str, np.ndarray] = {}
        rail_cmd: dict[str, float | None] = {}
        grip_cmd: dict[str, float] = {}
        for arm in self.arms:
            st: ArmState = snap.arms[arm.arm_id]
            q_meas = np.asarray(st.q, dtype=np.float64)
            t_w_b = self.kin.base_world(arm.arm_id, q_meas)
            ee_rec = self.converter.convert_pose(arm.arm_id, st.ee_pose, t_w_b)
            state_parts += [float(x) for x in q_meas[:7]]
            state_parts.append(float(st.gripper.open_frac))
            if arm.has_rail:
                state_parts.append(float(q_meas[7]))
            state_parts += [float(x) for x in ee_rec.position]
            state_parts += [float(x) for x in ee_rec.orientation]
            q_cmd = np.asarray(snap.q_cmd[arm.arm_id], dtype=np.float64)
            cmd_pose_b[arm.arm_id] = self.kin.tcp_base(arm.arm_id, q_cmd)
            base_quat_w[arm.arm_id] = t_w_b.orientation
            rail_cmd[arm.arm_id] = float(q_cmd[7]) if arm.has_rail else None
            grip = snap.gripper_frac.get(arm.arm_id)
            grip_cmd[arm.arm_id] = float(grip if grip is not None else st.gripper.open_frac)
        return _Capture(
            wallclock_ns=wallclock,
            tick=snap.tick,
            state_vec=np.asarray(state_parts, dtype=np.float32),
            images=images,
            cmd_pose_b=cmd_pose_b,
            base_quat_w=base_quat_w,
            rail_cmd=rail_cmd,
            grip_cmd=grip_cmd,
        )

    def _frame_from(self, prev: _Capture, cur: _Capture) -> dict[str, Any]:
        """Dataset frame: obs at ``prev``, delta_ee action prev -> cur (§3.2)."""
        action: list[float] = []
        for arm in self.arms:
            a, b = prev.cmd_pose_b[arm.arm_id], cur.cmd_pose_b[arm.arm_id]
            dp_b = b.position - a.position
            dr_b = se3.quat_to_rotvec(se3.quat_mul(b.orientation, se3.quat_conj(a.orientation)))
            dp_f, dr_f = self.converter.convert_delta(
                arm.arm_id, dp_b, dr_b, prev.base_quat_w[arm.arm_id]
            )
            action += [float(x) for x in dp_f]
            action += [float(x) for x in dr_f]
            action.append(prev.grip_cmd[arm.arm_id])  # absolute open-frac target
            if arm.has_rail:
                action.append(float(cur.rail_cmd[arm.arm_id] - prev.rail_cmd[arm.arm_id]))
        frame: dict[str, Any] = {
            "action": np.asarray(action, dtype=np.float32),
            "observation.state": prev.state_vec,
            "intervention": np.array([False]),
            "action_source": np.array([ACTION_SOURCE_TELEOP], dtype=np.int8),
            "wallclock_ns": np.array([prev.wallclock_ns], dtype=np.int64),
        }
        for cam_id, img in prev.images.items():
            frame[f"observation.images.{cam_id}"] = img
        return frame


__all__ = ["RecorderThread", "ACTION_SOURCE_TELEOP"]
