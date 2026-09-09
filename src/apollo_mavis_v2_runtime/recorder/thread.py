"""RecorderThread — the single owner of the episode being written (04-runtime §10.1).

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

Since 2026-09-07 (04-runtime §10.5/§10.6; 10-frames §11) the recorder is an
``EpisodeDirRecorder`` (one directory per saved episode; LeRobot v3 is an
export) and the thread also (a) mirrors the dataset manifest into
``EpisodeStatus`` (``repo_id`` / ``total_episodes`` / ``total_frames``), (b)
reports a ``returning`` state while the SessionManager drives the arms back to
the return profile after a save / discard (``episode_new`` is refused
meanwhile; the manager's hook ``on_episode_done`` starts that motion,
``set_returning`` bounds it) and (c) hands the microphone's
:class:`EpisodeAudioSink` to ``save`` so ``audio.wav`` lands INSIDE the episode
directory. The thread assembles the whole ``episode.json`` sidecar payload
(extrinsics, ``frames_dropped``, DAgger extras, the session's meta) BEFORE it
calls ``recorder.save(sidecar, audio)``, so the sidecar is inside the directory
at publication; deletion is immediate and lives in ``DatasetStore``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from apollo_mavis_v2_core import ArmState, Pose, se3
from apollo_mavis_v2_core.interfaces.recorder import EpisodeRecorder
from apollo_mavis_v2_core.protocol import ActionFilterConfig, EpisodeStatus

from .action_filter import ActionFilter
from .features import ArmMeta
from .frames import RecordingFrameConverter

if TYPE_CHECKING:
    from apollo_mavis_v2_core.interfaces.camera import CameraInterface

    from ..bus import RuntimeBus

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
        episode_meta_base: dict[str, Any] | None = None,
        extrinsics_fn: Callable[[dict[str, ArmState]], dict[str, Any]] | None = None,
        clock: Callable[[], float] = time.monotonic,
        audio: Any | None = None,  # EpisodeAudioSink (begin / finish(path, root) / abort)
        repo_id: str | None = None,
        dataset_root: Path | None = None,
        action_filter: ActionFilterConfig | None = None,  # idle-frame filter (04-runtime §10.5)
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
        self.episode_meta_base = dict(episode_meta_base or {})
        self.extrinsics_fn = extrinsics_fn
        self._clock = clock
        self.audio = audio
        self.repo_id = repo_id if repo_id is not None else getattr(recorder, "repo_id", None)
        self.dataset_root = Path(dataset_root) if dataset_root is not None else None
        # Idle-frame filter: decisions run `window` captures behind the live stream
        # (gripper look-ahead); kept captures feed the one-frame action lookahead below.
        self.filter = ActionFilter(action_filter, self.fps, [a.arm_id for a in self.arms])
        self._pending_prepare = False  # open the encoder on the recorder thread after N
        # SessionManager hook (recorder thread): ("saved" | "discarded", index | None)
        # after the writer op completed — the return-to-initial motion starts here.
        self.on_episode_done: Callable[[str, int | None], None] | None = None

        self._lock = threading.Lock()
        self._state: str = "idle"  # idle | recording | saving
        self._pending_op: str | None = None  # depth-1: "save" | "discard"
        self._returning = False  # arms driving back to the return profile
        self._detail = ""
        self._episode_index: int | None = None
        self._last_saved_index: int | None = None
        self._last_saved_id: str | None = None
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
                if self._returning:
                    return False, "returning to the initial configuration"
                try:
                    self.recorder.start({})  # mints the episode id; no filesystem op
                except Exception as e:
                    return False, f"start failed: {e!r}"
                self._episode_index = int(getattr(self.recorder, "episodes_saved", 0) or 0)
                self._frames = 0
                self._frames_dropped = 0
                self._pending = None
                self._need_extrinsics = True
                self._episode_extrinsics = {}
                self.filter.reset()
                self._pending_prepare = True
                self._state = "recording"
                self._detail = ""
                if self.audio is not None:
                    try:
                        self.audio.begin()
                    except Exception:  # noqa: BLE001 - audio never blocks recording
                        logger.exception("audio sink begin failed; episode records without audio")
                return True, f"episode {self._episode_index}"  # capture-order ordinal
            if op == "save":
                if self._state != "recording":
                    return False, self._state
                # a lone pending capture has no action yet; captures still inside the
                # filter's look-ahead buffer count (the save flushes them)
                if self._frames == 0 and self.filter.buffered + (self._pending is not None) < 2:
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
            state = self._state
            if state == "idle" and self._returning:
                state = "returning"
            return EpisodeStatus(
                state=state,  # type: ignore[arg-type]
                index=self._episode_index if self._state != "idle" else self._last_saved_index,
                frames=self._frames,
                duration_s=self._frames / self.fps,
                repo_id=self.repo_id,
                total_episodes=int(getattr(self.recorder, "episodes_saved", 0) or 0),
                total_frames=int(getattr(self.recorder, "total_frames", 0) or 0),
                detail=self._detail,
                frames_skipped=self.filter.frames_skipped,
            )

    @property
    def open_episode_id(self) -> str | None:
        """The id of the episode being recorded / saved (``DatasetStore`` marks it
        ``open`` and refuses to delete it); None when idle."""
        with self._lock:
            if self._state == "idle":
                return None
        return getattr(self.recorder, "episode_id", None)

    @property
    def last_saved_id(self) -> str | None:
        with self._lock:
            return self._last_saved_id

    # -- return-to-start (SessionManager; 04-runtime §10.5) --------------------------------
    def set_returning(self, active: bool, detail: str = "") -> None:
        """Flag the return motion: ``status().state`` reads ``returning`` while the
        recorder is idle and ``episode_new`` is refused. Any thread."""
        with self._lock:
            self._returning = bool(active)
            self._detail = detail

    @property
    def returning(self) -> bool:
        with self._lock:
            return self._returning

    # -- lifecycle ------------------------------------------------------------------
    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, name="recorder", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Teardown path: discard any open buffer, finalize exactly once — on the
        recorder thread's own ``finally`` when it is still alive (a long
        ``finish_episode``): running ``_shutdown`` concurrently would race the save
        and could publish a directory holding only ``episode.json``."""
        self._running = False
        thread = self._thread
        if thread is not None:
            thread.join(timeout=30.0)
            if thread.is_alive():
                logger.warning(
                    "recorder thread still busy after 30 s; leaving the shutdown to its finally"
                )
                return
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
        funnel here; discard-then-finalize, exactly once (04-runtime §10.4). An
        episode still open is discarded like an operator discard — the subclass hook
        runs with reason ``"session teardown"`` so an Online DAgger rollout is announced
        (``events.episode_discarded``, 15-online-dagger §3) before the coordinator closes."""
        with self._lock:
            if self._shutdown_done:
                return
            self._shutdown_done = True
            open_index = self._episode_index if self._state != "idle" else None
            was_open = self._state != "idle"
            self._state = "idle"
            self._pending_op = None
        episode_id = getattr(self.recorder, "episode_id", None) if was_open else None
        if self.audio is not None:
            try:
                self.audio.abort()
            except Exception:  # noqa: BLE001
                logger.exception("audio sink abort during shutdown failed")
        try:
            self.recorder.discard()
        except Exception:
            logger.exception("discard during recorder shutdown failed")
        if was_open:
            try:
                self._episode_discarded(open_index, episode_id, "session teardown")
            except Exception:
                logger.exception("episode-discarded hook failed during shutdown")
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
            prepare = self._pending_prepare and recording
            self._pending_prepare = False
        if prepare:
            # Open the temp dir + video encoder right after N, on THIS thread, before
            # any frame: the encoder's start stall lands here, not under a moving arm.
            prepare_fn = getattr(self.recorder, "prepare", None)  # optional on the protocol
            try:
                if prepare_fn is not None:
                    prepare_fn()
            except Exception:
                logger.exception("recorder prepare failed; the first frame will open lazily")
        if op == "save":
            self._do_save()
        elif op == "discard":
            self._do_discard()
        elif recording:
            self._capture(now)

    def _do_save(self) -> None:
        """Assemble the sidecar, then ``recorder.save(sidecar, audio)`` — one atomic
        publication (10-frames §11.6). A failed save keeps the buffer and is retried
        ONCE; a second failure degrades recording (04-runtime §15)."""
        # Flush the filter's look-ahead: the last captures are decided without a future.
        for kept in self.filter.flush():
            self._emit(kept)
        with self._lock:
            n_frames = self._frames
        if n_frames == 0:  # everything was filtered / a lone capture: nothing to save
            logger.info("episode save with no frames (all filtered): discarding instead")
            self._do_discard(reason="empty episode discarded")
            return
        payload = dict(self.episode_meta_base)
        payload["extrinsics"] = dict(self._episode_extrinsics)
        payload["frames_dropped"] = self._frames_dropped
        payload["filter"] = self.filter.summary()
        payload["success"] = None
        try:
            payload.update(self._sidecar_extra())
        except Exception:
            logger.exception("sidecar extras failed; saving without them")
        result: tuple[int, str] | None = None
        try:
            result = self.recorder.save(payload, self.audio)
        except Exception:
            logger.exception("episode save failed; buffer kept, retrying once (§15)")
            try:
                result = self.recorder.save(payload, self.audio)
            except Exception:
                logger.exception("episode save failed twice; recording degraded")
                self._degrade_after_failed_save()
                return
        index, episode_id = result
        try:
            self._episode_saved(index, episode_id)
        except Exception:
            logger.exception("episode-saved hook failed")
        # The manager's hook runs BEFORE the flip to idle so telemetry reads
        # saving -> returning -> idle, never a spurious idle frame in between
        # (set_returning takes the lock; it is not held here).
        self._notify_done("saved", index)
        with self._lock:
            self._pending_op = None
            self._state = "idle"
            self._last_saved_index = index
            self._last_saved_id = episode_id
            self._episode_index = None
            self._pending = None

    def _degrade_after_failed_save(self) -> None:
        """Second consecutive save failure (04-runtime §15): recording is OFF for the rest
        of the session; the episode buffer and its ``episodes/.tmp-<id>/`` directory are
        KEPT for recovery (never ``recorder.discard()`` here — §15 says so explicitly; the
        next open sweeps it). The episode still ends: the subclass hook fires with a
        reason, so an Online DAgger trainer that watched the rollout live is told
        (``events.episode_discarded``) that it will never be published as saved, and the
        executor's boundary reads the flip to idle as the discard it is. No return-to-start
        motion is started on this failure path."""
        with self._lock:
            index = self._episode_index
        episode_id = getattr(self.recorder, "episode_id", None)
        try:
            self._episode_discarded(
                index, episode_id, "save failed twice - recording degraded (buffer kept)"
            )
        except Exception:
            logger.exception("episode-discarded hook failed after the degraded save")
        if self.audio is not None:
            try:
                self.audio.abort()
            except Exception:  # noqa: BLE001
                logger.exception("audio sink abort after the degraded save failed")
        with self._lock:
            self.degraded = True
            self._pending_op = None
            self._state = "idle"
            self._detail = "recorder degraded (save failed twice)"

    def _notify_done(self, outcome: str, index: int | None) -> None:
        cb = self.on_episode_done
        if cb is None:
            return
        try:
            cb(outcome, index)
        except Exception:
            logger.exception("on_episode_done(%r) failed", outcome)

    # -- subclass hooks (DaggerRecorderThread, phase-08) ---------------------------
    def _sidecar_extra(self) -> dict[str, Any]:
        return {}

    def _episode_saved(self, index: int, episode_id: str) -> None:
        pass

    def _episode_discarded(self, index: int | None, episode_id: str | None, reason: str) -> None:
        """Mirror of :meth:`_episode_saved` for a discard (phase-14; 15-online-dagger §3):
        ``DaggerRecorderThread`` forwards it so ``events.episode_discarded`` is published.
        Runs on the recorder thread BEFORE the flip to idle; ``episode_id`` is the id the
        discarded episode had (None when the recorder never minted one)."""

    def _do_discard(self, reason: str = "") -> None:
        with self._lock:
            index = self._episode_index
        episode_id = getattr(self.recorder, "episode_id", None)  # before discard() clears it
        try:
            self.recorder.discard()  # cancels the encoder, removes the temp dir; free
        except Exception:
            logger.exception("episode discard failed")
        try:
            self._episode_discarded(index, episode_id, reason)
        except Exception:
            logger.exception("episode-discarded hook failed")
        if self.audio is not None:
            try:
                self.audio.abort()
            except Exception:  # noqa: BLE001
                logger.exception("audio sink abort failed")
        self.filter.reset()
        self._notify_done("discarded", None)  # before the flip to idle (see _do_save)
        with self._lock:
            self._pending_op = None
            self._state = "idle"
            self._episode_index = None
            self._frames = 0
            self._pending = None
            if reason:
                self._detail = reason

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
        self._last_snap_tick = snap.tick
        self._last_wallclock = wallclock
        for kept in self.filter.push(capture):  # idle frames never reach the writer
            self._emit(kept)

    def _emit(self, capture: _Capture) -> None:
        """A KEPT capture: the previous kept one gets its delta action and is written."""
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

    def _build_capture(self, snap: Any, images: dict[str, np.ndarray], wallclock: int) -> _Capture:
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
