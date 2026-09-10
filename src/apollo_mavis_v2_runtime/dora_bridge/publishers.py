"""Publishers feeding :class:`DoraBridge` (14-dora §2.5, §4).

- :class:`PoseStamper` - per-frame camera pose metadata (``q`` /
  ``tcp_pose_world`` / ``camera_pose_world`` / ``intrinsics`` / ``pose_source``)
  from a 32-entry ``(t_mono, q_by_arm)`` ring and its OWN ``RecorderKinematics``
  (one ``MjData`` per thread; ``stamp`` runs on ``dora-bus``, ``push`` on
  ``dora-publisher`` - a lock guards the ring only).
- :class:`CameraTap` - the ``EncoderWorker.taps`` callback: drops the frame
  REFERENCE into the topic slot; the Arrow copy + stamping happen on ``dora-bus``.
- :class:`MicTap` - the ``MicrophoneReader.taps`` callback -> ``mic_<id>``.
- :class:`SnapshotPublisher` - the ``dora-publisher`` thread: ``arm_state`` /
  ``arm_cmd`` (``state_hz``; idle snapshots at ``idle_state_hz``), ``obs_state``
  (``obs_hz``, ``observation_id`` monotonic), ``events`` (gate diffs, episode
  saves, version changes), ``session`` (change + 1 Hz), ``telemetry``
  (``telemetry_hz``) and ``heartbeat`` (1 Hz).

Nothing here runs on the control thread.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from apollo_mavis_v2_core import ArmState, Pose, se3
from apollo_mavis_v2_core.protocol import external as ext
from apollo_mavis_v2_core.protocol.external import (
    CameraAnnounce,
    EventEnvelope,
    SessionAnnounce,
)

from ..config import DoraConfig
from . import codec
from .bridge import DoraBridge

logger = logging.getLogger(__name__)

POSE_RING = 32  # (t_mono, q_by_arm) samples ~ 320 ms at 100 Hz
IDLE_TICK = -1
SESSION_REPEAT_S = 1.0
HEARTBEAT_S = 1.0
PUBLISHER_IDLE_WAIT_S = 0.1
QUEUED_EVENTS = 256  # enqueue_event bound (gate events of one session: a few per rollout)


def camera_arm(camera_id: str, arm_ids: Sequence[str]) -> str | None:
    """``grip_wrist`` / ``grip_wrist_cam`` -> ``grip`` (``None`` = static camera)."""
    for arm in arm_ids:
        if camera_id == arm or camera_id.startswith(f"{arm}_"):
            return arm
    return None


def _quat7(pose: Pose) -> list[float]:
    return [float(v) for v in pose.position] + [float(v) for v in pose.orientation]


# -- pose stamping ---------------------------------------------------------------------------------
class PoseStamper:
    """Camera pose metadata for wrist-camera frames, in and out of sessions (§4.2, §7).

    ``kin_factory(scene_id) -> RecorderKinematics``-like (``camera_world(cam, q_by_arm)``,
    ``tcp_base(arm, q)``, ``base_world(arm, q)``, ``camera_static(cam)``); built lazily
    per scene on the calling (bus) thread. ``mjcf_camera`` maps a published camera id to
    the MJCF camera name (hardware ``grip_wrist`` -> ``grip_wrist_cam``).
    """

    def __init__(
        self,
        kin_factory: Callable[[str], Any],
        *,
        intrinsics: Mapping[str, list[float]] | None = None,
        distortion: Mapping[str, list[float]] | None = None,
        mjcf_camera: Mapping[str, str] | None = None,
        arm_ids: Sequence[str] = (),
        image_pose: bool = True,
    ) -> None:
        self._kin_factory = kin_factory
        self._kin: Any = None
        self._scene_id: str | None = None
        self.intrinsics = dict(intrinsics or {})
        self.distortion = dict(distortion or {})
        self.mjcf_camera = dict(mjcf_camera or {})
        self.arm_ids = list(arm_ids)
        self.image_pose = image_pose
        self._cameras: set[str] = set()
        self._ring: deque[tuple[float, dict[str, np.ndarray], str]] = deque(maxlen=POSE_RING)
        self._lock = threading.Lock()
        # bus-thread cache: FK of the last used sample (several frames share one sample)
        self._fk_key: float | None = None
        self._fk_tcps: dict[str, Any] = {}
        self._fk_cams: dict[str, Any] = {}

    def set_scene(self, scene_id: str | None) -> None:
        """Switch the FK scene. The kinematics (a MuJoCo model compile, ~100 ms) are built
        HERE on the caller's thread so the bus thread never stalls its camera sends on a
        session boundary; the bus only ever uses the finished object."""
        with self._lock:
            if scene_id == self._scene_id:
                return
            kin = None
            if scene_id is not None:
                try:
                    kin = self._kin_factory(scene_id)
                except Exception:  # noqa: BLE001 - sim extra absent: frames go out without a pose
                    logger.exception("pose stamper: kinematics for %r unavailable", scene_id)
            self._scene_id = scene_id
            self._kin = kin
            self._ring.clear()
            self._fk_key = None

    def push(self, t_mono: float, q_by_arm: Mapping[str, np.ndarray], source: str) -> None:
        with self._lock:
            self._ring.append(
                (float(t_mono), {a: np.array(q) for a, q in q_by_arm.items()}, source)
            )

    def park(self) -> None:
        """Session end: re-tag the newest joint sample as ``idle`` at the current time so
        every frame captured from now on is stamped with the parked pose and
        ``pose_source: idle`` (the idle reader's own samples follow within one period)."""
        with self._lock:
            if not self._ring:
                return
            t, q_by_arm, _ = self._ring[-1]
            self._ring.append((time.monotonic(), q_by_arm, "idle"))

    def _nearest(self, t: float) -> tuple[float, dict[str, np.ndarray], str] | None:
        with self._lock:
            if not self._ring:
                return None
            return min(self._ring, key=lambda e: abs(e[0] - t))

    def _kinematics(self) -> Any:
        return self._kin

    def stamp(self, camera_id: str, frame_t_mono: float) -> dict[str, Any]:
        """Metadata for one frame of ``camera_id`` (may be partial when no FK is possible)."""
        arm = camera_arm(camera_id, self.arm_ids)
        meta: dict[str, Any] = {
            "camera_id": camera_id,
            "frame_ref": f"camera:{camera_id}",
            "mount": f"ee:{arm}" if arm else "world",
            "quat_order": "wxyz",
        }
        k = self.intrinsics.get(camera_id)
        if k is not None:
            meta["intrinsics"] = [float(v) for v in k]
            meta["distortion"] = [float(v) for v in self.distortion.get(camera_id, [])]
        if not self.image_pose:
            return meta
        sample = self._nearest(frame_t_mono)
        kin = self._kinematics()
        if sample is None or kin is None:
            return meta
        t, q_by_arm, source = sample
        meta["pose_t_mono"] = t
        meta["pose_source"] = source
        mj = self.mjcf_camera.get(camera_id, camera_id)
        tcps, cams = self._fk(kin, t, q_by_arm)
        if arm is not None and arm in q_by_arm:
            q = q_by_arm[arm]
            q8 = list(float(v) for v in q) + ([0.0] if len(q) < 8 else [])
            meta["q"] = q8[:8]
            if arm in tcps:
                meta["tcp_pose_world"] = _quat7(tcps[arm])
        if mj in cams:
            meta["camera_pose_world"] = _quat7(cams[mj])
        return meta

    def _fk(self, kin: Any, t: float, q_by_arm: Mapping[str, np.ndarray]) -> tuple[dict, dict]:
        """One forward pass per snapshot sample, shared by every camera frame stamped from it."""
        if self._fk_key == t:
            return self._fk_tcps, self._fk_cams
        cams = sorted(set(self.mjcf_camera.values()) | set(self.mjcf_camera) | set(self._cameras))
        try:
            if hasattr(kin, "world_poses"):
                tcps, cam_poses = kin.world_poses(dict(q_by_arm), cams)
            else:  # generic kinematics duck type (tests)
                tcps = {
                    a: se3.pose_mul(kin.base_world(a, q), kin.tcp_base(a, q))
                    for a, q in q_by_arm.items()
                }
                cam_poses = {}
                for c in cams:
                    try:
                        cam_poses[c] = kin.camera_world(c, dict(q_by_arm))
                    except Exception:  # noqa: BLE001
                        pass
        except Exception:  # noqa: BLE001 - never break a frame on FK trouble
            logger.exception("pose stamper FK failed")
            tcps, cam_poses = {}, {}
        self._fk_key, self._fk_tcps, self._fk_cams = t, tcps, cam_poses
        return tcps, cam_poses

    def note_camera(self, camera_id: str) -> None:
        """Register a published camera id (its MJCF pose is computed per sample)."""
        self._cameras.add(camera_id)


# -- taps ----------------------------------------------------------------------------------------
class CameraTap:
    """``EncoderWorker.taps`` callback for one camera stream (§2.5): reference only."""

    def __init__(
        self,
        bridge: DoraBridge,
        camera_id: str,
        stamper: PoseStamper,
        *,
        publish_depth: bool,
        seen: dict[str, int],
    ) -> None:
        self.bridge = bridge
        self.camera_id = camera_id
        self.output_id = ext.camera_output_id(camera_id)
        self.depth_id = ext.depth_output_id(camera_id) if publish_depth else None
        self.stamper = stamper
        self.stamper.note_camera(camera_id)
        self.seen = seen  # shared camera_id -> newest CameraFrame.seq (obs_state image_seq)
        self._first: float | None = None

    def __call__(self, frame: Any) -> None:
        self.seen[self.camera_id] = int(frame.seq)
        if not self.bridge.attached:
            return
        if self._first is None:
            self._first = time.monotonic()
            logger.info(
                "camera tap %s: first frame seq %d at t_mono %.3f (capture age %.0f ms)",
                self.camera_id,
                int(frame.seq),
                self._first,
                (self._first - frame.t_mono) * 1e3,
            )
        stamper, cam = self.stamper, self.camera_id

        def build_rgb() -> tuple[Any, dict[str, Any]]:
            arr, meta = codec.encode_rgb8(frame.rgb)
            meta.update(stamper.stamp(cam, frame.t_mono))
            meta["frame_seq"] = int(frame.seq)
            meta["frame_t_mono"] = float(frame.t_mono)
            meta["frame_wallclock_ns"] = int(frame.wallclock_ns)
            return arr, meta

        self.bridge.try_publish(self.output_id, build_rgb, {ext.META_T_MONO: frame.t_mono})
        if self.depth_id is not None and getattr(frame, "depth", None) is not None:

            def build_depth() -> tuple[Any, dict[str, Any]]:
                arr, meta = codec.encode_mono16(frame.depth, frame.depth_scale_m)
                meta.update(stamper.stamp(cam, frame.t_mono))
                meta["frame_seq"] = int(frame.seq)
                meta["frame_t_mono"] = float(frame.t_mono)
                meta["frame_wallclock_ns"] = int(frame.wallclock_ns)
                return arr, meta

            self.bridge.try_publish(self.depth_id, build_depth, {ext.META_T_MONO: frame.t_mono})


class MicTap:
    """``MicrophoneReader.taps`` callback -> ``mic_<id>`` ``Float32[N]`` (§4.1)."""

    def __init__(
        self, bridge: DoraBridge, mic_id: str, sample_rate: int, status: Callable[[], str]
    ) -> None:
        self.bridge = bridge
        self.output_id = ext.mic_output_id(mic_id)
        self.sample_rate = int(sample_rate)
        self._status = status

    def __call__(self, frame: Any) -> None:
        samples = getattr(frame, "samples", None)
        if samples is None or not self.bridge.attached:
            return
        sr = self.sample_rate

        def build() -> tuple[Any, dict[str, Any]]:
            n = int(len(samples))
            return codec.encode_f32(samples), {
                "sample_rate": sr,
                "channels": 1,
                "sample_type": "f32",
                "block_seq": int(frame.seq),
                "t_mono_first_sample": float(frame.rx_mono) - n / float(sr),
                "rms_dbfs": float(frame.rms_dbfs),
                "peak_dbfs": float(frame.peak_dbfs),
                "overruns": int(getattr(frame, "overruns", 0)),
                "status": str(self._status()),
            }

        # a bounded FIFO, not a depth-1 slot: mic blocks are a continuous stream whose
        # block_seq must stay gap-free across a GIL stall of the bus thread (~40 ms buys one
        # extra block); order is kept, the FIFO bound (64) is > 2 s of audio
        self.bridge.publish_event(self.output_id, build, {ext.META_T_MONO: frame.rx_mono})


# -- session facts -------------------------------------------------------------------------------
@dataclass
class SessionFacts:
    """What the publisher needs to know about the running session (set by the manager)."""

    session_id: str
    spec: Any  # SessionSpec
    kind: str
    scene_id: str | None
    arm_ids: list[str]
    has_rail: dict[str, bool]
    frames: dict[str, str]
    action_names: list[str]
    state_names: list[str]
    camera_ids: list[str]
    cameras: dict[str, CameraAnnounce]
    policy_source: str
    dataset_root: str | None = None
    run_id: str | None = None
    action_space: str = "delta_ee"
    converter: Any = None  # RecordingFrameConverter for obs_state ee dims
    engaged_arm: Callable[[], str | None] = lambda: None
    policy_version: Callable[[], int | None] = lambda: None
    gate_events: Callable[[], list] = list  # the supervisor's CollisionEvents (with `source`)
    online_dagger: Any = None  # OnlineDaggerAnnounce (phase-14; 15-online-dagger §6) or None


class SnapshotPublisher:
    """The ``dora-publisher`` thread (§2.5): decimates the snapshot slot into the bus."""

    def __init__(
        self,
        cfg: DoraConfig,
        bridge: DoraBridge,
        bus: Any,
        *,
        kin_factory: Callable[[str], Any],
        telemetry_json: Callable[[], str],
        session_state: Callable[[], str],
        epoch: str,
        telemetry_hz: float = 25.0,
        camera_seen: dict[str, int] | None = None,
        idle_arm_ids: Sequence[str] = (),
        idle_has_rail: Mapping[str, bool] | None = None,
        idle_scene: str | None = None,
        stamper: PoseStamper | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self.bridge = bridge
        self.bus = bus
        self._kin_factory = kin_factory
        self._telemetry_json = telemetry_json
        self._session_state = session_state
        self.epoch = epoch
        self.telemetry_hz = float(telemetry_hz)
        self.camera_seen = camera_seen if camera_seen is not None else {}
        self.idle_arm_ids = list(idle_arm_ids)
        self.idle_has_rail = dict(idle_has_rail or {})
        self.idle_scene = idle_scene
        self.stamper = stamper
        self._clock = clock
        self.facts: SessionFacts | None = None
        self._facts_lock = threading.Lock()
        self._kin: Any = None
        self._kin_scene: str | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._next: dict[str, float] = {}
        self._last_tick: int | None = None
        self._last_snap_put = -1.0
        self.observation_id = 0
        self._gate_events_seen = 0
        self._last_session_key: tuple | None = None
        self._last_version: int | None = None
        self.version_changes_mid_episode = 0
        self._recording = False
        self.events_published = 0
        self.observation_times: dict[int, float] = {}
        # events handed over from the control tick (15-online-dagger §3: every TakeoverGate
        # event of a plain external session): appended there, published HERE, in order
        self._queued: deque[tuple[str, dict[str, Any], str | None]] = deque(maxlen=QUEUED_EVENTS)

    # -- lifecycle -----------------------------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="dora-publisher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    # -- session hooks (manager thread) ------------------------------------------------------------
    def session_started(self, facts: SessionFacts) -> None:
        with self._facts_lock:
            self.facts = facts
            self.observation_id = 0
            self.observation_times.clear()  # ids restart per session: never age-check stale ones
            self._last_version = None
            self.version_changes_mid_episode = 0
            self._recording = False
            self._gate_events_seen = 0
        if self.stamper is not None:
            self.stamper.set_scene(facts.scene_id)
        self._force_session_announce()

    def session_ended(self) -> None:
        with self._facts_lock:
            self.facts = None
        if self.stamper is not None:
            self.stamper.set_scene(self.idle_scene)
            self.stamper.park()  # frames from now on: the parked pose, source "idle"
        self._force_session_announce()

    def _force_session_announce(self) -> None:
        self._last_session_key = None
        self._next["session"] = 0.0

    def publish_event(
        self, kind: str, payload: dict[str, Any], session_id: str | None = None
    ) -> None:
        if not self.cfg.publish.events:
            return
        now = self._clock()
        env = EventEnvelope(
            kind=kind,
            t_mono=now,
            wallclock_ns=time.time_ns(),
            session_id=session_id if session_id is not None else self._session_id(),
            epoch=self.epoch,
            payload=payload,
        )
        if self.bridge.publish_event(
            ext.OUT_EVENTS, lambda: (codec.encode_json(env), {"kind": kind})
        ):
            self.events_published += 1

    def enqueue_event(self, kind: str, payload: dict[str, Any]) -> None:
        """Queue an ``events`` publish from another thread (the control tick): one deque
        append; the publisher thread drains the queue in order on its next iteration
        (:meth:`_drain_queued`). The session id is captured now, so an event of a session
        that ends before the drain still spells its session."""
        self._queued.append((kind, payload, self._session_id() or None))

    def _drain_queued(self) -> None:
        while self._queued:
            try:
                kind, payload, session_id = self._queued.popleft()
            except IndexError:
                return
            self.publish_event(kind, payload, session_id=session_id)

    def _session_id(self) -> str:
        f = self.facts
        return f.session_id if f is not None else ""

    # -- thread ------------------------------------------------------------------------------------
    def _run(self) -> None:
        while self._running:
            try:
                wait_s = min(PUBLISHER_IDLE_WAIT_S, 0.5 / self.telemetry_hz)
                got = self.bus.snapshot.wait_fresh(wait_s)
                now = self._clock()
                if got is not None:
                    snap, put_mono = got
                    if put_mono != self._last_snap_put:
                        self._last_snap_put = put_mono
                        self._on_snapshot(snap, now)
                if self.bridge.attached:
                    self._drain_queued()
                self._periodic(now)
            except Exception:  # noqa: BLE001 - the publisher must never die
                logger.exception("dora publisher iteration failed")
                time.sleep(0.05)

    def _due(self, topic: str, period: float, now: float) -> bool:
        nxt = self._next.get(topic, 0.0)
        if now < nxt:
            return False
        # phase-lock to the period so decimation stays regular under jitter
        self._next[topic] = (nxt + period) if now - nxt < period else (now + period)
        return True

    def _kinematics(self, scene_id: str | None) -> Any:
        if scene_id is None:
            return None
        if self._kin is None or self._kin_scene != scene_id:
            self._kin = self._kin_factory(scene_id)
            self._kin_scene = scene_id
        return self._kin

    # -- snapshot handling -------------------------------------------------------------------------
    def _on_snapshot(self, snap: Any, now: float) -> None:
        idle = int(snap.tick) < 0
        facts = self.facts
        if not idle and facts is None:
            return  # a loop snapshot before the manager announced the session: skip
        if idle and facts is not None:
            return  # an idle snapshot leaking into a session window: ignore
        arm_ids = facts.arm_ids if facts is not None else self.idle_arm_ids
        arm_ids = [a for a in arm_ids if a in snap.arms]
        if not arm_ids:
            return
        source = "idle" if idle else "loop"
        q_by_arm = {a: np.asarray(snap.arms[a].q, dtype=np.float64) for a in arm_ids}
        if self.stamper is not None:
            self.stamper.push(float(snap.t_mono), q_by_arm, source)
        if not self.bridge.attached:
            return
        # idle snapshots are already paced at idle_state_hz by the IdleArmReader: publish
        # each one (decimating a 10 Hz stream by a 10 Hz gate drops ~10 % on jitter)
        if idle or self._due("arm_state", 1.0 / self.cfg.publish.state_hz, now):
            self._publish_arm_state(snap, arm_ids, source, now)
            if not idle:
                self._publish_arm_cmd(snap, arm_ids)
        if not idle:
            self._publish_obs_state(snap, arm_ids, now)
            self._gate_events(snap)
            self._version_tracking(snap)

    def _publish_arm_state(self, snap: Any, arm_ids: list[str], source: str, now: float) -> None:
        facts = self.facts
        scene = facts.scene_id if facts is not None else self.idle_scene
        kin = self._kinematics(scene)
        has_rail = facts.has_rail if facts is not None else self.idle_has_rail
        vec: list[float] = []
        stale: list[int] = []
        errs: list[int] = []
        warns: list[int] = []
        rails: list[int] = []
        arm_source: list[str] = []
        sources = snap.session_extra.get("arm_source") or {}
        for a in arm_ids:
            st: ArmState = snap.arms[a]
            vec.extend(_arm_block(st, kin, a, bool(has_rail.get(a, st.has_rail))))
            stale.append(int(bool(st.stale)))
            errs.append(int(st.error_code))
            warns.append(int(st.warn_code))
            rails.append(int(bool(st.has_rail)))
            arm_source.append(str(sources.get(a, "")) if source == "loop" else "")
        gate = snap.gate
        meta = {
            "arm_ids": list(arm_ids),
            "layout": list(ext.ARM_STATE_LAYOUT),
            "has_rail": rails,
            "error_code": errs,
            "warn_code": warns,
            "stale": stale,
            "arm_source": arm_source,
            "active_arm": (snap.active_arm or "") if source == "loop" else "",
            "gate_severity": str(gate.severity) if source == "loop" else "ok",
            "gate_blocked": bool(gate.blocked) if source == "loop" else False,
            "watchdog_tripped": bool(snap.watchdog_tripped) if source == "loop" else False,
            "tick": int(snap.tick) if source == "loop" else IDLE_TICK,
            "source": source,
            "quat_order": "wxyz",
            ext.META_T_MONO: float(snap.t_mono),
            ext.META_WALLCLOCK_NS: int(snap.wallclock_ns),
        }
        arr = codec.encode_f64(np.asarray(vec, dtype=np.float64))
        self.bridge.try_publish_now(ext.OUT_ARM_STATE, arr, meta)

    def _publish_arm_cmd(self, snap: Any, arm_ids: list[str]) -> None:
        parts: list[np.ndarray] = []
        dof: list[int] = []
        for a in arm_ids:
            q = snap.q_cmd.get(a)
            if q is None:
                q = snap.arms[a].q
            q = np.asarray(q, dtype=np.float64)
            parts.append(q)
            dof.append(int(q.shape[0]))
        arr = codec.encode_f64(np.concatenate(parts) if parts else np.zeros(0))
        self.bridge.try_publish_now(
            ext.OUT_ARM_CMD,
            arr,
            {
                "arm_ids": list(arm_ids),
                "dof": dof,
                "tick": int(snap.tick),
                ext.META_T_MONO: float(snap.t_mono),
                ext.META_WALLCLOCK_NS: int(snap.wallclock_ns),
            },
        )

    def _publish_obs_state(self, snap: Any, arm_ids: list[str], now: float) -> None:
        facts = self.facts
        if facts is None:
            return
        mode = facts.spec.mode
        if mode not in ("dagger", "inference") and not (
            mode == "collect" and self.cfg.publish.obs_in_collect
        ):
            return
        if not self._due("obs_state", 1.0 / self.cfg.publish.obs_hz, now):
            return
        kin = self._kinematics(facts.scene_id)
        if kin is None or facts.converter is None:
            return
        parts: list[float] = []
        for a in arm_ids:
            st: ArmState = snap.arms[a]
            q = np.asarray(st.q, dtype=np.float64)
            t_w_b = kin.base_world(a, q)
            ee = facts.converter.convert_pose(a, st.ee_pose, t_w_b)
            parts += [float(x) for x in q[:7]]
            parts.append(float(st.gripper.open_frac))
            if facts.has_rail.get(a, False):
                parts.append(float(q[7]))
            parts += [float(x) for x in ee.position]
            parts += [float(x) for x in ee.orientation]
        self.observation_id += 1
        oid = self.observation_id
        cams = [c for c in facts.camera_ids if c in self.camera_seen]
        meta = {
            "observation_id": oid,
            "tick": int(snap.tick),
            "state_names": list(facts.state_names),
            "arm_ids": list(arm_ids),
            "frames": [facts.frames.get(a, f"arm_base:{a}") for a in arm_ids],
            "has_rail": [int(bool(facts.has_rail.get(a, False))) for a in arm_ids],
            "image_camera_ids": cams,
            "image_seq": [int(self.camera_seen[c]) for c in cams],
            "engaged_arm": facts.engaged_arm() or "",
            "episode_state": str(snap.episode.state) if snap.episode is not None else "idle",
            "quat_order": "wxyz",
            ext.META_T_MONO: float(snap.t_mono),
            ext.META_WALLCLOCK_NS: int(snap.wallclock_ns),
        }
        self.bridge.try_publish_now(ext.OUT_OBS_STATE, codec.encode_f32(parts), meta)
        self.observation_times[oid] = float(snap.t_mono)
        if len(self.observation_times) > 4096:
            for k in sorted(self.observation_times)[:2048]:
                self.observation_times.pop(k, None)

    def observation_t_mono(self, observation_id: int) -> float | None:
        return self.observation_times.get(int(observation_id))

    def _gate_events(self, snap: Any) -> None:
        """Publish every NEW ``CollisionEvent`` the safety supervisor recorded (rising edges,
        pair-set changes, releases) as ``events`` ``kind: collision`` — the gate stamps
        ``source`` (teleop / policy / ...) on them, which the twin's report rows lack."""
        facts = self.facts
        if facts is None:
            return
        try:
            events = list(facts.gate_events())
        except Exception:  # noqa: BLE001
            return
        if len(events) < self._gate_events_seen:  # a new supervisor: start over
            self._gate_events_seen = 0
        for ev in events[self._gate_events_seen :]:
            payload = ev.model_dump(mode="json") if hasattr(ev, "model_dump") else dict(ev)
            self.publish_event("collision", payload)
        self._gate_events_seen = len(events)

    def _version_tracking(self, snap: Any) -> None:
        facts = self.facts
        if facts is None:
            return
        recording = snap.episode is not None and snap.episode.state == "recording"
        version = facts.policy_version()
        if version is not None and self._last_version is not None and version != self._last_version:
            if recording and self._recording:
                self.version_changes_mid_episode += 1
                self.publish_event(
                    "policy_version_changed",
                    {"from": self._last_version, "to": version, "mid_episode": True},
                )
            else:
                self.publish_event(
                    "policy_version_changed",
                    {"from": self._last_version, "to": version, "mid_episode": False},
                )
        if version is not None:
            self._last_version = version
        self._recording = recording

    # -- periodic (session / telemetry / heartbeat) ------------------------------------------------
    def _periodic(self, now: float) -> None:
        if not self.bridge.attached:
            return
        state = self._session_state()
        facts = self.facts
        key = (facts.session_id if facts is not None else None, state)
        if key != self._last_session_key or self._due("session", SESSION_REPEAT_S, now):
            self._last_session_key = key
            self._next["session"] = now + SESSION_REPEAT_S
            ann = self._announce(state)
            self.bridge.try_publish_now(
                ext.OUT_SESSION, codec.encode_json(ann), {"state": ann.state}
            )
        if self.cfg.publish.telemetry and self._due("telemetry", 1.0 / self.telemetry_hz, now):
            try:
                text = self._telemetry_json()
            except Exception:  # noqa: BLE001
                logger.exception("telemetry build for dora failed")
                text = None
            if text:
                self.bridge.try_publish_now(ext.OUT_TELEMETRY, codec.encode_json(text))
        if self._due("heartbeat", HEARTBEAT_S, now):
            seq = self.bridge.publish_seq.get(ext.OUT_HEARTBEAT, 0) + 1
            self.bridge.try_publish_now(
                ext.OUT_HEARTBEAT,
                codec.encode_i64(seq),
                {
                    "attached_since": float(self.bridge.attached_since or 0.0),
                    "dataflow_id": str(self.bridge.plane.dataflow_id or "")
                    if self.bridge.plane
                    else "",
                    "state": self.bridge.state,
                },
            )

    def _announce(self, state: str) -> SessionAnnounce:
        facts = self.facts
        if facts is None:
            # No session: still tell a late joiner the cell's layout (the delta_ee action /
            # state names every session over ALL configured arms uses), so a policy node can
            # declare a matching spec before the operator starts the session (§6.1 step 2).
            from ..recorder.features import arm_action_names, arm_state_names

            arms = list(self.idle_arm_ids)
            rail = {a: bool(self.idle_has_rail.get(a, False)) for a in arms}
            return SessionAnnounce(
                epoch=self.epoch,
                session_id=None,
                state="idle" if state in ("idle", "IDLE") else state,
                arm_ids=arms,
                has_rail=rail,
                frames={a: f"arm_base:{a}" for a in arms},  # the default recording frames
                action_space="delta_ee",
                action_names=[n for a in arms for n in arm_action_names(a, rail[a], "delta_ee")],
                state_names=[n for a in arms for n in arm_state_names(a, rail[a])],
                camera_ids=sorted(self.camera_seen) if self.camera_seen else [],
            )
        return SessionAnnounce(
            epoch=self.epoch,
            session_id=facts.session_id,
            state=state,
            spec=facts.spec,
            kind=facts.kind,  # type: ignore[arg-type]
            arm_ids=list(facts.arm_ids),
            has_rail=dict(facts.has_rail),
            frames=dict(facts.frames),
            action_space=facts.action_space,
            action_names=list(facts.action_names),
            state_names=list(facts.state_names),
            camera_ids=list(facts.camera_ids),
            cameras=dict(facts.cameras),
            policy_source=facts.policy_source,  # type: ignore[arg-type]
            dataset_root=facts.dataset_root,
            run_id=facts.run_id,
            online_dagger=facts.online_dagger,
        )


def _arm_block(st: ArmState, kin: Any, arm_id: str, has_rail: bool) -> list[float]:
    """The 32-value ``arm_state`` block of one arm (14-dora §4.2)."""
    q = np.asarray(st.q, dtype=np.float64)
    dq = np.asarray(st.dq, dtype=np.float64)
    rail = float(q[7]) if q.shape[0] > 7 else 0.0
    block = [float(v) for v in q[:7]] + [rail]
    block += [float(v) for v in dq[:7]] + [0.0]
    block += _quat7(st.ee_pose)
    if kin is not None:
        try:
            ee_w = se3.pose_mul(kin.base_world(arm_id, q), st.ee_pose)
            block += _quat7(ee_w)
        except Exception:  # noqa: BLE001 - arm not in this scene
            block += [math.nan] * 7
    else:
        block += [math.nan] * 7
    block.append(float(st.gripper.open_frac))
    block.append(float(st.rail_pos_m) if st.rail_pos_m is not None else math.nan)
    assert len(block) == ext.ARM_STATE_BLOCK
    return block


__all__ = [
    "CameraTap",
    "MicTap",
    "PoseStamper",
    "SessionFacts",
    "SnapshotPublisher",
    "camera_arm",
]
