"""Vive-tracker reader (13-tracker §2/§4): libsurvive poses -> ``LatestSlot``.

A daemon thread around libsurvive's blocking event API publishes
:class:`TrackerSample` (meters + **wxyz**, same as core ``Pose``) stamped with
``time.monotonic()`` on receipt. Only objects of type OBJECT whose codename
equals ``object_name`` are used; lighthouses are ignored. Backends:
``libsurvive`` (real), ``fake`` (scripted 0.15 m circle at 100 Hz) and
``none``. ``pysurvive`` is imported lazily and ONLY here (ruff banned-api
elsewhere); a missing module means status ``no_backend``, never a crash.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from apollo_xarm7_core import LatestSlot, Pose

from ..config import TrackerConfig

logger = logging.getLogger(__name__)

TrackerStatus = Literal["no_backend", "starting", "searching", "tracking", "stale", "error"]

FAKE_RATE_HZ = 100.0
FAKE_RADIUS_M = 0.15
FAKE_PERIOD_S = 20.0  # one slow lap; ~0.047 m/s tangential
FAKE_CENTER = np.array([0.5, 0.0, 0.3])
LIBSURVIVE_NO_OBJECT_GRACE_S = 3.0  # init ok but zero objects -> not openable / not paired


@dataclass(frozen=True)
class TrackerSample:
    """One tracker pose in the lighthouse world (13-tracker §4)."""

    pose: Pose  # m + wxyz
    vel_lin: np.ndarray  # m/s, world
    vel_ang: np.ndarray  # rad/s axis-angle, world
    t_dev: float  # libsurvive run time, s (NOT wall clock)
    rx_mono: float  # time.monotonic() on receipt (staleness feed)
    seq: int
    valid: bool = True  # False: jump > max_jump_m vs the previous sample


@dataclass(frozen=True)
class TrackerSettingsValues:
    """Immutable snapshot of the live tracker teleop settings."""

    yaw_deg: float
    pos_scale: float
    follow_rotation: bool


class TrackerSettings:
    """Live, thread-safe yaw/scale/rotation settings (process lifetime).

    ``get`` returns an immutable snapshot; ``update`` applies the non-None
    fields (``tracker_settings`` action semantics: omitted = unchanged).
    """

    def __init__(
        self, yaw_deg: float = 0.0, pos_scale: float = 1.0, follow_rotation: bool = True
    ) -> None:
        self._lock = threading.Lock()
        self._values = TrackerSettingsValues(
            float(yaw_deg), float(pos_scale), bool(follow_rotation)
        )

    @classmethod
    def from_config(cls, cfg: TrackerConfig) -> TrackerSettings:
        return cls(cfg.yaw_deg, cfg.pos_scale, cfg.follow_rotation)

    def get(self) -> TrackerSettingsValues:
        with self._lock:
            return self._values

    def update(
        self,
        *,
        yaw_deg: float | None = None,
        pos_scale: float | None = None,
        follow_rotation: bool | None = None,
    ) -> TrackerSettingsValues:
        with self._lock:
            v = self._values
            if yaw_deg is not None:
                v = replace(v, yaw_deg=float(yaw_deg))
            if pos_scale is not None:
                v = replace(v, pos_scale=float(pos_scale))
            if follow_rotation is not None:
                v = replace(v, follow_rotation=bool(follow_rotation))
            self._values = v
            return v


@dataclass(frozen=True)
class TrackerDeviceStatus:
    """Device-side telemetry fields (available without a session)."""

    backend: Literal["libsurvive", "fake", "none"]
    status: TrackerStatus
    detail: str
    object_name: str
    seq: int
    rate_hz: float
    age_s: float | None
    pose_raw: Pose | None


class TrackerReader:
    """Owns the device thread for the process lifetime; publishes to ``slot``.

    ``start()`` is a no-op for backend ``none``. ``status(now)`` derives
    ``stale`` from the newest sample's age (``cfg.stale_s``); everything else
    is set by the backend thread under ``_lock``.
    """

    def __init__(
        self,
        cfg: TrackerConfig,
        slot: LatestSlot[TrackerSample],
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self.slot = slot
        self._clock = clock
        self._lock = threading.Lock()
        self._status: TrackerStatus = "no_backend"
        self._detail = "tracker disabled (backend: none)" if cfg.backend == "none" else ""
        self._seq = 0
        self._last: TrackerSample | None = None
        self._rx_times: deque[float] = deque(maxlen=64)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_error_log = ""  # newest libsurvive error/warn line
        self._log_cb = None  # keep the ctypes callback alive for the ctx lifetime

    @property
    def backend(self) -> str:
        return self.cfg.backend

    # -- lifecycle -----------------------------------------------------------------
    def start(self) -> None:
        if self.cfg.backend == "none" or self._thread is not None:
            return
        self._stop.clear()
        self._set_status("starting", "")
        target = self._run_fake if self.cfg.backend == "fake" else self._run_libsurvive
        self._thread = threading.Thread(target=target, name="tracker-reader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    # -- status ----------------------------------------------------------------------
    def _set_status(self, status: TrackerStatus, detail: str) -> None:
        with self._lock:
            self._status = status
            self._detail = detail

    def status(self, now: float | None = None) -> TrackerDeviceStatus:
        now = self._clock() if now is None else now
        with self._lock:
            status, detail, last = self._status, self._detail, self._last
            rx = list(self._rx_times)
        age = None if last is None else max(0.0, now - last.rx_mono)
        if status == "tracking" and age is not None and age > self.cfg.stale_s:
            status = "stale"
        rate = (len(rx) - 1) / (rx[-1] - rx[0]) if len(rx) > 1 and rx[-1] > rx[0] else 0.0
        return TrackerDeviceStatus(
            backend=self.cfg.backend,
            status=status,
            detail=detail,
            object_name=self.cfg.object_name if self.cfg.backend != "none" else "",
            seq=last.seq if last is not None else 0,
            rate_hz=float(rate),
            age_s=age,
            pose_raw=last.pose if last is not None else None,
        )

    # -- publishing (backend threads) ----------------------------------------------------
    def _publish(self, pose: Pose, vel_lin, vel_ang, t_dev: float) -> TrackerSample:
        rx = self._clock()
        with self._lock:
            prev = self._last
            self._seq += 1
            valid = prev is None or (
                float(np.linalg.norm(pose.position - prev.pose.position)) <= self.cfg.max_jump_m
            )
            sample = TrackerSample(
                pose=pose,
                vel_lin=np.asarray(vel_lin, dtype=np.float64),
                vel_ang=np.asarray(vel_ang, dtype=np.float64),
                t_dev=float(t_dev),
                rx_mono=rx,
                seq=self._seq,
                valid=valid,
            )
            self._last = sample
            self._rx_times.append(rx)
            self._status = "tracking"
            self._detail = "" if valid else f"jump > {self.cfg.max_jump_m} m: sample invalid"
        if not valid:
            logger.warning(
                "tracker sample %d invalid (jump > %.2f m)", sample.seq, self.cfg.max_jump_m
            )
        self.slot.put(sample)
        return sample

    # -- fake backend: slow circle, identity orientation ----------------------------------
    def _run_fake(self) -> None:
        period = 1.0 / FAKE_RATE_HZ
        omega = 2.0 * math.pi / FAKE_PERIOD_S
        t0 = self._clock()
        next_t = t0
        while not self._stop.is_set():
            t = self._clock() - t0
            th = omega * t
            pos = FAKE_CENTER + FAKE_RADIUS_M * np.array([math.cos(th), math.sin(th), 0.0])
            vel = FAKE_RADIUS_M * omega * np.array([-math.sin(th), math.cos(th), 0.0])
            self._publish(Pose(pos, np.array([1.0, 0.0, 0.0, 0.0])), vel, np.zeros(3), t)
            next_t += period
            delay = next_t - self._clock()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_t = self._clock()

    # -- libsurvive backend --------------------------------------------------------------
    def _on_survive_log(self, _ctx, level, msg) -> None:
        """libsurvive logger callback: keep the newest error/warning line."""
        try:
            text = msg.decode(errors="replace") if isinstance(msg, bytes | bytearray) else str(msg)
        except Exception:  # never raise into C
            return
        if int(level) <= 1:  # SURVIVE_LOG_LEVEL_ERROR / _WARNING
            with self._lock:
                self._last_error_log = text.strip()
            logger.warning("libsurvive: %s", text.strip())
        else:
            logger.debug("libsurvive: %s", text.strip())

    def _error_detail(self, what: str) -> str:
        with self._lock:
            tail = self._last_error_log
        return f"{what}: {tail}" if tail else what

    def _run_libsurvive(self) -> None:
        import ctypes

        try:
            import pysurvive as ps
        except Exception as e:  # ImportError or a broken native lib
            self._set_status("no_backend", f"pysurvive not importable: {e!r}")
            return
        ptr = None
        try:
            args = ["apollo-xarm7-runtime", *self.cfg.libsurvive_args]
            argv = (ctypes.POINTER(ctypes.c_char) * (len(args) + 1))()
            for i, arg in enumerate(args):
                argv[i] = ctypes.create_string_buffer(arg.encode("utf-8"))
            self._log_cb = ps.SurviveSimpleLogFn(self._on_survive_log)
            ptr = ps.simple_init_with_logger(len(args), argv, self._log_cb)
            if not ptr:
                self._set_status("error", self._error_detail("libsurvive init failed"))
                return
            ps.simple_start_thread(ptr)
            self._set_status("searching", f"waiting for {self.cfg.object_name!r} poses")
            self._libsurvive_events(ps, ptr, ctypes)
        except Exception as e:
            logger.exception("tracker reader failed")
            self._set_status("error", self._error_detail(f"libsurvive failure: {e!r}"))
        finally:
            if ptr:
                try:
                    ps.simple_close(ptr)
                except Exception:
                    logger.exception("libsurvive close failed")
            self._log_cb = None

    def _libsurvive_events(self, ps, ptr, ctypes) -> None:
        ev = ps.SurviveSimpleEvent()
        t_start = self._clock()
        seen_object = False
        while not self._stop.is_set():
            et = ps.simple_wait_for_event(ptr, ctypes.byref(ev))
            if et == ps.SurviveSimpleEventType_Shutdown:
                if not self._stop.is_set():
                    self._set_status("error", self._error_detail("libsurvive shut down"))
                return
            if et != ps.SurviveSimpleEventType_PoseUpdateEvent:
                if not seen_object and self._clock() - t_start > LIBSURVIVE_NO_OBJECT_GRACE_S:
                    if ps.simple_get_object_count(ptr) == 0:
                        self._set_status(
                            "error",
                            self._error_detail(
                                "libsurvive found no devices: USB 28de:2101 not openable "
                                "(udev rule; scripts/tracker/01-sudo-udev-and-deps.sh) or "
                                "tracker off/unpaired"
                            ),
                        )
                    seen_object = True  # report once; poses (if any) flip to tracking
                continue
            pe = getattr(ev.d, "__private_pose_event")  # noqa: B009 - name-mangling guard
            obj = pe.object
            if ps.simple_object_get_type(obj) != ps.SurviveSimpleObject_OBJECT:
                continue  # lighthouses / HMD / external
            name = ps.simple_object_name(obj)
            name = name.decode(errors="replace") if isinstance(name, bytes) else str(name)
            if name != self.cfg.object_name:
                continue
            seen_object = True
            pose = Pose(np.array(pe.pose.Pos[:3]), np.array(pe.pose.Rot[:4]))  # m + wxyz
            self._publish(
                pose, np.array(pe.velocity.Pos[:3]), np.array(pe.velocity.AxisAngleRot[:3]),
                float(pe.time),
            )


__all__ = [
    "TrackerDeviceStatus",
    "TrackerReader",
    "TrackerSample",
    "TrackerSettings",
    "TrackerSettingsValues",
    "TrackerStatus",
]
