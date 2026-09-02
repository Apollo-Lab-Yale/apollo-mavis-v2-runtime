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
from apollo_xarm7_core.protocol import KEYMAP

from ..config import ControllerInput, TrackerConfig

logger = logging.getLogger(__name__)

TrackerStatus = Literal["no_backend", "starting", "searching", "tracking", "stale", "error"]

FAKE_RATE_HZ = 100.0
FAKE_RADIUS_M = 0.15
FAKE_PERIOD_S = 20.0  # one slow lap; ~0.047 m/s tangential
FAKE_CENTER = np.array([0.5, 0.0, 0.3])
LIBSURVIVE_NO_OBJECT_GRACE_S = 3.0  # init ok but zero objects -> not openable / not paired

# libsurvive button events, verified on the lab Vive Pro controller (13-tracker §1.1).
EVENT_BUTTON_UP = 2
EVENT_BUTTON_DOWN = 3
EVENT_TOUCH_UP = 4
EVENT_TOUCH_DOWN = 5
EVENT_AXIS_CHANGED = 8  # button_id 255 + axis_ids/axis_val arrays
BUTTON_TRIGGER = 0
BUTTON_TRACKPAD = 1
BUTTON_SYSTEM = 3
BUTTON_MENU = 6
BUTTON_GRIP = 7
AXIS_TRIGGER = 1  # 0..1
AXIS_TRACKPAD_X = 2  # -1..1
AXIS_TRACKPAD_Y = 3  # -1..1, +y = top


def _code_for(action: str) -> str:
    return next(e.code for e in KEYMAP if e.action == action)


# Device-held codes are the keymap codes of the actions they stand in for.
CLUTCH_CODE: str = _code_for("tracker_clutch")
GRIPPER_OPEN_CODE: str = _code_for("gripper_open")
GRIPPER_CLOSE_CODE: str = _code_for("gripper_close")


@dataclass(frozen=True)
class ControllerState:
    """Latest Vive-controller input state (mirrors core ``ControllerTelemetry``)."""

    trigger: float = 0.0  # analog pull, 0..1 (axis 1)
    trigger_pressed: bool = False  # button 0 down/up edges
    trackpad_touch: bool = False  # button 1 TOUCH_DOWN/UP
    trackpad_click: bool = False  # button 1 BUTTON_DOWN/UP
    trackpad_x: float = 0.0  # axis 2, -1..1
    trackpad_y: float = 0.0  # axis 3, -1..1, +y = top
    grip: bool = False  # button 7
    menu: bool = False  # button 6
    system: bool = False  # button 3
    rx_mono: float = 0.0  # time.monotonic() of the newest event folded in

    def buttons(self) -> tuple[bool, ...]:
        """Boolean inputs only (edge detection ignores the analog axes)."""
        return (
            self.trigger_pressed, self.trackpad_touch, self.trackpad_click,
            self.grip, self.menu, self.system,
        )


_BUTTON_FIELDS = {
    BUTTON_TRIGGER: "trigger_pressed",
    BUTTON_TRACKPAD: "trackpad_click",
    BUTTON_GRIP: "grip",
    BUTTON_MENU: "menu",
    BUTTON_SYSTEM: "system",
}
_AXIS_FIELDS = {
    AXIS_TRIGGER: "trigger", AXIS_TRACKPAD_X: "trackpad_x", AXIS_TRACKPAD_Y: "trackpad_y"
}


def apply_button_event(
    prev: ControllerState,
    event_type: int,
    button_id: int,
    axis_ids,
    axis_vals,
    rx_mono: float,
) -> ControllerState:
    """Fold one libsurvive button/touch/axis event into ``prev``.

    ``axis_ids`` / ``axis_vals`` are the event's ``axis_count`` leading entries;
    they are applied for every event type (button events may carry axes too).
    Unknown button or axis ids are ignored.
    """
    changes: dict = {"rx_mono": float(rx_mono)}
    for axis_id, val in zip(axis_ids, axis_vals, strict=True):
        field = _AXIS_FIELDS.get(int(axis_id))
        if field is not None:
            changes[field] = float(val)
    if event_type in (EVENT_BUTTON_DOWN, EVENT_BUTTON_UP):
        field = _BUTTON_FIELDS.get(int(button_id))
        if field is not None:
            changes[field] = event_type == EVENT_BUTTON_DOWN
    elif event_type in (EVENT_TOUCH_DOWN, EVENT_TOUCH_UP) and int(button_id) == BUTTON_TRACKPAD:
        changes["trackpad_touch"] = event_type == EVENT_TOUCH_DOWN
    return replace(prev, **changes)


def active_inputs(state: ControllerState, deadzone: float) -> frozenset[ControllerInput]:
    """Which bindable inputs (``ControllerInput``) are active in ``state``."""
    out: set[ControllerInput] = set()
    if state.trigger_pressed:
        out.add("trigger_click")
    if state.trackpad_click:
        if state.trackpad_y > deadzone:
            out.add("trackpad_up")
        elif state.trackpad_y < -deadzone:
            out.add("trackpad_down")
    return frozenset(out)


def derive_held_codes(state: ControllerState | None, cfg: TrackerConfig) -> frozenset[str]:
    """Key codes the controller currently stands in for (13-tracker §1.1):
    ``cfg.controller_map`` binds each action to an input; ``none`` = unbound."""
    if state is None:
        return frozenset()
    active = active_inputs(state, cfg.trackpad_deadzone)
    m = cfg.controller_map
    bindings = (
        (m.clutch, CLUTCH_CODE), (m.gripper_open, GRIPPER_OPEN_CODE),
        (m.gripper_close, GRIPPER_CLOSE_CODE),
    )
    return frozenset(code for src, code in bindings if src != "none" and src in active)


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
    controller: ControllerState | None = None  # None: backend reports no controller
    held_codes: frozenset[str] = frozenset()  # device-held codes (13-tracker §1.1)


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
    controller: ControllerState | None = None  # newest controller state (edge or pose)
    device_held: frozenset[str] = frozenset()  # newest sample's codes; empty when stale


class TrackerReader:
    """Owns the device thread for the process lifetime; publishes to ``slot``.

    ``start()`` is a no-op for backend ``none``. ``status(now)`` derives
    ``stale`` from the newest sample's age (``cfg.stale_s``); everything else
    is set by the backend thread under ``_lock``.

    Controller inputs (13-tracker §1.1): the libsurvive backend folds button
    events into a :class:`ControllerState`; every published sample carries the
    newest state plus the derived ``held_codes``, and a button edge re-publishes
    the last pose immediately so the loop sees the edge within one tick. The
    fake backend has no buttons (``controller=None``) unless a
    ``controller_provider`` (test hook, polled every fake tick) supplies one.
    """

    def __init__(
        self,
        cfg: TrackerConfig,
        slot: LatestSlot[TrackerSample],
        *,
        clock: Callable[[], float] = time.monotonic,
        controller_provider: Callable[[], ControllerState | None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.slot = slot
        self._clock = clock
        self.controller_provider = controller_provider  # fake backend only; settable live
        self._controller: ControllerState | None = None
        self._held_codes: frozenset[str] = frozenset()
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
            controller = self._controller
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
            controller=controller,
            device_held=(
                last.held_codes
                if last is not None and age is not None and age <= self.cfg.stale_s
                else frozenset()
            ),
        )

    # -- publishing (backend threads) ----------------------------------------------------
    def _publish(
        self, pose: Pose, vel_lin, vel_ang, t_dev: float, *, valid_override: bool | None = None
    ) -> TrackerSample:
        rx = self._clock()
        with self._lock:
            prev = self._last
            self._seq += 1
            valid = prev is None or (
                float(np.linalg.norm(pose.position - prev.pose.position)) <= self.cfg.max_jump_m
            )
            if valid_override is not None:
                valid = valid_override
            sample = TrackerSample(
                pose=pose,
                vel_lin=np.asarray(vel_lin, dtype=np.float64),
                vel_ang=np.asarray(vel_ang, dtype=np.float64),
                t_dev=float(t_dev),
                rx_mono=rx,
                seq=self._seq,
                valid=valid,
                controller=self._controller,
                held_codes=self._held_codes,
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

    # -- controller inputs (13-tracker §1.1) ----------------------------------------------
    def _on_controller(self, state: ControllerState | None) -> bool:
        """Adopt a new controller state; on a button edge (or a change of the
        derived codes) re-publish the last pose so the loop sees the edge now.

        The re-published sample keeps the pose's own validity, but a pose older
        than ``stale_s`` is flagged invalid: the codes still count (gripper,
        clutch *held*), while the clutch cannot anchor on a stale pose.
        Returns True when a sample was re-published.
        """
        codes = derive_held_codes(state, self.cfg)
        with self._lock:
            prev, prev_codes, last = self._controller, self._held_codes, self._last
            self._controller = state
            self._held_codes = codes
        edge = codes != prev_codes or (
            (state.buttons() if state is not None else None)
            != (prev.buttons() if prev is not None else None)
        )
        if not edge or last is None:
            return False
        fresh = self._clock() - last.rx_mono <= self.cfg.stale_s
        self._publish(
            last.pose, last.vel_lin, last.vel_ang, last.t_dev,
            valid_override=last.valid and fresh,
        )
        return True

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
            provider = self.controller_provider
            if provider is not None:
                self._on_controller(provider())  # scripted buttons: same edge path
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
            if et == ps.SurviveSimpleEventType_ButtonEvent:
                be = getattr(ev.d, "__private_button_event")  # noqa: B009 - name-mangling guard
                if self._is_tracked_object(ps, be.object):
                    self._on_button_event(be)
                continue
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
            if not self._is_tracked_object(ps, pe.object):
                continue  # lighthouses / HMD / external / other trackers
            seen_object = True
            pose = Pose(np.array(pe.pose.Pos[:3]), np.array(pe.pose.Rot[:4]))  # m + wxyz
            self._publish(
                pose, np.array(pe.velocity.Pos[:3]), np.array(pe.velocity.AxisAngleRot[:3]),
                float(pe.time),
            )

    def _is_tracked_object(self, ps, obj) -> bool:
        """Type OBJECT and codename == ``cfg.object_name`` (13-tracker §4)."""
        if ps.simple_object_get_type(obj) != ps.SurviveSimpleObject_OBJECT:
            return False
        name = ps.simple_object_name(obj)
        name = name.decode(errors="replace") if isinstance(name, bytes) else str(name)
        return name == self.cfg.object_name

    def _on_button_event(self, be) -> None:
        """``SurviveSimpleButtonEvent`` (time, object, event_type, button_id,
        axis_count, axis_ids[8], axis_val[8]) -> controller state (+ edge publish)."""
        n = max(0, min(int(be.axis_count), 8))
        prev = self._controller or ControllerState()
        state = apply_button_event(
            prev, int(be.event_type), int(be.button_id),
            [be.axis_ids[i] for i in range(n)], [be.axis_val[i] for i in range(n)],
            self._clock(),
        )
        self._on_controller(state)


__all__ = [
    "CLUTCH_CODE",
    "GRIPPER_CLOSE_CODE",
    "GRIPPER_OPEN_CODE",
    "ControllerState",
    "TrackerDeviceStatus",
    "TrackerReader",
    "TrackerSample",
    "TrackerSettings",
    "TrackerSettingsValues",
    "TrackerStatus",
    "active_inputs",
    "apply_button_event",
    "derive_held_codes",
]
