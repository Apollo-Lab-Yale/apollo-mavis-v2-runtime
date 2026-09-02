"""Vive-tracker reader (13-tracker §2/§4): libsurvive poses -> ``LatestSlot``.

A daemon thread around libsurvive's blocking event API publishes
:class:`TrackerSample` (meters + **wxyz**, same as core ``Pose``) stamped with
``time.monotonic()`` on receipt. Only objects of type OBJECT whose codename
equals ``object_name`` are used; lighthouses are ignored. Backends:
``libsurvive`` (real), ``fake`` (scripted 0.15 m circle at 100 Hz) and
``none``. ``pysurvive`` is imported lazily and ONLY here (ruff banned-api
elsewhere); a missing module means status ``no_backend``, never a crash.

Robustness (13-tracker §4 "Reader robustness"): every libsurvive event is
guarded (finite position, unit-norm quaternion; bad events are dropped and
counted, exceptions never kill the thread), the libsurvive loop auto-restarts
with backoff when it dies, ``start()`` is restartable, libsurvive warnings are
rate-limited per message class, and the device status distinguishes ``error``
(no OBJECT-type device) from ``searching`` (device present, no pose yet).
"""

from __future__ import annotations

import logging
import math
import re
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
LIBSURVIVE_OBJECT_RECHECK_S = 1.0  # re-enumerate objects while no pose has arrived
LIBSURVIVE_RESTART_BACKOFF_S = (0.5, 5.0)  # (first, max) delay before restarting a dead loop
LIBSURVIVE_RESTART_RESET_S = 30.0  # a run longer than this resets the backoff
RATE_WINDOW_S = 1.0  # rate_hz is measured over this window; decays to 0 when samples stop
LOG_RATE_LIMIT_S = 1.0  # forwarded libsurvive warnings: <= 1 line/s per message class
QUAT_NORM_TOL = 1e-2  # |‖q‖ - 1| above this -> bad event

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

TrackpadDir = Literal["trackpad_left", "trackpad_right", "trackpad_up", "trackpad_down"]


def _code_for(action: str) -> str:
    return next(e.code for e in KEYMAP if e.action == action)


# Device-held codes are the keymap codes of the actions they stand in for.
CLUTCH_CODE: str = _code_for("tracker_clutch")
GRIPPER_OPEN_CODE: str = _code_for("gripper_open")
GRIPPER_CLOSE_CODE: str = _code_for("gripper_close")

# Discrete device actions -> the ActionName the loop executes (13-tracker §1.1).
DISCRETE_ACTIONS: dict[str, str] = {"arm_next": "switch_arm", "arm_prev": "switch_arm_prev"}


@dataclass(frozen=True)
class ControllerState:
    """Latest Vive-controller input state (mirrors core ``ControllerTelemetry``).

    ``click_seq`` counts trackpad press edges and ``click_dir`` is the
    classification of the newest one (13-tracker §1.1: dominant axis at the
    press edge; ``None`` when both axes were inside the deadzone). Both are
    maintained by :func:`classify_click` (the reader applies it to every new
    state), so held codes follow ``click_dir`` while ``trackpad_click`` is True
    and the loop fires discrete actions when ``click_seq`` advances.
    """

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
    click_seq: int = 0  # number of trackpad press edges so far
    click_dir: TrackpadDir | None = None  # classification of the newest press edge

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
    Unknown button or axis ids are ignored. Raw state only: the trackpad click
    classification is added by :func:`classify_click`.
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


def classify_trackpad(x: float, y: float, deadzone: float) -> TrackpadDir | None:
    """Trackpad position -> direction by the dominant axis; ``None`` when both
    ``|x|`` and ``|y|`` are within ``deadzone`` (13-tracker §1.1)."""
    ax, ay = abs(float(x)), abs(float(y))
    if ax <= deadzone and ay <= deadzone:
        return None
    if ax >= ay:
        return "trackpad_right" if x > 0.0 else "trackpad_left"
    return "trackpad_up" if y > 0.0 else "trackpad_down"


def classify_click(
    prev: ControllerState | None, state: ControllerState, deadzone: float
) -> ControllerState:
    """Fix the trackpad click classification ONCE at its press edge and carry
    it (with the press counter) until release; a click that starts inside the
    deadzone stays ignored however far the finger moves afterwards."""
    prev_click = prev.trackpad_click if prev is not None else False
    seq = prev.click_seq if prev is not None else 0
    if state.trackpad_click and not prev_click:  # press edge: classify now
        return replace(
            state, click_seq=seq + 1,
            click_dir=classify_trackpad(state.trackpad_x, state.trackpad_y, deadzone),
        )
    if prev is not None:  # held or released: keep the edge accounting
        return replace(state, click_seq=seq, click_dir=prev.click_dir)
    return state


def active_inputs(state: ControllerState) -> frozenset[ControllerInput]:
    """Which bindable held inputs (``ControllerInput``) are active in ``state``:
    the trigger click and, while the trackpad is clicked, its press-edge
    classification (``click_dir``)."""
    out: set[ControllerInput] = set()
    if state.trigger_pressed:
        out.add("trigger_click")
    if state.trackpad_click and state.click_dir is not None:
        out.add(state.click_dir)
    return frozenset(out)


def derive_held_codes(state: ControllerState | None, cfg: TrackerConfig) -> frozenset[str]:
    """Key codes the controller currently stands in for (13-tracker §1.1):
    ``cfg.controller_map`` binds each held action to an input; ``none`` = unbound."""
    if state is None:
        return frozenset()
    active = active_inputs(state)
    m = cfg.controller_map
    bindings = (
        (m.clutch, CLUTCH_CODE), (m.gripper_open, GRIPPER_OPEN_CODE),
        (m.gripper_close, GRIPPER_CLOSE_CODE),
    )
    return frozenset(code for src, code in bindings if src != "none" and src in active)


def derive_click_action(state: ControllerState | None, cfg: TrackerConfig) -> str | None:
    """ActionName (``switch_arm`` / ``switch_arm_prev``) bound to the newest
    trackpad press edge (``state.click_dir``), or ``None`` (unbound / deadzone).
    The loop fires it once when ``state.click_seq`` advances."""
    if state is None or state.click_dir is None:
        return None
    m = cfg.controller_map
    for field, action in DISCRETE_ACTIONS.items():
        if getattr(m, field) == state.click_dir:
            return action
    return None


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
    click_action: str | None = None  # action bound to controller.click_dir (press edge)


@dataclass(frozen=True)
class TrackerSettingsValues:
    """Immutable snapshot of the live tracker teleop settings (13-tracker §4).

    ``filter_*`` are the live-tunable One Euro pose-filter fields (enabled /
    min cutoff / speed coefficient); the static filter fields (``d_cutoff_hz``,
    deadbands) stay in ``TrackerConfig.filter``.
    """

    yaw_deg: float
    pos_scale: float
    follow_rotation: bool
    filter_enabled: bool = True
    filter_min_cutoff_hz: float = 1.0
    filter_beta: float = 0.05


class TrackerSettings:
    """Live, thread-safe yaw/scale/rotation/filter settings (process lifetime).

    ``get`` returns an immutable snapshot; ``update`` applies the non-None
    fields (``tracker_settings`` action semantics: omitted = unchanged).
    """

    def __init__(
        self,
        yaw_deg: float = 0.0,
        pos_scale: float = 1.0,
        follow_rotation: bool = True,
        *,
        filter_enabled: bool = True,
        filter_min_cutoff_hz: float = 1.0,
        filter_beta: float = 0.05,
    ) -> None:
        self._lock = threading.Lock()
        self._values = TrackerSettingsValues(
            float(yaw_deg), float(pos_scale), bool(follow_rotation),
            bool(filter_enabled), float(filter_min_cutoff_hz), float(filter_beta),
        )

    @classmethod
    def from_config(cls, cfg: TrackerConfig) -> TrackerSettings:
        return cls(
            cfg.yaw_deg, cfg.pos_scale, cfg.follow_rotation,
            filter_enabled=cfg.filter.enabled,
            filter_min_cutoff_hz=cfg.filter.min_cutoff_hz,
            filter_beta=cfg.filter.beta,
        )

    def get(self) -> TrackerSettingsValues:
        with self._lock:
            return self._values

    def update(
        self,
        *,
        yaw_deg: float | None = None,
        pos_scale: float | None = None,
        follow_rotation: bool | None = None,
        filter_enabled: bool | None = None,
        filter_min_cutoff_hz: float | None = None,
        filter_beta: float | None = None,
    ) -> TrackerSettingsValues:
        with self._lock:
            v = self._values
            if yaw_deg is not None:
                v = replace(v, yaw_deg=float(yaw_deg))
            if pos_scale is not None:
                v = replace(v, pos_scale=float(pos_scale))
            if follow_rotation is not None:
                v = replace(v, follow_rotation=bool(follow_rotation))
            if filter_enabled is not None:
                v = replace(v, filter_enabled=bool(filter_enabled))
            if filter_min_cutoff_hz is not None:
                v = replace(v, filter_min_cutoff_hz=float(filter_min_cutoff_hz))
            if filter_beta is not None:
                v = replace(v, filter_beta=float(filter_beta))
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
    bad_events: int = 0  # libsurvive events dropped by the per-event guard
    restarts: int = 0  # libsurvive loop auto-restarts so far


class TrackerReader:
    """Owns the device thread for the process lifetime; publishes to ``slot``.

    ``start()`` is a no-op for backend ``none`` and while a thread is alive; a
    finished thread clears ``_thread`` so ``start()`` can be called again.
    ``status(now)`` derives ``stale`` from the newest sample's age
    (``cfg.stale_s``) and a ``rate_hz`` over the last ``RATE_WINDOW_S`` that
    decays to 0 when samples stop; everything else is set by the backend thread
    under ``_lock``.

    Controller inputs (13-tracker §1.1): the libsurvive backend folds button
    events into a :class:`ControllerState`; every published sample carries the
    newest state plus the derived ``held_codes`` / ``click_action``, and a
    button edge re-publishes the last pose immediately so the loop sees the
    edge within one tick. The fake backend has no buttons (``controller=None``)
    unless a ``controller_provider`` (test hook, polled every fake tick)
    supplies one.
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
        self._click_action: str | None = None
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
        self._log_last: dict[str, float] = {}  # message class -> last forwarded time
        self._log_suppressed: dict[str, int] = {}
        self.bad_events = 0  # dropped libsurvive events (guard)
        self.restarts = 0  # libsurvive loop auto-restarts

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
        self._thread = threading.Thread(
            target=self._thread_main, args=(target,), name="tracker-reader", daemon=True
        )
        self._thread.start()

    def _thread_main(self, target: Callable[[], None]) -> None:
        try:
            target()
        except Exception:  # last line of defence: the thread must end cleanly
            logger.exception("tracker reader thread crashed")
            self._set_status("error", self._error_detail("tracker reader crashed"))
        finally:
            self._thread = None  # start() may spawn a fresh thread

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)
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
            rx = [t for t in self._rx_times if t > now - RATE_WINDOW_S]
            controller = self._controller
        age = None if last is None else max(0.0, now - last.rx_mono)
        if status == "tracking" and age is not None and age > self.cfg.stale_s:
            status = "stale"
        # Samples within the window over the time since the oldest of them: a
        # dead stream decays toward 0 and reads exactly 0 once the window is empty.
        rate = (len(rx) - 1) / (now - rx[0]) if len(rx) > 1 and now > rx[0] else 0.0
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
            bad_events=self.bad_events,
            restarts=self.restarts,
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
                click_action=self._click_action,
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
        """Adopt a new controller state (trackpad click classified at its press
        edge, :func:`classify_click`); on a button edge (or a change of the
        derived codes) re-publish the last pose so the loop sees the edge now.

        The re-published sample keeps the pose's own validity, but a pose older
        than ``stale_s`` is flagged invalid: the codes still count (gripper,
        clutch *held*), while the clutch cannot anchor on a stale pose.
        Returns True when a sample was re-published.
        """
        with self._lock:
            prev, prev_codes, last = self._controller, self._held_codes, self._last
        if state is not None:
            state = classify_click(prev, state, self.cfg.trackpad_deadzone)
        codes = derive_held_codes(state, self.cfg)
        with self._lock:
            self._controller = state
            self._held_codes = codes
            self._click_action = derive_click_action(state, self.cfg)
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
        """libsurvive logger callback: keep the newest error/warning line and
        forward it rate-limited (<= 1 line/s per message class, digits masked)."""
        try:
            text = msg.decode(errors="replace") if isinstance(msg, bytes | bytearray) else str(msg)
            text = text.strip()
            if int(level) <= 1:  # SURVIVE_LOG_LEVEL_ERROR / _WARNING
                with self._lock:
                    self._last_error_log = text
                self._forward_warning(text)
            else:
                logger.debug("libsurvive: %s", text)
        except Exception:  # never raise into C
            return

    def _forward_warning(self, text: str) -> None:
        key = re.sub(r"\d+", "#", text)[:48]
        now = self._clock()
        with self._lock:
            last = self._log_last.get(key)
            if last is not None and now - last < LOG_RATE_LIMIT_S:
                self._log_suppressed[key] = self._log_suppressed.get(key, 0) + 1
                return
            self._log_last[key] = now
            suppressed = self._log_suppressed.pop(key, 0)
        if suppressed:
            logger.warning("libsurvive: %s (+%d similar suppressed)", text, suppressed)
        else:
            logger.warning("libsurvive: %s", text)

    def _error_detail(self, what: str) -> str:
        with self._lock:
            tail = self._last_error_log
        return f"{what}: {tail}" if tail else what

    def _run_libsurvive(self) -> None:
        """Supervisor: run the libsurvive loop and restart it with backoff when
        it dies (dongle busy/unplugged, libsurvive shutdown, unexpected
        exception). A missing ``pysurvive`` is final (``no_backend``)."""
        try:
            import pysurvive as ps
        except Exception as e:  # ImportError or a broken native lib
            self._set_status("no_backend", f"pysurvive not importable: {e!r}")
            return
        first, cap = LIBSURVIVE_RESTART_BACKOFF_S
        backoff = first
        while not self._stop.is_set():
            t0 = self._clock()
            self._run_libsurvive_once(ps)
            if self._stop.is_set():
                return
            if self._clock() - t0 > LIBSURVIVE_RESTART_RESET_S:
                backoff = first  # a long healthy run: start the ladder over
            self.restarts += 1
            logger.warning(
                "tracker: libsurvive loop ended (%s); restart %d in %.1f s",
                self.status().detail or "no detail", self.restarts, backoff,
            )
            self._stop.wait(backoff)
            backoff = min(backoff * 2.0, cap)

    def _run_libsurvive_once(self, ps) -> None:
        import ctypes

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
                    ps.simple_close(ptr)  # releases the USB interface (13-tracker §6)
                except Exception:
                    logger.exception("libsurvive close failed")
            self._log_cb = None

    def _libsurvive_events(self, ps, ptr, ctypes) -> None:
        """Drain libsurvive's event queue (non-blocking ``simple_next_event`` +
        1 ms idle wait, so the no-device check and ``stop()`` never hang on a
        silent dongle). Every event is guarded: a malformed one is dropped and
        counted, an exception is logged (rate-limited) and the loop goes on."""
        ev = ps.SurviveSimpleEvent()
        t_start = self._clock()
        seen_pose = False
        next_check = t_start + LIBSURVIVE_NO_OBJECT_GRACE_S
        while not self._stop.is_set():
            et = ps.simple_next_event(ptr, ctypes.byref(ev))
            if et == ps.SurviveSimpleEventType_Shutdown:
                if not self._stop.is_set():
                    self._set_status("error", self._error_detail("libsurvive shut down"))
                return
            if et == ps.SurviveSimpleEventType_None:
                if not seen_pose and self._clock() >= next_check:
                    self._report_no_pose(ps, ptr)
                    next_check = self._clock() + LIBSURVIVE_OBJECT_RECHECK_S
                self._stop.wait(0.001)
                continue
            try:
                if et == ps.SurviveSimpleEventType_ButtonEvent:
                    be = getattr(ev.d, "__private_button_event")  # noqa: B009 - name-mangling guard
                    if self._is_tracked_object(ps, be.object):
                        self._on_button_event(be)
                elif et == ps.SurviveSimpleEventType_PoseUpdateEvent:
                    pe = getattr(ev.d, "__private_pose_event")  # noqa: B009 - name-mangling guard
                    if self._is_tracked_object(ps, pe.object):  # not lighthouses/HMD/others
                        seen_pose = self._on_pose_event(pe) or seen_pose
            except Exception:
                self.bad_events += 1
                self._forward_warning(f"event {int(et)} handling failed: dropped")
                logger.debug("libsurvive event handling failed", exc_info=True)

    def _on_pose_event(self, pe) -> bool:
        """Guarded pose event -> publish; False when the event was dropped."""
        pos = np.array(pe.pose.Pos[:3], dtype=np.float64)
        rot = np.array(pe.pose.Rot[:4], dtype=np.float64)  # wxyz, same as core Pose
        if (
            pos.shape != (3,) or rot.shape != (4,)
            or not np.all(np.isfinite(pos)) or not np.all(np.isfinite(rot))
            or abs(float(np.linalg.norm(rot)) - 1.0) > QUAT_NORM_TOL
        ):
            self.bad_events += 1
            self._forward_warning("pose event with non-finite / non-unit data: dropped")
            return False
        vel_lin = np.nan_to_num(np.array(pe.velocity.Pos[:3], dtype=np.float64))
        vel_ang = np.nan_to_num(np.array(pe.velocity.AxisAngleRot[:3], dtype=np.float64))
        self._publish(Pose(pos, rot / np.linalg.norm(rot)), vel_lin, vel_ang, float(pe.time))
        return True

    def _object_names(self, ps, ptr) -> list[str]:
        """Codenames of every OBJECT-type libsurvive object (trackers/controllers/
        HMD; lighthouses excluded)."""
        names: list[str] = []
        obj = ps.simple_get_first_object(ptr)
        while obj:
            if ps.simple_object_get_type(obj) == ps.SurviveSimpleObject_OBJECT:
                name = ps.simple_object_name(obj)
                if isinstance(name, bytes):
                    name = name.decode(errors="replace")
                names.append(str(name))
            obj = ps.simple_get_next_object(ptr, obj)
        return names

    def _report_no_pose(self, ps, ptr) -> None:
        """No pose from the tracked object yet (after the grace period): ``error``
        when libsurvive sees no OBJECT-type device at all, else ``searching``
        with the reason (name mismatch / base stations) spelled out."""
        wanted = self.cfg.object_name
        try:
            names = self._object_names(ps, ptr)
        except Exception as e:
            self._set_status("error", self._error_detail(f"object enumeration failed: {e!r}"))
            return
        if not names:
            self._set_status(
                "error",
                self._error_detail(
                    "libsurvive found no tracked device (OBJECT type): dongle busy "
                    "(LIBUSB_ERROR_BUSY - a killed libsurvive keeps the USB interface "
                    "claimed; fuser /dev/bus/usb/<bus>/<dev>) or not openable (udev rule; "
                    "scripts/tracker/01-sudo-udev-and-deps.sh), tracker off, or unpaired "
                    "(survive-cli --pair-device)"
                ),
            )
        elif wanted not in names:
            self._set_status(
                "searching",
                f"no poses from {wanted!r}; OBJECT devices present: {', '.join(names)} "
                f"(none named {wanted!r}: check tracker.object_name)",
            )
        else:
            self._set_status(
                "searching",
                f"waiting for {wanted!r} poses (device present; needs >= 2 base stations "
                "visible and a still tracker on first start)",
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
    "DISCRETE_ACTIONS",
    "GRIPPER_CLOSE_CODE",
    "GRIPPER_OPEN_CODE",
    "ControllerState",
    "TrackerDeviceStatus",
    "TrackerReader",
    "TrackerSample",
    "TrackerSettings",
    "TrackerSettingsValues",
    "TrackerStatus",
    "TrackpadDir",
    "active_inputs",
    "apply_button_event",
    "classify_click",
    "classify_trackpad",
    "derive_click_action",
    "derive_held_codes",
]
