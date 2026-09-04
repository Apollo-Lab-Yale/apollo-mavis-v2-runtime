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

Controller inputs (13-tracker §1.1): button/axis events fold into a
:class:`ControllerState`; :func:`note_edges` classifies a trackpad click once
at its press edge and records the press edges of the trackpad, menu and grip
buttons as a short ``(seq, input)`` history (``edges``; ``edge_seq`` /
``edge_input`` are its newest entry). ``TrackerConfig.controller_map`` binds
inputs to held actions (clutch, gripper_open/close, rail_neg/pos -> the keymap
codes of those actions, published as ``TrackerSample.held_codes``) and to the
discrete actions arm_next / arm_prev (``TrackerSample.click_actions``: the
``(seq, action)`` of every remembered bound edge; the control loop fires those
newer than the last ``edge_seq`` it saw on a fresh sample, so two edges inside
one tick both fire). A controller edge re-publishes the last pose so the loop
sees it within a tick, but never refreshes the pose's age: a pose older than
``stale_s`` is re-published invalid and status / age / rate follow the real
pose stream only.

Calibration hooks (13-tracker §4 "Calibration modes"; used by
``devices.tracker_calibration``): :meth:`TrackerReader.restart` swaps the
libsurvive arguments (a per-reader ``TrackerConfig`` copy, never the shared
one) after a clean ``simple_close``; libsurvive INFO lines are kept ANSI-free
in ``info_lines`` (and handed to an optional ``on_info`` callback); and the
reader thread refreshes a :class:`LighthouseSnapshot` list every
``LIBSURVIVE_LIGHTHOUSE_POLL_S`` from libsurvive's LIGHTHOUSE objects.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
from apollo_xarm7_core import LatestSlot, Pose
from apollo_xarm7_core.protocol import KEYMAP

from ..config import (
    CONTROLLER_DISCRETE_ACTIONS,
    CONTROLLER_HELD_ACTIONS,
    ControllerInput,
    TrackerConfig,
)

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
CHARGE_POLL_S = 2.0  # re-read simple_object_charging at most this often (cheap device read)
LIBSURVIVE_LIGHTHOUSE_POLL_S = 0.5  # refresh the lighthouse snapshot (name/serial/pose) this often
INFO_LINES_MAX = 256  # libsurvive INFO lines kept for the calibration FSM (ANSI stripped)
STOP_JOIN_TIMEOUT_S = 5.0  # stop(): wait this long for the backend thread (simple_close)
RESTART_JOIN_TIMEOUT_S = 20.0  # restart(): a slow simple_close (USB stall) gets this long
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")  # libsurvive colours its log text
QUAT_NORM_TOL = 1e-2  # |‖q‖ - 1| above this -> bad event
EDGE_HISTORY = 8  # press edges remembered on ControllerState.edges (lossless up to 8 per tick)

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
RAIL_NEG_CODE: str = _code_for("rail_neg")
RAIL_POS_CODE: str = _code_for("rail_pos")

# Held controller_map actions -> the code they inject while their input is active.
HELD_ACTION_CODES: dict[str, str] = {
    "clutch": CLUTCH_CODE,
    "gripper_open": GRIPPER_OPEN_CODE,
    "gripper_close": GRIPPER_CLOSE_CODE,
    "rail_neg": RAIL_NEG_CODE,
    "rail_pos": RAIL_POS_CODE,
}

# Discrete device actions -> the ActionName the loop executes (13-tracker §1.1).
DISCRETE_ACTIONS: dict[str, str] = {"arm_next": "switch_arm", "arm_prev": "switch_arm_prev"}

if (
    tuple(HELD_ACTION_CODES) != CONTROLLER_HELD_ACTIONS
    or tuple(DISCRETE_ACTIONS) != CONTROLLER_DISCRETE_ACTIONS
):  # pragma: no cover - import-time invariant: config and reader agree on the action names
    raise RuntimeError(
        "controller_map action tables out of sync: config.CONTROLLER_*_ACTIONS vs "
        "devices.tracker HELD_ACTION_CODES / DISCRETE_ACTIONS"
    )


@dataclass(frozen=True)
class ControllerState:
    """Latest Vive-controller input state (mirrors core ``ControllerTelemetry``).

    Discrete edge accounting (13-tracker §1.1): ``edges`` remembers the last
    ``EDGE_HISTORY`` press edges of the trackpad click, the menu button and the
    grip button as ``(seq, input)``, newest last, where ``seq`` counts every
    press edge so far and ``input`` is the ``ControllerInput`` that produced it
    — the trackpad click's classification (dominant axis at the press edge;
    ``None`` when both axes were inside the deadzone, the counter still
    advances), ``menu_click`` or ``grip_click``. ``edge_seq`` / ``edge_input``
    are the newest entry. Keeping a history (not just the newest edge) makes
    the accounting lossless when two buttons edge inside one 10 ms loop tick
    (the slot is depth 1, the loop reads it once per tick). ``trackpad_dir``
    carries the current click's classification until release so held codes
    follow it while ``trackpad_click`` is True even if another button edges
    meanwhile. All are maintained by :func:`note_edges` (the reader applies it
    to every new state); the loop fires the discrete actions of the edges newer
    than the ``edge_seq`` it saw last. The trigger is held-only and registers
    no edge.
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
    trackpad_dir: TrackpadDir | None = None  # current click's classification, held to release
    # Last EDGE_HISTORY press edges (trackpad click, menu, grip) as (seq, input), newest last.
    edges: tuple[tuple[int, ControllerInput | None], ...] = ()

    @property
    def edge_seq(self) -> int:
        """Press edges so far (trackpad click, menu, grip); 0 before the first."""
        return self.edges[-1][0] if self.edges else 0

    @property
    def edge_input(self) -> ControllerInput | None:
        """Input of the newest press edge (``None``: none yet / deadzone click)."""
        return self.edges[-1][1] if self.edges else None

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
    Unknown button or axis ids are ignored. Raw state only: the edge accounting
    and the trackpad click classification are added by :func:`note_edges`.
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


def note_edges(
    prev: ControllerState | None, state: ControllerState, deadzone: float
) -> ControllerState:
    """Discrete press-edge accounting + trackpad click classification.

    Compares ``state`` with ``prev`` (the previously adopted state; ``None`` =
    first observation, every pressed button counts as an edge): a trackpad
    press edge is classified ONCE from the pad position (``trackpad_dir``,
    carried until release; a click that starts inside the deadzone stays
    ignored however far the finger moves afterwards), and every press edge of
    the trackpad click, the menu button and the grip button appends
    ``(seq, input)`` to ``edges`` (the deadzone click appends ``None`` as the
    input); the history keeps the newest ``EDGE_HISTORY`` entries.
    Simultaneous edges in one state (scripted states only: libsurvive delivers
    one button per event) are appended in the order trackpad, menu, grip.
    Held or released states carry the previous accounting unchanged.
    """
    seq = prev.edge_seq if prev is not None else 0
    history = prev.edges if prev is not None else ()
    trackpad_dir = prev.trackpad_dir if prev is not None else None
    new: list[tuple[int, ControllerInput | None]] = []
    if state.trackpad_click and not (prev is not None and prev.trackpad_click):
        trackpad_dir = classify_trackpad(state.trackpad_x, state.trackpad_y, deadzone)
        new.append((seq + len(new) + 1, trackpad_dir))
    if state.menu and not (prev is not None and prev.menu):
        new.append((seq + len(new) + 1, "menu_click"))
    if state.grip and not (prev is not None and prev.grip):
        new.append((seq + len(new) + 1, "grip_click"))
    if not new and prev is None:
        return state  # first observation, nothing pressed: adopt as is
    edges = (*history, *new)[-EDGE_HISTORY:]
    return replace(state, edges=edges, trackpad_dir=trackpad_dir)


def active_inputs(state: ControllerState) -> frozenset[ControllerInput]:
    """Which bindable inputs (``ControllerInput``) are active in ``state``: the
    trigger click, the trackpad's press-edge classification while it is
    clicked, ``menu_click`` while menu is down and ``grip_click`` while grip is
    down."""
    out: set[ControllerInput] = set()
    if state.trigger_pressed:
        out.add("trigger_click")
    if state.trackpad_click and state.trackpad_dir is not None:
        out.add(state.trackpad_dir)
    if state.menu:
        out.add("menu_click")
    if state.grip:
        out.add("grip_click")
    return frozenset(out)


def derive_held_codes(state: ControllerState | None, cfg: TrackerConfig) -> frozenset[str]:
    """Key codes the controller currently stands in for (13-tracker §1.1):
    ``cfg.controller_map`` binds each held action (clutch, gripper_open,
    gripper_close, rail_neg, rail_pos) to an input; ``none`` = unbound."""
    if state is None:
        return frozenset()
    active = active_inputs(state)
    m = cfg.controller_map
    return frozenset(
        code
        for action, code in HELD_ACTION_CODES.items()
        if (src := getattr(m, action)) != "none" and src in active
    )


def _discrete_action_for(edge_input: ControllerInput | None, cfg: TrackerConfig) -> str | None:
    """ActionName bound to ``edge_input`` through the discrete bindings, or
    ``None`` (unbound / held binding / deadzone click)."""
    if edge_input is None:
        return None
    m = cfg.controller_map
    for name, action in DISCRETE_ACTIONS.items():
        if getattr(m, name) == edge_input:
            return action
    return None


def derive_click_actions(
    state: ControllerState | None, cfg: TrackerConfig
) -> tuple[tuple[int, str], ...]:
    """``(seq, ActionName)`` for every remembered press edge (``state.edges``)
    whose input is bound to a discrete action (``switch_arm`` /
    ``switch_arm_prev``), oldest first. The loop fires the entries newer than
    the ``edge_seq`` it saw last, so no bound edge is lost when several buttons
    edge inside one tick (13-tracker §1.1)."""
    if state is None:
        return ()
    out = []
    for seq, edge_input in state.edges:
        action = _discrete_action_for(edge_input, cfg)
        if action is not None:
            out.append((seq, action))
    return tuple(out)


def derive_click_action(state: ControllerState | None, cfg: TrackerConfig) -> str | None:
    """ActionName bound to the input of the NEWEST press edge
    (``state.edge_input``), or ``None``; see :func:`derive_click_actions`."""
    return None if state is None else _discrete_action_for(state.edge_input, cfg)


@dataclass(frozen=True)
class TrackerSample:
    """One tracker pose in the lighthouse world (13-tracker §4)."""

    pose: Pose  # m + wxyz
    vel_lin: np.ndarray  # m/s, world
    vel_ang: np.ndarray  # rad/s axis-angle, world
    t_dev: float  # libsurvive run time, s (NOT wall clock)
    rx_mono: float  # time.monotonic() this SAMPLE was published (held-codes heartbeat)
    seq: int
    valid: bool = True  # False: jump > max_jump_m vs the previous pose, or pose older than stale_s
    controller: ControllerState | None = None  # None: backend reports no controller
    held_codes: frozenset[str] = frozenset()  # device-held codes (13-tracker §1.1)
    # (edge_seq, action) of every remembered bound press edge, oldest first
    # (``derive_click_actions``); the loop fires those newer than its last seen edge_seq.
    click_actions: tuple[tuple[int, str], ...] = ()
    # time.monotonic() the POSE was received: == rx_mono for a pose event, the older
    # pose's time for a controller-edge re-publish. The clutch's staleness feed
    # (13-tracker §4): a button edge never makes an old pose look fresh.
    pose_rx_mono: float = field(kw_only=True)


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
class LighthouseSnapshot:
    """One libsurvive LIGHTHOUSE object as last enumerated by the reader thread
    (13-tracker §4 "Calibration modes"). ``pose`` is the lighthouse-world pose
    (m, wxyz) or ``None`` while unsolved (libsurvive reports an all-zero
    quaternion then); ``serial`` is ``simple_serial_number`` (``None`` when the
    build does not expose it or OOTX has not decoded yet)."""

    index: int  # trailing number of the object name ("LH2" -> 2), else enumeration order
    name: str
    serial: str | None
    pose: Pose | None
    t_mono: float  # time.monotonic() of the enumeration


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
    charging: bool | None = None  # controller on external (USB) power; None = not reported


class TrackerReader:
    """Owns the device thread for the process lifetime; publishes to ``slot``.

    ``start()`` is a no-op for backend ``none`` and while a thread is alive; a
    finished thread clears ``_thread`` so ``start()`` can be called again.
    ``status(now)`` derives ``age_s`` / ``stale`` from the receive time of the
    last REAL pose (``cfg.stale_s``) and a ``rate_hz`` over the last
    ``RATE_WINDOW_S`` of real poses that decays to 0 when they stop;
    everything else is set by the backend thread under ``_lock``.

    Controller inputs (13-tracker §1.1): the libsurvive backend folds button
    events into a :class:`ControllerState` (press edges noted by
    :func:`note_edges`); every published sample carries the newest state plus
    the derived ``held_codes`` / ``click_actions``, and a button edge
    re-publishes the last pose immediately so the loop sees the edge within
    one tick — without refreshing the pose's age (see :meth:`_publish`). The
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
        self._click_actions: tuple[tuple[int, str], ...] = ()
        self._lock = threading.Lock()
        self._status: TrackerStatus = "no_backend"
        self._detail = "tracker disabled (backend: none)" if cfg.backend == "none" else ""
        self._seq = 0
        self._last: TrackerSample | None = None
        self._last_pose_rx: float | None = None  # receive time of the last REAL pose (age feed)
        self._pose_valid = False  # validity (jump check) of the last REAL pose
        self._rx_times: deque[float] = deque(maxlen=64)  # real poses only (rate feed)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # start()/stop()/restart() are serialised: a restart must never interleave
        # with another start (two libsurvive contexts = LIBUSB_ERROR_BUSY).
        self._lifecycle = threading.RLock()
        self._last_error_log = ""  # newest libsurvive error/warn line
        self._log_cb = None  # keep the ctypes callback alive for the ctx lifetime
        self._log_last: dict[str, float] = {}  # message class -> last forwarded time
        self._log_suppressed: dict[str, int] = {}
        self.bad_events = 0  # dropped libsurvive events (guard)
        self.restarts = 0  # libsurvive loop auto-restarts
        self._charging: bool | None = None  # controller USB-power flag (simple_object_charging)
        self._charging_next = 0.0  # next monotonic time to re-poll charging
        # Calibration hooks: libsurvive INFO lines (ANSI stripped) newest last, the
        # optional per-line callback (runs on libsurvive's C thread; exceptions are
        # swallowed) and the lighthouse snapshot refreshed by the reader thread.
        self.info_lines: deque[tuple[float, str]] = deque(maxlen=INFO_LINES_MAX)
        self.on_info: Callable[[float, str], None] | None = None
        self._lighthouses: list[LighthouseSnapshot] = []

    @property
    def backend(self) -> str:
        return self.cfg.backend

    # -- lifecycle -----------------------------------------------------------------
    def start(self) -> None:
        """Spawn the backend thread; a no-op for backend ``none`` and while a
        thread is still alive (including one whose ``simple_close`` is pending
        after a timed-out :meth:`stop` — it keeps its handle so no second
        libsurvive context is ever opened)."""
        with self._lifecycle:
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
            # Only the thread that owns the handle clears it: a lingering thread
            # must never orphan a newer one (start() may spawn a fresh thread).
            if self._thread is threading.current_thread():
                self._thread = None

    def stop(self, timeout: float = STOP_JOIN_TIMEOUT_S) -> bool:
        """Signal the backend thread and join it for up to ``timeout`` s.

        Returns True once the thread is gone — for libsurvive that means
        ``simple_close`` ran and the dongle is free. On a timeout the handle is
        KEPT (False): the thread is still inside the old libsurvive context, so
        a later :meth:`start`/:meth:`restart` stays a no-op / refuses instead of
        opening a second context (LIBUSB_ERROR_BUSY, 13-tracker §6); the
        thread clears the handle itself when it finally exits."""
        with self._lifecycle:
            self._stop.set()
            thread = self._thread
            if thread is None:
                return True
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning(
                    "tracker reader thread did not stop within %.1f s (libsurvive close pending)",
                    timeout,
                )
                return False
            self._thread = None
            return True

    def restart(
        self, libsurvive_args: list[str], *, timeout: float = RESTART_JOIN_TIMEOUT_S
    ) -> None:
        """Stop the backend thread (joined: ``simple_close`` must release the
        dongle before the next ``simple_init``, else LIBUSB_ERROR_BUSY), swap
        the libsurvive arguments on a per-reader COPY of the config (the
        ``TrackerConfig`` handed in is shared with the Runtime / SessionManager
        and is never mutated) and start again. ``status()`` goes through
        ``starting`` / ``searching`` meanwhile; the control loop holds on the
        aged-out sample. The lighthouse snapshot is cleared (it belongs to the
        old libsurvive context).

        Raises ``RuntimeError`` (status ``error``) when the old thread is still
        alive after ``timeout`` s: nothing is restarted, the arguments stay as
        they were, and the old thread keeps owning the handle until its close
        completes (the calibration FSM reports the failure; a later restart
        recovers once the close has finished)."""
        with self._lifecycle:
            if not self.stop(timeout=timeout):
                detail = (
                    f"tracker reader did not stop within {timeout:.0f} s "
                    "(libsurvive close pending; not restarted)"
                )
                self._set_status("error", detail)
                raise RuntimeError(detail)
            self.cfg = self.cfg.model_copy(update={"libsurvive_args": list(libsurvive_args)})
            with self._lock:
                self._lighthouses = []
            self.start()

    def lighthouses(self) -> list[LighthouseSnapshot]:
        """Newest LIGHTHOUSE-object snapshot (refreshed by the reader thread every
        ``LIBSURVIVE_LIGHTHOUSE_POLL_S`` while the libsurvive loop runs; empty for
        the other backends and right after a (re)start)."""
        with self._lock:
            return list(self._lighthouses)

    # -- status ----------------------------------------------------------------------
    def _set_status(self, status: TrackerStatus, detail: str) -> None:
        with self._lock:
            self._status = status
            self._detail = detail

    def status(self, now: float | None = None) -> TrackerDeviceStatus:
        now = self._clock() if now is None else now
        with self._lock:
            status, detail, last = self._status, self._detail, self._last
            pose_rx = self._last_pose_rx
            rx = [t for t in self._rx_times if t > now - RATE_WINDOW_S]
            controller = self._controller
            charging = self._charging
        # Pose age: the last REAL pose (an edge re-publish carries an old pose).
        age = None if pose_rx is None else max(0.0, now - pose_rx)
        if status == "tracking" and age is not None and age > self.cfg.stale_s:
            status = "stale"
        # Held codes ride on the controller stream: fresh by the newest SAMPLE's age.
        codes_fresh = last is not None and now - last.rx_mono <= self.cfg.stale_s
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
            device_held=last.held_codes if codes_fresh else frozenset(),
            bad_events=self.bad_events,
            restarts=self.restarts,
            charging=charging,
        )

    # -- publishing (backend threads) ----------------------------------------------------
    def _publish(
        self, pose: Pose, vel_lin, vel_ang, t_dev: float, *, edge: bool = False
    ) -> TrackerSample:
        """Publish a sample carrying the current controller state / codes / actions.

        A pose event (``edge=False``) is checked for a jump against the previous
        pose (``valid``), refreshes the pose age (``_last_pose_rx``), feeds the
        rate window and sets the status to ``tracking``. A controller edge
        (``edge=True``, :meth:`_on_controller`) re-publishes the LAST pose so
        the loop sees the edge within one tick: the pose keeps its validity
        unless it is older than ``stale_s`` (then the sample is invalid so the
        clutch cannot anchor on it), and neither the pose age, the rate nor the
        status are touched — button edges are the heartbeat of the held codes,
        not of the pose, so a chain of edges can never keep a dead pose stream
        looking fresh.
        """
        rx = self._clock()
        with self._lock:
            prev = self._last
            self._seq += 1
            jump = stale_edge = False
            if edge and self._last_pose_rx is not None:
                pose_rx = self._last_pose_rx
                pose_fresh = rx - pose_rx <= self.cfg.stale_s
                valid = self._pose_valid and pose_fresh
                stale_edge = self._pose_valid and not pose_fresh
            else:  # a pose event (an edge before any pose is not re-published)
                pose_rx = rx
                jump = prev is not None and (
                    float(np.linalg.norm(pose.position - prev.pose.position)) > self.cfg.max_jump_m
                )
                valid = not jump
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
                click_actions=self._click_actions,
                pose_rx_mono=pose_rx,
            )
            self._last = sample
            if not edge:
                self._last_pose_rx = rx
                self._pose_valid = valid
                self._rx_times.append(rx)
                self._status = "tracking"
                self._detail = "" if valid else f"jump > {self.cfg.max_jump_m} m: sample invalid"
            elif stale_edge:
                self._detail = (
                    f"controller edge on a pose older than {self.cfg.stale_s} s: sample invalid"
                )
        if jump:
            logger.warning(
                "tracker sample %d invalid (jump > %.2f m)", sample.seq, self.cfg.max_jump_m
            )
        elif stale_edge:
            logger.debug(
                "tracker sample %d re-published on a controller edge with a pose older than "
                "%.2f s: invalid", sample.seq, self.cfg.stale_s,
            )
        self.slot.put(sample)
        return sample

    # -- controller inputs (13-tracker §1.1) ----------------------------------------------
    def _on_controller(self, state: ControllerState | None) -> bool:
        """Adopt a new controller state (press edges noted and the trackpad
        click classified at its edge, :func:`note_edges`); on a button edge (or
        a change of the derived codes) re-publish the last pose so the loop
        sees the edge now.

        The re-published sample keeps the pose's own validity, but a pose older
        than ``stale_s`` (measured from the last REAL pose, never from an
        earlier re-publish) is flagged invalid: the codes still count (gripper,
        clutch *held*), while the clutch cannot anchor on a stale pose.
        Returns True when a sample was re-published.
        """
        with self._lock:
            prev, prev_codes, last = self._controller, self._held_codes, self._last
        if state is not None:
            state = note_edges(prev, state, self.cfg.trackpad_deadzone)
        codes = derive_held_codes(state, self.cfg)
        with self._lock:
            self._controller = state
            self._held_codes = codes
            self._click_actions = derive_click_actions(state, self.cfg)
        edge = codes != prev_codes or (
            (state.buttons() if state is not None else None)
            != (prev.buttons() if prev is not None else None)
        )
        if not edge or last is None:
            return False
        self._publish(last.pose, last.vel_lin, last.vel_ang, last.t_dev, edge=True)
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
            else:  # SURVIVE_LOG_LEVEL_INFO and up: keep for the calibration FSM
                logger.debug("libsurvive: %s", text)
                text = _ANSI_RE.sub("", text).strip()
                now = self._clock()
                with self._lock:
                    self.info_lines.append((now, text))
                cb = self.on_info
                if cb is not None:
                    try:
                        cb(now, text)
                    except Exception:
                        logger.debug("on_info callback failed", exc_info=True)
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
        next_lh = t_start  # first lighthouse snapshot right away, then every poll period
        while not self._stop.is_set():
            et = ps.simple_next_event(ptr, ctypes.byref(ev))
            if et == ps.SurviveSimpleEventType_Shutdown:
                if not self._stop.is_set():
                    self._set_status("error", self._error_detail("libsurvive shut down"))
                return
            if (now := self._clock()) >= next_lh:
                next_lh = now + LIBSURVIVE_LIGHTHOUSE_POLL_S
                self._refresh_lighthouses(ps, ptr, ctypes, now)
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
                        self._poll_charging(ps, pe.object)
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

    def _poll_charging(self, ps, obj) -> None:
        """Throttled read of the controller's USB-power (charging) flag.

        The only battery-related datum this libsurvive build exposes through
        the simple API: a bool, whether the controller is on external power.
        The charge *level* lives on the opaque ``SurviveObject`` behind the
        simple handle and ``simple_object_charge_percet`` is not compiled in,
        so a percentage is not available here. Polled at most every
        ``CHARGE_POLL_S`` and guarded: a failure keeps the last value and
        never kills the reader thread."""
        now = self._clock()
        if now < self._charging_next:
            return
        self._charging_next = now + CHARGE_POLL_S
        try:
            charging = bool(ps.simple_object_charging(obj))
        except Exception:
            self._forward_warning("simple_object_charging read failed: dropped")
            return
        with self._lock:
            self._charging = charging

    def _refresh_lighthouses(self, ps, ptr, ctypes, now: float) -> None:
        """Enumerate libsurvive's LIGHTHOUSE objects (``ps.<constant>`` comparison
        only: the test stub's enum values differ from the real library) into the
        lock-protected snapshot. Every accessor is guarded so a missing symbol or
        a half-initialised station degrades to ``None``, never kills the thread."""
        try:
            out: list[LighthouseSnapshot] = []
            obj = ps.simple_get_first_object(ptr)
            order = 0
            while obj:
                if ps.simple_object_get_type(obj) == ps.SurviveSimpleObject_LIGHTHOUSE:
                    name = ps.simple_object_name(obj)
                    name = name.decode(errors="replace") if isinstance(name, bytes) else str(name)
                    m = re.search(r"(\d+)$", name)
                    index = int(m.group(1)) if m else order
                    serial: str | None = None
                    try:
                        raw = ps.simple_serial_number(obj)
                        if raw:
                            serial = (
                                raw.decode(errors="replace") if isinstance(raw, bytes) else str(raw)
                            )
                    except Exception:
                        serial = None
                    pose: Pose | None = None
                    try:
                        lp = ps.SurvivePose()
                        ps.simple_object_get_latest_pose(obj, ctypes.byref(lp))
                        pos = np.array(lp.Pos[:3], dtype=np.float64)
                        rot = np.array(lp.Rot[:4], dtype=np.float64)  # wxyz
                        if (
                            np.all(np.isfinite(pos)) and np.all(np.isfinite(rot))
                            and abs(float(np.linalg.norm(rot)) - 1.0) <= QUAT_NORM_TOL
                        ):
                            pose = Pose(pos, rot)
                    except Exception:
                        pose = None
                    out.append(LighthouseSnapshot(index, name, serial, pose, now))
                    order += 1
                obj = ps.simple_get_next_object(ptr, obj)
            with self._lock:
                self._lighthouses = out
        except Exception:
            logger.debug("lighthouse enumeration failed", exc_info=True)

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
    "EDGE_HISTORY",
    "GRIPPER_CLOSE_CODE",
    "GRIPPER_OPEN_CODE",
    "HELD_ACTION_CODES",
    "RAIL_NEG_CODE",
    "RAIL_POS_CODE",
    "ControllerState",
    "LighthouseSnapshot",
    "TrackerDeviceStatus",
    "TrackerReader",
    "TrackerSample",
    "TrackerSettings",
    "TrackerSettingsValues",
    "TrackerStatus",
    "TrackpadDir",
    "active_inputs",
    "apply_button_event",
    "classify_trackpad",
    "derive_click_action",
    "derive_click_actions",
    "derive_held_codes",
    "note_edges",
]
