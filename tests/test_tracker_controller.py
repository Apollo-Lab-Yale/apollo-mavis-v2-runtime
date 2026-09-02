"""Vive-controller inputs in the tracker reader (13-tracker §1.1): libsurvive
button/touch/axis event parsing with synthetic pysurvive-like structs (no
pysurvive import), ``held_codes`` derivation (map, deadzone, ``none``), edge
re-publish, the fake backend's scripted controller hook and the libsurvive
event loop driven by a stub ``ps`` module."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest
from apollo_xarm7_core import LatestSlot, Pose
from apollo_xarm7_core.protocol import KEYMAP

from apollo_xarm7_runtime.config import ControllerMapConfig, TrackerConfig
from apollo_xarm7_runtime.devices.tracker import (
    AXIS_TRACKPAD_X,
    AXIS_TRACKPAD_Y,
    AXIS_TRIGGER,
    BUTTON_GRIP,
    BUTTON_MENU,
    BUTTON_SYSTEM,
    BUTTON_TRACKPAD,
    BUTTON_TRIGGER,
    CLUTCH_CODE,
    EVENT_AXIS_CHANGED,
    EVENT_BUTTON_DOWN,
    EVENT_BUTTON_UP,
    EVENT_TOUCH_DOWN,
    EVENT_TOUCH_UP,
    GRIPPER_CLOSE_CODE,
    GRIPPER_OPEN_CODE,
    ControllerState,
    TrackerReader,
    active_inputs,
    apply_button_event,
    derive_held_codes,
)

IDENT = np.array([1.0, 0.0, 0.0, 0.0])


def _code(action: str) -> str:
    return next(e.code for e in KEYMAP if e.action == action)


def test_injected_codes_come_from_the_keymap_not_letters():
    assert CLUTCH_CODE == _code("tracker_clutch")
    assert GRIPPER_OPEN_CODE == _code("gripper_open")
    assert GRIPPER_CLOSE_CODE == _code("gripper_close")
    assert len({CLUTCH_CODE, GRIPPER_OPEN_CODE, GRIPPER_CLOSE_CODE}) == 3


# -- event parsing ------------------------------------------------------------------------
def _ev(state, et, bid, axes=(), t=1.0):
    ids = [a for a, _ in axes]
    vals = [v for _, v in axes]
    return apply_button_event(state, et, bid, ids, vals, t)


def test_button_touch_and_axis_events_fold_into_state():
    s = ControllerState()
    s = _ev(s, EVENT_AXIS_CHANGED, 255, [(AXIS_TRIGGER, 0.4)])
    assert s.trigger == pytest.approx(0.4) and not s.trigger_pressed and s.rx_mono == 1.0
    s = _ev(s, EVENT_BUTTON_DOWN, BUTTON_TRIGGER, [(AXIS_TRIGGER, 1.0)], t=2.0)
    assert s.trigger_pressed and s.trigger == 1.0 and s.rx_mono == 2.0
    s = _ev(s, EVENT_BUTTON_UP, BUTTON_TRIGGER, [(AXIS_TRIGGER, 0.0)])
    assert not s.trigger_pressed and s.trigger == 0.0
    s = _ev(s, EVENT_TOUCH_DOWN, BUTTON_TRACKPAD, [(AXIS_TRACKPAD_X, -0.2), (AXIS_TRACKPAD_Y, 0.8)])
    assert s.trackpad_touch and not s.trackpad_click
    assert (s.trackpad_x, s.trackpad_y) == (pytest.approx(-0.2), pytest.approx(0.8))
    s = _ev(s, EVENT_BUTTON_DOWN, BUTTON_TRACKPAD)
    assert s.trackpad_click and s.trackpad_touch
    s = _ev(s, EVENT_BUTTON_UP, BUTTON_TRACKPAD)
    s = _ev(s, EVENT_TOUCH_UP, BUTTON_TRACKPAD)
    assert not s.trackpad_click and not s.trackpad_touch
    assert s.trackpad_y == pytest.approx(0.8)  # axes keep their last value
    for bid, field in ((BUTTON_GRIP, "grip"), (BUTTON_MENU, "menu"), (BUTTON_SYSTEM, "system")):
        down = _ev(s, EVENT_BUTTON_DOWN, bid)
        assert getattr(down, field) is True and down.buttons() != s.buttons()
        assert getattr(_ev(down, EVENT_BUTTON_UP, bid), field) is False
    # Unknown button / axis ids and touch events on non-trackpad buttons are ignored.
    same = _ev(s, EVENT_BUTTON_DOWN, 9, [(5, 0.7)])
    assert same == ControllerState(**{**s.__dict__, "rx_mono": same.rx_mono})
    assert _ev(s, EVENT_TOUCH_DOWN, BUTTON_TRIGGER).trackpad_touch is False


# -- held_codes derivation ------------------------------------------------------------------
def _pad(y: float, click: bool = True, touch: bool = True) -> ControllerState:
    return ControllerState(trackpad_touch=touch, trackpad_click=click, trackpad_y=y)


def test_derive_held_codes_defaults_and_deadzone():
    cfg = TrackerConfig()
    assert derive_held_codes(None, cfg) == frozenset()
    assert derive_held_codes(ControllerState(), cfg) == frozenset()
    assert derive_held_codes(ControllerState(trigger_pressed=True), cfg) == {CLUTCH_CODE}
    assert derive_held_codes(ControllerState(trigger=0.9), cfg) == frozenset()  # analog only
    assert derive_held_codes(_pad(0.9), cfg) == {GRIPPER_OPEN_CODE}
    assert derive_held_codes(_pad(-0.9), cfg) == {GRIPPER_CLOSE_CODE}
    for y in (0.3, -0.3, 0.29, -0.1, 0.0):  # |y| <= deadzone: neither
        assert derive_held_codes(_pad(y), cfg) == frozenset(), y
    assert derive_held_codes(_pad(0.9, click=False), cfg) == frozenset()  # touch w/o click
    assert derive_held_codes(ControllerState(grip=True, menu=True, system=True), cfg) == set()
    both = ControllerState(trigger_pressed=True, trackpad_click=True, trackpad_y=-0.5)
    assert derive_held_codes(both, cfg) == {CLUTCH_CODE, GRIPPER_CLOSE_CODE}
    assert active_inputs(both, 0.3) == {"trigger_click", "trackpad_down"}
    assert derive_held_codes(_pad(0.4), TrackerConfig(trackpad_deadzone=0.5)) == frozenset()


def test_derive_held_codes_honours_map_and_none():
    cfg = TrackerConfig(
        controller_map=ControllerMapConfig(
            clutch="trackpad_up", gripper_open="none", gripper_close="trigger_click"
        )
    )
    assert derive_held_codes(_pad(0.9), cfg) == {CLUTCH_CODE}
    assert derive_held_codes(ControllerState(trigger_pressed=True), cfg) == {GRIPPER_CLOSE_CODE}
    assert derive_held_codes(_pad(-0.9), cfg) == frozenset()
    off = TrackerConfig(controller_map=ControllerMapConfig(
        clutch="none", gripper_open="none", gripper_close="none"))
    assert derive_held_codes(both := ControllerState(trigger_pressed=True, trackpad_click=True,
                                                     trackpad_y=1.0), off) == frozenset()
    assert derive_held_codes(both, TrackerConfig()) == {CLUTCH_CODE, GRIPPER_OPEN_CODE}
    with pytest.raises(ValueError):
        ControllerMapConfig(clutch="grip")  # grip/menu/system are not bindable


# -- reader: state, edge re-publish, status ---------------------------------------------------
class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


def _reader(cfg=None, clock=None, **kw):
    slot: LatestSlot = LatestSlot()
    clock = clock or Clock()
    return TrackerReader(cfg or TrackerConfig(backend="fake"), slot, clock=clock, **kw), slot, clock


def test_button_edge_republishes_last_pose_with_codes():
    reader, slot, clock = _reader()
    pressed = ControllerState(trigger=1.0, trigger_pressed=True, rx_mono=clock.t)
    assert reader._on_controller(pressed) is False  # no pose yet: nothing to publish
    st = reader.status()
    assert st.controller == pressed and st.device_held == frozenset() and st.seq == 0
    reader._publish(Pose(np.array([0.1, 0.2, 0.3]), IDENT), np.zeros(3), np.zeros(3), 0.0)
    s1 = slot.get()[0]
    assert s1.controller == pressed and s1.held_codes == {CLUTCH_CODE} and s1.seq == 1
    assert reader.status().device_held == {CLUTCH_CODE}
    # Same buttons, analog change only: no edge, no re-publish.
    assert reader._on_controller(ControllerState(trigger=0.7, trigger_pressed=True)) is False
    assert slot.get()[0].seq == 1
    clock.t += 0.05
    released = ControllerState(trigger=0.0)
    assert reader._on_controller(released) is True
    s2 = slot.get()[0]
    assert s2.seq == 2 and s2.valid and s2.rx_mono == clock.t and s2.held_codes == frozenset()
    assert np.allclose(s2.pose.position, [0.1, 0.2, 0.3]) and s2.t_dev == 0.0
    assert s2.controller == released and reader.status().device_held == frozenset()
    # Trackpad touch alone is an edge (telemetry), but injects no code.
    assert reader._on_controller(ControllerState(trackpad_touch=True, trackpad_y=0.9)) is True
    assert slot.get()[0].held_codes == frozenset()
    # Pose older than stale_s at the edge: codes still ride along, pose flagged invalid.
    clock.t += 1.0
    assert reader._on_controller(_pad(-0.9)) is True
    s4 = slot.get()[0]
    assert s4.valid is False and s4.held_codes == {GRIPPER_CLOSE_CODE}
    assert reader.status().device_held == {GRIPPER_CLOSE_CODE}  # fresh by rx_mono
    assert reader.status(clock.t + 0.5).device_held == frozenset()  # stale -> nothing


def test_on_button_event_reads_pysurvive_like_struct():
    reader, slot, clock = _reader()
    reader._publish(Pose(np.zeros(3), IDENT), np.zeros(3), np.zeros(3), 0.0)
    be = SimpleNamespace(
        time=1.5, object=object(), event_type=EVENT_BUTTON_DOWN, button_id=BUTTON_TRIGGER,
        axis_count=1, axis_ids=[AXIS_TRIGGER, 0, 0, 0, 0, 0, 0, 0],
        axis_val=[1.0, 0, 0, 0, 0, 0, 0, 0],
    )
    reader._on_button_event(be)
    s = slot.get()[0]
    assert s.seq == 2 and s.controller.trigger_pressed and s.controller.trigger == 1.0
    assert s.controller.rx_mono == clock.t and s.held_codes == {CLUTCH_CODE}
    be2 = SimpleNamespace(
        time=1.6, object=object(), event_type=EVENT_AXIS_CHANGED, button_id=255,
        axis_count=99,  # bogus count is clamped to the 8-slot arrays
        axis_ids=[AXIS_TRACKPAD_X, AXIS_TRACKPAD_Y, 0, 0, 0, 0, 0, 0],
        axis_val=[0.25, -0.5, 0, 0, 0, 0, 0, 0],
    )
    reader._on_button_event(be2)
    assert slot.get()[0].seq == 2  # axis-only: state updated, no re-publish
    c = reader.status().controller
    assert (c.trackpad_x, c.trackpad_y) == (0.25, -0.5) and c.trigger_pressed


def test_fake_backend_scripted_controller_provider():
    script = {"state": None}
    slot: LatestSlot = LatestSlot()
    reader = TrackerReader(
        TrackerConfig(backend="fake"), slot, controller_provider=lambda: script["state"]
    )
    reader.start()
    try:
        deadline = time.monotonic() + 3.0
        while slot.get() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        s = slot.get()[0]
        assert s.controller is None and s.held_codes == frozenset()
        script["state"] = ControllerState(trigger=1.0, trigger_pressed=True)
        while time.monotonic() < deadline:
            s = slot.get()[0]
            if s.held_codes:
                break
            time.sleep(0.01)
        assert s.controller == script["state"] and s.held_codes == {CLUTCH_CODE}
        assert reader.status().device_held == {CLUTCH_CODE}
        script["state"] = _pad(0.9)
        while time.monotonic() < deadline and GRIPPER_OPEN_CODE not in slot.get()[0].held_codes:
            time.sleep(0.01)
        assert slot.get()[0].held_codes == {GRIPPER_OPEN_CODE}
        script["state"] = None
        while time.monotonic() < deadline and slot.get()[0].controller is not None:
            time.sleep(0.01)
        assert slot.get()[0].controller is None and slot.get()[0].held_codes == frozenset()
    finally:
        reader.stop()


# -- libsurvive event loop with a stub ``ps`` module ----------------------------------------------
class StubPS:
    """The subset of ``pysurvive`` the reader touches, driven by a scripted event list."""

    SurviveSimpleEventType_ButtonEvent = 1
    SurviveSimpleEventType_PoseUpdateEvent = 3
    SurviveSimpleEventType_Shutdown = 5
    SurviveSimpleObject_OBJECT = 1
    SurviveSimpleObject_LIGHTHOUSE = 2

    def __init__(self, events):
        self.events = list(events)

    def SurviveSimpleEvent(self):  # noqa: N802 - mirrors the C API name
        return SimpleNamespace(d=None)

    def simple_wait_for_event(self, ptr, ev):
        if not self.events:
            return self.SurviveSimpleEventType_Shutdown
        et, payload = self.events.pop(0)
        key = "__private_button_event" if et == 1 else "__private_pose_event"
        ev.d = SimpleNamespace(**{key: payload})
        return et

    def simple_object_get_type(self, obj):
        return obj.kind

    def simple_object_name(self, obj):
        return obj.name.encode()

    def simple_get_object_count(self, ptr):
        return 2


def _obj(name, kind=StubPS.SurviveSimpleObject_OBJECT):
    return SimpleNamespace(name=name, kind=kind)


def _pose(obj, pos, t):
    return SimpleNamespace(
        object=obj, time=t, pose=SimpleNamespace(Pos=list(pos), Rot=[1.0, 0.0, 0.0, 0.0]),
        velocity=SimpleNamespace(Pos=[0.0, 0.0, 0.0], AxisAngleRot=[0.0, 0.0, 0.0]),
    )


def _button(obj, et, bid, axes=()):
    ids = [a for a, _ in axes] + [0] * (8 - len(axes))
    vals = [v for _, v in axes] + [0.0] * (8 - len(axes))
    return SimpleNamespace(
        time=0.0, object=obj, event_type=et, button_id=bid, axis_count=len(axes),
        axis_ids=ids, axis_val=vals,
    )


def test_libsurvive_event_loop_routes_pose_and_button_events_by_object():
    wm0, wm1, lh = _obj("WM0"), _obj("WM1"), _obj("LH0", StubPS.SurviveSimpleObject_LIGHTHOUSE)
    ps = StubPS([
        (1, _button(wm0, EVENT_BUTTON_DOWN, BUTTON_TRIGGER)),  # before any pose: state only
        (3, _pose(lh, [9.0, 9.0, 9.0], 0.1)),  # lighthouse: ignored
        (3, _pose(wm1, [5.0, 0.0, 0.0], 0.2)),  # other tracker: ignored
        (3, _pose(wm0, [0.1, 0.0, 0.0], 0.3)),  # seq 1, carries trigger pressed
        (1, _button(wm1, EVENT_BUTTON_DOWN, BUTTON_TRACKPAD, [(AXIS_TRACKPAD_Y, 1.0)])),  # ignored
        (1, _button(wm0, EVENT_BUTTON_UP, BUTTON_TRIGGER)),  # edge: seq 2 re-publish
        (1, _button(wm0, EVENT_AXIS_CHANGED, 255, [(AXIS_TRACKPAD_Y, -0.9)])),  # no edge
        (1, _button(wm0, EVENT_BUTTON_DOWN, BUTTON_TRACKPAD)),  # edge: KeyF, seq 3
        (3, _pose(wm0, [0.11, 0.0, 0.0], 0.4)),  # seq 4 keeps KeyF
    ])
    reader, slot, clock = _reader(TrackerConfig(backend="libsurvive"))
    seen = []
    orig = reader._publish

    def spy(*a, **kw):
        s = orig(*a, **kw)
        seen.append(s)
        return s

    reader._publish = spy
    reader._libsurvive_events(ps, object(), SimpleNamespace(byref=lambda x: x))
    assert [s.seq for s in seen] == [1, 2, 3, 4]
    assert seen[0].controller.trigger_pressed and seen[0].held_codes == {CLUTCH_CODE}
    assert np.allclose(seen[0].pose.position, [0.1, 0.0, 0.0])
    assert not seen[1].controller.trigger_pressed and seen[1].held_codes == frozenset()
    assert np.allclose(seen[1].pose.position, [0.1, 0.0, 0.0]) and seen[1].t_dev == 0.3
    assert seen[2].controller.trackpad_click and seen[2].held_codes == {GRIPPER_CLOSE_CODE}
    assert seen[3].held_codes == {GRIPPER_CLOSE_CODE} and seen[3].t_dev == 0.4
    assert np.allclose(seen[3].pose.position, [0.11, 0.0, 0.0]) and seen[3].valid
    assert reader.status().status == "error" and "shut down" in reader.status().detail
    assert reader.status().controller.trackpad_y == pytest.approx(-0.9)
