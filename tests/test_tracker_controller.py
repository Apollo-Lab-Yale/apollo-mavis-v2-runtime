"""Vive-controller inputs in the tracker reader (13-tracker §1.1): libsurvive
button/touch/axis event parsing with synthetic pysurvive-like structs (no
pysurvive import), trackpad click classification (dominant axis, deadzone,
fixed at the press edge), ``held_codes`` / ``click_action`` derivation (map,
``none``, validation), edge re-publish, the fake backend's scripted controller
hook, and the libsurvive event loop driven by a stub ``ps`` module incl. the
robustness rules (bad events, error-vs-searching status, rate decay, log rate
limit, auto-restart)."""

from __future__ import annotations

import ctypes
import logging
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
from apollo_xarm7_core import LatestSlot, Pose
from apollo_xarm7_core.protocol import KEYMAP

import apollo_xarm7_runtime.devices.tracker as tracker_mod
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
    classify_click,
    classify_trackpad,
    derive_click_action,
    derive_held_codes,
)

IDENT = np.array([1.0, 0.0, 0.0, 0.0])
DZ = 0.3


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
    assert s.click_seq == 0 and s.click_dir is None  # raw: classify_click adds these
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


# -- trackpad click classification ------------------------------------------------------------
def _pad(x: float = 0.0, y: float = 0.0, click: bool = True, trigger: bool = False):
    """Raw (unclassified) controller state with the pad at (x, y)."""
    return ControllerState(
        trigger=1.0 if trigger else 0.0, trigger_pressed=trigger,
        trackpad_touch=True, trackpad_click=click, trackpad_x=x, trackpad_y=y,
    )


def _click(x: float, y: float, prev=None, **kw) -> ControllerState:
    """Classified state after a press edge at (x, y)."""
    return classify_click(prev, _pad(x, y, **kw), DZ)


@pytest.mark.parametrize(
    ("x", "y", "expect"),
    [
        (0.9, 0.0, "trackpad_right"), (-0.9, 0.0, "trackpad_left"),
        (0.0, 0.9, "trackpad_up"), (0.0, -0.9, "trackpad_down"),
        (0.6, 0.5, "trackpad_right"), (-0.6, 0.5, "trackpad_left"),  # x dominant
        (0.5, 0.6, "trackpad_up"), (-0.5, -0.6, "trackpad_down"),  # y dominant
        (0.5, -0.5, "trackpad_right"),  # tie: x wins
        (0.31, 0.0, "trackpad_right"), (0.0, -0.31, "trackpad_down"),  # just outside
        (0.3, 0.3, None), (0.2, -0.2, None), (0.0, 0.0, None), (-0.3, 0.0, None),  # inside
    ],
)
def test_classify_trackpad_dominant_axis_and_deadzone(x, y, expect):
    assert classify_trackpad(x, y, DZ) == expect


def test_classify_trackpad_honours_custom_deadzone():
    assert classify_trackpad(0.4, 0.0, 0.5) is None
    assert classify_trackpad(0.4, 0.0, 0.3) == "trackpad_right"
    assert classify_trackpad(0.0, 0.9, 1.0) is None  # deadzone 1.0: every click ignored


def test_click_classified_once_at_press_edge_and_held_until_release():
    s1 = _click(0.9, 0.0)  # press edge at the right
    assert s1.click_seq == 1 and s1.click_dir == "trackpad_right"
    assert active_inputs(s1) == {"trackpad_right"}
    s2 = classify_click(s1, _pad(0.0, 0.9), DZ)  # finger slides to the top while clicked
    assert s2.click_seq == 1 and s2.click_dir == "trackpad_right"  # classification held
    assert active_inputs(s2) == {"trackpad_right"}
    s3 = classify_click(s2, _pad(0.0, 0.9, click=False), DZ)  # release
    assert not s3.trackpad_click and s3.click_seq == 1 and s3.click_dir == "trackpad_right"
    assert active_inputs(s3) == frozenset()  # released: nothing held
    s4 = classify_click(s3, _pad(0.0, -0.9), DZ)  # new press at the bottom
    assert s4.click_seq == 2 and s4.click_dir == "trackpad_down"
    s5 = classify_click(s4, _pad(0.0, -0.9, click=False), DZ)
    s6 = classify_click(s5, _pad(0.1, 0.1), DZ)  # press inside the deadzone: ignored
    assert s6.click_seq == 3 and s6.click_dir is None and active_inputs(s6) == frozenset()
    s7 = classify_click(s6, _pad(0.9, 0.0), DZ)  # slides out afterwards: still ignored
    assert s7.click_seq == 3 and s7.click_dir is None and active_inputs(s7) == frozenset()
    # Trigger is independent of the pad; a raw (never classified) click holds nothing.
    assert active_inputs(_pad(0.9, 0.0, trigger=True)) == {"trigger_click"}
    assert active_inputs(classify_click(None, _pad(0.9, 0.0, trigger=True), DZ)) == {
        "trigger_click", "trackpad_right"
    }
    # A scripted state arriving with prev=None while already clicked counts as a press edge.
    assert classify_click(None, _pad(0.0, 0.9), DZ).click_seq == 1
    # Unclicked states carry the previous accounting and never classify.
    s8 = classify_click(s7, ControllerState(trigger_pressed=True), DZ)
    assert s8.click_seq == 3 and s8.click_dir is None


# -- held_codes / click_action derivation -----------------------------------------------------
def test_derive_held_codes_and_click_action_defaults():
    cfg = TrackerConfig()
    assert derive_held_codes(None, cfg) == frozenset()
    assert derive_held_codes(ControllerState(), cfg) == frozenset()
    assert derive_held_codes(ControllerState(trigger_pressed=True), cfg) == {CLUTCH_CODE}
    assert derive_held_codes(ControllerState(trigger=0.9), cfg) == frozenset()  # analog only
    assert derive_held_codes(_click(0.9, 0.0), cfg) == {GRIPPER_OPEN_CODE}  # right
    assert derive_held_codes(_click(-0.9, 0.0), cfg) == {GRIPPER_CLOSE_CODE}  # left
    assert derive_held_codes(_click(0.0, 0.9), cfg) == frozenset()  # up: discrete only
    assert derive_held_codes(_click(0.0, -0.9), cfg) == frozenset()  # down: discrete only
    assert derive_held_codes(_click(0.2, 0.2), cfg) == frozenset()  # deadzone
    assert derive_held_codes(_pad(0.9, 0.0), cfg) == frozenset()  # raw, never classified
    assert derive_held_codes(_pad(0.9, 0.0, click=False), cfg) == frozenset()  # touch only
    assert derive_held_codes(ControllerState(grip=True, menu=True, system=True), cfg) == set()
    both = _click(-0.9, 0.0, trigger=True)
    assert derive_held_codes(both, cfg) == {CLUTCH_CODE, GRIPPER_CLOSE_CODE}
    assert derive_click_action(_click(0.0, 0.9), cfg) == "switch_arm"
    assert derive_click_action(_click(0.0, -0.9), cfg) == "switch_arm_prev"
    assert derive_click_action(_click(0.9, 0.0), cfg) is None  # held binding, not discrete
    assert derive_click_action(_click(0.2, 0.2), cfg) is None
    assert derive_click_action(None, cfg) is None
    # The classification (and thus the action) persists after release for edge accounting.
    released = classify_click(_click(0.0, 0.9), _pad(0.0, 0.9, click=False), DZ)
    assert derive_click_action(released, cfg) == "switch_arm"
    assert derive_held_codes(released, cfg) == frozenset()
    dz = TrackerConfig(trackpad_deadzone=0.5)
    assert derive_held_codes(classify_click(None, _pad(0.4, 0.0), 0.5), dz) == frozenset()


def test_derive_honours_map_none_and_validation():
    cfg = TrackerConfig(
        controller_map=ControllerMapConfig(
            clutch="trackpad_left", gripper_open="none", gripper_close="trigger_click",
            arm_next="trackpad_right", arm_prev="none",
        )
    )
    assert derive_held_codes(_click(-0.9, 0.0), cfg) == {CLUTCH_CODE}
    assert derive_held_codes(ControllerState(trigger_pressed=True), cfg) == {GRIPPER_CLOSE_CODE}
    assert derive_held_codes(_click(0.0, 0.9), cfg) == frozenset()
    assert derive_click_action(_click(0.9, 0.0), cfg) == "switch_arm"
    assert derive_click_action(_click(0.0, -0.9), cfg) is None
    off = TrackerConfig(controller_map=ControllerMapConfig(
        clutch="none", gripper_open="none", gripper_close="none", arm_next="none", arm_prev="none"
    ))
    assert derive_held_codes(_click(0.9, 0.0, trigger=True), off) == frozenset()
    assert derive_click_action(_click(0.0, 0.9), off) is None
    with pytest.raises(ValueError):
        ControllerMapConfig(clutch="grip")  # grip/menu/system are not bindable
    with pytest.raises(ValueError, match="trackpad"):
        ControllerMapConfig(clutch="none", arm_next="trigger_click")  # discrete: trackpad only
    with pytest.raises(ValueError, match="more than one"):
        ControllerMapConfig(gripper_open="trackpad_up")  # clashes with arm_next default
    with pytest.raises(ValueError, match="more than one"):
        ControllerMapConfig(gripper_close="trackpad_right")  # clashes with gripper_open


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
    assert s1.click_action is None
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
    assert reader._on_controller(_pad(0.9, 0.0, click=False)) is True
    assert slot.get()[0].held_codes == frozenset()
    # Trackpad click at the top: classified up -> discrete switch_arm, no held code.
    assert reader._on_controller(_pad(0.0, 0.9)) is True
    s3 = slot.get()[0]
    assert s3.controller.click_seq == 1 and s3.controller.click_dir == "trackpad_up"
    assert s3.held_codes == frozenset() and s3.click_action == "switch_arm"
    assert reader._on_controller(_pad(0.0, 0.9, click=False)) is True  # release edge
    assert slot.get()[0].click_action == "switch_arm"  # persists for edge accounting
    # Pose older than stale_s at the edge: codes still ride along, pose flagged invalid.
    clock.t += 1.0
    assert reader._on_controller(_pad(-0.9, 0.0)) is True  # left -> gripper_close
    s5 = slot.get()[0]
    assert s5.valid is False and s5.held_codes == {GRIPPER_CLOSE_CODE}
    assert s5.controller.click_seq == 2 and s5.click_action is None
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
    # Trackpad press: classified from the axes in force (y dominant, negative -> down).
    be3 = SimpleNamespace(
        time=1.7, object=object(), event_type=EVENT_BUTTON_DOWN, button_id=BUTTON_TRACKPAD,
        axis_count=0, axis_ids=[0] * 8, axis_val=[0.0] * 8,
    )
    reader._on_button_event(be3)
    s3 = slot.get()[0]
    assert s3.seq == 3 and s3.controller.click_seq == 1
    assert s3.controller.click_dir == "trackpad_down" and s3.click_action == "switch_arm_prev"
    assert s3.held_codes == {CLUTCH_CODE}  # trigger still held; down injects no code


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
        script["state"] = _pad(0.9, 0.0)  # right -> gripper_open (classified by the reader)
        while time.monotonic() < deadline and GRIPPER_OPEN_CODE not in slot.get()[0].held_codes:
            time.sleep(0.01)
        s = slot.get()[0]
        assert s.held_codes == {GRIPPER_OPEN_CODE} and s.controller.click_seq == 1
        assert s.controller.click_dir == "trackpad_right" and s.click_action is None
        script["state"] = _pad(0.9, 0.0, click=False)  # release
        while time.monotonic() < deadline and slot.get()[0].held_codes:
            time.sleep(0.01)
        script["state"] = _pad(0.0, 0.9)  # up -> discrete switch_arm
        while time.monotonic() < deadline and slot.get()[0].click_action != "switch_arm":
            time.sleep(0.01)
        s = slot.get()[0]
        assert s.controller.click_seq == 2 and s.held_codes == frozenset()
        script["state"] = None
        while time.monotonic() < deadline and slot.get()[0].controller is not None:
            time.sleep(0.01)
        assert slot.get()[0].controller is None and slot.get()[0].held_codes == frozenset()
    finally:
        reader.stop()
    assert reader._thread is None


# -- libsurvive event loop with a stub ``ps`` module ----------------------------------------------
class StubPS:
    """The subset of ``pysurvive`` the reader touches, driven by a scripted event
    list; ``objects`` is what libsurvive would enumerate. ``on_poll`` runs on
    every ``simple_next_event`` (tests use it to advance the clock)."""

    SurviveSimpleEventType_None = 0
    SurviveSimpleEventType_ButtonEvent = 1
    SurviveSimpleEventType_PoseUpdateEvent = 3
    SurviveSimpleEventType_Shutdown = 5
    SurviveSimpleObject_OBJECT = 1
    SurviveSimpleObject_LIGHTHOUSE = 2

    def __init__(self, events, objects=(), on_poll=None):
        self.events = list(events)
        self.objects = list(objects)
        self.on_poll = on_poll

    def SurviveSimpleEvent(self):  # noqa: N802 - mirrors the C API name
        return SimpleNamespace(d=None)

    def simple_next_event(self, ptr, ev):
        if self.on_poll is not None:
            self.on_poll()
        if not self.events:
            return self.SurviveSimpleEventType_Shutdown
        et, payload = self.events.pop(0)
        if et == self.SurviveSimpleEventType_None:
            return et
        key = "__private_button_event" if et == 1 else "__private_pose_event"
        ev.d = SimpleNamespace(**{key: payload})
        return et

    def simple_object_get_type(self, obj):
        return obj.kind

    def simple_object_name(self, obj):
        return obj.name.encode()

    def simple_get_object_count(self, ptr):
        return len(self.objects)

    def simple_get_first_object(self, ptr):
        return self.objects[0] if self.objects else None

    def simple_get_next_object(self, ptr, obj):
        i = self.objects.index(obj) + 1
        return self.objects[i] if i < len(self.objects) else None


NONE_EV = (StubPS.SurviveSimpleEventType_None, None)
FAKE_CTYPES = SimpleNamespace(byref=lambda x: x)


def _obj(name, kind=StubPS.SurviveSimpleObject_OBJECT):
    return SimpleNamespace(name=name, kind=kind)


def _pose(obj, pos, t, rot=(1.0, 0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0)):
    return SimpleNamespace(
        object=obj, time=t, pose=SimpleNamespace(Pos=list(pos), Rot=list(rot)),
        velocity=SimpleNamespace(Pos=list(vel), AxisAngleRot=[0.0, 0.0, 0.0]),
    )


def _button(obj, et, bid, axes=()):
    ids = [a for a, _ in axes] + [0] * (8 - len(axes))
    vals = [v for _, v in axes] + [0.0] * (8 - len(axes))
    return SimpleNamespace(
        time=0.0, object=obj, event_type=et, button_id=bid, axis_count=len(axes),
        axis_ids=ids, axis_val=vals,
    )


def _spy_publish(reader):
    seen = []
    orig = reader._publish

    def spy(*a, **kw):
        s = orig(*a, **kw)
        seen.append(s)
        return s

    reader._publish = spy
    return seen


def test_libsurvive_event_loop_routes_pose_and_button_events_by_object():
    wm0, wm1, lh = _obj("WM0"), _obj("WM1"), _obj("LH0", StubPS.SurviveSimpleObject_LIGHTHOUSE)
    ps = StubPS([
        (1, _button(wm0, EVENT_BUTTON_DOWN, BUTTON_TRIGGER)),  # before any pose: state only
        (3, _pose(lh, [9.0, 9.0, 9.0], 0.1)),  # lighthouse: ignored
        (3, _pose(wm1, [5.0, 0.0, 0.0], 0.2)),  # other tracker: ignored
        (3, _pose(wm0, [0.1, 0.0, 0.0], 0.3)),  # seq 1, carries trigger pressed
        (1, _button(wm1, EVENT_BUTTON_DOWN, BUTTON_TRACKPAD, [(AXIS_TRACKPAD_Y, 1.0)])),  # ignored
        (1, _button(wm0, EVENT_BUTTON_UP, BUTTON_TRIGGER)),  # edge: seq 2 re-publish
        (1, _button(wm0, EVENT_AXIS_CHANGED, 255, [(AXIS_TRACKPAD_X, -0.9)])),  # no edge
        NONE_EV,  # idle poll: nothing happens
        (1, _button(wm0, EVENT_BUTTON_DOWN, BUTTON_TRACKPAD)),  # edge: left -> KeyF, seq 3
        (3, _pose(wm0, [0.11, 0.0, 0.0], 0.4)),  # seq 4 keeps KeyF
    ], objects=[lh, wm0, wm1])
    reader, slot, clock = _reader(TrackerConfig(backend="libsurvive"))
    seen = _spy_publish(reader)
    reader._libsurvive_events(ps, object(), FAKE_CTYPES)
    assert [s.seq for s in seen] == [1, 2, 3, 4]
    assert seen[0].controller.trigger_pressed and seen[0].held_codes == {CLUTCH_CODE}
    assert np.allclose(seen[0].pose.position, [0.1, 0.0, 0.0])
    assert not seen[1].controller.trigger_pressed and seen[1].held_codes == frozenset()
    assert np.allclose(seen[1].pose.position, [0.1, 0.0, 0.0]) and seen[1].t_dev == 0.3
    assert seen[2].controller.trackpad_click and seen[2].held_codes == {GRIPPER_CLOSE_CODE}
    assert seen[2].controller.click_dir == "trackpad_left" and seen[2].click_action is None
    assert seen[3].held_codes == {GRIPPER_CLOSE_CODE} and seen[3].t_dev == 0.4
    assert np.allclose(seen[3].pose.position, [0.11, 0.0, 0.0]) and seen[3].valid
    assert reader.status().status == "error" and "shut down" in reader.status().detail
    assert reader.status().controller.trackpad_x == pytest.approx(-0.9)
    assert reader.bad_events == 0


def test_libsurvive_loop_survives_bad_events(caplog):
    caplog.set_level(logging.WARNING, logger="apollo_xarm7_runtime.devices.tracker")
    wm0 = _obj("WM0")
    broken = SimpleNamespace(kind=StubPS.SurviveSimpleObject_OBJECT)  # no .name -> raises
    ps = StubPS([
        (3, _pose(wm0, [0.1, 0.0, 0.0], 0.1)),  # seq 1
        (3, _pose(wm0, [float("nan"), 0.0, 0.0], 0.2)),  # NaN position: dropped
        (3, _pose(wm0, [0.1, 0.0, 0.0], 0.3, rot=(0.5, 0.0, 0.0, 0.0))),  # non-unit: dropped
        (3, _pose(wm0, [0.1, float("inf"), 0.0], 0.4)),  # inf: dropped
        (3, _pose(broken, [0.1, 0.0, 0.0], 0.5)),  # handler raises: dropped
        (1, _button(broken, EVENT_BUTTON_DOWN, BUTTON_TRIGGER)),  # handler raises: dropped
        (3, _pose(wm0, [0.1, 0.0, 0.0], 0.6, rot=(0.0, 0.0, 0.0, 1.0),
                  vel=(float("nan"), 0.0, 0.0))),  # seq 2: NaN velocity sanitised to 0
        (3, _pose(wm0, [0.12, 0.0, 0.0], 0.7)),  # seq 3
    ], objects=[wm0])
    reader, slot, clock = _reader(TrackerConfig(backend="libsurvive"))
    seen = _spy_publish(reader)
    reader._libsurvive_events(ps, object(), FAKE_CTYPES)  # must return, never raise
    assert [s.seq for s in seen] == [1, 2, 3]
    assert reader.bad_events == 5 and reader.status().bad_events == 5
    assert np.allclose(seen[1].pose.orientation, [0.0, 0.0, 0.0, 1.0]) and seen[1].valid
    assert np.all(np.isfinite(seen[1].vel_lin)) and seen[1].vel_lin[0] == 0.0
    assert np.allclose(seen[2].pose.position, [0.12, 0.0, 0.0])
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("non-finite / non-unit" in m for m in msgs)
    assert any("handling failed" in m for m in msgs)
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)


def _run_no_pose(objects, clock_step=4.0, polls=3, log_line=None):
    """Drive the loop through the grace period with idle polls only; the stub
    stops the reader (instead of a Shutdown event) so the no-pose status survives."""
    reader, slot, clock = _reader(TrackerConfig(backend="libsurvive"))
    reader._set_status("searching", "waiting for 'WM0' poses")
    if log_line is not None:
        reader._on_survive_log(None, 0, log_line)

    def poll():
        clock.t += clock_step  # every idle poll advances the clock
        if not ps.events:
            reader._stop.set()

    ps = StubPS([NONE_EV] * polls, objects=objects, on_poll=poll)
    reader._libsurvive_events(ps, object(), FAKE_CTYPES)
    return reader


def test_no_pose_status_error_vs_searching_by_object_enumeration():
    lh0, lh1 = _obj("LH0", StubPS.SurviveSimpleObject_LIGHTHOUSE), _obj("LH1", 2)
    # Lighthouses only (count 2, but no OBJECT-type device): error with the how-to.
    st = _run_no_pose([lh0, lh1]).status()
    assert st.status == "error", st
    for needle in ("no tracked device", "dongle busy", "LIBUSB_ERROR_BUSY", "udev",
                   "tracker off", "unpaired", "01-sudo-udev-and-deps.sh"):
        assert needle in st.detail, (needle, st.detail)
    # The libusb line from the logger callback is appended.
    st = _run_no_pose([], log_line=b"libusb: LIBUSB_ERROR_BUSY claiming interface").status()
    assert st.status == "error"
    assert st.detail.endswith(": libusb: LIBUSB_ERROR_BUSY claiming interface")
    # An OBJECT device with another name: searching, mismatch spelled out.
    st = _run_no_pose([lh0, _obj("WM1")]).status()
    assert st.status == "searching" and "WM1" in st.detail
    assert "none named 'WM0'" in st.detail and "object_name" in st.detail
    # The right device, no pose yet: searching (base stations / still tracker hint).
    st = _run_no_pose([lh0, _obj("WM0")]).status()
    assert st.status == "searching" and "device present" in st.detail
    # Before the grace period (3 polls x 0.5 s < 3 s) nothing is reported.
    st = _run_no_pose([], clock_step=0.5).status()
    assert st.status == "searching" and st.detail == "waiting for 'WM0' poses"


def test_rate_hz_decays_to_zero_when_samples_stop():
    reader, slot, clock = _reader()
    for _ in range(30):
        clock.t += 0.01
        reader._publish(Pose(np.zeros(3), IDENT), np.zeros(3), np.zeros(3), 0.0)
    assert reader.status().rate_hz == pytest.approx(100.0, rel=0.02)
    r_half = reader.status(clock.t + 0.5).rate_hz
    assert 30.0 < r_half < 60.0, r_half  # 29 intervals over ~0.8 s: decaying
    assert reader.status(clock.t + 0.9).rate_hz < r_half
    assert reader.status(clock.t + 1.1).rate_hz == 0.0  # window empty: exactly 0
    assert reader.status(clock.t + 1.1).status == "stale"


def test_libsurvive_warnings_are_rate_limited_per_message_class(caplog):
    caplog.set_level(logging.DEBUG, logger="apollo_xarm7_runtime.devices.tracker")
    reader, slot, clock = _reader(TrackerConfig(backend="libsurvive"))
    for i in range(10):
        reader._on_survive_log(None, 1, f"Dropped {i} packets from WM0".encode())
    reader._on_survive_log(None, 1, b"Lighthouse 2 not seen")
    reader._on_survive_log(None, 1, b"Lighthouse 3 not seen")
    reader._on_survive_log(None, 5, b"info line")  # info: debug only
    warns = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warns == ["libsurvive: Dropped 0 packets from WM0", "libsurvive: Lighthouse 2 not seen"]
    assert reader.status().detail == "" and reader._last_error_log == "Lighthouse 3 not seen"
    clock.t += 1.1
    reader._on_survive_log(None, 1, b"Dropped 11 packets from WM0")
    warns = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warns[-1] == "libsurvive: Dropped 11 packets from WM0 (+9 similar suppressed)"
    reader._on_survive_log(None, 1, b"Dropped 12 packets from WM0")  # within 1 s: suppressed
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 3
    assert any("info line" in r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG)


class RestartStubModule:
    """A ``pysurvive`` stand-in whose event loop shuts down immediately."""

    SurviveSimpleEventType_None = 0
    SurviveSimpleEventType_ButtonEvent = 1
    SurviveSimpleEventType_PoseUpdateEvent = 3
    SurviveSimpleEventType_Shutdown = 5
    SurviveSimpleObject_OBJECT = 1
    SurviveSimpleEvent = ctypes.c_int  # a real ctypes instance for byref()

    def __init__(self):
        self.inits = 0
        self.closes = 0

    @staticmethod
    def SurviveSimpleLogFn(fn):  # noqa: N802 - mirrors the C API name
        return fn

    def simple_init_with_logger(self, argc, argv, cb):
        self.inits += 1
        return 1

    def simple_start_thread(self, ptr):
        pass

    def simple_next_event(self, ptr, ev):
        return self.SurviveSimpleEventType_Shutdown

    def simple_close(self, ptr):
        self.closes += 1


def test_libsurvive_loop_auto_restarts_with_backoff_and_start_is_restartable(monkeypatch):
    stub = RestartStubModule()
    monkeypatch.setitem(sys.modules, "pysurvive", stub)
    monkeypatch.setattr(tracker_mod, "LIBSURVIVE_RESTART_BACKOFF_S", (0.005, 0.02))
    reader = TrackerReader(TrackerConfig(backend="libsurvive"), LatestSlot())
    reader.start()
    try:
        deadline = time.monotonic() + 3.0
        while stub.inits < 4 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert reader.restarts >= 3 and stub.inits >= 4 and stub.closes >= stub.inits - 1
        assert reader.status().status == "error" and "shut down" in reader.status().detail
        assert reader.status().restarts == reader.restarts
    finally:
        reader.stop()
    assert reader._thread is None
    inits = stub.inits
    reader.start()  # restartable: a fresh thread picks up where the old one left
    try:
        deadline = time.monotonic() + 3.0
        while stub.inits < inits + 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert stub.inits >= inits + 2
    finally:
        reader.stop()
    assert reader._thread is None and stub.closes == stub.inits
