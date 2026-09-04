"""Vive-controller inputs in the tracker reader (13-tracker §1.1): libsurvive
button/touch/axis event parsing with synthetic pysurvive-like structs (no
pysurvive import), trackpad click classification (dominant axis, deadzone,
fixed at the press edge), press-edge accounting (the ``edges`` history with
``edge_seq`` / ``edge_input`` over the trackpad, menu and grip buttons),
``held_codes`` / ``click_actions`` derivation (map, ``none``, validation), edge
re-publish (never refreshing the pose's age / rate / status), the fake backend's
scripted controller hook, and the libsurvive event loop driven by a stub ``ps``
module incl. the robustness rules (bad events, error-vs-searching status, rate
decay, log rate limit, auto-restart) and the calibration hooks (lighthouse
snapshot, INFO-line queue, ``restart`` with new libsurvive arguments)."""

from __future__ import annotations

import ctypes
import logging
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from apollo_mavis_v2_core import LatestSlot, Pose
from apollo_mavis_v2_core.protocol import KEYMAP

import apollo_mavis_v2_runtime.devices.tracker as tracker_mod
from apollo_mavis_v2_runtime.config import (
    CONTROLLER_DISCRETE_ACTIONS,
    CONTROLLER_HELD_ACTIONS,
    ControllerMapConfig,
    TrackerConfig,
)
from apollo_mavis_v2_runtime.devices.tracker import (
    AXIS_TRACKPAD_X,
    AXIS_TRACKPAD_Y,
    AXIS_TRIGGER,
    BUTTON_GRIP,
    BUTTON_MENU,
    BUTTON_SYSTEM,
    BUTTON_TRACKPAD,
    BUTTON_TRIGGER,
    CLUTCH_CODE,
    DISCRETE_ACTIONS,
    EDGE_HISTORY,
    EVENT_AXIS_CHANGED,
    EVENT_BUTTON_DOWN,
    EVENT_BUTTON_UP,
    EVENT_TOUCH_DOWN,
    EVENT_TOUCH_UP,
    GRIPPER_CLOSE_CODE,
    GRIPPER_OPEN_CODE,
    HELD_ACTION_CODES,
    RAIL_NEG_CODE,
    RAIL_POS_CODE,
    ControllerState,
    LighthouseSnapshot,
    TrackerReader,
    active_inputs,
    apply_button_event,
    classify_trackpad,
    derive_click_action,
    derive_click_actions,
    derive_held_codes,
    note_edges,
)

IDENT = np.array([1.0, 0.0, 0.0, 0.0])
DZ = 0.3


def _code(action: str) -> str:
    return next(e.code for e in KEYMAP if e.action == action)


def test_injected_codes_come_from_the_keymap_not_letters():
    assert CLUTCH_CODE == _code("tracker_clutch")
    assert GRIPPER_OPEN_CODE == _code("gripper_open")
    assert GRIPPER_CLOSE_CODE == _code("gripper_close")
    assert RAIL_NEG_CODE == _code("rail_neg") and RAIL_POS_CODE == _code("rail_pos")
    codes = {CLUTCH_CODE, GRIPPER_OPEN_CODE, GRIPPER_CLOSE_CODE, RAIL_NEG_CODE, RAIL_POS_CODE}
    assert len(codes) == 5


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
    assert s.edge_seq == 0 and s.edge_input is None  # raw: note_edges adds these
    assert s.trackpad_dir is None
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
    return note_edges(prev, _pad(x, y, **kw), DZ)


def _btn(prev=None, **fields) -> ControllerState:
    """State with the given buttons (menu / grip / trigger...) passed through note_edges."""
    return note_edges(prev, ControllerState(**fields), DZ)


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
    assert s1.edge_seq == 1 and s1.edge_input == "trackpad_right"
    assert s1.trackpad_dir == "trackpad_right" and active_inputs(s1) == {"trackpad_right"}
    s2 = note_edges(s1, _pad(0.0, 0.9), DZ)  # finger slides to the top while clicked
    assert s2.edge_seq == 1 and s2.trackpad_dir == "trackpad_right"  # classification held
    assert active_inputs(s2) == {"trackpad_right"}
    s3 = note_edges(s2, _pad(0.0, 0.9, click=False), DZ)  # release
    assert not s3.trackpad_click and s3.edge_seq == 1 and s3.edge_input == "trackpad_right"
    assert active_inputs(s3) == frozenset()  # released: nothing held
    s4 = note_edges(s3, _pad(0.0, -0.9), DZ)  # new press at the bottom
    assert s4.edge_seq == 2 and s4.edge_input == "trackpad_down" == s4.trackpad_dir
    s5 = note_edges(s4, _pad(0.0, -0.9, click=False), DZ)
    s6 = note_edges(s5, _pad(0.1, 0.1), DZ)  # press inside the deadzone: counted, unclassified
    assert s6.edge_seq == 3 and s6.edge_input is None and s6.trackpad_dir is None
    assert active_inputs(s6) == frozenset()
    s7 = note_edges(s6, _pad(0.9, 0.0), DZ)  # slides out afterwards: still ignored
    assert s7.edge_seq == 3 and s7.edge_input is None and active_inputs(s7) == frozenset()
    # Trigger is independent of the pad and registers no edge; a raw (never
    # classified) click holds nothing.
    assert active_inputs(_pad(0.9, 0.0, trigger=True)) == {"trigger_click"}
    t = note_edges(None, _pad(0.9, 0.0, trigger=True), DZ)
    assert active_inputs(t) == {"trigger_click", "trackpad_right"} and t.edge_seq == 1
    assert _btn(trigger_pressed=True).edge_seq == 0
    # A scripted state arriving with prev=None while already clicked counts as a press edge.
    assert note_edges(None, _pad(0.0, 0.9), DZ).edge_seq == 1
    # Unclicked states carry the previous accounting and never classify.
    s8 = note_edges(s7, ControllerState(trigger_pressed=True), DZ)
    assert s8.edge_seq == 3 and s8.edge_input is None


def test_menu_and_grip_press_edges_advance_edge_seq_and_keep_trackpad_dir():
    m1 = _btn(menu=True)
    assert m1.edge_seq == 1 and m1.edge_input == "menu_click"
    assert active_inputs(m1) == {"menu_click"}
    m2 = note_edges(m1, ControllerState(menu=True), DZ)  # held: no new edge
    assert m2.edge_seq == 1 and active_inputs(m2) == {"menu_click"}
    m3 = note_edges(m2, ControllerState(), DZ)  # release: accounting carried
    assert m3.edge_seq == 1 and m3.edge_input == "menu_click" and active_inputs(m3) == set()
    g1 = note_edges(m3, ControllerState(grip=True), DZ)
    assert g1.edge_seq == 2 and g1.edge_input == "grip_click"
    assert active_inputs(g1) == {"grip_click"}
    # Menu pressed while the pad is clicked: newest edge is the menu, the pad
    # classification is carried so its held code keeps flowing.
    up = _click(0.0, 0.9)
    both = note_edges(
        up, ControllerState(trackpad_touch=True, trackpad_click=True, trackpad_y=0.9, menu=True), DZ
    )
    assert both.edge_seq == 2 and both.edge_input == "menu_click"
    assert both.trackpad_dir == "trackpad_up"
    assert active_inputs(both) == {"trackpad_up", "menu_click"}
    # The system button never registers an edge nor an input.
    sysb = _btn(system=True)
    assert sysb.edge_seq == 0 and sysb.edge_input is None and active_inputs(sysb) == set()
    # Simultaneous edges (scripted) each count; the order trackpad, menu, grip resolves.
    multi = note_edges(None, ControllerState(trackpad_click=True, trackpad_y=0.9, menu=True), DZ)
    assert multi.edge_seq == 2 and multi.edge_input == "menu_click"
    assert multi.trackpad_dir == "trackpad_up"
    assert multi.edges == ((1, "trackpad_up"), (2, "menu_click"))


def test_edge_history_is_lossless_up_to_edge_history_entries():
    cfg = TrackerConfig(controller_map=ControllerMapConfig(arm_prev="grip_click"))
    m = _btn(menu=True)
    g = note_edges(m, ControllerState(menu=True, grip=True), DZ)  # grip edges while menu is down
    assert g.edges == ((1, "menu_click"), (2, "grip_click"))
    assert (g.edge_seq, g.edge_input) == (2, "grip_click")
    assert derive_click_actions(g, cfg) == ((1, "switch_arm"), (2, "switch_arm_prev"))
    assert derive_click_actions(g, TrackerConfig()) == ((1, "switch_arm"),)  # grip unbound
    assert derive_click_action(g, TrackerConfig()) is None  # newest edge only
    assert derive_click_actions(None, cfg) == ()
    assert derive_click_actions(ControllerState(), cfg) == ()
    # Released / held states carry the history; a deadzone click appends (seq, None).
    r = note_edges(g, ControllerState(), DZ)
    assert r.edges == g.edges and derive_click_actions(r, cfg) == derive_click_actions(g, cfg)
    d = note_edges(r, _pad(0.1, 0.1), DZ)
    assert d.edges[-1] == (3, None) and derive_click_actions(d, cfg)[-1] == (2, "switch_arm_prev")
    # The history is bounded: only the newest EDGE_HISTORY edges are kept, seq stays absolute.
    st, prev = None, None
    for i in range(EDGE_HISTORY + 4):
        st = note_edges(prev, ControllerState(menu=(i % 2 == 0)), DZ)
        prev = st
    presses = (EDGE_HISTORY + 4 + 1) // 2
    assert st.edge_seq == presses and len(st.edges) == min(presses, EDGE_HISTORY)
    assert st.edges[0][0] == presses - len(st.edges) + 1 and st.edges[-1][0] == presses
    assert all(inp == "menu_click" for _, inp in st.edges)


def test_controller_map_action_tables_agree_across_config_and_reader():
    # config.CONTROLLER_*_ACTIONS is the single list of action names; the reader's
    # code / action tables and the ControllerMapConfig fields must match it exactly
    # (both modules raise at import time otherwise).
    assert tuple(HELD_ACTION_CODES) == CONTROLLER_HELD_ACTIONS
    assert tuple(DISCRETE_ACTIONS) == CONTROLLER_DISCRETE_ACTIONS
    assert set(ControllerMapConfig.model_fields) == set(CONTROLLER_HELD_ACTIONS) | set(
        CONTROLLER_DISCRETE_ACTIONS
    )
    cfg_mod = __import__("apollo_mavis_v2_runtime.config", fromlist=["x"])
    assert not hasattr(cfg_mod, "TRACKPAD_INPUTS")


# -- controller_map config ----------------------------------------------------------------------
def test_controller_map_defaults_match_spec():
    assert ControllerMapConfig().model_dump() == {
        "clutch": "trigger_click",
        "gripper_open": "trackpad_up",
        "gripper_close": "trackpad_down",
        "rail_neg": "trackpad_left",
        "rail_pos": "trackpad_right",
        "arm_next": "menu_click",
        "arm_prev": "none",
    }
    assert TrackerConfig().trackpad_deadzone == 0.3


def test_controller_map_validation():
    with pytest.raises(ValueError, match="trigger_click"):
        ControllerMapConfig(clutch="none", arm_next="trigger_click")  # held-only input
    with pytest.raises(ValueError, match="held-only"):
        ControllerMapConfig(clutch="none", arm_prev="trigger_click")
    with pytest.raises(ValueError, match="more than one"):
        ControllerMapConfig(gripper_open="trackpad_left")  # clashes with rail_neg default
    with pytest.raises(ValueError, match="more than one"):
        ControllerMapConfig(arm_prev="menu_click")  # clashes with arm_next default
    with pytest.raises(ValueError):
        ControllerMapConfig(clutch="system")  # system is not bindable
    with pytest.raises(ValueError):
        ControllerMapConfig(clutch="grip")  # not an input name
    # Discrete actions accept menu / grip / trackpad; held actions accept menu / grip too.
    ok = ControllerMapConfig(arm_next="menu_click", arm_prev="grip_click")
    assert ok.arm_prev == "grip_click"
    ok = ControllerMapConfig(arm_next="trackpad_left", rail_neg="none")
    assert ok.arm_next == "trackpad_left"
    ok = ControllerMapConfig(clutch="grip_click", arm_next="none", gripper_open="menu_click")
    assert ok.clutch == "grip_click" and ok.gripper_open == "menu_click"
    all_none = ControllerMapConfig(**{k: "none" for k in ControllerMapConfig.model_fields})
    assert set(all_none.model_dump().values()) == {"none"}


# -- held_codes / click_action derivation -----------------------------------------------------
def test_derive_held_codes_and_click_action_defaults():
    cfg = TrackerConfig()
    assert derive_held_codes(None, cfg) == frozenset()
    assert derive_held_codes(ControllerState(), cfg) == frozenset()
    assert derive_held_codes(ControllerState(trigger_pressed=True), cfg) == {CLUTCH_CODE}
    assert derive_held_codes(ControllerState(trigger=0.9), cfg) == frozenset()  # analog only
    assert derive_held_codes(_click(-0.9, 0.0), cfg) == {RAIL_NEG_CODE}  # left -> ArrowLeft
    assert derive_held_codes(_click(0.9, 0.0), cfg) == {RAIL_POS_CODE}  # right -> ArrowRight
    assert derive_held_codes(_click(0.0, 0.9), cfg) == {GRIPPER_OPEN_CODE}  # up -> KeyH
    assert derive_held_codes(_click(0.0, -0.9), cfg) == {GRIPPER_CLOSE_CODE}  # down -> KeyF
    assert derive_held_codes(_click(0.2, 0.2), cfg) == frozenset()  # deadzone
    assert derive_held_codes(_pad(0.9, 0.0), cfg) == frozenset()  # raw, never classified
    assert derive_held_codes(_pad(0.9, 0.0, click=False), cfg) == frozenset()  # touch only
    assert derive_held_codes(_btn(menu=True), cfg) == frozenset()  # menu: discrete only
    assert derive_held_codes(_btn(grip=True, system=True), cfg) == frozenset()  # unbound
    both = _click(0.0, -0.9, trigger=True)
    assert derive_held_codes(both, cfg) == {CLUTCH_CODE, GRIPPER_CLOSE_CODE}
    assert derive_click_action(_btn(menu=True), cfg) == "switch_arm"
    assert derive_click_action(_btn(grip=True), cfg) is None  # grip unbound by default
    for held_only in (_click(0.0, 0.9), _click(0.0, -0.9), _click(-0.9, 0.0), _click(0.9, 0.0)):
        assert derive_click_action(held_only, cfg) is None  # held bindings, not discrete
    assert derive_click_action(_click(0.2, 0.2), cfg) is None
    assert derive_click_action(None, cfg) is None
    # The edge accounting (and thus the action) persists after release.
    released = note_edges(_btn(menu=True), ControllerState(), DZ)
    assert derive_click_action(released, cfg) == "switch_arm"
    assert derive_held_codes(released, cfg) == frozenset()
    dz = TrackerConfig(trackpad_deadzone=0.5)
    assert derive_held_codes(note_edges(None, _pad(0.4, 0.0), 0.5), dz) == frozenset()


def test_derive_click_edge_examples_and_deadzone_counts():
    cfg = TrackerConfig()
    left, right = _click(-0.9, 0.0), _click(0.9, 0.0)
    assert derive_held_codes(left, cfg) == {"ArrowLeft"} == {RAIL_NEG_CODE}
    assert derive_held_codes(right, cfg) == {"ArrowRight"} == {RAIL_POS_CODE}
    up, down = _click(0.0, 0.9), _click(0.0, -0.9)
    assert derive_held_codes(up, cfg) == {"KeyH"} and derive_held_codes(down, cfg) == {"KeyF"}
    dead = note_edges(_click(0.9, 0.0, prev=None), _pad(0.1, 0.1, click=False), DZ)
    dead = note_edges(dead, _pad(0.1, 0.1), DZ)  # second press, inside the deadzone
    assert dead.edge_seq == 2 and dead.edge_input is None
    assert derive_held_codes(dead, cfg) == frozenset()
    assert derive_click_action(dead, cfg) is None


def test_derive_honours_custom_map_and_none():
    cfg = TrackerConfig(
        controller_map=ControllerMapConfig(
            clutch="grip_click",
            gripper_open="none",
            gripper_close="trigger_click",
            rail_neg="none",
            rail_pos="menu_click",
            arm_next="trackpad_right",
            arm_prev="trackpad_down",
        )
    )
    assert derive_held_codes(_btn(grip=True), cfg) == {CLUTCH_CODE}
    assert derive_held_codes(ControllerState(trigger_pressed=True), cfg) == {GRIPPER_CLOSE_CODE}
    assert derive_held_codes(_btn(menu=True), cfg) == {RAIL_POS_CODE}
    assert derive_held_codes(_click(0.0, 0.9), cfg) == frozenset()  # gripper_open unbound
    assert derive_held_codes(_click(-0.9, 0.0), cfg) == frozenset()  # rail_neg unbound
    assert derive_held_codes(_click(0.9, 0.0), cfg) == frozenset()  # right is discrete here
    assert derive_click_action(_click(0.9, 0.0), cfg) == "switch_arm"
    assert derive_click_action(_click(0.0, -0.9), cfg) == "switch_arm_prev"
    assert derive_click_action(_btn(menu=True), cfg) is None  # menu is a held binding here
    assert derive_click_action(_btn(grip=True), cfg) is None
    off = TrackerConfig(
        controller_map=ControllerMapConfig(
            clutch="none",
            gripper_open="none",
            gripper_close="none",
            rail_neg="none",
            rail_pos="none",
            arm_next="none",
            arm_prev="none",
        )
    )
    assert derive_held_codes(_click(0.9, 0.0, trigger=True), off) == frozenset()
    assert derive_held_codes(_btn(menu=True, grip=True), off) == frozenset()
    assert derive_click_action(_btn(menu=True), off) is None


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
    assert s1.click_actions == ()
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
    # Trackpad click at the top: classified up -> gripper_open held, no discrete action.
    assert reader._on_controller(_pad(0.0, 0.9)) is True
    s3 = slot.get()[0]
    assert s3.controller.edge_seq == 1 and s3.controller.edge_input == "trackpad_up"
    assert s3.controller.trackpad_dir == "trackpad_up"
    assert s3.held_codes == {GRIPPER_OPEN_CODE} and s3.click_actions == ()
    assert reader._on_controller(_pad(0.0, 0.9, click=False)) is True  # release edge
    assert slot.get()[0].held_codes == frozenset() and slot.get()[0].click_actions == ()
    # Menu press edge: discrete switch_arm, no held code.
    assert reader._on_controller(ControllerState(menu=True)) is True
    s4 = slot.get()[0]
    assert s4.controller.edge_seq == 2 and s4.controller.edge_input == "menu_click"
    assert s4.held_codes == frozenset() and s4.click_actions == ((2, "switch_arm"),)
    assert reader._on_controller(ControllerState()) is True  # release edge
    assert slot.get()[0].click_actions == ((2, "switch_arm"),)  # persists for edge accounting
    # Pose older than stale_s at the edge: codes still ride along, pose flagged invalid.
    clock.t += 1.0
    assert reader._on_controller(_pad(-0.9, 0.0)) is True  # left -> rail_neg
    s5 = slot.get()[0]
    assert s5.valid is False and s5.held_codes == {RAIL_NEG_CODE}
    assert s5.controller.edge_seq == 3 and s5.click_actions == ((2, "switch_arm"),)
    assert reader.status().device_held == {RAIL_NEG_CODE}  # fresh by the SAMPLE's rx_mono
    assert reader.status(clock.t + 0.5).device_held == frozenset()  # stale -> nothing
    st = reader.status()  # ... but the POSE is 1.05 s old: the edge did not refresh it
    assert st.status == "stale" and st.age_s == pytest.approx(1.05)
    assert "older than" in st.detail and "jump" not in st.detail


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
    assert s3.seq == 3 and s3.controller.edge_seq == 1
    assert s3.controller.edge_input == "trackpad_down" and s3.click_actions == ()
    assert s3.held_codes == {CLUTCH_CODE, GRIPPER_CLOSE_CODE}  # trigger still held; down = KeyF
    # Menu press: discrete switch_arm rides on the re-published sample.
    be4 = SimpleNamespace(
        time=1.8, object=object(), event_type=EVENT_BUTTON_DOWN, button_id=BUTTON_MENU,
        axis_count=0, axis_ids=[0] * 8, axis_val=[0.0] * 8,
    )
    reader._on_button_event(be4)
    s4 = slot.get()[0]
    assert s4.seq == 4 and s4.controller.edge_seq == 2 and s4.controller.menu
    assert s4.controller.edge_input == "menu_click" and s4.click_actions == ((2, "switch_arm"),)
    assert s4.held_codes == {CLUTCH_CODE, GRIPPER_CLOSE_CODE}  # pad classification carried
    # Grip DOWN 3 ms after the menu: the slot holds only the grip sample, whose
    # click_actions still carry the bound menu edge (lossless within a tick).
    clock.t += 0.003
    be5 = SimpleNamespace(
        time=1.803, object=object(), event_type=EVENT_BUTTON_DOWN, button_id=BUTTON_GRIP,
        axis_count=0, axis_ids=[0] * 8, axis_val=[0.0] * 8,
    )
    reader._on_button_event(be5)
    s5 = slot.get()[0]
    assert s5.seq == 5 and s5.controller.edge_seq == 3 and s5.controller.edge_input == "grip_click"
    assert s5.click_actions == ((2, "switch_arm"),)


def test_edge_republish_never_refreshes_pose_age_rate_or_status(caplog):
    caplog.set_level(logging.DEBUG, logger="apollo_mavis_v2_runtime.devices.tracker")
    reader, slot, clock = _reader()  # stale_s 0.2, max_jump_m 0.1
    reader._publish(Pose(np.array([0.1, 0.0, 0.0]), IDENT), np.zeros(3), np.zeros(3), 0.0)
    st0 = reader.status()
    assert st0.status == "tracking" and st0.age_s == 0.0 and st0.detail == ""
    touch, seen = False, []
    for _ in range(6):  # the thumb taps the pad every 0.15 s; no pose arrives for 0.9 s
        clock.t += 0.15
        touch = not touch
        assert reader._on_controller(ControllerState(trackpad_touch=touch)) is True
        seen.append(slot.get()[0])
    assert [s.valid for s in seen] == [True, False, False, False, False, False]
    assert all(np.allclose(s.pose.position, [0.1, 0.0, 0.0]) for s in seen)
    assert [s.seq for s in seen] == [2, 3, 4, 5, 6, 7]
    st = reader.status()
    assert st.status == "stale" and st.age_s == pytest.approx(0.9) and st.rate_hz == 0.0
    assert st.seq == 7 and st.detail == "controller edge on a pose older than 0.2 s: sample invalid"
    assert st.device_held == frozenset()  # nothing bound to the touch
    # A stale-pose edge is not a jump: no WARNING at all, one DEBUG line per invalid re-publish.
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    dbg = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert len([m for m in dbg if "controller edge" in m and "invalid" in m]) == 5
    # A new pose restores tracking; the jump check runs against the last real pose.
    clock.t += 0.01
    reader._publish(Pose(np.array([0.12, 0.0, 0.0]), IDENT), np.zeros(3), np.zeros(3), 1.0)
    st = reader.status()
    assert st.status == "tracking" and st.age_s == 0.0 and st.detail == ""
    assert slot.get()[0].valid and slot.get()[0].seq == 8
    # A genuine jump still warns with the jump detail; an edge on that (fresh, invalid)
    # pose stays invalid and keeps the jump detail rather than claiming staleness.
    reader._publish(Pose(np.array([0.5, 0.0, 0.0]), IDENT), np.zeros(3), np.zeros(3), 1.1)
    assert slot.get()[0].valid is False and reader.status().detail.startswith("jump > 0.1 m")
    warns = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warns == ["tracker sample 9 invalid (jump > 0.10 m)"]
    assert reader._on_controller(ControllerState(trackpad_touch=False, menu=True)) is True
    assert slot.get()[0].valid is False and reader.status().detail.startswith("jump > 0.1 m")
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


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
        script["state"] = _pad(0.9, 0.0)  # right -> rail_pos (classified by the reader)
        while time.monotonic() < deadline and RAIL_POS_CODE not in slot.get()[0].held_codes:
            time.sleep(0.01)
        s = slot.get()[0]
        assert s.held_codes == {RAIL_POS_CODE} and s.controller.edge_seq == 1
        assert s.controller.edge_input == "trackpad_right" and s.click_actions == ()
        script["state"] = _pad(0.9, 0.0, click=False)  # release
        while time.monotonic() < deadline and slot.get()[0].held_codes:
            time.sleep(0.01)
        script["state"] = ControllerState(menu=True)  # menu -> discrete switch_arm
        while time.monotonic() < deadline and slot.get()[0].click_actions != ((2, "switch_arm"),):
            time.sleep(0.01)
        s = slot.get()[0]
        assert s.controller.edge_seq == 2 and s.held_codes == frozenset()
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

    class SurvivePose:  # LinmathPose: Pos[3] + Rot[4] (wxyz); zeros until solved
        def __init__(self):
            self.Pos = [0.0, 0.0, 0.0]
            self.Rot = [0.0, 0.0, 0.0, 0.0]

    def simple_serial_number(self, obj):
        serial = getattr(obj, "serial", None)
        return serial.encode() if serial is not None else None

    def simple_object_get_latest_pose(self, obj, pose):
        lh_pose = getattr(obj, "pose", None)
        if lh_pose is not None:
            pose.Pos[:] = list(lh_pose[0])
            pose.Rot[:] = list(lh_pose[1])
        return 1.5

    def simple_get_object_count(self, ptr):
        return len(self.objects)

    def simple_get_first_object(self, ptr):
        return self.objects[0] if self.objects else None

    def simple_get_next_object(self, ptr, obj):
        i = self.objects.index(obj) + 1
        return self.objects[i] if i < len(self.objects) else None


NONE_EV = (StubPS.SurviveSimpleEventType_None, None)
FAKE_CTYPES = SimpleNamespace(byref=lambda x: x)


def _obj(name, kind=StubPS.SurviveSimpleObject_OBJECT, serial=None, pose=None):
    return SimpleNamespace(name=name, kind=kind, serial=serial, pose=pose)


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
        (1, _button(wm0, EVENT_BUTTON_DOWN, BUTTON_TRACKPAD)),  # edge: left -> ArrowLeft, seq 3
        (3, _pose(wm0, [0.11, 0.0, 0.0], 0.4)),  # seq 4 keeps ArrowLeft
    ], objects=[lh, wm0, wm1])
    reader, slot, clock = _reader(TrackerConfig(backend="libsurvive"))
    seen = _spy_publish(reader)
    reader._libsurvive_events(ps, object(), FAKE_CTYPES)
    assert [s.seq for s in seen] == [1, 2, 3, 4]
    assert seen[0].controller.trigger_pressed and seen[0].held_codes == {CLUTCH_CODE}
    assert np.allclose(seen[0].pose.position, [0.1, 0.0, 0.0])
    assert not seen[1].controller.trigger_pressed and seen[1].held_codes == frozenset()
    assert np.allclose(seen[1].pose.position, [0.1, 0.0, 0.0]) and seen[1].t_dev == 0.3
    assert seen[2].controller.trackpad_click and seen[2].held_codes == {RAIL_NEG_CODE}
    assert seen[2].controller.trackpad_dir == "trackpad_left" and seen[2].click_actions == ()
    assert seen[2].controller.edge_seq == 1 and seen[2].controller.edge_input == "trackpad_left"
    assert seen[3].held_codes == {RAIL_NEG_CODE} and seen[3].t_dev == 0.4
    assert np.allclose(seen[3].pose.position, [0.11, 0.0, 0.0]) and seen[3].valid
    assert reader.status().status == "error" and "shut down" in reader.status().detail
    assert reader.status().controller.trackpad_x == pytest.approx(-0.9)
    assert reader.bad_events == 0


def test_libsurvive_loop_survives_bad_events(caplog):
    caplog.set_level(logging.WARNING, logger="apollo_mavis_v2_runtime.devices.tracker")
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
    caplog.set_level(logging.DEBUG, logger="apollo_mavis_v2_runtime.devices.tracker")
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


# -- calibration hooks: lighthouse snapshot, INFO lines, restart -----------------------------------
def test_libsurvive_loop_snapshots_lighthouse_objects_by_ps_constants():
    lh0 = _obj("LH0", StubPS.SurviveSimpleObject_LIGHTHOUSE, serial="2684858188",
               pose=([3.8, 1.2, 1.8], [0.5, 0.5, 0.5, 0.5]))
    lh1 = _obj("LH1", StubPS.SurviveSimpleObject_LIGHTHOUSE)  # unsolved: zero quaternion
    wm0 = _obj("WM0", serial="LHR-FFFFFFFF", pose=([0.1, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]))
    reader, slot, clock = _reader(TrackerConfig(backend="libsurvive"))
    assert reader.lighthouses() == []

    def poll():
        clock.t += 0.3  # 3 polls: snapshot at t0, not at t0+0.3, again at t0+0.6
        if not ps.events:
            reader._stop.set()

    ps = StubPS([NONE_EV] * 3, objects=[lh0, wm0, lh1], on_poll=poll)
    reader._libsurvive_events(ps, object(), FAKE_CTYPES)
    snap = reader.lighthouses()
    assert [type(s) for s in snap] == [LighthouseSnapshot, LighthouseSnapshot]
    assert [(s.index, s.name, s.serial) for s in snap] == [
        (0, "LH0", "2684858188"), (1, "LH1", None)
    ]
    assert snap[0].pose is not None and np.allclose(snap[0].pose.position, [3.8, 1.2, 1.8])
    assert np.allclose(snap[0].pose.orientation, [0.5, 0.5, 0.5, 0.5])
    assert snap[1].pose is None  # zero quaternion = not solved
    assert snap[0].t_mono == snap[1].t_mono == clock.t - 0.3  # the latest refresh
    assert reader.bad_events == 0  # the snapshot path never counts as a bad event
    # An enumeration failure degrades to the previous snapshot, never an exception.
    ps = StubPS([NONE_EV], objects=[SimpleNamespace(kind=2)], on_poll=poll)  # no .name
    reader._stop.clear()
    reader._libsurvive_events(ps, object(), FAKE_CTYPES)
    assert len(reader.lighthouses()) == 2 and reader.bad_events == 0
    # restart() with the fake backend clears the snapshot and swaps the args on a COPY.
    reader.stop()
    cfg = reader.cfg
    reader.restart(["--configfile", "/tmp/x.json"])
    try:
        assert reader.lighthouses() == []
        assert reader.cfg is not cfg
        assert reader.cfg.libsurvive_args == ["--configfile", "/tmp/x.json"]
        assert cfg.libsurvive_args == ["--lighthousecount", "2"]  # shared config untouched
        assert reader._thread is not None and reader._thread.is_alive()
    finally:
        reader.stop()


def test_info_lines_are_kept_ansi_free_and_handed_to_on_info(caplog):
    caplog.set_level(logging.DEBUG, logger="apollo_mavis_v2_runtime.devices.tracker")
    reader, slot, clock = _reader(TrackerConfig(backend="libsurvive"))
    seen = []
    reader.on_info = lambda t, text: seen.append((t, text))
    reader._on_survive_log(None, 2, b"\x1b[0;32mInfo: Global solve with 3 scenes for 1\x1b[0m")
    reader._on_survive_log(None, 1, b"Lighthouse 2 not seen")  # warning: not an INFO line
    reader._on_survive_log(
        None, 3, "Using LH 2 (\x1b[0;31m596c9a8b\x1b[0m) as reference lighthouse"  # str, level 3
    )
    assert list(reader.info_lines) == [
        (clock.t, "Info: Global solve with 3 scenes for 1"),
        (clock.t, "Using LH 2 (596c9a8b) as reference lighthouse"),
    ]
    assert seen == list(reader.info_lines)
    assert reader._last_error_log == "Lighthouse 2 not seen"
    # A raising callback never propagates into the C caller and the line is still kept.
    reader.on_info = lambda t, text: 1 / 0
    reader._on_survive_log(None, 2, b"Info: Force calibrate flag set")
    assert reader.info_lines[-1][1] == "Info: Force calibrate flag set"
    reader.on_info = None
    for i in range(300):
        reader._on_survive_log(None, 2, f"line {i}".encode())
    assert len(reader.info_lines) == 256 and reader.info_lines[-1][1] == "line 299"
    assert any("Global solve with 3 scenes" in r.getMessage() for r in caplog.records)


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


class IdleStubModule(RestartStubModule):
    """A ``pysurvive`` stand-in whose event loop idles (None events) until the
    reader stops; records the argv of every ``simple_init_with_logger``."""

    SurviveSimpleObject_LIGHTHOUSE = 2

    def __init__(self):
        super().__init__()
        self.argvs: list[list[str]] = []

    def simple_init_with_logger(self, argc, argv, cb):
        self.argvs.append(
            [ctypes.cast(argv[i], ctypes.c_char_p).value.decode() for i in range(argc)]
        )
        return super().simple_init_with_logger(argc, argv, cb)

    def simple_next_event(self, ptr, ev):
        time.sleep(0.001)
        return self.SurviveSimpleEventType_None

    def simple_get_first_object(self, ptr):
        return None


def test_restart_swaps_libsurvive_args_after_a_clean_close(monkeypatch):
    stub = IdleStubModule()
    monkeypatch.setitem(sys.modules, "pysurvive", stub)
    cfg = TrackerConfig(backend="libsurvive", libsurvive_args=["--lighthousecount", "3"])
    reader = TrackerReader(cfg, LatestSlot())
    reader.start()
    try:
        deadline = time.monotonic() + 3.0
        while stub.inits < 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert stub.argvs == [["apollo-mavis-v2-runtime", "--lighthousecount", "3"]]
        assert reader.status().status == "searching"
        reader.restart(["--lighthousecount", "3", "--configfile", "/tmp/bs.json",
                        "--force-calibrate", "1", "--globalscenesolver", "1"])
        deadline = time.monotonic() + 3.0
        while stub.inits < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert stub.closes == 1 and stub.inits == 2  # closed BEFORE the new init (dongle owner)
        assert stub.argvs[1] == [
            "apollo-mavis-v2-runtime", "--lighthousecount", "3", "--configfile", "/tmp/bs.json",
            "--force-calibrate", "1", "--globalscenesolver", "1",
        ]
        assert cfg.libsurvive_args == ["--lighthousecount", "3"]  # shared config untouched
        assert reader.cfg is not cfg and reader.restarts == 0  # operator restarts are not failures
        assert reader.status().status == "searching"
    finally:
        reader.stop()
    assert stub.closes == stub.inits == 2 and reader._thread is None


class BlockingCloseStubModule(IdleStubModule):
    """``simple_close`` blocks until ``release`` is set (a USB stall on close)."""

    def __init__(self):
        super().__init__()
        self.closing = threading.Event()
        self.release = threading.Event()

    def simple_close(self, ptr):
        self.closing.set()
        self.release.wait(5.0)
        super().simple_close(ptr)


def test_restart_refuses_while_the_old_libsurvive_close_is_pending(monkeypatch):
    """A restart must never open a second libsurvive context while the old one is
    still closing (LIBUSB_ERROR_BUSY, 13-tracker §6): a timed-out join keeps the
    thread handle, refuses the restart (status ``error``), leaves the arguments
    alone and lets the thread clear its own handle when the close completes."""
    stub = BlockingCloseStubModule()
    monkeypatch.setitem(sys.modules, "pysurvive", stub)
    cfg = TrackerConfig(backend="libsurvive", libsurvive_args=["--lighthousecount", "3"])
    reader = TrackerReader(cfg, LatestSlot())
    reader.start()
    try:
        deadline = time.monotonic() + 3.0
        while stub.inits < 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        old = reader._thread
        assert old is not None and stub.inits == 1
        with pytest.raises(RuntimeError, match="did not stop within 0 s"):
            reader.restart(["--lighthousecount", "3", "--configfile", "/tmp/bs.json"], timeout=0.05)
        assert stub.closing.wait(1.0)  # the old thread is inside simple_close
        assert stub.inits == 1 and stub.closes == 0  # no second context
        assert reader._thread is old and old.is_alive()  # handle kept by the stuck thread
        assert reader.cfg.libsurvive_args == ["--lighthousecount", "3"]  # args not swapped
        st = reader.status()
        assert st.status == "error" and "did not stop" in st.detail and "close pending" in st.detail
        reader.start()  # a no-op while the old thread lives
        assert stub.inits == 1 and reader._thread is old
        assert reader.stop(timeout=0.05) is False
        stub.release.set()  # the close completes: the thread clears its own handle
        deadline = time.monotonic() + 3.0
        while reader._thread is not None and time.monotonic() < deadline:
            time.sleep(0.005)
        assert reader._thread is None and stub.closes == 1
        reader.restart(["--lighthousecount", "3", "--configfile", "/tmp/bs.json"])
        deadline = time.monotonic() + 3.0
        while stub.inits < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert stub.inits == 2 and stub.argvs[1][-2:] == ["--configfile", "/tmp/bs.json"]
        assert reader.status().status == "searching"
    finally:
        stub.release.set()
        reader.stop()
    assert reader._thread is None and stub.closes == stub.inits == 2
