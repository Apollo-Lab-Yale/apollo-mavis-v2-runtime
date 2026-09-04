"""Tracker calibration back end (13-tracker §4 "Calibration modes"; phase-10):
``fit_yaw`` (synthetic gestures, the +180° stance pitfall, short legs, reversed
up/down, noise), the libsurvive argument sets, the persisted file, and the
``TrackerCalibration`` state machine over a duck-typed fake reader — base
station (per-phase restart arguments, scene counting from INFO lines, the
lighthouse snapshot merge, validation pass/fail, install with backup + byte copy
in ``tmp_path``, ``yaw_valid`` reset, abort restoring the normal arguments) and
yaw (trigger rising edges + REST capture, fit, apply persistence, startup
override, redo, abort)."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import deque

import numpy as np
import pytest
from apollo_mavis_v2_core import LatestSlot, Pose, se3
from apollo_mavis_v2_core.protocol import TrackerCalibrationCommand

from apollo_mavis_v2_runtime.config import (
    RuntimeConfig,
    TrackerCalibrationConfig,
    TrackerConfig,
)
from apollo_mavis_v2_runtime.control.tracker_teleop import yaw_quat
from apollo_mavis_v2_runtime.devices.tracker import (
    ControllerState,
    LighthouseSnapshot,
    TrackerSample,
    TrackerSettings,
)
from apollo_mavis_v2_runtime.devices.tracker_calibration import (
    YAW_POINT_ORDER,
    CalibrationError,
    PersistedCalibration,
    TrackerCalibration,
    apply_persisted_yaw,
    capture_args,
    fit_yaw,
    lighthouse_channels,
    strip_calibration_args,
    validation_args,
)

IDENT = np.array([1.0, 0.0, 0.0, 0.0])
CCFG = TrackerCalibrationConfig()
NORMAL = ["--lighthousecount", "3", "--globalscenesolver", "0", "--disable-calibrate", "1"]
# libsurvive's config.json is quasi-JSON: bare top-level "k":"v" lines, then one block per
# station libsurvive KNOWS (mode = channel); json.loads rejects it.
CONFIG_TEXT = (
    '"v":"0",\n"poser":"MPFIT",\n"configed-lighthouse-gen":"2",\n'
    '"floor-offset":"-0.340191096067"\n'
    '"lighthouse0":{\n"index":"0",\n"id":"2684858188",\n"mode":"3",\n'
    '"pose":["3.838969230652","1.266503095627","1.806747436523","0.310048612152",'
    '"0.263429196541","0.474210623260","0.780768340354"],\n'
    '"variance":["0.000220616654","0.000717556453","0.000069961083","0.000040179981",'
    '"0.000064019638","0.000066193439"],\n"OOTXSet":"1",\n"PositionSet":"1"\n},\n'
    '"lighthouse1":{\n"index":"1",\n"id":"1500617355",\n"mode":"7",\n'
    '"OOTXSet":"1",\n"PositionSet":"1"\n},\n'
    '"lighthouse2":{\n"index":"2",\n"id":"3921211122",\n"mode":"14",\n'
    '"OOTXSet":"1",\n"PositionSet":"1"\n}\n'
)
ORIGINAL = CONFIG_TEXT.encode()


def _rz(deg: float, v) -> np.ndarray:
    return se3.quat_rotate(yaw_quat(deg), np.asarray(v, dtype=np.float64))


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def gesture_world(
    leg: float = 0.25, *, left=(1, 0, 0), forward=(0, -1, 0), start=(0.3, 0.1, 0.2)
) -> np.ndarray:
    """Operator gesture in MJCF world: start, left, forward, right, back, up, down.
    ``left`` / ``forward`` are the operator's directions in world coordinates
    (defaults = the lab stance: +X / -Y)."""
    left, forward = np.asarray(left, float) * leg, np.asarray(forward, float) * leg
    up = np.array([0.0, 0.0, leg])
    p = [np.asarray(start, float)]
    for d in (left, forward, -left, -forward, up, -up):
        p.append(p[-1] + d)
    return np.asarray(p)


def gesture_raw(yaw_true: float, **kw) -> np.ndarray:
    """The same gesture as libsurvive sees it: ``p_world = Rz(yaw_true) p_raw``."""
    return np.asarray([_rz(-yaw_true, p) for p in gesture_world(**kw)])


# -- fit_yaw ----------------------------------------------------------------------------------
@pytest.mark.parametrize("yaw_true", [0.0, 45.0, 102.1, -77.9, 179.5, -170.0, 90.0])
def test_fit_yaw_recovers_the_alignment_of_a_clean_gesture(yaw_true):
    yaw, residual, checks = fit_yaw(gesture_raw(yaw_true), CCFG)
    assert checks == []
    assert abs(_wrap(yaw - yaw_true)) < 1e-6, (yaw, yaw_true)
    assert residual < 1e-6
    # Same definition as align_pose (p_world = Rz(yaw) p_raw): the raw LEFT leg lands on +X.
    raw = gesture_raw(yaw_true)
    assert np.allclose(_rz(yaw, raw[1] - raw[0]), [0.25, 0.0, 0.0], atol=1e-9)
    assert np.allclose(_rz(yaw, raw[2] - raw[1]), [0.0, -0.25, 0.0], atol=1e-9)


def test_fit_yaw_plus_180_pitfall_operator_on_the_arm_side():
    """Standing on the inner (arm) side, 'left' is -X and 'forward' is +Y: the fit
    is off by exactly 180° (the 2026-09-02 lesson: -77.9° vs the correct 102.1°)."""
    yaw_true = 102.1
    yaw_ok, _, checks_ok = fit_yaw(gesture_raw(yaw_true), CCFG)
    yaw_wrong, res, checks_wrong = fit_yaw(
        gesture_raw(yaw_true, left=(-1, 0, 0), forward=(0, 1, 0)), CCFG
    )
    assert checks_ok == [] and checks_wrong == [] and res < 1e-6  # a perfect fit either way
    assert abs(_wrap(yaw_ok - 102.1)) < 1e-6
    assert abs(_wrap(yaw_wrong - (-77.9))) < 1e-6
    assert abs(abs(_wrap(yaw_wrong - yaw_ok)) - 180.0) < 1e-6


def test_fit_yaw_tolerates_noise():
    rng = np.random.default_rng(7)
    raw = gesture_raw(30.0, leg=0.3) + rng.normal(scale=0.005, size=(7, 3))
    yaw, residual, checks = fit_yaw(raw, CCFG)
    assert checks == []
    assert abs(_wrap(yaw - 30.0)) < 3.0
    assert residual < 5.0


def test_fit_yaw_checks_short_legs_and_still_returns_the_yaw():
    yaw, _, checks = fit_yaw(gesture_raw(20.0, leg=0.05), CCFG)
    assert abs(_wrap(yaw - 20.0)) < 1e-6  # the estimate is still given
    assert [c.split(" ")[0] for c in checks] == ["left", "forward", "right", "back"]
    assert all("too short" in c and "5 cm < 10 cm" in c for c in checks)


def test_fit_yaw_checks_reversed_up_down_and_non_horizontal_legs():
    raw = gesture_raw(0.0)
    reversed_updown = raw.copy()
    reversed_updown[5] = raw[4] - [0, 0, 0.25]  # "up" goes down ...
    reversed_updown[6] = reversed_updown[5] + [0, 0, 0.25]  # ... and "down" goes up
    _, _, checks = fit_yaw(reversed_updown, CCFG)
    assert any("up leg does not rise" in c for c in checks)
    assert any("down leg does not descend" in c for c in checks)
    tilted = raw.copy()
    tilted[1] = raw[0] + [0.25, 0.0, 0.30]  # left leg climbs more than half its length
    _, _, checks = fit_yaw(tilted, CCFG)
    assert any(c.startswith("left leg not horizontal") for c in checks)
    shallow_up = raw.copy()
    shallow_up[5] = raw[4] + [0.25, 0.0, 0.05]  # up leg mostly horizontal
    _, _, checks = fit_yaw(shallow_up, CCFG)
    assert any("up leg not vertical enough" in c for c in checks)


def test_fit_yaw_checks_the_residual_and_input_shape():
    raw = gesture_world()
    raw[2] = raw[1] + [0.25, 0.0, 0.0]  # "forward" moved the same way as "left"
    raw[3] = raw[2] - [0.25, 0.0, 0.0]
    raw[4] = raw[3] - [0.25, 0.0, 0.0]
    raw[5] = raw[4] + [0.0, 0.0, 0.25]
    raw[6] = raw[5] - [0.0, 0.0, 0.25]
    _, residual, checks = fit_yaw(raw, CCFG)
    assert residual > CCFG.yaw_max_residual_deg
    assert any(c.startswith("fit residual") for c in checks)
    with pytest.raises(ValueError):
        fit_yaw(raw[:6], CCFG)
    with pytest.raises(ValueError):
        fit_yaw(np.full((7, 3), np.nan), CCFG)


# -- argument sets / persistence ------------------------------------------------------------------
def test_calibration_argument_sets_strip_the_owned_options():
    assert strip_calibration_args(NORMAL) == ["--lighthousecount", "3"]
    assert strip_calibration_args(
        ["--configfile", "/x.json", "--force-calibrate", "1", "--v", "5",
         "--use-stationary-sensor-window", "0", "--globalscenesolver", "1"]
    ) == ["--v", "5"]
    # A flag given without a value (followed by another option) is stripped alone.
    assert strip_calibration_args(["--disable-calibrate", "--lighthousecount", "2"]) == [
        "--lighthousecount", "2"
    ]
    tmp = "/tmp/base_station-x.json"
    assert capture_args(NORMAL, tmp, force=True) == [
        "--lighthousecount", "3", "--configfile", tmp, "--force-calibrate", "1",
        "--globalscenesolver", "1",
    ]
    assert capture_args(NORMAL, tmp, force=False) == [
        "--lighthousecount", "3", "--configfile", tmp, "--globalscenesolver", "1"
    ]
    assert validation_args(NORMAL, tmp) == [
        "--lighthousecount", "3", "--configfile", tmp, "--globalscenesolver", "0",
        "--disable-calibrate", "1", "--use-stationary-sensor-window", "0",
    ]


def test_lighthouse_channels_reads_index_and_mode_from_the_quasi_json_config():
    assert lighthouse_channels(CONFIG_TEXT) == {0: 3, 1: 7, 2: 14}
    assert lighthouse_channels("") == {} and lighthouse_channels('"v":"0",\n') == {}
    # Block number when "index" is missing; unset mode (255) and mode-less blocks skipped.
    text = (
        '"lighthouse4":{"id":"1","mode":"9"},\n'
        '"lighthouse5":{"index":"5","id":"2","mode":"255"},\n'
        '"lighthouse6":{"index":"6","id":"3"}\n'
    )
    assert lighthouse_channels(text) == {4: 9}


def test_persisted_calibration_roundtrip_and_tolerance(tmp_path, caplog):
    d = tmp_path / "calib"
    assert PersistedCalibration.load(d) == PersistedCalibration()  # missing -> defaults
    rec = PersistedCalibration(
        yaw_deg=102.1, yaw_valid=False, yaw_calibrated_at=1.5,
        base_station_installed_at=2.5, lighthouse_config_sha256="ab" * 32,
    )
    path = rec.save(d)
    assert path == d / "tracker_calibration.json"
    assert json.loads(path.read_text()) == {
        "yaw_deg": 102.1, "yaw_valid": False, "yaw_calibrated_at": 1.5,
        "base_station_installed_at": 2.5, "lighthouse_config_sha256": "ab" * 32,
    }
    assert PersistedCalibration.load(d) == rec
    path.write_text("{not json")
    assert PersistedCalibration.load(d) == PersistedCalibration()
    assert any("unreadable" in r.getMessage() for r in caplog.records)
    path.write_text('{"yaw_deg": "nope", "yaw_valid": 1}')
    assert PersistedCalibration.load(d) == PersistedCalibration(yaw_deg=None, yaw_valid=True)


def test_apply_persisted_yaw_overrides_the_yaml_default_only_when_valid(tmp_path):
    d = tmp_path / "calib"
    cfg = RuntimeConfig(calibration_dir=d, tracker=TrackerConfig(yaw_deg=5.0))
    assert apply_persisted_yaw(cfg).yaw_deg is None and cfg.tracker.yaw_deg == 5.0
    PersistedCalibration(yaw_deg=102.1, yaw_valid=True, yaw_calibrated_at=1.0).save(d)
    apply_persisted_yaw(cfg)
    assert cfg.tracker.yaw_deg == 102.1
    cfg = RuntimeConfig(calibration_dir=d, tracker=TrackerConfig(yaw_deg=5.0))
    PersistedCalibration(yaw_deg=102.1, yaw_valid=False).save(d)  # after a base-station install
    apply_persisted_yaw(cfg)
    assert cfg.tracker.yaw_deg == 5.0
    assert TrackerSettings.from_config(cfg.tracker).get().yaw_deg == 5.0


# -- state machine over a duck-typed fake reader ---------------------------------------------------
class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class FakeReader:
    """What ``TrackerCalibration`` needs from the reader: ``backend``,
    ``restart(args)`` / ``stop()`` (recorded), ``lighthouses()`` (settable),
    ``on_info`` (the FSM's INFO sink) and ``info_lines``."""

    def __init__(self, backend: str = "libsurvive") -> None:
        self.backend = backend
        self.restarts: list[list[str]] = []
        self.stops = 0
        self.on_info = None
        self.info_lines: deque[tuple[float, str]] = deque(maxlen=256)
        self.snapshot: list[LighthouseSnapshot] = []
        self.fail_restart = False
        self.stop_ok = True  # False = the reader thread is stuck in simple_close (join timed out)

    def restart(self, args):
        if self.fail_restart:
            raise RuntimeError("dongle busy")
        self.stops += 1
        self.restarts.append(list(args))

    def stop(self) -> bool:
        self.stops += 1
        return self.stop_ok

    def lighthouses(self):
        return list(self.snapshot)

    def info(self, line: str) -> None:
        self.info_lines.append((0.0, line))
        if self.on_info is not None:
            self.on_info(0.0, line)


class Harness:
    def __init__(self, tmp_path, *, backend="libsurvive", ccfg=None, session=False):
        self.clock = Clock()
        self.wall = Clock(1_700_000_000.0)
        self.session_active = session
        self.reader = FakeReader(backend)
        self.slot: LatestSlot[TrackerSample] = LatestSlot()
        self.config_path = tmp_path / "libsurvive" / "config.json"
        self.calib_dir = tmp_path / "calibration"
        self.cfg = RuntimeConfig(
            calibration_dir=self.calib_dir,
            tracker=TrackerConfig(
                backend=backend, libsurvive_args=NORMAL, libsurvive_config_path=self.config_path,
                calibration=ccfg or TrackerCalibrationConfig(
                    min_scenes=2, validation_seconds=0.1, validation_skip_seconds=0.05,
                ),
            ),
        )
        self.settings = TrackerSettings.from_config(self.cfg.tracker)
        self.cal = TrackerCalibration(
            self.reader, self.settings, self.cfg, self.slot,
            lambda: self.session_active, clock=self.clock, wall=self.wall,
        )
        self.seq = 0

    def cmd(self, kind, op, point=None):
        return self.cal.command(TrackerCalibrationCommand(kind=kind, op=op, point=point))

    def publish(self, pos, *, trigger=None, valid=True, dt=0.01):
        self.clock.t += dt
        self.seq += 1
        ctrl = None
        if trigger is not None:
            ctrl = ControllerState(
                trigger=1.0 if trigger else 0.0, trigger_pressed=trigger, rx_mono=self.clock.t
            )
        s = TrackerSample(
            pose=Pose(np.asarray(pos, float), IDENT), vel_lin=np.zeros(3), vel_ang=np.zeros(3),
            t_dev=self.clock.t, rx_mono=self.clock.t, seq=self.seq, valid=valid,
            controller=ctrl, pose_rx_mono=self.clock.t,
        )
        self.slot.put(s)
        return s

    def wait(self, pred, timeout=3.0, what=""):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            st = self.cal.status()
            if pred(st):
                return st
            time.sleep(0.005)
        st = self.cal.status()
        assert pred(st), (what, st.model_dump())
        return st

    def wait_restarts(self, n):
        deadline = time.monotonic() + 3.0
        while len(self.reader.restarts) < n and time.monotonic() < deadline:
            time.sleep(0.005)
        assert len(self.reader.restarts) == n, self.reader.restarts
        return self.reader.restarts[-1]

    def feed_still(self, pos, n=12, noise_mm=0.3):
        """Publish n still samples one worker period apart (the depth-1 slot is
        read by the 50 Hz worker, so the test paces itself to it)."""
        rng = np.random.default_rng(1)
        for _ in range(n):
            self.publish(np.asarray(pos) + rng.normal(scale=noise_mm / 1000.0, size=3))
            time.sleep(0.025)


def test_base_station_start_guards(tmp_path):
    h = Harness(tmp_path, backend="fake")
    with pytest.raises(CalibrationError, match="backend is not libsurvive"):
        h.cmd("base_station", "start")
    h = Harness(tmp_path, session=True)
    with pytest.raises(CalibrationError, match="stop the session first"):
        h.cmd("base_station", "start")
    with pytest.raises(CalibrationError, match="stop the session first"):
        h.cmd("yaw", "start")
    h = Harness(tmp_path)
    for op in ("capture", "validate", "install", "abort"):
        with pytest.raises(CalibrationError, match="no base_station calibration in progress"):
            h.cmd("base_station", op)
    with pytest.raises(CalibrationError, match="apply is a yaw calibration op"):
        h.cmd("base_station", "apply")
    st = h.cal.status()
    assert st.kind == "none" and st.phase == "idle" and st.yaw_valid is True
    assert st.lighthouses == [] and st.yaw_points == [] and not h.cal.active
    h.cal.close()


def test_base_station_full_flow_capture_validate_install(tmp_path):
    h = Harness(tmp_path)
    h.config_path.parent.mkdir(parents=True)
    h.config_path.write_bytes(ORIGINAL)
    st = h.cmd("base_station", "start")
    assert st.kind == "base_station" and st.phase == "starting" and h.cal.active
    assert st.started_at == h.wall.t and st.elapsed_s == 0.0
    # The stations libsurvive will know (index + channel from the config copy) are listed
    # right away — but none counts as visible until the solver reports light from it.
    assert [(lh.index, lh.channel, lh.scenes) for lh in st.lighthouses] == [
        (0, 3, 0), (1, 7, 0), (2, 14, 0)
    ]
    assert st.stations_visible == 0 and st.scenes == 0
    with pytest.raises(CalibrationError, match="base_station calibration in progress"):
        h.cmd("yaw", "start")
    with pytest.raises(CalibrationError, match="base_station calibration in progress"):
        h.cmd("base_station", "start")
    args = h.wait_restarts(1)
    tmps = sorted(h.calib_dir.glob("base_station-*.json"))
    assert len(tmps) == 1 and re.fullmatch(r"base_station-\d{8}-\d{6}\.json", tmps[0].name)
    tmp = tmps[0]
    assert tmp.read_bytes() == ORIGINAL  # a COPY; libsurvive rewrites this one
    assert args == [
        "--lighthousecount", "3", "--configfile", str(tmp), "--force-calibrate", "1",
        "--globalscenesolver", "1",
    ]
    assert h.reader.on_info is not None
    # 'Force calibrate flag set' flips starting -> capturing (ANSI already stripped by the reader).
    h.reader.info("Info: Force calibrate flag set -- clearing position on all lighthouses")
    st = h.wait(lambda s: s.phase == "capturing", what="capturing")
    assert st.detail == "scenes 0/2 — park the controller still ≥ 3 s at another spot"
    assert st.controller_still is None  # no fresh samples yet
    # Scene counting per station (max over stations), reference station; a station with
    # scenes was seen by the controller -> visible (LH2 is configured but dark).
    h.reader.info("Info: Global solve with 1 scenes for 0 with error of 4.1/9.7 (acc err 0.0002)")
    h.reader.info("Info: Global solve with 1 scenes for 1 with error of 4.1/9.7 (acc err 0.0002)")
    h.reader.info("Info: Using LH 1 (596c9a8b) as reference lighthouse")
    st = h.wait(lambda s: s.scenes == 1 and s.stations_visible == 2, what="scenes")
    assert st.detail.startswith("scenes 1/2")
    assert [lh.index for lh in st.lighthouses] == [0, 1, 2]
    assert [lh.reference for lh in st.lighthouses] == [False, True, False]
    assert [lh.channel for lh in st.lighthouses] == [3, 7, 14]
    assert st.lighthouses[1].serial == "596c9a8b"
    with pytest.raises(CalibrationError, match=r"need 2 scenes, have 1 \(1 more\)"):
        h.cmd("base_station", "validate")
    with pytest.raises(CalibrationError, match="cannot install while capturing"):
        h.cmd("base_station", "install")
    with pytest.raises(CalibrationError, match="cannot capture while capturing"):
        h.cmd("base_station", "capture")
    # The reader's lighthouse snapshot fills serial / pose. libsurvive enumerates every
    # CONFIGURED station (unplugged or not), so three entries do NOT mean three visible.
    lh_pose = Pose(np.array([3.8, 1.2, 1.8]), np.array([0.31, 0.26, 0.47, 0.78]))
    h.reader.snapshot = [
        LighthouseSnapshot(0, "LH0", "LHB-A00E2B4C", lh_pose, 0.0),
        LighthouseSnapshot(1, "LH1", None, None, 0.0),
        LighthouseSnapshot(2, "LH2", "LHB-E9BFDF83", None, 0.0),
    ]
    st = h.wait(lambda s: s.lighthouses[0].pose is not None, what="snapshot")
    assert st.stations_visible == 2 and len(st.lighthouses) == 3
    assert st.lighthouses[0].serial == "LHB-A00E2B4C"
    assert np.allclose(st.lighthouses[0].pose.position, [3.8, 1.2, 1.8])
    assert st.lighthouses[1].serial == "596c9a8b"  # INFO-line serial kept
    assert st.lighthouses[2].serial == "LHB-E9BFDF83" and st.lighthouses[2].pose is None
    # controller_still from the sample window.
    h.feed_still([0.5, 0.1, 0.2])
    st = h.wait(lambda s: s.controller_still is True, what="still")
    for i in range(8):
        h.publish([0.5 + 0.02 * i, 0.1, 0.2])  # 2 cm per sample: moving
        time.sleep(0.025)
    st = h.wait(lambda s: s.controller_still is False, what="moving")
    h.reader.info("Info: Global solve with 2 scenes for 0 with error of 1.0/2.0 (acc err 0.0001)")
    h.reader.info("Info: Global solve with 3 scenes for 1 with error of 1.0/2.0 (acc err 0.0001)")
    st = h.wait(lambda s: s.scenes == 3, what="scenes 3")
    assert [lh.scenes for lh in st.lighthouses] == [2, 3, 0]
    # validate: frozen-solution restart, skip window, then the scatter statistics.
    st = h.cmd("base_station", "validate")
    assert st.phase == "validating" and st.validation is None
    args = h.wait_restarts(2)
    assert args == [
        "--lighthousecount", "3", "--configfile", str(tmp), "--globalscenesolver", "0",
        "--disable-calibrate", "1", "--use-stationary-sensor-window", "0",
    ]
    tmp.write_bytes(b"SOLVED-CONFIG\n")  # what libsurvive would have written on the restart
    time.sleep(0.05)
    assert h.cal.status().detail.startswith("waiting for tracking")
    h.feed_still([0.5, 0.1, 0.2], n=25)  # 0.25 s of samples > skip 0.05 + window 0.1
    st = h.wait(lambda s: s.phase == "done", what="validated")
    assert st.validation is not None and st.validation.passed is True
    assert st.validation.samples >= 2 and max(st.validation.std_mm) < 5.0
    assert st.validation.max_step_mm < 20.0
    assert st.validation.threshold_std_mm == 5.0 and st.validation.threshold_step_mm == 20.0
    assert st.detail == "validation passed — install"
    assert h.cal.active
    # capture more: no --force-calibrate (keep the solution); validation cleared.
    st = h.cmd("base_station", "capture")
    assert st.phase == "starting" and st.validation is None
    args = h.wait_restarts(3)
    assert args == ["--lighthousecount", "3", "--configfile", str(tmp), "--globalscenesolver", "1"]
    h.publish([0.5, 0.1, 0.2])  # first pose after the restart -> capturing (no force line)
    st = h.wait(lambda s: s.phase == "capturing", what="capturing again")
    assert st.scenes == 3  # scene counts survive a capture-more restart
    # A failing validation: a 50 mm step in the window.
    h.cmd("base_station", "validate")
    h.wait_restarts(4)
    h.feed_still([0.5, 0.1, 0.2], n=10)
    h.feed_still([0.55, 0.1, 0.2], n=15)
    st = h.wait(lambda s: s.phase == "done", what="validated (fail)")
    assert st.validation.passed is False and st.validation.max_step_mm > 20.0
    assert st.detail == "validation failed — capture more spots"
    with pytest.raises(CalibrationError, match="validation has not passed"):
        h.cmd("base_station", "install")
    # Validate again (allowed from done), pass, install.
    h.cmd("base_station", "validate")
    h.wait_restarts(5)
    h.feed_still([0.5, 0.1, 0.2], n=25)
    st = h.wait(lambda s: s.phase == "done" and s.validation is not None, what="revalidated")
    assert st.validation.passed is True
    stops_before = h.reader.stops
    st = h.cmd("base_station", "install")
    assert st.phase == "installing"
    st = h.wait(lambda s: s.phase == "done" and s.installed_path is not None, what="installed")
    assert st.detail == "installed — run Yaw alignment"
    assert not h.cal.active and h.reader.on_info is None
    assert h.reader.stops >= stops_before + 2  # stop (flush tmp) + restart(normal)
    assert h.reader.restarts[-1] == NORMAL and len(h.reader.restarts) == 6
    assert st.installed_path == str(h.config_path)
    assert h.config_path.read_bytes() == b"SOLVED-CONFIG\n"
    assert st.backup_path is not None
    backup = tmp_path / st.backup_path
    assert re.fullmatch(r"config\.json\.bak-\d{8}-\d{6}", backup.name), backup
    assert backup.parent == h.config_path.parent and backup.read_bytes() == ORIGINAL
    installed_copy = h.calib_dir / tmp.name.replace(".json", "-installed.json")
    assert installed_copy.read_bytes() == b"SOLVED-CONFIG\n"
    assert tmp.exists()  # kept for diagnosis
    persisted = json.loads((h.calib_dir / "tracker_calibration.json").read_text())
    assert persisted["yaw_valid"] is False and persisted["yaw_deg"] is None
    assert persisted["base_station_installed_at"] == h.wall.t
    assert persisted["lighthouse_config_sha256"] == hashlib.sha256(b"SOLVED-CONFIG\n").hexdigest()
    assert st.yaw_valid is False and st.base_station_installed_at == h.wall.t
    assert st.elapsed_s == 0.0  # frozen at finish (manual wall clock)
    h.cal.close()


def test_base_station_abort_restores_normal_args_and_keeps_tmp(tmp_path):
    h = Harness(tmp_path)  # no official config file yet: tmp is created by libsurvive itself
    h.cmd("base_station", "start")
    args = h.wait_restarts(1)
    tmp = args[args.index("--configfile") + 1]
    assert not (h.calib_dir / tmp).exists() and h.calib_dir.is_dir()
    st = h.cmd("base_station", "abort")
    assert st.phase == "aborted" and not h.cal.active
    assert "normal libsurvive arguments restored" in st.detail and tmp in st.detail
    assert h.wait_restarts(2) == NORMAL
    assert h.reader.on_info is None
    with pytest.raises(CalibrationError, match="no base_station calibration in progress"):
        h.cmd("base_station", "abort")
    # A new start is possible afterwards; a failing restart ends in `failed`.
    h.reader.fail_restart = True
    h.cmd("base_station", "start")
    st = h.wait(lambda s: s.phase == "failed", what="failed")
    assert "dongle busy" in st.detail and not h.cal.active
    h.cal.close()


def test_base_station_validation_without_tracking_times_out(tmp_path, monkeypatch):
    import apollo_mavis_v2_runtime.devices.tracker_calibration as mod

    monkeypatch.setattr(mod, "VALIDATION_TRACKING_TIMEOUT_S", 0.5)
    h = Harness(tmp_path)
    h.cmd("base_station", "start")
    h.wait_restarts(1)
    h.reader.info("Info: Global solve with 5 scenes for 2 with error of 1.0/2.0 (acc err 0.0001)")
    h.publish([0.1, 0.2, 0.3])
    h.wait(lambda s: s.phase == "capturing" and s.scenes == 5, what="capturing")
    h.cmd("base_station", "validate")
    h.wait_restarts(2)
    h.clock.t += 1.0  # nothing tracks after the restart
    st = h.wait(lambda s: s.phase == "done", what="timed out")
    assert st.validation is not None and st.validation.passed is False
    assert st.validation.samples == 0 and "no tracking" in st.detail
    h.cmd("base_station", "abort")
    h.cal.close()


def test_close_mid_calibration_restores_normal_args(tmp_path):
    h = Harness(tmp_path)
    h.cmd("base_station", "start")
    h.wait_restarts(1)
    h.cal.close()
    assert h.reader.restarts[-1] == NORMAL and not h.cal.active
    st = h.cal.status()
    assert st.phase == "aborted" and st.detail == "runtime shutdown"
    with pytest.raises(CalibrationError, match="shutting down"):
        h.cmd("yaw", "start")


def test_stations_visible_counts_light_not_configured_stations(tmp_path):
    """libsurvive creates one LIGHTHOUSE object per station in the config file at
    init (survive_api.c create_lighthouse over activeLighthouses), so the snapshot
    length is 3 the moment capture is up even with a station unplugged. Visible =
    the solver got light from it: scenes solved, a pose appearing during the run,
    a station added / decoded from light."""
    h = Harness(tmp_path)
    h.config_path.parent.mkdir(parents=True)
    h.config_path.write_bytes(ORIGINAL)
    h.cmd("base_station", "start")
    h.wait_restarts(1)
    lh_pose = Pose(np.array([3.8, 1.2, 1.8]), np.array([0.31, 0.26, 0.47, 0.78]))
    # First snapshot of the run = baseline: poses already present came from the file.
    h.reader.snapshot = [
        LighthouseSnapshot(0, "LH0", "LHB-A00E2B4C", lh_pose, 0.0),
        LighthouseSnapshot(1, "LH1", "LHB-59703E8B", None, 0.0),
        LighthouseSnapshot(2, "LH2", "LHB-E9BFDF83", None, 0.0),
    ]
    h.publish([0.1, 0.2, 0.3])  # first pose -> capturing
    st = h.wait(lambda s: s.phase == "capturing" and s.lighthouses[0].pose is not None)
    assert len(st.lighthouses) == 3 and st.stations_visible == 0
    assert [lh.channel for lh in st.lighthouses] == [3, 7, 14]
    # A pose APPEARING during the run (unsolved -> solved) means the solver saw the station.
    h.reader.snapshot = [
        LighthouseSnapshot(0, "LH0", "LHB-A00E2B4C", lh_pose, 0.0),
        LighthouseSnapshot(1, "LH1", "LHB-59703E8B", lh_pose, 0.0),
        LighthouseSnapshot(2, "LH2", "LHB-E9BFDF83", None, 0.0),
    ]
    st = h.wait(lambda s: s.stations_visible == 1, what="LH1 solved")
    assert st.lighthouses[1].pose is not None
    # Scenes solved for a station: seen.
    h.reader.info("Info: Global solve with 1 scenes for 0 with error of 4.1/9.7 (acc err 0.0002)")
    st = h.wait(lambda s: s.stations_visible == 2, what="LH0 scenes")
    # A station NOT in the config, created from its light, with its channel; the OOTX line
    # for that channel is attributed to it (no double count).
    h.reader.info("Info: Adding lighthouse ch 5 (idx: 3, cnt: 4)")
    h.reader.info("Info: OOTX not set for LH in channel 5; attaching ootx decoder using device WM0")
    st = h.wait(lambda s: s.stations_visible == 3 and len(s.lighthouses) == 4, what="LH3 added")
    assert (st.lighthouses[3].index, st.lighthouses[3].channel) == (3, 5)
    time.sleep(0.05)
    assert h.cal.status().stations_visible == 3
    # A channel seen from light that no known station has counts as one extra station;
    # the gen1 wording names the index directly.
    h.reader.info("Info: OOTX not set for LH in channel 9; attaching ootx decoder using device WM0")
    st = h.wait(lambda s: s.stations_visible == 4, what="unknown channel")
    h.reader.info("Info: OOTX not set for LH 2; attaching ootx decoder using device WM0")
    st = h.wait(lambda s: s.stations_visible == 5, what="gen1 index")
    assert [lh.index for lh in st.lighthouses] == [0, 1, 2, 3]
    # Visibility is per calibration: a validate restart (poses all loaded from the file
    # again) neither resets nor inflates it.
    h.reader.info("Info: Global solve with 2 scenes for 0 with error of 1.0/2.0 (acc err 0.0001)")
    h.wait(lambda s: s.scenes == 2)
    h.cmd("base_station", "validate")
    h.wait_restarts(2)
    h.reader.snapshot = [
        LighthouseSnapshot(i, f"LH{i}", None, lh_pose, 0.0) for i in range(4)
    ]
    time.sleep(0.1)
    assert h.cal.status().stations_visible == 5
    h.cmd("base_station", "abort")
    h.cal.close()


def _to_validation(h: Harness, dt: float = 0.01):
    """start -> capturing (one scene) -> validate -> first pose after the restart."""
    n = len(h.reader.restarts)
    h.cmd("base_station", "start")
    h.wait_restarts(n + 1)
    h.reader.info("Info: Global solve with 5 scenes for 0 with error of 1.0/2.0 (acc err 0.0001)")
    h.publish([0.5, 0.1, 0.2])
    h.wait(lambda s: s.phase == "capturing" and s.scenes == 5, what="capturing")
    h.cmd("base_station", "validate")
    h.wait_restarts(n + 2)
    h.publish([0.5, 0.1, 0.2], dt=dt)  # val_t0
    time.sleep(0.05)


def test_validation_fails_when_tracking_drops_out_inside_the_window(tmp_path):
    """Two still samples inside the collect window, then the controller is occluded:
    the scatter of two points is ~0 mm but that is no evidence — 'tracking lost',
    not 'passed — install'."""
    ccfg = TrackerCalibrationConfig(
        min_scenes=1, validation_seconds=1.0, validation_skip_seconds=0.1
    )
    h = Harness(tmp_path, ccfg=ccfg)
    _to_validation(h)
    h.publish([0.5, 0.1, 0.2], dt=0.15)  # inside the collect window (skip 0.1 s)
    time.sleep(0.03)
    h.publish([0.5, 0.1, 0.2], dt=0.05)
    time.sleep(0.03)
    assert h.cal.status().phase == "validating"
    h.clock.t += 2.0  # occluded: no more samples; the window elapses
    st = h.wait(lambda s: s.phase == "done", what="window over")
    assert st.validation is not None and st.validation.passed is False
    assert st.validation.samples == 2 and max(st.validation.std_mm) < 0.01
    assert st.detail.startswith("validation failed — tracking lost (")
    assert "2 samples" in st.detail
    with pytest.raises(CalibrationError, match="validation has not passed"):
        h.cmd("base_station", "install")
    # A gap between adjacent samples fails the same way ...
    h.cmd("base_station", "validate")
    h.wait_restarts(3)
    h.publish([0.5, 0.1, 0.2])
    for _ in range(6):
        h.publish([0.5, 0.1, 0.2], dt=0.05)  # 0.1 .. 0.3: tracked
        time.sleep(0.025)
    h.publish([0.5, 0.1, 0.2], dt=0.9)  # 0.9 s without a pose, then the window is over
    st = h.wait(lambda s: s.phase == "done", what="gap")
    assert st.validation.passed is False and "tracking lost (0.9 s gap" in st.detail
    # ... a 20 Hz still stream over the whole window passes.
    h.cmd("base_station", "validate")
    h.wait_restarts(4)
    h.publish([0.5, 0.1, 0.2])
    for _ in range(25):
        h.publish([0.5, 0.1, 0.2], dt=0.05)
        time.sleep(0.025)
    st = h.wait(lambda s: s.phase == "done", what="healthy")
    assert st.validation.passed is True and st.validation.samples >= 10
    assert st.detail == "validation passed — install"
    h.cmd("base_station", "abort")
    h.cal.close()


def test_validation_fails_with_too_few_samples_for_the_window(tmp_path):
    """No gap wider than VALIDATION_MAX_GAP_S, but a 5 Hz trickle over a 1 s window
    is far below what a tracked controller produces (>= VALIDATION_MIN_RATE_HZ)."""
    ccfg = TrackerCalibrationConfig(
        min_scenes=1, validation_seconds=1.0, validation_skip_seconds=0.1
    )
    h = Harness(tmp_path, ccfg=ccfg)
    _to_validation(h)
    for _ in range(6):
        h.publish([0.5, 0.1, 0.2], dt=0.2)  # 0.2 .. 1.2 s: 6 samples in a 1 s window
        time.sleep(0.03)
    st = h.wait(lambda s: s.phase == "done", what="window over")
    assert st.validation.passed is False and st.validation.samples == 6
    assert st.detail.startswith("validation failed — only 6 samples (need ≥ 10)")
    h.cmd("base_station", "abort")
    h.cal.close()


def test_install_fails_when_the_reader_does_not_stop(tmp_path):
    """``install`` reads the temporary config that ``simple_close`` flushes; a reader
    thread still stuck in the close (stop() False) must not be trusted — the
    official file stays untouched and the flow fails with the reason."""
    h = Harness(tmp_path)
    h.config_path.parent.mkdir(parents=True)
    h.config_path.write_bytes(ORIGINAL)
    _to_validation(h)
    h.feed_still([0.5, 0.1, 0.2], n=25)
    st = h.wait(lambda s: s.phase == "done" and s.validation is not None, what="validated")
    assert st.validation.passed is True
    h.reader.stop_ok = False
    h.cmd("base_station", "install")
    st = h.wait(lambda s: s.phase == "failed", what="failed")
    assert "did not stop" in st.detail and not h.cal.active
    assert st.installed_path is None and h.config_path.read_bytes() == ORIGINAL
    assert list(h.config_path.parent.glob("config.json.bak-*")) == []
    assert h.reader.restarts[-1] == NORMAL
    assert not (h.calib_dir / "tracker_calibration.json").exists()  # nothing persisted
    h.cal.close()


def test_session_active_covers_bringup_so_calibration_cannot_slip_in(tmp_path, monkeypatch):
    """``SessionManager.session`` is assigned only AFTER bringup returns; the
    calibration guard reads ``session_active``, which is raised under the manager
    lock before validation/bringup and cleared even when bringup fails."""
    from conftest import make_runtime_config

    from apollo_mavis_v2_runtime.runtime import Runtime
    from apollo_mavis_v2_runtime.session.manager import SessionError
    from apollo_mavis_v2_runtime.session.types import SessionSpec

    cfg = make_runtime_config(tmp_path, scene="mavis_v2", tracker=TrackerConfig(backend="none"))
    rt = Runtime(cfg)
    try:
        seen: dict[str, object] = {}

        def bringup_probe(spec, wc):
            seen["active"] = rt.manager.session_active
            seen["session"] = rt.manager.session
            with pytest.raises(CalibrationError, match="stop the session first"):
                rt.tracker_calibration.command(TrackerCalibrationCommand(kind="yaw", op="start"))
            raise SessionError("bringup aborted by the test")

        monkeypatch.setattr(rt.manager, "_bringup_sim", bringup_probe)
        assert rt.manager.session_active is False
        spec = SessionSpec(
            mode="teleop", kind="sim", arms=["view", "grip"],
            frames={"view": "arm_base:view", "grip": "arm_base:grip"}, sim_scene="mavis_v2",
        )
        with pytest.raises(SessionError, match="aborted by the test"):
            rt.manager.create(spec)
        assert seen == {"active": True, "session": None}
        assert rt.manager.session_active is False and rt.manager.session is None
        st = rt.tracker_calibration.command(TrackerCalibrationCommand(kind="yaw", op="start"))
        assert st.phase == "capturing"
    finally:
        rt.stop()


# -- yaw -------------------------------------------------------------------------------------------
def _capture_at(h: Harness, pos, *, via: str, point=None):
    """Move the (synthetic) controller to ``pos`` and capture: ``via='rest'`` uses
    the command, ``via='trigger'`` a trigger rising edge on the sample stream."""
    h.clock.t += 1.0  # previous point's samples fall out of the averaging window
    for _ in range(3):
        h.publish(pos, trigger=False if via == "trigger" else None)
    if via == "rest":
        time.sleep(0.03)
        return h.cmd("yaw", "capture", point)
    n = len(h.cal.status().yaw_points)
    h.publish(pos, trigger=True)  # rising edge
    return h.wait(lambda s: len(s.yaw_points) == n + 1, what=f"trigger capture {point}")


def test_yaw_full_flow_trigger_edges_rest_capture_apply_and_startup_override(tmp_path):
    h = Harness(tmp_path, backend="fake")
    yaw_true = 102.1
    raw = gesture_raw(yaw_true)
    st = h.cmd("yaw", "start")
    assert st.kind == "yaw" and st.phase == "capturing" and st.next_point == "start"
    assert st.yaw_points == [] and h.cal.active and "START" in st.detail
    for op in ("validate", "install"):
        with pytest.raises(CalibrationError, match="base-station calibration op"):
            h.cmd("yaw", op)
    with pytest.raises(CalibrationError, match="no fresh tracker sample"):
        h.cmd("yaw", "capture")  # nothing published yet
    with pytest.raises(CalibrationError, match="capture all 7 points first"):
        h.cmd("yaw", "apply")
    # start via REST (point omitted = next_point), left via a trigger edge, ...
    st = _capture_at(h, raw[0], via="rest")
    assert [p.label for p in st.yaw_points] == ["start"] and st.next_point == "left"
    assert np.allclose(st.yaw_points[0].pose.position, raw[0], atol=1e-9)
    assert st.detail.startswith("captured start — move LEFT")
    st = _capture_at(h, raw[1], via="trigger", point="left")
    assert [p.label for p in st.yaw_points] == ["start", "left"]
    # ... a HELD trigger is not a second click, a stale press is ignored ...
    h.publish(raw[1], trigger=True)
    h.publish(raw[1], trigger=True)
    time.sleep(0.08)
    assert len(h.cal.status().yaw_points) == 2
    h.publish(raw[1], trigger=False)
    time.sleep(0.05)
    # ... the wrong explicit label is refused, the right one accepted.
    with pytest.raises(CalibrationError, match="next point is 'forward'"):
        _capture_at(h, raw[2], via="rest", point="up")
    st = _capture_at(h, raw[2], via="rest", point="forward")
    st = _capture_at(h, raw[3], via="trigger", point="right")
    st = _capture_at(h, raw[4], via="rest")
    st = _capture_at(h, raw[5], via="trigger", point="up")
    assert st.next_point == "down" and st.phase == "capturing" and st.fitted_yaw_deg is None
    # Capture averages the raw positions inside yaw_capture_average_s (0.3 s); the
    # slot is depth 1, so pace the publishes to the 50 Hz worker.
    h.clock.t += 1.0
    h.publish(raw[6] + [0.002, 0.0, 0.0])
    time.sleep(0.05)
    h.publish(raw[6] - [0.002, 0.0, 0.0])
    time.sleep(0.05)
    st = h.cmd("yaw", "capture")
    assert [p.label for p in st.yaw_points] == list(YAW_POINT_ORDER)
    assert np.allclose(st.yaw_points[6].pose.position, raw[6], atol=1e-9)
    assert st.phase == "done" and st.next_point is None
    assert st.fitted_yaw_deg == pytest.approx(yaw_true, abs=1e-6)
    assert st.fit_residual_deg == pytest.approx(0.0, abs=1e-6) and st.fit_checks == []
    assert st.applied_yaw_deg is None and h.cal.active
    assert h.settings.get().yaw_deg == 0.0  # not applied yet
    with pytest.raises(CalibrationError, match="all 7 points captured"):
        h.cmd("yaw", "capture")
    h.publish(raw[6], trigger=True)  # a trigger click in `done` changes nothing
    time.sleep(0.05)
    assert len(h.cal.status().yaw_points) == 7
    st = h.cmd("yaw", "apply")
    assert st.phase == "done" and st.applied_yaw_deg == pytest.approx(yaw_true, abs=1e-6)
    assert not h.cal.active and st.yaw_valid is True and st.yaw_calibrated_at == h.wall.t
    assert h.settings.get().yaw_deg == pytest.approx(yaw_true, abs=1e-6)
    persisted = json.loads((h.calib_dir / "tracker_calibration.json").read_text())
    assert persisted["yaw_deg"] == pytest.approx(yaw_true, abs=1e-6)
    assert persisted["yaw_valid"] is True and persisted["yaw_calibrated_at"] == h.wall.t
    assert persisted["base_station_installed_at"] is None
    # Startup override: a new process over the same calibration_dir seeds yaw from the file.
    cfg2 = RuntimeConfig(calibration_dir=h.calib_dir, tracker=TrackerConfig(yaw_deg=0.0))
    apply_persisted_yaw(cfg2)
    assert cfg2.tracker.yaw_deg == pytest.approx(yaw_true, abs=1e-6)
    cal2 = TrackerCalibration(
        FakeReader("fake"), TrackerSettings.from_config(cfg2.tracker), cfg2, LatestSlot(),
        lambda: False,
    )
    st2 = cal2.status()
    assert st2.kind == "none" and st2.yaw_valid is True and st2.yaw_calibrated_at == h.wall.t
    cal2.close()
    h.cal.close()


def test_yaw_bad_gesture_blocks_apply_and_start_redoes(tmp_path):
    h = Harness(tmp_path, backend="fake")
    h.cmd("yaw", "start")
    raw = gesture_raw(30.0, leg=0.04)  # 4 cm legs: too short
    for i in range(7):
        _capture_at(h, raw[i], via="rest")
    st = h.cal.status()
    assert st.phase == "done" and st.fitted_yaw_deg == pytest.approx(30.0, abs=1e-6)
    assert len(st.fit_checks) == 4 and all("too short" in c for c in st.fit_checks)
    assert "checks failed" in st.detail
    with pytest.raises(CalibrationError, match="fit checks failed: left leg too short"):
        h.cmd("yaw", "apply")
    assert h.settings.get().yaw_deg == 0.0 and h.cal.active
    # Restart the gesture (Redo): points cleared, still active; abort clears everything.
    st = h.cmd("yaw", "start")
    assert st.phase == "capturing" and st.yaw_points == [] and st.next_point == "start"
    assert st.fitted_yaw_deg is None and st.fit_checks == [] and h.cal.active
    _capture_at(h, raw[0], via="rest")
    st = h.cmd("yaw", "abort")
    assert st.phase == "aborted" and st.yaw_points == [] and not h.cal.active
    assert st.yaw_valid is True and not (h.calib_dir / "tracker_calibration.json").exists()
    with pytest.raises(CalibrationError, match="no yaw calibration in progress"):
        h.cmd("yaw", "capture")
    h.cal.close()


def test_yaw_start_with_a_held_trigger_does_not_capture_immediately(tmp_path):
    h = Harness(tmp_path, backend="fake")
    h.publish([0.1, 0.2, 0.3], trigger=True)
    h.cmd("yaw", "start")
    h.publish([0.1, 0.2, 0.3], trigger=True)
    time.sleep(0.08)
    assert h.cal.status().yaw_points == []
    h.publish([0.1, 0.2, 0.3], trigger=False)
    time.sleep(0.05)
    h.publish([0.1, 0.2, 0.3], trigger=True)
    st = h.wait(lambda s: len(s.yaw_points) == 1, what="edge after release")
    assert st.yaw_points[0].label == "start"
    # Invalid (jump) samples are never averaged into a point.
    h.clock.t += 1.0
    h.publish([0.9, 0.9, 0.9], valid=False)
    time.sleep(0.03)
    with pytest.raises(CalibrationError, match="no fresh tracker sample"):
        h.cmd("yaw", "capture")
    h.cal.close()


def test_fit_matches_align_pose_convention_for_the_lab_stance():
    """CLAUDE.md: the operator stands at +Y facing -Y; a hand moved LEFT (+X in
    the world) must move the EE to +X after ``align_pose`` with the fitted yaw."""
    raw = gesture_raw(-77.9 + 180.0)
    yaw, _, _ = fit_yaw(raw, CCFG)
    assert yaw == pytest.approx(102.1, abs=1e-6)
    hand_left_raw = raw[1] - raw[0]
    assert np.allclose(_rz(yaw, hand_left_raw), [0.25, 0.0, 0.0], atol=1e-9)
    assert math.isclose(np.linalg.norm(hand_left_raw), 0.25)
