"""GELLO calibration (phase-15; 16-gello §4 / D10): the π/2 offset rounding, the gripper
fraction, and the var/gello_calibration.json store (round trip, atomic write, clear,
corrupt-file tolerance)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from apollo_mavis_v2_runtime.gello.calibration import (
    EMPTY_CALIBRATION,
    QUARTER_TURN_RAD,
    GelloCalibration,
    GelloCalibrationStore,
    gripper_frac,
    match_arm_offsets,
)

PI = math.pi


def test_match_arm_offsets_snap_to_the_nearest_quarter_turn():
    q_arm = np.array([PI, 0.1, -0.2, 0.5, 0.0, 0.3, -0.4])
    signs = np.array([1, -1, 1, 1, -1, 1, 1])
    true_offsets = np.array([PI / 2, PI, 0.0, -PI / 2, 3 * PI / 2, 0.0, -PI])
    # raw = sign * q_arm + offset  (so sign * (raw - offset) == q_arm), plus operator error
    noise = np.array([0.1, -0.12, 0.05, 0.2, -0.3, 0.0, 0.15])  # well below pi/4
    raw = signs * q_arm + true_offsets + noise
    off = match_arm_offsets(raw, q_arm, signs)
    assert np.allclose(off, true_offsets)
    assert np.allclose(off / QUARTER_TURN_RAD, np.round(off / QUARTER_TURN_RAD))
    assert np.allclose(signs * (raw - off) - q_arm, noise * signs)  # residual = the posing error
    assert QUARTER_TURN_RAD == PI / 2
    with pytest.raises(ValueError):
        match_arm_offsets([float("nan")] * 7, q_arm, signs)
    with pytest.raises(ValueError):
        match_arm_offsets([0.0] * 6, q_arm, signs)


def test_gripper_frac_is_clipped_and_none_without_both_endpoints():
    assert gripper_frac(1.0, None, 0.0) is None
    assert gripper_frac(1.0, 2.0, None) is None
    assert gripper_frac(None, 2.0, 0.0) is None
    assert gripper_frac(1.0, 1.0, 1.0) is None  # degenerate endpoints
    assert gripper_frac(1.0, 2.0, 0.0) == 0.5
    assert gripper_frac(2.5, 2.0, 0.0) == 1.0 and gripper_frac(-1.0, 2.0, 0.0) == 0.0
    assert gripper_frac(0.5, 0.0, 2.0) == 0.75  # inverted endpoints (open < closed) work


def test_store_round_trip_clear_and_corrupt_file_tolerance(tmp_path):
    store = GelloCalibrationStore(tmp_path / "var" / "gello_calibration.json")
    assert store.load() == EMPTY_CALIBRATION and not store.exists()
    assert not EMPTY_CALIBRATION.calibrated and not EMPTY_CALIBRATION.gripper_calibrated
    cal = store.update(joint_offsets_rad=(PI / 2, 0.0, 0.0, -PI, 0.0, 0.0, 0.0))
    assert cal.calibrated and cal.saved_at and store.exists()
    on_disk = json.loads(store.path.read_text())
    assert set(on_disk) == {
        "joint_offsets_rad", "gripper_open_rad", "gripper_closed_rad", "saved_at",
    }
    assert on_disk["joint_offsets_rad"][3] == -PI and on_disk["gripper_open_rad"] is None
    assert not list(store.path.parent.glob("*.tmp"))  # atomic write left no temp file
    cal2 = store.update(gripper_open_rad=1.5)
    assert cal2.joint_offsets_rad == cal.joint_offsets_rad and cal2.gripper_open_rad == 1.5
    assert not cal2.gripper_calibrated
    cal3 = store.update(gripper_closed_rad=0.25)
    assert cal3.gripper_calibrated and store.load() == cal3
    assert store.clear() is True and not store.exists() and store.load() == EMPTY_CALIBRATION
    assert store.clear() is False  # idempotent
    # a corrupt / malformed file reads as uncalibrated and is reported, never raised
    store.path.write_text("{not json")
    assert store.load() == EMPTY_CALIBRATION and "unreadable" in store.last_error
    store.path.write_text(json.dumps({"joint_offsets_rad": [0.0] * 6}))
    assert store.load() == EMPTY_CALIBRATION and "7 floats" in store.last_error
    store.path.write_text("[]")
    assert store.load() == EMPTY_CALIBRATION and "JSON object" in store.last_error
    store.save(GelloCalibration((0.0,) * 7))
    assert store.last_error == "" and store.load().calibrated
    with pytest.raises(ValueError):
        GelloCalibration.from_json({"gripper_open_rad": float("inf")})
