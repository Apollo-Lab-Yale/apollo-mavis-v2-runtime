"""TrackerReader backends (none/fake/libsurvive-missing), jump detection,
live settings, and the pysurvive import confinement (13-tracker §2/§4)."""

from __future__ import annotations

import ast
import math
import os
import sys
import time
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from apollo_xarm7_core import LatestSlot, Pose

import apollo_xarm7_runtime
from apollo_xarm7_runtime.config import TrackerConfig
from apollo_xarm7_runtime.devices.tracker import (
    FAKE_CENTER,
    FAKE_PERIOD_S,
    FAKE_RADIUS_M,
    TrackerReader,
    TrackerSettings,
)

SRC = Path(apollo_xarm7_runtime.__file__).parent


def _wait(pred, timeout_s: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


def test_none_backend_reports_no_backend_and_never_threads():
    reader = TrackerReader(TrackerConfig(backend="none"), LatestSlot())
    reader.start()
    st = reader.status(0.0)
    assert st.backend == "none" and st.status == "no_backend" and "none" in st.detail
    assert st.pose_raw is None and st.seq == 0 and st.age_s is None and st.object_name == ""
    assert reader._thread is None
    reader.stop()


def test_fake_backend_emits_valid_circle_at_100hz():
    slot: LatestSlot = LatestSlot()
    reader = TrackerReader(TrackerConfig(backend="fake"), slot)
    reader.start()
    try:
        assert _wait(lambda: reader.status().seq >= 60, 4.0)
        st = reader.status()
        assert st.backend == "fake" and st.status == "tracking" and st.object_name == "WM0"
        assert 70.0 <= st.rate_hz <= 130.0, st.rate_hz
        assert st.age_s is not None and st.age_s < 0.1
        sample = slot.get()[0]
        assert sample.valid and sample.seq == st.seq
        r = float(np.linalg.norm(sample.pose.position - FAKE_CENTER))
        assert abs(r - FAKE_RADIUS_M) < 1e-9
        assert np.allclose(sample.pose.orientation, [1.0, 0.0, 0.0, 0.0])
        speed = FAKE_RADIUS_M * 2.0 * math.pi / FAKE_PERIOD_S
        assert abs(float(np.linalg.norm(sample.vel_lin)) - speed) < 1e-9
        assert np.allclose(sample.vel_ang, 0.0) and sample.t_dev >= 0.0
    finally:
        reader.stop()
    assert reader._thread is None
    # No new samples after stop: the newest one ages into "stale".
    assert reader.status(time.monotonic() + 1.0).status == "stale"


def test_publish_marks_jump_invalid_then_recovers():
    slot: LatestSlot = LatestSlot()
    reader = TrackerReader(TrackerConfig(backend="fake", max_jump_m=0.1), slot)
    ident = np.array([1.0, 0.0, 0.0, 0.0])
    s1 = reader._publish(Pose(np.zeros(3), ident), np.zeros(3), np.zeros(3), 0.0)
    s2 = reader._publish(Pose(np.array([0.5, 0.0, 0.0]), ident), np.zeros(3), np.zeros(3), 0.01)
    s3 = reader._publish(Pose(np.array([0.5, 0.01, 0.0]), ident), np.zeros(3), np.zeros(3), 0.02)
    assert (s1.valid, s2.valid, s3.valid) == (True, False, True)
    assert (s1.seq, s2.seq, s3.seq) == (1, 2, 3)
    assert slot.get()[0] is s3
    assert reader.status().status == "tracking"


def test_settings_thread_safe_snapshot_and_partial_update():
    s = TrackerSettings.from_config(TrackerConfig(yaw_deg=10.0, pos_scale=1.5))
    v0 = s.get()
    assert (v0.yaw_deg, v0.pos_scale, v0.follow_rotation) == (10.0, 1.5, True)
    v1 = s.update(pos_scale=2.0, follow_rotation=False)
    assert (v1.yaw_deg, v1.pos_scale, v1.follow_rotation) == (10.0, 2.0, False)
    assert v0.pos_scale == 1.5  # snapshots are immutable
    assert s.update() == v1  # omitted = unchanged
    with pytest.raises(FrozenInstanceError):
        v1.pos_scale = 3.0  # frozen dataclass


def test_libsurvive_backend_without_pysurvive_is_no_backend(monkeypatch):
    monkeypatch.setitem(sys.modules, "pysurvive", None)  # import -> ImportError
    reader = TrackerReader(TrackerConfig(backend="libsurvive"), LatestSlot())
    reader.start()
    try:
        assert _wait(lambda: reader.status().status != "starting")
        st = reader.status()
        assert st.status == "no_backend" and "pysurvive" in st.detail
    finally:
        reader.stop()


@pytest.mark.skipif(
    os.environ.get("APOLLO_TRACKER_HW") != "1",
    reason="real libsurvive/dongle probe; set APOLLO_TRACKER_HW=1",
)
def test_libsurvive_backend_never_crashes_on_this_box():
    reader = TrackerReader(TrackerConfig(backend="libsurvive"), LatestSlot())
    reader.start()
    try:
        assert _wait(lambda: reader.status().status not in ("starting", "searching"), 8.0)
        assert reader.status().status in ("tracking", "error", "no_backend")
    finally:
        reader.stop()


def test_pysurvive_import_confined_to_devices_tracker():
    offenders = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(n.split(".")[0] == "pysurvive" for n in names):
                offenders.append(str(path.relative_to(SRC)))
    assert offenders == ["devices/tracker.py"], offenders
