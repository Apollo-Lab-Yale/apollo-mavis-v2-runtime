"""One Euro pose filter (13-tracker-teleop §4): jitter rejection at rest, low lag
in motion, rest deadband, orientation slerp behaviour, reset/retune."""

from __future__ import annotations

import numpy as np
from apollo_mavis_v2_core import Pose, se3

from apollo_mavis_v2_runtime.control.pose_filter import (
    OneEuroVector,
    PoseFilter,
    PoseFilterConfig,
    smoothing_factor,
)

DT = 0.01  # 100 Hz tick


def _run(f: PoseFilter, poses: list[Pose]) -> list[Pose]:
    return [f.step(p, i * DT) for i, p in enumerate(poses)]


def test_smoothing_factor_monotone_in_cutoff():
    assert 0.0 < smoothing_factor(1.0, DT) < smoothing_factor(10.0, DT) < 1.0


def test_resting_jitter_is_suppressed_by_an_order_of_magnitude():
    rng = np.random.default_rng(0)
    centre = np.array([0.3, -0.1, 0.9])
    noise = rng.normal(0.0, 0.003, size=(600, 3))  # 3 mm lighthouse jitter
    raw = [Pose(centre + n, np.array([1.0, 0.0, 0.0, 0.0])) for n in noise]
    f = PoseFilter(PoseFilterConfig(deadband_m=0.0, deadband_rad=0.0))
    out = _run(f, raw)
    raw_std = np.std(noise[200:], axis=0).mean()
    out_std = np.std([p.position for p in out[200:]], axis=0).mean()
    assert out_std < raw_std / 4  # ~1 Hz first-order at 100 Hz: std / 5-6 expected
    assert np.linalg.norm(np.mean([p.position for p in out[200:]], axis=0) - centre) < 1e-3


def test_fast_motion_passes_with_small_lag():
    # constant 0.3 m/s ramp along x: One Euro raises the cutoff with speed
    raw = [Pose(np.array([0.3 * i * DT, 0.0, 0.0]), np.array([1.0, 0, 0, 0])) for i in range(300)]
    f = PoseFilter(PoseFilterConfig(deadband_m=0.0, deadband_rad=0.0, beta=0.5))
    out = _run(f, raw)
    lag_m = raw[-1].position[0] - out[-1].position[0]
    assert 0.0 <= lag_m < 0.03  # < 3 cm behind at 0.3 m/s
    # and the same ramp with beta = 0 (plain 1 Hz low-pass) lags much more
    f0 = PoseFilter(PoseFilterConfig(deadband_m=0.0, deadband_rad=0.0, beta=0.0))
    out0 = _run(f0, raw)
    assert raw[-1].position[0] - out0[-1].position[0] > lag_m * 2


def test_rest_deadband_yields_exactly_zero_motion():
    rng = np.random.default_rng(1)
    centre = np.array([0.1, 0.2, 0.8])
    raw = [Pose(centre + rng.normal(0, 0.0004, 3), np.array([1.0, 0, 0, 0])) for _ in range(300)]
    f = PoseFilter(PoseFilterConfig(deadband_m=0.002, deadband_rad=0.005))
    out = _run(f, raw)
    # after the first emission every returned pose is the SAME object -> no creep
    assert all(o is out[50] for o in out[50:])


def test_deadband_releases_on_real_motion():
    f = PoseFilter(PoseFilterConfig(deadband_m=0.002))
    a = f.step(Pose(np.zeros(3), np.array([1.0, 0, 0, 0])), 0.0)
    for i in range(1, 200):  # move 5 cm over 2 s
        b = f.step(Pose(np.array([0.05 * i / 200, 0, 0]), np.array([1.0, 0, 0, 0])), i * DT)
    assert b is not a and b.position[0] > 0.03


def test_orientation_tracks_slow_rotation_and_rejects_jitter():
    rng = np.random.default_rng(2)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    # jitter of ~0.5 deg around identity
    raw = [Pose(np.zeros(3), se3.rotvec_to_quat(rng.normal(0, 0.009, 3))) for _ in range(400)]
    f = PoseFilter(PoseFilterConfig(deadband_m=0.0, deadband_rad=0.0))
    out = _run(f, raw)
    raw_ang = np.mean([se3.quat_geodesic(p.orientation, q0) for p in raw[200:]])
    out_ang = np.mean([se3.quat_geodesic(p.orientation, q0) for p in out[200:]])
    assert out_ang < raw_ang / 5
    # slow constant rotation about z (0.5 rad/s) is followed with bounded lag
    f.reset()
    raw = [
        Pose(np.zeros(3), se3.rotvec_to_quat(np.array([0, 0, 0.5 * i * DT]))) for i in range(300)
    ]
    out = _run(f, raw)
    lag = se3.quat_geodesic(out[-1].orientation, raw[-1].orientation)
    assert lag < 0.15  # < ~9 deg behind at 0.5 rad/s


def test_reset_forgets_state_and_disabled_is_passthrough():
    f = PoseFilter()
    f.step(Pose(np.array([1.0, 0, 0]), np.array([1.0, 0, 0, 0])), 0.0)
    f.reset()
    p = Pose(np.array([5.0, 0, 0]), np.array([1.0, 0, 0, 0]))
    assert np.allclose(f.step(p, 1.0).position, p.position)  # first sample after reset: passthrough
    g = PoseFilter(PoseFilterConfig(enabled=False))
    assert g.step(p, 0.0) is p


def test_non_monotonic_timestamps_hold_the_estimate():
    v = OneEuroVector(1.0, 0.0, 1.0)
    v.step(np.array([0.0]), 0.0)
    a = v.step(np.array([1.0]), DT)
    b = v.step(np.array([5.0]), DT)  # same stamp: ignored
    assert np.allclose(a, b)


def test_retune_changes_smoothing_without_reset():
    f = PoseFilter(PoseFilterConfig(deadband_m=0.0, deadband_rad=0.0))
    raw = [Pose(np.array([0.3 * i * DT, 0, 0]), np.array([1.0, 0, 0, 0])) for i in range(200)]
    for i, p in enumerate(raw):
        f.step(p, i * DT)
    f.retune(min_cutoff_hz=20.0, beta=1.0)  # much more responsive
    for i, p in enumerate(raw[:50]):
        out = f.step(Pose(p.position + 0.5, p.orientation), 2.0 + i * DT)
    assert abs(out.position[0] - (raw[49].position[0] + 0.5)) < 0.01
