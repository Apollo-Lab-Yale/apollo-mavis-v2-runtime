"""The shipped ``configs/*.yaml`` load and carry the lab facts (2026-09-04):
two control boxes (Manipulation Arm ``grip`` 192.168.1.201 with the force-capable
Gripper G2, Perception Arm ``view`` 192.168.2.219 with the microphone), camera
paths still placeholders, user-facing microphone label."""

from __future__ import annotations

from pathlib import Path

import pytest

from apollo_mavis_v2_runtime.config import load_runtime_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_mavis_v2_hardware_workcell_matches_the_lab():
    cfg = load_runtime_config(CONFIGS / "mavis_v2.yaml")
    hw = cfg.workcells["hardware"]
    assert hw.kind == "hardware" and hw.digital_twin_scene == "mavis_v2"
    by_id = {a.id: a for a in hw.arms}
    assert [a.id for a in hw.arms] == ["grip", "view"]  # Manipulation Arm first
    assert (by_id["grip"].ip, by_id["grip"].gripper) == ("192.168.1.201", "xarm_g2")
    assert (by_id["view"].ip, by_id["view"].gripper) == ("192.168.2.219", "none")
    assert by_id["view"].microphone is True and by_id["grip"].microphone is False
    assert [c.id for c in hw.cameras] == ["grip_wrist", "view_wrist"]
    # RealSense D435i colour over UVC: by USB serial (no by-id path), YUYV only.
    assert all(c.kind == "v4l2" and c.device_path is None and c.serial for c in hw.cameras)
    assert {c.fourcc for c in hw.cameras} == {"YUYV"}
    assert cfg.microphone.enabled and cfg.microphone.mic_id == "mic_view"
    assert cfg.microphone.label == "Perception Arm microphone"
    assert cfg.hardware_probe.enabled and cfg.hardware_probe.port == 502
    # Phase-09a: D435i COLOUR intrinsics per wrist camera (rs-enumerate-devices -c),
    # read-only monitor + twin overlay blocks.
    intr = {c.id: c.intrinsics for c in hw.cameras}
    assert (intr["grip_wrist"].fx, intr["grip_wrist"].fy) == (608.19, 608.23)
    assert (intr["grip_wrist"].cx, intr["grip_wrist"].cy) == (327.39, 247.90)
    assert (intr["view_wrist"].fx, intr["view_wrist"].fy) == (606.36, 606.38)
    assert (intr["view_wrist"].cx, intr["view_wrist"].cy) == (311.90, 249.45)
    mon = cfg.hardware_monitor
    assert (mon.enabled, mon.poll_hz, mon.stale_s, mon.reconnect_s) == (True, 10.0, 0.5, 2.0)
    ov = cfg.twin_overlay
    assert ov.enabled and ov.fps == 12.0 and ov.alpha == 0.5 and ov.stream_suffix == "_align"
    assert ov.tint_rgb == (255, 235, 140) and ov.edge_rgb == (255, 220, 60)
    assert ov.env_outline and ov.env_rgb == (90, 200, 250)
    assert ov.stale_tint_rgb == (170, 170, 170)
    assert ov.joint1_offset_rad == 0.0 and ov.rail_flip is False
    assert ov.rail_fallback_m == {"grip": 0.65, "view": 0.0}
    sim = cfg.workcells["sim"]
    assert {a.id for a in sim.arms} == {"view", "grip"} and sim.sim_scene == "mavis_v2"


@pytest.mark.parametrize("name", ["mavis_v2.yaml", "sim.yaml"])
def test_shipped_configs_load(name):
    cfg = load_runtime_config(CONFIGS / name)
    assert cfg.port == 8765
