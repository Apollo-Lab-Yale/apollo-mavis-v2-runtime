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
    assert [c.id for c in hw.cameras] == ["camera1", "camera2"]  # placeholders kept
    assert all(str(c.device_path).startswith("/dev/v4l/by-id/TODO-") for c in hw.cameras)
    assert cfg.microphone.enabled and cfg.microphone.mic_id == "mic_view"
    assert cfg.microphone.label == "Perception Arm microphone"
    assert cfg.hardware_probe.enabled and cfg.hardware_probe.port == 502
    sim = cfg.workcells["sim"]
    assert {a.id for a in sim.arms} == {"view", "grip"} and sim.sim_scene == "mavis_v2"


@pytest.mark.parametrize("name", ["mavis_v2.yaml", "sim.yaml"])
def test_shipped_configs_load(name):
    cfg = load_runtime_config(CONFIGS / name)
    assert cfg.port == 8765
