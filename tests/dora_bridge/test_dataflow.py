"""render_dataflow snapshot + shape rules (14-dora §2.3); dora validate when the CLI exists."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml
from apollo_mavis_v2_core.protocol import external as ext

from apollo_mavis_v2_runtime.config import DoraConfig, DoraMachineConfig, load_runtime_config
from apollo_mavis_v2_runtime.dora_bridge.dataflow import (
    placeholders_for,
    render_dataflow,
    runtime_outputs,
)
from dora_bridge.harness import dora_cli

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "dataflows" / "mavis_v2.example.dora.yml"
SIM_CAMERAS = [
    "cam_front",
    "cam_top",
    "view_wrist_cam",
    "grip_wrist_cam",
    "grip_wrist",
    "view_wrist",
]
EXAMPLE_PYTHON = (
    "python3"  # the example pins no absolute interpreter; the runtime pins sys.executable
)


def example_text() -> str:
    cfg = load_runtime_config(REPO / "configs" / "mavis_v2.yaml")
    return render_dataflow(cfg.dora, SIM_CAMERAS, cfg.microphone.mic_id, EXAMPLE_PYTHON)


def test_example_file_matches_the_renderer_byte_for_byte():
    assert EXAMPLE.is_file(), "run: uv run python -m tests.dora_bridge.test_dataflow to regenerate"
    assert EXAMPLE.read_text(encoding="utf-8") == example_text()


def test_shape_rules_of_the_rendered_dataflow():
    doc = yaml.safe_load(example_text())
    nodes = {n["id"]: n for n in doc["nodes"]}
    rt = nodes[ext.EXTERNAL_NODE_ID]
    assert rt["path"] == "dynamic"
    assert set(rt["inputs"]) == set(ext.RUNTIME_INPUTS)  # exactly the six; no view_*
    assert not any(k.startswith("view_") for k in rt["inputs"])
    assert rt["inputs"]["tick"] == ext.TICK_TIMER
    assert rt["inputs"]["probe_heartbeat"] == "probe/heartbeat"
    assert rt["inputs"]["policy_action"] == {
        "source": "policy/action",
        "queue_size": 1,
        "queue_policy": "drop_oldest",
    }
    assert rt["inputs"]["policy_status"]["queue_size"] == 8
    # phase-14 (15-online-dagger §6): the trainer role's status rides policy/trainer_status, queue 8
    assert rt["inputs"]["policy_trainer_status"] == {
        "source": "policy/trainer_status",
        "queue_size": 8,
    }
    outs = rt["outputs"]
    assert outs[: len(ext.RUNTIME_FIXED_OUTPUTS)] == list(ext.RUNTIME_FIXED_OUTPUTS)
    assert "cam_view_wrist_cam" in outs and "cam_view_wrist_cam_depth" in outs
    assert "cam_grip_wrist_cam" in outs and "cam_grip_wrist_cam_depth" not in outs
    assert outs[-1] == "mic_mic_view"
    for pid in ext.PLACEHOLDERS:
        assert nodes[pid]["path"] == "dynamic"
        assert nodes[pid]["deploy"] == {"machine": "lab"}
    assert "outputs" not in nodes["viewer"]  # inputs only
    assert nodes["policy"]["outputs"] == list(ext.POLICY_OUTPUTS)
    assert set(nodes["observer"]["inputs"]) == set(outs)  # receives everything
    assert nodes["observer"]["inputs"]["events"]["queue_size"] == 64
    probe = nodes["probe"]
    assert probe["path"] == EXAMPLE_PYTHON and "nodes.probe" in probe["args"]
    assert probe["inputs"] == {"tick": ext.PROBE_TIMER} and probe["outputs"] == ["heartbeat"]
    text = example_text()
    assert "input_timeout" not in text and "enable_debug_inspection" not in text
    assert all(n["deploy"]["machine"] == "lab" for n in doc["nodes"])
    assert runtime_outputs(["a"], ["a"], None) == [
        *ext.RUNTIME_FIXED_OUTPUTS,
        "cam_a",
        "cam_a_depth",
    ]


def test_remote_placeholders_only_for_registered_machines():
    cfg = DoraConfig(
        enabled=True,
        machines=[
            DoraMachineConfig(id="remote", placeholders=["viewer"]),
            DoraMachineConfig(id="gpubox"),
        ],
    )
    none = yaml.safe_load(render_dataflow(cfg, ["view_wrist_cam"], None, "py", []))
    assert not any(n["id"].startswith(("viewer_", "observer_")) for n in none["nodes"])
    some = yaml.safe_load(render_dataflow(cfg, ["view_wrist_cam"], None, "py", ["remote"]))
    ids = [n["id"] for n in some["nodes"]]
    assert "viewer_remote" in ids and "observer_remote" not in ids and "viewer_gpubox" not in ids
    vr = next(n for n in some["nodes"] if n["id"] == "viewer_remote")
    assert vr["deploy"] == {"machine": "remote"} and vr["path"] == "dynamic"
    assert set(vr["inputs"]) == {
        "arm_state",
        "session",
        "cam_view_wrist_cam",
        "cam_view_wrist_cam_depth",
    }
    both = render_dataflow(cfg, ["view_wrist_cam"], None, "py", ["remote", "gpubox"])
    assert "observer_gpubox" in both and "viewer_gpubox" in both
    assert placeholders_for(cfg, ["gpubox", "unknown"]) == {
        "gpubox": ["viewer_gpubox", "observer_gpubox"]
    }


def test_dora_validate_strict_types_accepts_the_example():
    cli = dora_cli()
    if cli is None:
        import pytest

        pytest.skip("dora CLI missing")
    res = subprocess.run(
        [cli, "validate", "--strict-types", str(EXAMPLE)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert res.returncode == 0, res.stdout + res.stderr


if __name__ == "__main__":  # regenerate the committed example
    EXAMPLE.parent.mkdir(parents=True, exist_ok=True)
    EXAMPLE.write_text(example_text(), encoding="utf-8")
    print(f"wrote {EXAMPLE}")
