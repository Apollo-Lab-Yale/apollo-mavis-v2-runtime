"""render_dataflow snapshot + shape rules (14-dora §2.3); dora validate when the CLI exists."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml
from apollo_mavis_v2_core.protocol import external as ext

from apollo_mavis_v2_runtime.config import DoraConfig, DoraMachineConfig, load_runtime_config
from apollo_mavis_v2_runtime.dora_bridge.dataflow import (
    placeholders_for,
    policy_outputs,
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


# v1.3: the example renders the sim workcell's arms (WorkcellConfig order: view, grip) -> one
# policy_action_<arm> runtime input and one action_<arm> policy output per arm
SIM_ARMS = ["view", "grip"]


def example_text() -> str:
    cfg = load_runtime_config(REPO / "configs" / "mavis_v2.yaml")
    assert [a.id for a in cfg.workcells["sim"].arms] == SIM_ARMS
    return render_dataflow(
        cfg.dora, SIM_CAMERAS, cfg.microphone.mic_id, EXAMPLE_PYTHON, arm_ids=SIM_ARMS
    )


def test_example_file_matches_the_renderer_byte_for_byte():
    assert EXAMPLE.is_file(), (
        "run: PYTHONPATH=tests .venv/bin/python tests/dora_bridge/test_dataflow.py to regenerate"
        " (never `uv run` while a runtime is live)"
    )
    assert EXAMPLE.read_text(encoding="utf-8") == example_text()


def test_shape_rules_of_the_rendered_dataflow():
    doc = yaml.safe_load(example_text())
    nodes = {n["id"]: n for n in doc["nodes"]}
    rt = nodes[ext.EXTERNAL_NODE_ID]
    assert rt["path"] == "dynamic"
    # the six fixed inputs + one per-arm action input per configured arm (v1.3); no view_*
    per_arm = {ext.policy_arm_action_input_id(a) for a in SIM_ARMS}
    assert set(rt["inputs"]) == set(ext.RUNTIME_INPUTS) | per_arm
    assert not any(k.startswith("view_") for k in rt["inputs"])
    assert rt["inputs"]["policy_action_grip"] == {
        "source": "policy/action_grip",
        "queue_size": 1,
        "queue_policy": "drop_oldest",
    }
    assert list(rt["inputs"])[-2:] == ["policy_action_view", "policy_action_grip"]  # after the six
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
    # v1.3: the fixed four, then action_<arm> per configured arm; the policy also hears the mic
    assert nodes["policy"]["outputs"] == [
        *ext.POLICY_OUTPUTS,
        *(ext.arm_action_output_id(a) for a in SIM_ARMS),
    ]
    assert nodes["policy"]["outputs"] == policy_outputs(SIM_ARMS)
    assert nodes["policy"]["inputs"]["mic_mic_view"] == {
        "source": "mavis_runtime/mic_mic_view",
        "queue_size": 32,
    }
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


def test_no_arm_ids_renders_the_v1_2_shape_and_no_mic_without_audio():
    """Backwards compatibility: without ``arm_ids`` the runtime inputs are exactly the six
    and the policy outputs the four (a v1.2 dataflow); the policy's mic row follows
    ``publish.audio`` like the runtime's own mic output."""
    cfg = DoraConfig(enabled=True)
    doc = yaml.safe_load(render_dataflow(cfg, ["view_wrist_cam"], "mic_view", "py"))
    nodes = {n["id"]: n for n in doc["nodes"]}
    assert set(nodes[ext.EXTERNAL_NODE_ID]["inputs"]) == set(ext.RUNTIME_INPUTS)
    assert nodes["policy"]["outputs"] == list(ext.POLICY_OUTPUTS)
    assert "mic_mic_view" in nodes["policy"]["inputs"]
    quiet = DoraConfig(enabled=True, publish={"audio": False})
    doc = yaml.safe_load(render_dataflow(quiet, ["view_wrist_cam"], "mic_view", "py"))
    nodes = {n["id"]: n for n in doc["nodes"]}
    assert "mic_mic_view" not in nodes["policy"]["inputs"]
    assert "mic_mic_view" not in nodes[ext.EXTERNAL_NODE_ID]["outputs"]


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
