"""AST scan (14-dora §1 "Import confinement"): dora only inside dora_bridge/; pyarrow there plus
the two sanctioned parquet sites (dagger spool, recorder episode store)."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

from apollo_mavis_v2_core.protocol import external as ext

import apollo_mavis_v2_runtime

SRC = Path(apollo_mavis_v2_runtime.__file__).parent
BANNED = {"dora", "pyarrow"}
# pyarrow is also sanctioned in the dagger package (trainer spool parquet) and - since the
# episode-directory datasets (2026-09-07; 10-frames §11, 04-runtime §10) - in the recorder
# package (episode frames.parquet + the LeRobot v3 export), the same sites pyproject's
# TID251 ban message names. `dora` itself stays confined to dora_bridge/.
PYARROW_ALSO_OK = {SRC / "dagger", SRC / "recorder"}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                found.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


def test_dora_and_pyarrow_imports_live_only_in_dora_bridge():
    offenders = {}
    for path in SRC.rglob("*.py"):
        mods = _imports(path) & BANNED
        if not mods:
            continue
        inside = SRC / "dora_bridge" in path.parents
        pyarrow_ok = mods == {"pyarrow"} and any(p in path.parents for p in PYARROW_ALSO_OK)
        if not inside and not pyarrow_ok:
            offenders[str(path.relative_to(SRC))] = sorted(mods)
    assert not offenders, f"dora/pyarrow imported outside dora_bridge/: {offenders}"


def test_dora_imports_inside_the_bridge_are_lazy():
    """No module-level `import dora` anywhere: importing the package must not import dora."""
    for path in (SRC / "dora_bridge").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:  # module level only
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            assert "dora" not in names, f"module-level dora import in {path.name}"


def test_importing_the_runtime_never_imports_dora():
    child = (
        "import sys, json\n"
        "import apollo_mavis_v2_runtime.runtime, apollo_mavis_v2_runtime.dora_bridge.wiring\n"
        "import apollo_mavis_v2_runtime.dagger.loop\n"
        "print(json.dumps({'dora': 'dora' in sys.modules}))\n"
    )
    out = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip().splitlines()[-1]) == {"dora": False}


def test_core_spellings_match_the_shared_contract_golden():
    """The same golden the apollo-mavis-v2-policy-node repo tests its contract.py against."""
    golden = json.loads((Path(__file__).parent / "golden" / "contract_golden.json").read_text())
    assert golden["mavis_schema"] == ext.MAVIS_SCHEMA
    assert golden["node_id"] == ext.EXTERNAL_NODE_ID
    assert golden["placeholders"] == list(ext.PLACEHOLDERS)
    assert golden["runtime_inputs"] == list(ext.RUNTIME_INPUTS)
    assert golden["runtime_fixed_outputs"] == list(ext.RUNTIME_FIXED_OUTPUTS)
    assert golden["policy_outputs"] == list(ext.POLICY_OUTPUTS)
    assert golden["arm_state_layout"] == list(ext.ARM_STATE_LAYOUT)
    assert golden["policy_action_required_metadata"] == list(ext.POLICY_ACTION_REQUIRED_METADATA)
    assert golden["common_output_metadata"] == list(ext.COMMON_OUTPUT_METADATA)
    assert golden["reserved_ids"] == list(ext.RESERVED_IDS)
    assert golden["policy_reset_reasons"] == list(ext.PolicyResetReason.__args__)
    assert golden["event_kinds"] == list(ext.EVENT_KINDS)
    assert golden["policy_spec_announce_fields"] == list(ext.PolicySpecAnnounce.model_fields)
    assert golden["session_announce_fields"] == list(ext.SessionAnnounce.model_fields)
    assert golden["policy_reset_fields"] == list(ext.PolicyResetMsg.model_fields)
    assert golden["event_envelope_fields"] == list(ext.EventEnvelope.model_fields)
    # phase-14 (15-online-dagger §6): the two trainer-contract models, placed right after
    # EventEnvelope (the policy-node repo's test pins the same position)
    keys = list(golden)
    i = keys.index("event_envelope_fields")
    assert keys[i + 1 : i + 3] == ["TrainerStatusAnnounce", "OnlineDaggerAnnounce"]
    assert "RefGradStatus" not in golden and "ProDaggerAnnounce" not in golden
    assert golden["TrainerStatusAnnounce"] == list(ext.TrainerStatusAnnounce.model_fields) == [
        "mavis_schema", "trainer_id", "node_version", "state", "session_id", "policy_version",
        "progress", "metrics", "detail", "uptime_s",
    ]
    assert golden["OnlineDaggerAnnounce"] == list(ext.OnlineDaggerAnnounce.model_fields) == [
        "session_name", "session_dir", "rollouts_dir",
    ]
    assert golden["policy_spec_announce_fields"][-1] == "capabilities"
    assert golden["session_announce_fields"][-1] == "online_dagger"
    assert golden["runtime_inputs"][-1] == "policy_trainer_status"
    assert golden["policy_outputs"][-1] == "trainer_status"
    assert golden["event_kinds"][-1] == "train_now" and len(golden["event_kinds"]) == 10
    assert golden["camera_output_prefix"] == ext.CAMERA_OUTPUT_PREFIX
    assert golden["depth_output_suffix"] == ext.DEPTH_OUTPUT_SUFFIX
    assert golden["mic_output_prefix"] == ext.MIC_OUTPUT_PREFIX
