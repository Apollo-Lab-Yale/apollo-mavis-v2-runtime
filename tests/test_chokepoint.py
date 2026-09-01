"""AST scan: ArmInterface.command_* call sites are pinned (11-safety §4)."""

from __future__ import annotations

import ast
from pathlib import Path

import apollo_xarm7_runtime

SRC = Path(apollo_xarm7_runtime.__file__).parent
COMMANDS = {"command_joints", "command_rail", "command_gripper"}

# The chokepoint dispatch path only: the ControlLoop resolves + gates every
# command; the per-arm ArmSender threads are its dispatch arm (04-runtime §3).
ALLOWED = {
    SRC / "control" / "arm_sender.py",
}


def _call_sites(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in COMMANDS
        ):
            found.add(node.func.attr)
    return found


def test_command_calls_only_in_dispatch_path():
    offenders = {}
    for path in SRC.rglob("*.py"):
        sites = _call_sites(path)
        if sites and path not in ALLOWED:
            offenders[str(path.relative_to(SRC))] = sorted(sites)
    assert not offenders, f"command_* called outside the chokepoint: {offenders}"


def test_dispatch_path_actually_calls_command_joints():
    assert "command_joints" in _call_sites(SRC / "control" / "arm_sender.py")
