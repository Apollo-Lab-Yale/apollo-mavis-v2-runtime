"""AST scan: ArmInterface.command_* call sites are pinned (11-safety §4)."""

from __future__ import annotations

import ast
from pathlib import Path

import apollo_mavis_v2_runtime

SRC = Path(apollo_mavis_v2_runtime.__file__).parent
COMMANDS = {"command_joints", "command_rail", "command_gripper"}

# The chokepoint dispatch path only: the ControlLoop resolves + gates every
# command; the per-arm ArmSender threads are its dispatch arm (04-runtime §3).
# session/hardware.py holds RailFlipArm (phase-09c): a pure ArmInterface adapter on
# the dispatch path that mirrors the rail slot of the ALREADY-GATED command the
# ArmSender hands it (q_sim = 0.65 - q_track) and forwards it - no command source.
# RailHoldArm (phase-09d) is the second adapter: it pins the rail slot of the gated
# command to the reported position (the rail-homing job never moves the carriage)
# and forwards; command_rail is refused there, never forwarded.
ALLOWED = {
    SRC / "control" / "arm_sender.py",
    SRC / "session" / "hardware.py",
}
ADAPTERS = {"RailFlipArm": 3, "RailHoldArm": 2}  # class -> command_* forwards inside it


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


def test_rail_adapters_only_forward_inside_their_classes():
    """The session/hardware.py allowance is for the two rail adapters alone
    (RailFlipArm: 3 forwards; RailHoldArm: command_joints + command_gripper, its
    command_rail raises): every command_* call in the module must sit inside one
    of them (no other code path there may address an arm)."""
    tree = ast.parse((SRC / "session" / "hardware.py").read_text(encoding="utf-8"))
    classes = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    }
    inside: set[int] = set()
    for name, expected in ADAPTERS.items():
        calls = {
            id(n)
            for n in ast.walk(classes[name])
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in COMMANDS
        }
        assert len(calls) == expected, (name, len(calls))
        inside |= calls
    every = {
        id(n)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in COMMANDS
    }
    assert every == inside
