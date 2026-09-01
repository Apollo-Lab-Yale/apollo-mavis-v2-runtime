"""Runtime CI wires phase-03's guardrail regression (11-safety §5.1).

``python -m apollo_xarm7_sim.tools.guardrail_check --all`` must exit 0. The
runtime SafetyGate is behaviorally consistent with that reference gate; this
test pins the dependency so a gate regression fails runtime CI too.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytest.importorskip("apollo_xarm7_sim")


@pytest.mark.egl
def test_guardrail_check_all_passes():
    proc = subprocess.run(
        [sys.executable, "-m", "apollo_xarm7_sim.tools.guardrail_check", "--all"],
        capture_output=True,
        text=True,
        timeout=300,
        env={"MUJOCO_GL": "egl", **_env()},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def _env() -> dict:
    import os

    return dict(os.environ)
