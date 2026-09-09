"""Fixtures for the dora tests; helpers live in ``harness.py`` (see its docstring)."""

from __future__ import annotations

import pytest

from dora_bridge.harness import ProcessSet


@pytest.fixture
def procs():
    ps = ProcessSet()
    yield ps
    ps.close()
