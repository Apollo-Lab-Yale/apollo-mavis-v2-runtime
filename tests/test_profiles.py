"""Profile flows: save, set-initial overwrite semantics (04-runtime §9)."""

from __future__ import annotations

import numpy as np
from apollo_xarm7_core import Command, ProfileStore
from apollo_xarm7_core.testing import FakeArm, FakeWorkcell
from conftest import run_ticks

from apollo_xarm7_runtime.profiles.store import (
    INITIAL_PROFILE_NAME,
    save_from_states,
    save_initial_overwrite,
)


def states_at(q0: float):
    cell = FakeWorkcell({"arm0": FakeArm("arm0", has_rail=True, q0=np.full(8, q0))})
    cell.start()
    return cell.states()


def test_save_from_states_roundtrip(tmp_path):
    store = ProfileStore(tmp_path)
    p = save_from_states(store, states_at(0.3), ["arm0"], "sim", "wide", "notes!")
    loaded = store.get(p.profile_id)
    assert loaded.name == "wide" and loaded.workcell_kind == "sim"
    assert np.allclose(loaded.arms["arm0"].q, 0.3)
    assert loaded.arms["arm0"].rail_pos_m == 0.3


def test_set_initial_overwrites_same_name(tmp_path):
    store = ProfileStore(tmp_path)
    p1 = save_initial_overwrite(store, states_at(0.1), ["arm0"], "sim")
    assert p1.name == INITIAL_PROFILE_NAME and p1.is_initial_condition
    p2 = save_initial_overwrite(store, states_at(0.2), ["arm0"], "sim")
    assert p2.profile_id == p1.profile_id  # OVERWRITE, not a second profile
    assert np.allclose(store.get(p2.profile_id).arms["arm0"].q, 0.2)
    initials = [p for p in store.list() if p.is_initial_condition]
    assert len(initials) == 1  # unique per kind


def test_set_initial_uniqueness_across_profiles(tmp_path):
    store = ProfileStore(tmp_path)
    other = save_from_states(store, states_at(0.4), ["arm0"], "sim", "other")
    store.set_initial(other.profile_id)
    save_initial_overwrite(store, states_at(0.1), ["arm0"], "sim")
    assert not store.get(other.profile_id).is_initial_condition
    initials = [p for p in store.list() if p.is_initial_condition]
    assert len(initials) == 1 and initials[0].name == INITIAL_PROFILE_NAME


def test_loop_save_profile_and_set_initial_ops(fake_loop):
    cell, bus, loop = fake_loop
    f1 = bus.commands.submit(Command(op="save_profile", args={"name": "posture-a"}))
    f2 = bus.commands.submit(Command(op="set_initial_condition", args={}))
    run_ticks(loop, cell, 1)
    r1, r2 = f1.result(0), f2.result(0)
    assert r1.ok and r1.detail  # detail carries profile_id
    assert r2.ok and r2.detail
    store = loop.profile_store
    assert store.get(r2.detail).is_initial_condition
    f3 = bus.commands.submit(
        Command(op="set_initial_condition", args={"profile_id": r1.detail})
    )
    run_ticks(loop, cell, 1)
    assert f3.result(0).ok
    assert store.get(r1.detail).is_initial_condition
    assert not store.get(r2.detail).is_initial_condition


def test_save_profile_requires_name(fake_loop):
    cell, bus, loop = fake_loop
    fut = bus.commands.submit(Command(op="save_profile", args={}))
    run_ticks(loop, cell, 1)
    assert not fut.result(0).ok
