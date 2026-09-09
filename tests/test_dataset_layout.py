"""Per-namespace dataset roots (15-online-dagger §7 / D5; 04-runtime §10.6; 2026-09-08).

``DatasetStore(root, default_namespace=.., namespaces=..)``: ``root_of`` spells
``<mapped root>/<name>[/<subdir>]`` for a mapped namespace and
``<datasets_root>/<ns>/<name>`` otherwise; ``list()`` / ``sweep()`` walk the generic
root AND every mapped root; ``DatasetInfo`` rows carry ``namespace`` / ``path``;
``GET /api/datasets/layout`` publishes the layout (declared before the ``{ns}/{name}``
routes); a collect session with ``default_namespace: bc_demo`` records under
``<bc_demo root>/<name>`` and ``dataset: online_dagger/<s>`` under
``<online_dagger root>/<s>/rollouts``. Every root here is a tmp dir - never ``~/data``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pytest
from apollo_mavis_v2_core.protocol import DatasetInfo, DatasetLayoutInfo
from conftest import make_runtime_config
from starlette.testclient import TestClient

from apollo_mavis_v2_runtime.config import DatasetNamespaceConfig, DatasetsConfig
from apollo_mavis_v2_runtime.recorder.datasets import DatasetStore, NamespaceRoot
from apollo_mavis_v2_runtime.recorder.manifest import (
    manifest_path,
    new_manifest,
    read_manifest,
    write_json_atomic,
)
from apollo_mavis_v2_runtime.runtime import Runtime
from apollo_mavis_v2_runtime.server.app import create_app

ROBOT = "xarm7_1arm_rail_mujoco"
T1, T2, T3, T9 = (f"2026-09-0{d}T00:00:00Z" for d in (1, 2, 3, 9))


def make_dataset(ds_root: Path, repo_id: str, modified_at: str) -> Path:
    """A manifest-only episode-directory dataset (the store reads nothing else)."""
    (ds_root / "episodes").mkdir(parents=True, exist_ok=True)
    m = new_manifest(repo_id, 25, ROBOT, {}, {})
    m["modified_at"] = modified_at
    write_json_atomic(manifest_path(ds_root), m)
    return ds_root


def make_legacy(ds_root: Path) -> Path:
    (ds_root / "meta").mkdir(parents=True, exist_ok=True)
    (ds_root / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v3.0", "fps": 25, "robot_type": ROBOT,
        "total_episodes": 2, "total_frames": 20, "features": {},
    }))
    return ds_root


@pytest.fixture()
def roots(tmp_path):
    return {
        "generic": tmp_path / "datasets",
        "bc_demo": tmp_path / "bc_demo",
        "online_dagger": tmp_path / "online_dagger",
    }


@pytest.fixture()
def store(roots):
    return DatasetStore(
        roots["generic"],
        default_namespace="bc_demo",
        namespaces={
            "bc_demo": DatasetNamespaceConfig(root=roots["bc_demo"]),
            "online_dagger": DatasetNamespaceConfig(
                root=roots["online_dagger"], subdir="rollouts"
            ),
        },
    )


# -- naming -----------------------------------------------------------------------------------
def test_root_of_maps_both_namespaces_and_falls_back_to_the_generic_root(store, roots):
    assert store.resolve("pick_cube") == "bc_demo/pick_cube"  # bare name -> default namespace
    assert store.resolve("apollo/x") == "apollo/x"
    assert store.root_of("bc_demo/pick_cube") == roots["bc_demo"] / "pick_cube"
    assert store.root_of("pick_cube") == roots["bc_demo"] / "pick_cube"  # resolves first
    assert store.root_of("online_dagger/s1") == roots["online_dagger"] / "s1" / "rollouts"
    assert store.root_of("apollo/xarm7_task") == roots["generic"] / "apollo" / "xarm7_task"
    assert store.root_of("other/thing") == roots["generic"] / "other" / "thing"
    assert store.namespace == "bc_demo"  # alias of default_namespace
    assert store.split("online_dagger/s1") == ("online_dagger", "s1")
    # the layout the REST publishes
    lay = store.layout()
    assert isinstance(lay, DatasetLayoutInfo)
    assert lay.default_namespace == "bc_demo" and lay.generic_root == str(roots["generic"])
    assert lay.namespaces["bc_demo"].root == str(roots["bc_demo"])
    assert lay.namespaces["bc_demo"].subdir is None
    assert lay.namespaces["online_dagger"].root == str(roots["online_dagger"])
    assert lay.namespaces["online_dagger"].subdir == "rollouts"


def test_namespace_specs_are_duck_typed(tmp_path):
    """The store accepts config models, NamespaceRoot, mappings, pairs and bare paths
    (it must stay importable without config.py)."""
    st = DatasetStore(
        tmp_path / "g",
        default_namespace="apollo",
        namespaces={
            "a": DatasetNamespaceConfig(root=tmp_path / "a"),
            "b": NamespaceRoot(tmp_path / "b", "sub"),
            "c": {"root": tmp_path / "c", "subdir": "roll"},
            "d": (tmp_path / "d", None),
            "e": tmp_path / "e",
        },
    )
    assert st.root_of("a/x") == tmp_path / "a" / "x"
    assert st.root_of("b/x") == tmp_path / "b" / "x" / "sub"
    assert st.root_of("c/x") == tmp_path / "c" / "x" / "roll"
    assert st.root_of("d/x") == tmp_path / "d" / "x"
    assert st.root_of("e/x") == tmp_path / "e" / "x"
    assert st.root_of("x") == tmp_path / "g" / "apollo" / "x"
    plain = DatasetStore(tmp_path / "g")  # pre-D4 signature still works
    assert plain.default_namespace == "apollo" and plain.namespaces == {}
    assert plain.layout().namespaces == {}
    assert plain.root_of("apollo/x") == tmp_path / "g" / "apollo" / "x"


# -- listing across roots -----------------------------------------------------------------------
def test_list_walks_the_generic_root_and_every_mapped_root(store, roots):
    make_dataset(roots["generic"] / "apollo" / "old_task", "apollo/old_task", T1)
    make_dataset(roots["bc_demo"] / "pick_cube", "bc_demo/pick_cube", T3)
    make_dataset(roots["online_dagger"] / "s1" / "rollouts", "online_dagger/s1", T2)
    make_legacy(roots["bc_demo"] / "old_v3")  # a legacy tree under a mapped root lists too
    (roots["online_dagger"] / "s1" / "trainer").mkdir()  # trainer-owned sibling: not a dataset
    (roots["online_dagger"] / "s1" / "session.json").write_text("{}")
    # a bc_demo dataset sitting under the GENERIC root is unaddressable -> not listed
    make_dataset(roots["generic"] / "bc_demo" / "shadow", "bc_demo/shadow", T9)

    rows = store.list()
    by_id = {r.repo_id: r for r in rows}
    newest_first = ["bc_demo/pick_cube", "online_dagger/s1", "apollo/old_task"]
    # newest first across roots (the legacy tree's modified_at is its file mtime = now)
    assert [r.repo_id for r in rows if r.layout == "episode_dirs"] == newest_first
    assert set(by_id) == {*newest_first, "bc_demo/old_v3"}
    assert "bc_demo/shadow" not in by_id
    assert all(isinstance(r, DatasetInfo) for r in rows)
    pick = by_id["bc_demo/pick_cube"]
    assert pick.namespace == "bc_demo" and pick.path == str(roots["bc_demo"] / "pick_cube")
    assert pick.root == pick.path and pick.layout == "episode_dirs"
    s1 = by_id["online_dagger/s1"]
    assert s1.namespace == "online_dagger"
    assert s1.path == str(roots["online_dagger"] / "s1" / "rollouts")
    old = by_id["apollo/old_task"]
    assert old.namespace == "apollo" and old.path == str(roots["generic"] / "apollo" / "old_task")
    legacy = by_id["bc_demo/old_v3"]
    assert legacy.layout == "lerobot_v3" and legacy.namespace == "bc_demo"
    assert legacy.path == str(roots["bc_demo"] / "old_v3")
    # describe / episodes / layout_of address every root the same way
    assert store.describe("online_dagger/s1").total_episodes == 0
    assert store.episodes("online_dagger/s1") == []
    assert store.layout_of("bc_demo/old_v3") == "lerobot_v3"
    assert store.layout_of("bc_demo/shadow") is None  # the generic-root copy is invisible
    assert store.exists("apollo/old_task") and not store.exists("apollo/nope")


def test_list_with_no_generic_root_still_lists_mapped_roots(store, roots):
    make_dataset(roots["bc_demo"] / "only", "bc_demo/only", T3)
    assert not roots["generic"].exists()
    assert [r.repo_id for r in store.list()] == ["bc_demo/only"]


def test_delete_dataset_keeps_a_mapped_root_and_the_online_dagger_session_dir(store, roots):
    make_dataset(roots["bc_demo"] / "gone", "bc_demo/gone", T3)
    make_dataset(roots["online_dagger"] / "s1" / "rollouts", "online_dagger/s1", T2)
    (roots["online_dagger"] / "s1" / "trainer").mkdir()
    make_dataset(roots["generic"] / "apollo" / "g", "apollo/g", T1)
    store.delete_dataset("bc_demo/gone")
    assert not (roots["bc_demo"] / "gone").exists() and roots["bc_demo"].exists()  # root stays
    store.delete_dataset("online_dagger/s1")
    assert not (roots["online_dagger"] / "s1" / "rollouts").exists()
    assert (roots["online_dagger"] / "s1" / "trainer").is_dir()  # the session dir is not ours
    store.delete_dataset("apollo/g")
    assert not (roots["generic"] / "apollo").exists()  # emptied generic <ns> dir pruned
    assert store.list() == []


# -- crash sweep across roots ---------------------------------------------------------------------
def test_sweep_removes_tmp_episodes_under_every_root_but_the_open_one(store, roots):
    ds = {
        "apollo/a": make_dataset(roots["generic"] / "apollo" / "a", "apollo/a", T1),
        "bc_demo/b": make_dataset(roots["bc_demo"] / "b", "bc_demo/b", T1),
        "online_dagger/c": make_dataset(
            roots["online_dagger"] / "c" / "rollouts", "online_dagger/c", T1
        ),
    }
    crashed = {}
    for repo_id, root in ds.items():
        tmp = root / "episodes" / ".tmp-20260908T000000.000Z-000000"
        tmp.mkdir()
        crashed[repo_id] = tmp
    live = ds["online_dagger/c"] / "episodes" / ".tmp-20260908T000001.000Z-111111"
    live.mkdir()
    store.in_use_repo = lambda: "online_dagger/c"
    store.open_episode = lambda: "20260908T000001.000Z-111111"
    removed = store.sweep()
    # every .tmp-* of the IN-USE dataset is kept (pre-existing rule: the recorder owns
    # that directory while the session runs); the other roots' crashed ones go
    assert set(removed) == {crashed["apollo/a"], crashed["bc_demo/b"]}
    assert not crashed["apollo/a"].exists() and not crashed["bc_demo/b"].exists()
    assert crashed["online_dagger/c"].exists() and live.exists()
    # Runtime.start() runs the same sweep through the manager's store; with no session
    # running nothing is open, so the in-use dataset's directories go too
    again = ds["bc_demo/b"] / "episodes" / ".tmp-x"
    again.mkdir()
    cfg = make_runtime_config(roots["generic"].parent / "rt").model_copy(update={
        "datasets_root": roots["generic"],
        "datasets": DatasetsConfig(
            default_namespace="bc_demo",
            namespaces={
                "bc_demo": DatasetNamespaceConfig(root=roots["bc_demo"]),
                "online_dagger": DatasetNamespaceConfig(
                    root=roots["online_dagger"], subdir="rollouts"
                ),
            },
        ),
    })
    rt = Runtime(cfg)
    try:
        rt.start()
        assert not again.exists() and not live.exists() and not crashed["online_dagger/c"].exists()
    finally:
        rt.stop()


def test_sweep_without_a_session_removes_every_tmp_dir(store, roots):
    root = make_dataset(roots["online_dagger"] / "c" / "rollouts", "online_dagger/c", T1)
    tmp = root / "episodes" / ".tmp-1"
    tmp.mkdir()
    assert store.sweep() == [tmp] and not tmp.exists()
    assert store.sweep() == []


# -- REST + a collect session ------------------------------------------------------------------
SPEC = {
    "mode": "collect",
    "kind": "sim",
    "arms": ["arm0"],
    "frames": {"arm0": "arm_base:arm0"},
    "sim_scene": "guardrail_env",  # 1 arm (rail) + 1 camera
    "task": "layout e2e",
    "dataset_resume": False,
    "return_to_start": False,  # no profile in this runtime (else 409)
}


@pytest.fixture(scope="module")
def layout_env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("layout")
    cfg = make_runtime_config(tmp / "rt", scene="guardrail_env").model_copy(update={
        "datasets_root": tmp / "datasets",
        "datasets": DatasetsConfig(
            default_namespace="bc_demo",
            namespaces={
                "bc_demo": DatasetNamespaceConfig(root=tmp / "bc_demo"),
                "online_dagger": DatasetNamespaceConfig(
                    root=tmp / "online_dagger", subdir="rollouts"
                ),
            },
        ),
    })
    app = create_app(Runtime(cfg))
    with TestClient(app) as c:
        yield tmp, c
        c.delete("/api/session")


def wait_running(client) -> None:
    for _ in range(400):
        if client.get("/api/session").json()["state"] == "running":
            return
        time.sleep(0.05)
    raise AssertionError("session never reached running")


def wait_idle(client) -> None:
    for _ in range(400):
        if client.get("/api/session").status_code == 404:
            return
        time.sleep(0.05)
    raise AssertionError("session never ended")


def test_get_datasets_layout_body_and_routing(layout_env):
    tmp, client = layout_env
    r = client.get("/api/datasets/layout")
    assert r.status_code == 200, r.text
    body = DatasetLayoutInfo.model_validate(r.json())
    assert body.default_namespace == "bc_demo"
    assert body.generic_root == str(tmp / "datasets")
    assert body.namespaces["bc_demo"].root == str(tmp / "bc_demo")
    assert body.namespaces["bc_demo"].subdir is None
    assert body.namespaces["online_dagger"].root == str(tmp / "online_dagger")
    assert body.namespaces["online_dagger"].subdir == "rollouts"
    assert r.json() == {
        "default_namespace": "bc_demo",
        "generic_root": str(tmp / "datasets"),
        "namespaces": {
            "bc_demo": {"root": str(tmp / "bc_demo"), "subdir": None},
            "online_dagger": {"root": str(tmp / "online_dagger"), "subdir": "rollouts"},
        },
    }
    # "layout" is a literal, never swallowed as a namespace; the parametrised routes
    # still answer their own shapes
    assert client.get("/api/datasets/layout/x").status_code == 404  # ns=layout, name=x
    assert client.get("/api/datasets/bc_demo/nope").status_code == 404
    assert client.get("/api/datasets").json() == []


def test_collect_session_records_under_the_mapped_namespace_roots(layout_env):
    tmp, client = layout_env
    # bare name -> bc_demo/<name> -> <bc_demo root>/<name>
    r = client.post("/api/session", json={**SPEC, "dataset": "pick_cube"})
    assert r.status_code == 200, r.text
    wait_running(client)
    try:
        ds_root = tmp / "bc_demo" / "pick_cube"
        assert manifest_path(ds_root).exists()
        assert read_manifest(ds_root)["repo_id"] == "bc_demo/pick_cube"
        assert not list((tmp / "datasets").rglob("manifest.json"))  # nothing in the generic root
        rows = client.get("/api/datasets").json()
        assert [d["repo_id"] for d in rows] == ["bc_demo/pick_cube"]
        assert rows[0]["namespace"] == "bc_demo" and rows[0]["path"] == str(ds_root)
        assert rows[0]["root"] == str(ds_root) and rows[0]["in_use"] is True
        one = client.get("/api/datasets/bc_demo/pick_cube").json()
        assert one["path"] == str(ds_root) and one["namespace"] == "bc_demo"
        assert client.get("/api/datasets/bc_demo/pick_cube/episodes").json() == []
        # the same name again is 409 (exists) - through root_of, not the generic root
        r = client.post("/api/session", json={**SPEC, "dataset": "pick_cube"})
        assert r.status_code == 409
    finally:
        client.delete("/api/session")
        wait_idle(client)
    # resuming finds it under the mapped root
    r = client.post("/api/session", json={**SPEC, "dataset": "pick_cube", "dataset_resume": True})
    assert r.status_code == 200, r.text
    wait_running(client)
    client.delete("/api/session")
    wait_idle(client)
    r = client.post("/api/session", json={**SPEC, "dataset": "never", "dataset_resume": True})
    assert r.status_code == 409 and "unknown dataset 'bc_demo/never'" in r.text

    # an explicit online_dagger/<s> id -> <online_dagger root>/<s>/rollouts
    r = client.post("/api/session", json={**SPEC, "dataset": "online_dagger/sess1"})
    assert r.status_code == 200, r.text
    wait_running(client)
    try:
        roll = tmp / "online_dagger" / "sess1" / "rollouts"
        assert manifest_path(roll).exists()
        assert read_manifest(roll)["repo_id"] == "online_dagger/sess1"
        rows = {d["repo_id"]: d for d in client.get("/api/datasets").json()}
        assert set(rows) == {"bc_demo/pick_cube", "online_dagger/sess1"}
        assert rows["online_dagger/sess1"]["path"] == str(roll)
        assert rows["online_dagger/sess1"]["namespace"] == "online_dagger"
        assert rows["online_dagger/sess1"]["in_use"] and not rows["bc_demo/pick_cube"]["in_use"]
    finally:
        client.delete("/api/session")
        wait_idle(client)
    # deletion goes through the same roots; the online_dagger session dir survives its
    # rollouts dataset, the bc_demo root survives its dataset
    assert client.delete("/api/datasets/online_dagger/sess1").status_code == 204
    assert not roll.exists() and (tmp / "online_dagger" / "sess1").is_dir()
    assert client.delete("/api/datasets/bc_demo/pick_cube").status_code == 204
    assert not (tmp / "bc_demo" / "pick_cube").exists() and (tmp / "bc_demo").is_dir()
    assert client.get("/api/datasets").json() == []


# -- export CLI ------------------------------------------------------------------------------------
def test_export_cli_spells_paths_through_the_store(tmp_path, monkeypatch):
    from apollo_mavis_v2_runtime.tools.export_lerobot import _dataset_store, main

    monkeypatch.delenv("APOLLO_CONFIG", raising=False)
    # --root: one generic tree, bare names -> bc_demo (the config default) or --namespace
    st = _dataset_store(argparse.Namespace(root=tmp_path / "g", namespace=None, config=None))
    assert st.default_namespace == "bc_demo" and st.namespaces == {}
    assert st.root_of("x") == tmp_path / "g" / "bc_demo" / "x"
    st = _dataset_store(argparse.Namespace(root=tmp_path / "g", namespace="apollo", config=None))
    assert st.root_of("x") == tmp_path / "g" / "apollo" / "x"
    # --config: the config's generic root + namespace map
    monkeypatch.setenv("APOLLO_HOME", str(tmp_path))
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "datasets_root: ${APOLLO_HOME}/var/datasets\n"
        "datasets:\n"
        "  default_namespace: bc_demo\n"
        "  namespaces:\n"
        '    bc_demo: {root: "${APOLLO_HOME}/data/bc_demo"}\n'
        '    online_dagger: {root: "${APOLLO_HOME}/data/online_dagger", subdir: rollouts}\n'
    )
    st = _dataset_store(argparse.Namespace(root=None, namespace=None, config=cfg_file))
    assert st.root_of("pick") == tmp_path / "data" / "bc_demo" / "pick"
    assert st.root_of("online_dagger/s1") == tmp_path / "data" / "online_dagger" / "s1" / "rollouts"
    assert st.root_of("apollo/old") == tmp_path / "var" / "datasets" / "apollo" / "old"
    st = _dataset_store(argparse.Namespace(root=None, namespace="apollo", config=cfg_file))
    assert st.root_of("old") == tmp_path / "var" / "datasets" / "apollo" / "old"
    # a missing dataset is reported at the MAPPED path (exit 2)
    assert main(["online_dagger/nope", "--config", str(cfg_file)]) == 2
