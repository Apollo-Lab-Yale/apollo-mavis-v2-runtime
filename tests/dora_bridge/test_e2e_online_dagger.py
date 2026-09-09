"""Tier 4 (15-online-dagger §11; phase-14): an Online DAgger session over a REAL server + REAL
dora control plane, with ``nodes/fake_policy.py`` in trainer mode (``FAKE_TRAINER=1``)
playing the policy repo's generic trainer and an ``observer`` node logging ``events``.

The rig uses the SHIPPED dataset layout (D5) in a tmp: ``bc_demo`` -> ``<tmp>/data/bc_demo``
(the default namespace) and ``online_dagger`` -> ``<tmp>/data/online_dagger/<s>/rollouts``. A
teleop session designates the initial condition (D6 needs one), then the Online DAgger
session ``s1`` walks: the 409 matrix (no policy / the session-directory rules / no
``online_dagger`` capability) -> ``waiting_trainer`` while the fake prepares (``episode_new``
refused with the documented reason) -> ``rollout`` once it reports ``ready`` -> rollout 1 with
the explicit ``takeover`` / ``handback`` actions -> return-to-start (``returning`` walked) ->
rollout 2 with the Space toggle -> the fake trains after its 2nd kept rollout (``training``:
``episode_new`` refused) -> ``ready`` with a version bump (``policy_version_acting`` 2) -> a
rollout discarded while the expert holds the arm (NOTHING on disk, ``episode_discarded`` +
the ``episode_reset`` gate event on the bus) -> **Train now** (``events.train_now``; the fake
retrains -> v3). Then the wire (every event kind with its exact payload keys; nothing but
the ten generic kinds), the disk (``session.json`` rows, ``actor`` in frames + spool, the
sidecar block, no trainer artefact) and the session-less REST (sessions listing, skill
markdown + tarball; ``/api/pro_dagger/*`` is gone).
"""

from __future__ import annotations

import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
from apollo_mavis_v2_core.protocol.external import EVENT_KINDS
from conftest import LiveServer
from test_e2e_teleop import PulsingCtl, Tele

from apollo_mavis_v2_runtime.config import DatasetNamespaceConfig, DatasetsConfig
from dora_bridge.harness import (
    dora_runtime_config,
    node_env,
    requires_dora,
    wait_until,
)

pq = pytest.importorskip("pyarrow.parquet")

pytestmark = [pytest.mark.dora, pytest.mark.egl, requires_dora]

MOD = "apollo_mavis_v2_runtime.dora_bridge.nodes"
TASK = "online-dagger e2e"
BASE = {
    "kind": "sim",
    "arms": ["arm0"],
    "frames": {"arm0": "arm_base:arm0"},
    "sim_scene": "guardrail_env",  # 1 arm (rail) + 1 camera: video + spool like the lab
}
OD = {"session_name": "s1", "wait_for_trainer_ready": True}
SPEC = {
    **BASE,
    "mode": "dagger",
    "task": TASK,
    "policy_source": "external",
    "return_to_start": True,  # D6: back to the initial condition after every save / discard
    "online_dagger": OD,
}
RATE_HZ = 15.0
PREPARE_S = 3.0  # long enough to catch `waiting_trainer` and its refusal from the test
TRAIN_S = 3.0  # a 3 s `training` window for the refusal (telemetry at 25 Hz)
EVERY = 2  # the fake trains after every 2nd kept rollout
TRAINER_ID = "runtime-fake/online_dagger"
NO_CAPABILITY = (
    "no Online DAgger trainer attached (the policy node does not report the online_dagger "
    "capability)"
)
GATE_KEYS = {"arm_id", "mode", "seq", "source", "episode_id"}
SAVED_KEYS = {"episode_index", "summary", "dataset_root", "spool_path", "run_id", "online_dagger"}
BLOCK_KEYS = {"episode_id", "rollouts_saved", "actor_counts", "policy_version", "spool_path"}


# -- rig -------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def server(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rt")
    cfg = dora_runtime_config(tmp, scene="guardrail_env", arms=("arm0",))
    cfg.datasets = DatasetsConfig(
        default_namespace="bc_demo",
        namespaces={
            "bc_demo": DatasetNamespaceConfig(root=tmp / "data" / "bc_demo"),
            "online_dagger": DatasetNamespaceConfig(
                root=tmp / "data" / "online_dagger", subdir="rollouts"
            ),
        },
    )
    srv = LiveServer(cfg)
    with httpx.Client(base_url=srv.http, timeout=30) as api:
        wait_until(lambda: api.get("/api/dora").json()["state"] == "attached", 10.0, "attached")
    yield srv, cfg, tmp
    bridge = srv.runtime.dora.bridge
    plane_pids = list(bridge.plane.child_pids()) if bridge.plane is not None else []
    srv.stop()
    time.sleep(0.3)
    alive = [p for p in plane_pids if os.path.exists(f"/proc/{p}")]
    assert not alive, f"runtime's dora children survived stop: {alive}"


@pytest.fixture(scope="module")
def api(server):
    with httpx.Client(base_url=server[0].http, timeout=60.0) as client:
        yield client


class FakePolicy:
    """A ``fake_policy`` subprocess on the server's private daemon; ``trainer=True`` turns
    the Online DAgger trainer role on (``FAKE_TRAINER=1`` + the timing knobs)."""

    def __init__(self, server, *args: str, trainer: bool = False, env: dict | None = None):
        srv, cfg, _tmp = server
        self.srv = srv
        info = httpx.get(f"{srv.http}/api/dora", timeout=10).json()
        if "--action-frame" not in args:
            args = (*args, "--action-frame", SPEC["frames"]["arm0"])
        extra = dict(env or {})
        if trainer:
            extra.setdefault("FAKE_TRAINER", "1")
            extra.setdefault("FAKE_TRAINER_PREPARE_S", str(PREPARE_S))
            extra.setdefault("FAKE_TRAINER_TRAIN_S", str(TRAIN_S))
            extra.setdefault("FAKE_TRAINER_EVERY", str(EVERY))
        self.stats_path = cfg.dora.var_dir / f"fake_stats_{'trainer' if trainer else 'plain'}.json"
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                f"{MOD}.fake_policy",
                "--daemon-port",
                str(info["daemon_port"]),
                "--rate-hz",
                str(RATE_HZ),
                "--stats-out",
                str(self.stats_path),
                *args,
            ],
            env=node_env(info["bind_host"], info["zenoh_port"], extra),
            stdout=open(cfg.dora.var_dir / "fake_policy.log", "ab"),  # noqa: SIM115
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        wait_until(lambda: external(srv)["policy_attached"] is True, 10.0, "fake policy spec")
        if trainer:
            wait_until(lambda: "online_dagger" in external(srv)["capabilities"], 10.0, "capability")

    def close(self) -> None:
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)

    def stats(self) -> dict:
        return json.loads(self.stats_path.read_text()) if self.stats_path.exists() else {}


class EventsObserver:
    """An ``observer`` node appending every ``events`` envelope to a JSON-lines file."""

    def __init__(self, server, out: Path, seconds: float = 240.0) -> None:
        srv, _cfg, _tmp = server
        info = httpx.get(f"{srv.http}/api/dora", timeout=10).json()
        self.out = out
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                f"""
import json, time
from dora import Node
node = Node("observer", daemon_port={info["daemon_port"]})
t_end = time.monotonic() + {seconds}
with open({json.dumps(str(out))}, "a") as fh:
    while time.monotonic() < t_end:
        ev = node.next(timeout=1.0)
        if ev is None or ev.get("type") == "STOP": break
        if ev.get("type") == "INPUT" and ev["id"] == "events":
            fh.write(json.dumps(json.loads(ev["value"][0].as_py())) + "\\n"); fh.flush()
""",
            ],
            env=node_env(info["bind_host"], info["zenoh_port"]),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    def events(self) -> list[dict]:
        if not self.out.exists():
            return []
        return [json.loads(line) for line in self.out.read_text().splitlines() if line.strip()]

    def of(self, kind: str) -> list[dict]:
        return [e for e in self.events() if e["kind"] == kind]

    def close(self) -> None:
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)


# -- helpers ---------------------------------------------------------------------------------------
def external(srv) -> dict:
    tele = Tele(srv)
    try:
        return tele.latest()["external"]
    finally:
        tele.close()


def wait_running(api: httpx.Client, timeout: float = 30.0) -> None:
    wait_until(lambda: api.get("/api/session").json()["state"] == "running", timeout, "running")


def wait_tele(tele: Tele, pred, timeout: float, what: str) -> dict:
    end = time.monotonic() + timeout
    last = None
    while time.monotonic() < end:
        last = tele.latest()
        if pred(last):
            return last
    raise AssertionError(
        f"telemetry never satisfied: {what}; last dagger={last and last['dagger']}"
    )


def od_of(msg: dict) -> dict:
    dagger = msg.get("dagger") or {}
    return dagger.get("online_dagger") or {}


def watch_until_idle(tele: Tele, timeout: float) -> list[str]:
    """Every distinct episode state seen until the recorder is idle again."""
    seen: list[str] = []
    t0 = time.monotonic()
    while True:
        msg = tele.latest()
        st = msg["episode"]["state"]
        if not seen or seen[-1] != st:
            seen.append(st)
        if st == "idle" and len(seen) > 1:
            return seen
        assert time.monotonic() - t0 < timeout, seen


def wait_human(tele: Tele) -> None:
    wait_tele(
        tele,
        lambda m: m["dagger"]["control_mode"] == "human" and m["dagger"]["engaged_arm"] == "arm0",
        5.0,
        "HUMAN visible",
    )
    assert tele.latest()["dagger"]["takeover_rate_ep"] > 0.0


def ints(table, name: str) -> np.ndarray:
    vals = table.column(name).to_pylist()
    return np.asarray([v[0] if isinstance(v, list) else v for v in vals]).astype(int)


# -- stage 1: the initial condition ----------------------------------------------------------------
@pytest.fixture(scope="module")
def initial_condition(server, api) -> str:
    """A teleop session designates the initial-condition profile the rollouts return to."""
    srv, _cfg, tmp = server
    r = api.post("/api/session", json={**BASE, "mode": "teleop"})
    assert r.status_code == 200, r.text
    wait_running(api)
    ctl = PulsingCtl(srv)
    try:
        ctl.hold(["KeyE", "KeyA"], 1.2)  # up + left: away from the pedestal
        time.sleep(0.5)
        ack = ctl.action("set_initial_condition")
        assert ack["ok"], ack
        profile_id = ack["detail"]
    finally:
        ctl.close()
        assert api.delete("/api/session").status_code == 204
    rows = {p["profile_id"]: p for p in api.get("/api/profiles").json()}
    assert rows[profile_id]["is_initial_condition"]
    layout = api.get("/api/datasets/layout").json()
    assert layout["default_namespace"] == "bc_demo"
    assert layout["namespaces"]["online_dagger"] == {
        "root": str(tmp / "data" / "online_dagger"),
        "subdir": "rollouts",
    }
    return profile_id


# -- stage 2: the session over the bus -------------------------------------------------------------
def test_409_matrix_before_any_trainer(server, api, initial_condition):
    srv, _cfg, tmp = server
    root = tmp / "data" / "online_dagger"
    # nothing attached yet: the shared external 409
    r = api.post("/api/session", json=SPEC)
    assert r.status_code == 409 and r.json()["detail"].startswith("no external policy attached"), (
        r.text
    )
    # the session-directory rules come BEFORE any trainer check (no side effect either way)
    (root / "taken").mkdir(parents=True)
    try:
        r = api.post("/api/session", json={**SPEC, "online_dagger": {"session_name": "taken"}})
        assert r.status_code == 409, r.text
        assert r.json()["detail"] == (
            "Online DAgger session 'taken' already exists - resume it or pick another name"
        )
        assert sorted(p.name for p in (root / "taken").iterdir()) == []
    finally:
        shutil.rmtree(root / "taken")
    r = api.post(
        "/api/session", json={**SPEC, "online_dagger": {"session_name": "nope", "resume": True}}
    )
    assert r.status_code == 409 and r.json()["detail"] == "Online DAgger session 'nope' not found"
    # the core rules ride the wire as 422s
    r = api.post("/api/session", json={**SPEC, "policy_source": "checkpoint"})
    assert r.status_code == 422 and "online_dagger requires policy_source 'external'" in r.text
    r = api.post("/api/session", json={**SPEC, "dataset": "demo1"})
    assert r.status_code == 422 and "leave dataset unset" in r.text
    # a plain policy node (no trainer role): fresh spec without the capability
    plain = FakePolicy(server, "--mode", "hold")
    try:
        ext = external(srv)
        assert (
            ext["policy_attached"] and ext["capabilities"] == [] and ext["trainer_status"] is None
        )
        r = api.post("/api/session", json=SPEC)
        assert r.status_code == 409 and r.json()["detail"] == NO_CAPABILITY, r.text
        assert api.get("/api/session").status_code == 404
    finally:
        plain.close()
    assert not (root / "s1").exists()  # no side effect of a 409


def test_online_dagger_rollouts_over_the_bus(server, api, initial_condition, tmp_path):
    srv, cfg, tmp = server
    sdir = tmp / "data" / "online_dagger" / "s1"
    rollouts = sdir / "rollouts"
    fake = FakePolicy(server, "--mode", "scripted", "--amplitude-m", "0.01", trainer=True)
    observer = EventsObserver(server, tmp_path / "events.jsonl")
    ctl = tele = None
    phases: list[str] = []

    def note_phase(msg: dict) -> dict:
        ph = od_of(msg).get("phase")
        if ph and (not phases or phases[-1] != ph):
            phases.append(ph)
        return msg

    try:
        # -- the trainer's spec is fresh: capability, no status before a session --------------
        ext = external(srv)
        assert ext["capabilities"] == ["online_dagger"] and ext["policy_id"] == "fake-policy"
        assert ext["trainer_status"] is None
        r = api.post("/api/session", json=SPEC)
        assert r.status_code == 200, r.text
        info = r.json()
        assert info["mode"] == "dagger" and info["policy_source"] == "external"
        assert info["online_dagger"] == {
            "session_name": "s1",
            "resume": False,
            "pause_while_training": True,
            "wait_for_trainer_ready": True,
        }
        wait_running(api)
        session = srv.runtime.manager.session
        sid = session.session_id
        assert sorted(p.name for p in sdir.iterdir()) == ["rollouts", "session.json"]
        assert session.recorder_thread.repo_id == "online_dagger/s1"
        ctl, tele = PulsingCtl(srv), Tele(srv)
        # -- waiting_trainer: the fake prepares; episode_new is refused with the reason -------
        msg = wait_tele(
            tele,
            lambda m: note_phase(m) and od_of(m).get("trainer_alive") is True,
            10.0,
            "trainer status reaches the coordinator",
        )
        odt = od_of(msg)
        assert odt["session_name"] == "s1" and odt["session_dir"] == str(sdir)
        assert odt["phase"] == "waiting_trainer" and odt["rollouts_saved"] == 0
        assert odt["policy_version_acting"] == 1
        assert (odt["expert_frames_session"], odt["novice_frames_session"]) == (0, 0)
        assert odt["trainer"]["trainer_id"] == TRAINER_ID
        assert odt["trainer"]["session_id"] == sid and odt["trainer"]["state"] == "preparing"
        assert odt["detail"].startswith("waiting for the trainer to report ready (warming up")
        assert msg["session"]["trainer_alive"] is True
        assert msg["dagger"]["policy_version"] == "fake-policy/v000001"
        assert msg["external"]["trainer_status"]["session_id"] == sid
        assert msg["external"]["capabilities"] == ["online_dagger"]
        ack = ctl.action("episode_new")
        assert ack["ok"] is False, ack
        assert ack["detail"].startswith("waiting for the trainer to report ready (warming up"), ack
        # -- ready -> rollout ------------------------------------------------------------------
        msg = wait_tele(
            tele,
            lambda m: note_phase(m) and od_of(m).get("phase") == "rollout",
            PREPARE_S + 15.0,
            "rollout after the trainer reported ready",
        )
        assert phases[:2] == ["waiting_trainer", "rollout"], phases
        odt = od_of(msg)
        assert odt["trainer"]["state"] == "ready" and odt["trainer"]["progress"] == 1.0
        assert odt["trainer"]["policy_version"] == 1 and odt["trainer"]["metrics"] == {}
        # nothing refused: `detail` is the trainer's own
        assert odt["detail"] == odt["trainer"]["detail"]
        assert odt["detail"].startswith("ready (policy v1")
        # -- rollout 1: the explicit takeover / handback actions, save, return-to-start --------
        assert ctl.action("episode_new")["ok"]
        eid1 = session.recorder_thread.open_episode_id
        assert eid1
        time.sleep(0.8)  # the recorder's first frame opens the video encoder (GIL stall)
        ack = ctl.action("takeover")
        assert ack["ok"] and ack["detail"] == "takeover_transition", ack
        assert ctl.action("takeover") == {**ack, "detail": "already taken over"}  # idempotent
        ctl.hold(["KeyW"], 1.4)  # 0.3 s TRANSITION then HUMAN frames with motion
        wait_human(tele)
        ack = ctl.action("handback")
        assert ack["ok"] and ack["detail"] == "policy", ack
        assert ctl.action("handback")["detail"] == "policy already driving"  # idempotent
        time.sleep(0.6)  # a few more policy frames after the handback
        assert ctl.action("train_now") == {
            "t": "ack",
            "name": "train_now",
            "ok": False,
            "detail": "save or discard the episode first",
        }
        assert ctl.action("episode_save")["ok"]
        seen = watch_until_idle(tele, 60.0)
        assert "saving" in seen and "returning" in seen, seen  # D6 walked `returning`
        msg = wait_tele(tele, lambda m: od_of(m).get("rollouts_saved") == 1, 10.0, "rollout 1")
        assert msg["episode"]["total_episodes"] == 1
        assert od_of(msg)["expert_frames_session"] > 0 and od_of(msg)["novice_frames_session"] > 0
        assert od_of(msg)["phase"] == "rollout"  # one rollout does not trigger the fake
        wait_until(lambda: len(observer.of("episode_saved")) == 1, 10.0, "episode_saved on the bus")
        # -- rollout 2: the Space toggle; the 2nd kept rollout makes the fake train ------------
        assert ctl.action("episode_new")["ok"]
        eid2 = session.recorder_thread.open_episode_id
        assert eid2 and eid2 != eid1
        time.sleep(0.8)
        assert ctl.action("takeover_toggle")["detail"] == "takeover_transition"
        ctl.hold(["KeyW"], 1.4)
        wait_human(tele)
        assert ctl.action("takeover_toggle")["detail"] == "policy"
        time.sleep(0.6)
        assert ctl.action("episode_save")["ok"]
        msg = wait_tele(
            tele,
            lambda m: note_phase(m) and od_of(m).get("phase") == "training",
            30.0,
            "training after the 2nd save",
        )
        ack = ctl.action("episode_new")
        assert ack["ok"] is False and ack["detail"].startswith("training in progress ("), ack
        odt = od_of(msg)
        assert odt["trainer"]["state"] == "training" and odt["rollouts_saved"] == 2
        assert odt["detail"].startswith("training in progress (")
        assert 0.0 <= odt["trainer"]["progress"] <= 1.0
        assert odt["trainer"]["metrics"]["rollouts_trained"] == 2.0
        assert odt["trainer"]["metrics"]["n_expert_frames"] == odt["expert_frames_session"]
        # -- ready with the bumped version: the announced spec is the acting one --------------
        msg = wait_tele(
            tele,
            lambda m: (
                note_phase(m)
                and od_of(m).get("phase") == "rollout"
                and od_of(m).get("policy_version_acting") == 2
            ),
            TRAIN_S + 30.0,
            "rollout with policy v2",
        )
        odt = od_of(msg)
        assert odt["trainer"]["state"] == "ready" and odt["trainer"]["policy_version"] == 2
        assert odt["trainer"]["metrics"]["rollouts_trained"] == 2.0
        assert 0.0 < odt["trainer"]["metrics"]["loss"] < 1.0
        assert odt["trainer"]["metrics"]["n_expert_frames"] == odt["expert_frames_session"] > 0
        assert msg["dagger"]["policy_version"] == "fake-policy/v000002"
        assert phases == ["waiting_trainer", "rollout", "training", "rollout"], phases
        # the second return finished too (it ran while the fake trained)
        wait_tele(tele, lambda m: m["episode"]["state"] == "idle", 30.0, "idle after return")
        assert tele.latest()["episode"]["total_episodes"] == 2
        # -- a rollout discarded while the expert holds the arm: NOTHING on disk --------------
        assert ctl.action("episode_new")["ok"]
        eid3 = session.recorder_thread.open_episode_id
        assert eid3 and eid3 not in (eid1, eid2)
        time.sleep(0.5)
        assert ctl.action("takeover_toggle")["detail"] == "takeover_transition"
        ctl.hold(["KeyW"], 0.8)
        wait_human(tele)
        assert ctl.action("episode_discard")["ok"]  # HUMAN at the boundary: the gate resets
        watch_until_idle(tele, 60.0)
        wait_until(lambda: observer.of("episode_discarded"), 10.0, "episode_discarded on the bus")
        assert not (rollouts / "episodes" / eid3).exists()
        assert not (rollouts / "episodes" / f".tmp-{eid3}").exists()
        assert not (rollouts / "trainer_spool" / f"ep_{eid3}.parquet").exists()
        assert sorted(p.name for p in (rollouts / "episodes").iterdir()) == sorted([eid1, eid2])
        assert od_of(tele.latest())["rollouts_saved"] == 2
        assert tele.latest()["dagger"]["control_mode"] == "policy"
        # -- Train now: the event goes out, the fake retrains on its buffer -> v3 -------------
        ack = ctl.action("train_now")
        assert ack["ok"] is True, ack
        assert ack["detail"] == "asked the trainer to train (2 rollouts saved)"
        wait_until(lambda: observer.of("train_now"), 10.0, "train_now on the bus")
        msg = wait_tele(
            tele,
            lambda m: (
                note_phase(m)
                and od_of(m).get("phase") == "rollout"
                and od_of(m).get("policy_version_acting") == 3
            ),
            TRAIN_S + 30.0,
            "rollout with policy v3",
        )
        assert phases == [
            "waiting_trainer",
            "rollout",
            "training",
            "rollout",
            "training",
            "rollout",
        ]
        assert od_of(msg)["trainer"]["policy_version"] == 3 and od_of(msg)["rollouts_saved"] == 2
        assert msg["dagger"]["policy_version"] == "fake-policy/v000003"
        # -- the wire: the ten generic kinds only, every payload with its exact keys ----------
        wait_until(lambda: len(observer.of("policy_version_changed")) == 2, 10.0, "v3 on the bus")
        events = observer.events()
        kinds = [e["kind"] for e in events]
        assert set(kinds) <= set(EVENT_KINDS), sorted(set(kinds) - set(EVENT_KINDS))
        ours = {"gate", "episode_saved", "episode_discarded", "train_now"}
        assert all(e["session_id"] == sid for e in events if e["kind"] in ours)
        saved = observer.of("episode_saved")
        assert len(saved) == 2
        eids = [e["payload"]["online_dagger"]["episode_id"] for e in saved]
        assert eids == [eid1, eid2]
        for k, e in enumerate(saved, start=1):
            p = e["payload"]
            assert set(p) == SAVED_KEYS
            block = p["online_dagger"]
            assert set(block) == BLOCK_KEYS
            assert block["rollouts_saved"] == k and block["policy_version"] == 1
            assert block["actor_counts"]["expert"] > 0 and block["actor_counts"]["novice"] > 0
            assert block["episode_id"] == p["summary"]["episode_id"]
            assert block["spool_path"] == p["spool_path"] and Path(p["spool_path"]).is_file()
            assert p["summary"]["n_expert_frames"] == block["actor_counts"]["expert"]
            assert p["summary"]["n_novice_frames"] == block["actor_counts"]["novice"]
            assert p["summary"]["n_frames"] == sum(block["actor_counts"].values())
            assert p["dataset_root"] == str(rollouts) and p["run_id"] == sid[:8]
        # the recorder's index counts SAVED episodes: the discarded rollout took none
        assert [e["payload"]["episode_index"] for e in saved] == [0, 1]
        [d] = observer.of("episode_discarded")
        assert d["payload"] == {"episode_index": 2, "episode_id": eid3, "reason": ""}
        [tn] = observer.of("train_now")
        assert tn["payload"] == {"rollouts_saved": 2, "requested_by": "operator"}
        gates = [e["payload"] for e in observer.of("gate")]
        assert all(set(g) == GATE_KEYS and g["arm_id"] == "arm0" for g in gates)
        seqs = [g["seq"] for g in gates]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)  # one session-wide counter
        by_episode = {
            eid: [(g["mode"], g["source"]) for g in gates if g["episode_id"] == eid]
            for eid in (eid1, eid2, eid3)
        }
        assert by_episode[eid1] == [
            ("takeover_transition", "action"),
            ("human", "auto_advance"),
            ("policy", "action"),
        ]
        assert by_episode[eid2] == [
            ("takeover_transition", "keyboard"),
            ("human", "auto_advance"),
            ("policy", "keyboard"),
        ]
        assert by_episode[eid3][:2] == [
            ("takeover_transition", "keyboard"),
            ("human", "auto_advance"),
        ]
        resets = [g for g in gates if g["source"] == "episode_reset"]
        assert [(g["mode"], g["seq"]) for g in resets] == [("policy", seqs[-1])]
        assert resets[0]["episode_id"] == eid3  # the boundary names the episode it closed
        assert {g["source"] for g in gates} == {
            "action",
            "keyboard",
            "auto_advance",
            "episode_reset",
        }
        assert len(gates) == 9
        # the boundary reset spells "episode_boundary" (§6); the handbacks "handback"
        reasons = [e["payload"].get("reason") for e in observer.of("reset_watermark")]
        assert "episode_boundary" in reasons and "handback" in reasons, reasons
        # both version changes landed between episodes (the swap is never mid-episode)
        vcs = [e["payload"] for e in observer.of("policy_version_changed")]
        assert [(v["from"], v["to"]) for v in vcs] == [(1, 2), (2, 3)]
        assert all(v["mid_episode"] is False for v in vcs)
        assert tele.latest()["external"]["version_changes_mid_episode"] == 0
        # -- disk: session.json, the rollouts dataset, actor in frames + spool ----------------
        wait_until(
            lambda: (
                json.loads((sdir / "session.json").read_text())["current"]["phase"] == "rollout"
            ),
            10.0,
            "session.json caught up",
        )
        doc = json.loads((sdir / "session.json").read_text())
        assert doc["session_id"] == sid and doc["session_name"] == "s1" and doc["task"] == TASK
        n_exp = sum(e["payload"]["online_dagger"]["actor_counts"]["expert"] for e in saved)
        n_nov = sum(e["payload"]["online_dagger"]["actor_counts"]["novice"] for e in saved)
        assert doc["current"] == {
            "phase": "rollout",
            "rollouts_saved": 2,
            "expert_frames_session": n_exp,
            "novice_frames_session": n_nov,
        }
        assert doc["paths"] == {"session_dir": str(sdir), "rollouts": str(rollouts)}
        assert (
            doc["spec"]["online_dagger"]["session_name"] == "s1" and doc["spec"]["mode"] == "dagger"
        )
        assert [row["episode_id"] for row in doc["rollouts"]] == eids
        for row, e in zip(doc["rollouts"], saved, strict=True):
            block = e["payload"]["online_dagger"]
            assert row["actor_counts"] == block["actor_counts"] and row["policy_version"] == 1
            assert row["spool_path"] == block["spool_path"] and row["saved_at"].endswith("Z")
        states = [row["state"] for row in doc["trainer_log"]]
        assert [s for s in states if s != "swapped"] == [
            "preparing",
            "ready",
            "training",
            "ready",
            "training",
            "ready",
        ], states
        swaps = [row for row in doc["trainer_log"] if row["state"] == "swapped"]
        assert [(s["policy_version"], s["detail"]) for s in swaps] == [
            (2, "policy v1 -> v2"),
            (3, "policy v2 -> v3"),
        ]
        assert sorted(p.name for p in sdir.iterdir()) == ["rollouts", "session.json"]  # no trainer
        #   artefact: the fake writes nothing; the runtime creates nothing else
        manifest = json.loads((rollouts / "manifest.json").read_text())
        assert manifest["episodes"] == 2 and "actor" in manifest["features"]
        assert manifest["features"]["actor"]["info"]["labels"] == {"0": "novice", "1": "expert"}
        frames = sorted(rollouts.glob("episodes/*/frames.parquet"))
        assert [f.parent.name for f in frames] == sorted(eids)  # ids are capture-time stamps
        for f in frames:
            table = pq.read_table(f)
            actor, mode = ints(table, "actor"), ints(table, "control_mode")
            assert set(actor.tolist()) == {0, 1}, f
            assert np.array_equal(actor, (mode != 0).astype(int)), f  # actor = 1 iff not policy
            sidecar = json.loads((f.parent / "episode.json").read_text())
            k = eids.index(f.parent.name) + 1
            counts = {
                "novice": int(np.count_nonzero(actor == 0)),
                "expert": int(np.count_nonzero(actor == 1)),
            }
            assert sidecar["online_dagger"] == {
                "session_name": "s1",
                "rollouts_saved": k,
                "policy_version": 1,
                "actor_counts": counts,
            }
            assert sidecar["episode_summary"]["n_expert_frames"] == counts["expert"]
            assert [g["seq"] for g in sidecar["gate_events"]] == [
                g["seq"] for g in gates if g["episode_id"] == f.parent.name
            ]
        spools = sorted((rollouts / "trainer_spool").glob("ep_*.parquet"))
        assert [s.name for s in spools] == [f"ep_{e}.parquet" for e in sorted(eids)]
        for s in spools:
            table = pq.read_table(s)
            assert "actor" in table.column_names and set(ints(table, "actor").tolist()) == {0, 1}
        # -- the fake's view: one session, two trainings, the discard remembered --------------
        fake.close()
        stats = fake.stats()["trainer"]
        assert stats["sessions"] == 1 and stats["prepares"] == 1
        assert stats["rollouts_counted"] == 2 and stats["trainings"] == 2 and stats["swaps"] == 2
        assert stats["train_now"] == 1 and stats["discards"] == 1
        assert stats["errors"] == 0 and stats["ignored_events"] == 0
    finally:
        if ctl is not None:
            ctl.close()
        if tele is not None:
            tele.close()
        fake.close()
        r = api.delete("/api/session")
        observer.close()
    assert r.status_code == 204
    # -- teardown wrote the last session.json; the listing + the datasets see s1 -------------
    doc = json.loads((sdir / "session.json").read_text())
    assert doc["current"]["rollouts_saved"] == 2 and doc["last_used_at"].endswith("Z")
    rows = api.get("/api/online_dagger/sessions").json()
    assert rows == [
        {
            "session_name": "s1",
            "path": str(sdir),
            "created_at": doc["created_at"],
            "task": TASK,
            "rollouts": 2,
            "last_used_at": doc["last_used_at"],
        }
    ]
    ds = {d["repo_id"]: d for d in api.get("/api/datasets").json()}
    assert ds["online_dagger/s1"]["namespace"] == "online_dagger"
    assert ds["online_dagger/s1"]["path"] == str(rollouts)
    assert ds["online_dagger/s1"]["total_episodes"] == 2
    # the name is taken now (resume: false); the trainer is gone -> the session-directory rule
    # still comes first (before any trainer 409)
    r = api.post("/api/session", json=SPEC)
    assert r.status_code == 409, r.text
    assert (
        r.json()["detail"]
        == "Online DAgger session 's1' already exists - resume it or pick another name"
    )


def test_skill_endpoints_are_session_less(api):
    r = api.get("/api/online_dagger/skill")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/markdown")
    assert r.text.startswith("---\nname: mavis-online-dagger-trainer\n")
    r = api.get("/api/online_dagger/skill.tgz")
    assert r.status_code == 200 and r.headers["content-type"] == "application/gzip"
    assert r.content[:2] == b"\x1f\x8b"  # gzip magic
    with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tar:
        names = tar.getnames()
        assert "mavis-online-dagger-trainer/SKILL.md" in names
        assert "mavis-online-dagger-trainer/references/contract.md" in names
        text = tar.extractfile("mavis-online-dagger-trainer/SKILL.md").read().decode("utf-8")
    assert text == api.get("/api/online_dagger/skill").text
    # the v1.0 prefix is gone, not aliased
    assert api.get("/api/pro_dagger/skill").status_code == 404
    assert api.get("/api/pro_dagger/sessions").status_code == 404
