"""An Online DAgger session end to end INSIDE the runtime (15-online-dagger §3, §7; phase-14)
over a real sim server, with the dora wiring replaced by a fake (the ``ExternalPolicyHub``
is real; this test plays the bus thread): create -> session directory (``rollouts/`` ONLY)
+ ``session.json`` + the ``SessionAnnounce.online_dagger`` facts; ``episode_new`` refused
with the documented reasons until the trainer reports ``ready`` for THIS session; the
explicit ``takeover`` / ``handback`` actions and Space all publish ``events.gate``; a kept
rollout publishes ``episode_saved`` (+ ``online_dagger`` block), files a ``session.json``
row and carries the sidecar block + the ``actor`` column; ``training`` pauses new rollouts;
the announced version is the acting one ("swapped" in the log); a discard publishes
``episode_discarded`` and leaves NOTHING on disk; ``train_now``; teardown writes the last
``session.json`` and detaches the sink; ``resume: false`` on the name is 409 and
``resume: true`` continues the count; a failed bring-up of a FRESH name removes its
directory again. Stage 2's ``tests/dora_bridge/test_e2e_online_dagger.py`` drives the same
loop over the real control plane with the fake trainer node."""

from __future__ import annotations

import json
import threading
import time

import httpx
import pytest
from conftest import LiveServer, make_runtime_config
from dora_bridge.hubfakes import (
    announce,
    make_bridge_and_hub,
    spec_event,
    trainer_event,
    trainer_status,
)
from test_e2e_teleop import Ctl, Tele

from apollo_mavis_v2_runtime.config import DatasetNamespaceConfig, DatasetsConfig
from apollo_mavis_v2_runtime.dagger.online_dagger import EPISODE_OPEN, NO_TRAINER, NOT_SERVING
from apollo_mavis_v2_runtime.errors import SessionError
from apollo_mavis_v2_runtime.session.manager import SessionManager

pq = pytest.importorskip("pyarrow.parquet")

SPEC = {
    "mode": "dagger",
    "kind": "sim",
    "arms": ["arm0"],
    "frames": {"arm0": "arm_base:arm0"},
    "sim_scene": "single_rail",
    "task": "pick",
    "policy_source": "external",
    "return_to_start": False,  # no initial-condition profile in this rig (D6 is unit-tested)
    "action_filter": {"enabled": False},  # a held arm records every frame
    "online_dagger": {"session_name": "s1"},
}
CAPS = ["online_dagger"]


class FakePublisher:
    """The ``SnapshotPublisher`` surface the coordinator / external source / wiring touch."""

    def __init__(self) -> None:
        self.observation_id = 0
        self.events: list[tuple[str, dict]] = []
        self.session_ids: dict[str, set] = {}  # kind -> the ids the caller pinned

    def observation_t_mono(self, oid):
        return None

    def publish_event(self, kind, payload, session_id=None):
        self.events.append((kind, payload))
        self.session_ids.setdefault(kind, set()).add(session_id)

    def enqueue_event(self, kind, payload):  # plain sessions' gate events (not used here)
        self.events.append((kind, payload))

    def kinds(self):
        return [k for k, _ in self.events]

    def of(self, kind):
        return [p for k, p in self.events if k == kind]


class FakeDora:
    """What ``SessionManager`` touches on an ENABLED ``DoraWiring``, without a bus thread."""

    def __init__(self, tmp) -> None:
        self.enabled = True
        # the hub stamps trainer statuses with ITS clock and the coordinator ages them with
        # its own: both are time.monotonic in the runtime, so here too (a 3 s staleness
        # window - the test heartbeats like a real trainer would)
        self.clock, self.bridge, self.policy_hub, self.node = make_bridge_and_hub(
            tmp / "dora", clock=time.monotonic
        )
        self.publisher = FakePublisher()
        self.depth_camera_ids: list[str] = []
        self.calls: list[str] = []
        self.facts = None

    def before_bringup(self) -> None:
        self.calls.append("before_bringup")

    def after_session_start(self, facts) -> None:
        self.calls.append("after_session_start")
        self.facts = facts

    def after_teardown(self) -> None:
        self.calls.append("after_teardown")

    def camera_announces(self, cams, scene_id, q_by_arm):
        return {}

    def publish_gate_events(self, payloads):
        for p in payloads:
            self.publisher.enqueue_event("gate", p)

    def beat(self, state: str = "idle", **kw) -> None:
        """One trainer_status heartbeat (the real node sends them at 1 Hz); ``session_id``
        is the announce it echoes (None = up, serving nobody: alive, drives nothing)."""
        self.policy_hub._on_trainer_status(trainer_event(trainer_status(state, **kw)))

    def spec(self, version: int) -> None:
        self.policy_hub._on_spec(spec_event(announce(version=version, capabilities=CAPS)))


def wait_until(pred, timeout: float = 10.0, what: str = ""):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = pred()
        if v:
            return v
        time.sleep(0.02)
    raise AssertionError(f"timed out: {what}")


def wait_running(api: httpx.Client) -> None:
    wait_until(lambda: api.get("/api/session").json()["state"] == "running", 30.0, "running")


def wait_tele(tele: Tele, pred, timeout: float = 10.0, what: str = "") -> dict:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        msg = tele.latest()
        if pred(msg):
            return msg
    raise AssertionError(f"telemetry never satisfied: {what}")


def od_of(msg: dict) -> dict | None:
    dagger = msg.get("dagger")
    return dagger.get("online_dagger") if dagger else None


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rt")
    cfg = make_runtime_config(tmp)
    # the shipped layout for the online_dagger namespace: <root>/<s>/rollouts
    cfg.datasets = DatasetsConfig(
        default_namespace="apollo",
        namespaces={"online_dagger": DatasetNamespaceConfig(root=tmp / "od", subdir="rollouts")},
    )
    srv = LiveServer(cfg)
    dora = FakeDora(tmp)
    srv.runtime.manager.dora = dora  # the REST / telemetry `external` block keeps runtime.dora
    yield srv, dora, tmp
    srv.stop()


@pytest.fixture(scope="module")
def api(rig):
    with httpx.Client(base_url=rig[0].http, timeout=60.0) as client:
        yield client


def test_online_dagger_session_lifecycle(rig, api):
    srv, dora, tmp = rig
    hub, pub = dora.policy_hub, dora.publisher
    store = srv.runtime.manager.dataset_store
    sdir = tmp / "od" / "s1"
    dora.spec(1)
    r = api.post("/api/session", json=SPEC)
    assert r.status_code == 200, r.text
    info = r.json()
    assert info["online_dagger"] == {
        "session_name": "s1",
        "resume": False,
        "pause_while_training": True,
        "wait_for_trainer_ready": True,
    }
    assert info["policy_source"] == "external" and info["mode"] == "dagger"
    wait_running(api)
    session = srv.runtime.manager.session
    sid = info["session_id"]  # the trainer echoes it; only such statuses drive the phase
    od = session.online_dagger
    coord = od.coordinator
    # -- the session directory (§0 item 6): rollouts/ + session.json and NOTHING else ---------
    assert od.paths.session_dir == sdir and od.created_fresh
    assert (sdir / "rollouts").is_dir() and (sdir / "session.json").is_file()
    assert sorted(p.name for p in sdir.iterdir()) == ["rollouts", "session.json"]
    assert od.repo_id == "online_dagger/s1"
    assert session.recorder_thread.repo_id == "online_dagger/s1"
    assert store.root_of("online_dagger/s1") == sdir / "rollouts"
    assert session.loop.coordinator is coord and hub.trainer_sink is coord
    assert session.loop.on_gate_events == coord.on_gate_events
    doc = json.loads((sdir / "session.json").read_text(encoding="utf-8"))
    assert doc["session_id"] == sid and doc["task"] == "pick"
    assert doc["current"] == {
        "phase": "waiting_trainer",
        "rollouts_saved": 0,
        "expert_frames_session": 0,
        "novice_frames_session": 0,
    }
    assert doc["spec"]["online_dagger"]["session_name"] == "s1"
    assert doc["paths"] == {"session_dir": str(sdir), "rollouts": str(sdir / "rollouts")}
    assert doc["rollouts"] == [] and doc["trainer_log"] == []
    # -- the announce facts (SessionAnnounce.online_dagger; §6) ------------------------------
    assert dora.calls == ["before_bringup", "after_session_start"]
    facts = dora.facts
    assert facts.policy_source == "external" and facts.dataset_root == str(sdir / "rollouts")
    assert facts.run_id == sid[:8]
    assert facts.online_dagger.model_dump() == {
        "session_name": "s1",
        "session_dir": str(sdir),
        "rollouts_dir": str(sdir / "rollouts"),
    }
    time.sleep(0.3)
    assert pub.kinds() == ["reset_watermark"]  # session_start reset; NO phase event (v2.0)
    ctl, tele = Ctl(srv), Tele(srv)
    try:
        # -- waiting for the trainer: refused with the reasons; telemetry mirrors it -----------
        ack = ctl.action("episode_new")
        assert ack["ok"] is False and ack["detail"] == NO_TRAINER
        ack = ctl.action("train_now")
        assert ack["ok"] is False and ack["detail"] == NO_TRAINER
        # a trainer up but not serving this session yet: alive, refuses with that reason
        dora.beat("idle", session_id=None)
        ack = ctl.action("episode_new")
        assert ack["detail"] == f"waiting for the trainer to report ready ({NOT_SERVING})"
        dora.beat("preparing", progress=0.5, detail="offline pool 1/2", session_id=sid)
        ack = ctl.action("episode_new")
        assert ack["detail"] == "waiting for the trainer to report ready (offline pool 1/2)"
        waiting = "waiting for the trainer to report ready (offline pool 1/2)"
        msg = wait_tele(tele, lambda m: (od_of(m) or {}).get("detail") == waiting, what="alive")
        odt = od_of(msg)
        assert odt["trainer_alive"] is True and odt["phase"] == "waiting_trainer"
        assert odt["rollouts_saved"] == 0 and odt["session_name"] == "s1"
        assert odt["session_dir"] == str(sdir)
        assert odt["trainer"]["progress"] == 0.5 and odt["trainer"]["state"] == "preparing"
        assert odt["policy_version_acting"] == 1
        assert msg["session"]["trainer_alive"] is True  # from the same freshness (§5)
        assert msg["dagger"]["policy_version"] == "fake/v000001"
        # Train now is allowed while waiting (the trainer decides): the event goes out
        ack = ctl.action("train_now")
        assert ack["ok"] is True, ack
        wait_until(lambda: pub.of("train_now"), what="train_now")
        assert pub.of("train_now") == [{"rollouts_saved": 0, "requested_by": "operator"}]
        # -- ready -> rollout ------------------------------------------------------------------
        dora.beat("ready", session_id=sid)
        wait_tele(tele, lambda m: (od_of(m) or {}).get("phase") == "rollout", what="rollout")
        assert ctl.action("episode_new")["ok"]
        time.sleep(0.6)  # the novice "drives" (no action arrives: hold); every frame is kept
        eid = session.recorder_thread.open_episode_id
        assert eid
        dora.beat("ready", session_id=sid)
        assert ctl.action("train_now")["detail"] == EPISODE_OPEN
        # -- the gate API: explicit actions + Space, every instant on the bus -----------------
        ack = ctl.action("takeover")
        assert ack["ok"] and ack["detail"] == "takeover_transition"
        assert ctl.action("takeover")["detail"] == "already taken over"  # idempotent
        time.sleep(0.6)  # T_blend 0.3 s -> HUMAN (auto_advance); the expert "drives"
        ack = ctl.action("handback")
        assert ack["ok"] and ack["detail"] == "policy"
        assert ctl.action("handback")["detail"] == "policy already driving"  # idempotent
        assert ctl.action("takeover_toggle")["detail"] == "takeover_transition"  # Space
        time.sleep(0.5)
        assert ctl.action("takeover_toggle")["detail"] == "policy"
        wait_until(lambda: len(pub.of("gate")) >= 6, what="gate events")
        gates = pub.of("gate")
        assert [(g["mode"], g["source"]) for g in gates[:6]] == [
            ("takeover_transition", "action"),
            ("human", "auto_advance"),
            ("policy", "action"),
            ("takeover_transition", "keyboard"),
            ("human", "auto_advance"),
            ("policy", "keyboard"),
        ]
        assert [g["seq"] for g in gates[:6]] == [1, 2, 3, 4, 5, 6]
        assert all(g["arm_id"] == "arm0" and g["episode_id"] == eid for g in gates[:6])
        assert ctl.action("episode_save")["ok"]
        wait_tele(
            tele,
            lambda m: m["episode"]["state"] == "idle" and m["episode"]["total_episodes"] == 1,
            30.0,
            "saved",
        )
        wait_until(lambda: pub.of("episode_saved"), what="episode_saved")
        # -- the kept rollout: episode_saved + online_dagger block, a session.json row ---------
        [ep] = pub.of("episode_saved")
        n_frames = ep["summary"]["n_frames"]
        n_exp, n_nov = ep["summary"]["n_expert_frames"], ep["summary"]["n_novice_frames"]
        assert n_frames > 10 and ep["summary"]["episode_id"] == eid
        assert n_exp > 0 and n_nov > 0 and n_exp + n_nov == n_frames  # both actors drove
        spool = sdir / "rollouts" / "trainer_spool" / f"ep_{eid}.parquet"
        assert ep["online_dagger"] == {
            "episode_id": eid,
            "rollouts_saved": 1,
            "actor_counts": {"novice": n_nov, "expert": n_exp},
            "policy_version": 1,
            "spool_path": str(spool),
        }
        assert ep["episode_index"] == 0 and ep["run_id"] == sid[:8]
        assert ep["dataset_root"] == str(sdir / "rollouts")
        assert ep["spool_path"] == str(spool) and spool.is_file()
        # the shell publishes rollout-level kinds only (never an iteration / phase event)
        assert set(pub.kinds()) <= {
            "reset_watermark", "train_now", "gate", "episode_saved", "episode_discarded",
        }
        # the episode directory: sidecar block (§4), actor column, spool
        ep_dir = sdir / "rollouts" / "episodes" / eid
        sidecar = json.loads((ep_dir / "episode.json").read_text(encoding="utf-8"))
        assert sidecar["online_dagger"] == {
            "session_name": "s1",
            "rollouts_saved": 1,
            "policy_version": 1,
            "actor_counts": {"novice": n_nov, "expert": n_exp},
        }
        assert sidecar["episode_summary"]["n_expert_frames"] == n_exp
        assert [g["seq"] for g in sidecar["gate_events"]] == [1, 2, 3, 4, 5, 6]
        table = pq.read_table(ep_dir / "frames.parquet")
        assert "actor" in table.column_names and table.num_rows == n_frames
        actors = [v[0] if isinstance(v, list) else v for v in table.column("actor").to_pylist()]
        assert set(actors) == {0, 1} and actors.count(1) == n_exp
        assert "actor" in pq.read_table(spool).column_names
        manifest = json.loads((sdir / "rollouts" / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["episodes"] == 1 and "actor" in manifest["features"]
        # the boundary reset is on the bus too (a held gate would have been reset; here the
        # gate was already in POLICY, so no reset event: the count stays 6)
        assert len(pub.of("gate")) == 6
        # every coordinator event is published with THIS session's id pinned at submit
        # (never resolved later from the publisher's live facts, which teardown clears)
        assert pub.session_ids["gate"] == {sid} and pub.session_ids["episode_saved"] == {sid}
        assert pub.session_ids["train_now"] == {sid}
        # a SAVED rollout of the running session cannot be deleted under the trainer's feet
        r = api.delete(f"/api/datasets/online_dagger/s1/episodes/{eid}")
        assert r.status_code == 409, r.text
        assert r.json()["detail"] == (
            "dataset 'online_dagger/s1' is in use by the running Online DAgger session - end "
            "the session first (the trainer is told about discards, not deletions)"
        )
        assert ep_dir.is_dir()
        wait_until(lambda: json.loads((sdir / "session.json").read_text())["rollouts"], what="row")
        doc = json.loads((sdir / "session.json").read_text(encoding="utf-8"))
        [row] = doc["rollouts"]
        assert row["episode_id"] == eid and row["policy_version"] == 1
        assert row["actor_counts"] == {"novice": n_nov, "expert": n_exp}
        assert row["spool_path"] == str(spool) and row["saved_at"].endswith("Z")
        assert doc["current"]["rollouts_saved"] == 1
        assert doc["current"]["expert_frames_session"] == n_exp
        msg = wait_tele(tele, lambda m: (od_of(m) or {}).get("rollouts_saved") == 1, what="count")
        odt = od_of(msg)
        assert (odt["expert_frames_session"], odt["novice_frames_session"]) == (n_exp, n_nov)
        # -- training pauses new rollouts; ready resumes; the announced version acts -----------
        dora.beat("training", progress=0.5, detail="epoch 4/8", session_id=sid)
        wait_tele(tele, lambda m: (od_of(m) or {}).get("phase") == "training", what="training")
        assert ctl.action("episode_new")["detail"] == "training in progress (epoch 4/8)"
        dora.beat("training", progress=0.75, session_id=sid)
        assert ctl.action("episode_new")["detail"] == "training in progress (75%)"
        # another session's `ready` never moves OUR phase
        dora.beat("ready", policy_version=9, session_id="someone-else")
        time.sleep(0.2)
        assert (coord.phase, coord.trainer.session_id) == ("training", sid)
        dora.beat("ready", policy_version=2, metrics={"loss": 0.3}, session_id=sid)
        wait_tele(tele, lambda m: (od_of(m) or {}).get("phase") == "rollout", what="rollout 2")
        assert od_of(tele.latest())["policy_version_acting"] == 1  # the trainer's claim is not it
        dora.spec(2)  # the spec heartbeat carries v2: THAT is the acting version
        msg = wait_tele(
            tele, lambda m: (od_of(m) or {}).get("policy_version_acting") == 2, what="swapped"
        )
        assert msg["dagger"]["policy_version"] == "fake/v000002"
        assert od_of(msg)["trainer"]["metrics"] == {"loss": 0.3}
        wait_until(
            lambda: any(
                r["state"] == "swapped"
                for r in json.loads((sdir / "session.json").read_text())["trainer_log"]
            ),
            what="swapped row",
        )
        log = json.loads((sdir / "session.json").read_text())["trainer_log"]
        assert [r["state"] for r in log] == ["preparing", "ready", "training", "ready", "swapped"]
        assert log[-1]["policy_version"] == 2 and log[-1]["detail"] == "policy v1 -> v2"
        # -- a discard: episode_discarded, the counter does not move, NOTHING on disk ----------
        dora.beat("ready", policy_version=2, session_id=sid)
        assert ctl.action("episode_new")["ok"]
        time.sleep(0.4)
        eid2 = session.recorder_thread.open_episode_id
        assert eid2 and eid2 != eid
        assert ctl.action("episode_discard")["ok"]
        wait_until(lambda: pub.of("episode_discarded"), what="episode_discarded")
        [d] = pub.of("episode_discarded")
        assert d == {"episode_index": 1, "episode_id": eid2, "reason": ""}
        assert pub.session_ids["episode_discarded"] == {sid}
        wait_tele(tele, lambda m: m["episode"]["state"] == "idle", what="idle after discard")
        assert not (sdir / "rollouts" / "episodes" / eid2).exists()
        assert not (sdir / "rollouts" / "episodes" / f".tmp-{eid2}").exists()
        assert not (sdir / "rollouts" / "trainer_spool" / f"ep_{eid2}.parquet").exists()
        assert sorted(p.name for p in (sdir / "rollouts" / "episodes").iterdir()) == [eid]
        assert od_of(tele.latest())["rollouts_saved"] == 1
        assert len(pub.of("episode_saved")) == 1
        doc = json.loads((sdir / "session.json").read_text(encoding="utf-8"))
        assert doc["current"]["rollouts_saved"] == 1 and len(doc["rollouts"]) == 1
        # Train now after the rollouts: the count rides the event
        ack = ctl.action("train_now")
        assert ack["ok"]
        wait_until(lambda: len(pub.of("train_now")) == 2, what="second train_now")
        assert pub.of("train_now")[-1] == {"rollouts_saved": 1, "requested_by": "operator"}
    finally:
        ctl.close()
        tele.close()
    # -- teardown: last session.json, sink detached, no motion --------------------------------
    assert api.delete("/api/session").status_code == 204
    assert hub.trainer_sink is None and dora.calls[-1] == "after_teardown"
    doc = json.loads((sdir / "session.json").read_text(encoding="utf-8"))
    assert doc["current"]["rollouts_saved"] == 1 and doc["last_used_at"].endswith("Z")
    created_at = doc["created_at"]
    # -- the name again: resume: false is 409, resume: true continues the count ----------------
    r = api.post("/api/session", json=SPEC)
    assert r.status_code == 409, r.text
    assert r.json()["detail"] == (
        "Online DAgger session 's1' already exists - resume it or pick another name"
    )
    resume = {**SPEC, "online_dagger": {**SPEC["online_dagger"], "resume": True}}
    dora.spec(2)  # the spec heartbeat must be fresh at POST (3 s)
    # a running export of the rollouts dataset refuses the resume like any recording POST
    # (before any side effect: the record is untouched)
    record_before = (sdir / "session.json").read_bytes()
    store._exporting = "online_dagger/s1"
    try:
        r = api.post("/api/session", json=resume)
    finally:
        store._exporting = None
    assert r.status_code == 409, r.text
    assert r.json()["detail"] == "dataset 'online_dagger/s1' is being exported - retry in a moment"
    assert (sdir / "session.json").read_bytes() == record_before
    # the trainer is still up, its last heartbeat (`ready`) echoes the PREVIOUS session: the
    # hub replays it into the new coordinator, which must not take it
    dora.beat("ready", policy_version=2, session_id=sid)
    r = api.post("/api/session", json=resume)
    assert r.status_code == 200, r.text
    assert r.json()["online_dagger"]["resume"] is True
    sid2 = r.json()["session_id"]
    assert sid2 != sid
    wait_running(api)
    od2 = srv.runtime.manager.session.online_dagger
    coord2 = od2.coordinator
    assert not od2.created_fresh and coord2.resumed and coord2.created_at == created_at
    # the resumed record is rewritten once the session RUNS (on_session_start), not before
    wait_until(
        lambda: json.loads((sdir / "session.json").read_text())["session_id"] == sid2,
        what="resumed session.json names the new session",
    )
    assert (coord2.rollouts_saved, coord2.phase) == (1, "waiting_trainer")
    assert coord2.expert_frames_session == n_exp
    assert [r["episode_id"] for r in coord2.rollouts] == [eid]
    assert coord2.trainer is None and not coord2.trainer_alive()
    assert coord2.refuse_episode_new() == NO_TRAINER
    assert hub.trainer_sink is coord2 and coord2.policy_version_acting == 2
    # the trainer picks the new announce up and reports ready -> rollout
    dora.beat("ready", policy_version=2, session_id=sid2)
    wait_until(lambda: coord2.phase == "rollout", what="rollout after the trainer re-attached")
    assert coord2.trainer_alive()
    assert srv.runtime.manager.session.recorder_thread.status().total_episodes == 1  # continued
    assert api.delete("/api/session").status_code == 204
    # -- listings ----------------------------------------------------------------------------
    rows = api.get("/api/online_dagger/sessions").json()
    assert [row["session_name"] for row in rows] == ["s1"]
    assert rows[0] == {
        "session_name": "s1",
        "path": str(sdir),
        "created_at": created_at,
        "task": "pick",
        "rollouts": 1,
        "last_used_at": json.loads((sdir / "session.json").read_text())["last_used_at"],
    }
    ds = [d for d in api.get("/api/datasets").json() if d["repo_id"] == "online_dagger/s1"]
    assert len(ds) == 1 and ds[0]["namespace"] == "online_dagger"
    assert ds[0]["path"] == str(sdir / "rollouts") and ds[0]["total_episodes"] == 1


def test_failed_bringup_of_a_fresh_name_removes_its_directory(rig, api, monkeypatch):
    srv, dora, tmp = rig
    hub = dora.policy_hub
    dora.spec(2)

    def boom(self, *a, **k):
        raise SessionError("recorder refused (test)")

    monkeypatch.setattr(SessionManager, "_build_collect_recorder", boom)
    fresh = {**SPEC, "online_dagger": {"session_name": "s9"}}
    r = api.post("/api/session", json=fresh)
    assert r.status_code == 409 and "recorder refused" in r.json()["detail"], r.text
    assert not (tmp / "od" / "s9").exists()  # the fresh name stays usable
    assert hub.trainer_sink is None and api.get("/api/session").status_code == 404
    # a RESUMED session's directory is never removed by a failed bring-up - nor is its
    # session.json rewritten for a session that never ran (the listing order, the
    # session_id and the spec stay those of the last session that did)
    assert (tmp / "od" / "s1" / "session.json").is_file()
    record_before = (tmp / "od" / "s1" / "session.json").read_bytes()
    resume = {**SPEC, "online_dagger": {**SPEC["online_dagger"], "resume": True}}
    dora.spec(2)
    r = api.post("/api/session", json=resume)
    assert r.status_code == 409 and "recorder refused" in r.json()["detail"], r.text
    assert (tmp / "od" / "s1" / "session.json").read_bytes() == record_before
    assert (tmp / "od" / "s1" / "rollouts" / "episodes").is_dir()
    assert api.get("/api/online_dagger/sessions").json()[0]["session_name"] == "s1"


def test_resume_refuses_an_unreadable_session_json_instead_of_overwriting_it(rig, api):
    """The rollouts rows and the counters a resume continues live only in ``session.json``:
    a corrupt record is a 409 at POST (before any side effect), never replaced."""
    srv, dora, tmp = rig
    path = tmp / "od" / "s1" / "session.json"
    original = path.read_bytes()
    resume = {**SPEC, "online_dagger": {**SPEC["online_dagger"], "resume": True}}
    try:
        path.write_text("{not json", encoding="utf-8")
        dora.spec(2)
        r = api.post("/api/session", json=resume)
        assert r.status_code == 409, r.text
        assert r.json()["detail"] == (
            "Online DAgger session 's1': session.json is unreadable - fix or remove it"
        )
        assert path.read_text(encoding="utf-8") == "{not json"  # untouched
        assert api.get("/api/session").status_code == 404
        path.write_text("[]", encoding="utf-8")  # valid JSON, not a session document
        dora.spec(2)
        r = api.post("/api/session", json=resume)
        assert r.status_code == 409 and "unreadable" in r.json()["detail"], r.text
        assert path.read_text(encoding="utf-8") == "[]"
    finally:
        path.write_bytes(original)


def test_a_start_failure_after_the_stack_is_built_rolls_the_bringup_back(rig, api, monkeypatch):
    """The rollback covers the WHOLE sim bring-up, not only the recorder build: a failing
    ``policy_session.start()`` (loop and recorder already running) stops them, drops the
    FRESH session directory so the name stays usable, and leaves no session behind."""
    import shutil

    from apollo_mavis_v2_core.protocol import SessionSpec

    from apollo_mavis_v2_runtime.dagger.loop import DaggerSession

    srv, dora, tmp = rig
    manager = srv.runtime.manager
    hub = dora.policy_hub
    fresh = {**SPEC, "online_dagger": {"session_name": "s8"}}

    def boom(self):
        raise RuntimeError("policy source refused (test)")

    try:
        with monkeypatch.context() as m:
            m.setattr(DaggerSession, "start", boom)
            dora.spec(2)
            with pytest.raises(RuntimeError, match="policy source refused"):
                manager.create(SessionSpec.model_validate(fresh))
        assert not (tmp / "od" / "s8").exists()  # the fresh name stays usable
        assert manager.session is None and hub.trainer_sink is None
        assert api.get("/api/session").status_code == 404
        assert not any(t.name == "recorder" and t.is_alive() for t in threading.enumerate())
        # ... and the same name comes up normally afterwards
        dora.spec(2)
        r = api.post("/api/session", json=fresh)
        assert r.status_code == 200, r.text
        wait_running(api)
        assert (tmp / "od" / "s8" / "session.json").is_file()
        assert sorted(p.name for p in (tmp / "od" / "s8").iterdir()) == ["rollouts", "session.json"]
        assert api.delete("/api/session").status_code == 204
    finally:
        shutil.rmtree(tmp / "od" / "s8", ignore_errors=True)
