"""``OnlineDaggerCoordinator`` (15-online-dagger §3; phase-14) with an injected clock,
publisher and session-file writer: the start phase and the four refusal strings verbatim;
which trainer statuses count (this session's id drives, ``null`` = alive only, another id
ignored, older-than-newest dropped); ``ready`` / ``training`` / ``error`` / recovery;
``pause_while_training`` / ``wait_for_trainer_ready`` off; a kept rollout's counters,
``episode_saved`` block and ``session.json`` row; a discard = ``episode_discarded`` event
ONLY with nothing on disk; ``train_now`` event + refusals; gate events; the acting policy
version + ``trainer_log``; the ``session.json`` shape and resume; ``scan_sessions`` rows;
the serial worker and the atomic writer."""

from __future__ import annotations

import dataclasses
import json
import threading

import pytest
from apollo_mavis_v2_core.dagger import EpisodeSummary
from apollo_mavis_v2_core.protocol import (
    OnlineDaggerConfig,
    OnlineDaggerSessionInfo,
    OnlineDaggerStatus,
    SessionSpec,
)
from apollo_mavis_v2_core.protocol.external import (
    EVENT_KINDS,
    OnlineDaggerAnnounce,
    TrainerStatusAnnounce,
)

from apollo_mavis_v2_runtime.dagger.online_dagger import (
    EPISODE_OPEN,
    NO_STATUS_YET,
    NO_TRAINER,
    NOT_SERVING,
    PHASES,
    TRAINER_LOG_MAX,
    OnlineDaggerCoordinator,
    OnlineDaggerPaths,
    SerialWorker,
    write_session_json_atomic,
)

T0 = 1000.0
WALL0 = 1_757_000_000.0


class Clock:
    def __init__(self) -> None:
        self.t = T0

    def __call__(self) -> float:
        return self.t


def spec_for(cfg: OnlineDaggerConfig) -> SessionSpec:
    return SessionSpec(
        mode="dagger",
        kind="sim",
        arms=["arm0"],
        frames={"arm0": "arm_base:arm0"},
        sim_scene="single_rail",
        task="pick",
        policy_source="external",
        online_dagger=cfg,
    )


def paths_for(tmp_path, name: str = "s1") -> OnlineDaggerPaths:
    d = tmp_path / "online_dagger" / name
    return OnlineDaggerPaths(session_dir=d, rollouts_dir=d / "rollouts")


def make(tmp_path, *, policy_version: int | None = 1, **over):
    cfg = OnlineDaggerConfig(session_name="s1", **over)
    clock = Clock()
    events: list[tuple[str, dict]] = []
    writes: list[dict] = []
    coord = OnlineDaggerCoordinator(
        cfg,
        session_id="sess1",
        paths=paths_for(tmp_path),
        spec=spec_for(cfg),
        run_id="run1",
        spec_stale_s=3.0,
        policy_version=policy_version,
        clock=clock,
        wallclock=lambda: WALL0 + (clock.t - T0),
        publish=lambda k, p: events.append((k, p)),
        write_session=writes.append,
    )
    return coord, events, writes, clock


def ts(state="idle", *, session_id="sess1", version=1, progress=0.0, detail="", **kw):
    """A trainer status; ``session_id`` defaults to the coordinator's own id (``make``);
    None / another id = not ours."""
    return TrainerStatusAnnounce(
        trainer_id="repo/online_dagger",
        node_version="0.1",
        state=state,
        session_id=session_id,
        policy_version=version,
        progress=progress,
        detail=detail,
        **kw,
    )


def summary(i: int, expert: int = 4, novice: int = 6) -> EpisodeSummary:
    return EpisodeSummary(
        episode_index=i,
        n_frames=expert + novice,
        n_intervention_frames=expert,
        n_label_frames=max(0, expert - 1),
        takeover_segments=1,
        segment_doubts=[0.1],
        success=None,
        episode_id=f"ep{i}",
        n_expert_frames=expert,
        n_novice_frames=novice,
    )


def save(coord, i: int, expert: int = 4, novice: int = 6):
    return coord.on_episode_saved(
        f"ep{i}", i, summary(i, expert, novice), f"/spool/ep_ep{i}.parquet"
    )


def kinds(events):
    return [k for k, _ in events]


# -- start + refusals ------------------------------------------------------------------------------
def test_start_is_waiting_trainer_and_refuses_with_the_documented_reasons(tmp_path):
    coord, events, writes, clock = make(tmp_path)
    assert PHASES == ("waiting_trainer", "rollout", "training", "error")
    assert coord.phase == "waiting_trainer" and coord.rollouts_saved == 0
    assert coord.on_session_start() == [] and events == []  # no phase events on the wire
    assert coord.started and len(writes) == 1
    assert writes[0]["current"] == {
        "phase": "waiting_trainer",
        "rollouts_saved": 0,
        "expert_frames_session": 0,
        "novice_frames_session": 0,
    }
    # no status at all: not attached
    assert coord.refuse_episode_new() == NO_TRAINER == "no Online DAgger trainer attached"
    assert coord.trainer_alive() is False and coord.trainer_age() is None
    assert NO_STATUS_YET == "no trainer status yet"  # the detail while no status arrived yet
    assert coord.status().detail == NO_TRAINER  # ... but the refusal itself is "no trainer"
    # a trainer up but serving nobody yet: alive, still waiting, the detail says so
    assert coord.on_trainer_status(ts("idle", session_id=None), clock()) == []
    assert coord.trainer_alive() is True and coord.phase == "waiting_trainer"
    assert coord.refuse_episode_new() == (
        f"waiting for the trainer to report ready ({NOT_SERVING})"
    )
    # our session, preparing with a detail: the detail rides the refusal
    coord.on_trainer_status(ts("preparing", detail="loading the offline pool"), clock())
    assert coord.refuse_episode_new() == (
        "waiting for the trainer to report ready (loading the offline pool)"
    )
    coord.on_trainer_status(ts("preparing"), clock())  # no detail -> the state
    assert coord.refuse_episode_new() == (
        "waiting for the trainer to report ready (trainer preparing)"
    )
    # training BEFORE any ready (pause on): the training refusal, progress as the detail
    coord.on_trainer_status(ts("training", progress=0.42), clock())
    assert coord.phase == "training"
    assert coord.refuse_episode_new() == "training in progress (42%)"
    coord.on_trainer_status(ts("training", progress=0.5, detail="epoch 2/8"), clock())
    assert coord.refuse_episode_new() == "training in progress (epoch 2/8)"
    # error: the trainer's detail, "unknown" without one
    coord.on_trainer_status(ts("error", detail="loss diverged"), clock())
    assert coord.phase == "error" and coord.refuse_episode_new() == "trainer error: loss diverged"
    coord.on_trainer_status(ts("error"), clock())
    assert coord.refuse_episode_new() == "trainer error: unknown"
    # recovery without a ready yet: back to waiting
    coord.on_trainer_status(ts("idle"), clock())
    assert coord.phase == "waiting_trainer"
    # ready: rollouts admitted
    coord.on_trainer_status(ts("ready"), clock())
    assert coord.phase == "rollout" and coord.refuse_episode_new() is None
    assert coord.trainer_seen_ready is True
    # stale (> spec_stale_s): the phase stays, the refusal is "no trainer"
    clock.t += 3.1
    assert coord.phase == "rollout" and coord.trainer_alive() is False
    assert coord.refuse_episode_new() == NO_TRAINER
    assert coord.trainer_age() == pytest.approx(3.1)
    assert events == []  # nothing above published an event


def test_ready_latch_and_pause_semantics(tmp_path):
    coord, events, writes, clock = make(tmp_path)
    coord.on_session_start()
    coord.on_trainer_status(ts("ready"), clock())
    assert coord.phase == "rollout"
    # idle / preparing AFTER ready keep rolling out (the trainer is not training)
    coord.on_trainer_status(ts("idle"), clock())
    assert coord.phase == "rollout" and coord.refuse_episode_new() is None
    coord.on_trainer_status(ts("preparing"), clock())
    assert coord.phase == "rollout"
    # training pauses (default), ready resumes
    coord.on_trainer_status(ts("training", progress=0.1), clock())
    assert coord.phase == "training" and coord.refuse_episode_new().startswith("training in")
    coord.on_trainer_status(ts("ready", version=2), clock())
    assert coord.phase == "rollout"
    # error after ready recovers to rollout
    coord.on_trainer_status(ts("error", detail="x"), clock())
    assert coord.phase == "error"
    coord.on_trainer_status(ts("idle"), clock())
    assert coord.phase == "rollout"
    assert events == []


def test_pause_while_training_false_keeps_rolling_out_and_the_pill_just_shows_it(tmp_path):
    coord, events, writes, clock = make(tmp_path, pause_while_training=False)
    coord.on_session_start()
    coord.on_trainer_status(ts("training", progress=0.3), clock())
    assert coord.phase == "waiting_trainer"  # never saw ready
    coord.on_trainer_status(ts("ready"), clock())
    coord.on_trainer_status(ts("training", progress=0.3), clock())
    assert coord.phase == "rollout" and coord.refuse_episode_new() is None
    st = coord.status()
    assert st.trainer.state == "training" and st.detail == ""  # allowed: no refusal text
    coord.on_trainer_status(ts("error", detail="boom"), clock())
    assert coord.phase == "error" and coord.refuse_episode_new() == "trainer error: boom"


def test_wait_for_trainer_ready_false_starts_in_rollout(tmp_path):
    coord, events, writes, clock = make(tmp_path, wait_for_trainer_ready=False)
    assert coord.phase == "rollout"
    coord.on_session_start()
    assert coord.refuse_episode_new() == NO_TRAINER  # still needs a fresh status
    coord.on_trainer_status(ts("idle"), clock())
    assert coord.phase == "rollout" and coord.refuse_episode_new() is None
    coord.on_trainer_status(ts("preparing"), clock())
    assert coord.phase == "rollout"
    coord.on_trainer_status(ts("training"), clock())
    assert coord.phase == "training"  # pause still applies
    coord.on_trainer_status(ts("idle"), clock())
    assert coord.phase == "rollout"


# -- which status counts ---------------------------------------------------------------------------
def test_only_a_status_echoing_this_session_drives_the_state_machine(tmp_path):
    coord, events, writes, clock = make(tmp_path)
    coord.on_session_start()
    # another session's ready: ignored outright (not even alive)
    assert coord.on_trainer_status(ts("ready", session_id="someone-else"), clock()) == []
    assert coord.trainer is None and coord.trainer_alive() is False
    assert coord.phase == "waiting_trainer"
    # null: alive, shown verbatim, drives nothing
    coord.on_trainer_status(ts("ready", session_id=None), clock())
    assert coord.trainer_alive() and coord.trainer.session_id is None
    assert coord.phase == "waiting_trainer" and coord.trainer_seen_ready is False
    assert coord.status().detail == f"waiting for the trainer to report ready ({NOT_SERVING})"
    # ours
    coord.on_trainer_status(ts("ready"), clock())
    assert coord.phase == "rollout"
    # another session's error never parks us
    coord.on_trainer_status(ts("error", session_id="someone-else", detail="x"), clock())
    assert coord.phase == "rollout" and coord.trainer.state == "ready"


def test_a_status_older_than_the_newest_seen_is_dropped(tmp_path):
    coord, events, writes, clock = make(tmp_path)
    coord.on_session_start()
    coord.on_trainer_status(ts("ready"), clock())
    t_new = clock()
    # the hub's attach replay lands with an OLDER receive time: dropped
    assert coord.on_trainer_status(ts("error", detail="old"), t_new - 0.5) == []
    assert coord.phase == "rollout" and coord.trainer.state == "ready"
    assert coord.trainer_age() == 0.0
    # equal time is accepted (same-instant ordering)
    coord.on_trainer_status(ts("training"), t_new)
    assert coord.phase == "training"


# -- rollouts --------------------------------------------------------------------------------------
def test_kept_rollout_counts_publishes_the_block_and_files_a_row(tmp_path):
    coord, events, writes, clock = make(tmp_path)
    coord.on_session_start()
    coord.on_trainer_status(ts("ready"), clock())
    del writes[:]
    clock.t += 1.5
    out = save(coord, 0, expert=4, novice=6)
    assert kinds(out) == ["episode_saved"] and events == out
    payload = out[0][1]
    assert payload["episode_index"] == 0 and payload["run_id"] == "run1"
    assert payload["dataset_root"] == str(tmp_path / "online_dagger" / "s1" / "rollouts")
    assert payload["spool_path"] == "/spool/ep_ep0.parquet"
    assert payload["summary"] == dataclasses.asdict(summary(0, 4, 6))
    assert payload["online_dagger"] == {
        "episode_id": "ep0",
        "rollouts_saved": 1,
        "actor_counts": {"novice": 6, "expert": 4},
        "policy_version": 1,
        "spool_path": "/spool/ep_ep0.parquet",
    }
    assert (coord.rollouts_saved, coord.expert_frames_session, coord.novice_frames_session) == (
        1, 4, 6,
    )
    assert coord.phase == "rollout"  # the shell never counts iterations
    # session.json written at once with the row
    assert len(writes) == 1
    doc = writes[0]
    assert doc["current"]["rollouts_saved"] == 1
    assert doc["rollouts"] == [
        {
            "episode_id": "ep0",
            "saved_at": "2025-09-04T15:33:21.500Z",
            "actor_counts": {"novice": 6, "expert": 4},
            "policy_version": 1,
            "spool_path": "/spool/ep_ep0.parquet",
        }
    ]
    save(coord, 1, expert=0, novice=10)
    assert coord.rollouts_saved == 2 and coord.expert_frames_session == 4
    assert coord.novice_frames_session == 16
    assert events[-1][1]["online_dagger"]["rollouts_saved"] == 2
    assert [r["episode_id"] for r in writes[-1]["rollouts"]] == ["ep0", "ep1"]


def test_a_rollout_whose_spool_write_failed_is_still_a_kept_rollout(tmp_path):
    """The recorder hands ``spool_path=""`` when the trainer spool could not be written
    (pyarrow / disk full): the rollout IS on disk, so it counts, is filed and announced -
    with ``spool_path: null`` on the wire and in the row, never a stray ``""`` path."""
    coord, events, writes, clock = make(tmp_path)
    coord.on_session_start()
    coord.on_trainer_status(ts("ready"), clock())
    out = coord.on_episode_saved("ep0", 0, summary(0), "")
    payload = out[0][1]
    assert payload["spool_path"] is None and payload["online_dagger"]["spool_path"] is None
    assert payload["online_dagger"]["rollouts_saved"] == 1 and coord.rollouts_saved == 1
    assert writes[-1]["rollouts"][0]["spool_path"] is None
    assert coord.sidecar_block({})["rollouts_saved"] == 2  # the next rollout numbers on


def test_sidecar_block_matches_the_episode_saved_block(tmp_path):
    coord, events, writes, clock = make(tmp_path)
    coord.on_session_start()
    coord.on_trainer_status(ts("ready"), clock())
    block = coord.sidecar_block({"novice": 6, "expert": 4})
    assert block == {
        "session_name": "s1",
        "rollouts_saved": 1,
        "policy_version": 1,
        "actor_counts": {"novice": 6, "expert": 4},
    }
    save(coord, 0)
    assert events[-1][1]["online_dagger"]["rollouts_saved"] == block["rollouts_saved"]
    assert coord.sidecar_block({})["rollouts_saved"] == 2
    assert coord.sidecar_block({})["actor_counts"] == {"novice": 0, "expert": 0}


def test_discard_publishes_the_event_only_and_leaves_nothing_on_disk(tmp_path):
    coord, events, writes, clock = make(tmp_path)
    paths = coord.paths
    paths.mkdirs()
    coord.on_session_start()
    coord.on_trainer_status(ts("ready"), clock())
    n_writes = len(writes)
    out = coord.on_episode_discarded("ep7", 3, "")
    assert out == [
        ("episode_discarded", {"episode_index": 3, "episode_id": "ep7", "reason": ""})
    ] and events == out
    assert "iteration" not in out[0][1]  # the shell has no iterations
    assert coord.rollouts_saved == 0 and coord.rollouts == []  # the counter does not move
    assert len(writes) == n_writes  # nothing to persist
    # NOTHING is persisted for a discarded rollout: no episode directory, no spool row
    assert not (paths.rollouts_dir / "episodes" / "ep7").exists()
    assert not (paths.rollouts_dir / "episodes" / ".tmp-ep7").exists()
    assert not (paths.rollouts_dir / "trainer_spool" / "ep_ep7.parquet").exists()
    assert sorted(p.name for p in paths.session_dir.iterdir()) == ["rollouts"]
    assert list(paths.rollouts_dir.iterdir()) == []
    # the teardown discard spells its reason; a never-minted id is None
    out = coord.on_episode_discarded(None, None, "session teardown")
    assert out[0][1] == {"episode_index": None, "episode_id": None, "reason": "session teardown"}
    # a following save still counts from 0 -> 1
    save(coord, 0)
    assert coord.rollouts_saved == 1 and events[-1][1]["online_dagger"]["rollouts_saved"] == 1


# -- train now -------------------------------------------------------------------------------------
def test_train_now_publishes_the_event_and_its_refusals(tmp_path):
    coord, events, writes, clock = make(tmp_path)
    coord.on_session_start()
    assert coord.request_train_now(episode_open=True) == (False, EPISODE_OPEN)
    assert EPISODE_OPEN == "save or discard the episode first"
    assert coord.request_train_now() == (False, NO_TRAINER)
    coord.on_trainer_status(ts("ready"), clock())
    save(coord, 0)
    save(coord, 1)
    n = len(writes)
    ok, detail = coord.request_train_now()
    assert ok and detail == "asked the trainer to train (2 rollouts saved)"
    assert events[-1] == ("train_now", {"rollouts_saved": 2, "requested_by": "operator"})
    assert len(writes) == n  # no file write for a request
    # the open-episode rule wins over everything
    assert coord.request_train_now(episode_open=True) == (False, EPISODE_OPEN)
    # allowed with zero rollouts (the trainer decides), and while training / waiting
    coord2, events2, _w, clock2 = make(tmp_path)
    coord2.on_trainer_status(ts("preparing"), clock2())
    assert coord2.request_train_now() == (True, "asked the trainer to train (0 rollouts saved)")
    coord2.on_trainer_status(ts("training"), clock2())
    assert coord2.request_train_now()[0] is True
    assert kinds(events2) == ["train_now", "train_now"]
    # stale trainer: refused
    clock.t += 3.1
    assert coord.request_train_now() == (False, NO_TRAINER)
    assert "train_now" in EVENT_KINDS and "gate" in EVENT_KINDS


# -- gate events -----------------------------------------------------------------------------------
def test_gate_events_are_published_verbatim(tmp_path):
    coord, events, writes, clock = make(tmp_path)
    coord.on_session_start()
    n = len(writes)
    p1 = {"arm_id": "arm0", "mode": "takeover_transition", "seq": 1, "source": "keyboard",
          "episode_id": "ep0"}
    p2 = {"arm_id": "arm0", "mode": "human", "seq": 2, "source": "auto_advance",
          "episode_id": "ep0"}
    assert coord.on_gate_events([p1, p2]) == [("gate", p1), ("gate", p2)]
    assert events == [("gate", p1), ("gate", p2)]
    assert coord.on_gate_events([]) == [] and len(events) == 2
    assert len(writes) == n  # gate events never touch the file
    assert coord.events_published == 2


# -- policy version + trainer log ------------------------------------------------------------------
def test_spec_version_is_the_acting_version_and_a_change_is_logged(tmp_path):
    coord, events, writes, clock = make(tmp_path)
    coord.on_session_start()
    assert coord.policy_version_acting == 1
    assert coord.on_spec_version(1) == [] and coord.trainer_log == []
    n = len(writes)
    clock.t += 2.0
    coord.on_spec_version(2)
    assert coord.policy_version_acting == 2
    assert coord.trainer_log == [
        {"at": "2025-09-04T15:33:22.000Z", "state": "swapped", "policy_version": 2,
         "detail": "policy v1 -> v2"}
    ]
    assert len(writes) == n + 1 and writes[-1]["trainer_log"] == coord.trainer_log
    # the trainer's own claim never moves the acting version
    coord.on_trainer_status(ts("ready", version=9), clock())
    assert coord.policy_version_acting == 2 and coord.status().policy_version_acting == 2
    # trainer state changes are logged once per change (heartbeats of the same state are not)
    coord.on_trainer_status(ts("ready", version=9), clock())
    coord.on_trainer_status(ts("training", version=9, detail="epoch 1"), clock())
    coord.on_trainer_status(ts("training", version=9, detail="epoch 2"), clock())
    assert [(r["state"], r["policy_version"], r["detail"]) for r in coord.trainer_log] == [
        ("swapped", 2, "policy v1 -> v2"),
        ("ready", 9, ""),
        ("training", 9, "epoch 1"),
    ]
    # a coordinator without an initial version adopts the first announced one silently
    c2, _e, w2, _c = make(tmp_path, policy_version=None)
    assert c2.policy_version_acting is None
    c2.on_spec_version(3)
    assert c2.policy_version_acting == 3 and c2.trainer_log == [] and w2 == []


def test_trainer_log_is_capped(tmp_path):
    coord, events, writes, clock = make(tmp_path)
    for i in range(TRAINER_LOG_MAX + 25):
        coord.on_spec_version(i + 2)
    assert len(coord.trainer_log) == TRAINER_LOG_MAX
    assert coord.trainer_log[-1]["policy_version"] == TRAINER_LOG_MAX + 26
    assert len(coord.to_session_json()["trainer_log"]) == TRAINER_LOG_MAX


# -- status ----------------------------------------------------------------------------------------
def test_status_round_trips_through_the_wire_model(tmp_path):
    coord, events, writes, clock = make(tmp_path)
    coord.on_session_start()
    st = coord.status()
    assert isinstance(st, OnlineDaggerStatus)
    assert st.model_dump() == {
        "session_name": "s1",
        "phase": "waiting_trainer",
        "rollouts_saved": 0,
        "detail": NO_TRAINER,
        "trainer_alive": False,
        "trainer_age_s": None,
        "trainer": None,
        "policy_version_acting": 1,
        "expert_frames_session": 0,
        "novice_frames_session": 0,
        "session_dir": str(tmp_path / "online_dagger" / "s1"),
    }
    msg = ts("training", progress=0.25, detail="epoch 2/8", metrics={"loss": 0.5})
    coord.on_trainer_status(msg, clock())
    save(coord, 0, expert=3, novice=7)
    clock.t += 0.5
    st = coord.status()
    assert st.phase == "training" and st.detail == "training in progress (epoch 2/8)"
    assert st.trainer == msg and st.trainer_alive and st.trainer_age_s == pytest.approx(0.5)
    assert (st.rollouts_saved, st.expert_frames_session, st.novice_frames_session) == (1, 3, 7)
    again = OnlineDaggerStatus.model_validate_json(st.model_dump_json())
    assert again == st and again.trainer.metrics == {"loss": 0.5}
    # allowed: the detail is the trainer's own text
    coord.on_trainer_status(ts("ready", detail="v2 installed"), clock())
    assert coord.status().detail == "v2 installed"
    assert coord.status(now=clock() + 10.0).trainer_alive is False


# -- session.json / resume / scan ------------------------------------------------------------------
def test_session_json_shape_and_resume(tmp_path):
    coord, events, writes, clock = make(tmp_path)
    coord.on_session_start()
    coord.on_trainer_status(ts("ready", version=1), clock())
    save(coord, 0, expert=4, novice=6)
    coord.on_spec_version(2)
    save(coord, 1, expert=1, novice=9)
    coord.on_trainer_status(ts("training"), clock())
    clock.t += 7.0
    coord.close()
    doc = writes[-1]
    sdir = tmp_path / "online_dagger" / "s1"
    assert list(doc) == [
        "session_name", "created_at", "session_id", "task", "spec", "paths", "rollouts",
        "trainer_log", "current", "last_used_at",
    ]
    assert doc["session_name"] == "s1" and doc["session_id"] == "sess1" and doc["task"] == "pick"
    assert doc["created_at"] == "2025-09-04T15:33:20.000Z"
    assert doc["last_used_at"] == "2025-09-04T15:33:27.000Z"
    assert doc["spec"]["online_dagger"] == {
        "session_name": "s1", "resume": False, "pause_while_training": True,
        "wait_for_trainer_ready": True,
    }
    assert doc["paths"] == {"session_dir": str(sdir), "rollouts": str(sdir / "rollouts")}
    assert [(r["episode_id"], r["policy_version"]) for r in doc["rollouts"]] == [
        ("ep0", 1),
        ("ep1", 2),
    ]
    assert doc["current"] == {
        "phase": "training", "rollouts_saved": 2, "expert_frames_session": 5,
        "novice_frames_session": 15,
    }
    assert [r["state"] for r in doc["trainer_log"]] == ["ready", "swapped", "training"]
    json.dumps(doc)  # JSON-serialisable
    # -- resume: the counters and rows continue; the phase restarts; ready must be re-seen
    cfg2 = OnlineDaggerConfig(session_name="s1", resume=True)
    events2: list = []
    writes2: list = []
    c2 = OnlineDaggerCoordinator(
        cfg2, session_id="sess2", paths=paths_for(tmp_path), spec=spec_for(cfg2),
        policy_version=2, clock=clock, wallclock=lambda: WALL0 + 100,
        publish=lambda k, p: events2.append((k, p)), write_session=writes2.append,
    )
    c2.from_session_json(doc)
    assert c2.resumed and c2.created_at == doc["created_at"] and c2.task == "pick"
    assert (c2.rollouts_saved, c2.expert_frames_session, c2.novice_frames_session) == (2, 5, 15)
    assert [r["episode_id"] for r in c2.rollouts] == ["ep0", "ep1"]
    assert c2.trainer_log == doc["trainer_log"]
    assert c2.phase == "waiting_trainer" and not c2.trainer_seen_ready
    c2.on_session_start()
    assert c2.refuse_episode_new() == NO_TRAINER
    # the OLD session's ready (replayed by the hub) does not count
    c2.on_trainer_status(ts("ready", session_id="sess1"), clock())
    assert c2.phase == "waiting_trainer" and not c2.trainer_alive()
    c2.on_trainer_status(ts("ready", session_id="sess2"), clock())
    assert c2.phase == "rollout"
    save(c2, 2)
    assert c2.rollouts_saved == 3 and events2[-1][1]["online_dagger"]["rollouts_saved"] == 3
    assert writes2[-1]["current"]["rollouts_saved"] == 3 and len(writes2[-1]["rollouts"]) == 3
    assert writes2[-1]["session_id"] == "sess2" and writes2[-1]["created_at"] == doc["created_at"]
    # a file with fewer rows than the counter keeps the counter; a bare file resets to 0
    c3, *_ = make(tmp_path)
    c3.from_session_json({"current": {"rollouts_saved": 5}})
    assert c3.rollouts_saved == 5 and c3.rollouts == []
    c4, *_ = make(tmp_path, wait_for_trainer_ready=False)
    c4.from_session_json({})
    assert c4.rollouts_saved == 0 and c4.phase == "rollout"


def test_scan_sessions_reads_session_json_rows_newest_first(tmp_path):
    root = tmp_path / "online_dagger"
    assert OnlineDaggerCoordinator.scan_sessions(None) == []
    assert OnlineDaggerCoordinator.scan_sessions(root) == []  # missing root
    docs = {
        "old": {
            "session_name": "old", "created_at": "2026-09-08T10:00:00.000Z", "task": "pick",
            "rollouts": [{"episode_id": "a"}, {"episode_id": "b"}],
            "current": {"phase": "rollout", "rollouts_saved": 2},
            "last_used_at": "2026-09-08T11:00:00.000Z",
        },
        "new": {
            "session_name": "new", "created_at": "2026-09-08T12:00:00.000Z",
            "spec": {"task": "place", "online_dagger": {"session_name": "new"}},
            "current": {"phase": "waiting_trainer", "rollouts_saved": 0},
            "last_used_at": "2026-09-08T12:30:00.000Z",
        },
        "bare": {"rollouts": [{"episode_id": "x"}]},  # counts fall back to the rows
    }
    for name, doc in docs.items():
        write_session_json_atomic(root / name / "session.json", doc)
    (root / "junk").mkdir()
    (root / "junk" / "session.json").write_text("nope")
    (root / "list").mkdir()
    (root / "list" / "session.json").write_text("[]")
    (root / "nofile").mkdir()
    rows = OnlineDaggerCoordinator.scan_sessions(root)
    assert all(isinstance(r, OnlineDaggerSessionInfo) for r in rows)
    # "bare" has no created_at / last_used_at: it sorts by the file's mtime (now)
    assert [r.session_name for r in rows if r.session_name != "bare"] == ["new", "old"]
    by_name = {r.session_name: r for r in rows}
    assert set(by_name) == {"new", "old", "bare"}  # junk / list / nofile skipped
    assert by_name["old"].model_dump() == {
        "session_name": "old", "path": str(root / "old"),
        "created_at": "2026-09-08T10:00:00.000Z", "task": "pick", "rollouts": 2,
        "last_used_at": "2026-09-08T11:00:00.000Z",
    }
    assert by_name["new"].task == "place" and by_name["new"].rollouts == 0
    bare = by_name["bare"]
    assert bare.rollouts == 1 and bare.task is None and bare.last_used_at is None
    assert bare.created_at.endswith("Z")  # the file's mtime


def test_announce(tmp_path):
    coord, *_ = make(tmp_path)
    ann = coord.announce()
    assert isinstance(ann, OnlineDaggerAnnounce)
    sdir = tmp_path / "online_dagger" / "s1"
    assert ann.model_dump() == {
        "session_name": "s1", "session_dir": str(sdir), "rollouts_dir": str(sdir / "rollouts"),
    }
    assert list(OnlineDaggerAnnounce.model_fields) == [
        "session_name", "session_dir", "rollouts_dir",
    ]
    assert coord.session_name == "s1"


# -- side effects ----------------------------------------------------------------------------------
def test_session_file_rewrites_outside_transitions_are_rate_limited(tmp_path):
    coord, events, writes, clock = make(tmp_path)
    coord.on_session_start()
    coord.on_trainer_status(ts("ready"), clock())  # a transition: written at once
    n = len(writes)
    for _ in range(5):  # heartbeats of the same state within the period: no rewrite
        clock.t += 0.1
        coord.on_trainer_status(ts("ready"), clock())
    assert len(writes) == n
    clock.t += 1.0
    coord.on_trainer_status(ts("ready"), clock())
    assert len(writes) == n + 1
    # a null-session heartbeat also only writes when due
    coord.on_trainer_status(ts("idle", session_id=None), clock())
    assert len(writes) == n + 1


def test_coordinator_without_side_effects_still_returns_its_events(tmp_path):
    cfg = OnlineDaggerConfig(session_name="s1")
    coord = OnlineDaggerCoordinator(cfg, session_id="s", paths=paths_for(tmp_path))
    assert coord.on_session_start() == []
    assert kinds(save(coord, 0)) == ["episode_saved"]
    assert kinds(coord.on_episode_discarded("e", 1)) == ["episode_discarded"]
    assert coord.events_published == 2 and coord.task is None


def test_transitions_are_serialised_across_threads(tmp_path):
    coord, events, writes, clock = make(tmp_path)
    coord.on_session_start()
    coord.on_trainer_status(ts("ready"), clock())
    start = threading.Barrier(3)

    def saver():
        start.wait()
        for i in range(50):
            coord.on_episode_saved(f"s{i}", i, summary(i, 1, 1), "")

    def statuser():
        start.wait()
        for i in range(50):
            coord.on_trainer_status(ts("training" if i % 2 else "ready"), clock())
            coord.on_spec_version(i)

    threads = [threading.Thread(target=saver), threading.Thread(target=statuser)]
    for t in threads:
        t.start()
    start.wait()
    for t in threads:
        t.join(5.0)
    assert coord.rollouts_saved == 50 and coord.expert_frames_session == 50
    saved = [p for k, p in events if k == "episode_saved"]
    assert [p["online_dagger"]["rollouts_saved"] for p in saved] == list(range(1, 51))
    assert writes[-1]["current"]["rollouts_saved"] == 50


def test_serial_worker_runs_in_submission_order_and_inline_once_closed():
    seen: list[int] = []
    done = threading.Event()
    w = SerialWorker()
    for i in range(20):
        w.submit(seen.append, i)
    w.submit(lambda: done.set())
    assert done.wait(5.0) and seen == list(range(20))
    w.submit(lambda: 1 / 0)  # an exception is logged, never raised, never kills the worker
    w.close()
    w.submit(seen.append, 99)  # closed: runs inline
    assert seen[-1] == 99
    w.close()  # idempotent


def test_write_session_json_atomic_leaves_no_temp_file(tmp_path):
    path = tmp_path / "x" / "session.json"
    write_session_json_atomic(path, {"b": 1, "a": [1, 2]})
    assert json.loads(path.read_text()) == {"a": [1, 2], "b": 1}
    assert path.read_text().startswith("{\n")  # indent=2, sort_keys
    assert [p.name for p in path.parent.iterdir()] == ["session.json"]
    write_session_json_atomic(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 2}
