"""``nodes/fake_policy.py``'s trainer role in-process (15-online-dagger §10): the generic
Online DAgger trainer the sim e2e spawns, driven here without dora — ``on_session`` /
``on_event`` / ``pump`` on this thread, its one worker thread doing the fake prepare /
training. Every status it emits must validate as the core ``TrainerStatusAnnounce``
(the 10 generic fields, finite floats) and echo the served ``session_id``; the training
trigger (every ``FAKE_TRAINER_EVERY`` kept rollouts or ``train_now``), the version bump,
the discarded-id rule, the ``FAKE_TRAINER_FAIL_AT`` error + recovery and the final
``idle`` are the contract the e2e relies on."""

from __future__ import annotations

import json
import time

import pytest
from apollo_mavis_v2_core.protocol.external import TrainerStatusAnnounce

from apollo_mavis_v2_runtime.dora_bridge.nodes import fake_policy as fp

SID = "11111111-2222-3333-4444-555555555555"
FIELDS = list(TrainerStatusAnnounce.model_fields)


class StubNode:
    version = 1


def announce(state: str = "running", session_id: str | None = SID, online_dagger: bool = True):
    ann = {"mavis_schema": 1, "epoch": "e", "session_id": session_id, "state": state}
    if online_dagger:
        ann["online_dagger"] = {
            "session_name": "s1",
            "session_dir": "/tmp/od/s1",
            "rollouts_dir": "/tmp/od/s1/rollouts",
        }
    return ann


def event(kind: str, payload: dict, session_id: str | None = SID) -> dict:
    return {
        "mavis_schema": 1,
        "kind": kind,
        "t_mono": 0.0,
        "wallclock_ns": 0,
        "session_id": session_id,
        "epoch": "e",
        "payload": payload,
    }


def saved(eid: str, expert: int, novice: int, n: int) -> dict:
    return event(
        "episode_saved",
        {
            "episode_index": n - 1,
            "summary": {"episode_id": eid, "n_expert_frames": expert, "n_novice_frames": novice},
            "dataset_root": "/tmp/od/s1/rollouts",
            "spool_path": f"/tmp/od/s1/rollouts/trainer_spool/ep_{eid}.parquet",
            "run_id": SID[:8],
            "online_dagger": {
                "episode_id": eid,
                "rollouts_saved": n,
                "actor_counts": {"novice": novice, "expert": expert},
                "policy_version": 1,
                "spool_path": f"/tmp/od/s1/rollouts/trainer_spool/ep_{eid}.parquet",
            },
        },
    )


class Harness:
    """Collects every status the role emits (validated as ``TrainerStatusAnnounce``)."""

    def __init__(self, role: fp.FakeTrainerRole) -> None:
        self.role = role
        self.statuses: list[dict] = []

    def pump(self) -> list[dict]:
        out = []
        for text in self.role.pump(time.monotonic()):
            doc = json.loads(text)
            assert list(doc) == FIELDS, list(doc)  # the exact generic contract, in order
            TrainerStatusAnnounce.model_validate(doc)
            out.append(doc)
        self.statuses.extend(out)
        return out

    def wait_state(self, state: str, timeout: float = 3.0, **match) -> dict:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            for doc in self.pump():
                if doc["state"] == state and all(doc[k] == v for k, v in match.items()):
                    return doc
            time.sleep(0.01)
        raise AssertionError(f"no status {state} {match}; last {self.statuses[-3:]}")

    def settle(self, seconds: float = 0.2) -> list[dict]:
        end = time.monotonic() + seconds
        out = []
        while time.monotonic() < end:
            out += self.pump()
            time.sleep(0.01)
        return out


@pytest.fixture
def role(monkeypatch):
    monkeypatch.setenv("FAKE_TRAINER_PREPARE_S", "0.05")
    monkeypatch.setenv("FAKE_TRAINER_TRAIN_S", "0.05")
    monkeypatch.setenv("FAKE_TRAINER_EVERY", "2")
    monkeypatch.setenv("FAKE_TRAINER_FAIL_AT", "2")
    node = StubNode()
    r = fp.FakeTrainerRole(node)
    yield r, node
    r.close()


def test_constants_are_the_generic_contract():
    assert fp.TRAINER_ID == "runtime-fake/online_dagger"
    assert fp.ONLINE_DAGGER_CAPABILITY == "online_dagger"
    args = fp.build_parser().parse_args(["--trainer"])
    assert args.trainer is True
    assert fp.build_parser().parse_args([]).trainer is False


def test_trainer_role_walks_the_generic_state_machine(role):
    r, node = role
    h = Harness(r)
    assert r.every == 2 and r.fail_at == 2 and r.prepare_s == 0.05
    # -- a plain announce (no online_dagger block) is not served -----------------------------
    r.on_session(announce(online_dagger=False))
    assert not r.active and h.settle(0.05) == []
    # -- the running announce: preparing at once (echoing the id), ready after PREPARE_S ------
    r.on_session(announce())
    first = h.pump()
    assert first and first[0]["state"] == "preparing" and first[0]["session_id"] == SID
    assert first[0]["progress"] == 0.0 and first[0]["policy_version"] == 1
    ready = h.wait_state("ready")
    assert ready["session_id"] == SID and ready["progress"] == 1.0
    assert ready["policy_version"] == 1 and ready["metrics"] == {}
    assert r.stats["sessions"] == 1 and r.stats["prepares"] == 1
    r.on_session(announce())  # the 1 Hz heartbeat of the served session: nothing happens
    assert h.settle(0.05) == [] and r.stats["prepares"] == 1
    # -- one kept rollout: counted, no training yet ------------------------------------------
    r.on_event(saved("e1", expert=30, novice=70, n=1))
    assert h.settle(0.1) == [] and r.stats["rollouts_counted"] == 1 and r.state == "ready"
    # -- a discarded id is remembered and never counted; another session's event is ignored --
    r.on_event(event("episode_discarded", {"episode_index": 1, "episode_id": "e2", "reason": ""}))
    r.on_event(saved("e2", expert=5, novice=5, n=2))
    r.on_event(saved("e9", expert=5, novice=5, n=2))
    r.on_event(event("episode_saved", saved("e8", 1, 1, 9)["payload"], session_id="other"))
    assert r.stats["discards"] == 1 and r.stats["ignored_events"] >= 2
    # e9 was the 2nd new rollout -> a training started (e2 never counted, "other" ignored)
    training = h.wait_state("training")
    assert training["session_id"] == SID and training["policy_version"] == 1
    assert training["metrics"]["rollouts_trained"] == 2.0
    assert training["metrics"]["n_expert_frames"] == 35.0  # 30 (e1) + 5 (e9)
    assert training["metrics"]["loss"] == 1.0 and "train" in training["detail"]
    swapped = h.wait_state("ready", policy_version=2)
    assert node.version == 2 and r.take_spec_dirty() is True and r.take_spec_dirty() is False
    assert swapped["metrics"]["rollouts_trained"] == 2.0
    assert 0.0 < swapped["metrics"]["loss"] < 1.0
    assert r.stats["trainings"] == 1 and r.stats["swaps"] == 1 and r.stats["errors"] == 0
    # -- train_now: trains on the whole buffer; the 2nd training is the configured failure ----
    r.on_event(event("train_now", {"rollouts_saved": 2, "requested_by": "operator"}))
    err = h.wait_state("error")
    assert "FAKE_TRAINER_FAIL_AT=2" in err["detail"] and err["policy_version"] == 2
    assert err["session_id"] == SID and node.version == 2  # no swap on a failed training
    assert r.stats["trainings"] == 2 and r.stats["errors"] == 1 and r.stats["train_now"] == 1
    # the 1 Hz heartbeat keeps saying error while the session is served
    r.heartbeat_s = 0.02
    beats = h.settle(0.1)
    assert beats and all(b["state"] == "error" and b["session_id"] == SID for b in beats)
    # -- the next train_now recovers: training -> ready with another version bump ------------
    r.on_event(event("train_now", {"rollouts_saved": 2, "requested_by": "operator"}))
    h.wait_state("training")
    recovered = h.wait_state("ready", policy_version=3)
    assert node.version == 3 and recovered["metrics"]["rollouts_trained"] == 2.0
    assert recovered["metrics"]["loss"] < swapped["metrics"]["loss"]  # keeps decreasing
    assert r.take_spec_dirty() is True
    # -- transient states of the served session keep the context; the end -> one final idle --
    r.on_session(announce(state="start_from"))
    assert r.active
    r.on_session(announce(state="teardown"))
    assert not r.active
    idle = h.wait_state("idle")
    assert idle["session_id"] is None and idle["policy_version"] == 3 and idle["metrics"] == {}
    assert h.settle(0.1) == []  # no heartbeat without a session
    # every status of the session echoed its id (the final idle serves nobody)
    assert all(s["session_id"] == SID for s in h.statuses if s["state"] != "idle")
    assert [s["state"] for s in h.statuses if s["state"] == "idle"] == ["idle"]
    r.on_event(saved("e10", 1, 1, 3))  # no session: ignored
    assert r.stats["ignored_events"] >= 3


def test_train_now_without_rollouts_is_ignored_and_a_late_discard_leaves_the_buffer(role):
    r, node = role
    h = Harness(r)
    r.on_session(announce())
    h.wait_state("ready")
    r.on_event(event("train_now", {"rollouts_saved": 0, "requested_by": "operator"}))
    docs = h.settle(0.1)
    assert docs and docs[-1]["state"] == "ready" and "no rollouts saved yet" in docs[-1]["detail"]
    assert r.stats["trainings"] == 0 and node.version == 1
    r.on_event(saved("e1", 3, 3, 1))
    r.on_event(event("episode_discarded", {"episode_index": 1, "episode_id": "e1", "reason": ""}))
    assert r.stats["rollouts_counted"] == 1 and not r._rollouts  # left the buffer again
    r.on_event(event("train_now", {"rollouts_saved": 1, "requested_by": "operator"}))
    assert h.settle(0.1)[-1]["state"] == "ready" and r.stats["trainings"] == 0
    r.on_session(announce(state="idle"))
    assert h.wait_state("idle")["session_id"] is None
