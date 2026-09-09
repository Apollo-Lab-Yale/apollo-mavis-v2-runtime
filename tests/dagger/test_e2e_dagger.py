"""DAgger-mode e2e over a real server + real AsyncTrainer process:
scripted run with takeovers -> dataset schema (12-dagger §4), trainer burst,
episode-boundary hot-swap, kill -9 crash isolation, teardown/finalize."""

from __future__ import annotations

import json
import os
import signal
import time

import httpx
import numpy as np
import pytest
from conftest import LiveServer, make_runtime_config
from helpers import free_port, make_net, write_checkpoint
from test_e2e_teleop import PulsingCtl, Tele

TASK = "dagger e2e"
SPEC = {
    "mode": "dagger",
    "kind": "sim",
    "arms": ["arm0"],
    "frames": {"arm0": "arm_base:arm0"},
    "sim_scene": "guardrail_env",
    "task": TASK,
    # D6 (15-online-dagger, 2026-09-08): return-to-start is ON by default for dagger too and
    # 409s without an initial-condition profile; this suite tests the policy plumbing.
    "return_to_start": False,
}


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    cfg = make_runtime_config(tmp_path_factory.mktemp("rt"), scene="guardrail_env")
    cfg.dagger.trainer.port = free_port()
    cfg.dagger.trainer.device = "cpu"
    cfg.dagger.trainer.cuda_visible_devices = None
    cfg.dagger.trainer.min_new_labels = 20
    cfg.dagger.trainer.push_period_s = 0.5
    cfg.dagger.trainer.lr = 1e-3
    srv = LiveServer(cfg)
    yield srv
    srv.stop()


@pytest.fixture(scope="module")
def api(server):
    with httpx.Client(base_url=server.http, timeout=120.0) as client:
        yield client


def wait(pred, timeout=30.0, dt=0.1, what=""):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = pred()
        if v:
            return v
        time.sleep(dt)
    raise TimeoutError(f"timed out: {what}")


def wait_idle(tele, t0):
    while tele.latest()["episode"]["state"] != "idle":
        assert time.monotonic() - t0 < 10.0


@pytest.fixture(scope="module")
def run_artifacts(server, api):
    """One scripted DAgger session; downstream tests assert on the leftovers."""
    net = make_net(0)
    net.body[-1].weight.data *= 0.01
    net.body[-1].bias.data.zero_()
    write_checkpoint(server.runtime.cfg.checkpoints_root, "seed0", 1, net=net)
    r = api.post("/api/session", json=SPEC)  # policy=None -> latest sanity_ok
    assert r.status_code == 200, r.text
    wait(lambda: api.get("/api/session").json()["state"] == "running", 20.0)
    session = server.runtime.manager.session
    loop = session.loop
    run_id = loop.run_id
    store = session.policy_session.reloader.store
    client = session.policy_session.trainer_client
    # PulsingCtl = the Cockpit's 25 Hz heartbeat: without it the deadman trips between
    # two holds and the second takeover's keys are IGNORED (AWAIT_EMPTY) - the arm then
    # stands still and the idle-frame filter (default ON) records no labels at all.
    ctl, tele = PulsingCtl(server), Tele(server)
    art = {"run_id": run_id, "repo_id": None}
    try:
        msg = tele.latest()
        assert msg["dagger"]["policy_version"] == f"{run_id}/v000000"
        assert msg["dagger"]["control_mode"] == "policy"
        wait(lambda: (m := tele.latest()["dagger"]["trainer"]) and m["state"] != "dead",
             30.0, what="trainer up")

        # -- episode 0: policy drive + TWO takeovers -> labels ------------------
        assert ctl.action("episode_new")["ok"]
        time.sleep(1.0)  # policy driving, counterfactuals flowing
        for held, dur in ((["KeyW"], 1.4), (["KeyS"], 1.1)):
            assert ctl.action("takeover_toggle")["detail"] == "takeover_transition"
            ctl.hold(held, dur)  # 0.3 s TRANSITION then HUMAN labels
            # poll: unread WS windows can leave stale frames in the buffer
            wait(lambda: (m := tele.latest()["dagger"])["control_mode"] == "human"
                 and m["engaged_arm"] == "arm0", 5.0, what="HUMAN visible")
            assert tele.latest()["dagger"]["takeover_rate_ep"] > 0.0
            assert ctl.action("takeover_toggle")["detail"] == "policy"
            time.sleep(0.5)
        assert ctl.action("episode_save")["ok"]
        wait_idle(tele, time.monotonic())

        # -- trainer: burst -> v000001, staged, NOT applied until next boundary --
        wait(lambda: store.latest() == 1, 60.0, what="v000001")
        reloader = session.policy_session.reloader
        wait(lambda: reloader.staged_version() == 1, 10.0, what="staged")
        assert tele.latest()["dagger"]["policy_version"] == f"{run_id}/v000000"
        assert tele.latest()["dagger"]["staged_version"] == f"{run_id}/v000001"

        # -- episode 1: plain policy episode; save = swap point ------------------
        assert ctl.action("episode_new")["ok"]
        time.sleep(0.8)
        assert ctl.action("episode_save")["ok"]
        wait_idle(tele, time.monotonic())
        wait(lambda: tele.latest()["dagger"]["policy_version"] == f"{run_id}/v000001",
             10.0, what="hot-swap at boundary")

        # -- episode 2: frames must record the NEW policy_version ----------------
        assert ctl.action("episode_new")["ok"]
        time.sleep(0.8)
        assert ctl.action("episode_save")["ok"]
        wait_idle(tele, time.monotonic())

        # -- crash isolation: kill -9, tick rate steady, ONE --resume restart ----
        pid1 = client.proc.pid
        ticks0, t0 = loop.tick_count, time.monotonic()
        os.kill(pid1, signal.SIGKILL)
        time.sleep(2.0)
        rate = (loop.tick_count - ticks0) / (time.monotonic() - t0)
        assert 99.0 <= rate <= 101.0, f"loop {rate:.1f} Hz during trainer death"
        wait(lambda: client.restarts == 1, 20.0, what="one auto-restart")
        assert "--resume" in client.proc.args
        wait(lambda: client.proc.pid != pid1 and client.status().state != "dead",
             30.0, what="restarted trainer alive")
        os.kill(client.proc.pid, signal.SIGKILL)  # second death -> degraded
        wait(lambda: client.degraded, 20.0, what="degraded")
        assert client.restarts == 1  # never a third restart
        wait(lambda: tele.latest()["dagger"]["trainer"]["state"] == "dead", 10.0)
        assert tele.latest()["session"]["trainer_alive"] is False
        m = tele.latest()["dagger"]
        assert m["episodes_labeled"] == 1 and 0 <= m["takeover_rate_run"] <= 1
        art["repo_id"] = session.recorder_thread.recorder.repo_id
    finally:
        ctl.close()
        tele.close()
        api.delete("/api/session")
    art["root"] = server.runtime.cfg.datasets_root / art["repo_id"]
    art["ckpt_root"] = server.runtime.cfg.checkpoints_root / run_id
    return art


def _col(table, name) -> np.ndarray:
    values = table.column(name).to_numpy(zero_copy_only=False)
    return np.stack([np.atleast_1d(np.asarray(v)) for v in values]).squeeze()


def test_dataset_schema_and_modes(run_artifacts):
    import pyarrow.parquet as pq

    root, run_id = run_artifacts["root"], run_artifacts["run_id"]
    assert f"_dagger_{run_id}" in run_artifacts["repo_id"]  # dedicated repo (§4)
    feats = json.loads((root / "manifest.json").read_text())["features"]
    assert feats["control_mode"]["info"]["labels"] == {
        "0": "policy", "1": "human", "2": "takeover_transition"}
    assert feats["policy_action"]["names"] == feats["action"]["names"]
    assert feats["policy_action"]["info"] == {"counterfactual": True}
    assert feats["policy_version"]["info"]["run_id"] == run_id

    import pyarrow as pa

    files = sorted(root.glob("episodes/*/frames.parquet"))  # one directory per episode
    assert len(files) == 3
    tables = [pq.read_table(f) for f in files]
    table = pa.concat_tables(tables)
    modes = _col(table, "control_mode").astype(int)
    ep_idx = np.concatenate([np.full(t.num_rows, i) for i, t in enumerate(tables)])
    m0 = modes[ep_idx == 0]
    assert set(m0.tolist()) == {0, 1, 2}  # all three modes in episode 0
    interv = _col(table, "intervention").astype(bool)
    assert (interv == (modes != 0)).all()  # transition counts as intervention
    src = _col(table, "action_source").astype(int)
    assert ((src == 3) == (modes != 0)).all() and set(src.tolist()) <= {0, 3}
    # transition run lengths ~ 0.3 s * 25 fps per takeover
    trans = int((m0 == 2).sum())
    assert 2 * 6 <= trans <= 2 * 10
    pa = _col(table, "policy_action")
    assert pa.shape[1] == 8
    assert np.isfinite(pa[modes == 1]).all()  # queried during HUMAN too (§4)
    ver = _col(table, "policy_version").astype(int)
    assert set(ver[ep_idx == 0].tolist()) == {0}
    assert set(ver[ep_idx == 2].tolist()) == {1}  # post-swap episode records v1
    # human frames executed real motion (the held-key drive left ee deltas)
    action = _col(table, "action")
    assert np.abs(action[modes == 1, 0]).max() > 1e-4


def test_sidecars_and_spool(run_artifacts):
    root = run_artifacts["root"]
    dirs = sorted(p for p in (root / "episodes").iterdir() if not p.name.startswith("."))
    eps = [d / "episode.json" for d in dirs]
    assert len(eps) == 3
    ep0 = json.loads(eps[0].read_text())
    modes = [e["mode"] for e in ep0["gate_events"]]
    assert modes == ["takeover_transition", "human", "policy"] * 2
    assert ep0["episode_summary"]["takeover_segments"] == 2
    assert ep0["episode_summary"]["n_label_frames"] >= 20
    assert len(ep0["episode_summary"]["segment_doubts"]) == 2
    ep1 = json.loads(eps[1].read_text())
    assert ep1["gate_events"] == [] and ep1["episode_summary"]["n_label_frames"] == 0
    spools = sorted((root / "trainer_spool").glob("ep_*.parquet"))
    # keyed by the episode id (12-dagger §7, 2026-09-07), one per saved episode dir
    assert [p.name for p in spools] == [f"ep_{d.name}.parquet" for d in dirs]
    assert json.loads(eps[0].read_text())["episode_id"] == dirs[0].name


def test_checkpoint_run_layout(run_artifacts):
    ck = run_artifacts["ckpt_root"]
    assert (ck / "v000000" / "state_dict.pt").exists()  # session seed = rollback target
    assert (ck / "v000001" / "state_dict.pt").exists()
    assert (ck / "v000001" / "trainer_state.pt").exists()
    manifest = json.loads((ck / "v000001" / "manifest.json").read_text())
    assert manifest["sanity_ok"] is True and manifest["parent_version"] == 0
    assert (ck / "LATEST").read_text().strip() == "v000001"
    assert (ck / "LAST_KNOWN_GOOD").exists()


def test_dataset_exports_and_reloads(run_artifacts):
    """The DAgger dataset is the same episode-directory store; its LeRobot v3 export
    (built on demand, 12-dagger §4 / 10-frames §11.8) reads back with lerobot."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from apollo_mavis_v2_runtime.recorder.export_lerobot import export_lerobot_v3

    root = run_artifacts["root"]
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["episodes"] == 3 and not (root / "recorder_state.json").exists()
    assert not list((root / "episodes").glob(".tmp-*"))
    result = export_lerobot_v3(root, run_artifacts["repo_id"], validate=True)
    assert result.episodes == 3
    ds = LeRobotDataset(run_artifacts["repo_id"], root=root / "exports" / "lerobot_v3")
    assert ds.num_episodes == 3
    row = ds[0]
    assert row["task"] == TASK
