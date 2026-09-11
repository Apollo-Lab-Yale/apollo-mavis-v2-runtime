"""``tools.backfill_abs_ee`` against episodes the REAL recorder wrote (10-frames §6 /
§11, 2026-09-11): the backfilled ``action.abs_ee`` equals the live recorder's column
bit-for-bit on an exactly-representable fixture (the recorder thread and the backfill
share ONE FakeKin), ``observation.state`` ee.* is replaced by FK, backups land outside
the episode directory, the manifest is patched LAST (never after a failure),
idempotence, ``--force``, ``--dry-run`` writes nothing, the CLI, mixed-rail offsets,
and the real twin (``single_rail``) round trip within float tolerance."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa  # noqa: TID251 - reads / rewrites the recorder's frames.parquet fixtures
import pyarrow.parquet as pq  # noqa: TID251
import pytest
from apollo_mavis_v2_core import CollisionReport, se3
from test_recorder_thread import FakeCam, FakeKin, make_state

from apollo_mavis_v2_runtime.bus import RuntimeBus
from apollo_mavis_v2_runtime.config import RecorderConfig
from apollo_mavis_v2_runtime.control.snapshot import StateSnapshot
from apollo_mavis_v2_runtime.recorder import backfill_abs_ee as bf
from apollo_mavis_v2_runtime.recorder.episode_recorder import (
    EpisodeDirRecorder,
    dataset_incompatibility,
)
from apollo_mavis_v2_runtime.recorder.features import ABS_EE_KEY, ArmMeta, build_features
from apollo_mavis_v2_runtime.recorder.frames import RecordingFrameConverter
from apollo_mavis_v2_runtime.recorder.manifest import (
    read_manifest,
    write_json_atomic,
    write_manifest,
)
from apollo_mavis_v2_runtime.recorder.sidecars import SidecarWriter
from apollo_mavis_v2_runtime.recorder.thread import RecorderThread

REPO = "bc_demo/backfill_test"
CAM = "cam0"
W, H = 64, 48
FRAMES = {"arm0": "arm_base:arm0"}
ARMS = [ArmMeta("arm0", True)]
HW_ROBOT = "xarm7_1arm_rail"  # hardware-kind: the SDK ee.* gets replaced by FK
FEATURES = build_features(ARMS, FRAMES, {CAM: (W, H)})
# Every value a dyadic rational: FakeKin's FK, the recorder's float64 -> float32 casts and
# the backfill's cumsum are then all exact, so the two columns must agree bit for bit.
Q0 = np.array([0.125, 0.25, 0.375, 0.0, 0.0, 0.0, 0.0, 0.125])
STEP = np.array([2.0**-9, -(2.0**-10), 2.0**-11, 0.0, 0.0, 0.0, 0.0, 2.0**-9])
ROT_STEP = np.array([0.0, 0.0, 0.0, 0.01, -0.02, 0.03, 0.0, 0.0])


def snap(tick: int, q: np.ndarray, t: float, grip: float) -> StateSnapshot:
    ns = int(round(t * 1e9))
    return StateSnapshot(
        t_mono=t, wallclock_ns=ns, tick=tick,
        arms={"arm0": make_state(q, ns)},  # q_meas == q_cmd: the arm tracks perfectly
        q_cmd={"arm0": np.asarray(q, dtype=np.float64)},
        active_arm="arm0", gate=CollisionReport.ok(), gripper_frac={"arm0": grip},
    )


def record_dataset(
    root: Path, kin, *, episodes: int = 2, n: int = 6, rotate: bool = False,
    robot_type: str = HW_ROBOT, scene: str = "fake_scene",
) -> list[str]:
    """Drive the real EpisodeDirRecorder through a RecorderThread with ``kin``."""
    rec = EpisodeDirRecorder(RecorderConfig(vcodec="libsvtav1"), FEATURES, root, REPO,
                             robot_type, "backfill task")
    bus, cam = RuntimeBus(), FakeCam()
    thread = RecorderThread(
        rec, bus, {CAM: cam}, ARMS, RecordingFrameConverter(FRAMES), kin, fps=25,
        episode_meta_base={"session_id": "s1", "frames": FRAMES},
    )
    ids: list[str] = []
    t, tick = 1.0, 0
    for e in range(episodes):
        assert thread.request("new")[0]
        q = Q0.copy()
        for i in range(n):
            tick += 1
            t += 0.04
            cam.refresh(t)
            bus.snapshot.put(snap(tick, q, t, 0.5 if i % 2 else 0.75))
            thread.run_iteration(t)
            q = q + STEP * (1 + e) + (ROT_STEP if rotate else 0.0)
        assert thread.request("save")[0]
        t += 0.04
        thread.run_iteration(t)
        assert thread.status().state == "idle", thread.status().detail
        ids.append(thread.last_saved_id)
    thread.stop()
    side = SidecarWriter(root)
    side.write_session("s1", "collect", {"task": "t", "sim_scene": scene},
                       {"kind": "hardware", "arm_ids": ["arm0"]}, {})
    return ids


def strip_abs_ee(root: Path, ids: list[str]) -> dict[str, np.ndarray]:
    """Turn the dataset into one recorded before 2026-09-11; returns the live columns."""
    live: dict[str, np.ndarray] = {}
    for eid in ids:
        d = root / "episodes" / eid
        table = pq.read_table(d / "frames.parquet")
        live[eid] = np.asarray(table.column(ABS_EE_KEY).to_pylist(), dtype=np.float32)
        stripped = table.drop_columns([ABS_EE_KEY])
        pq.write_table(stripped, d / "frames.parquet", compression="snappy",
                       row_group_size=table.num_rows)
        ep = json.loads((d / "episode.json").read_text())
        del ep["stats"][ABS_EE_KEY]
        write_json_atomic(d / "episode.json", ep)
    m = read_manifest(root)
    del m["features"][ABS_EE_KEY]
    write_manifest(root, m)
    return live


def read_matrix(path: Path, key: str) -> np.ndarray:
    return np.asarray(pq.read_table(path).column(key).to_pylist(), dtype=np.float32)


@pytest.fixture(scope="module")
def kin():
    return FakeKin()


@pytest.fixture
def stripped(tmp_path, kin):
    root = tmp_path / REPO
    ids = record_dataset(root, kin)
    live = strip_abs_ee(root, ids)
    for eid in ids:
        names = pq.ParquetFile(root / "episodes" / eid / "frames.parquet").schema_arrow.names
        assert ABS_EE_KEY not in names
    assert ABS_EE_KEY not in read_manifest(root)["features"]
    return root, ids, live


def test_live_recorder_writes_the_column_and_the_backfill_reproduces_it_bitwise(stripped, kin):
    root, ids, live = stripped
    result = bf.backfill_dataset(root, kin_factory=lambda scene_id: kin)
    assert result.ok and result.manifest_patched and not result.dry_run
    assert [r.status for r in result.reports] == ["backfilled", "backfilled"]
    for rep, eid in zip(result.reports, ids, strict=True):
        assert rep.episode_id == eid and rep.rows == 5 and rep.scene == "fake_scene"
        assert rep.flags == [] and rep.error is None
        d = root / "episodes" / eid
        # the episode directory's file set is untouched (no .bak, no temp file)
        assert {p.name for p in d.iterdir()} == {"episode.json", "frames.parquet", "video"}
        pf = pq.ParquetFile(d / "frames.parquet")
        assert pf.metadata.num_row_groups == 1 and pf.metadata.num_rows == 5
        assert pf.schema_arrow.names == [
            "action", ABS_EE_KEY, "observation.state", "intervention", "action_source",
            "wallclock_ns", "timestamp", "frame_index", "task",
        ]
        assert pf.schema_arrow.field(ABS_EE_KEY).type == pa.list_(pa.float32(), 11)
        assert pf.schema_arrow.field("observation.state").type == pa.list_(pa.float32(), 16)
        got = read_matrix(d / "frames.parquet", ABS_EE_KEY)
        assert got.dtype == np.float32 and np.array_equal(got, live[eid])  # bit for bit
        # observation.state ee.* := FK(q_meas) (the thread wrote the fake driver's fixed pose)
        state = read_matrix(d / "frames.parquet", "observation.state")
        assert np.array_equal(state[:, 9:12], state[:, :3])  # FakeKin: tcp = q[:3]
        assert np.array_equal(state[:, 12:16], np.tile([1.0, 0.0, 0.0, 0.0], (5, 1)))
        # the label is the commanded pose at k+1 = the measured pose of row k+1 here
        assert np.array_equal(got[:-1, :3], state[1:, 9:12])
        assert np.array_equal(got[:, 9], read_matrix(d / "frames.parquet", "action")[:, 6])
        assert np.array_equal(got[:-1, 10], state[1:, 8])  # rail at k+1
        ep = json.loads((d / "episode.json").read_text())
        marker = ep["backfill"]["abs_ee"]
        assert marker["version"] == 1 and marker["state_ee_recomputed"] is True
        assert marker["tool"] == "apollo_mavis_v2_runtime.tools.backfill_abs_ee"
        assert marker["fk_scene"] == "fake_scene" and marker["at"].endswith("Z")
        assert marker["terminal_residual_m"]["arm0"] == pytest.approx(
            rep.terminal_residual_m["arm0"]
        )
        assert 0 < marker["terminal_residual_m"]["arm0"] < 0.02
        assert set(ep["stats"]) >= {ABS_EE_KEY, "observation.state", "action"}
        assert np.asarray(ep["stats"][ABS_EE_KEY]["mean"]).shape == (11,)
        assert np.allclose(ep["stats"][ABS_EE_KEY]["mean"], got.mean(axis=0), atol=1e-6)
        assert np.allclose(ep["stats"]["observation.state"]["min"], state.min(axis=0), atol=1e-6)
    # backups OUTSIDE episodes/: the pre-backfill files + the pre-patch manifest
    bdir = result.backup_dir
    assert bdir is not None and bdir.parent == root / "backups"
    assert {p.name for p in bdir.iterdir()} == {*ids, "manifest.json"}
    for eid in ids:
        names = pq.ParquetFile(bdir / eid / "frames.parquet").schema_arrow.names
        assert ABS_EE_KEY not in names
        assert ABS_EE_KEY not in json.loads((bdir / eid / "episode.json").read_text())["stats"]
    assert ABS_EE_KEY not in json.loads((bdir / "manifest.json").read_text())["features"]
    # the manifest declares the column exactly as build_features does -> resume is allowed
    m = read_manifest(root)
    assert set(m["features"]) == set(FEATURES)  # (written sort_keys: no order to pin)
    expected = dict(FEATURES[ABS_EE_KEY])
    expected["shape"] = list(expected["shape"])
    assert m["features"][ABS_EE_KEY] == expected
    assert dataset_incompatibility(m, FEATURES, 25, HW_ROBOT) is None
    assert m["last_export"] is None
    # the report reads
    text = bf.format_report(result)
    assert "2 backfilled" in text and "action.abs_ee feature added" in text and str(bdir) in text


def test_second_run_is_a_no_op_and_force_redoes_it(stripped, kin):
    root, ids, live = stripped
    bf.backfill_dataset(root, kin_factory=lambda s: kin)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    again = bf.backfill_dataset(root, kin_factory=lambda s: kin)
    assert [r.status for r in again.reports] == ["skipped", "skipped"]
    assert again.ok and not again.manifest_patched and again.backup_dir is None
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before
    assert "already declares" in bf.format_report(again)
    forced = bf.backfill_dataset(root, force=True, kin_factory=lambda s: kin)
    assert [r.status for r in forced.reports] == ["backfilled", "backfilled"]
    assert forced.backup_dir is not None and forced.backup_dir != again.backup_dir
    for eid in ids:
        assert np.array_equal(read_matrix(root / "episodes" / eid / "frames.parquet", ABS_EE_KEY),
                              live[eid])
        pf = pq.ParquetFile(root / "episodes" / eid / "frames.parquet")
        assert pf.schema_arrow.names.count(ABS_EE_KEY) == 1  # replaced in place, not duplicated


def test_manifest_is_patched_only_after_every_episode_succeeded(stripped, kin, monkeypatch):
    root, ids, live = stripped
    real = bf._write_parquet_atomic

    def fail_second(path, table):
        if ids[1] in str(path):
            raise OSError("disk full (test)")
        return real(path, table)

    monkeypatch.setattr(bf, "_write_parquet_atomic", fail_second)
    result = bf.backfill_dataset(root, kin_factory=lambda s: kin)
    assert not result.ok and not result.manifest_patched
    assert [r.status for r in result.reports] == ["backfilled", "failed"]
    assert "OSError: disk full" in result.reports[1].error
    assert ABS_EE_KEY not in read_manifest(root)["features"]  # never half-declared
    assert "NOT patched" in bf.format_report(result)
    # the failed episode is intact (its backup was taken, its files untouched)
    d1 = root / "episodes" / ids[1]
    assert ABS_EE_KEY not in pq.ParquetFile(d1 / "frames.parquet").schema_arrow.names
    assert "backfill" not in json.loads((d1 / "episode.json").read_text())
    assert (result.backup_dir / ids[1] / "frames.parquet").exists()
    assert not (result.backup_dir / "manifest.json").exists()
    monkeypatch.undo()
    rerun = bf.backfill_dataset(root, kin_factory=lambda s: kin)
    assert rerun.ok and rerun.manifest_patched
    assert [r.status for r in rerun.reports] == ["skipped", "backfilled"]
    assert np.array_equal(read_matrix(d1 / "frames.parquet", ABS_EE_KEY), live[ids[1]])
    assert ABS_EE_KEY in read_manifest(root)["features"]


def test_dry_run_computes_but_writes_nothing(stripped, kin):
    root, ids, live = stripped
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = bf.backfill_dataset(root, dry_run=True, kin_factory=lambda s: kin)
    assert result.ok and result.dry_run and not result.manifest_patched
    assert [r.status for r in result.reports] == ["would-backfill"] * 2
    assert result.reports[0].terminal_residual_m["arm0"] > 0  # the FK ran
    assert result.backup_dir is None and not (root / "backups").exists()
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before
    assert "dry run" in bf.format_report(result)


def test_cli_and_refusals(stripped, kin, capsys, tmp_path):
    root, ids, live = stripped
    # the CLI on a dataset already backfilled: every episode skipped, no FK needed
    bf.backfill_dataset(root, kin_factory=lambda s: kin)
    assert bf.main([str(root)]) == 0
    out = capsys.readouterr().out
    assert "2 skipped" in out and ids[0] in out
    # an open episode (.tmp-*) -> refused before anything is touched
    (root / "episodes" / ".tmp-live").mkdir()
    assert bf.main([str(root), "--dry-run"]) == 2
    assert "being recorded" in capsys.readouterr().err
    shutil.rmtree(root / "episodes" / ".tmp-live")
    # not a dataset
    assert bf.main([str(tmp_path / "nowhere")]) == 2
    with pytest.raises(bf.BackfillError, match="manifest.json"):
        bf.backfill_dataset(tmp_path / "nowhere")
    # a failing episode -> exit 1 and the manifest stays put
    strip_abs_ee(root, ids)
    (root / "episodes" / ids[0] / "frames.parquet").write_bytes(b"not parquet")
    assert bf.main([str(root)]) == 1
    assert ABS_EE_KEY not in read_manifest(root)["features"]


def test_rotations_integrate_to_the_commanded_orientation(tmp_path, kin):
    """With non-zero rotation deltas the integrated quaternion (left composition) decodes
    to the commanded rotation: not bit-exact (trig), but within float32 tolerance."""
    root = tmp_path / REPO
    ids = record_dataset(root, kin, episodes=1, n=8, rotate=True)
    live = strip_abs_ee(root, ids)
    result = bf.backfill_dataset(root, kin_factory=lambda s: kin)
    assert result.ok
    got = read_matrix(root / "episodes" / ids[0] / "frames.parquet", ABS_EE_KEY)
    assert np.allclose(got, live[ids[0]], atol=2e-6)
    state = read_matrix(root / "episodes" / ids[0] / "frames.parquet", "observation.state")
    for k in range(got.shape[0] - 1):
        r_cmd = se3.rot6d_to_mat(got[k, 3:9].astype(np.float64))
        r_fk = se3.quat_to_mat(state[k + 1, 12:16].astype(np.float64))  # meas at k+1 = cmd
        assert np.allclose(r_cmd, r_fk, atol=1e-5)
        assert np.allclose(r_cmd @ r_cmd.T, np.eye(3), atol=1e-6)  # Gram-Schmidt decode


def test_compute_columns_mixed_rail_offsets(kin):
    """Two arms, one without a rail: block widths 16 + 15 / 8 + 7 / 11 + 10 and the
    per-arm rules (gripper verbatim, rail cumsum from the first measured position)."""
    arms = [ArmMeta("a", True), ArmMeta("b", False)]
    conv = RecordingFrameConverter({"a": "arm_base:a", "b": "arm_base:b"})
    n = 4
    state = np.zeros((n, 31), dtype=np.float32)
    action = np.zeros((n, 15), dtype=np.float32)
    qa = np.array([0.1, 0.2, 0.3, 0, 0, 0, 0, 0.4])
    qb = np.array([0.5, 0.6, 0.7, 0, 0, 0, 0])
    for k in range(n):
        state[k, :7] = qa[:7]
        state[k, :3] += k * 0.01  # translation only (FakeKin: q[3:6] is the rotation)
        state[k, 7] = 0.9
        state[k, 8] = qa[7] + k * 0.02
        state[k, 16:23] = qb
        state[k, 16:19] += k * 0.03
        state[k, 23] = 0.1
        action[k, :3] = 0.01
        action[k, 6] = 0.33
        action[k, 7] = 0.02
        action[k, 8:11] = 0.03
        action[k, 14] = 0.66
    cols = bf.compute_columns(state, action, arms, conv, kin)
    assert cols.state.shape == (n, 31) and cols.abs_ee.shape == (n, 21)
    assert cols.state.dtype == np.float32 and cols.abs_ee.dtype == np.float32
    # arm a: ee at 9..15, rail-aware FK (FakeKin ignores the rail for tcp_base)
    assert np.allclose(cols.state[:, 9:12], state[:, :3]) and np.allclose(cols.state[:, 12], 1.0)
    # arm b: ee at 16 + 8 = 24..30
    assert np.allclose(cols.state[:, 24:27], state[:, 16:19])
    for k in range(n):
        assert np.allclose(cols.abs_ee[k, :3], qa[:3] + (k + 1) * 0.01, atol=1e-6)
        assert np.allclose(cols.abs_ee[k, 3:9], [1, 0, 0, 0, 1, 0])
        assert cols.abs_ee[k, 9] == np.float32(0.33)
        assert cols.abs_ee[k, 10] == pytest.approx(qa[7] + (k + 1) * 0.02, abs=1e-6)
        assert np.allclose(cols.abs_ee[k, 11:14], qb[:3] + (k + 1) * 0.03, atol=1e-6)
        assert cols.abs_ee[k, 20] == np.float32(0.66)
    assert set(cols.terminal_residual_m) == {"a", "b"}
    assert cols.terminal_residual_m["a"] == pytest.approx(np.sqrt(3) * 0.01, abs=1e-6)
    assert cols.state_ee_max_change_m["b"] > 0  # the zero ee.* got replaced


def test_dataset_layout_refuses_a_foreign_layout():
    m = {"features": build_features(ARMS, FRAMES, {}), "robot_type": HW_ROBOT}
    layout = bf.dataset_layout(m)
    assert layout.arms == ARMS and layout.frames == FRAMES and not layout.sim
    assert layout.abs_feature["names"] == FEATURES[ABS_EE_KEY]["names"]
    bad = json.loads(json.dumps(m))
    bad["features"]["action"]["names"][0] = "arm1_ee.dx"
    with pytest.raises(bf.BackfillError, match="do not match"):
        bf.dataset_layout(bad)
    with pytest.raises(bf.BackfillError, match="delta_ee"):
        bf.dataset_layout({"features": {"action": {"info": {"action_space": "joint"}},
                                        "observation.state": {"names": ["x"]}}})


def test_real_twin_round_trip(tmp_path):
    """The default kinematics path: record with ``RecorderKinematics(single_rail)``, strip,
    backfill through ``default_kin_factory`` (scene from the session sidecar) - float32
    rounding of the stored deltas keeps this within tolerance, not bit-exact."""
    pytest.importorskip("apollo_mavis_v2_sim")
    real_kin = bf.default_kin_factory("single_rail")
    root = tmp_path / REPO
    ids = record_dataset(root, real_kin, episodes=1, n=6, rotate=True, scene="single_rail")
    live = strip_abs_ee(root, ids)
    result = bf.backfill_dataset(root, scene="nonexistent_scene")  # the sidecar's scene wins
    assert result.ok and result.reports[0].scene == "single_rail" and result.reports[0].flags == []
    path = root / "episodes" / ids[0] / "frames.parquet"
    got = read_matrix(path, ABS_EE_KEY)
    assert np.allclose(got, live[ids[0]], atol=1e-5)
    # ee.* := FK(q_meas) at link_tcp (the fixture's fake driver pose is gone)
    state = read_matrix(path, "observation.state")
    for k in range(state.shape[0]):
        q = np.concatenate([state[k, :7].astype(np.float64), [float(state[k, 8])]])
        tcp = real_kin.tcp_base("arm0", q)
        assert np.allclose(state[k, 9:12], tcp.position, atol=1e-6)
        assert np.allclose(state[k, 12:16], se3.quat_normalize(tcp.orientation), atol=1e-6)
    # a fresh recording already stores FK(q_meas): the recompute is a no-op (the alignment
    # guarantee between the live recorder and the backfill)
    assert result.reports[0].state_ee_max_change_m["arm0"] < 1e-6
    # a hardware episode recorded BEFORE 2026-09-11 stored the SDK flange pose (0.172 m short
    # along tool z): corrupt ee.* the same way and check the backfill REPLACES it
    table = pq.read_table(path)
    st = np.array(table.column("observation.state").to_pylist(), dtype=np.float32)
    st[:, 11] -= 0.172
    cols = {name: table.column(name) for name in table.schema.names}
    cols["observation.state"] = pa.FixedSizeListArray.from_arrays(
        pa.array(st.reshape(-1), type=pa.float32()), st.shape[1]
    )
    pq.write_table(pa.table(cols), path)
    result2 = bf.backfill_dataset(root, force=True)
    assert result2.ok
    assert result2.reports[0].state_ee_max_change_m["arm0"] == pytest.approx(0.172, abs=1e-4)
    fixed = read_matrix(path, "observation.state")
    assert np.allclose(fixed[:, 9:16], state[:, 9:16], atol=1e-6)
