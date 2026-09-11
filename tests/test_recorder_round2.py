"""Round-2 review fixes on the recorder / store / export (2026-09-07):
1. manifest counters are REBUILT from episodes/ at save (a mid-session delete survives);
2. unique temp names + one manifest lock for concurrent writers;
3. a failure AFTER publication does not leave the recorder open; the audio block is
   re-entrant; 4. delete while exporting -> 409, derived staleness, task-derived repo
   checks; 5. a bad manifest never breaks the listing; 6. an incomplete episode
   directory is skipped by the export, not fatal; 7. the export validates the staging
   tree before it replaces the previous export; leftovers are swept; 9. save with an
   audio sink writes audio.wav inside the directory."""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from test_audio_sink import StubReader

from apollo_mavis_v2_runtime.config import RecorderConfig
from apollo_mavis_v2_runtime.recorder import export_lerobot as ex
from apollo_mavis_v2_runtime.recorder.datasets import DatasetError, DatasetStore
from apollo_mavis_v2_runtime.recorder.episode_recorder import EpisodeDirRecorder
from apollo_mavis_v2_runtime.recorder.features import ArmMeta, build_features
from apollo_mavis_v2_runtime.recorder.manifest import (
    MANIFEST_LOCK,
    read_manifest,
    write_json_atomic,
    write_manifest,
)

REPO = "apollo/round2"
CAM = "cam0"
W, H = 64, 48
FEATS = build_features([ArmMeta("arm0", True)], {"arm0": "arm_base:arm0"}, {CAM: (W, H)})
SOFT = RecorderConfig(vcodec="libsvtav1")


def frame(i: int) -> dict:
    return {
        "action": np.full(8, 0.001 * i, dtype=np.float32),
        "action.abs_ee": np.full(11, 0.002 * i, dtype=np.float32),
        "observation.state": np.full(16, 0.01 * i, dtype=np.float32),
        f"observation.images.{CAM}": np.full((H, W, 3), i % 255, dtype=np.uint8),
        "intervention": np.array([False]),
        "action_source": np.array([1], dtype=np.int8),
        "wallclock_ns": np.array([1_700_000_000_000_000_000 + i * 40_000_000], dtype=np.int64),
    }


def make_recorder(root: Path) -> EpisodeDirRecorder:
    return EpisodeDirRecorder(SOFT, FEATS, root, REPO, "xarm7_1arm_rail_mujoco", "t")


def record(rec: EpisodeDirRecorder, n: int, audio=None, prepare: bool = True) -> str:
    rec.start({})
    if prepare:
        rec.prepare()
    for i in range(n):
        rec.add_frame(frame(i))
    return rec.save({"session_id": "s", "frames_dropped": 0}, audio)[1]


# -- fix 1: counters rebuilt from the directories -------------------------------------------------
def test_save_rebuilds_counters_after_a_mid_session_delete(tmp_path):
    root = tmp_path / REPO
    rec = make_recorder(root)
    ids = [record(rec, 4) for _ in range(3)]
    time.sleep(0.002)
    store = DatasetStore(tmp_path)
    store.delete_episode(REPO, ids[1])  # REST delete while the session records
    assert read_manifest(root)["episodes"] == 2
    ids.append(record(rec, 6))
    m = read_manifest(root)
    assert m["episodes"] == 3 and m["frames"] == 4 + 4 + 6  # rebuilt, not 4/18
    assert rec.episodes_saved == 3 and rec.total_frames == 14
    assert sorted(p.name for p in (root / "episodes").iterdir()) == sorted(ids[:1] + ids[2:])
    rec.finalize()


# -- fix 2: unique temp names, one lock -----------------------------------------------------------
def test_concurrent_manifest_writers_never_collide(tmp_path):
    root = tmp_path / REPO
    rec = make_recorder(root)
    record(rec, 3)
    errors: list[BaseException] = []

    def hammer(k: int) -> None:
        try:
            for i in range(60):
                with MANIFEST_LOCK:
                    m = read_manifest(root)
                    m[f"probe_{k}"] = i
                    write_manifest(root, m)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=hammer, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    m = read_manifest(root)
    assert m is not None and all(m[f"probe_{k}"] == 59 for k in range(4))
    assert not list(root.glob("manifest.json.*.tmp"))  # every temp file was renamed away
    rec.finalize()


def test_write_json_atomic_uses_unique_temp_names(tmp_path, monkeypatch):
    target = tmp_path / "x.json"
    seen: list[str] = []
    real = os.replace

    def spy(src, dst):
        seen.append(Path(src).name)
        return real(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    write_json_atomic(target, {"a": 1})
    write_json_atomic(target, {"a": 2})
    assert len(set(seen)) == 2 and all(n.startswith("x.json.") and n.endswith(".tmp") for n in seen)
    assert json.loads(target.read_text()) == {"a": 2}


# -- fix 3: post-publication failures, audio re-entrancy ------------------------------------------
def test_failure_after_publication_does_not_leave_the_recorder_open(tmp_path, monkeypatch):
    root = tmp_path / REPO
    rec = make_recorder(root)
    rec.start({})
    for i in range(5):
        rec.add_frame(frame(i))
    calls = {"n": 0}
    real = ex.write_manifest  # same function object as episode_recorder's import

    from apollo_mavis_v2_runtime.recorder import episode_recorder as er

    def boom(*a, **kw):
        calls["n"] += 1
        raise OSError("disk full right after the rename")

    monkeypatch.setattr(er, "write_manifest", boom)
    ordinal, eid = rec.save({"session_id": "s"}, None)  # published; the manifest write failed
    monkeypatch.setattr(er, "write_manifest", real)
    assert calls["n"] == 1 and (root / "episodes" / eid / "episode.json").exists()
    assert not rec.recording and rec.episode_id is None  # NOT left open on a vanished tmp
    assert ordinal == 0
    # the next open / save rebuilds the counters
    record(rec, 2)
    assert read_manifest(root)["episodes"] == 2
    rec.finalize()


def test_retry_after_a_parquet_failure_keeps_the_audio_block(tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    root = tmp_path / REPO
    rec = make_recorder(root)
    reader = StubReader()
    from apollo_mavis_v2_runtime.recorder.audio import EpisodeAudioSink

    sink = EpisodeAudioSink(reader)
    rec.start({})
    sink.begin()
    for i in range(4):
        reader.push(np.full(1920, 0.1, np.float32), 1.0 + 0.04 * i)
        rec.add_frame(frame(i))
    n = {"c": 0}
    real = pq.write_table

    def flaky(*a, **kw):
        n["c"] += 1
        if n["c"] == 1:
            raise OSError("hiccup")
        return real(*a, **kw)

    monkeypatch.setattr(pq, "write_table", flaky)
    with pytest.raises(OSError):
        rec.save({"session_id": "s"}, sink)
    # audio.finish() ran once and drained the sink; the retry must reuse its block
    _, eid = rec.save({"session_id": "s"}, sink)
    ep = json.loads((root / "episodes" / eid / "episode.json").read_text())
    assert ep["audio"] is not None and ep["audio"]["samples"] == 4 * 1920
    assert (root / "episodes" / eid / "audio.wav").exists()
    rec.finalize()


# -- fix 9(i): save(audio=StubReader sink) --------------------------------------------------------
def test_save_with_an_audio_sink_writes_audio_wav_inside_the_directory(tmp_path):
    root = tmp_path / REPO
    rec = make_recorder(root)
    reader = StubReader()
    from apollo_mavis_v2_runtime.recorder.audio import EpisodeAudioSink

    sink = EpisodeAudioSink(reader)
    rec.start({})
    rec.prepare()
    sink.begin()
    for i in range(6):
        reader.push(np.full(1920, 0.2, np.float32), 1.0 + 0.04 * i)
        rec.add_frame(frame(i))
    _, eid = rec.save({"session_id": "s"}, sink)
    d = root / "episodes" / eid
    ep = json.loads((d / "episode.json").read_text())
    assert ep["audio"]["path"] == "audio.wav" and (d / "audio.wav").stat().st_size > 44
    assert ep["audio"]["duration_s"] == pytest.approx(6 * 1920 / 48000)
    assert DatasetStore(tmp_path).episodes(REPO)[0].audio is True
    rec.finalize()


# -- fix 4: exporting 409 / derived stale ---------------------------------------------------------
def test_delete_episode_is_refused_while_the_dataset_is_being_exported(tmp_path, monkeypatch):
    root = tmp_path / REPO
    rec = make_recorder(root)
    ids = [record(rec, 4) for _ in range(2)]
    rec.finalize()
    store = DatasetStore(tmp_path)
    monkeypatch.setattr(type(store), "exporting", property(lambda self: REPO))
    with pytest.raises(DatasetError, match="being exported"):
        store.delete_episode(REPO, ids[0])
    with pytest.raises(DatasetError, match="being exported"):
        store.delete_dataset(REPO)
    assert (root / "episodes" / ids[0]).exists()


def test_export_staleness_is_derived_from_the_ids_on_disk(tmp_path):
    root = tmp_path / REPO
    rec = make_recorder(root)
    ids = [record(rec, 4) for _ in range(2)]
    result = ex.export_lerobot_v3(root, REPO, validate=False)
    assert result.episode_ids == ids
    assert read_manifest(root)["last_export"]["stale"] is False
    # a save that lands DURING the job: recorded stale straight away
    real_record = ex._record_export

    def record_after_a_save(root_, out, result_, error):
        record(rec, 3)  # a new episode appeared before last_export was written
        return real_record(root_, out, result_, error)

    ex._record_export = record_after_a_save
    try:
        ex.export_lerobot_v3(root, REPO, validate=False)
    finally:
        ex._record_export = real_record
    assert read_manifest(root)["last_export"]["stale"] is True
    rec.finalize()


# -- fix 5: one bad manifest does not 500 the listing ---------------------------------------------
def test_listing_skips_a_corrupt_or_unsupported_manifest(tmp_path):
    root = tmp_path / REPO
    rec = make_recorder(root)
    record(rec, 2)
    rec.finalize()
    bad = tmp_path / "apollo" / "future"
    (bad / "episodes").mkdir(parents=True)
    (bad / "manifest.json").write_text(
        json.dumps({"apollo_dataset_layout": 99, "repo_id": "apollo/future"})
    )
    corrupt = tmp_path / "apollo" / "corrupt"
    corrupt.mkdir(parents=True)
    (corrupt / "manifest.json").write_text("{ not json")
    store = DatasetStore(tmp_path)
    assert [d.repo_id for d in store.list()] == [REPO]
    with pytest.raises(ValueError):
        store.describe("apollo/future")


# -- fix 6: incomplete episode directories are skipped by the export ------------------------------
def test_export_skips_incomplete_directories_and_reports_them(tmp_path):
    root = tmp_path / REPO
    rec = make_recorder(root)
    ids = [record(rec, 4) for _ in range(3)]
    rec.finalize()
    (root / "episodes" / ids[0] / "frames.parquet").unlink()  # a directory holding only sidecars
    (root / "episodes" / ids[1] / "video" / f"{CAM}.mp4").unlink()
    progress = ex.ExportProgress(REPO)
    result = ex.export_lerobot_v3(root, REPO, progress=progress, validate=False)
    assert result.episodes == 1 and result.episode_ids == [ids[2]]
    assert set(result.incomplete) == {ids[0], ids[1]}
    assert "frames.parquet" in result.incomplete[ids[0]] and "video" in result.incomplete[ids[1]]
    assert "2 incomplete" in progress.telemetry().detail
    assert read_manifest(root)["last_export"]["episodes"] == 1


# -- fix 7: validate before swap; leftovers -------------------------------------------------------
def test_export_validates_the_staging_tree_and_keeps_the_previous_export(tmp_path, monkeypatch):
    root = tmp_path / REPO
    rec = make_recorder(root)
    record(rec, 4)
    rec.finalize()
    out = root / "exports" / "lerobot_v3"
    ex.export_lerobot_v3(root, REPO, validate=False)
    marker = out / "meta" / "info.json"
    before = marker.read_bytes()
    seen: list[Path] = []

    def failing_validate(repo_id, path, expected):
        seen.append(Path(path))
        raise ex.ExportError("validation says no")

    monkeypatch.setattr(ex, "validate_export", failing_validate)
    progress = ex.ExportProgress(REPO)
    with pytest.raises(ex.ExportError):
        ex.export_lerobot_v3(root, REPO, progress=progress, validate=True)
    assert seen and seen[0] != out and seen[0].name.startswith(".lerobot_v3.tmp-")  # staging
    assert marker.read_bytes() == before  # the previous export is untouched
    assert not list(out.parent.glob(".lerobot_v3.tmp-*"))  # staging removed
    assert progress.telemetry().phase == "failed"
    assert read_manifest(root)["last_export"]["error"]
    # leftovers from a crashed job are swept at the next start
    (out.parent / ".lerobot_v3.tmp-deadbeef").mkdir()
    (out.parent / ".lerobot_v3.old-deadbeef").mkdir()
    monkeypatch.undo()
    ex.export_lerobot_v3(root, REPO, validate=False)
    assert not list(out.parent.glob(".lerobot_v3.*"))


def test_validate_export_runs_on_the_final_tree_when_no_previous_export(tmp_path):
    root = tmp_path / REPO
    rec = make_recorder(root)
    record(rec, 4)
    rec.finalize()
    progress = ex.ExportProgress(REPO)
    res = ex.export_lerobot_v3(root, REPO, progress=progress, validate=True)
    assert res.episodes == 1 and progress.telemetry().phase == "done"
    assert "validating" in progress.phase_started and "videos" in progress.phase_started
    assert progress.phase_started["validating"] - progress.phase_started["videos"] < 2.0


# -- fix 6(b): the recorder-level lock serialises a discard against a save ------------------------
def test_discard_during_a_slow_save_cannot_publish_a_half_directory(tmp_path, monkeypatch):
    root = tmp_path / REPO
    rec = make_recorder(root)
    rec.start({})
    rec.prepare()
    for i in range(4):
        rec.add_frame(frame(i))
    enc = rec._encoder
    real_finish = enc.finish_episode
    started = threading.Event()

    def slow_finish():
        started.set()
        time.sleep(0.4)
        return real_finish()

    monkeypatch.setattr(enc, "finish_episode", slow_finish)
    result: dict = {}

    def save():
        result["r"] = rec.save({"session_id": "s"}, None)

    t = threading.Thread(target=save)
    t.start()
    started.wait(2.0)
    rec.discard()  # concurrent teardown path: blocks on the lock until save completed
    t.join(10.0)
    _, eid = result["r"]
    d = root / "episodes" / eid
    assert {p.name for p in d.iterdir()} == {"episode.json", "frames.parquet", "video"}
    assert not list((root / "episodes").glob(".tmp-*"))
    rec.finalize()


def test_prepare_opens_the_encoder_before_the_first_frame(tmp_path):
    root = tmp_path / REPO
    rec = make_recorder(root)
    rec.start({})
    assert rec.open_tmp_dir is None
    rec.prepare()
    assert rec.open_tmp_dir is not None and rec.open_tmp_dir.is_dir()
    assert rec._encoder is not None and rec._encoder_open
    rec.prepare()  # idempotent
    for i in range(3):
        rec.add_frame(frame(i))
    _, eid = rec.save({"session_id": "s"}, None)
    assert (root / "episodes" / eid / "video" / f"{CAM}.mp4").exists()
    rec.finalize()
    shutil.rmtree(root)
