"""EpisodeDirRecorder over the REAL lerobot streaming encoder + pyarrow (10-frames §11;
04-runtime §10.1/§10.4): directory layout, ``frames == rows``, atomic publication,
``.tmp-*`` crash sweep, manifest counters, encoder identity, resume + codec family
pinning, ``export_ok`` on a short video, idempotent finalize."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from apollo_mavis_v2_runtime.config import RecorderConfig
from apollo_mavis_v2_runtime.recorder import stats as rstats
from apollo_mavis_v2_runtime.recorder.episode_recorder import (
    DEFAULT_PROBE_SIZE,
    NVENC_EXTRA_OPTIONS,
    EpisodeDirRecorder,
    _encoder_opens,
    codec_family,
    dataset_incompatibility,
    encoder_extra_options,
    make_rgb_encoder,
    min_video_frame_size,
    resolve_vcodec,
    resolve_vcodec_for_dataset,
    stored_video_codec,
    sweep_incomplete_episodes,
)
from apollo_mavis_v2_runtime.recorder.features import ArmMeta, build_features
from apollo_mavis_v2_runtime.recorder.manifest import (
    EPISODE_JSON,
    FRAMES_PARQUET,
    MANIFEST_FILENAME,
    mint_episode_id,
    read_manifest,
)

REPO_ID = "apollo/xarm7_test_1arm_dee-base"
FRAMES = {"arm0": "arm_base:arm0"}
CAM = "cam0"
W, H = 64, 48
ROBOT_TYPE = "xarm7_1arm_rail_mujoco"
SOFT = RecorderConfig(vcodec="libsvtav1")  # tiny frames: keep CI off the hardware encoders
NVENC_OK = _encoder_opens("h264_nvenc", *DEFAULT_PROBE_SIZE)


def features(width: int = W, height: int = H):
    return build_features([ArmMeta("arm0", True)], FRAMES, {CAM: (width, height)})


def make_frame(i: int, width: int = W, height: int = H) -> dict:
    return {
        "action": np.full(8, 0.001 * i, dtype=np.float32),
        "action.abs_ee": np.full(11, 0.002 * i, dtype=np.float32),
        "observation.state": np.full(16, 0.01 * i, dtype=np.float32),
        f"observation.images.{CAM}": np.full((height, width, 3), i % 255, dtype=np.uint8),
        "intervention": np.array([False]),
        "action_source": np.array([1], dtype=np.int8),
        "wallclock_ns": np.array([1_700_000_000_000_000_000 + i * 40_000_000], dtype=np.int64),
    }


def make_recorder(root, cfg: RecorderConfig | None = None, size=(W, H)) -> EpisodeDirRecorder:
    return EpisodeDirRecorder(cfg or SOFT, features(*size), root, REPO_ID, ROBOT_TYPE, "test task")


def record_episode(rec: EpisodeDirRecorder, n: int, size=(W, H), sidecar=None) -> tuple[int, str]:
    rec.start({})
    for i in range(n):
        rec.add_frame(make_frame(i, *size))
    return rec.save(sidecar or {"session_id": "s1", "frames_dropped": 0}, None)


def decoded_frames(path: Path) -> int:
    import av

    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    """One recorder session: save ep0 (10 frames), discard one, save ep1 (6), finalize."""
    root = tmp_path_factory.mktemp("ds") / REPO_ID
    rec = make_recorder(root)
    assert rec.episodes_saved == 0 and rec.total_frames == 0 and not rec.recording
    rec.start({"task": "test task"})
    assert rec.recording and rec.episode_id is not None
    for i in range(10):
        rec.add_frame(make_frame(i))
    assert rec.frames_in_buffer == 10 and rec.open_tmp_dir is not None
    ord0, id0 = rec.save({"session_id": "s1", "frames_dropped": 2}, None)
    assert (ord0, rec.recording, rec.episode_id) == (0, False, None)
    rec.start({})
    for i in range(5):
        rec.add_frame(make_frame(i))
    rec.discard()  # cancels the encoder; leaves no trace
    assert not rec.recording and not list((root / "episodes").glob(".tmp-*"))
    ord1, id1 = record_episode(rec, 6)
    assert ord1 == 1 and id1 > id0  # ids sort in capture order
    rec.finalize()
    rec.finalize()  # idempotent
    return root, [id0, id1]


def test_episode_id_grammar():
    eid = mint_episode_id()
    assert len(eid) == 27 and eid[15] == "." and eid[19:21] == "Z-"
    assert all(c in "0123456789abcdef" for c in eid[21:])


def test_directory_layout_and_frames_equal_rows(saved):
    root, ids = saved
    assert sorted(p.name for p in (root / "episodes").iterdir()) == ids  # no .tmp-*
    assert (root / MANIFEST_FILENAME).exists()
    assert not (root / "meta" / "info.json").exists()
    assert not (root / "recorder_state.json").exists()
    for eid, n in zip(ids, (10, 6), strict=True):
        d = root / "episodes" / eid
        assert {p.name for p in d.iterdir()} == {EPISODE_JSON, FRAMES_PARQUET, "video"}
        assert [p.name for p in (d / "video").iterdir()] == [f"{CAM}.mp4"]
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(d / FRAMES_PARQUET)
        assert pf.metadata.num_rows == n and pf.metadata.num_row_groups == 1
        assert pf.schema_arrow.names == [
            "action", "action.abs_ee", "observation.state", "intervention", "action_source",
            "wallclock_ns", "timestamp", "frame_index", "task",
        ]
        table = pf.read()
        import pyarrow as pa

        assert table.schema.field("action").type == pa.list_(pa.float32(), 8)
        assert table.schema.field("action.abs_ee").type == pa.list_(pa.float32(), 11)
        assert np.allclose(table.column("action.abs_ee")[n - 1].as_py(), [0.002 * (n - 1)] * 11)
        assert str(table.schema.field("timestamp").type) == "float"
        assert table.column("frame_index").to_pylist() == list(range(n))
        assert table.column("task").to_pylist() == ["test task"] * n
        assert np.allclose(table.column("timestamp").to_numpy(), np.arange(n) / 25.0)
        assert decoded_frames(d / "video" / f"{CAM}.mp4") == n
        ep = json.loads((d / EPISODE_JSON).read_text())
        assert ep["episode_id"] == eid and ep["episode_index"] is None
        assert ep["length"] == n and ep["fps"] == 25 and ep["duration_s"] == n / 25
        assert ep["tasks"] == ["test task"] and ep["export_ok"] is True
        assert ep["export_note"] is None
        assert ep["recorded_at"] == "2023-11-14T22:13:20.000Z"  # the first frame's wallclock_ns
        assert ep["audio"] is None and ep["session_id"] == "s1"
        v = ep["video"][CAM]
        assert v["file"] == f"video/{CAM}.mp4" and v["frames"] == n and v["encoder_drops"] == 0
        assert (v["codec"], v["encoder"], v["pix_fmt"]) == ("av1", "libsvtav1", "yuv420p")
        assert v["g"] == 2
        assert (v["width"], v["height"]) == (W, H)
        # stats: lerobot semantics — (3,1,1) image stats in [0,1], vectors per dim, count (1,)
        st = ep["stats"]
        assert set(st) == {
            "action", "action.abs_ee", "observation.state", "intervention", "action_source",
            "wallclock_ns", f"observation.images.{CAM}",
        }
        assert np.asarray(st["action.abs_ee"]["mean"]).shape == (11,)
        assert st["action.abs_ee"]["count"] == [n]
        img = st[f"observation.images.{CAM}"]
        assert np.asarray(img["mean"]).shape == (3, 1, 1) and 0.0 <= img["mean"][0][0][0] <= 1.0
        assert img["count"][0] >= n  # lerobot's writer stores the downsampled PIXEL count
        assert np.asarray(st["action"]["mean"]).shape == (8,) and st["action"]["count"] == [n]
        assert np.asarray(st["intervention"]["min"]).shape == (1,)
        assert set(st["action"]) == set(rstats.STAT_KEYS)
    first = json.loads((root / "episodes" / ids[0] / EPISODE_JSON).read_text())
    assert first["frames_dropped"] == 2


def test_manifest_counters_identity_and_stale_flag(saved):
    root, ids = saved
    m = read_manifest(root)
    assert m["apollo_dataset_layout"] == 1 and m["repo_id"] == REPO_ID
    assert (m["episodes"], m["frames"], m["fps"], m["robot_type"]) == (2, 16, 25, ROBOT_TYPE)
    assert m["arms"] == ["arm0"] and m["cameras"] == [CAM]
    assert m["video"] == {
        "codec": "av1", "encoder": "libsvtav1", "pix_fmt": "yuv420p", "g": 2, "crf": 30,
        "extra_options": {}, "backend": "pyav",
    }
    assert m["features"]["action"]["shape"] == [8]
    assert m["features"]["action"]["info"]["apollo_schema"] == 1
    assert "action.abs_ee" in m["features"]  # the manifest is written sort_keys, no order pin
    assert m["features"]["action.abs_ee"]["shape"] == [11]
    assert m["features"]["action.abs_ee"]["info"]["action_space"] == "abs_ee"
    assert m["last_export"] is None  # never exported: nothing to mark stale
    assert stored_video_codec(root) == "av1"


def test_resume_appends_and_pins_the_codec_family(saved):
    root, ids = saved
    rec = make_recorder(root, RecorderConfig(vcodec="auto"))  # resume: family pinned to av1
    assert rec.vcodec == "libsvtav1" and rec.episodes_saved == 2 and rec.total_frames == 16
    ordinal, eid = record_episode(rec, 4)
    assert ordinal == 2 and eid not in ids
    rec.finalize()
    m = read_manifest(root)
    assert (m["episodes"], m["frames"]) == (3, 20)
    shutil.rmtree(root / "episodes" / eid)  # put the fixture back for the other tests
    from apollo_mavis_v2_runtime.recorder.manifest import refresh_manifest

    refresh_manifest(root)
    assert read_manifest(root)["episodes"] == 2
    with pytest.raises(ValueError, match="mix"):
        resolve_vcodec_for_dataset("h264_nvenc", root, (W, H))
    assert resolve_vcodec_for_dataset("libsvtav1", root, (W, H)) == "libsvtav1"


def test_dataset_incompatibility_reads_the_manifest_incl_info_blocks(saved):
    root, _ = saved
    m = read_manifest(root)
    assert dataset_incompatibility(None, features(), 25, ROBOT_TYPE) is None
    assert dataset_incompatibility(m, features(), 25, ROBOT_TYPE) is None
    assert "fps" in dataset_incompatibility(m, features(), 30, ROBOT_TYPE)
    assert "robot_type" in dataset_incompatibility(m, features(), 25, "xarm7_1arm_rail")
    other = build_features([ArmMeta("arm0", True)], {"arm0": "world"}, {CAM: (W, H)})
    why = dataset_incompatibility(m, other, 25, ROBOT_TYPE)
    assert why is not None and "feature 'action' differs" in why  # the frames info block
    two = build_features([ArmMeta("arm0", True), ArmMeta("arm1", False)],
                         {"arm0": "arm_base:arm0", "arm1": "arm_base:arm1"}, {CAM: (W, H)})
    assert "differs" in dataset_incompatibility(m, two, 25, ROBOT_TYPE)
    # a dataset recorded before action.abs_ee existed: the refusal names the backfill tool
    old_manifest = json.loads(json.dumps(m))
    del old_manifest["features"]["action.abs_ee"]
    why = dataset_incompatibility(old_manifest, features(), 25, ROBOT_TYPE)
    assert why is not None and "this session adds ['action.abs_ee']" in why
    assert "apollo_mavis_v2_runtime.tools.backfill_abs_ee" in why
    # ... but not when anything else differs too
    del old_manifest["features"]["wallclock_ns"]
    why2 = dataset_incompatibility(old_manifest, features(), 25, ROBOT_TYPE)
    assert why2 is not None and "backfill_abs_ee" not in why2
    with pytest.raises(ValueError, match="cannot be continued"):
        EpisodeDirRecorder(SOFT, other, root, REPO_ID, ROBOT_TYPE, "t")


def test_crash_leaves_one_tmp_dir_that_the_next_open_sweeps(tmp_path):
    root = tmp_path / REPO_ID
    rec = make_recorder(root)
    record_episode(rec, 4)
    rec.start({})
    rec.add_frame(make_frame(0))
    tmp = rec.open_tmp_dir
    assert tmp is not None and tmp.name.startswith(".tmp-") and tmp.is_dir()
    # "process dies": drop the recorder without discard/finalize
    rec._encoder.cancel_episode()  # release the encoder thread; the directory stays
    del rec
    assert [p.name for p in (root / "episodes").glob(".tmp-*")] == [tmp.name]
    swept = sweep_incomplete_episodes(tmp_path)
    assert swept == [tmp] and not tmp.exists()
    assert len([p for p in (root / "episodes").iterdir()]) == 1  # the saved one is intact
    assert sweep_incomplete_episodes(tmp_path) == []
    # a keep-set protects the running session's open episode
    keep = root / "episodes" / ".tmp-live"
    keep.mkdir()
    assert sweep_incomplete_episodes(tmp_path, keep={keep}) == []
    assert keep.exists()
    # the recorder's own open sweeps too
    rec2 = make_recorder(root)
    assert not keep.exists() and rec2.episodes_saved == 1
    rec2.finalize()


def test_short_video_marks_export_ok_false_but_keeps_the_episode(tmp_path, monkeypatch):
    """The encoder queue dropped a frame (``_dropped_frames``): fed - dropped != rows ->
    kept with export_ok false + a note; the export skips it (10-frames §11.4)."""
    root = tmp_path / REPO_ID
    rec = make_recorder(root)
    rec.start({})
    for i in range(6):
        rec.add_frame(make_frame(i))
    enc = rec._encoder
    real_finish = enc.finish_episode

    def finish_with_a_drop():
        out = real_finish()
        enc._dropped_frames[f"observation.images.{CAM}"] = 1
        return out

    monkeypatch.setattr(enc, "finish_episode", finish_with_a_drop)
    ordinal, eid = rec.save({"session_id": "s"}, None)
    ep = json.loads((root / "episodes" / eid / EPISODE_JSON).read_text())
    assert ep["export_ok"] is False
    assert "5 video frames for 6 rows" in ep["export_note"] and "1 dropped" in ep["export_note"]
    assert ep["video"][CAM]["frames"] == 5 and ep["video"][CAM]["encoder_drops"] == 1
    assert read_manifest(root)["episodes"] == 1  # kept and counted
    rec.finalize()


def test_save_retry_resumes_after_the_encoder_step(tmp_path, monkeypatch):
    """A failure AFTER finish_episode (parquet write) must not re-run the encoder:
    the RecorderThread retries once and the retry must succeed (04-runtime §15)."""
    import pyarrow.parquet as pq

    root = tmp_path / REPO_ID
    rec = make_recorder(root)
    rec.start({})
    for i in range(5):
        rec.add_frame(make_frame(i))
    calls = {"n": 0}
    real = pq.write_table

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk hiccup")
        return real(*a, **kw)

    monkeypatch.setattr(pq, "write_table", flaky)
    with pytest.raises(OSError):
        rec.save({"session_id": "s"}, None)
    assert rec.recording  # buffer kept
    ordinal, eid = rec.save({"session_id": "s"}, None)  # retry: no finish_episode again
    assert ordinal == 0 and (root / "episodes" / eid / FRAMES_PARQUET).exists()
    assert decoded_frames(root / "episodes" / eid / "video" / f"{CAM}.mp4") == 5
    rec.finalize()


def test_empty_save_and_double_start_are_errors(tmp_path):
    rec = make_recorder(tmp_path / REPO_ID)
    rec.start({})
    with pytest.raises(ValueError, match="empty"):
        rec.save({}, None)
    with pytest.raises(RuntimeError):
        rec.start({})
    rec.discard()
    rec.discard()  # idempotent
    rec.finalize()
    with pytest.raises(RuntimeError):
        rec.start({})


def test_legacy_v3_tree_is_refused(tmp_path):
    root = tmp_path / REPO_ID
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text("{}")
    with pytest.raises(ValueError, match="legacy"):
        make_recorder(root)


# -- encoder helpers (kept from phase-07) --------------------------------------------------------
def test_vcodec_auto_resolves_to_usable_encoder():
    codec = resolve_vcodec("auto", (16, 16))
    assert codec == "libsvtav1" or _encoder_opens.__wrapped__(codec, 16, 16)


def test_nvenc_extra_options_make_lerobot_short_gop_legal():
    assert NVENC_EXTRA_OPTIONS == {"bf": "0"}
    assert encoder_extra_options("h264_nvenc") == {"bf": "0"}
    assert encoder_extra_options("libsvtav1") == {}
    enc = make_rgb_encoder("h264_nvenc")
    assert enc.get_codec_options(None, as_strings=True)["bf"] == "0"
    with pytest.raises(ValueError):
        make_rgb_encoder("auto")


def test_min_video_frame_size_picks_smallest_stream():
    f = build_features([ArmMeta("arm0", True)], FRAMES, {"a": (640, 480), "b": (64, 48)})
    assert min_video_frame_size(f) == (64, 48)
    assert min_video_frame_size({}) == DEFAULT_PROBE_SIZE


def test_codec_family_and_fresh_root():
    assert codec_family("h264_nvenc") == "h264"
    assert codec_family("hevc_videotoolbox") == "hevc"
    assert codec_family("libsvtav1") == "av1"
    assert codec_family("libx264") == "h264"
    assert codec_family("bogus") is None
    assert stored_video_codec(Path("/nonexistent/apollo/xarm7")) is None
    assert resolve_vcodec_for_dataset("libsvtav1", Path("/nonexistent"), (64, 48)) == "libsvtav1"


@pytest.mark.skipif(not NVENC_OK, reason="no usable NVENC encoder on this host")
def test_nvenc_records_through_lerobot_streaming_encoder(tmp_path):
    """h264_nvenc opens with lerobot's g=2 only because bf=0 rides along; frames
    must encode AND decode, and the identity lands in episode.json + the manifest."""
    root = tmp_path / REPO_ID
    rec = make_recorder(root, RecorderConfig(vcodec="auto"), DEFAULT_PROBE_SIZE)
    assert rec.vcodec == "h264_nvenc"
    _, eid = record_episode(rec, 6, DEFAULT_PROBE_SIZE)
    rec.finalize()
    ep = json.loads((root / "episodes" / eid / EPISODE_JSON).read_text())
    v = ep["video"][CAM]
    assert (v["codec"], v["encoder"], v["extra_options"]) == ("h264", "h264_nvenc", {"bf": "0"})
    assert decoded_frames(root / "episodes" / eid / "video" / f"{CAM}.mp4") == 6
    assert read_manifest(root)["video"]["extra_options"] == {"bf": "0"}
