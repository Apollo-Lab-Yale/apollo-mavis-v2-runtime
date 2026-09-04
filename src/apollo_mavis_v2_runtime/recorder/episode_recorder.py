"""LeRobotEpisodeRecorder — core ``EpisodeRecorder`` over LeRobot v3.

Ownership: constructed and driven exclusively by the :class:`RecorderThread`
(single-owner writer; reading a dataset while a writer is open raises).
``lerobot`` is imported lazily HERE so no other runtime module pays the
import cost (04-runtime §10.1).

Crash safety (§10.4): the dataset dir keeps ``recorder_state.json``
``{repo_id, episodes_saved, finalized}``; an unfinalized dataset found at
startup is repaired via ``resume()`` + ``finalize()``
(:func:`repair_unfinalized_datasets`) — without finalize the parquet footers
are missing and the whole dataset is unreadable.
"""

from __future__ import annotations

import functools
import json
import logging
import os
from pathlib import Path
from typing import Any

from apollo_mavis_v2_core.interfaces.recorder import EpisodeRecorder

from ..config import RecorderConfig

logger = logging.getLogger(__name__)

STATE_FILENAME = "recorder_state.json"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def write_recorder_state(root: Path, repo_id: str, episodes_saved: int, finalized: bool) -> None:
    _write_json_atomic(
        Path(root) / STATE_FILENAME,
        {"repo_id": repo_id, "episodes_saved": episodes_saved, "finalized": finalized},
    )


_NVENC_CODECS = frozenset({"h264_nvenc", "hevc_nvenc"})

# lerobot pins ``g=2`` (two-frame GOP so any timestamp decodes with at most
# one reference frame) for EVERY codec.  NVENC's default B-frame count (3)
# violates its own "GOP length > bf + 1" rule, so ``avcodec_open2`` fails
# with EINVAL -- regardless of driver version or frame size.  ``bf=0`` gives
# a legal GOP while keeping hardware encoding.
NVENC_EXTRA_OPTIONS: dict[str, str] = {"bf": "0"}

DEFAULT_PROBE_SIZE: tuple[int, int] = (640, 480)  # (width, height)


def encoder_extra_options(vcodec: str) -> dict[str, str]:
    """Codec-specific ``extra_options`` merged into lerobot's encoder config."""
    return dict(NVENC_EXTRA_OPTIONS) if vcodec in _NVENC_CODECS else {}


def make_rgb_encoder(vcodec: str):
    """``RGBEncoderConfig`` for an already-RESOLVED codec (never ``"auto"``),
    carrying the extras the streaming encoder needs to open at all."""
    from lerobot.configs.video import RGBEncoderConfig

    if vcodec == "auto":
        raise ValueError("resolve_vcodec() first; make_rgb_encoder needs a concrete codec")
    return RGBEncoderConfig(vcodec=vcodec, extra_options=encoder_extra_options(vcodec))


def min_video_frame_size(
    features: dict[str, dict], default: tuple[int, int] = DEFAULT_PROBE_SIZE
) -> tuple[int, int]:
    """Smallest ``(width, height)`` over the video/image features (shape is
    ``(H, W, C)``).  Hardware encoders enforce a minimum frame size, so the
    probe must use the SMALLEST stream that will actually be encoded."""
    sizes = [
        (int(spec["shape"][1]), int(spec["shape"][0]))
        for spec in features.values()
        if spec.get("dtype") in ("video", "image") and len(spec.get("shape") or ()) == 3
    ]
    if not sizes:
        return default
    return (min(w for w, _ in sizes), min(h for _, h in sizes))


@functools.cache
def _encoder_opens(codec: str, width: int, height: int, fps: int = 30) -> bool:
    """True iff PyAV can OPEN ``codec`` with the EXACT options and pixel
    format lerobot's streaming encoder will pass.

    Listing is not enough, and neither is a bare open: ``h264_nvenc`` opens
    fine without options yet fails once lerobot's ``g=2`` is applied (see
    :data:`NVENC_EXTRA_OPTIONS`), so the probe must mirror the real call.
    Cached per (codec, size): an NVENC probe costs ~0.1-0.6 s (session setup).
    """
    import io

    import av
    import numpy as np

    try:
        encoder = make_rgb_encoder(codec)
        options = encoder.get_codec_options(None, as_strings=True)
        buf = io.BytesIO()
        out = av.open(buf, "w", format="mp4")
        stream = out.add_stream(codec, rate=fps, options=options)
        stream.width, stream.height = width, height
        stream.pix_fmt = encoder.pix_fmt
        frame = av.VideoFrame.from_ndarray(np.zeros((height, width, 3), np.uint8), format="rgb24")
        for pkt in stream.encode(frame):
            out.mux(pkt)
        for pkt in stream.encode(None):
            out.mux(pkt)
        out.close()
        return True
    except Exception:
        logger.debug("encoder %s does not open at %dx%d", codec, width, height, exc_info=True)
        return False


def resolve_vcodec(vcodec: str, frame_size: tuple[int, int] = DEFAULT_PROBE_SIZE) -> str:
    """Resolve ``"auto"`` with a REAL open-probe of the hardware encoders
    (NVENC on the 4090s) at ``frame_size`` -- the smallest stream to be
    encoded -- falling back to ``libsvtav1`` (10-frames §7.5).  Anything
    other than ``"auto"`` passes through untouched."""
    if vcodec != "auto":
        return vcodec
    from lerobot.configs.video import HW_VIDEO_CODECS

    width, height = int(frame_size[0]), int(frame_size[1])
    for codec in HW_VIDEO_CODECS:
        if _encoder_opens(codec, width, height):
            logger.info("vcodec auto -> %s (probed at %dx%d)", codec, width, height)
            return codec
    logger.warning("no hardware encoder opens at %dx%d; vcodec auto -> libsvtav1", width, height)
    return "libsvtav1"


# lerobot persists the CANONICAL codec name (``video.codec``: h264/hevc/av1);
# its offline merge (10-frames §8) refuses videos of different codecs, so a
# resumed dataset must keep its family whatever the current host can open.
_SOFTWARE_ENCODER_FOR_FAMILY: dict[str, str] = {"h264": "h264", "hevc": "hevc", "av1": "libsvtav1"}
_ENCODER_FAMILY_ALIASES: dict[str, str] = {
    "libx264": "h264",
    "libopenh264": "h264",
    "libx265": "hevc",
    "libsvtav1": "av1",
    "libaom-av1": "av1",
}


def codec_family(vcodec: str) -> str | None:
    """``h264`` / ``hevc`` / ``av1`` for an encoder name, ``None`` if unknown."""
    if vcodec in _ENCODER_FAMILY_ALIASES:
        return _ENCODER_FAMILY_ALIASES[vcodec]
    head = vcodec.split("_", 1)[0]  # h264_nvenc -> h264, hevc_videotoolbox -> hevc
    return head if head in _SOFTWARE_ENCODER_FOR_FAMILY else None


def stored_video_codec(root: Path) -> str | None:
    """Codec family of the video features already in ``root/meta/info.json``;
    ``None`` when there is no dataset (or no video feature) there."""
    info_path = Path(root) / "meta" / "info.json"
    if not info_path.exists():
        return None
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    codecs = {
        str(spec["info"]["video.codec"])
        for spec in (info.get("features") or {}).values()
        if isinstance(spec, dict)
        and spec.get("dtype") == "video"
        and isinstance(spec.get("info"), dict)
        and spec["info"].get("video.codec")
    }
    if len(codecs) > 1:
        raise ValueError(f"dataset at {root} already mixes video codecs {sorted(codecs)}")
    return codecs.pop() if codecs else None


def resolve_vcodec_for_dataset(
    vcodec: str, root: Path, frame_size: tuple[int, int] = DEFAULT_PROBE_SIZE
) -> str:
    """:func:`resolve_vcodec`, but a dataset that already exists at ``root``
    pins the codec FAMILY: ``"auto"`` re-probes only within it (hardware
    encoder first, then the software one), and an explicit codec of another
    family is rejected so the operator records into a fresh repo instead of
    producing a dataset lerobot cannot merge."""
    stored = stored_video_codec(root)
    if stored is None:
        return resolve_vcodec(vcodec, frame_size)
    width, height = int(frame_size[0]), int(frame_size[1])
    if vcodec != "auto":
        if codec_family(vcodec) != stored:
            raise ValueError(
                f"vcodec {vcodec!r} ({codec_family(vcodec)}) would mix with the {stored} "
                f"videos already in {root}; use vcodec 'auto' or a fresh repo"
            )
        return vcodec
    from lerobot.configs.video import HW_VIDEO_CODECS

    for codec in HW_VIDEO_CODECS:
        if codec_family(codec) == stored and _encoder_opens(codec, width, height):
            logger.info("resumed %s dataset: vcodec auto -> %s", stored, codec)
            return codec
    software = _SOFTWARE_ENCODER_FOR_FAMILY[stored]
    if _encoder_opens(software, width, height):
        logger.warning(
            "resumed %s dataset: no hardware encoder opens at %dx%d; vcodec auto -> %s",
            stored,
            width,
            height,
            software,
        )
        return software
    raise RuntimeError(f"no {stored} encoder opens at {width}x{height}; cannot resume {root}")


class LeRobotEpisodeRecorder(EpisodeRecorder):
    """Buffered episode recording over ``LeRobotDataset`` (04-runtime §10.1)."""

    def __init__(
        self,
        cfg: RecorderConfig,
        features: dict[str, dict],
        root: Path,
        repo_id: str,
        robot_type: str,
        default_task: str,
    ) -> None:
        # Lazy import: lerobot (and torch behind it) loads only when a
        # collect/dagger session actually constructs a recorder.
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.cfg = cfg
        self.root = Path(root)
        self.repo_id = repo_id
        self._task = default_task
        self._recording = False
        self._finalized = False
        self._frames_in_buffer = 0
        self.vcodec = resolve_vcodec_for_dataset(
            cfg.vcodec, self.root, min_video_frame_size(features)
        )
        rgb_encoder = make_rgb_encoder(self.vcodec)
        if (self.root / "meta" / "info.json").exists():
            self._ds = LeRobotDataset.resume(
                repo_id,
                root=self.root,
                rgb_encoder=rgb_encoder,
                streaming_encoding=True,
                image_writer_threads=cfg.image_writer_threads,
            )
        else:
            self._ds = LeRobotDataset.create(
                repo_id,
                fps=cfg.fps,
                features=features,
                root=self.root,
                robot_type=robot_type,
                use_videos=True,
                streaming_encoding=True,
                rgb_encoder=rgb_encoder,
                image_writer_threads=cfg.image_writer_threads,
            )
        self.episodes_saved = int(self._ds.num_episodes)
        write_recorder_state(self.root, repo_id, self.episodes_saved, finalized=False)

    # -- core EpisodeRecorder API ------------------------------------------------
    def start(self, meta: dict[str, object]) -> None:
        """Open a new episode buffer (no writer op until the first frame)."""
        if self._recording:
            raise RuntimeError("episode already recording")
        task = meta.get("task")
        if task:
            self._task = str(task)
        self._frames_in_buffer = 0
        self._recording = True

    def add_frame(self, frame: dict[str, object]) -> None:
        """Append one frame; ``task`` is injected here (required per frame)."""
        if not self._recording:
            raise RuntimeError("no episode buffer open")
        self._ds.add_frame({**frame, "task": self._task})
        self._frames_in_buffer += 1

    def save(self) -> int:
        """Commit the buffer; returns the episode index. On failure the
        buffer is KEPT (caller retries once, 04-runtime §15)."""
        if not self._recording:
            raise RuntimeError("no episode buffer open")
        self._ds.save_episode()
        self._recording = False
        self._frames_in_buffer = 0
        self.episodes_saved = int(self._ds.num_episodes)
        write_recorder_state(self.root, self.repo_id, self.episodes_saved, finalized=False)
        return self.episodes_saved - 1

    def discard(self) -> None:
        """Drop the buffer (cancels the streaming encoder; discard is free)."""
        if self._recording:
            self._ds.clear_episode_buffer()
            self._recording = False
            self._frames_in_buffer = 0

    def finalize(self) -> None:
        """MANDATORY at session end (parquet footers). Idempotent."""
        if self._finalized:
            return
        if self._recording:  # open buffer at teardown: discard, never half-save
            try:
                self.discard()
            except Exception:
                logger.exception("discard during finalize failed")
                self._recording = False
        self._ds.finalize()
        self._finalized = True
        write_recorder_state(self.root, self.repo_id, self.episodes_saved, finalized=True)

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def frames_in_buffer(self) -> int:
        return self._frames_in_buffer


def repair_unfinalized_datasets(datasets_root: Path) -> list[str]:
    """Startup repair (04-runtime §15): find ``recorder_state.json`` files
    with ``finalized == false`` and repair via ``resume()`` + ``finalize()``.

    Scanning is filesystem-only; lerobot is imported iff a repair is needed.
    Returns the repo_ids repaired.
    """
    root = Path(datasets_root)
    if not root.exists():
        return []
    repaired: list[str] = []
    for state_path in sorted(root.glob(f"**/{STATE_FILENAME}")):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("unreadable %s; skipping", state_path)
            continue
        if state.get("finalized", True):
            continue
        repo_id = str(state.get("repo_id", ""))
        ds_root = state_path.parent
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            ds = LeRobotDataset.resume(repo_id, root=ds_root)
            ds.finalize()
            write_recorder_state(
                ds_root, repo_id, int(state.get("episodes_saved", 0)), finalized=True
            )
            repaired.append(repo_id)
            logger.warning("repaired unfinalized dataset %s at %s", repo_id, ds_root)
        except Exception:
            logger.exception("failed to repair dataset at %s", ds_root)
    return repaired


__all__ = [
    "DEFAULT_PROBE_SIZE",
    "LeRobotEpisodeRecorder",
    "NVENC_EXTRA_OPTIONS",
    "codec_family",
    "encoder_extra_options",
    "make_rgb_encoder",
    "min_video_frame_size",
    "repair_unfinalized_datasets",
    "resolve_vcodec",
    "resolve_vcodec_for_dataset",
    "stored_video_codec",
    "write_recorder_state",
    "STATE_FILENAME",
]
