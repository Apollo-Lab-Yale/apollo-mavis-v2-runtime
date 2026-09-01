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

import json
import logging
import os
from pathlib import Path
from typing import Any

from apollo_xarm7_core.interfaces.recorder import EpisodeRecorder

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


def _encoder_opens(codec: str) -> bool:
    """True iff PyAV can actually OPEN this encoder (listing is not enough:
    ``h264_nvenc`` shows as available yet fails on old NVIDIA drivers)."""
    import io

    import av
    import numpy as np

    try:
        buf = io.BytesIO()
        out = av.open(buf, "w", format="mp4")
        stream = out.add_stream(codec, rate=25)
        stream.width, stream.height = 64, 64
        stream.pix_fmt = "yuv420p"
        frame = av.VideoFrame.from_ndarray(np.zeros((64, 64, 3), np.uint8), format="rgb24")
        for pkt in stream.encode(frame):
            out.mux(pkt)
        out.close()
        return True
    except Exception:
        return False


def resolve_vcodec(vcodec: str) -> str:
    """Resolve ``"auto"`` with a REAL open-probe of the hardware encoders
    (NVENC on the 4090s), falling back to ``libsvtav1`` (10-frames §7.5)."""
    if vcodec != "auto":
        return vcodec
    from lerobot.configs.video import HW_VIDEO_CODECS

    for codec in HW_VIDEO_CODECS:
        if _encoder_opens(codec):
            logger.info("vcodec auto -> %s", codec)
            return codec
    logger.warning("no hardware encoder usable; vcodec auto -> libsvtav1")
    return "libsvtav1"


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
        from lerobot.configs.video import RGBEncoderConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.cfg = cfg
        self.root = Path(root)
        self.repo_id = repo_id
        self._task = default_task
        self._recording = False
        self._finalized = False
        self._frames_in_buffer = 0
        rgb_encoder = RGBEncoderConfig(vcodec=resolve_vcodec(cfg.vcodec))
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
    "LeRobotEpisodeRecorder",
    "repair_unfinalized_datasets",
    "resolve_vcodec",
    "write_recorder_state",
    "STATE_FILENAME",
]
