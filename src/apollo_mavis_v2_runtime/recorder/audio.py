"""EpisodeAudioSink — per-episode WAV sidecar from the microphone reader
(10-frames §9 "audio"; 04-runtime §10.5).

lerobot 0.6 has no audio feature, so the Perception Arm microphone is recorded
as a sidecar of the episode directory: ``episodes/<episode_id>/audio.wav`` (mono,
PCM s16le, the reader's sample rate — 48 kHz for the RØDE NT-USB Mini), with
the alignment reference in ``episode.json`` (``audio`` block, 10-frames §11.4).
The recorder calls :meth:`finish` with the episode's temp directory before it
publishes the directory, so the WAV is inside it at the rename. The raw float32
blocks exist only inside
``MicrophoneReader._publish``; :meth:`on_block` is the sink it calls on the
capture thread, so this class buffers copies and never touches the writer.

Alignment (documented, binding): every dataset frame carries ``wallclock_ns``
(the arm report's wall clock); every microphone block carries ``rx_mono`` (the
host monotonic clock when the block was READ, i.e. its END). ``t0_mono`` /
``t0_wallclock_ns`` are one simultaneous reading of both clocks taken at
:meth:`begin`, so ``wallclock_ns(sample i) ≈ t0_wallclock_ns + (audio_start_mono
+ i / sample_rate - t0_mono) * 1e9`` with ``audio_start_mono = first block
rx_mono - block_len / sample_rate``. Block boundaries are contiguous unless
the reader reported overruns during the episode (``overruns_delta``).
"""

from __future__ import annotations

import logging
import threading
import time
import wave
from pathlib import Path
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class _Reader(Protocol):
    """The slice of :class:`~apollo_mavis_v2_runtime.devices.microphone.MicrophoneReader`
    the sink needs (tests hand in a stub)."""

    cfg: Any  # .sample_rate

    def add_sink(self, fn) -> None: ...
    def remove_sink(self, fn) -> None: ...
    def status(self, now: float | None = None) -> Any: ...  # .overruns, .status


class EpisodeAudioSink:
    """Buffer the microphone between ``begin()`` and ``finish()`` / ``abort()``."""

    def __init__(self, reader: _Reader, clock=time.monotonic) -> None:
        self.reader = reader
        self.sample_rate = int(reader.cfg.sample_rate)
        self._clock = clock
        self._lock = threading.Lock()
        self._blocks: list[np.ndarray] = []
        self._rx: list[float] = []
        self._open = False
        self._t0_mono = 0.0
        self._t0_wallclock_ns = 0
        self._overruns0 = 0
        # ONE bound-method object for add_sink / remove_sink: the reader removes by
        # identity, and ``self.on_block`` is a fresh object on every attribute access.
        self._sink_fn = self.on_block

    # -- capture-thread side ------------------------------------------------------------
    def on_block(self, samples: np.ndarray, rx_mono: float) -> None:
        with self._lock:
            if not self._open:
                return
            self._blocks.append(np.array(samples, dtype=np.float32, copy=True).reshape(-1))
            self._rx.append(float(rx_mono))

    # -- recorder-thread side -----------------------------------------------------------
    def begin(self) -> None:
        with self._lock:
            self._blocks, self._rx = [], []
            self._t0_mono = float(self._clock())
            self._t0_wallclock_ns = time.time_ns()
            self._overruns0 = self._overruns()
            self._open = True
        self.reader.add_sink(self._sink_fn)

    def abort(self) -> None:
        self.reader.remove_sink(self._sink_fn)
        with self._lock:
            self._open = False
            self._blocks, self._rx = [], []

    def finish(self, path: Path, root: Path | None = None) -> dict[str, Any] | None:
        """Write the buffered episode to ``path`` (PCM s16le mono); returns the
        sidecar ``audio`` block (``path`` RELATIVE to ``root`` when given, so a
        dataset directory can be moved / re-indexed), or ``None`` when no block
        arrived (the reader was not live — the sidecar then says so instead of
        pointing at a missing file)."""
        self.reader.remove_sink(self._sink_fn)
        with self._lock:
            self._open = False
            blocks, rx = self._blocks, self._rx
            self._blocks, self._rx = [], []
            t0_mono, t0_wall, over0 = self._t0_mono, self._t0_wallclock_ns, self._overruns0
        if not blocks:
            return None
        samples = np.concatenate(blocks)
        pcm = np.clip(samples * 32767.0, -32768.0, 32767.0).astype("<i2")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with wave.open(str(tmp), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(pcm.tobytes())
        tmp.replace(path)
        block_len = int(blocks[0].shape[0])
        audio_start_mono = rx[0] - block_len / self.sample_rate
        return {
            "path": str(path.relative_to(root)) if root is not None else str(path),
            "format": "wav",
            "sample_format": "pcm_s16le",
            "channels": 1,
            "sample_rate": self.sample_rate,
            "samples": int(samples.shape[0]),
            "duration_s": float(samples.shape[0] / self.sample_rate),
            "blocks": len(blocks),
            "block_len": block_len,
            "t0_mono": t0_mono,
            "t0_wallclock_ns": t0_wall,
            "audio_start_mono": audio_start_mono,
            "audio_start_wallclock_ns": int(t0_wall + (audio_start_mono - t0_mono) * 1e9),
            "last_block_rx_mono": rx[-1],
            "overruns_delta": self._overruns() - over0,
            "clock_note": (
                "rx_mono is the host monotonic clock at the END of each block; frames carry "
                "wallclock_ns; t0_* is one simultaneous reading of both clocks at episode start"
            ),
        }

    def _overruns(self) -> int:
        try:
            return int(getattr(self.reader.status(), "overruns", 0) or 0)
        except Exception:  # noqa: BLE001 - a stub reader without status()
            return 0


__all__ = ["EpisodeAudioSink"]
