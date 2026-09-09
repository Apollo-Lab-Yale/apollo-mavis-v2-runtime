"""VideoHub — encode-once JPEG pipelines (04-runtime §13.4).

One :class:`EncoderWorker` thread per stream id pulls the newest frame from
its FrameSource (any object with ``latest() -> CameraFrame | None``), JPEG-
encodes at q80 via ``cv2.imencode``, prepends the binding 12-byte ``<dI``
header (core ``pack_frame``), and publishes the single buffer into
``encoded[stream_id]`` — WS senders and the MJPEG debug endpoint share it.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import cv2
from apollo_mavis_v2_core import CameraFrame, LatestSlot
from apollo_mavis_v2_core.protocol import pack_frame

logger = logging.getLogger(__name__)


# A tick waits for the source's next frame in steps of ``period / _POLLS_PER_PERIOD``
# (>= _POLL_STEP_MIN_S); a source idle for ``fps`` consecutive ticks (~1 s) is polled
# once per tick until it produces again, so a stalled camera costs no extra wake-ups.
_POLLS_PER_PERIOD = 8
_POLL_STEP_MIN_S = 0.002


class EncoderWorker:
    """Paces one stream at its own fps (a CAP): latest-wins, drops on no-new-frame.

    A tick does not sample the source once on a fixed 1/fps grid: it waits INSIDE its
    period for the next distinct ``frame.seq`` and re-anchors the grid on that frame, so
    the encoder phase-locks to an equal-rate producer (the sim ``RenderService`` renders
    every preview camera at ``preview_fps`` and ``start_previews`` adds the streams
    milliseconds later, i.e. the two 1/fps grids start nearly coincident). Sampling a
    fixed grid there let render-time jitter decide each poll and silently lost 5-20 % of
    the output periods (2026-09-08: ``cam_view_wrist_cam`` at 11.9-14.5 Hz with
    contiguous seqs, frame age p99 = one full period; 14-dora §7 promises 15 Hz).
    Faster sources are still capped at ``fps`` (the poll is due only once per period).
    """

    def __init__(
        self,
        stream_id: str,
        source,  # .latest() -> CameraFrame | None
        slot: LatestSlot[bytes],
        fps: float,
        jpeg_quality: int = 80,
    ) -> None:
        self.stream_id = stream_id
        self.source = source
        self.slot = slot
        self.jpeg_quality = int(jpeg_quality)
        self._fps = float(fps)
        # phase-12 (14-dora §2.5): observers of the deduplicated frame, invoked AFTER the
        # seq dedup and BEFORE the colour conversion with the frame REFERENCE only (the
        # dora CameraTap drops it into a slot; the copy happens on the dora-bus thread).
        # A tap must never block or raise (exceptions are logged and do not stop the encode).
        self.taps: list[Callable[[CameraFrame], None]] = []
        self._last_seq: int | None = None
        self._idle_ticks = 0  # consecutive ticks without a new frame (see _await_new_frame)
        self._thread: threading.Thread | None = None
        self._running = False
        self._stop_event = threading.Event()  # wakes the pacing sleep so stop() is prompt

    def set_fps(self, fps: float) -> None:
        self._fps = float(fps)

    @property
    def fps(self) -> float:
        return self._fps

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"encoder-{self.stream_id}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _await_new_frame(self, deadline: float, period: float) -> CameraFrame | None:
        """The source's newest frame whose seq differs from the last encoded one, or
        ``None`` once ``deadline`` (the end of this tick's period) passes without one.
        Polls in short steps while the source is live; after ``fps`` empty ticks it
        polls once per tick (a disconnected camera must not spin the thread)."""
        step = max(_POLL_STEP_MIN_S, period / _POLLS_PER_PERIOD)
        idle_max = max(1, int(self._fps))
        while self._running:
            frame = self.source.latest()
            if frame is not None and frame.seq != self._last_seq:
                self._idle_ticks = 0
                return frame
            remaining = deadline - time.monotonic()
            if remaining <= 0.0 or self._idle_ticks >= idle_max:
                break
            self._stop_event.wait(min(step, remaining))  # returns at once on stop()
        self._idle_ticks += 1
        return None

    def _run(self) -> None:
        next_t = time.monotonic()
        params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        while self._running:
            period = 1.0 / self._fps
            try:
                frame = self._await_new_frame(next_t + period, period)
                if frame is not None:
                    next_t = time.monotonic()  # phase-lock the grid to the producer
                    self._last_seq = frame.seq
                    for tap in self.taps:
                        try:
                            tap(frame)
                        except Exception:  # noqa: BLE001 - a tap never breaks the encode
                            logger.exception("tap on stream %s failed", self.stream_id)
                    ok, jpeg = cv2.imencode(
                        ".jpg", cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR), params
                    )
                    if ok:
                        self.slot.put(pack_frame(frame.t_mono, jpeg.tobytes()))
            except Exception:
                logger.exception("encoder %s failed on a frame", self.stream_id)
            next_t += period
            lag = time.monotonic() - next_t
            if lag > period:
                next_t = time.monotonic()
            elif lag < 0.0:
                self._stop_event.wait(-lag)  # returns at once when stop() is called


class VideoHub:
    """stream_id -> (source, encoder, encoded slot); the video registry."""

    def __init__(self, bus, jpeg_quality: int = 80) -> None:
        self._bus = bus  # RuntimeBus (owns the encoded[stream] slots)
        self._jpeg_quality = int(jpeg_quality)
        self._workers: dict[str, EncoderWorker] = {}
        self._lock = threading.Lock()
        # phase-12: ``stream_hooks(stream_id, worker)`` run on every add_stream (before the
        # worker starts) so a process-lifetime observer (the dora bridge) can attach a tap
        # to streams created later - session cameras, hardware previews.
        self.stream_hooks: list[Callable[[str, EncoderWorker], None]] = []

    def ids(self) -> list[str]:
        with self._lock:
            return list(self._workers)

    def has(self, stream_id: str) -> bool:
        with self._lock:
            return stream_id in self._workers

    def slot(self, stream_id: str) -> LatestSlot[bytes] | None:
        with self._lock:
            if stream_id not in self._workers:
                return None
        return self._bus.encoded_slot(stream_id)

    def add_stream(self, stream_id: str, source, fps: float) -> None:
        with self._lock:
            if stream_id in self._workers:
                raise ValueError(f"duplicate stream {stream_id!r}")
            worker = EncoderWorker(
                stream_id,
                source,
                self._bus.encoded_slot(stream_id),
                fps,
                self._jpeg_quality,
            )
            self._workers[stream_id] = worker
            hooks = list(self.stream_hooks)
        for hook in hooks:
            try:
                hook(stream_id, worker)
            except Exception:  # noqa: BLE001 - a hook never blocks a stream
                logger.exception("stream hook failed for %s", stream_id)
        worker.start()

    def add_tap(self, stream_id: str, tap: Callable[[CameraFrame], None]) -> bool:
        """Attach ``tap`` to an existing stream (phase-12); False when unknown."""
        with self._lock:
            worker = self._workers.get(stream_id)
        if worker is None:
            return False
        worker.taps.append(tap)
        return True

    def remove_stream(self, stream_id: str) -> None:
        with self._lock:
            worker = self._workers.pop(stream_id, None)
        if worker is not None:
            worker.stop()

    def set_fps(self, stream_id: str, fps: float) -> None:
        with self._lock:
            worker = self._workers.get(stream_id)
        if worker is not None:
            worker.set_fps(fps)

    def stop(self) -> None:
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for worker in workers:
            worker.stop()


__all__ = ["EncoderWorker", "VideoHub"]
