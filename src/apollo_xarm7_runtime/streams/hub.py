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

import cv2
from apollo_xarm7_core import LatestSlot
from apollo_xarm7_core.protocol import pack_frame

logger = logging.getLogger(__name__)


class EncoderWorker:
    """Paces one stream at its own fps; latest-wins, drops on no-new-frame."""

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
        self._last_seq: int | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def set_fps(self, fps: float) -> None:
        self._fps = float(fps)

    @property
    def fps(self) -> float:
        return self._fps

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name=f"encoder-{self.stream_id}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        next_t = time.monotonic()
        params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        while self._running:
            period = 1.0 / self._fps
            try:
                frame = self.source.latest()
                if frame is not None and frame.seq != self._last_seq:
                    self._last_seq = frame.seq
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
                time.sleep(-lag)


class VideoHub:
    """stream_id -> (source, encoder, encoded slot); the video registry."""

    def __init__(self, bus, jpeg_quality: int = 80) -> None:
        self._bus = bus  # RuntimeBus (owns the encoded[stream] slots)
        self._jpeg_quality = int(jpeg_quality)
        self._workers: dict[str, EncoderWorker] = {}
        self._lock = threading.Lock()

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
                stream_id, source, self._bus.encoded_slot(stream_id), fps,
                self._jpeg_quality,
            )
            self._workers[stream_id] = worker
        worker.start()

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
