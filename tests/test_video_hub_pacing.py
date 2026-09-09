"""EncoderWorker pacing (04-runtime §13.4; 14-dora §7 promises the fixed-viewpoint
consumer ``preview_fps`` with contiguous seqs).

The sim ``RenderService`` renders every preview camera at ``preview_fps`` and
``start_previews`` adds the encoder streams milliseconds later, so the producer's and
the encoder's 1/fps grids start nearly coincident. Sampling the depth-1 ``latest()``
slot ONCE per fixed grid tick then lets render-time jitter decide each poll: a miss is
a lost output period (2026-09-08: ``cam_view_wrist_cam`` published at 11.9 Hz over
48 contiguous seqs; a phase sweep of the old loop gave 10.9-13.6 Hz within +/-6 ms of
the coincident phase and 15.0 Hz elsewhere). The worker now waits INSIDE its period
for the next distinct seq and re-anchors the grid on it; ``fps`` stays a cap.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from apollo_mavis_v2_core import CameraFrame, LatestSlot

from apollo_mavis_v2_runtime.streams.hub import EncoderWorker

FPS = 15.0
T = 1.0 / FPS
IMG = np.zeros((24, 32, 3), np.uint8)


def _frame(seq: int) -> CameraFrame:
    now = time.monotonic()
    return CameraFrame(camera_id="c", seq=seq, t_mono=now, wallclock_ns=time.time_ns(), rgb=IMG)


class GridProducer:
    """A new frame every ``period`` (+/- ``jitter_s``, uniform) on its own thread, like
    the RenderService's per-stream grid. ``latest()`` is a depth-1 slot."""

    def __init__(self, period: float, jitter_s: float = 0.0, seed: int = 0) -> None:
        self.period = period
        self.jitter_s = jitter_s
        self.produced = 0
        self.polls = 0
        self._slot: LatestSlot[CameraFrame] = LatestSlot()
        self._stop = threading.Event()
        self._rng = np.random.default_rng(seed)
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(2.0)

    def latest(self) -> CameraFrame | None:
        self.polls += 1
        got = self._slot.get()
        return None if got is None else got[0]

    def _run(self) -> None:
        next_t = time.monotonic()
        while not self._stop.is_set():
            j = self._rng.uniform(-self.jitter_s, self.jitter_s) if self.jitter_s else 0.0
            self._stop.wait(max(0.0, next_t + j - time.monotonic()))
            if self._stop.is_set():
                return
            self.produced += 1
            self._slot.put(_frame(self.produced))
            next_t += self.period
            if next_t < time.monotonic() - self.period:
                next_t = time.monotonic() + self.period


def _run_worker(source, fps: float, window_s: float, warmup_s: float = 0.4):
    """Encoded-frame arrival times + ages (ms) seen by a tap over ``window_s``."""
    slot: LatestSlot[bytes] = LatestSlot()
    worker = EncoderWorker("c", source, slot, fps)
    arrivals: list[float] = []
    ages_ms: list[float] = []

    def tap(fr: CameraFrame) -> None:
        now = time.monotonic()
        arrivals.append(now)
        ages_ms.append((now - fr.t_mono) * 1e3)

    worker.taps.append(tap)
    worker.start()
    time.sleep(warmup_s)
    n0 = len(arrivals)
    time.sleep(window_s)
    n1 = len(arrivals)
    worker.stop()
    return arrivals[n0:n1], ages_ms[n0:n1]


def _rate(arrivals: list[float]) -> float:
    return (len(arrivals) - 1) / (arrivals[-1] - arrivals[0]) if len(arrivals) > 1 else 0.0


@pytest.mark.perf
def test_equal_rate_source_is_followed_frame_for_frame_at_the_coincident_phase():
    # producer grid anchored at the worker's start (the preview hand-over case) with the
    # render-time jitter that lost 5-20 % of the output periods under the fixed-grid poll
    prod = GridProducer(T, jitter_s=0.008)
    prod.start()
    try:
        arrivals, ages = _run_worker(prod, FPS, window_s=3.0)
        rate = _rate(arrivals)
        gaps = np.diff(arrivals)
        assert abs(rate - FPS) <= 1.0, (rate, len(arrivals))
        # no output period lost: the worst inter-arrival stays well under two periods
        assert float(gaps.max()) < 1.6 * T, (float(gaps.max()) * 1e3, rate)
    finally:
        prod.stop()


@pytest.mark.perf
def test_faster_source_is_capped_at_fps_and_slower_source_is_encoded_once_per_frame():
    fast = GridProducer(T / 4)  # 60 Hz source, 15 fps cap
    fast.start()
    try:
        arrivals, _ = _run_worker(fast, FPS, window_s=2.0)
        assert _rate(arrivals) <= FPS * 1.1, _rate(arrivals)
        assert _rate(arrivals) >= FPS * 0.9, _rate(arrivals)
    finally:
        fast.stop()
    slow = GridProducer(T * 3)  # 5 Hz source: every frame once, none twice
    slow.start()
    try:
        arrivals, _ = _run_worker(slow, FPS, window_s=2.0)
        assert abs(_rate(arrivals) - FPS / 3) <= 0.5, _rate(arrivals)
    finally:
        slow.stop()


def test_frame_landing_just_after_a_poll_is_picked_up_within_the_same_tick():
    """Deterministic latency check: a frame lands 15 ms after a tick's first (empty) poll.
    The fixed-grid poll saw it only at the NEXT grid tick (age ~= period - 15 ms = 52 ms);
    the intra-tick wait picks it up within one poll step (period / 8 = 8.3 ms)."""

    class ReactiveSource:
        def __init__(self) -> None:
            self.seq = 0
            self._pending: CameraFrame | None = None
            self._lock = threading.Lock()
            self._timer: threading.Timer | None = None

        def _emit(self) -> None:
            with self._lock:
                self.seq += 1
                self._pending = _frame(self.seq)
                self._timer = None

        def latest(self) -> CameraFrame | None:
            with self._lock:
                if self._pending is None and self._timer is None:  # first empty poll
                    self._timer = threading.Timer(0.015, self._emit)
                    self._timer.daemon = True
                    self._timer.start()
                return self._pending

        def consume(self) -> None:
            with self._lock:
                self._pending = None

        def close(self) -> None:
            with self._lock:
                if self._timer is not None:
                    self._timer.cancel()

    src = ReactiveSource()
    slot: LatestSlot[bytes] = LatestSlot()
    worker = EncoderWorker("c", src, slot, FPS)
    ages_ms: list[float] = []

    def tap(fr: CameraFrame) -> None:
        ages_ms.append((time.monotonic() - fr.t_mono) * 1e3)
        src.consume()  # the next frame lands 15 ms after the next tick's first poll

    worker.taps.append(tap)
    worker.start()
    time.sleep(1.5)
    worker.stop()
    src.close()
    assert len(ages_ms) >= 10, ages_ms
    med = float(np.median(ages_ms[2:]))
    assert med < (T / 2) * 1e3, (med, ages_ms)  # not a full period behind


def test_idle_source_falls_back_to_one_poll_per_tick():
    """A stalled camera (no frame for ~1 s) must not spin the encoder thread."""

    class Stalled:
        def __init__(self) -> None:
            self.polls = 0

        def latest(self) -> None:
            self.polls += 1
            return None

    src = Stalled()
    worker = EncoderWorker("c", src, LatestSlot(), FPS)
    worker.start()
    time.sleep(1.5)  # > fps idle ticks: fine polling has stopped
    p0 = src.polls
    time.sleep(1.0)
    p1 = src.polls
    worker.stop()
    assert (p1 - p0) <= FPS * 1.5, p1 - p0  # ~one poll per tick, not eight
