"""Microphone reader (phase-11; 04-runtime §13.3): PulseAudio capture -> ``LatestSlot``.

Mirrors :class:`~apollo_mavis_v2_runtime.devices.tracker.TrackerReader`: a
daemon thread owned by ``Runtime`` for the process lifetime publishes one
:class:`MicFrame` per telemetry tick (frame length = ``sample_rate /
frame_hz``; 1920 samples at 48 kHz / 25 Hz) carrying peak / RMS (dBFS), a
clipping flag and a ``bins``-point int8 min/max envelope for the UI's
scrolling oscilloscope. The envelope is quantised RELATIVE TO THE FRAME PEAK
(the peak sample maps to +-127) so a quiet room at -58 dBFS still has shape;
absolute values are ``env / 127 * 10 ** (peak_dbfs / 20)``. ``status(now)``
derives ``stalled`` from the age of the last frame exactly like the tracker
derives ``stale``.

Capture ALWAYS goes through PulseAudio (the lab's PulseAudio 15.99 owns the
RØDE NT-USB Mini; opening ``hw:CARD=Mini`` fails with EBUSY and silently
stalls every other Pulse client). Backend order for ``auto``:

1. ``sounddevice`` -- ``InputStream(device="pulse", samplerate, channels=1,
   dtype="float32", blocksize=frame)`` over PortAudio's ALSA ``pulse`` plugin;
   the Pulse source is pinned with ``PULSE_SOURCE`` before the stream opens
   (PortAudio has no Pulse host API, so this is the only way to pick it).
2. ``parec`` subprocess -- ``parec -d <source> --format=s16le --rate=<sr>
   --channels=1 --raw --latency-msec=50`` read from a pipe.
3. ``fake`` -- amplitude-modulated 440 Hz sine synthesized at ``frame_hz``
   (tests; no audio hardware).

``sounddevice`` is imported lazily and ONLY here (ruff banned-api elsewhere
plus an AST guard test, same as ``pysurvive``). A missing module means status
``no_backend`` (explicit ``sounddevice`` backend) or a fall-through to
``parec`` (``auto``); when neither route exists ``auto`` also ends in
``no_backend`` -- never a crash.

Status machine (``MicStatus``): ``no_backend`` (disabled / nothing to capture
with) -> ``starting`` (resolving the source, opening) -> ``live`` (frames
arrive) / ``stalled`` (device present, no frame for > ``stale_s``) /
``absent`` (no Pulse source matches ``source_match``; presence is re-probed
at 1 Hz) / ``error`` (open or read failure; re-opened with 0.5 -> 5 s
backoff). Unplugging is detected by the 1 Hz presence probe plus a source
index/name check, because Pulse migrates a capture stream to the fallback
source instead of failing it.
"""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import re
import select
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from apollo_mavis_v2_core import LatestSlot
from apollo_mavis_v2_core.protocol import MicrophoneInfo, MicrophoneTelemetry, MicStatus

from ..config import MicrophoneConfig

logger = logging.getLogger(__name__)

MicKind = Literal["pulse", "fake", "none"]

PRESENCE_POLL_S = 1.0  # re-list Pulse sources this often (absent detection / unplug)
OPEN_BACKOFF_S = (0.5, 5.0)  # (first, max) delay before re-opening after an error
OPEN_BACKOFF_RESET_S = 30.0  # a capture run longer than this resets the backoff
RATE_WINDOW_S = 1.0  # rate_hz is measured over this window; decays to 0 when frames stop
READ_TIMEOUT_FACTOR = 2.0  # one read waits stale_s * this before counting as a stall
STALL_REOPEN_S = 3.0  # no audio for this long -> close and re-open (Pulse stalls silently)
STOP_JOIN_TIMEOUT_S = 5.0
PACTL_TIMEOUT_S = 3.0
PAREC_LATENCY_MSEC = 50
CLIP_DBFS = -1.0  # peak at/above this -> clipping
DBFS_FLOOR = -120.0  # dBFS of digital silence
INT8_MAX = 127
FAKE_CARRIER_HZ = 440.0
FAKE_MOD_HZ = 0.5  # amplitude modulation period 2 s
FAKE_AMP_MIN = 0.05
FAKE_AMP_MAX = 0.5


class NoBackendError(RuntimeError):
    """No capture route is available at all (module / binary missing)."""


@dataclass(frozen=True)
class MicFrame:
    """One analysed audio frame (``frame_len`` samples); published latest-wins."""

    seq: int
    rx_mono: float  # time.monotonic() when the frame was complete
    rms_dbfs: float
    peak_dbfs: float
    clipping: bool
    env_min: tuple[int, ...]  # bins x int8 (-127..127) relative to the frame peak, time-ordered
    env_max: tuple[int, ...]
    # additive (phase-12; 14-dora §4.2 "Microphone"): the frame's float32 mono PCM samples
    # in [-1, 1] (frame_len = sample_rate / frame_hz); None only for producers that never
    # retained them. Read-only by convention (published latest-wins).
    samples: np.ndarray | None = None


@dataclass(frozen=True)
class PulseSource:
    """One row of ``pactl list sources`` (monitors included; filtered by the matcher)."""

    index: int
    name: str
    description: str = ""
    properties: dict[str, str] = field(default_factory=dict)

    @property
    def is_monitor(self) -> bool:
        return self.properties.get("device.class") == "monitor" or self.name.endswith(".monitor")


@dataclass(frozen=True)
class MicrophoneDeviceStatus:
    """Device-side fields (available without a session); feeds both
    ``GET /api/microphones`` (:func:`to_info`) and telemetry (:func:`to_telemetry`)."""

    mic_id: str
    label: str
    backend: str  # configured backend
    kind: MicKind  # capture route actually in use
    status: MicStatus
    detail: str
    source: str | None  # resolved Pulse source name
    sample_rate: int
    channels: int
    seq: int
    rate_hz: float
    age_s: float | None
    overruns: int
    last: MicFrame | None


# -- frame analysis -------------------------------------------------------------------
def dbfs(value: float) -> float:
    """Full-scale level of a linear amplitude (0 dBFS = |1.0|), floored at ``DBFS_FLOOR``."""
    if not value > 0.0 or not math.isfinite(value):
        return DBFS_FLOOR
    return max(DBFS_FLOOR, 20.0 * math.log10(value))


def frame_stats(
    samples: np.ndarray, bins: int
) -> tuple[float, float, bool, np.ndarray, np.ndarray]:
    """``(peak_dbfs, rms_dbfs, clipping, env_min, env_max)`` of one frame.

    The envelope splits the frame into ``bins`` contiguous time slices
    (``arange(bins) * n // bins`` edges, exact when ``n`` is a multiple of
    ``bins``) and keeps each slice's min and max, quantised to int8
    (-127..127) **relative to the frame peak** for the wire: the loudest
    sample of the frame maps to +-127 whatever its level, so the oscilloscope
    keeps its shape at -58 dBFS (a full-scale int8 would round a quiet room to
    all zeros). ``peak_dbfs`` carries the scale back: absolute = ``env / 127 *
    10 ** (peak_dbfs / 20)``. Frames shorter than ``bins`` are sampled; a
    digitally silent frame yields an all-zero envelope.
    """
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    n = int(x.shape[0])
    if n == 0:
        zeros = np.zeros(bins, dtype=np.int64)
        return DBFS_FLOOR, DBFS_FLOOR, False, zeros, zeros.copy()
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    if n >= bins:
        edges = (np.arange(bins) * n) // bins
        mn = np.minimum.reduceat(x, edges)
        mx = np.maximum.reduceat(x, edges)
    else:
        idx = (np.arange(bins) * n) // bins
        mn = mx = x[idx]
    scale = INT8_MAX / peak if peak > 0.0 else 0.0  # peak -> +-127 (relative envelope)
    env_min = np.clip(np.rint(mn * scale), -INT8_MAX, INT8_MAX).astype(np.int64)
    env_max = np.clip(np.rint(mx * scale), -INT8_MAX, INT8_MAX).astype(np.int64)
    peak_dbfs = dbfs(peak)
    return peak_dbfs, dbfs(rms), peak_dbfs >= CLIP_DBFS, env_min, env_max


# -- PulseAudio source discovery --------------------------------------------------------
def list_pulse_sources(timeout_s: float = PACTL_TIMEOUT_S) -> list[PulseSource]:
    """``pactl -f json list sources`` (falls back to ``pactl list sources short``
    when the JSON form is unavailable). Raises ``OSError`` when ``pactl`` itself
    cannot run (missing binary / no Pulse server) so the caller can report
    ``error`` and retry."""
    proc = subprocess.run(
        ["pactl", "-f", "json", "list", "sources"],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if proc.returncode == 0:
        try:
            rows = json.loads(proc.stdout or "[]")
        except json.JSONDecodeError:
            rows = None
        if isinstance(rows, list):
            out = []
            for row in rows:
                if not isinstance(row, dict) or "name" not in row:
                    continue
                props = row.get("properties") or {}
                out.append(
                    PulseSource(
                        index=int(row.get("index", -1)),
                        name=str(row["name"]),
                        description=str(row.get("description") or ""),
                        properties={str(k): str(v) for k, v in props.items()},
                    )
                )
            return out
    short = subprocess.run(
        ["pactl", "list", "sources", "short"],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if short.returncode != 0:
        raise OSError(
            f"pactl failed (rc {short.returncode}): "
            f"{(short.stderr or proc.stderr or '').strip()[:200]}"
        )
    out = []
    for line in short.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].strip().isdigit():
            out.append(PulseSource(index=int(parts[0]), name=parts[1].strip()))
    return out


_NORM_RE = re.compile(r"[\s_\-]+")


def _norm(text: str) -> str:
    return _NORM_RE.sub(" ", text.lower()).strip()


def match_source(sources: list[PulseSource], needle: str) -> PulseSource | None:
    """First non-monitor source whose name / description / product name
    contains ``needle`` (case-insensitive; ``_``, ``-`` and whitespace are
    interchangeable, so ``"NT-USB Mini"`` matches
    ``alsa_input.usb-R__DE_..._NT-USB_Mini_750BFEE8-00.mono-fallback`` even when
    pactl blanks the non-ASCII description to ``(null)``)."""
    want = _norm(needle)
    if not want:
        return None
    for src in sources:
        if src.is_monitor:
            continue
        props = src.properties
        hay = " | ".join(
            [
                src.name,
                src.description,
                props.get("device.product.name", ""),
                props.get("device.description", ""),
                props.get("alsa.card_name", ""),
                props.get("alsa.long_card_name", ""),
            ]
        )
        if want in _norm(hay):
            return src
    return None


def parec_argv(source: str, sample_rate: int) -> list[str]:
    return [
        "parec",
        "-d",
        source,
        "--format=s16le",
        f"--rate={int(sample_rate)}",
        "--channels=1",
        "--raw",
        f"--latency-msec={PAREC_LATENCY_MSEC}",
    ]


# -- capture routes ---------------------------------------------------------------------
class _SounddeviceCapture:
    """PortAudio ``InputStream`` on the ALSA ``pulse`` device; the callback only
    copies the block into a queue (it runs on PortAudio's thread)."""

    def __init__(self, sd, source: str, sample_rate: int, frame_len: int) -> None:
        os.environ["PULSE_SOURCE"] = source  # read by the ALSA pulse plugin at PCM open
        self.overruns = 0
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)

        def callback(indata, frames, time_info, status) -> None:
            if status:
                self.overruns += 1
            try:
                self._q.put_nowait(np.array(indata[:, 0], dtype=np.float32, copy=True))
            except queue.Full:
                self.overruns += 1

        self._stream = sd.InputStream(
            device="pulse",
            samplerate=int(sample_rate),
            channels=1,
            dtype="float32",
            blocksize=int(frame_len),
            callback=callback,
        )
        self._stream.start()

    def read(self, timeout: float) -> np.ndarray | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        for op in (self._stream.stop, self._stream.close):
            try:
                op()
            except Exception:  # noqa: BLE001 - closing a dead PortAudio stream
                pass


class _ParecCapture:
    """``parec`` child process; s16le mono frames read from its stdout pipe with
    ``select`` so a silent Pulse stall surfaces as a timeout, not a hang."""

    def __init__(self, argv: list[str], frame_len: int) -> None:
        self.overruns = 0
        self._proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            bufsize=0,
        )
        assert self._proc.stdout is not None
        self._fd = self._proc.stdout.fileno()
        self._nbytes = int(frame_len) * 2
        self._buf = bytearray()

    def read(self, timeout: float) -> np.ndarray | None:
        deadline = time.monotonic() + timeout
        while len(self._buf) < self._nbytes:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return None
            ready, _, _ = select.select([self._fd], [], [], remaining)
            if not ready:
                return None
            chunk = os.read(self._fd, self._nbytes - len(self._buf))
            if not chunk:
                raise RuntimeError(f"parec exited (rc {self._proc.poll()})")
            self._buf += chunk
        data = bytes(self._buf[: self._nbytes])
        del self._buf[: self._nbytes]
        return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=1.0)
        try:
            self._proc.stdout.close()  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass


class MicrophoneReader:
    """Owns the capture thread for the process lifetime; publishes to ``slot``.

    ``start()`` is a no-op when the microphone is disabled, for backend
    ``none`` and while a thread is alive; ``stop()`` joins the thread (the
    stream / child process is closed on the thread). ``status(now)`` derives
    ``age_s`` / ``stalled`` from the receive time of the last frame
    (``cfg.stale_s``) and a ``rate_hz`` over the last ``RATE_WINDOW_S``; every
    other field is set by the capture thread under ``_lock``.

    Test seams: ``source_lister`` replaces ``pactl`` (returns
    :class:`PulseSource` rows), ``parec_command`` replaces the ``parec``
    command line, ``import_sounddevice`` replaces the lazy import.
    """

    def __init__(
        self,
        cfg: MicrophoneConfig,
        slot: LatestSlot[MicFrame],
        *,
        frame_hz: float = 25.0,
        clock: Callable[[], float] = time.monotonic,
        source_lister: Callable[[], list[PulseSource]] | None = None,
        parec_command: Callable[[str, int], list[str]] | None = None,
        import_sounddevice: Callable[[], object] | None = None,
    ) -> None:
        self.cfg = cfg
        self.slot = slot
        self.frame_hz = float(frame_hz)
        self.frame_len = max(int(self.cfg.bins), int(round(cfg.sample_rate / self.frame_hz)))
        self._clock = clock
        self._source_lister = source_lister or list_pulse_sources
        self._parec_argv = parec_command or parec_argv
        self._import_sounddevice = import_sounddevice or self._default_import_sounddevice
        self._lock = threading.Lock()
        self._status: MicStatus = "no_backend"
        if not cfg.enabled:
            self._detail = "microphone disabled (enabled: false)"
        elif cfg.backend == "none":
            self._detail = "microphone disabled (backend: none)"
        else:
            self._detail = ""
        self._kind: MicKind = (
            "none"
            if (not cfg.enabled or cfg.backend == "none")
            else ("fake" if cfg.backend == "fake" else "pulse")
        )
        self._source: str | None = None
        self._seq = 0
        self._sinks: list[Callable[[np.ndarray, float], None]] = []  # raw-block sinks
        self._last: MicFrame | None = None
        self._rx_times: deque[float] = deque(maxlen=256)
        self._overruns = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lifecycle = threading.RLock()
        # phase-12 (14-dora §2.5): observers invoked with every published MicFrame on the
        # capture thread (the dora MicTap); never block, never raise.
        self.taps: list[Callable[[MicFrame], None]] = []

    # -- lifecycle -----------------------------------------------------------------------
    @property
    def active(self) -> bool:
        """Capture configured (enabled and a backend other than ``none``)."""
        return self.cfg.enabled and self.cfg.backend != "none"

    def start(self) -> None:
        with self._lifecycle:
            if not self.active or self._thread is not None:
                return
            self._stop.clear()
            self._set_status("starting", "")
            target = self._run_fake if self.cfg.backend == "fake" else self._run_pulse
            self._thread = threading.Thread(
                target=self._thread_main, args=(target,), name="microphone-reader", daemon=True
            )
            self._thread.start()

    def _thread_main(self, target: Callable[[], None]) -> None:
        try:
            target()
        except Exception:  # last line of defence: the thread must end cleanly
            logger.exception("microphone reader thread crashed")
            self._set_status("error", "microphone reader crashed (see log)")
        finally:
            if self._thread is threading.current_thread():
                self._thread = None

    def stop(self, timeout: float = STOP_JOIN_TIMEOUT_S) -> bool:
        with self._lifecycle:
            self._stop.set()
            thread = self._thread
            if thread is None:
                return True
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("microphone reader thread did not stop within %.1f s", timeout)
                return False
            self._thread = None
            return True

    # -- status ----------------------------------------------------------------------------
    def _set_status(self, status: MicStatus, detail: str) -> None:
        with self._lock:
            self._status = status
            self._detail = detail

    def _set_route(self, kind: MicKind, source: str | None) -> None:
        with self._lock:
            self._kind = kind
            self._source = source

    def status(self, now: float | None = None) -> MicrophoneDeviceStatus:
        now = self._clock() if now is None else now
        with self._lock:
            status, detail, last = self._status, self._detail, self._last
            kind, source, overruns = self._kind, self._source, self._overruns
            rx = [t for t in self._rx_times if t > now - RATE_WINDOW_S]
        age = None if last is None else max(0.0, now - last.rx_mono)
        if status == "live" and age is not None and age > self.cfg.stale_s:
            status = "stalled"
        rate = (len(rx) - 1) / (now - rx[0]) if len(rx) > 1 and now > rx[0] else 0.0
        return MicrophoneDeviceStatus(
            mic_id=self.cfg.mic_id,
            label=self.cfg.label,
            backend=self.cfg.backend,
            kind=kind,
            status=status,
            detail=detail,
            source=source,
            sample_rate=self.cfg.sample_rate,
            channels=1,
            seq=last.seq if last is not None else 0,
            rate_hz=float(rate),
            age_s=age,
            overruns=overruns,
            last=last,
        )

    # -- raw-sample sinks (recorder audio sidecars, 04-runtime §10.5) ----------------------
    def add_sink(self, fn: Callable[[np.ndarray, float], None]) -> None:
        """Register ``fn(samples, rx_mono)`` for every captured block (called on the
        capture thread; the float32 mono block is the reader's own array — copy it).
        Telemetry keeps publishing stats only; sinks are how an episode recorder
        gets the raw audio without a second Pulse client."""
        with self._lock:
            if fn not in self._sinks:
                self._sinks = [*self._sinks, fn]

    def remove_sink(self, fn: Callable[[np.ndarray, float], None]) -> None:
        with self._lock:
            self._sinks = [s for s in self._sinks if s is not fn]

    # -- publishing (capture thread) -------------------------------------------------------
    def _publish(self, samples: np.ndarray, rx: float) -> MicFrame:
        peak_dbfs, rms_dbfs, clipping, env_min, env_max = frame_stats(samples, self.cfg.bins)
        with self._lock:
            self._seq += 1
            frame = MicFrame(
                seq=self._seq,
                rx_mono=rx,
                rms_dbfs=rms_dbfs,
                peak_dbfs=peak_dbfs,
                clipping=clipping,
                env_min=tuple(env_min.tolist()),
                env_max=tuple(env_max.tolist()),
                samples=np.ascontiguousarray(samples, dtype=np.float32),
            )
            self._last = frame
            self._rx_times.append(rx)
            if self._status != "live":
                self._status, self._detail = "live", ""
            sinks = self._sinks
        self.slot.put(frame)
        for fn in sinks:  # never let a sink break the capture loop
            try:
                fn(samples, rx)
            except Exception:  # noqa: BLE001
                logger.exception("microphone sink failed; removing it")
                self.remove_sink(fn)
        for tap in self.taps:
            try:
                tap(frame)
            except Exception:  # noqa: BLE001 - a tap never breaks capture
                logger.exception("microphone tap failed")
        return frame

    # -- fake backend: amplitude-modulated sine at frame_hz ----------------------------------
    def _run_fake(self) -> None:
        self._set_route("fake", None)
        period = 1.0 / self.frame_hz
        n, sr = self.frame_len, float(self.cfg.sample_rate)
        idx = 0
        next_t = self._clock()
        while not self._stop.is_set():
            t = (np.arange(n, dtype=np.float64) + idx) / sr
            amp = FAKE_AMP_MIN + (FAKE_AMP_MAX - FAKE_AMP_MIN) * 0.5 * (
                1.0 + np.sin(2.0 * math.pi * FAKE_MOD_HZ * t)
            )
            x = (amp * np.sin(2.0 * math.pi * FAKE_CARRIER_HZ * t)).astype(np.float32)
            idx += n
            self._publish(x, self._clock())
            next_t += period
            delay = next_t - self._clock()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_t = self._clock()

    # -- Pulse backends ----------------------------------------------------------------------
    @staticmethod
    def _default_import_sounddevice():
        import sounddevice  # lazy; the ONLY import site (ruff banned-api elsewhere)

        return sounddevice

    def _open_stream(self, source: PulseSource):
        """Open the first working route for ``cfg.backend``. Raises
        :class:`NoBackendError` when every candidate route is MISSING (module /
        binary), any other exception when a present route failed to open (the
        caller retries with backoff)."""
        backend = self.cfg.backend
        missing: list[str] = []
        failures: list[str] = []
        if backend in ("auto", "sounddevice"):
            try:
                sd = self._import_sounddevice()
            except Exception as e:  # ImportError, or OSError when libportaudio is missing
                sd = None
                msg = f"sounddevice unavailable ({e.__class__.__name__}: {e})"
                if backend == "sounddevice":
                    raise NoBackendError(msg) from e
                missing.append(msg)
            if sd is not None:
                try:
                    return _SounddeviceCapture(
                        sd, source.name, self.cfg.sample_rate, self.frame_len
                    )
                except Exception as e:
                    if backend == "sounddevice":
                        raise
                    failures.append(f"sounddevice: {e!r}")
        if backend in ("auto", "parec"):
            argv = self._parec_argv(source.name, self.cfg.sample_rate)
            if shutil.which(argv[0]) is None:
                msg = f"{argv[0]} not found"
                if backend == "parec":
                    raise NoBackendError(msg)
                missing.append(msg)
            else:
                try:
                    return _ParecCapture(argv, self.frame_len)
                except Exception as e:
                    if backend == "parec":
                        raise
                    failures.append(f"parec: {e!r}")
        if failures:
            raise RuntimeError("; ".join(failures + missing))
        raise NoBackendError("; ".join(missing) or "no capture backend")

    def _absent_detail(self) -> str:
        return f"no PulseAudio source matches {self.cfg.source_match!r} (microphone unplugged?)"

    def _run_pulse(self) -> None:
        backoff = OPEN_BACKOFF_S[0]
        while not self._stop.is_set():
            try:
                sources = self._source_lister()
            except Exception as e:
                self._set_route("pulse", None)
                self._set_status("error", f"pactl failed: {e}")
                self._stop.wait(backoff)
                backoff = min(backoff * 2.0, OPEN_BACKOFF_S[1])
                continue
            src = match_source(sources, self.cfg.source_match)
            if src is None:
                self._set_route("pulse", None)
                self._set_status("absent", self._absent_detail())
                self._stop.wait(PRESENCE_POLL_S)
                continue
            self._set_route("pulse", src.name)
            self._set_status("starting", f"opening {src.name}")
            try:
                stream = self._open_stream(src)
            except NoBackendError as e:
                self._set_route("none", src.name)
                self._set_status("no_backend", str(e))
                return
            except Exception as e:
                self._set_status("error", f"open failed: {e!r}")
                self._stop.wait(backoff)
                backoff = min(backoff * 2.0, OPEN_BACKOFF_S[1])
                continue
            opened_at = self._clock()
            outcome = "error"
            try:
                outcome = self._capture(stream, src)
            except Exception as e:
                logger.warning("microphone capture failed: %r", e)
                self._set_status("error", f"capture failed: {e!r}")
            finally:
                stream.close()
                with self._lock:
                    self._overruns += int(getattr(stream, "overruns", 0))
            if self._clock() - opened_at > OPEN_BACKOFF_RESET_S:
                backoff = OPEN_BACKOFF_S[0]
            if outcome == "error" and not self._stop.is_set():
                self._stop.wait(backoff)
                backoff = min(backoff * 2.0, OPEN_BACKOFF_S[1])

    def _capture(self, stream, src: PulseSource) -> str:
        """Read frames until stop / absent / source change / stall. Returns
        ``"stopped"``, ``"absent"``, ``"changed"`` or ``"error"``."""
        read_timeout = self.cfg.stale_s * READ_TIMEOUT_FACTOR
        next_presence = self._clock() + PRESENCE_POLL_S
        stall_since: float | None = None
        while not self._stop.is_set():
            block = stream.read(read_timeout)
            now = self._clock()
            if block is None:
                stall_since = now if stall_since is None else stall_since
                if now - stall_since > STALL_REOPEN_S:
                    self._set_status(
                        "error",
                        f"no audio from {src.name} for {now - stall_since:.1f} s "
                        "(source held by a direct ALSA client?)",
                    )
                    return "error"
            else:
                stall_since = None
                self._publish(block, now)
            if now >= next_presence:
                next_presence = now + PRESENCE_POLL_S
                try:
                    current = match_source(self._source_lister(), self.cfg.source_match)
                except Exception as e:  # one failed listing does not end the capture
                    logger.debug("pactl presence probe failed: %r", e)
                    continue
                if current is None:
                    self._set_route("pulse", None)
                    self._set_status("absent", self._absent_detail())
                    return "absent"
                if current.index != src.index or current.name != src.name:
                    self._set_status("starting", "source changed; reopening")
                    return "changed"
        return "stopped"


# -- wire conversions (REST + telemetry share one status) ----------------------------------
def to_info(st: MicrophoneDeviceStatus) -> MicrophoneInfo:
    return MicrophoneInfo(
        mic_id=st.mic_id,
        label=st.label,
        kind=st.kind,
        source=st.source,
        sample_rate=st.sample_rate,
        channels=st.channels,
        live=st.status == "live",
        status=st.status,
        detail=st.detail,
    )


def to_telemetry(st: MicrophoneDeviceStatus) -> MicrophoneTelemetry:
    last = st.last
    return MicrophoneTelemetry(
        mic_id=st.mic_id,
        status=st.status,
        detail=st.detail,
        seq=st.seq,
        age_s=st.age_s,
        rate_hz=st.rate_hz,
        sample_rate=st.sample_rate,
        rms_dbfs=last.rms_dbfs if last is not None else None,
        peak_dbfs=last.peak_dbfs if last is not None else None,
        clipping=last.clipping if last is not None else False,
        env_min=list(last.env_min) if last is not None else [],
        env_max=list(last.env_max) if last is not None else [],
        overruns=st.overruns,
    )


__all__ = [
    "MicKind",
    "MicFrame",
    "PulseSource",
    "MicrophoneDeviceStatus",
    "MicrophoneReader",
    "NoBackendError",
    "dbfs",
    "frame_stats",
    "list_pulse_sources",
    "match_source",
    "parec_argv",
    "to_info",
    "to_telemetry",
]
