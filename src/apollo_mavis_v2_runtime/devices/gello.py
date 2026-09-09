"""GELLO leader-arm reader (16-gello §4, D2): Dynamixel positions -> ``LatestSlot``.

A daemon thread (``gello-reader``) polls the passive leader's servos and publishes
:class:`GelloSample` (7 raw joints in rad, the mapped joints ``sign * (raw - offset)``, the
gripper fraction) stamped with ``time.monotonic()`` on receipt — the tracker pattern
(``devices/tracker.py``), not a workcell member. Backends:

* ``dynamixel`` — protocol 2.0, ONE ``GroupSyncRead`` of Present Position (address 132,
  4 bytes) over ``joint_ids`` + ``gripper_id`` at ``poll_hz``; ``ticks -> rad`` as
  ``ticks * 2*pi / 4096`` (two's complement above ``0x7FFFFFFF``: multi-turn servos read
  negative). Baud: ``cfg.baud`` or an auto-scan (``GELLO_BAUD_SCAN``: one ``ping`` per
  CONFIGURED id per rate - ~40 ms each instead of the SDK's 0.8-1.4 s busy-polling
  broadcast ping - the first rate at which a configured servo answers wins and is logged;
  ``stop()`` is honoured between pings, 2026-09-09 review). Port: ``cfg.port`` or, when
  ``cfg.usb_serial`` is set, the ``ttyUSB*`` node whose sysfs USB parent carries that
  serial (:func:`tty_nodes_by_usb_serial`, the cameras' by-serial precedent). At connect
  the FTDI ``latency_timer`` is read from sysfs and a WARNING logged above 2 ms (16 ms caps
  eight servos at ~30 Hz; the udev rule of 16-gello §9.3 sets 1 ms). The reader NEVER
  enables torque and NEVER writes a goal — GELLO stays passive (a test pins that no
  ``write*`` / ``GroupSyncWrite`` call exists in this module). Serial / bus errors set
  status ``error`` and the loop restarts with backoff (0.5 -> 5 s, like the tracker).
* ``fake`` — publishes a scripted posture at ``poll_hz``; default the ``mavis_v2`` keyframe
  of the Manipulation Arm (``[pi, 0, 0, 0, 0, 0, 0]``, gripper 1.0) so a sim GELLO session
  launches already synced; :meth:`GelloReader.fake_set` moves it. The fake needs no
  calibration: ``q == q_raw`` (signs / offsets are NOT applied), every sample is valid.
* ``none`` — status ``no_backend``, no thread.

``dynamixel_sdk`` (and ``serial`` underneath it) are imported lazily and ONLY here (ruff
banned-api elsewhere + an AST test); a missing module means status ``no_backend`` with the
reason, never a crash.

Calibration (16-gello §4): ``joint_signs`` come from the config (operator-owned);
``joint_offsets_rad`` from the config when set, else from ``var/gello_calibration.json``
(:class:`~apollo_mavis_v2_runtime.gello.calibration.GelloCalibrationStore`, written by
``POST /api/gello/calibrate``; :meth:`GelloReader.reload_calibration` picks up a REST op).
A dynamixel sample is ``valid = False`` when any joint jumped more than ``max_jump_rad``
since the previous sample (a dropped byte / wrong id never becomes a target; ``jump`` says
so separately) or when the offsets are missing. ``fresh_sample(require_calibrated=False)``
hands out a fresh, non-jump but UNCALIBRATED sample - what ``match_arm`` (the op that
CREATES the calibration) and the gripper endpoint ops read (2026-09-09 review: before, the
first calibration of a real leader was a chicken-and-egg 409). ``gripper_frac =
clip((raw - closed) / (open - closed), 0, 1)`` from the two calibrated endpoints, None
until both exist.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from apollo_mavis_v2_core import LatestSlot

from ..config import GELLO_BAUD_SCAN, GelloConfig
from ..gello.calibration import (
    EMPTY_CALIBRATION,
    GelloCalibration,
    GelloCalibrationStore,
    gripper_frac,
)

logger = logging.getLogger(__name__)

GelloStatus = Literal["no_backend", "starting", "connected", "stale", "error"]

PROTOCOL_VERSION = 2.0
ADDR_PRESENT_POSITION = 132  # X-series control table: Present Position (4 bytes, signed)
LEN_PRESENT_POSITION = 4
TICKS_PER_REV = 4096
RAD_PER_TICK = 2.0 * math.pi / TICKS_PER_REV
FAKE_DEFAULT_Q: tuple[float, ...] = (math.pi, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # mavis_v2 keyframe
FAKE_DEFAULT_GRIPPER = 1.0
RESTART_BACKOFF_S = (0.5, 5.0)  # (first, max) delay before restarting a dead bus loop
RESTART_RESET_S = 30.0  # a run longer than this resets the backoff ladder
RATE_WINDOW_S = 1.0  # rate_hz over this window; decays to 0 when samples stop
STOP_JOIN_TIMEOUT_S = 5.0
# Upper bound of ONE per-id ping (dynamixel_sdk 4.0.5: ``setPacketTimeout(11)`` = 11 bytes at
# the rate + 2 x LATENCY_TIMER (16 ms) + 2 ms = 36-38 ms; the bound leaves room for the serial
# re-open a ``setBaudRate`` costs). ``stop()`` waits at least one worst-case scan
# (``scan_budget_s``) so the port is never abandoned open.
PING_TIMEOUT_S = 0.1
SCAN_REOPEN_S = 0.2  # per rate: setBaudRate closes and re-opens the device
READ_FAILS_BEFORE_RESTART = 20  # consecutive sync-read failures -> close + reopen the port
SYSFS_TTY = "/sys/class/tty"
LATENCY_TIMER_WARN_MS = 2  # FTDI latency_timer above this -> WARNING (16-gello §4)
SERIAL_WALK_DEPTH = 4  # sysfs ancestors of a ttyUSB device dir searched for `serial`


class GelloPortError(OSError):
    """The serial node could not be resolved / opened (retried with backoff)."""


class GelloBusError(RuntimeError):
    """No servo answered / the sync read kept failing (retried with backoff)."""


def ticks_to_rad(ticks: int) -> float:
    """Present Position (uint32 on the wire) -> rad; values above ``0x7FFFFFFF`` are the
    two's complement of a negative multi-turn count."""
    t = int(ticks) & 0xFFFFFFFF
    if t > 0x7FFFFFFF:
        t -= 0x1_0000_0000
    return t * RAD_PER_TICK


def _read_attr(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def tty_nodes_by_usb_serial(serial: str, sysfs_root: str | Path = SYSFS_TTY) -> list[str]:
    """``/dev/ttyUSB*`` nodes of the USB device with this ``serial``, in node order.

    Walks ``<sysfs_root>/ttyUSBn/device`` (the usb-serial port device; for ftdi_sio its
    parent is the USB interface ``<bus>-<port>:1.<iface>`` and the grandparent the USB
    device holding ``serial``) up to :data:`SERIAL_WALK_DEPTH` ancestors and matches the
    first ``serial`` attribute found — the same by-serial route the wrist cameras take
    (``v4l2_nodes_by_usb_serial``), so a re-enumerated adapter keeps its identity.
    """
    found: list[tuple[int, str]] = []
    for node in Path(sysfs_root).glob("ttyUSB*"):
        try:
            dev = (node / "device").resolve()
        except OSError:
            continue
        d = dev
        matched = False
        for _ in range(SERIAL_WALK_DEPTH):
            s = _read_attr(d / "serial")
            if s is not None:
                matched = s == serial
                break
            if d.parent == d:
                break
            d = d.parent
        if not matched:
            continue
        try:
            n = int(node.name[len("ttyUSB") :])
        except ValueError:
            continue
        found.append((n, f"/dev/{node.name}"))
    return [p for _, p in sorted(found)]


def ftdi_latency_timer_ms(port: str, sysfs_root: str | Path = SYSFS_TTY) -> int | None:
    """The ``latency_timer`` (ms) of the FTDI behind ``port`` (a node or a by-id symlink);
    None when sysfs does not expose it (not an FTDI, no such node)."""
    try:
        name = Path(os.path.realpath(port)).name
    except OSError:
        return None
    raw = _read_attr(Path(sysfs_root) / name / "device" / "latency_timer")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def resolve_port(cfg: GelloConfig, sysfs_root: str | Path = SYSFS_TTY) -> str:
    """``cfg.port``, or the ``ttyUSB`` node carrying ``cfg.usb_serial`` (first match)."""
    if cfg.usb_serial:
        nodes = tty_nodes_by_usb_serial(cfg.usb_serial, sysfs_root)
        if not nodes:
            raise GelloPortError(
                f"no ttyUSB node with USB serial {cfg.usb_serial!r} under {sysfs_root} "
                "(adapter unplugged? check lsusb -d 0403:6014 / ls -la /dev/serial/by-id)"
            )
        if len(nodes) > 1:
            logger.warning(
                "gello: several ttyUSB nodes carry serial %s: %s; using %s",
                cfg.usb_serial,
                nodes,
                nodes[0],
            )
        return nodes[0]
    return cfg.port


@dataclass(frozen=True)
class GelloSample:
    """One leader reading (16-gello §4 "Sample"). ``q_raw`` / ``q`` are 7-vectors in rad;
    ``q = sign * (raw - offset)`` (the fake backend: ``q == q_raw``); ``gripper_frac`` is
    0 closed .. 1 open (None without calibrated endpoints / without a gripper channel);
    ``valid`` is False for a jump or an uncalibrated dynamixel reading — the loop treats an
    invalid sample as ``no_leader``; ``jump`` is True for the jump alone (``valid ==
    calibrated and not jump``), so the calibration ops can accept an uncalibrated reading
    while still refusing a jump-flagged one. ``gripper_raw`` (rad) is what the two gripper
    calibration ops store."""

    q_raw: np.ndarray
    q: np.ndarray
    gripper_frac: float | None
    rx_mono: float
    seq: int
    valid: bool
    gripper_raw: float | None = None
    jump: bool = False


@dataclass(frozen=True)
class GelloDeviceStatus:
    """Device-side fields (available without a session); ``last`` is the newest sample.
    ``joint_offsets_rad`` are the offsets IN FORCE (config override, else the file; zeros
    for the fake); ``calibration_source`` says which. ``gripper_open_rad`` /
    ``gripper_closed_rad`` echo the calibration file (16-gello §4: every op's result is
    echoed in ``GET /api/gello``)."""

    backend: Literal["dynamixel", "fake", "none"]
    status: GelloStatus
    detail: str
    port: str
    baud: int | None
    seq: int
    rate_hz: float
    age_s: float | None
    last: GelloSample | None
    calibrated: bool
    joint_offsets_rad: tuple[float, ...] | None
    joint_signs: tuple[int, ...]
    gripper_open_rad: float | None = None
    gripper_closed_rad: float | None = None
    calibration_source: Literal["config", "file", "fake", "none"] = "none"
    restarts: int = 0
    invalid_samples: int = 0


def device_telemetry_fields(st: GelloDeviceStatus) -> dict[str, Any]:
    """The ``GelloDeviceTelemetry`` fields (core ``protocol/gello.py``) shared by
    ``TelemetryMsg.gello`` and ``GET /api/gello``, from one status."""
    last = st.last
    return {
        "backend": st.backend,
        "status": st.status,
        "detail": st.detail,
        "port": st.port,
        "baud": st.baud,
        "seq": st.seq,
        "rate_hz": st.rate_hz,
        "age_s": st.age_s,
        "q_raw": [float(x) for x in last.q_raw] if last is not None else None,
        "q": [float(x) for x in last.q] if last is not None else None,
        "gripper_frac": last.gripper_frac if last is not None else None,
        "calibrated": st.calibrated,
        "joint_offsets_rad": (
            [float(x) for x in st.joint_offsets_rad] if st.joint_offsets_rad is not None else None
        ),
        "joint_signs": [int(x) for x in st.joint_signs],
    }


class GelloReader:
    """Owns the leader's device thread for the process lifetime; publishes to ``slot``.

    ``start()`` is a no-op for backend ``none`` and while a thread is alive; ``stop()``
    joins it; ``restart()`` = stop + start (a new port scan). ``status(now)`` derives
    ``age_s`` / ``stale`` from the receive time of the last sample (``cfg.stale_s``) and a
    ``rate_hz`` over the last ``RATE_WINDOW_S``; everything else is set by the backend
    thread under ``_lock``. ``fake_set(q, gripper_frac)`` moves the fake leader (published
    at once and on every fake tick).

    Test seams: ``clock``; ``import_dynamixel`` replaces the lazy ``import dynamixel_sdk``
    (a fake module with ``PortHandler`` / ``PacketHandler`` / ``GroupSyncRead`` /
    ``COMM_SUCCESS`` exercises the real backend's scan, parsing and jump rejection);
    ``calibration_store`` (None = no file: config offsets or uncalibrated); ``sysfs_root``
    for the by-serial port resolution and the latency-timer read.
    """

    def __init__(
        self,
        cfg: GelloConfig,
        slot: LatestSlot[GelloSample],
        *,
        clock: Callable[[], float] = time.monotonic,
        import_dynamixel: Callable[[], Any] | None = None,
        calibration_store: GelloCalibrationStore | None = None,
        sysfs_root: str | Path = SYSFS_TTY,
    ) -> None:
        self.cfg = cfg
        self.slot = slot
        self._clock = clock
        self._import_dynamixel = import_dynamixel or self._default_import_dynamixel
        self._store = calibration_store
        self._sysfs_root = sysfs_root
        self._lock = threading.Lock()
        self._lifecycle = threading.RLock()
        self._status: GelloStatus = "no_backend"
        self._detail = "GELLO disabled (backend: none)" if cfg.backend == "none" else ""
        self._port = ""
        self._baud: int | None = None
        self._seq = 0
        self._last: GelloSample | None = None
        self._rx_times: deque[float] = deque(maxlen=256)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.restarts = 0
        self.invalid_samples = 0
        self._signs = np.asarray(cfg.joint_signs, dtype=np.float64)
        self._config_offsets = (
            np.asarray(cfg.joint_offsets_rad, dtype=np.float64)
            if cfg.joint_offsets_rad is not None
            else None
        )
        self._calibration: GelloCalibration = (
            self._store.load() if self._store is not None else EMPTY_CALIBRATION
        )
        self._fake_q = np.asarray(FAKE_DEFAULT_Q, dtype=np.float64)
        self._fake_grip: float | None = FAKE_DEFAULT_GRIPPER

    @property
    def backend(self) -> str:
        return self.cfg.backend

    # -- calibration ------------------------------------------------------------------------
    @property
    def calibration(self) -> GelloCalibration:
        """The calibration FILE's content as last loaded (config offsets are not in it)."""
        with self._lock:
            return self._calibration

    def reload_calibration(self) -> GelloCalibration:
        """Re-read the calibration store (after a ``POST /api/gello/calibrate`` op); the
        next sample uses it. Returns the file's content."""
        cal = self._store.load() if self._store is not None else EMPTY_CALIBRATION
        with self._lock:
            self._calibration = cal
        return cal

    def _offsets_in_force(self) -> tuple[np.ndarray | None, str]:
        """(offsets, source) under ``_lock``: config override > file > none; fake = zeros."""
        if self.cfg.backend == "fake":
            return np.zeros(7), "fake"
        if self._config_offsets is not None:
            return self._config_offsets, "config"
        if self._calibration.joint_offsets_rad is not None:
            return np.asarray(self._calibration.joint_offsets_rad, dtype=np.float64), "file"
        return None, "none"

    # -- lifecycle --------------------------------------------------------------------------
    def start(self) -> None:
        """Spawn the backend thread; a no-op for backend ``none`` and while a thread is
        alive."""
        with self._lifecycle:
            if self.cfg.backend == "none" or self._thread is not None:
                return
            self._stop.clear()
            self._set_status("starting", "")
            target = self._run_fake if self.cfg.backend == "fake" else self._run_dynamixel
            self._thread = threading.Thread(
                target=self._thread_main, args=(target,), name="gello-reader", daemon=True
            )
            self._thread.start()

    def _thread_main(self, target: Callable[[], None]) -> None:
        try:
            target()
        except Exception:  # last line of defence: the thread must end cleanly
            logger.exception("gello reader thread crashed")
            self._set_status("error", "gello reader crashed (see runtime.log)")
        finally:
            if self._thread is threading.current_thread():
                self._thread = None

    def scan_budget_s(self) -> float:
        """Worst-case duration of one baud scan (every rate x every configured id at
        :data:`PING_TIMEOUT_S` + the serial re-open per rate): the floor of ``stop()``'s
        join timeout, so ``Runtime.stop()`` never abandons an open port mid-scan."""
        rates = 1 if self.cfg.baud is not None else len(GELLO_BAUD_SCAN)
        n_ids = len(self.cfg.joint_ids) + (1 if self.cfg.gripper_id is not None else 0)
        return rates * (n_ids * PING_TIMEOUT_S + SCAN_REOPEN_S)

    def stop(self, timeout: float | None = None) -> bool:
        """Signal the backend thread and join it; True once it is gone (the port closed).
        ``timeout`` None = ``max(STOP_JOIN_TIMEOUT_S, scan_budget_s())``; the scan itself
        checks the stop flag between pings, so the join normally returns within one ping."""
        if timeout is None:
            timeout = max(STOP_JOIN_TIMEOUT_S, self.scan_budget_s())
        with self._lifecycle:
            self._stop.set()
            thread = self._thread
            if thread is None:
                return True
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("gello reader thread did not stop within %.1f s", timeout)
                return False
            self._thread = None
            return True

    def restart(self, *, timeout: float | None = None) -> None:
        """Stop (joined) and start again: a fresh port resolution and baud scan."""
        with self._lifecycle:
            if timeout is None:
                timeout = max(STOP_JOIN_TIMEOUT_S, self.scan_budget_s())
            if not self.stop(timeout=timeout):
                detail = f"gello reader did not stop within {timeout:.0f} s (not restarted)"
                self._set_status("error", detail)
                raise RuntimeError(detail)
            with self._lock:
                self._port, self._baud = "", None
            self.start()

    # -- status -----------------------------------------------------------------------------
    def _set_status(self, status: GelloStatus, detail: str) -> None:
        with self._lock:
            self._status = status
            self._detail = detail

    def status(self, now: float | None = None) -> GelloDeviceStatus:
        now = self._clock() if now is None else now
        with self._lock:
            status, detail, last = self._status, self._detail, self._last
            port, baud = self._port, self._baud
            rx = [t for t in self._rx_times if t > now - RATE_WINDOW_S]
            offsets, source = self._offsets_in_force()
            cal = self._calibration
        age = None if last is None else max(0.0, now - last.rx_mono)
        if status == "connected" and age is not None and age > self.cfg.stale_s:
            status = "stale"
            detail = detail or f"no leader sample for {age:.2f} s"
        rate = (len(rx) - 1) / (now - rx[0]) if len(rx) > 1 and now > rx[0] else 0.0
        return GelloDeviceStatus(
            backend=self.cfg.backend,
            status=status,
            detail=detail,
            port=port,
            baud=baud,
            seq=last.seq if last is not None else 0,
            rate_hz=float(rate),
            age_s=age,
            last=last,
            calibrated=offsets is not None,
            joint_offsets_rad=tuple(float(x) for x in offsets) if offsets is not None else None,
            joint_signs=tuple(int(x) for x in self.cfg.joint_signs),
            gripper_open_rad=cal.gripper_open_rad,
            gripper_closed_rad=cal.gripper_closed_rad,
            calibration_source=source,  # type: ignore[arg-type]
            restarts=self.restarts,
            invalid_samples=self.invalid_samples,
        )

    def latest(self) -> GelloSample | None:
        with self._lock:
            return self._last

    def fresh_sample(
        self, now: float | None = None, *, require_calibrated: bool = True
    ) -> GelloSample | None:
        """The newest sample if it is no older than ``stale_s``, not jump-flagged and -
        by default - ``valid`` (calibrated); ``require_calibrated=False`` also hands out an
        UNCALIBRATED reading (still fresh, still no jump): what ``match_arm`` and the
        gripper endpoint ops need, since they exist to create the calibration."""
        now = self._clock() if now is None else now
        with self._lock:
            last = self._last
        if last is None or now - last.rx_mono > self.cfg.stale_s or last.jump:
            return None
        if require_calibrated and not last.valid:
            return None
        return last

    # -- publishing (backend thread; fake_set from any thread) --------------------------------
    def _publish(self, q_raw: np.ndarray, gripper_raw: float | None) -> GelloSample:
        """Map, validate (jump / calibration) and publish one dynamixel reading."""
        rx = self._clock()
        with self._lock:
            prev = self._last
            self._seq += 1
            offsets, _ = self._offsets_in_force()
            cal = self._calibration
            calibrated = offsets is not None
            q = self._signs * (q_raw - (offsets if calibrated else 0.0))
            jump = prev is not None and bool(
                np.max(np.abs(q_raw - prev.q_raw)) > self.cfg.max_jump_rad
            )
            valid = calibrated and not jump
            frac = gripper_frac(gripper_raw, cal.gripper_open_rad, cal.gripper_closed_rad)
            sample = GelloSample(
                q_raw=q_raw,
                q=q,
                gripper_frac=frac,
                rx_mono=rx,
                seq=self._seq,
                valid=valid,
                gripper_raw=gripper_raw,
                jump=jump,
            )
            self._last = sample
            self._rx_times.append(rx)
            self._status = "connected"
            if jump:
                self._detail = f"jump > {self.cfg.max_jump_rad:g} rad: sample invalid"
            elif not calibrated:
                self._detail = (
                    "uncalibrated: run POST /api/gello/calibrate {op: match_arm} with GELLO "
                    "posed like the Manipulation Arm"
                )
            else:
                self._detail = ""
            if not valid:
                self.invalid_samples += 1
        if jump:
            logger.warning(
                "gello sample %d invalid (jump > %g rad)", sample.seq, self.cfg.max_jump_rad
            )
        self.slot.put(sample)
        return sample

    def _publish_fake(self) -> GelloSample:
        rx = self._clock()
        with self._lock:
            self._seq += 1
            q = self._fake_q.copy()
            sample = GelloSample(
                q_raw=q,
                q=q.copy(),
                gripper_frac=self._fake_grip,
                rx_mono=rx,
                seq=self._seq,
                valid=True,
                gripper_raw=self._fake_grip,
            )
            self._last = sample
            self._rx_times.append(rx)
            self._status = "connected"
            self._detail = ""
            self._port, self._baud = "fake", None
        self.slot.put(sample)
        return sample

    def fake_set(self, q, gripper_frac: float | None = None) -> GelloSample | None:
        """Move the fake leader: ``q`` = 7 joints (rad), ``gripper_frac`` 0..1 (None =
        unchanged). Published immediately while the fake thread runs (and again at every
        fake tick). Raises ``RuntimeError`` for the other backends."""
        if self.cfg.backend != "fake":
            raise RuntimeError(f"fake_set needs backend 'fake' (backend is {self.cfg.backend!r})")
        arr = np.asarray(q, dtype=np.float64).reshape(-1)
        if arr.shape != (7,) or not np.all(np.isfinite(arr)):
            raise ValueError("fake_set needs 7 finite joint values (rad)")
        if gripper_frac is not None and not (0.0 <= float(gripper_frac) <= 1.0):
            raise ValueError("gripper_frac must be within [0, 1]")
        with self._lock:
            self._fake_q = arr.copy()
            if gripper_frac is not None:
                self._fake_grip = float(gripper_frac)
            running = self._thread is not None
        return self._publish_fake() if running else None

    # -- fake backend -------------------------------------------------------------------------
    def _run_fake(self) -> None:
        # Paced on the wall clock (``time.monotonic``): the ``clock`` seam stamps the
        # samples and feeds ``status()`` only, so a frozen test clock cannot stall the loop.
        period = 1.0 / self.cfg.poll_hz
        next_t = time.monotonic()
        while not self._stop.is_set():
            self._publish_fake()
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_t = time.monotonic()

    # -- dynamixel backend --------------------------------------------------------------------
    @staticmethod
    def _default_import_dynamixel():
        import dynamixel_sdk  # lazy; the ONLY import site (ruff banned-api elsewhere)

        return dynamixel_sdk

    def _run_dynamixel(self) -> None:
        """Supervisor: run the bus loop and restart it with backoff when it dies (port
        gone, no servo answering, read failures). A missing ``dynamixel_sdk`` is final."""
        try:
            dxl = self._import_dynamixel()
        except Exception as e:  # ImportError, or a broken pyserial underneath
            self._set_status(
                "no_backend",
                f"dynamixel_sdk not importable ({e.__class__.__name__}: {e}); install the "
                "runtime's [gello] extra",
            )
            return
        first, cap = RESTART_BACKOFF_S
        backoff = first
        while not self._stop.is_set():
            t0 = self._clock()
            try:
                self._run_dynamixel_once(dxl)
            except (GelloPortError, GelloBusError, OSError) as e:
                self._set_status("error", str(e))
            except Exception as e:  # noqa: BLE001 - unexpected: report, then retry
                logger.exception("gello bus loop failed")
                self._set_status("error", f"{e.__class__.__name__}: {e}")
            if self._stop.is_set():
                return
            if self._clock() - t0 > RESTART_RESET_S:
                backoff = first
            self.restarts += 1
            logger.warning(
                "gello: bus loop ended (%s); restart %d in %.1f s",
                self.status().detail or "no detail",
                self.restarts,
                backoff,
            )
            self._stop.wait(backoff)
            backoff = min(backoff * 2.0, cap)

    def _run_dynamixel_once(self, dxl) -> None:
        port = resolve_port(self.cfg, self._sysfs_root)
        if self._stop.is_set():
            raise GelloBusError("stopped before opening the port")
        with self._lock:
            self._port, self._baud = port, None
        self._set_status("starting", f"opening {port}")
        ph = dxl.PortHandler(port)
        if not ph.openPort():
            raise GelloPortError(f"cannot open {port} (permissions? is the adapter plugged in?)")
        try:
            pk = dxl.PacketHandler(PROTOCOL_VERSION)
            ids = self._configured_ids()
            baud, answered = self._select_baud(dxl, ph, pk, port)
            with self._lock:
                self._baud = baud
            missing = [i for i in ids if i not in answered]
            if missing:
                logger.warning(
                    "gello: servos %s did not answer the ping at %d baud on %s (answered: %s)",
                    missing,
                    baud,
                    port,
                    sorted(answered),
                )
            lat = ftdi_latency_timer_ms(port, self._sysfs_root)
            if lat is not None and lat > LATENCY_TIMER_WARN_MS:
                logger.warning(
                    "gello: FTDI latency_timer is %d ms on %s (> %d ms caps %d servos at ~%d Hz); "
                    "install the udev rule of 16-gello §9.3 (latency_timer 1)",
                    lat,
                    port,
                    LATENCY_TIMER_WARN_MS,
                    len(ids),
                    int(1000 / (lat * 2)),
                )
            gsr = dxl.GroupSyncRead(ph, pk, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
            for i in ids:
                gsr.addParam(i)
            self._set_status(
                "starting", f"bus at {baud} baud on {port}; waiting for the first sync read"
            )
            logger.info("gello: %s at %d baud, servos %s", port, baud, ids)
            self._read_loop(dxl, pk, gsr, ids)
        finally:
            try:
                ph.closePort()
            except Exception:  # noqa: BLE001
                logger.debug("gello: closePort failed", exc_info=True)

    def _configured_ids(self) -> list[int]:
        return [*self.cfg.joint_ids] + (
            [self.cfg.gripper_id] if self.cfg.gripper_id is not None else []
        )

    def _select_baud(self, dxl, ph, pk, port: str) -> tuple[int, set[int]]:
        """``cfg.baud`` (checked with one ping round) or the first rate of
        ``GELLO_BAUD_SCAN`` at which a CONFIGURED servo answers its ping.

        2026-09-09 review: the SDK's ``broadcastPing`` busy-polls for ``3528 x tx_time +
        3 x MAX_ID + 16`` ms (a 0.77 s floor, 1.4 s at 57600) per rate and could not be
        interrupted, so a bus where nothing answers (servos unpowered) spun a core for
        ~4.6 s per restart cycle and ``stop()`` mid-scan ran into its join budget. Now one
        ``ping`` per configured id (~40 ms each), the stop flag checked before every rate
        and every ping (``GelloBusError('stopped during the baud scan')``)."""
        ids = self._configured_ids()
        rates = (self.cfg.baud,) if self.cfg.baud is not None else GELLO_BAUD_SCAN
        for rate in rates:
            if self._stop.is_set():
                raise GelloBusError("stopped during the baud scan")
            if not ph.setBaudRate(int(rate)):
                logger.warning("gello: %s refused %d baud", port, rate)
                continue
            self._set_status("starting", f"pinging servos {ids} at {rate} baud on {port}")
            answered = self._ping_ids(dxl, ph, pk, ids)
            if answered:
                if self.cfg.baud is None:
                    logger.info(
                        "gello: baud scan -> %d (servos %s answered)", rate, sorted(answered)
                    )
                return int(rate), answered
        tried = " / ".join(str(r) for r in rates)
        raise GelloBusError(
            f"no servo answered the ping (ids {ids}) at {tried} baud on {port} "
            "(servo power off? wrong ids? wrong adapter?)"
        )

    def _ping_ids(self, dxl, ph, pk, ids: list[int]) -> set[int]:
        """One protocol-2.0 ``ping`` per configured id at the port's current baud; the set
        of ids that answered. Interruptible: ``stop()`` between two pings raises."""
        answered: set[int] = set()
        for i in ids:
            if self._stop.is_set():
                raise GelloBusError("stopped during the baud scan")
            try:
                _model, result, _error = pk.ping(ph, int(i))
            except Exception as e:  # noqa: BLE001 - a serial hiccup during the scan is "no answer"
                logger.debug("gello: ping of id %d raised %r", i, e)
                continue
            if result == dxl.COMM_SUCCESS:
                answered.add(int(i))
        return answered

    def _read_loop(self, dxl, pk, gsr, ids: list[int]) -> None:
        period = 1.0 / self.cfg.poll_hz
        next_t = time.monotonic()  # wall-clock pacing (see _run_fake)
        fails = 0
        n_joints = len(self.cfg.joint_ids)
        joints = np.zeros(7, dtype=np.float64)
        while not self._stop.is_set():
            result = gsr.txRxPacket()
            if result != dxl.COMM_SUCCESS:
                fails += 1
                if fails >= READ_FAILS_BEFORE_RESTART:
                    text = pk.getTxRxResult(result) if hasattr(pk, "getTxRxResult") else result
                    raise GelloBusError(f"sync read failed {fails} times in a row: {text}")
            else:
                missing = [
                    i
                    for i in ids
                    if not gsr.isAvailable(i, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                ]
                if missing:
                    fails += 1
                    if fails >= READ_FAILS_BEFORE_RESTART:
                        raise GelloBusError(
                            f"servos {missing} did not answer {fails} sync reads in a row"
                        )
                else:
                    fails = 0
                    for k, i in enumerate(self.cfg.joint_ids[:n_joints]):
                        joints[k] = ticks_to_rad(
                            gsr.getData(i, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                        )
                    grip = (
                        ticks_to_rad(
                            gsr.getData(
                                self.cfg.gripper_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
                            )
                        )
                        if self.cfg.gripper_id is not None
                        else None
                    )
                    self._publish(joints.copy(), grip)
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_t = time.monotonic()


__all__ = [
    "GelloStatus",
    "GelloSample",
    "GelloDeviceStatus",
    "GelloReader",
    "GelloPortError",
    "GelloBusError",
    "ADDR_PRESENT_POSITION",
    "LEN_PRESENT_POSITION",
    "TICKS_PER_REV",
    "FAKE_DEFAULT_Q",
    "FAKE_DEFAULT_GRIPPER",
    "RESTART_BACKOFF_S",
    "READ_FAILS_BEFORE_RESTART",
    "PING_TIMEOUT_S",
    "STOP_JOIN_TIMEOUT_S",
    "LATENCY_TIMER_WARN_MS",
    "SYSFS_TTY",
    "ticks_to_rad",
    "tty_nodes_by_usb_serial",
    "ftdi_latency_timer_ms",
    "resolve_port",
    "device_telemetry_fields",
]
