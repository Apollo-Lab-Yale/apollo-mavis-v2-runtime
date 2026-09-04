"""Tracker calibration wizard back end (13-tracker §4 "Calibration modes"; phase-10).

Two session-less device-management flows behind one REST endpoint
(``/api/tracker/calibration``, 04-runtime §13.1) with progress on telemetry
(``TrackerTelemetry.calibration``):

* **base_station** — re-solve the Lighthouse poses with libsurvive's global
  scene solver. The reader is restarted on a TEMPORARY copy of the libsurvive
  config (libsurvive rewrites whatever ``--configfile`` points at, so the
  official file is never handed to it) with ``--force-calibrate 1
  --globalscenesolver 1``; INFO lines (``Global solve with N scenes for L``,
  ``Using LH i (serial) as reference lighthouse``, ``Adding lighthouse ch C
  (idx: L, ...)``, ``OOTX not set for LH in channel C``) drive the progress
  fields. ``stations_visible`` counts stations the solver actually got light
  from since ``start`` (scenes solved, a station added/decoded from light, a
  pose appearing during a run) — NOT the stations libsurvive merely knows
  from the config file (it creates one LIGHTHOUSE object per configured
  station at init, unplugged or not); ``channel`` comes from the config copy
  (``"mode"``) or the ``Adding lighthouse`` line. ``validate`` freezes the
  solution (``--globalscenesolver 0 --disable-calibrate 1
  --use-stationary-sensor-window 0`` = the moving-mode sensor window teleop
  actually sees) and measures the scatter of a still controller; a tracking
  dropout inside the window (gap > ``VALIDATION_MAX_GAP_S``) or fewer samples
  than ``VALIDATION_MIN_RATE_HZ`` × window fails it (an occluded / lost
  controller must never pass); ``install`` backs the official file up, copies the temporary
  bytes over it, records the install (and invalidates the yaw alignment: the
  lighthouse world is re-anchored by a new solution) and restores the normal
  arguments.
* **yaw** — the 7-click gesture (start, left, forward, right, back, up, down)
  captured from trigger rising edges or REST ``capture``; :func:`fit_yaw` maps
  the operator's horizontal legs onto the MJCF axes (operator convention from
  CLAUDE.md "Hardware facts": left = +X, forward = -Y, right = -X, back = +Y);
  ``apply`` writes ``tracker_settings.yaw_deg`` and persists it.

Persistence: ``calibration_dir/tracker_calibration.json``
(:class:`PersistedCalibration`); :func:`apply_persisted_yaw` overrides the YAML
``tracker.yaw_deg`` at startup when a valid yaw was applied.

Threading: ``command()`` validates the transition under the lock and returns
immediately (illegal -> :class:`CalibrationError`, REST 409); a worker thread
runs the reader restarts (a ``simple_close`` join may take seconds), parses
INFO lines, times the validation window, detects trigger edges and performs
the install. ``status()`` is cheap and lock-protected (25 Hz telemetry).
No pysurvive here: everything reaches libsurvive through the reader.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from apollo_mavis_v2_core import LatestSlot, Pose
from apollo_mavis_v2_core.protocol import (
    CalibrationValidation,
    LighthouseStatus,
    PoseMsg,
    TrackerCalibrationCommand,
    TrackerCalibrationStatus,
    YawGesturePoint,
)

from ..config import RuntimeConfig, TrackerCalibrationConfig
from .tracker import TrackerSample

logger = logging.getLogger(__name__)

YAW_POINT_ORDER: tuple[str, ...] = ("start", "left", "forward", "right", "back", "up", "down")
# Operator convention (CLAUDE.md "Hardware facts"; 13-tracker §6): the operator stands at
# the +Y outer edge facing -Y, so left = +X, forward = -Y, right = -X, back = +Y.
YAW_LEG_TARGETS: tuple[tuple[str, tuple[float, float]], ...] = (
    ("left", (1.0, 0.0)),
    ("forward", (0.0, -1.0)),
    ("right", (-1.0, 0.0)),
    ("back", (0.0, 1.0)),
)
PERSIST_FILE = "tracker_calibration.json"
WORKER_PERIOD_S = 0.02  # ~50 Hz: trigger edges, INFO parsing, validation timing
RECENT_SAMPLES = 256  # raw samples remembered for still / averaging windows (>= 2 s at 100 Hz)
VALIDATION_TRACKING_TIMEOUT_S = 30.0  # no pose after the validation restart -> failed validation
VALIDATION_MIN_RATE_HZ = 10.0  # need >= this x validation_seconds samples (libsurvive: >= 50 Hz)
VALIDATION_MAX_GAP_S = 0.5  # adjacent samples (or window edges) farther apart = tracking lost
# libsurvive options the calibration modes own: stripped (with their value) from the
# normal arguments before the per-mode flags are appended.
CALIBRATION_ARGS: frozenset[str] = frozenset(
    {
        "--globalscenesolver",
        "--disable-calibrate",
        "--configfile",
        "--force-calibrate",
        "--use-stationary-sensor-window",
    }
)
# INFO-line contracts (pinned libsurvive commit, scripts/tracker/02-build-pysurvive.sh);
# the reader strips ANSI escapes before handing lines over.
RE_GLOBAL_SOLVE = re.compile(r"Global solve with (\d+) scenes for (\d+)")
RE_REFERENCE = re.compile(r"Using LH (\d+) \((\w+)\) as reference lighthouse")
# gen2 station seen from light but not decoded yet (survive_process_gen2.c); the channel is
# attributed to an index through the config / "Adding lighthouse" channel map.
RE_OOTX_CHANNEL = re.compile(r"OOTX not set for LH in channel (\d+)")
RE_OOTX_INDEX = re.compile(r"OOTX not set for LH (\d+)")  # gen1 wording: index directly
# A station NOT in the config file, created from light (survive.c survive_get_bsd_idx).
RE_ADD_LH = re.compile(r"Adding lighthouse ch (\d+) \(idx: (\d+)")
RE_FORCE_CALIBRATE = re.compile(r"Force calibrate flag set")
# libsurvive config file: quasi-JSON (top-level "k":"v", lines without braces) with
# "lighthouseN":{"index":"N","id":"...","mode":"<channel>",...} blocks; scanned with a
# tolerant regex like scripts/tracker/03-lh-consistency-check.sh (json.load fails on it).
RE_CONFIG_LIGHTHOUSE = re.compile(r'"lighthouse(\d+)"\s*:\s*\{(.*?)\}', re.S)


class CalibrationError(Exception):
    """Illegal calibration command / transition; ``rest.py`` maps it to 409 ``{detail}``."""


# -- libsurvive argument sets ---------------------------------------------------------
def strip_calibration_args(args: Sequence[str]) -> list[str]:
    """``args`` without the ``CALIBRATION_ARGS`` options (each with its value)."""
    out: list[str] = []
    skip_value = False
    for a in args:
        if skip_value:
            skip_value = False
            if not a.startswith("--"):
                continue
        if a in CALIBRATION_ARGS:
            skip_value = True
            continue
        out.append(a)
    return out


def capture_args(normal: Sequence[str], tmp_config: Path, *, force: bool) -> list[str]:
    """Capture mode: the global scene solver on a temporary config; ``force``
    clears the existing lighthouse solution (first capture), otherwise the
    current solution is kept and refined (``capture`` after validation)."""
    out = [*strip_calibration_args(normal), "--configfile", str(tmp_config)]
    if force:
        out += ["--force-calibrate", "1"]
    return [*out, "--globalscenesolver", "1"]


def validation_args(normal: Sequence[str], tmp_config: Path) -> list[str]:
    """Validation mode: frozen solution + the moving-mode sensor window."""
    return [
        *strip_calibration_args(normal),
        "--configfile", str(tmp_config),
        "--globalscenesolver", "0",
        "--disable-calibrate", "1",
        "--use-stationary-sensor-window", "0",
    ]


def lighthouse_channels(config_text: str) -> dict[int, int]:
    """``{index: channel}`` of the stations a libsurvive config file knows
    (``"lighthouseN":{... "index":"N", "mode":"<channel>" ...}``; libsurvive
    numbers gen2 channels 0-15 internally and prints them as such). Missing /
    malformed blocks are skipped; the text is never parsed as JSON."""
    out: dict[int, int] = {}
    for m in RE_CONFIG_LIGHTHOUSE.finditer(config_text):
        block = m.group(2)
        mode = re.search(r'"mode"\s*:\s*"(\d+)"', block)
        if mode is None:
            continue
        channel = int(mode.group(1))
        if not 0 <= channel < 16:
            continue  # 255 = libsurvive's "unset"
        idx = re.search(r'"index"\s*:\s*"(\d+)"', block)
        out[int(idx.group(1)) if idx else int(m.group(1))] = channel
    return out


# -- yaw fit (pure) -------------------------------------------------------------------
def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def fit_yaw(
    points: Sequence[Sequence[float]], cfg: TrackerCalibrationConfig
) -> tuple[float, float, list[str]]:
    """Yaw (deg) such that ``Rz(yaw) · p_raw`` puts the operator's gesture legs on
    the MJCF axes, from the 7 RAW positions of the gesture (start, left,
    forward, right, back, up, down).

    Horizontal legs ``left = P1-P0``, ``forward = P2-P1``, ``right = P3-P2``,
    ``back = P4-P3`` with targets ``+X, -Y, -X, +Y`` (operator convention);
    unit xy vectors ``u_i`` vs targets ``e_i``:
    ``yaw = atan2(Σ(u_x e_y - u_y e_x), Σ(u_x e_x + u_y e_y))`` (so
    ``Rz(yaw) u_i ≈ e_i``, matching ``align_pose``'s ``p_world = Rz(yaw) p_raw``).
    ``residual`` = mean angle (deg) between ``Rz(yaw) u_i`` and ``e_i``.

    Returns ``(yaw_deg, residual_deg, checks)``; ``checks`` lists the failed
    sanity checks (empty = ok): every horizontal leg >= ``yaw_min_leg_m`` and
    ``|dz| <= 0.5 |d|``; ``up = P5-P4`` has ``dz > 0`` and ``dz >= 0.5 |d|``;
    ``down = P6-P5`` has ``dz < 0`` (lighthouse z is up and the gesture was not
    done backwards); ``residual <= yaw_max_residual_deg``. The yaw is always
    returned so the operator can see it even when a check fails.

    The fit is only as good as the stance convention: an operator standing on
    the opposite (arm) side produces exactly ``yaw + 180°`` (2026-09-02 lesson:
    -77.9° vs the correct 102.1°).
    """
    P = np.asarray(points, dtype=np.float64)
    if P.shape != (7, 3) or not np.all(np.isfinite(P)):
        raise ValueError("fit_yaw needs the 7 finite gesture positions (start, left, ..., down)")
    checks: list[str] = []
    s_sin = s_cos = 0.0
    units: list[tuple[np.ndarray, np.ndarray]] = []
    for i, (label, e) in enumerate(YAW_LEG_TARGETS):
        d = P[i + 1] - P[i]
        length = float(np.linalg.norm(d))
        if length < cfg.yaw_min_leg_m:
            checks.append(
                f"{label} leg too short: {length * 100:.0f} cm < {cfg.yaw_min_leg_m * 100:.0f} cm"
            )
        elif abs(float(d[2])) > 0.5 * length:
            checks.append(f"{label} leg not horizontal: |dz| {abs(float(d[2])) * 100:.0f} cm")
        h = float(math.hypot(d[0], d[1]))
        if h < 1e-9:
            continue
        u = d[:2] / h
        ev = np.asarray(e)
        s_sin += float(u[0] * ev[1] - u[1] * ev[0])
        s_cos += float(u[0] * ev[0] + u[1] * ev[1])
        units.append((u, ev))
    up = P[5] - P[4]
    up_len = float(np.linalg.norm(up))
    if up[2] <= 0.0:
        checks.append("up leg does not rise (lighthouse z must be up; gesture reversed?)")
    elif up[2] < 0.5 * up_len:
        checks.append(
            f"up leg not vertical enough: dz {up[2] * 100:.0f} cm of {up_len * 100:.0f} cm"
        )
    down = P[6] - P[5]
    if down[2] >= 0.0:
        checks.append("down leg does not descend")
    if not units:
        checks.append("no usable horizontal legs")
        return 0.0, 180.0, checks
    theta = math.atan2(s_sin, s_cos)
    c, s = math.cos(theta), math.sin(theta)
    angles = []
    for u, ev in units:
        r = np.array([c * u[0] - s * u[1], s * u[0] + c * u[1]])
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(r, ev)))))))
    residual = float(np.mean(angles))
    if residual > cfg.yaw_max_residual_deg:
        checks.append(f"fit residual {residual:.1f}° > {cfg.yaw_max_residual_deg:.1f}°")
    return _wrap_deg(math.degrees(theta)), residual, checks


# -- persistence -----------------------------------------------------------------------
@dataclass
class PersistedCalibration:
    """``calibration_dir/tracker_calibration.json`` (13-tracker §4): the applied
    yaw and whether it is still valid for the installed lighthouse solution."""

    yaw_deg: float | None = None
    yaw_valid: bool = True
    yaw_calibrated_at: float | None = None
    base_station_installed_at: float | None = None
    lighthouse_config_sha256: str | None = None

    @staticmethod
    def path_for(calibration_dir: Path) -> Path:
        return Path(calibration_dir).expanduser() / PERSIST_FILE

    @classmethod
    def load(cls, calibration_dir: Path) -> PersistedCalibration:
        """Read the file; missing -> defaults; unreadable -> defaults + warning."""
        path = cls.path_for(calibration_dir)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return cls()
        except (OSError, ValueError) as e:
            logger.warning("tracker calibration file %s unreadable (%s): ignored", path, e)
            return cls()
        if not isinstance(data, dict):
            logger.warning("tracker calibration file %s is not an object: ignored", path)
            return cls()

        def _num(key: str) -> float | None:
            v = data.get(key)
            return float(v) if isinstance(v, int | float) and math.isfinite(float(v)) else None

        sha = data.get("lighthouse_config_sha256")
        return cls(
            yaw_deg=_num("yaw_deg"),
            yaw_valid=bool(data.get("yaw_valid", True)),
            yaw_calibrated_at=_num("yaw_calibrated_at"),
            base_station_installed_at=_num("base_station_installed_at"),
            lighthouse_config_sha256=str(sha) if isinstance(sha, str) else None,
        )

    def save(self, calibration_dir: Path) -> Path:
        """Atomic write (tmp + replace); creates the directory."""
        path = self.path_for(calibration_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "yaw_deg": self.yaw_deg,
                    "yaw_valid": self.yaw_valid,
                    "yaw_calibrated_at": self.yaw_calibrated_at,
                    "base_station_installed_at": self.base_station_installed_at,
                    "lighthouse_config_sha256": self.lighthouse_config_sha256,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
        return path


def apply_persisted_yaw(cfg: RuntimeConfig) -> PersistedCalibration:
    """Startup override (``Runtime.__init__`` before ``TrackerSettings.from_config``):
    a persisted, still-valid yaw replaces the YAML ``tracker.yaw_deg`` (the YAML
    value is only the first-start default). Returns the record read."""
    persisted = PersistedCalibration.load(cfg.calibration_dir)
    if persisted.yaw_valid and persisted.yaw_deg is not None:
        cfg.tracker.yaw_deg = float(persisted.yaw_deg)
        logger.info(
            "tracker yaw_deg %.2f from %s", cfg.tracker.yaw_deg,
            PersistedCalibration.path_for(cfg.calibration_dir),
        )
    return persisted


# -- state machine ----------------------------------------------------------------------
@dataclass
class _Lighthouse:
    index: int
    channel: int | None = None
    serial: str | None = None
    pose: Pose | None = None
    scenes: int = 0
    reference: bool = False


def _pose_msg(pose: Pose) -> PoseMsg:
    return PoseMsg(
        position=tuple(float(x) for x in pose.position),
        orientation=tuple(float(x) for x in pose.orientation),
    )


class TrackerCalibration:
    """Runtime-owned calibration FSM over a duck-typed reader (``backend``,
    ``restart(args)``, ``stop()``, ``lighthouses()``, ``on_info`` attribute),
    the live :class:`~.tracker.TrackerSettings`, the runtime config and the
    tracker :class:`LatestSlot`. ``session_active()`` gates both flows (a
    reader restart or a trigger click must never hit a running session)."""

    def __init__(
        self,
        reader,
        settings,
        cfg: RuntimeConfig,
        slot: LatestSlot[TrackerSample],
        session_active: Callable[[], bool],
        *,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ) -> None:
        self._reader = reader
        self._settings = settings
        self._cfg = cfg
        self._ccfg: TrackerCalibrationConfig = cfg.tracker.calibration
        self._slot = slot
        self._session_active = session_active
        self._clock = clock
        self._wall = wall
        self._dir = Path(cfg.calibration_dir).expanduser()
        self._config_path = Path(cfg.tracker.libsurvive_config_path).expanduser()
        self._normal_args = list(cfg.tracker.libsurvive_args)
        self._persisted = PersistedCalibration.load(self._dir)
        self._lock = threading.RLock()
        self._jobs: deque[Callable[[], None]] = deque()
        self._thread: threading.Thread | None = None
        self._wake = threading.Event()
        self._closing = False
        self._info_queue: deque[str] = deque(maxlen=1024)  # filled on libsurvive's C thread
        self._recent: deque[tuple[float, int, np.ndarray, np.ndarray]] = deque(
            maxlen=RECENT_SAMPLES
        )
        self._last_seq = -1
        self._prev_trigger = False
        # flow state (all under _lock)
        self._active = False
        self._kind = "none"
        self._phase = "idle"
        self._detail = ""
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._scenes = 0
        self._lh: dict[int, _Lighthouse] = {}
        self._seen: set[int] = set()  # station indices the solver got light from since start
        self._channels: set[int] = set()  # channels from "OOTX not set ... channel C" lines
        self._snap_had_pose: dict[int, bool] | None = None  # previous snapshot, this run
        self._stations_visible = 0
        self._controller_still: bool | None = None
        self._validation: CalibrationValidation | None = None
        self._installed_path: str | None = None
        self._backup_path: str | None = None
        self._yaw_points: list[tuple[str, Pose]] = []
        self._next_point: str | None = None
        self._fitted: float | None = None
        self._residual: float | None = None
        self._checks: list[str] = []
        self._applied: float | None = None
        # base-station bookkeeping
        self._tmp: Path | None = None
        self._ts = ""
        self._force = False
        self._saw_force = False
        self._restart_mono: float | None = None
        self._val_t0: float | None = None  # first pose after the validation restart
        self._val_seq = -1
        self._val_pos: list[np.ndarray] = []
        self._val_rx: list[float] = []  # receive times of _val_pos (dropout detection)

    # -- public ---------------------------------------------------------------------
    @property
    def active(self) -> bool:
        """A flow is in progress (started and neither applied/installed nor
        aborted/failed); ``POST /api/session`` is refused meanwhile."""
        with self._lock:
            return self._active

    @property
    def normal_args(self) -> list[str]:
        return list(self._normal_args)

    def status(self) -> TrackerCalibrationStatus:
        with self._lock:
            started = self._started_at
            end = self._finished_at if self._finished_at is not None else self._wall()
            return TrackerCalibrationStatus(
                kind=self._kind,
                phase=self._phase,
                detail=self._detail,
                started_at=started,
                elapsed_s=max(0.0, end - started) if started is not None else None,
                scenes=self._scenes,
                lighthouses=[
                    LighthouseStatus(
                        index=lh.index,
                        channel=lh.channel,
                        serial=lh.serial,
                        pose=_pose_msg(lh.pose) if lh.pose is not None else None,
                        scenes=lh.scenes,
                        reference=lh.reference,
                    )
                    for _, lh in sorted(self._lh.items())
                ],
                stations_visible=self._stations_visible,
                controller_still=self._controller_still,
                validation=self._validation,
                installed_path=self._installed_path,
                backup_path=self._backup_path,
                yaw_points=[
                    YawGesturePoint(label=label, pose=_pose_msg(pose))
                    for label, pose in self._yaw_points
                ],
                next_point=self._next_point,
                fitted_yaw_deg=self._fitted,
                fit_residual_deg=self._residual,
                fit_checks=list(self._checks),
                applied_yaw_deg=self._applied,
                yaw_valid=self._persisted.yaw_valid,
                yaw_calibrated_at=self._persisted.yaw_calibrated_at,
                base_station_installed_at=self._persisted.base_station_installed_at,
            )

    def command(self, cmd: TrackerCalibrationCommand) -> TrackerCalibrationStatus:
        """Apply one wizard command; illegal transitions raise :class:`CalibrationError`."""
        with self._lock:
            if self._closing:
                raise CalibrationError("runtime is shutting down")
            if cmd.kind == "base_station":
                self._bs_command(cmd)
            else:
                self._yaw_command(cmd)
            return self.status()

    def close(self) -> None:
        """Process exit: stop the worker; a base-station flow still in progress
        gets the normal libsurvive arguments back before the reader is stopped."""
        with self._lock:
            self._closing = True
            self._jobs.clear()
            thread = self._thread
        self._wake.set()
        if thread is not None:
            thread.join(timeout=10.0)
        with self._lock:
            active, kind = self._active, self._kind
        if active and kind == "base_station":
            try:
                self._reader.restart(self._normal_args)
            except Exception:
                logger.exception("restoring normal libsurvive arguments failed")
        with self._lock:
            if active:
                self._finish("aborted", "runtime shutdown")

    # -- command handlers (under _lock) --------------------------------------------------
    def _require_no_session(self) -> None:
        if self._session_active():
            raise CalibrationError("stop the session first")

    def _require_active(self, kind: str) -> None:
        if not self._active:
            raise CalibrationError(f"no {kind} calibration in progress")
        if self._kind != kind:
            raise CalibrationError(f"{self._kind} calibration in progress")

    def _bs_command(self, cmd: TrackerCalibrationCommand) -> None:
        op = cmd.op
        if op == "start":
            if self._reader.backend != "libsurvive":
                raise CalibrationError("backend is not libsurvive")
            self._require_no_session()
            if self._active:
                raise CalibrationError(f"{self._kind} calibration in progress")
            ts = time.strftime("%Y%m%d-%H%M%S", time.localtime(self._wall()))
            channels: dict[int, int] = {}
            try:
                self._dir.mkdir(parents=True, exist_ok=True)
                tmp = self._dir / f"base_station-{ts}.json"
                if self._config_path.exists():
                    shutil.copyfile(self._config_path, tmp)  # libsurvive rewrites tmp, not this
                    text = tmp.read_text(encoding="utf-8", errors="replace")
                    channels = lighthouse_channels(text)
            except OSError as e:
                raise CalibrationError(f"cannot prepare temporary config: {e}") from None
            self._begin("base_station", "starting", "restarting libsurvive in capture mode")
            # Stations libsurvive will KNOW (one LIGHTHOUSE object each, seen or not):
            # index + channel from the config; they count as visible only once the
            # solver reports light from them (_seen).
            self._lh = {idx: _Lighthouse(idx, channel=ch) for idx, ch in sorted(channels.items())}
            self._tmp, self._ts, self._force = tmp, ts, True
            self._info_queue.clear()
            self._reader.on_info = self._on_info
            args = capture_args(self._normal_args, tmp, force=True)
            self._enqueue(lambda: self._job_restart(args))
            return
        if op == "apply":
            raise CalibrationError("apply is a yaw calibration op")
        self._require_active("base_station")
        assert self._tmp is not None
        if op == "capture":
            if self._phase not in ("done", "validating"):
                raise CalibrationError(f"cannot capture while {self._phase}")
            self._phase, self._detail = "starting", "restarting libsurvive in capture mode"
            self._force = False
            self._validation = None
            args = capture_args(self._normal_args, self._tmp, force=False)
            self._enqueue(lambda: self._job_restart(args))
        elif op == "validate":
            if self._phase not in ("capturing", "done"):
                raise CalibrationError(f"cannot validate while {self._phase}")
            need = self._ccfg.min_scenes
            if self._scenes < need:
                raise CalibrationError(
                    f"need {need} scenes, have {self._scenes} ({need - self._scenes} more)"
                )
            self._phase = "validating"
            self._detail = "restarting libsurvive with the frozen solution"
            self._validation = None
            self._val_t0, self._val_seq, self._val_pos, self._val_rx = None, -1, [], []
            args = validation_args(self._normal_args, self._tmp)
            self._enqueue(lambda: self._job_restart(args))
        elif op == "install":
            if self._phase != "done":
                raise CalibrationError(f"cannot install while {self._phase}")
            if self._validation is None or not self._validation.passed:
                raise CalibrationError("validation has not passed")
            self._phase, self._detail = "installing", "installing the lighthouse calibration"
            self._enqueue(self._job_install)
        elif op == "abort":
            self._finish(
                "aborted",
                f"aborted — normal libsurvive arguments restored; temporary config kept at "
                f"{self._tmp}",
            )
            args = list(self._normal_args)
            self._enqueue(lambda: self._job_restart(args))
        else:  # pragma: no cover - the pydantic Literal excludes other ops
            raise CalibrationError(f"unknown op {op!r}")

    def _yaw_command(self, cmd: TrackerCalibrationCommand) -> None:
        op = cmd.op
        if op == "start":
            self._require_no_session()
            if self._active and self._kind != "yaw":
                raise CalibrationError(f"{self._kind} calibration in progress")
            self._begin("yaw", "capturing", "")
            self._next_point = YAW_POINT_ORDER[0]
            self._detail = self._yaw_prompt()
            now = self._clock()
            self._ingest_sample(now)
            self._prev_trigger = self._trigger_pressed(now)  # a held trigger is not a click
            self._ensure_worker()
            return
        if op in ("validate", "install"):
            raise CalibrationError(f"{op} is a base-station calibration op")
        self._require_active("yaw")
        if op == "capture":
            if self._phase != "capturing" or self._next_point is None:
                raise CalibrationError("all 7 points captured — apply, or start again")
            label = cmd.point or self._next_point
            if label != self._next_point:
                raise CalibrationError(f"next point is {self._next_point!r}")
            self._capture_point(self._clock(), from_trigger=False)
        elif op == "apply":
            if self._phase != "done" or self._fitted is None:
                raise CalibrationError("capture all 7 points first")
            if self._checks:
                raise CalibrationError("fit checks failed: " + "; ".join(self._checks))
            self._persisted.yaw_deg = float(self._fitted)
            self._persisted.yaw_valid = True
            self._persisted.yaw_calibrated_at = self._wall()
            try:
                self._persisted.save(self._dir)
            except OSError as e:
                raise CalibrationError(f"cannot persist calibration: {e}") from None
            self._settings.update(yaw_deg=float(self._fitted))
            self._applied = float(self._fitted)
            self._finish("done", f"applied yaw {self._applied:.1f}°")
        elif op == "abort":
            self._yaw_points, self._next_point = [], None
            self._fitted = self._residual = None
            self._checks = []
            self._finish("aborted", "aborted")
        else:  # pragma: no cover
            raise CalibrationError(f"unknown op {op!r}")

    # -- flow bookkeeping (under _lock) ---------------------------------------------------
    def _begin(self, kind: str, phase: str, detail: str) -> None:
        self._active = True
        self._kind, self._phase, self._detail = kind, phase, detail
        self._started_at, self._finished_at = self._wall(), None
        self._scenes, self._lh, self._channels, self._stations_visible = 0, {}, set(), 0
        self._seen, self._snap_had_pose = set(), None
        self._controller_still = None
        self._validation = self._installed_path = self._backup_path = None
        self._yaw_points, self._next_point = [], None
        self._fitted = self._residual = self._applied = None
        self._checks = []
        self._tmp, self._ts, self._force, self._saw_force = None, "", False, False
        self._restart_mono = None
        self._val_t0, self._val_seq, self._val_pos, self._val_rx = None, -1, [], []
        self._recent.clear()
        self._last_seq = -1

    def _finish(self, phase: str, detail: str) -> None:
        self._active = False
        self._phase, self._detail = phase, detail
        self._finished_at = self._wall()
        if self._kind == "base_station":
            self._reader.on_info = None
        self._wake.set()

    def _fail(self, detail: str) -> None:
        logger.error("tracker calibration failed: %s", detail)
        with self._lock:
            kind = self._kind
            self._finish("failed", detail)
        if kind == "base_station":
            try:
                self._reader.restart(self._normal_args)
            except Exception:
                logger.exception("restoring normal libsurvive arguments failed")

    def _enqueue(self, job: Callable[[], None]) -> None:
        self._jobs.append(job)
        self._ensure_worker()
        self._wake.set()

    def _ensure_worker(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._run, name="tracker-calibration", daemon=True
            )
            self._thread.start()

    def _on_info(self, _t: float, text: str) -> None:
        """libsurvive C thread: queue only (no lock, never raises)."""
        self._info_queue.append(text)

    # -- worker -------------------------------------------------------------------------
    def _run(self) -> None:
        while True:
            job: Callable[[], None] | None = None
            with self._lock:
                if self._closing:
                    self._thread = None
                    return
                if self._jobs:
                    job = self._jobs.popleft()
                elif not self._active:
                    self._thread = None
                    return
            if job is not None:
                try:
                    job()
                except Exception as e:
                    self._fail(f"{e.__class__.__name__}: {e}")
                continue
            try:
                self._tick()
            except Exception:
                logger.exception("tracker calibration tick failed")
            self._wake.wait(WORKER_PERIOD_S)
            self._wake.clear()

    def _job_restart(self, args: list[str]) -> None:
        self._reader.restart(args)
        with self._lock:
            self._restart_mono = self._clock()
            self._recent.clear()
            self._last_seq = -1
            self._controller_still = None
            self._snap_had_pose = None  # the next snapshot is this run's baseline

    def _job_install(self) -> None:
        with self._lock:
            tmp, ts = self._tmp, self._ts
        assert tmp is not None
        backup: Path | None = None
        try:
            if not self._reader.stop():  # simple_close flushes the temporary config
                raise RuntimeError(
                    "tracker reader did not stop — libsurvive has not flushed the temporary config"
                )
            data = tmp.read_bytes()
            now_ts = time.strftime("%Y%m%d-%H%M%S", time.localtime(self._wall()))
            if self._config_path.exists():
                backup = self._config_path.with_name(f"{self._config_path.name}.bak-{now_ts}")
                shutil.copyfile(self._config_path, backup)
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            self._config_path.write_bytes(data)
            (self._dir / f"base_station-{ts}-installed.json").write_bytes(data)
            with self._lock:
                self._persisted.base_station_installed_at = self._wall()
                self._persisted.lighthouse_config_sha256 = hashlib.sha256(data).hexdigest()
                self._persisted.yaw_valid = False  # the lighthouse world was re-anchored
                self._persisted.save(self._dir)
        finally:
            self._reader.restart(self._normal_args)
        with self._lock:
            self._installed_path = str(self._config_path)
            self._backup_path = str(backup) if backup is not None else None
            self._finish("done", "installed — run Yaw alignment")

    def _tick(self) -> None:
        now = self._clock()
        with self._lock:
            if not self._active:
                return
            self._ingest_sample(now)
            if self._kind == "yaw":
                self._yaw_tick(now)
                return
        # base-station: merge the reader's lighthouse snapshot (taken outside our lock)
        try:
            snapshot = list(self._reader.lighthouses())
        except Exception:
            snapshot = None
        with self._lock:
            if not self._active or self._kind != "base_station":
                return
            self._drain_info()
            if snapshot is not None:
                self._merge_snapshot(snapshot)
            self._controller_still = self._still(now)
            self._bs_tick(now)

    # -- sample bookkeeping (under _lock) --------------------------------------------------
    def _latest(self) -> TrackerSample | None:
        got = self._slot.get()
        return got[0] if got is not None else None

    def _ingest_sample(self, now: float) -> None:
        s = self._latest()
        if s is None or s.seq == self._last_seq:
            return
        self._last_seq = s.seq
        if s.valid and (self._restart_mono is None or s.pose_rx_mono > self._restart_mono):
            self._recent.append(
                (s.pose_rx_mono, s.seq, np.array(s.pose.position), np.array(s.pose.orientation))
            )

    def _trigger_pressed(self, now: float) -> bool:
        s = self._latest()
        if s is None or s.controller is None or now - s.rx_mono > self._cfg.tracker.stale_s:
            return False
        return bool(s.controller.trigger_pressed)

    def _still(self, now: float) -> bool | None:
        win = self._ccfg.still_window_s
        pts = [p for rx, _, p, _ in self._recent if rx >= now - win]
        if len(pts) < 3 or now - self._recent[-1][0] > self._cfg.tracker.stale_s:
            return None
        std_mm = np.std(np.asarray(pts), axis=0) * 1000.0
        return bool(np.max(std_mm) < self._ccfg.still_threshold_mm)

    # -- yaw flow (under _lock) -----------------------------------------------------------
    def _yaw_prompt(self) -> str:
        nxt = self._next_point
        if nxt is None:
            return ""
        if nxt == "start":
            return "hold the controller at the START point and click the trigger (or Capture)"
        return f"move {nxt.upper()} 20–30 cm, hold still, then click the trigger (or Capture)"

    def _yaw_tick(self, now: float) -> None:
        if self._phase != "capturing":
            return
        pressed = self._trigger_pressed(now)
        if pressed and not self._prev_trigger:
            try:
                self._capture_point(now, from_trigger=True)
            except CalibrationError as e:
                self._detail = f"{e} — hold still and click again"
        self._prev_trigger = pressed

    def _capture_point(self, now: float, *, from_trigger: bool) -> None:
        self._ingest_sample(now)
        stale = self._cfg.tracker.stale_s
        if not self._recent or now - self._recent[-1][0] > stale:
            raise CalibrationError("no fresh tracker sample")
        win = self._ccfg.yaw_capture_average_s
        pts = [p for rx, _, p, _ in self._recent if rx >= now - win]
        pos = np.mean(np.asarray(pts), axis=0)
        quat = self._recent[-1][3]
        label = self._next_point
        assert label is not None
        self._yaw_points.append((label, Pose(pos, quat)))
        n = len(self._yaw_points)
        if n < len(YAW_POINT_ORDER):
            self._next_point = YAW_POINT_ORDER[n]
            self._detail = f"captured {label} — " + self._yaw_prompt()
            return
        self._next_point = None
        self._phase = "fitting"
        yaw, residual, checks = fit_yaw([p.position for _, p in self._yaw_points], self._ccfg)
        self._fitted, self._residual, self._checks = yaw, residual, list(checks)
        self._phase = "done"
        self._detail = (
            f"fit {yaw:.1f}° (residual {residual:.1f}°) — apply"
            if not checks
            else f"fit {yaw:.1f}° but checks failed: " + "; ".join(checks)
        )

    # -- base-station flow (under _lock) ---------------------------------------------------
    def _merge_snapshot(self, snapshot) -> None:
        """Fold the reader's LIGHTHOUSE snapshot in: serial / pose per station.
        libsurvive creates one object per station in the config file whether or
        not it is powered, so the snapshot LENGTH says nothing about visibility;
        a pose that APPEARS during this run (unsolved -> solved, i.e. the solver
        got light from the station) does — poses already present in the first
        snapshot after a restart were loaded from the file."""
        had_pose: dict[int, bool] = {}
        prev = self._snap_had_pose
        for lh in snapshot:
            idx = int(lh.index)
            ent = self._lh.setdefault(idx, _Lighthouse(idx))
            if lh.serial and not ent.serial:
                ent.serial = str(lh.serial)
            has_pose = lh.pose is not None
            if has_pose:
                ent.pose = lh.pose
                if prev is not None and prev.get(idx) is False:
                    self._seen.add(idx)
            had_pose[idx] = has_pose
        self._snap_had_pose = had_pose
        self._recount_visible()

    def _recount_visible(self) -> None:
        """``stations_visible`` = stations the solver got light from since ``start``
        (``_seen``) plus channels seen from light (``OOTX not set ... channel C``)
        attributed through the channel map — or counted as one extra station each
        when no known station has that channel."""
        by_channel = {lh.channel: idx for idx, lh in self._lh.items() if lh.channel is not None}
        seen = set(self._seen)
        extra = 0
        for ch in self._channels:
            idx = by_channel.get(ch)
            if idx is None:
                extra += 1
            else:
                seen.add(idx)
        self._stations_visible = len(seen) + extra

    def _drain_info(self) -> None:
        while self._info_queue:
            line = self._info_queue.popleft()
            if RE_FORCE_CALIBRATE.search(line):
                self._saw_force = True
                continue
            m = RE_GLOBAL_SOLVE.search(line)
            if m:
                n, idx = int(m.group(1)), int(m.group(2))
                ent = self._lh.setdefault(idx, _Lighthouse(idx))
                ent.scenes = n
                self._seen.add(idx)  # scenes exist only for stations the controller saw
                self._scenes = max(lh.scenes for lh in self._lh.values())
                continue
            m = RE_REFERENCE.search(line)
            if m:
                idx, serial = int(m.group(1)), m.group(2)
                for lh in self._lh.values():
                    lh.reference = False
                ent = self._lh.setdefault(idx, _Lighthouse(idx))
                ent.reference = True
                if not ent.serial:
                    ent.serial = serial
                continue
            m = RE_ADD_LH.search(line)
            if m:  # a station not in the config, created from its light: seen by definition
                ch, idx = int(m.group(1)), int(m.group(2))
                self._lh.setdefault(idx, _Lighthouse(idx)).channel = ch
                self._seen.add(idx)
                continue
            m = RE_OOTX_CHANNEL.search(line)
            if m:
                self._channels.add(int(m.group(1)))
                continue
            m = RE_OOTX_INDEX.search(line)
            if m:
                idx = int(m.group(1))
                self._lh.setdefault(idx, _Lighthouse(idx))
                self._seen.add(idx)
        self._recount_visible()

    def _first_pose_after_restart(self) -> float | None:
        if self._restart_mono is None or not self._recent:
            return None
        rx = self._recent[-1][0]
        return rx if rx > self._restart_mono else None

    def _bs_tick(self, now: float) -> None:
        ccfg = self._ccfg
        if self._phase == "starting":
            if (self._force and self._saw_force) or self._first_pose_after_restart() is not None:
                self._phase = "capturing"
        if self._phase == "capturing":
            self._detail = (
                f"scenes {self._scenes}/{ccfg.min_scenes} — park the controller still ≥ 3 s at "
                "another spot"
            )
            return
        if self._phase != "validating":
            return
        if self._restart_mono is None:
            return  # restart job still pending
        if self._val_t0 is None:
            first = self._first_pose_after_restart()
            if first is None:
                if now - self._restart_mono > VALIDATION_TRACKING_TIMEOUT_S:
                    self._validation = CalibrationValidation(
                        threshold_std_mm=ccfg.validation_std_mm,
                        threshold_step_mm=ccfg.validation_step_mm,
                    )
                    self._phase = "done"
                    self._detail = "validation failed — no tracking after the restart"
                else:
                    self._detail = "waiting for tracking — keep the controller still"
                return
            self._val_t0 = first
        t_collect = self._val_t0 + ccfg.validation_skip_seconds
        for rx, seq, p, _ in self._recent:
            if rx >= t_collect and seq > self._val_seq:
                self._val_seq = seq
                self._val_pos.append(p)
                self._val_rx.append(rx)
        t_end = t_collect + ccfg.validation_seconds
        if now < t_end:
            left = t_end - now
            self._detail = (
                f"validating — {left:.0f} s left, keep the controller still"
                if now >= t_collect
                else f"tracking — skipping the first {ccfg.validation_skip_seconds:.0f} s"
            )
            return
        n = len(self._val_pos)
        # The scatter of a controller that was NOT tracked is not evidence: a dropout
        # inside the window (gap between adjacent samples or at either edge) or a sample
        # count below what a tracked controller produces fails the validation outright.
        min_samples = max(2, math.ceil(ccfg.validation_seconds * VALIDATION_MIN_RATE_HZ))
        gap = ccfg.validation_seconds  # no samples at all: the whole window is the gap
        if n:
            times = np.asarray(self._val_rx)
            gap = float(
                max(times[0] - t_collect, t_end - times[-1], np.diff(times).max(initial=0.0))
            )
        lost = gap > VALIDATION_MAX_GAP_S
        passed = False
        if n >= 2:
            P = np.asarray(self._val_pos) * 1000.0
            std = np.std(P, axis=0)
            steps = np.linalg.norm(np.diff(P, axis=0), axis=1)
            max_step = float(steps.max()) if len(steps) else 0.0
            passed = bool(
                np.all(std < ccfg.validation_std_mm) and max_step < ccfg.validation_step_mm
                and n >= min_samples and not lost
            )
            self._validation = CalibrationValidation(
                samples=n,
                std_mm=(float(std[0]), float(std[1]), float(std[2])),
                max_step_mm=max_step,
                threshold_std_mm=ccfg.validation_std_mm,
                threshold_step_mm=ccfg.validation_step_mm,
                passed=passed,
            )
        else:
            self._validation = CalibrationValidation(
                samples=n,
                threshold_std_mm=ccfg.validation_std_mm,
                threshold_step_mm=ccfg.validation_step_mm,
            )
        self._phase = "done"
        if passed:
            self._detail = "validation passed — install"
        elif lost:
            self._detail = (
                f"validation failed — tracking lost ({gap:.1f} s gap, {n} samples); "
                "check the controller is seen and validate again"
            )
        elif n < min_samples:
            self._detail = (
                f"validation failed — only {n} samples (need ≥ {min_samples}); "
                "check tracking and validate again"
            )
        else:
            self._detail = "validation failed — capture more spots"


__all__ = [
    "CALIBRATION_ARGS",
    "PERSIST_FILE",
    "YAW_LEG_TARGETS",
    "YAW_POINT_ORDER",
    "CalibrationError",
    "PersistedCalibration",
    "TrackerCalibration",
    "apply_persisted_yaw",
    "capture_args",
    "fit_yaw",
    "lighthouse_channels",
    "strip_calibration_args",
    "validation_args",
]
