"""Scriptable driver-event fakes for the FAULT -> RECOVERING -> RUNNING plumbing
(phase-09b; 04-runtime §15 / §16 "FakeWorkcell ... scriptable error codes").

The event classes mirror ``apollo_mavis_v2_hardware.events`` field-for-field and
BY NAME: the control loop dispatches on ``type(ev).__name__`` so the runtime never
imports the optional hardware package, and ``test_fault_recovery_loop.py`` pins
these names / fields against the hardware package whenever it is importable.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Literal

import numpy as np
from apollo_mavis_v2_core import CommandError
from apollo_mavis_v2_core.testing import FakeArm, FakeCamera, FakeWorkcell


@dataclass(frozen=True)
class FaultEvent:
    arm_id: str
    source: str  # "servo" | "monitor" | "report" | "external" | "user_stop" | "user" | "latch"
    code: int
    error_code: int = 0
    warn_code: int = 0
    detail: str = ""
    t_mono: float = 0.0


@dataclass(frozen=True)
class RecoveredEvent:
    arm_id: str
    error_code: int
    t_mono: float = 0.0


@dataclass(frozen=True)
class ReseedEvent:
    arm_id: str
    q: tuple[float, ...]
    t_mono: float = 0.0


@dataclass(frozen=True)
class StudioConflictWarning:
    arm_id: str
    mode: int
    state: int
    detail: str = "close UFACTORY Studio live control"
    t_mono: float = 0.0


@dataclass(frozen=True)
class RecoveryResult:
    """Mirror of ``apollo_mavis_v2_hardware.RecoveryResult``."""

    seq: int
    ok: bool
    error_code: int
    detail: str = ""
    user_initiated: bool = False
    t_mono: float = 0.0


@dataclass(frozen=True)
class SettingResult:
    """Mirror of ``apollo_mavis_v2_hardware.SettingResult`` (2026-09-11)."""

    seq: int
    ok: bool
    code: int
    level: int
    t_mono: float = 0.0
    detail: str = ""


class EventFakeWorkcell(FakeWorkcell):
    """``FakeWorkcell`` + the hardware workcell's event / recovery surface.

    ``queue(*events)`` scripts events for the next ``drain_events()``;
    ``fault(arm_id, code)`` injects a controller error the way the driver would
    (``ArmState.error_code`` + a FaultEvent); ``request_recovery(arm_id)``
    emulates ``HardwareWorkcell.request_recovery``: after ``recovery_latency_s``
    (0 = synchronously, on the caller's thread) it clears the arm's error and
    queues ``FaultEvent(source="user") -> ReseedEvent -> RecoveredEvent`` plus a
    ``RecoveryResult(ok=True)`` - or, when ``latch_next[arm_id]`` holds a reason,
    a latch ``FaultEvent`` plus ``RecoveryResult(ok=False, detail=reason)`` and
    the error stays latched. ``request_set_collision_sensitivity(arm_id, level)``
    (2026-09-11) mirrors ``HardwareWorkcell.request_set_collision_sensitivity``:
    after the same ``recovery_latency_s`` it publishes a :class:`SettingResult`
    (``setting_result(arm_id)``) whose code is ``setting_code_next.pop(arm_id, 0)``
    - 0 clean, 1 / 2 / 9 a status echo (``ok`` with the driver's note), anything
    else a failure; ``setting_requests`` logs ``(arm_id, level)``.
    """

    def __init__(
        self,
        arms: dict[str, FakeArm] | None = None,
        cameras: dict[str, FakeCamera] | None = None,
        kind: Literal["hardware", "sim"] = "hardware",
        *,
        recovery_latency_s: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(arms, cameras if cameras is not None else {}, kind)
        self._events: list[object] = []
        self._lock = threading.Lock()
        self.recovery_latency_s = float(recovery_latency_s)
        self._clock = clock
        self.latch_next: dict[str, str] = {}  # arm -> latch reason for the next recovery
        self._results: dict[str, RecoveryResult] = {}
        self._seq: dict[str, int] = {}
        self.recovery_requests: list[str] = []
        self.drains = 0
        # 2026-09-11 collision-sensitivity channel
        self.setting_requests: list[tuple[str, int]] = []
        self.setting_code_next: dict[str, int] = {}  # arm -> SDK code of the next write
        self._setting_results: dict[str, SettingResult] = {}
        self._setting_seq: dict[str, int] = {}

    # -- events -------------------------------------------------------------------------
    def queue(self, *events: object) -> None:
        with self._lock:
            self._events.extend(events)

    def drain_events(self) -> list[object]:
        with self._lock:
            out, self._events = self._events, []
        self.drains += 1
        return out

    def fault(
        self,
        arm_id: str,
        error_code: int = 24,
        *,
        source: str = "monitor",
        detail: str = "",
        code: int | None = None,
    ) -> FaultEvent:
        """Latch ``error_code`` on the arm and queue the matching FaultEvent."""
        arm = self.arms[arm_id]
        arm.inject_error(error_code)  # type: ignore[attr-defined]
        ev = FaultEvent(
            arm_id,
            source=source,
            code=error_code if code is None else code,
            error_code=error_code,
            detail=detail,
            t_mono=self._clock(),
        )
        self.queue(ev)
        return ev

    # -- recovery channel (HardwareWorkcell surface) -------------------------------------
    def request_recovery(self, arm_id: str) -> None:
        if arm_id not in self.arms:
            raise KeyError(arm_id)
        if not self.started:
            raise CommandError(f"{arm_id}: driver not connected")
        self.recovery_requests.append(arm_id)
        if self.recovery_latency_s > 0.0:
            timer = threading.Timer(self.recovery_latency_s, self._complete_recovery, (arm_id,))
            timer.daemon = True
            timer.start()
        else:
            self._complete_recovery(arm_id)

    def _complete_recovery(self, arm_id: str) -> None:
        arm = self.arms[arm_id]
        err = int(arm.get_state().error_code)
        now = self._clock()
        seq = self._seq.get(arm_id, 0) + 1
        self._seq[arm_id] = seq
        reason = self.latch_next.pop(arm_id, None)
        events: list[object] = [FaultEvent(arm_id, "user", code=0, error_code=err, t_mono=now)]
        if reason is None:
            arm.clear_errors()  # type: ignore[attr-defined]
            q = tuple(float(v) for v in arm.get_state().q)
            events += [ReseedEvent(arm_id, q, now), RecoveredEvent(arm_id, err, now)]
            result = RecoveryResult(seq, True, err, "", True, now)
        else:
            events.append(FaultEvent(arm_id, "latch", 0, error_code=err, detail=reason, t_mono=now))
            result = RecoveryResult(seq, False, err, reason, True, now)
        self.queue(*events)
        self._results[arm_id] = result

    def recovery_result(self, arm_id: str) -> RecoveryResult | None:
        return self._results.get(arm_id)

    # -- collision-sensitivity channel (HardwareWorkcell surface, 2026-09-11) ------------
    def request_set_collision_sensitivity(self, arm_id: str, level: int) -> None:
        if arm_id not in self.arms:
            raise KeyError(arm_id)
        if not self.started:
            raise CommandError(f"{arm_id}: driver not connected")
        if isinstance(level, bool) or int(level) != level or int(level) not in (1, 2, 3):
            raise CommandError(f"{arm_id}: collision sensitivity must be 1, 2 or 3 (got {level!r})")
        self.setting_requests.append((arm_id, int(level)))
        if self.recovery_latency_s > 0.0:
            timer = threading.Timer(
                self.recovery_latency_s, self._complete_setting, (arm_id, int(level))
            )
            timer.daemon = True
            timer.start()
        else:
            self._complete_setting(arm_id, int(level))

    def _complete_setting(self, arm_id: str, level: int) -> None:
        code = int(self.setting_code_next.pop(arm_id, 0))
        seq = self._setting_seq.get(arm_id, 0) + 1
        self._setting_seq[arm_id] = seq
        ok = code == 0 or code in (1, 2, 9)
        if code == 0:
            detail = ""
        elif ok:
            detail = (
                f"set_collision_sensitivity returned {code} (status echo: the controller has "
                "an error / warning latched or is not ready; the write went through - the "
                "read-only monitor verifies the value once it holds the box again)"
            )
        else:
            detail = f"set_collision_sensitivity returned {code}"
        self._setting_results[arm_id] = SettingResult(seq, ok, code, level, self._clock(), detail)

    def setting_result(self, arm_id: str) -> SettingResult | None:
        return self._setting_results.get(arm_id)


@dataclass(frozen=True)
class FakeRailHomeOutcome:
    """Duck-typed ``apollo_mavis_v2_hardware.RailHomeOutcome`` (same field names)."""

    ok: bool
    detail: str
    written: bool = True
    phase: str = "READY"
    on_zero: int | None = 1
    is_enabled: int | None = 1
    error: int | None = 0
    pos_m: float | None = 0.0
    sdk_codes: dict[str, int] | None = None
    duration_s: float = 0.0


@dataclass(frozen=True)
class RailEvent:
    """Mirror of ``apollo_mavis_v2_hardware.events.RailEvent``."""

    arm_id: str
    phase: str
    code: int = 0
    detail: str = ""
    t_mono: float = 0.0


@dataclass(frozen=True)
class FakeServoLimits:
    """Duck-typed hardware ``ServoLimits`` (same field names) with the EFFECTIVE,
    i.e. already speed-scaled, caps a connected ``XArmDriver.cfg.servo`` carries.
    Defaults = the hardware package's ORIGINAL first-run caps at scale 0.1 (0.03
    rad/s per joint, 0.2 mm per 10 ms tick over the conservative lever arms; the
    real defaults doubled on 2026-09-07 - these stay put, the tests pin them)."""

    rate_hz: float = 100.0
    max_joint_vel: tuple[float, ...] = (0.03,) * 7
    max_joint_acc: tuple[float, ...] = (2.0,) * 7
    lever_arm_m: tuple[float, ...] = (1.20, 1.20, 1.00, 0.75, 0.44, 0.30, 0.10)
    max_cart_step_m: float = 0.0002


class FakeRailDriverArm(FakeArm):
    """``FakeArm`` + the ``XArmDriver`` rail surface the phase-09d rail-homing job
    drives: ``rail_position_known`` / ``rail_phase`` (``DETECTED`` while unhomed -
    the published rail slot is then the driver's 0.0 placeholder, whatever the
    carriage really does), ``home_rail()`` (blocks ``homing_duration_s`` on the
    caller's thread, then homed + enabled at 0.0 m unless ``homing_fails``), a
    dropped rail slot while unhomed (like the driver) and ``command_rail``
    refusing. ``instant`` makes ``command_joints`` land immediately (a perfectly
    tracking servo stream, so the job's posture checks see the commanded joints
    without anybody stepping the fake). ``servo`` (a :class:`FakeServoLimits`)
    exposes ``cfg.servo`` like a real driver - the runtime derives its executor
    caps from it - and, with ``instant=False``, models the driver's
    ``_ServoStreamer`` on a thread at ``servo.rate_hz`` from ``connect()`` to
    ``disconnect()``: per-joint velocity clip, per-joint acceleration clip and
    the lever-weighted Cartesian scaling toward the latest ``command_joints``
    target; every streamed configuration is appended to ``path_log``."""

    def __init__(
        self,
        arm_id: str,
        *,
        q0,
        rail_homed: bool = False,
        homing_duration_s: float = 0.0,
        homing_fails: str | None = None,
        instant: bool = True,
        events: list | None = None,
        servo: FakeServoLimits | None = None,
        **kw,
    ) -> None:
        super().__init__(arm_id, has_rail=True, q0=q0, **kw)
        self._homed = bool(rail_homed)
        self.homing_duration_s = float(homing_duration_s)
        self.homing_fails = homing_fails
        self.instant = bool(instant)
        self.events = events if events is not None else []
        self.home_rail_calls: list[tuple[float, ...]] = []  # q at each home_rail()
        self.rail_targets_dropped = 0
        self.command_log: list[tuple[float, ...]] = []
        self.servo = servo
        if servo is not None:
            self.cfg = SimpleNamespace(servo=servo)  # what executor_caps_for reads
        self.path_log: list[np.ndarray] = []  # streamed q per streamer tick (instant=False)
        self._stream_lock = threading.Lock()
        self._streaming = False
        self._stream_thread: threading.Thread | None = None

    # -- the servo streamer model (instant=False) -------------------------------------------
    def connect(self) -> None:
        super().connect()
        if not self.instant and self.servo is not None and self._stream_thread is None:
            self._streaming = True
            self._stream_thread = threading.Thread(
                target=self._stream, name=f"fake-servo-{self.arm_id}", daemon=True
            )
            self._stream_thread.start()

    def disconnect(self) -> None:
        self._streaming = False
        t = self._stream_thread
        if t is not None:
            t.join(1.0)
            self._stream_thread = None
        super().disconnect()

    def _stream(self) -> None:
        servo = self.servo
        assert servo is not None
        dt = 1.0 / float(servo.rate_hz)
        vel_step = np.asarray(servo.max_joint_vel, dtype=np.float64) * dt
        acc_step = np.asarray(servo.max_joint_acc, dtype=np.float64) * dt * dt
        lever = np.asarray(servo.lever_arm_m, dtype=np.float64)
        prev_dq = np.zeros(7)
        next_t = time.monotonic() + dt  # fixed-rate schedule like the driver's streamer:
        while self._streaming:  # sleep to the next slot, re-anchor after a stall, no bursts
            time.sleep(max(0.0, next_t - time.monotonic()))
            now = time.monotonic()
            if now - next_t > 2 * dt:
                next_t = now
            next_t += dt
            with self._stream_lock:
                q = self._q.copy()
                target = self._target.copy()
                dq = np.clip(target[:7] - q[:7], -vel_step, vel_step)
                dq = np.clip(dq, prev_dq - acc_step, prev_dq + acc_step)
                cart = float(np.sum(np.abs(dq) * lever))
                if cart > servo.max_cart_step_m:
                    dq = dq * (servo.max_cart_step_m / cart)
                q[:7] = q[:7] + dq
                prev_dq = dq
                self._q = q
                self.path_log.append(q.copy())

    @property
    def rail_position_known(self) -> bool:
        return self._homed

    @property
    def rail_phase(self) -> str:
        return "READY" if self._homed else "DETECTED"

    def command_joints(self, q) -> None:
        q = np.asarray(q, dtype=np.float64).copy()
        self.command_log.append(tuple(float(v) for v in q))
        if not self._homed and q.shape[0] > 7:
            if abs(float(q[7]) - float(self._q[7])) > 1e-9:
                self.rail_targets_dropped += 1
            q[7] = self._q[7]  # the driver drops the rail slot while unhomed
        with self._stream_lock:
            super().command_joints(q)
            if self.instant:
                self._q = self._target.copy()

    def command_rail(self, pos_m: float) -> None:
        if not self._homed:
            raise CommandError(f"{self.arm_id}: linear track not homed (position unknown)")
        super().command_rail(pos_m)

    def get_state(self):
        st = super().get_state()
        if self._homed:
            return st
        q = np.array(st.q)
        q[7] = 0.0  # UNKNOWN_RAIL_POS_M placeholder while unhomed
        return replace(st, q=q, rail_pos_m=0.0)

    def home_rail(self) -> FakeRailHomeOutcome:
        self.home_rail_calls.append(tuple(float(v) for v in self._q[:7]))
        if self.homing_duration_s > 0.0:
            time.sleep(self.homing_duration_s)
        codes = {
            "set_linear_track_back_origin": 0,
            "set_linear_track_enable": 0,
            "set_linear_track_speed": 0,
        }
        if self.homing_fails:
            self.events.append(RailEvent(self.arm_id, "RAIL_ERROR", 80, self.homing_fails))
            return FakeRailHomeOutcome(
                False,
                f"{self.arm_id}: rail homing failed: {self.homing_fails}",
                phase="RAIL_ERROR",
                on_zero=0,
                is_enabled=0,
                error=80,
                pos_m=None,
                sdk_codes={**codes, "set_linear_track_back_origin": 80},
            )
        self._homed = True
        self._q[7] = 0.0
        self._target[7] = 0.0
        detail = (
            f"{self.arm_id}: rail homed: carriage at 0.000 m, track enabled, positioning "
            "speed 5 mm/s"
        )
        self.events.append(RailEvent(self.arm_id, "READY", 0, detail))
        return FakeRailHomeOutcome(True, detail, sdk_codes=codes)


@dataclass
class FakeBringupStatus:
    """Duck-typed ``apollo_mavis_v2_hardware.ArmBringupStatus`` (same field names)."""

    arm_id: str
    network: str = "pending"
    connected: bool = False
    fw_version: str | None = None
    sn: str | None = None
    rail: str = "unknown"
    gripper: str = "unknown"
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def model_copy(self, deep: bool = False) -> FakeBringupStatus:
        return replace(self, warnings=list(self.warnings))


class HardwareFakeWorkcell(EventFakeWorkcell):
    """``EventFakeWorkcell`` + the ``HardwareWorkcell.bring_up(status_cb,
    timeout_s)`` surface the phase-09c hardware bring-up drives (04-runtime §5).

    ``bring_up`` connects every arm (``start()``), streams one
    :class:`FakeBringupStatus` per stage through ``status_cb`` and returns the
    final statuses. Knobs: ``fail`` = ``{arm_id: "[step] message"}`` makes that
    arm fail (``connected False``, ``rail 'unhomed'`` for a ``[rail]`` error);
    ``rail_status`` = ``{arm_id: "error" | "detected" | ...}`` overrides the rail
    value of a CONNECTED arm (``error`` = a driver that latched RAIL_ERROR at
    connect and kept going: ``connected True``, ``error None`` - the runtime must
    refuse it); ``bringup_delay_s`` holds the connect (the monitor-supervisor race test) and
    ``on_bring_up`` is called mid-connect (a probe of the runtime's state while the
    bring-up is in flight); ``fw_version`` / ``sn`` are echoed. ``states()`` stamps
    ``t_mono`` with the live clock (like a driver's 30003 report) so the
    ``ArmReportWatchdog`` sees fresh states. ``stop()`` is the D6 hand-back
    (counted by the base class).
    """

    def __init__(
        self,
        *args,
        fail: dict[str, str] | None = None,
        rail_status: dict[str, str] | None = None,
        bringup_delay_s: float = 0.0,
        on_bring_up: Callable[[], None] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.fail = dict(fail or {})
        self.rail_status = dict(rail_status or {})
        self.bringup_delay_s = float(bringup_delay_s)
        self.on_bring_up = on_bring_up
        self.bringup_calls: list[float] = []  # timeout_s per call
        self.statuses: dict[str, FakeBringupStatus] = {}
        self._follow_lock = threading.Lock()
        self._followed_at: float | None = None

    def states(self):
        now = self._clock()
        self._follow(now)
        return {
            arm_id: replace(arm.get_state(), t_mono=now, wallclock_ns=time.time_ns())
            for arm_id, arm in self.arms.items()
        }

    def _follow(self, now: float) -> None:
        """A real driver's servo stream follows the command between two state reads; a
        plain ``FakeArm`` only moves when stepped and nothing steps it in a live session.
        Step every plain ``FakeArm`` by the wall time since the last read (measured q
        trails the command at the fake's 1 rad/s / 0.2 m/s), so the manager's
        measured-arrival hand-over (2026-09-08) sees the arms arrive. A
        ``FakeRailDriverArm`` has its own instant / servo-streamer model and is left
        alone. Bounded to 0.1 s per read (a paused test must not teleport the fake)."""
        with self._follow_lock:
            last, self._followed_at = self._followed_at, now
            if last is None:
                return
            dt = min(now - last, 0.1)
            if dt <= 0.0:
                return
            for arm in self.arms.values():
                if type(arm) is FakeArm:
                    arm.step(dt)

    def bring_up(self, status_cb=None, timeout_s: float = 180.0) -> dict[str, FakeBringupStatus]:
        self.bringup_calls.append(float(timeout_s))
        statuses = {
            a: FakeBringupStatus(
                arm_id=a, network="ok", warnings=["netsetup skipped (no NetSetup wired)"]
            )
            for a in self.arms
        }

        def push(st: FakeBringupStatus) -> None:
            if status_cb is not None:
                status_cb(st.model_copy(deep=True))

        for st in statuses.values():
            push(st)
        if self.bringup_delay_s > 0.0:
            time.sleep(self.bringup_delay_s)
        if self.on_bring_up is not None:
            self.on_bring_up()
        for arm_id, arm in self.arms.items():
            st = statuses[arm_id]
            error = self.fail.get(arm_id)
            if error is not None:
                st.error = error
                if error.startswith("[rail]"):
                    st.rail = "unhomed"
                elif error.startswith("[gripper]"):
                    st.gripper = "error"
                push(st)
                continue
            arm.connect()
            st.fw_version, st.sn = "1.12.10", f"FAKE-{arm_id}"
            if getattr(arm, "has_rail", False):
                # a FakeRailDriverArm connected unhomed reports "unhomed" (09d allow_unhomed)
                default_rail = "ready" if getattr(arm, "rail_position_known", True) else "unhomed"
            else:
                default_rail = "none"
            st.rail = self.rail_status.get(arm_id, default_rail)
            st.gripper = "xarm_g2" if arm_id == "grip" else "none"
            push(st)
            st.connected = True
            push(st)
        self.started = True
        self.start_calls += 1
        self.statuses = statuses
        return {a: s.model_copy(deep=True) for a, s in statuses.items()}


__all__ = [
    "EventFakeWorkcell",
    "FakeBringupStatus",
    "FakeRailDriverArm",
    "FakeRailHomeOutcome",
    "FakeServoLimits",
    "FaultEvent",
    "HardwareFakeWorkcell",
    "RailEvent",
    "RecoveredEvent",
    "RecoveryResult",
    "ReseedEvent",
    "StudioConflictWarning",
]
