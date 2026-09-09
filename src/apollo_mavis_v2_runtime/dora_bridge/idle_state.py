"""``IdleArmReader`` — arm states between sessions (14-dora §4.2, §7, §8).

04-runtime §5 tears the workcell down at session end, so publishing ``arm_state``
(and the per-frame camera pose) for the process lifetime needs a session-less
state source. The reader runs its own thread at ``dora.publish.idle_state_hz``
and puts an idle :class:`StateSnapshot` (``tick == -1``,
``session_extra["source"] == "idle"``) into ``bus.snapshot`` — the slot the
control loop feeds in a session — so the publisher and the pose stamper work
identically in and out of sessions. It is paused before BRINGUP (its source
releases every connection) and resumed after TEARDOWN.

Sources (``IdleSource`` duck type: ``connect() / disconnect() / states() ->
dict[str, ArmState] / detail``):

- :class:`SimIdleSource` — the preview scene: the manager's parked joint vector
  (the last session's final ``q`` when the preview scene equals the session
  scene, the keyframe otherwise) run through this thread's own FK.
- :class:`MonitorIdleSource` — the phase-09a read-only ``HardwareStateMonitor``
  samples (zero additional SDK clients; the monitor already pauses itself for
  hardware sessions). Default on hardware (``dora.publish.idle_source: monitor``).
- :class:`DriverIdleSource` — one ``XArmDriver.connect(readonly=True)`` per arm
  (``idle_source: driver``; report stream + read-only polls, never a write),
  reconnecting with a 1 -> 10 s backoff; a lost box marks that arm ``stale``
  with its last ``q`` frozen.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import numpy as np
from apollo_mavis_v2_core import ArmState, CollisionReport, GripperState, Pose, se3

from ..control.snapshot import StateSnapshot

logger = logging.getLogger(__name__)

IDLE_TICK = -1
STALE_AFTER_S = 1.0  # a sample older than this freezes the arm as stale
BACKOFF = (1.0, 10.0)
IdleStatus = str  # "off" | "running" | "paused" | "stale"


class IdleSource(Protocol):
    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def states(self) -> dict[str, ArmState]: ...

    @property
    def detail(self) -> str: ...


def _state(
    arm_id: str,
    q: np.ndarray,
    kin: Any,
    *,
    has_rail: bool,
    gripper: float | None,
    stale: bool,
    error_code: int = 0,
    warn_code: int = 0,
    t_mono: float | None = None,
    dq: np.ndarray | None = None,
) -> ArmState:
    q = np.asarray(q, dtype=np.float64)
    if has_rail and q.shape[0] == 7:
        q = np.append(q, 0.0)
    if not has_rail:
        q = q[:7]
    ee = kin.tcp_base(arm_id, q) if kin is not None else Pose.identity()
    now = time.monotonic() if t_mono is None else t_mono
    return ArmState(
        arm_id=arm_id,
        q=q,
        dq=np.zeros_like(q) if dq is None else np.asarray(dq, dtype=np.float64),
        ee_pose=ee,
        gripper=GripperState(
            open_frac=1.0 if gripper is None else min(1.0, max(0.0, float(gripper)))
        ),
        rail_pos_m=float(q[7]) if has_rail else None,
        error_code=int(error_code),
        warn_code=int(warn_code),
        mode=0,
        state=0,
        stale=bool(stale),
        t_mono=now,
        wallclock_ns=time.time_ns(),
    )


class SimIdleSource:
    """Preview-scene states from the manager's parked ``q`` per arm."""

    def __init__(
        self,
        q_provider: Callable[[], Mapping[str, np.ndarray]],
        kin: Any,
        has_rail: Mapping[str, bool],
    ) -> None:
        self._q = q_provider
        self._kin = kin
        self._has_rail = dict(has_rail)
        self.detail = "sim preview scene"

    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def states(self) -> dict[str, ArmState]:
        out: dict[str, ArmState] = {}
        for arm_id, q in self._q().items():
            has_rail = self._has_rail.get(arm_id, len(q) > 7)
            out[arm_id] = _state(arm_id, q, self._kin, has_rail=has_rail, gripper=None, stale=False)
        return out


class MonitorIdleSource:
    """``HardwareStateMonitor`` samples -> ``ArmState`` (read-only by construction)."""

    def __init__(
        self,
        monitor: Any,
        kin: Any,
        has_rail: Mapping[str, bool],
        rail_fallback_m: Mapping[str, float],
        *,
        rail_flip: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.monitor = monitor
        self._kin = kin
        self._has_rail = dict(has_rail)
        self._fallback = dict(rail_fallback_m)
        self._flip = rail_flip
        self._clock = clock
        self._last: dict[str, ArmState] = {}
        self.detail = "hardware monitor samples"

    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def states(self) -> dict[str, ArmState]:
        samples = self.monitor.snapshot() or {}
        now = self._clock()
        out: dict[str, ArmState] = {}
        for arm_id, has_rail in self._has_rail.items():
            s = samples.get(arm_id)
            status = (
                self.monitor.status_of(arm_id)[0]
                if hasattr(self.monitor, "status_of")
                else "running"
            )
            if s is None:
                if arm_id in self._last:
                    out[arm_id] = _frozen(self._last[arm_id])
                continue
            q7 = np.asarray(list(s.q)[:7], dtype=np.float64)
            if has_rail:
                pos = getattr(s, "rail_pos_m", None)
                if pos is None:
                    pos = float(self._fallback.get(arm_id, 0.0))
                elif self._flip:
                    pos = se3.RAIL_TRAVEL_M - float(pos)
                q = np.append(q7, min(se3.RAIL_TRAVEL_M, max(0.0, float(pos))))
            else:
                q = q7
            stale = status != "running" or (now - float(s.t_mono)) > STALE_AFTER_S
            st = _state(
                arm_id,
                q,
                self._kin,
                has_rail=has_rail,
                gripper=getattr(s, "gripper_open_frac", None),
                stale=stale,
                error_code=int(getattr(s, "error_code", 0) or 0),
                warn_code=int(getattr(s, "warn_code", 0) or 0),
                t_mono=float(s.t_mono),
            )
            self._last[arm_id] = st
            out[arm_id] = st
        return out


class DriverIdleSource:
    """One read-only ``XArmDriver`` per arm (``connect(readonly=True)``)."""

    def __init__(
        self,
        driver_factory: Callable[[str], Any],
        arm_ids: Sequence[str],
        kin: Any,
        has_rail: Mapping[str, bool],
        gripper_arms: Sequence[str],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._factory = driver_factory  # arm_id -> driver (not yet connected)
        self.arm_ids = list(arm_ids)
        self._kin = kin
        self._has_rail = dict(has_rail)
        self._gripper_arms = set(gripper_arms)
        self._clock = clock
        self.drivers: dict[str, Any] = {}
        self._retry_at: dict[str, float] = {}
        self._backoff: dict[str, float] = {}
        self._last: dict[str, ArmState] = {}
        self.detail = "read-only drivers"
        self.connect_attempts = 0

    def connect(self) -> None:
        for arm_id in self.arm_ids:
            self._try_connect(arm_id)

    def _try_connect(self, arm_id: str) -> None:
        if arm_id in self.drivers:
            return
        now = self._clock()
        if now < self._retry_at.get(arm_id, 0.0):
            return
        self.connect_attempts += 1
        try:
            drv = self._factory(arm_id)
            drv.connect(readonly=True)
        except Exception as exc:  # noqa: BLE001 - box off / cable out
            delay = self._backoff.get(arm_id, BACKOFF[0])
            self._backoff[arm_id] = min(delay * 2.0, BACKOFF[1])
            self._retry_at[arm_id] = now + delay
            self.detail = (
                f"{arm_id}: read-only connect failed ({type(exc).__name__}: {exc}); "
                f"retry in {delay:.0f} s"
            )
            logger.warning("idle reader: %s", self.detail)
            return
        self.drivers[arm_id] = drv
        self._backoff.pop(arm_id, None)
        self._retry_at.pop(arm_id, None)
        self.detail = "read-only drivers connected"

    def disconnect(self) -> None:
        drivers, self.drivers = self.drivers, {}
        for drv in drivers.values():
            try:
                drv.disconnect()
            except Exception:  # noqa: BLE001
                logger.exception("idle reader: read-only disconnect failed")
        self._retry_at.clear()
        self._backoff.clear()

    def states(self) -> dict[str, ArmState]:
        out: dict[str, ArmState] = {}
        for arm_id in self.arm_ids:
            drv = self.drivers.get(arm_id)
            if drv is None:
                self._try_connect(arm_id)
                drv = self.drivers.get(arm_id)
            if drv is None:
                if arm_id in self._last:
                    out[arm_id] = _frozen(self._last[arm_id])
                continue
            try:
                st = drv.get_state()
            except Exception as exc:  # noqa: BLE001 - lost box: drop the driver, back off
                logger.warning("idle reader: %s get_state failed: %s", arm_id, exc)
                self.drivers.pop(arm_id, None)
                try:
                    drv.disconnect()
                except Exception:  # noqa: BLE001
                    pass
                self._retry_at[arm_id] = self._clock() + BACKOFF[0]
                if arm_id in self._last:
                    out[arm_id] = _frozen(self._last[arm_id])
                continue
            has_rail = self._has_rail.get(arm_id, st.has_rail)
            grip = float(st.gripper.open_frac) if arm_id in self._gripper_arms else None
            state = _state(
                arm_id,
                st.q,
                self._kin,
                has_rail=has_rail,
                gripper=grip,
                stale=bool(st.stale),
                error_code=st.error_code,
                warn_code=st.warn_code,
                t_mono=float(st.t_mono),
                dq=None if st.stale else st.dq,
            )
            if state.stale and arm_id in self._last:
                state = _frozen(self._last[arm_id])
            self._last[arm_id] = state
            out[arm_id] = state
        return out


def _frozen(st: ArmState) -> ArmState:
    """The last good state, marked stale, velocities zero (the pose is frozen)."""
    return ArmState(
        arm_id=st.arm_id,
        q=st.q,
        dq=np.zeros_like(st.q),
        ee_pose=st.ee_pose,
        gripper=st.gripper,
        rail_pos_m=st.rail_pos_m,
        error_code=st.error_code,
        warn_code=st.warn_code,
        mode=st.mode,
        state=st.state,
        stale=True,
        t_mono=st.t_mono,
        wallclock_ns=st.wallclock_ns,
    )


class IdleArmReader:
    """Publish idle snapshots at ``idle_state_hz`` while no session exists."""

    def __init__(
        self,
        bus: Any,
        source: IdleSource,
        *,
        hz: float,
        kind: str,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.bus = bus
        self.source = source
        self.period = 1.0 / float(hz)
        self.kind = kind
        self._clock = clock
        self._thread: threading.Thread | None = None
        self._running = False
        self._paused = threading.Event()  # set = paused
        self._resumed = threading.Event()  # set = may publish
        self._resumed.set()
        self._lock = threading.Lock()
        self._status: IdleStatus = "off"
        self.snapshots = 0
        self.last_states: dict[str, ArmState] = {}

    # -- lifecycle -----------------------------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._status = "running"
        self._thread = threading.Thread(target=self._run, name="dora-idle-arms", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._resumed.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        try:
            self.source.disconnect()
        except Exception:  # noqa: BLE001
            logger.exception("idle reader: source disconnect failed")
        self._status = "off"

    def pause(self) -> None:
        """Before BRINGUP: stop publishing and release every connection (synchronous)."""
        self._paused.set()
        self._resumed.clear()
        with self._lock:  # wait for an in-flight states() to finish
            try:
                self.source.disconnect()
            except Exception:  # noqa: BLE001
                logger.exception("idle reader: pause disconnect failed")
            self._status = "paused" if self._running else "off"

    def resume(self) -> None:
        """After TEARDOWN: reconnect and publish again (<= one period later)."""
        self._paused.clear()
        if self._running:
            self._status = "running"
        self._resumed.set()

    @property
    def status(self) -> IdleStatus:
        return self._status

    @property
    def detail(self) -> str:
        return str(getattr(self.source, "detail", ""))

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    # -- thread ------------------------------------------------------------------------------------
    def _run(self) -> None:
        next_t = self._clock()
        connected = False
        while self._running:
            if self._paused.is_set():
                connected = False
                self._resumed.wait(timeout=0.2)
                next_t = self._clock()
                continue
            with self._lock:
                if self._paused.is_set():
                    continue
                try:
                    if not connected:
                        self.source.connect()
                        connected = True
                    states = self.source.states()
                except Exception:  # noqa: BLE001 - never die
                    logger.exception("idle reader: states() failed")
                    states = {}
                if states:
                    self._publish(states)
            next_t += self.period
            delay = next_t - self._clock()
            if delay > 0:
                time.sleep(min(delay, self.period))
            else:
                next_t = self._clock()

    def _publish(self, states: dict[str, ArmState]) -> None:
        now = self._clock()
        stale = any(st.stale for st in states.values())
        self._status = "stale" if stale else "running"
        self.last_states = dict(states)
        snap = StateSnapshot(
            t_mono=now,
            wallclock_ns=time.time_ns(),
            tick=IDLE_TICK,
            arms=states,
            q_cmd={a: np.array(st.q) for a, st in states.items()},
            active_arm=None,
            gate=CollisionReport.ok(),
            clearances=[],
            gripper_frac={a: float(st.gripper.open_frac) for a, st in states.items()},
            episode=None,
            watchdog_tripped=False,
            plan_status={},
            session_extra={"source": "idle", "kind": self.kind},
        )
        self.snapshots += 1
        self.bus.snapshot.put(snap)


__all__ = [
    "IDLE_TICK",
    "DriverIdleSource",
    "IdleArmReader",
    "IdleSource",
    "MonitorIdleSource",
    "SimIdleSource",
]
