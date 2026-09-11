"""Hardware-session helpers for ``SessionManager._bringup_hardware`` (phase-09c;
04-runtime §5 BRINGUP).

Pure functions and small adapters, kept out of ``manager.py`` so they can be
unit-tested without a Runtime:

* :func:`scale_control_config` / :func:`scale_driver_config` - the
  ``SessionSpec.speed_scale`` (D2) applied host-side (teleop rates, target
  rate, ``dq_max_rad``, jog slews) and driver-side (``servo.max_joint_vel``,
  ``servo.max_cart_step_m``, ``rail_speed_mm_s``; duck-typed on the hardware
  package's pydantic ``XArmDriverConfig`` so the runtime never imports it);
* :class:`RailFlipWorkcell` / :class:`RailFlipArm` - ``hardware_session.rail_flip``
  applied at the workcell boundary (``q_sim = 0.65 - q_track`` on every
  ``get_state`` and back on every ``command_joints`` / ``command_rail``), so
  the twin, IK, gate and loop all live in the twin's rail convention and the
  alignment overlay (which applies the flip itself) reads the INNER workcell;
* :class:`RailHoldWorkcell` / :class:`RailHoldArm` - phase-09d: the rail-homing
  job's adapter for an arm connected with its track UNHOMED (``rail_homing:
  allow_unhomed``): ``get_state`` shows the configured ``rail_fallback_m`` in
  the rail slot while the driver's ``rail_position_known`` is False (the
  driver publishes a 0.0 placeholder the twin must not believe) and
  ``command_joints`` / ``command_rail`` never move the carriage (the rail slot
  is pinned to the reported position) - the only rail motion of the job is
  ``home_rail()`` itself;
* :func:`frozen_state` - D1: the arm NOT part of the maintenance motion (09d;
  in 09c the unselected session arm) posed from its last monitor sample as an
  ``ArmState`` the gate twin is synced with once and never updated;
* :func:`plan_duration_s` - the executor-time estimate of a planned joint path
  at the executor caps (``PrePositionPlan.duration_s``);
* :class:`ExecutorCaps` / :func:`servo_executor_caps` / :func:`executor_caps_for` /
  :func:`apply_executor_caps` - the host-side ``PlanExecutor`` bounded by the
  connected driver's ``ServoLimits`` (per-joint velocity AND the lever-weighted
  Cartesian step the servo streamer enforces), so the commanded path is the
  validated straight segment and never a streamer-clipped bend of it;
* :class:`SessionStateProvider` - the overlay's frame/state source during a
  hardware session (monitor paused): session arms from ``workcell.states()``,
  frozen arms from their last sample;
* :func:`bringup_rows` - ``ArmBringupStatus`` -> ``ArmBringupTelemetry`` rows.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from apollo_mavis_v2_core import ArmState, CommandError, GripperCommand, GripperState, Pose, se3
from apollo_mavis_v2_core.interfaces import ArmInterface, WorkcellInterface
from apollo_mavis_v2_core.protocol import ArmBringupTelemetry

from ..config import ControlConfig

RAIL_TRAVEL_M = 0.65
ARM_LABELS = {"grip": "Manipulation Arm", "view": "Perception Arm"}  # user-facing names
MIN_RAIL_SPEED_MM_S = 1
# phase-09d: the measured posture must match the planned start this closely before the
# pre-positioning path executes (the hardware monitor's home_rail uses the same 0.02 rad)
START_POSTURE_TOL_RAD = 0.02


def arm_label(arm_id: str) -> str:
    """User-facing arm name (ids stay internal)."""
    return ARM_LABELS.get(arm_id, arm_id)


# -- speed scale (D2) ------------------------------------------------------------------------
def scale_control_config(cfg: ControlConfig, scale: float) -> ControlConfig:
    """Host-side ``speed_scale``: ``teleop.linear_mps / angular_rps / rail_mps``,
    ``target_rate.v_mps / w_radps``, ``dq_max_rad`` and
    ``jog.slew_rad_per_tick / rail_m_per_tick`` multiplied by ``scale``
    (``gripper_frac_ps``, the leash, the watchdog and the residual caps are
    not speeds and stay)."""
    s = float(scale)
    if not 0.0 < s <= 1.0:
        raise ValueError(f"speed_scale must be in (0, 1], got {scale}")
    teleop = cfg.teleop.model_copy(
        update={
            "linear_mps": cfg.teleop.linear_mps * s,
            "angular_rps": cfg.teleop.angular_rps * s,
            "rail_mps": cfg.teleop.rail_mps * s,
        }
    )
    target_rate = cfg.target_rate.model_copy(
        update={"v_mps": cfg.target_rate.v_mps * s, "w_radps": cfg.target_rate.w_radps * s}
    )
    jog = cfg.jog.model_copy(
        update={
            "slew_rad_per_tick": cfg.jog.slew_rad_per_tick * s,
            "rail_m_per_tick": cfg.jog.rail_m_per_tick * s,
        }
    )
    return cfg.model_copy(
        update={
            "teleop": teleop,
            "target_rate": target_rate,
            "jog": jog,
            "dq_max_rad": cfg.dq_max_rad * s,
        }
    )


def scale_driver_config(cfg: Any, scale: float) -> Any:
    """Driver-side ``speed_scale`` on a hardware ``XArmDriverConfig`` (duck-typed
    pydantic model): ``servo.max_joint_vel`` (per joint), ``servo.max_cart_step_m``
    and ``rail_speed_mm_s`` (integer, floor 1 mm/s) multiplied by ``scale``. The
    config's own values are the scale-1.0 hardware caps (0.3 rad/s, 2 mm/tick,
    50 mm/s)."""
    s = float(scale)
    if not 0.0 < s <= 1.0:
        raise ValueError(f"speed_scale must be in (0, 1], got {scale}")
    servo = cfg.servo.model_copy(
        update={
            "max_joint_vel": tuple(float(v) * s for v in cfg.servo.max_joint_vel),
            "max_cart_step_m": float(cfg.servo.max_cart_step_m) * s,
        }
    )
    return cfg.model_copy(
        update={
            "rail_speed_mm_s": max(MIN_RAIL_SPEED_MM_S, round(int(cfg.rail_speed_mm_s) * s)),
            "servo": servo,
        }
    )


# -- rail_flip at the workcell boundary ---------------------------------------------------------
def flip_rail(q: np.ndarray, travel_m: float = RAIL_TRAVEL_M) -> np.ndarray:
    """Copy of ``q`` with the rail slot (index 7, when present) mirrored."""
    out = np.array(q, dtype=np.float64)
    if out.shape[0] > 7:
        out[7] = travel_m - out[7]
    return out


class RailFlipArm(ArmInterface):
    """``ArmInterface`` view of a driver whose rail convention is mirrored:
    ``get_state().q[7] / rail_pos_m = 0.65 - track``, ``command_joints`` /
    ``command_rail`` map back. Everything else is forwarded (incl. the driver's
    ``drain_events`` / ``request_recovery`` / ``recovery_result`` / ``sn`` ...
    via ``__getattr__``)."""

    def __init__(self, inner: ArmInterface, travel_m: float = RAIL_TRAVEL_M) -> None:
        self.inner = inner
        self.travel_m = float(travel_m)

    def __getattr__(self, name: str) -> Any:  # forwarded driver surface
        return getattr(self.inner, name)

    def connect(self) -> None:
        self.inner.connect()

    def disconnect(self) -> None:
        self.inner.disconnect()

    def stop(self) -> None:
        self.inner.stop()

    def clear_errors(self) -> None:
        self.inner.clear_errors()

    def adapt(self, st: ArmState) -> ArmState:
        """The inner arm's state in the mirrored rail convention."""
        if not self.inner.has_rail or st.q.shape[0] < 8:
            return st
        q = flip_rail(st.q, self.travel_m)
        return ArmState(
            arm_id=st.arm_id,
            q=q,
            dq=np.array(st.dq, dtype=np.float64),
            ee_pose=st.ee_pose,
            gripper=st.gripper,
            rail_pos_m=float(q[7]),
            error_code=st.error_code,
            warn_code=st.warn_code,
            mode=st.mode,
            state=st.state,
            stale=st.stale,
            t_mono=st.t_mono,
            wallclock_ns=st.wallclock_ns,
        )

    def get_state(self) -> ArmState:
        return self.adapt(self.inner.get_state())

    def command_joints(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=np.float64)
        self.inner.command_joints(flip_rail(q, self.travel_m) if self.inner.has_rail else q)

    def command_gripper(self, cmd: GripperCommand) -> None:
        self.inner.command_gripper(cmd)

    def command_rail(self, pos_m: float) -> None:
        self.inner.command_rail(self.travel_m - float(pos_m))

    @property
    def dof(self) -> int:
        return self.inner.dof

    @property
    def has_rail(self) -> bool:
        return self.inner.has_rail

    @property
    def gripper_force_capable(self) -> bool:
        return self.inner.gripper_force_capable


class _WrappedWorkcell(WorkcellInterface):
    """``WorkcellInterface`` over a hardware workcell with every railed arm wrapped
    by :meth:`_wrap`; ``cameras`` / events / recovery forwarded. ``states()``
    reads the INNER workcell's batch (``HardwareWorkcell.states`` - the drivers'
    cached reports; a fake may stamp them) and runs each state through its
    arm's ``adapt``, so the wrapped view never bypasses the inner batch read."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.arms: dict[str, ArmInterface] = {
            arm_id: self._wrap(arm_id, arm) if getattr(arm, "has_rail", False) else arm
            for arm_id, arm in inner.arms.items()
        }

    def _wrap(self, arm_id: str, arm: ArmInterface) -> ArmInterface:  # pragma: no cover
        raise NotImplementedError

    def __getattr__(self, name: str) -> Any:  # bring_up, request_recovery, recovery_result ...
        return getattr(self.inner, name)

    @property
    def kind(self):
        return self.inner.kind

    @property
    def cameras(self) -> dict:
        return self.inner.cameras

    def start(self) -> None:
        self.inner.start()

    def stop(self) -> None:
        self.inner.stop()

    def states(self) -> dict[str, ArmState]:
        out: dict[str, ArmState] = {}
        for arm_id, st in self.inner.states().items():
            adapt = getattr(self.arms.get(arm_id), "adapt", None)
            out[arm_id] = adapt(st) if adapt is not None else st
        return out

    def drain_events(self) -> list[Any]:
        drain = getattr(self.inner, "drain_events", None)
        return list(drain()) if drain is not None else []


class RailFlipWorkcell(_WrappedWorkcell):
    """Every railed arm wrapped in :class:`RailFlipArm` (``hardware_session.rail_flip``)."""

    def __init__(self, inner: Any, travel_m: float = RAIL_TRAVEL_M) -> None:
        self.travel_m = float(travel_m)
        super().__init__(inner)

    def _wrap(self, arm_id: str, arm: ArmInterface) -> ArmInterface:
        return RailFlipArm(arm, self.travel_m)


# -- phase-09d: the rail-homing job's arm adapter (track unhomed, position unknown) --------------
class RailHoldArm(ArmInterface):
    """``ArmInterface`` view of a railed driver whose carriage position is UNKNOWN
    (connected with ``rail_homing: allow_unhomed`` for the rail-homing job).

    The driver publishes ``q[7] == rail_pos_m == 0.0`` as a placeholder until
    ``home_rail()`` succeeds (``rail_position_known`` False); the twin must not
    believe it, so ``get_state`` substitutes ``fallback_m`` (the configured
    ``rail_fallback_m`` in the convention of the wrapped arm, i.e. AFTER a
    :class:`RailFlipArm` when ``rail_flip`` is on). Once the position is known
    the measured value passes through. Commands never move the carriage: the rail
    slot of ``command_joints`` is pinned to the wrapped arm's CURRENT rail slot
    and ``command_rail`` raises - the position-agnostic path check is the only
    safety basis of the motion, and the only rail motion of the job is the
    homing itself (the driver drops rail targets while unhomed anyway; after the
    homing this adapter keeps a stale 8-dof hold from moving the freshly homed
    carriage). Everything else is forwarded (``home_rail`` / ``rail_phase`` /
    ``rail_position_known`` / ``drain_events`` ... via ``__getattr__``).
    """

    def __init__(self, inner: ArmInterface, fallback_m: float, travel_m: float = RAIL_TRAVEL_M):
        self.inner = inner
        self.fallback_m = min(float(travel_m), max(0.0, float(fallback_m)))
        self.travel_m = float(travel_m)
        self.rail_commands_dropped = 0  # diagnostics / tests

    def __getattr__(self, name: str) -> Any:  # forwarded driver surface
        return getattr(self.inner, name)

    @property
    def position_known(self) -> bool:
        return bool(getattr(self.inner, "rail_position_known", True))

    def connect(self) -> None:
        self.inner.connect()

    def disconnect(self) -> None:
        self.inner.disconnect()

    def stop(self) -> None:
        self.inner.stop()

    def clear_errors(self) -> None:
        self.inner.clear_errors()

    def adapt(self, st: ArmState) -> ArmState:
        """The inner arm's state with the fallback in the rail slot while unknown."""
        if not self.inner.has_rail or self.position_known or st.q.shape[0] < 8:
            return st
        q = np.array(st.q, dtype=np.float64)
        q[7] = self.fallback_m
        return ArmState(
            arm_id=st.arm_id,
            q=q,
            dq=np.array(st.dq, dtype=np.float64),
            ee_pose=st.ee_pose,
            gripper=st.gripper,
            rail_pos_m=self.fallback_m,
            error_code=st.error_code,
            warn_code=st.warn_code,
            mode=st.mode,
            state=st.state,
            stale=st.stale,
            t_mono=st.t_mono,
            wallclock_ns=st.wallclock_ns,
        )

    def get_state(self) -> ArmState:
        return self.adapt(self.inner.get_state())

    def command_joints(self, q: np.ndarray) -> None:
        q = np.array(q, dtype=np.float64)
        if self.inner.has_rail and q.shape[0] > 7:
            current = self.inner.get_state()
            if current.q.shape[0] > 7 and abs(float(q[7]) - float(current.q[7])) > 1e-9:
                self.rail_commands_dropped += 1
            q[7] = float(current.q[7])  # never a rail move: pin to the reported slot
        self.inner.command_joints(q)

    def command_gripper(self, cmd: GripperCommand) -> None:
        self.inner.command_gripper(cmd)

    def command_rail(self, pos_m: float) -> None:
        raise CommandError(
            f"{getattr(self.inner, 'arm_id', 'arm')}: rail locked during the rail-homing "
            "maintenance motion (the carriage position is unknown until home_rail)"
        )

    @property
    def dof(self) -> int:
        return self.inner.dof

    @property
    def has_rail(self) -> bool:
        return self.inner.has_rail

    @property
    def gripper_force_capable(self) -> bool:
        return self.inner.gripper_force_capable


class RailHoldWorkcell(_WrappedWorkcell):
    """Every railed arm wrapped in :class:`RailHoldArm` with its configured
    ``rail_fallback_m`` (phase-09d rail-homing job)."""

    def __init__(self, inner: Any, rail_fallback_m: Mapping[str, float]) -> None:
        self.rail_fallback_m = dict(rail_fallback_m)
        super().__init__(inner)

    def _wrap(self, arm_id: str, arm: ArmInterface) -> ArmInterface:
        return RailHoldArm(arm, self.rail_fallback_m.get(arm_id, 0.0))


def plan_duration_s(
    waypoints: Iterable[Iterable[float]],
    slew_rad_per_tick: float,
    rate_hz: float,
    *,
    cart_step_m: float | None = None,
    lever_arm_m: Sequence[float] | None = None,
) -> float:
    """Executor-time estimate of a joint path: the ``PlanExecutor`` slews every
    joint by at most ``slew_rad_per_tick`` per tick and - with the hardware
    executor caps (:class:`ExecutorCaps`) - bounds ``sum|dq_j| * lever_j`` per
    tick by ``cart_step_m`` like the driver's servo streamer, so a segment takes
    ``max(max|dq| / slew, sum|dq| lever / cart_step)`` ticks (rail slot
    excluded - the job never moves it)."""
    qs = [np.asarray(list(w), dtype=np.float64)[:7] for w in waypoints]
    if len(qs) < 2 or slew_rad_per_tick <= 0.0 or rate_hz <= 0.0:
        return 0.0
    lever = (
        np.asarray(list(lever_arm_m), dtype=np.float64)[:7]
        if cart_step_m is not None and lever_arm_m is not None and cart_step_m > 0.0
        else None
    )
    ticks = 0.0
    for a, b in zip(qs[:-1], qs[1:], strict=True):
        dq = np.abs(b - a)
        seg = float(np.max(dq)) / float(slew_rad_per_tick)
        if lever is not None:
            seg = max(seg, float(np.sum(dq * lever)) / float(cart_step_m))
        ticks += seg
    return ticks / float(rate_hz)


# -- hardware executor caps: the host-side plan slew bounded by the driver's servo stream ------
@dataclass(frozen=True)
class ExecutorCaps:
    """Per-tick caps of the ``PlanExecutor`` on a HARDWARE loop, derived from the
    connected driver's ``ServoLimits`` (already speed-scaled inside the driver
    config) so the COMMANDED path is one the servo streamer can follow tick for
    tick: the streamer clips every joint at ``max_joint_vel * dt`` and scales the
    whole step so ``sum|dq_j| * lever_j <= max_cart_step_m`` - a host step beyond
    either cap makes the streamer bend the straight joint-space segment the
    planner (and the rail-homing job's position-agnostic ``check_path``)
    validated, while the gate only ever sees the commanded posture. With the caps
    applied the streamer's clips are inactive (its per-joint acceleration ramp
    lags by at most ``max_joint_vel^2 / (2 max_joint_acc)`` rad, ~1e-4 rad at the
    defaults - far below the 0.05 rad path densification)."""

    slew_rad_per_tick: float  # max per-joint step per loop tick (<= the host slew)
    cart_step_m: float | None  # sum|dq_j| * lever_j bound per loop tick (None = unbounded)
    lever_arm_m: tuple[float, ...] | None  # conservative lever per joint (7)
    source: str = "host"  # "servo" when derived from a driver's ServoLimits
    # The streamer's own per-joint step per loop tick (max_joint_vel / rate_hz), NOT
    # bounded by the host jog slew like ``slew_rad_per_tick`` is - the teleop caps
    # (``apply_teleop_caps``) must follow the servo bound alone, never the jog's.
    # None = no driver published one (host-only caps: teleop is left untouched).
    joint_step_rad: float | None = None
    # The carriage's per-loop-tick step the TRACK can actually follow: the driver
    # config's ``rail_speed_mm_s`` (already speed-scaled) / the loop rate. Without it
    # (2026-09-08 review) the executor slewed the rail slot at ``jog.rail_m_per_tick``
    # x scale = 0.2 m/s at 100 % while the track positions at 50 mm/s x scale toward
    # latest-wins targets, so an arm was declared arrived up to ~10 s before its
    # carriage physically got there - and the next arm of a sequence started moving
    # against a carriage the gate believed parked (it checks the COMMANDED q). None =
    # no driver published a rail speed (sim, fakes): the host value stands.
    rail_m_per_tick: float | None = None


def servo_executor_caps(
    servo: Any,
    rate_hz: float,
    host_slew_rad_per_tick: float,
    *,
    scale: float = 1.0,
) -> ExecutorCaps:
    """:class:`ExecutorCaps` from a duck-typed hardware ``ServoLimits`` (``rate_hz``,
    ``max_joint_vel`` (7, rad/s), ``lever_arm_m`` (7), ``max_cart_step_m`` per
    STREAMER tick), scaled by ``scale`` (1.0 for a driver config that is already
    speed-scaled) and converted to the loop's tick (``rate_hz``)."""
    s = float(scale)
    loop_hz = float(rate_hz)
    servo_hz = float(getattr(servo, "rate_hz", loop_hz) or loop_hz)
    ticks_per_loop_tick = servo_hz / loop_hz  # streamer ticks per loop tick
    vel = min(float(v) for v in servo.max_joint_vel) * s / loop_hz  # rad per loop tick
    slew = min(float(host_slew_rad_per_tick), vel)
    cart = float(servo.max_cart_step_m) * s * ticks_per_loop_tick
    return ExecutorCaps(
        slew_rad_per_tick=slew,
        cart_step_m=cart,
        lever_arm_m=tuple(float(v) for v in servo.lever_arm_m)[:7],
        source="servo",
        joint_step_rad=vel,
    )


def default_servo_limits() -> Any | None:
    """The hardware package's ``ServoLimits()`` defaults (``None`` without the
    ``[hardware]`` extra). The workcell's ``ArmConfig -> XArmDriverConfig``
    mapping forwards no speed field, so these ARE every real driver's caps
    before the session / job speed scale."""
    try:
        from apollo_mavis_v2_hardware.config import ServoLimits  # [hardware] extra
    except Exception:  # noqa: BLE001 - optional extra absent or broken
        return None
    return ServoLimits()


def executor_caps_for(
    arms: Iterable[Any], cfg: ControlConfig, *, fallback_servo: Any = None, scale: float = 1.0
) -> ExecutorCaps:
    """Caps for a hardware loop over ``arms`` (connected driver objects): the
    tightest of every driver exposing ``cfg.servo`` (``XArmDriver.cfg`` - already
    speed-scaled); a driver without one (fakes) contributes nothing; when none
    does, ``fallback_servo`` (scaled by ``scale``) or else the host slew alone."""
    host = float(cfg.jog.slew_rad_per_tick)
    found: list[ExecutorCaps] = []
    rail_caps: list[float] = []  # m per loop tick the connected tracks can follow
    for arm in arms:
        driver_cfg = getattr(arm, "cfg", None)
        servo = getattr(driver_cfg, "servo", None)
        if servo is not None and hasattr(servo, "max_joint_vel"):
            found.append(servo_executor_caps(servo, cfg.rate_hz, host))
        rail_mm_s = getattr(driver_cfg, "rail_speed_mm_s", None)
        if rail_mm_s is not None and float(rail_mm_s) > 0.0:
            rail_caps.append(float(rail_mm_s) / 1000.0 / float(cfg.rate_hz))
    rail = min(rail_caps) if rail_caps else None
    if not found and fallback_servo is not None:
        found.append(servo_executor_caps(fallback_servo, cfg.rate_hz, host, scale=scale))
    if not found:
        return ExecutorCaps(
            slew_rad_per_tick=host, cart_step_m=None, lever_arm_m=None, rail_m_per_tick=rail
        )
    tight = min(found, key=lambda c: c.slew_rad_per_tick)
    cart = min(c.cart_step_m for c in found if c.cart_step_m is not None)
    lever = max((c.lever_arm_m for c in found if c.lever_arm_m is not None), key=lambda lv: sum(lv))
    joint = min(c.joint_step_rad for c in found if c.joint_step_rad is not None)
    return ExecutorCaps(
        tight.slew_rad_per_tick,
        cart,
        lever,
        source="servo",
        joint_step_rad=joint,
        rail_m_per_tick=rail,
    )


def teleop_rate_caps(cfg: ControlConfig, caps: ExecutorCaps) -> tuple[float | None, float | None]:
    """``(tcp_mps, joint_radps)`` the servo streamer can actually execute, from
    per-loop-tick :class:`ExecutorCaps` and the loop rate. Either is ``None``
    when the driver published no such bound (host-only caps: both ``None``).
    The joint rate comes from ``joint_step_rad`` (the streamer's own bound),
    not from ``slew_rad_per_tick``, which the PlanExecutor also bounds by the
    host JOG slew - a jog cap must not leak into teleop."""
    hz = float(cfg.rate_hz)
    tcp = None if caps.cart_step_m is None else float(caps.cart_step_m) * hz
    joint = None if caps.joint_step_rad is None else float(caps.joint_step_rad) * hz
    return tcp, joint


def apply_teleop_caps(cfg: ControlConfig, caps: ExecutorCaps) -> ControlConfig:
    """``cfg`` with the TELEOP command chain bounded by what the connected
    driver's servo streamer can execute (2026-09-07).

    WHY. The clutched tracker target is rate-limited to ``target_rate`` (1.0 m/s
    / 2.0 rad/s in the lab config) and then leash-clamped to ``leash.pos_m``
    (0.025 m) around the MEASURED TCP, while the streamer executes at most
    ``max_cart_step_m`` per streamer tick - 0.002 m, i.e. 0.2 m/s at
    ``speed_scale`` 1.0 and 0.02 m/s at the 0.1 default. Hand motion faster than
    that ran the target into the leash and
    :meth:`~apollo_mavis_v2_runtime.control.tracker_teleop.TrackerTeleop.slip`
    folded the truncation into the engagement anchor, so the excess hand travel
    was silently DISCARDED; the part that did get through kept arriving for up
    to one leash (0.125 s at scale 1.0, 1.25 s at 0.1) after the hand stopped.
    Those are exactly the two symptoms the operator reported after the
    2026-09-06 session: moving the controller down barely moved the end
    effector, and the arm kept going after the trigger was released.

    Capping the host chain at the streamer's own rate makes the mapping
    FAITHFUL instead of clipped: slower than the hand, but never truncated, so
    the commanded direction is the hand's direction and releasing the clutch
    stops the arm within one streamer tick. The arm's top speed is unchanged -
    it always was the streamer's - so no safety bound is relaxed here. Raising
    the FEEL means raising ``ServoLimits`` / ``speed_scale``, deliberately.

    Capped: ``target_rate.v_mps`` / ``.w_radps`` (tracker chain),
    ``teleop.linear_mps`` / ``.angular_rps`` (keyboard chain) and
    ``dq_max_rad`` (the loop's per-tick joint clamp). ``teleop.rail_mps`` is a
    separate track axis with its own controller-side speed and is left alone.
    """
    tcp_mps, joint_radps = teleop_rate_caps(cfg, caps)
    if tcp_mps is None and joint_radps is None:
        return cfg  # host-only caps (no driver bound published): nothing to cap against

    def cap(value: float, bound: float | None) -> float:
        return float(value) if bound is None else min(float(value), float(bound))

    teleop = cfg.teleop.model_copy(
        update={
            "linear_mps": cap(cfg.teleop.linear_mps, tcp_mps),
            "angular_rps": cap(cfg.teleop.angular_rps, joint_radps),
        }
    )
    target_rate = cfg.target_rate.model_copy(
        update={
            "v_mps": cap(cfg.target_rate.v_mps, tcp_mps),
            "w_radps": cap(cfg.target_rate.w_radps, joint_radps),
        }
    )
    return cfg.model_copy(
        update={
            "teleop": teleop,
            "target_rate": target_rate,
            "dq_max_rad": cap(cfg.dq_max_rad, caps.joint_step_rad),
        }
    )


def apply_executor_caps(cfg: ControlConfig, caps: ExecutorCaps) -> ControlConfig:
    """``cfg`` with the ``PlanExecutor`` bounded by ``caps`` (``jog.slew_rad_per_tick``
    lowered to the cap; ``jog.plan_cart_step_m`` / ``plan_lever_arm_m`` set;
    ``jog.rail_m_per_tick`` lowered to the track's own positioning speed when a driver
    published one - the commanded carriage then never runs ahead of the track)."""
    rail = float(cfg.jog.rail_m_per_tick)
    if caps.rail_m_per_tick is not None:
        rail = min(rail, float(caps.rail_m_per_tick))
    jog = cfg.jog.model_copy(
        update={
            "slew_rad_per_tick": min(float(cfg.jog.slew_rad_per_tick), caps.slew_rad_per_tick),
            "rail_m_per_tick": rail,
            "plan_cart_step_m": caps.cart_step_m,
            "plan_lever_arm_m": caps.lever_arm_m,
        }
    )
    return cfg.model_copy(update={"jog": jog})


# -- D1: the arm not part of the motion frozen at its last monitor sample -----------------------
def frozen_state(
    arm_id: str,
    sample: Any,
    *,
    has_rail: bool,
    rail_fallback_m: Mapping[str, float],
    rail_flip: bool = False,
    travel_m: float = RAIL_TRAVEL_M,
    gripper: bool | None = None,
) -> tuple[ArmState, str | None]:
    """``ArmState`` posing an arm that is NOT commanded (09c: the unselected
    session arm; 09d: the other arm during a rail-homing job) from its last
    monitor sample (q7 + rail position, ``rail_fallback_m`` when the rail is
    unknown; ``rail_flip`` applied like the overlay) -> ``(state, assumption | None)``.

    ``ee_pose`` (2026-09-11) is the twin's ``link_tcp`` built from the sample's FLANGE
    ``tcp_pose`` (``[x, y, z m, roll, pitch, yaw rad]``, extrinsic-XYZ RPY) through
    ``se3.rpy_to_quat`` + ``se3.flange_to_tcp``; ``gripper`` says whether the arm carries
    one (``ArmConfig.gripper != "none"``), ``None`` = infer it from the sample
    (``gripper_open_frac`` is ``None`` exactly for gripper ``"none"``). Identity when
    the sample carries no pose (the pre-2026-09-11 behaviour for every sample)."""
    q7 = np.asarray(list(sample.q)[:7], dtype=np.float64)
    note: str | None = None
    if has_rail:
        pos = getattr(sample, "rail_pos_m", None)
        if pos is None:
            fb = float(rail_fallback_m.get(arm_id, 0.0))
            why = "rail not enabled" if getattr(sample, "rail_homed", False) else "rail not homed"
            note = f"{why} - twin assumes {fb:.2f} m"
            pos = fb
        elif rail_flip:
            pos = travel_m - float(pos)
        q = np.append(q7, min(travel_m, max(0.0, float(pos))))
    else:
        q = q7
    frac = getattr(sample, "gripper_open_frac", None)
    tcp = tuple(getattr(sample, "tcp_pose", ()) or ())
    if len(tcp) >= 6 and all(np.isfinite(v) for v in tcp[:6]):
        flange = Pose(np.asarray(tcp[:3], dtype=np.float64), se3.rpy_to_quat(tcp[3:6]))
        has_gripper = (frac is not None) if gripper is None else bool(gripper)
        ee_pose = se3.flange_to_tcp(flange, gripper=has_gripper)
    else:
        ee_pose = Pose.identity()
    now = time.monotonic()
    state = ArmState(
        arm_id=arm_id,
        q=q,
        dq=np.zeros_like(q),
        ee_pose=ee_pose,
        gripper=GripperState(open_frac=1.0 if frac is None else min(1.0, max(0.0, float(frac)))),
        rail_pos_m=float(q[7]) if has_rail else None,
        error_code=int(getattr(sample, "error_code", 0) or 0),
        warn_code=int(getattr(sample, "warn_code", 0) or 0),
        mode=int(getattr(sample, "mode", None) or 0),
        state=int(getattr(sample, "state", None) or 0),
        stale=False,
        t_mono=now,
        wallclock_ns=time.time_ns(),
    )
    return state, note


# -- overlay state source during a hardware session ---------------------------------------------
@dataclass(frozen=True)
class StateSample:
    """Monitor-sample-shaped view of a driver ``ArmState`` (the overlay poses the
    twin from these fields; the same names as ``ArmMonitorSample``)."""

    arm_id: str
    seq: int
    t_mono: float
    q: tuple[float, ...]
    tcp_pose: tuple[float, ...] = ()
    error_code: int = 0
    warn_code: int = 0
    state: int | None = None
    mode: int | None = None
    rail_present: bool | None = None
    rail_homed: bool | None = None
    rail_enabled: bool | None = None
    rail_pos_m: float | None = None
    rail_raw_mm: float | None = None
    rail_error: int | None = None
    gripper_open_frac: float | None = None
    gripper_raw: float | None = None
    collision_sensitivity: int | None = None
    tcp_load_kg: float | None = None
    tcp_load_cog_mm: tuple[float, ...] = ()


def sample_from_state(
    state: ArmState, seq: int, *, has_gripper: bool, rail_known: bool = True
) -> StateSample:
    """Driver ``ArmState`` (TRACK rail convention, i.e. the inner workcell) ->
    :class:`StateSample`. A railed arm reports its rail as homed + enabled with
    the measured position while ``rail_known`` (a session driver only connects
    to a homed track); ``rail_known=False`` (phase-09d: the rail-homing job's
    driver connected with ``allow_unhomed``, ``XArmDriver.rail_position_known``
    False - its ``q[7]`` is the 0.0 placeholder) reports the rail present but
    NOT homed / enabled with ``rail_pos_m None``, so the overlay applies its own
    ``rail_fallback_m`` + "rail not homed - twin assumes X m" caption exactly as
    for the monitor's unhomed sample (``rail_raw_mm`` keeps the raw register)."""
    q = np.asarray(state.q, dtype=np.float64)
    railed = q.shape[0] > 7
    known = bool(rail_known)
    return StateSample(
        arm_id=state.arm_id,
        seq=int(seq),
        t_mono=float(state.t_mono),
        q=tuple(float(v) for v in q[:7]),
        error_code=int(state.error_code),
        warn_code=int(state.warn_code),
        state=int(state.state),
        mode=int(state.mode),
        rail_present=railed,
        rail_homed=known if railed else None,
        rail_enabled=known if railed else None,
        rail_pos_m=float(q[7]) if railed and known else None,
        rail_raw_mm=float(q[7]) * 1000.0 if railed else None,
        rail_error=0 if railed else None,
        gripper_open_frac=float(state.gripper.open_frac) if has_gripper else None,
    )


class SessionStateProvider:
    """Overlay state source while a hardware session owns the boxes: session arms
    from the INNER workcell's ``states()`` (track rail convention - the overlay
    applies ``rail_flip`` itself), unselected arms from their frozen sample.
    ``status_of`` mirrors the monitor's vocabulary: ``running`` / ``stale`` for
    the session arms, ``stale`` + "frozen" for the others."""

    def __init__(
        self,
        workcell: Any,
        session_arms: Iterable[str],
        frozen: Mapping[str, Any],
        gripper_arms: Iterable[str],
    ) -> None:
        self.workcell = workcell
        self.session_arms = tuple(session_arms)
        self.frozen = dict(frozen)
        self.gripper_arms = frozenset(gripper_arms)
        self._seq = 0
        self._stale: dict[str, bool] = {}

    def samples(self) -> dict[str, Any]:
        out: dict[str, Any] = dict(self.frozen)
        try:
            states = self.workcell.states()
        except Exception:  # noqa: BLE001 - a driver hiccup must not kill the overlay
            states = {}
        self._seq += 1
        arms = getattr(self.workcell, "arms", {}) or {}
        for arm_id in self.session_arms:
            st = states.get(arm_id)
            if st is None:
                continue
            self._stale[arm_id] = bool(st.stale)
            # phase-09d: a driver connected with its track unhomed publishes a 0.0
            # placeholder in the rail slot - the overlay must not believe it
            known = bool(getattr(arms.get(arm_id), "rail_position_known", True))
            out[arm_id] = sample_from_state(
                st, self._seq, has_gripper=arm_id in self.gripper_arms, rail_known=known
            )
        return out

    def status_of(self, arm_id: str) -> tuple[str, str]:
        if arm_id in self.session_arms:
            if self._stale.get(arm_id, False):
                return "stale", "hardware session: report stream stale"
            return "running", ""
        if arm_id in self.frozen:
            return "stale", f"{arm_label(arm_id)} frozen at last sample (hardware session)"
        return "paused", "not part of the hardware session"


# -- bring-up progress rows -----------------------------------------------------------------------
def bringup_rows(status: Any) -> list[ArmBringupTelemetry]:
    """``ArmBringupStatus``-like (``network`` / ``connected`` / ``rail`` /
    ``gripper`` / ``warnings`` / ``error`` / ``fw_version`` / ``sn``) -> one
    ``ArmBringupTelemetry`` row per stage, in stage order."""
    arm_id = str(status.arm_id)
    error = getattr(status, "error", None) or ""
    rows: list[ArmBringupTelemetry] = []

    def row(step: str, st: str, detail: str = "") -> None:
        rows.append(ArmBringupTelemetry(arm_id=arm_id, step=step, status=st, detail=detail))

    network = str(getattr(status, "network", "pending"))
    if network == "ok":
        row("network", "ok", "control box reachable")
    elif network == "failed":
        row("network", "error", error or "network failed")
    elif network == "booting":
        row("network", "pending", "control box booting (polling TCP 502)")
    else:
        row("network", "pending", "probing")

    connected = bool(getattr(status, "connected", False))
    fw = getattr(status, "fw_version", None)
    sn = getattr(status, "sn", None)
    ident = ", ".join(p for p in (fw and f"fw {fw}", sn and f"sn {sn}") if p)
    rail = str(getattr(status, "rail", "unknown"))
    gripper = str(getattr(status, "gripper", "unknown"))
    connect_failed = bool(error) and not connected and network != "failed"
    if connected or rail not in ("unknown",) or gripper != "unknown":
        # the driver's connect() returned (rail / gripper are known) or failed
        if connect_failed and rail != "unhomed" and gripper != "error" and "[report]" not in error:
            row("connect", "error", error)
        else:
            row("connect", "ok", f"connected ({ident})" if ident else "connected")
    elif connect_failed:
        row("connect", "error", error)
    else:
        row("connect", "pending", "connecting")

    if rail == "none":
        row("rail", "ok", "no linear track")
    elif rail == "ready":
        row("rail", "ok", "linear track homed and enabled")
    elif rail == "unhomed":
        row("rail", "error", error or "rail not homed")
    elif rail == "error":
        row("rail", "error", error or "linear track error")
    elif rail == "detected":
        row("rail", "pending", "linear track detected")
    else:
        row("rail", "pending", "")

    if gripper == "error":
        row("gripper", "error", error or "gripper error")
    elif gripper == "unknown":
        row("gripper", "pending", "")
    else:
        row("gripper", "ok", "no gripper" if gripper == "none" else f"gripper {gripper}")

    if connected:
        row("report", "ok", "30003 report stream fresh")
    elif "[report]" in error or "report stream" in error:
        row("report", "error", error)
    else:
        row("report", "pending", "")

    warnings = [str(w) for w in (getattr(status, "warnings", None) or ())]
    if warnings:
        row("warnings", "warning", "; ".join(warnings))
    return rows


__all__ = [
    "ARM_LABELS",
    "RAIL_TRAVEL_M",
    "START_POSTURE_TOL_RAD",
    "ExecutorCaps",
    "RailFlipArm",
    "RailFlipWorkcell",
    "RailHoldArm",
    "RailHoldWorkcell",
    "SessionStateProvider",
    "StateSample",
    "apply_executor_caps",
    "apply_teleop_caps",
    "arm_label",
    "bringup_rows",
    "default_servo_limits",
    "executor_caps_for",
    "flip_rail",
    "frozen_state",
    "plan_duration_s",
    "sample_from_state",
    "scale_control_config",
    "scale_driver_config",
    "servo_executor_caps",
    "teleop_rate_caps",
]
