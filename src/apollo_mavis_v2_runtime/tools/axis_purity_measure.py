"""``python -m apollo_mavis_v2_runtime.tools.axis_purity_measure [--streamer] [--cap VARIANT]
[--no-pullback] [--tracker] [--scales 1.0 0.5] [--postures ...] [--arms ...] [--keys ...]``
- axis purity of the teleop chain on the ``mavis_v2`` sim in LOCKSTEP (04-runtime §6
"Per-tick joint step cap"; operator report 2026-09-09: "when I press forward/back with the
keyboard the arm also drifts up/down").

Runs the REAL ``ControlLoop`` + ``MinkIKSolver`` + servo-faithful ``SimWorkcell`` (as a sim
session builds it) with no wall clock: one loop tick, dispatch, one physics tick. From each
posture it holds every translate key for ``--hold`` s and reports the Manipulation /
Perception Arm TCP travel along the key's axis, off it, and the orientation drift (world
frame), plus the loop's cap ticks and IK slips. Needs the ``[sim]`` and ``[hardware]``
extras (the caps come from the driver's ``ServoLimits``); not collected by pytest.

Controls (the isolation experiments behind the §6 root cause; all optional):

- ``--streamer``: pass every host command through an EMULATION of the hardware
  ``_ServoStreamer`` (per-joint velocity clip, per-joint acceleration clip, lever-weighted
  Cartesian scale, joint limits; one streamer tick per loop tick, ideal timing) before the
  sim executes it. The sim itself has NO streamer, so without this flag the run measures
  the host-side caps only - which is exactly how the first uniform-scaling fix looked
  right in the sim and still bent on the real arm (the streamer's lever estimate is 5-10x
  conservative for a keyboard step; its per-joint clip then bent the direction one layer
  down). The columns ``vel/acc/lever`` count the streamer ticks each clip acted on;
  ``maxlag`` is the largest command-vs-streamer gap (rad).
- ``--cap``: ``uniform`` (the shipped rule: whole step / max(joint ratio, lever-weighted
  Cartesian ratio)), ``joint-only`` (the first 2026-09-09 fix: dq_max alone),
  ``per-joint-clip`` (the pre-2026-09-09 ``np.clip``). ``--no-pullback`` disables the
  keyboard target pull-back (``_hold_key_target``). The shipped behaviour is the default;
  ``--cap per-joint-clip --no-pullback`` reproduces the operator's report.
- ``--config`` overrides: ``dq=0.04`` (cap inactive), ``leash=1`` (metres), ``nores``
  (residual freeze off) - the suspects (a)-(c) of the §6 history.
- ``--tracker``: instead of keys, a scripted HAND (tracker sample stream, filter off,
  yaw 0, scale 1) moving along each world axis at ``--hand-mps`` while the clutch is
  held; reports the TCP's on-axis rate vs the hand's, the off-axis drift and the
  provider's leash slips - the trade-off check for the tracker path, which the uniform
  cap slows whenever a joint saturates but never bends.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import types
from collections.abc import Callable, Iterable, Sequence

import numpy as np
from apollo_mavis_v2_core import ArmConfig, HeldState, Pose, Twist, WorkcellConfig, se3

from ..bus import RuntimeBus
from ..config import ControlConfig, LeashConfig
from ..control.fk import SceneKinematics
from ..control.loop import RAIL_TRAVEL_M, ControlLoop
from ..control.teleop import held_to_twist, twist_to_control_frame
from ..control.tracker_teleop import TRACKER_CLUTCH_CODE, TrackerTeleop
from ..devices.tracker import TrackerSample, TrackerSettings
from ..safety.gate import NullGate
from ..safety.supervisor import SafetySupervisor
from ..safety.watchdog import InputWatchdog
from ..session.hardware import (
    apply_executor_caps,
    apply_teleop_caps,
    executor_caps_for,
    scale_control_config,
)

DT = 0.01
ARMS = ("grip", "view")
KEYS = ("KeyW", "KeyS", "KeyA", "KeyD", "KeyE", "KeyQ")
AXES = {
    "+x": (1, 0, 0),
    "-x": (-1, 0, 0),
    "+y": (0, 1, 0),
    "-y": (0, -1, 0),
    "+z": (0, 0, 1),
    "-z": (0, 0, -1),
}
IDENT = np.array([1.0, 0.0, 0.0, 0.0])
# The operator's initial-condition posture (profiles/seed_initial), degrees.
INITIAL_DEG = {
    "grip": [-180.0, -12.0, -20.0, 30.0, -5.0, 35.0, -8.9],
    "view": [0.0, 0.8, 0.0, 28.9, 0.0, 28.2, 0.0],
}
# Postures beyond the initial condition, reached by holding keys (world frame) from it
# with the UNCONSTRAINED sim config, so every configuration measures the same q.
SCRIPTS: dict[str, list[tuple[str, float]]] = {
    "P0 initial": [],
    "P1 W1.5s": [("KeyW", 1.5)],
    "P2 E1s A1s": [("KeyE", 1.0), ("KeyA", 1.0)],
    "P3 S1s Q0.6s D0.8s": [("KeyS", 1.0), ("KeyQ", 0.6), ("KeyD", 0.8)],
}


# -- configs ------------------------------------------------------------------------------------
def hw_equivalent(scale: float, over: dict | None = None) -> ControlConfig:
    """The ``ControlConfig`` a HARDWARE bring-up hands the loop at ``speed_scale``: host
    scaling, then the executor + teleop caps of the driver's ``ServoLimits``."""
    from apollo_mavis_v2_hardware.config import ServoLimits

    cfg = scale_control_config(ControlConfig(translate_frame="world"), scale)
    caps = executor_caps_for([], cfg, fallback_servo=ServoLimits(), scale=scale)
    cfg = apply_teleop_caps(apply_executor_caps(cfg, caps), caps)
    return cfg.model_copy(update=over) if over else cfg


def config_overrides(names: Iterable[str]) -> dict:
    over: dict = {}
    for name in names:
        if name.startswith("dq="):
            over["dq_max_rad"] = float(name[3:])
        elif name.startswith("leash="):
            over["leash"] = LeashConfig(pos_m=float(name[6:]), rot_rad=10.0)
        elif name == "nores":
            over["residual_max_pos_m"] = 10.0
            over["residual_max_rot_rad"] = 10.0
        else:
            raise SystemExit(f"unknown --config {name!r} (dq=<rad>, leash=<m>, nores)")
    return over


# -- cap variants (bound on the loop INSTANCE; the shipped code is untouched) -------------------
def _cap_joint_only(self: ControlLoop, q: np.ndarray, q_last: np.ndarray):
    """The first 2026-09-09 fix: uniform scaling against ``dq_max`` alone."""
    dq_max = self.cfg.dq_max_rad
    q = np.array(q, dtype=np.float64)
    dq = q[:7] - q_last[:7]
    ratio = float(np.max(np.abs(dq))) / dq_max
    capped = ratio > 1.0
    if capped:
        q[:7] = q_last[:7] + dq / ratio
    if q.shape[0] > 7:
        q[7] = min(max(q[7], q_last[7] - dq_max), q_last[7] + dq_max)
        q[7] = min(max(q[7], 0.0), RAIL_TRAVEL_M)
    return q, capped


def _cap_per_joint_clip(self: ControlLoop, q: np.ndarray, q_last: np.ndarray):
    """The pre-2026-09-09 rule: every joint clipped independently."""
    dq_max = self.cfg.dq_max_rad
    q = np.array(q, dtype=np.float64)
    clipped = np.clip(q[:7], q_last[:7] - dq_max, q_last[:7] + dq_max)
    capped = bool(np.any(clipped != q[:7]))
    q[:7] = clipped
    if q.shape[0] > 7:
        q[7] = min(max(q[7], q_last[7] - dq_max), q_last[7] + dq_max)
        q[7] = min(max(q[7], 0.0), RAIL_TRAVEL_M)
    return q, capped


CAP_VARIANTS: dict[str, Callable | None] = {
    "uniform": None,  # the shipped ControlLoop._cap_joint_step
    "joint-only": _cap_joint_only,
    "per-joint-clip": _cap_per_joint_clip,
}


# -- the hardware servo streamer, emulated ------------------------------------------------------
class StreamerEmu:
    """``apollo_mavis_v2_hardware.driver._ServoStreamer._run`` on one arm, one streamer
    tick per loop tick: per-joint vel clip -> per-joint acc clip -> lever-weighted
    Cartesian scale -> joint limits. Velocity and the Cartesian step are speed-scaled
    like the driver factory does; acceleration is not."""

    def __init__(self, scale: float) -> None:
        from apollo_mavis_v2_hardware.config import XARM7_JOINT_LIMITS_RAD, ServoLimits

        lim = ServoLimits()
        dt = 1.0 / lim.rate_hz
        self.vel = np.asarray(lim.max_joint_vel) * scale * dt
        self.acc = np.asarray(lim.max_joint_acc) * dt * dt
        self.lever = np.asarray(lim.lever_arm_m)
        self.cart = lim.max_cart_step_m * scale
        lims = np.asarray(XARM7_JOINT_LIMITS_RAD)
        self.lo = lims[:, 0] + lim.joint_limit_margin_rad
        self.hi = lims[:, 1] - lim.joint_limit_margin_rad
        self.last: np.ndarray | None = None
        self.prev_dq = np.zeros(7)
        self.reset_counters()

    def reset_counters(self) -> None:
        self.n = self.n_vel = self.n_acc = self.n_lever = 0
        self.max_lag = 0.0

    def reseed(self, q7: np.ndarray) -> None:
        self.last = np.array(q7, dtype=np.float64)
        self.prev_dq = np.zeros(7)

    def step(self, target7: np.ndarray) -> np.ndarray:
        assert self.last is not None
        self.n += 1
        raw = target7 - self.last
        self.max_lag = max(self.max_lag, float(np.max(np.abs(raw))))
        dq = np.clip(raw, -self.vel, self.vel)
        self.n_vel += int(np.any(np.abs(raw) > self.vel + 1e-12))
        dq2 = np.clip(dq, self.prev_dq - self.acc, self.prev_dq + self.acc)
        self.n_acc += int(np.any(np.abs(dq2 - dq) > 1e-12))
        dq = dq2
        est = float(np.sum(np.abs(dq) * self.lever))
        if est > self.cart:
            dq = dq * (self.cart / est)
            # counted only when the scale does something: the host cap lands the step
            # exactly on the bound, and float drift alone trips the driver's strict `>`
            self.n_lever += int(est > self.cart * (1.0 + 1e-9))
        q = np.clip(self.last + dq, self.lo, self.hi)
        self.prev_dq = q - self.last
        self.last = q
        return q

    def line(self) -> str:
        return f"{self.n_vel:4d} {self.n_acc:4d} {self.n_lever:5d}  {self.max_lag:.4f}"


# -- the lockstep harness -----------------------------------------------------------------------
class Harness:
    def __init__(
        self,
        cfg: ControlConfig,
        *,
        scale: float = 1.0,
        streamer: bool = False,
        cap: str = "uniform",
        pullback: bool = True,
        tracker: bool = False,
    ) -> None:
        from apollo_mavis_v2_sim import REGISTRY, IKParams, MinkIKSolver, SimWorkcell

        from ..session.manager import _servo_faithful_scene

        self.cfg = cfg
        wc = WorkcellConfig(
            kind="sim",
            sim_scene="mavis_v2",
            arms=[
                ArmConfig(id="grip", base_in_world={}),
                ArmConfig(id="view", base_in_world={}, gripper="none"),
            ],
            cameras=[],
        )
        self.cell = SimWorkcell(_servo_faithful_scene("mavis_v2", None), wc)
        for arm in self.cell.arms.values():
            arm.connect()
        stock = REGISTRY.build("mavis_v2", None)
        self.ik = MinkIKSolver(
            stock, IKParams(min_distance_m=0.010, lock_rail=True), collision_pairs=None
        )
        self.kin = SceneKinematics(stock)
        self.bus = RuntimeBus()
        self.sup = SafetySupervisor(NullGate(), InputWatchdog())
        self.t = 0.0
        self.seq = 0
        self._sent: dict[str, float] = {}
        self.tracker: TrackerTeleop | None = None
        if tracker:
            self.tracker = TrackerTeleop(
                self.bus.tracker,
                TrackerSettings(filter_enabled=False),
                stale_s=0.2,
                leash_pos_m=cfg.leash.pos_m,
                leash_rot_rad=cfg.leash.rot_rad,
            )
        self.loop = ControlLoop(
            self.cell,
            cfg,
            self.bus,
            self.sup,
            list(ARMS),
            ik=self.ik,
            kin=self.kin,
            tracker=self.tracker,
            workcell_kind="sim",
            gripper_arms=["grip"],
            clock=lambda: self.t,
        )
        variant = CAP_VARIANTS[cap]
        if variant is not None:
            self.loop._cap_joint_step = types.MethodType(variant, self.loop)  # type: ignore[method-assign]
        if not pullback:
            self.loop._hold_key_target = types.MethodType(  # type: ignore[method-assign]
                lambda self, arm_id, q: None, self.loop
            )
        self.streamers = {a: StreamerEmu(scale) for a in ARMS} if streamer else {}
        self.loop._seed_from_measured()

    # -- plumbing --------------------------------------------------------------------------------
    def tick(self, n: int = 1, held: Sequence[str] = (), hand: Pose | None = None) -> None:
        for _ in range(n):
            self.t += DT
            self.seq += 1
            hs = HeldState(held=frozenset(held), seq=self.seq, rx_mono=self.t)
            self.bus.held_keys.put(hs)
            self.sup.watchdog.on_keys(hs)
            if hand is not None:
                self.bus.tracker.put(
                    TrackerSample(
                        hand,
                        np.zeros(3),
                        np.zeros(3),
                        self.t,
                        self.t,
                        self.seq,
                        True,
                        None,
                        frozenset(),
                        (),
                        pose_rx_mono=self.t,
                    )
                )
            self.loop.run_tick(self.t)
            for arm_id in ARMS:
                got = self.bus.arm_slot(arm_id).get()
                if got is None:
                    continue
                q, put_mono = got
                if self._sent.get(arm_id) == put_mono:
                    continue
                self._sent[arm_id] = put_mono
                q = np.asarray(q, dtype=np.float64)
                st = self.streamers.get(arm_id)
                if st is not None:
                    if st.last is None:
                        st.reseed(q[:7])
                    out = q.copy()
                    out[:7] = st.step(q[:7])
                    q = out
                self.cell.arms[arm_id].command_joints(q)
            self.cell.step_virtual(1)

    def teleport(self, arm: str, q: np.ndarray) -> None:
        import mujoco

        a = self.cell.scene.addressing[arm]
        data = self.cell._data
        q = np.asarray(q, dtype=np.float64)
        data.qpos[a.qpos_adr] = q
        data.qvel[a.dof_adr] = 0.0
        self.cell._targets[a.ctrl_adr] = q
        data.ctrl[:] = self.cell._targets
        mujoco.mj_forward(self.cell._model, data)
        self.cell._publish_snapshot()
        self.loop.reseed_arm(arm)
        st = self.streamers.get(arm)
        if st is not None:
            st.reseed(q[:7])

    def activate(self, arm: str) -> None:
        self.loop.active_arm = arm
        self.loop._teleop_seeded.discard(arm)

    def q(self, arm: str) -> np.ndarray:
        return np.array(self.cell.states()[arm].q)

    def tcp(self, arm: str, q: np.ndarray | None = None) -> Pose:
        return self.kin.tcp_world(arm, self.q(arm) if q is None else q)

    def key_dir(self, arm: str, key: str) -> np.ndarray:
        tcp = self.tcp(arm)
        tw = held_to_twist(frozenset({key}), self.cfg.teleop)
        out = twist_to_control_frame(
            Twist(v=tw.v, w=tw.w),
            self.kin.base_quat_world(arm),
            tcp.orientation,
            self.cfg.translate_frame,
            self.kin.wrist_cam_quat_world(arm, tcp.orientation),
        )
        return np.asarray(out.v) / float(np.linalg.norm(out.v))


def initial_q(h: Harness, arm: str) -> np.ndarray:
    q = h.q(arm)
    q[:7] = np.radians(INITIAL_DEG[arm])
    return q


def reach(h: Harness, arm: str, q0: np.ndarray, script: list[tuple[str, float]]) -> np.ndarray:
    """Hold keys in sequence from ``q0`` (with the harness config); the reached q."""
    h.teleport(arm, q0)
    h.activate(arm)
    h.tick(5)
    for key, secs in script:
        h.tick(int(round(secs / DT)), (key,))
    h.tick(50)
    return h.q(arm)


# -- measurements -------------------------------------------------------------------------------
def _split(dp: np.ndarray, d: np.ndarray) -> tuple[float, float]:
    on = float(dp @ d)
    return on, float(np.linalg.norm(dp - on * d))


def measure_key(h: Harness, arm: str, q0: np.ndarray, key: str, hold_s: float, settle_s: float):
    h.teleport(arm, q0)
    h.activate(arm)
    h.tick(10)
    st = h.streamers.get(arm)
    if st is not None:
        st.reset_counters()
    p0 = h.tcp(arm)
    d = h.key_dir(arm, key)
    c0, s0 = h.loop.clamp_ticks, h.loop.ik_slips
    h.tick(int(round(hold_s / DT)), (key,))
    h.tick(int(round(settle_s / DT)))
    p1 = h.tcp(arm)
    on, off = _split(p1.position - p0.position, d)
    return {
        "label": key,
        "on_mm": on * 1e3,
        "off_mm": off * 1e3,
        "off_per_100": off / max(abs(on), 1e-9) * 100.0,
        "rot_deg": math.degrees(se3.quat_geodesic(p0.orientation, p1.orientation)),
        "cap": h.loop.clamp_ticks - c0,
        "slips": h.loop.ik_slips - s0,
        "streamer": st.line() if st is not None else "",
        "requested_mm": h.cfg.teleop.linear_mps * hold_s * 1e3,
    }


def measure_hand(
    h: Harness, arm: str, q0: np.ndarray, axis: str, v_mps: float, hold_s: float, settle_s: float
):
    """Clutch held, the hand moving along a world axis at ``v_mps``: the arm should
    follow one to one (slower when capped), never off the line."""
    assert h.tracker is not None
    d = np.asarray(AXES[axis], dtype=np.float64)
    h.teleport(arm, q0)
    h.activate(arm)
    h.tick(10)
    st = h.streamers.get(arm)
    if st is not None:
        st.reset_counters()
    hand0 = np.array([0.5, 0.5, 1.0])
    h.tick(3, (TRACKER_CLUTCH_CODE,), Pose(hand0, IDENT))  # engage with a still hand
    p0 = h.tcp(arm)
    c0, s0 = h.loop.clamp_ticks, h.loop.ik_slips
    slips0 = h.tracker.slip_pos_total_m
    n = int(round(hold_s / DT))
    for i in range(1, n + 1):
        h.tick(1, (TRACKER_CLUTCH_CODE,), Pose(hand0 + d * v_mps * i * DT, IDENT))
    still = Pose(hand0 + d * v_mps * n * DT, IDENT)
    h.tick(int(round(settle_s / DT)), (TRACKER_CLUTCH_CODE,), still)
    p1 = h.tcp(arm)
    on, off = _split(p1.position - p0.position, d)
    return {
        "label": axis,
        "on_mm": on * 1e3,
        "off_mm": off * 1e3,
        "off_per_100": off / max(abs(on), 1e-9) * 100.0,
        "rot_deg": math.degrees(se3.quat_geodesic(p0.orientation, p1.orientation)),
        "cap": h.loop.clamp_ticks - c0,
        "slips": h.loop.ik_slips - s0,
        "streamer": st.line() if st is not None else "",
        "requested_mm": v_mps * hold_s * 1e3,
        "leash_slip_mm": (h.tracker.slip_pos_total_m - slips0) * 1e3,
    }


HDR = (
    f"{'arm':5} {'input':5} {'on mm':>8} {'req mm':>7} {'off mm':>7} {'off/100':>8} "
    f"{'rot deg':>8} {'cap':>4} {'slip':>4}"
)
HDR_STREAMER = "  | vel  acc  lever  maxlag(rad)"


def fmt(arm: str, r: dict, streamer: bool) -> str:
    line = (
        f"{arm:5} {r['label']:5} {r['on_mm']:8.1f} {r['requested_mm']:7.1f} {r['off_mm']:7.2f} "
        f"{r['off_per_100']:8.2f} {r['rot_deg']:8.3f} {r['cap']:4d} {r['slips']:4d}"
    )
    if "leash_slip_mm" in r:
        line += f"  leash_slip {r['leash_slip_mm']:6.1f} mm"
    if streamer:
        line += f"  | {r['streamer']}"
    return line


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scales", nargs="*", type=float, default=[1.0, 0.5])
    ap.add_argument(
        "--postures", nargs="*", default=["P0 initial", "P1 W1.5s"], choices=list(SCRIPTS)
    )
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--keys", nargs="*", default=list(KEYS), choices=list(KEYS))
    ap.add_argument("--hold", type=float, default=2.0, help="seconds each key / hand run is held")
    ap.add_argument("--settle", type=float, default=0.5)
    ap.add_argument("--streamer", action="store_true", help="emulate the hardware servo streamer")
    ap.add_argument("--cap", choices=list(CAP_VARIANTS), default="uniform")
    ap.add_argument("--no-pullback", action="store_true")
    ap.add_argument("--config", nargs="*", default=[], help="dq=<rad> leash=<m> nores")
    ap.add_argument("--tracker", action="store_true", help="scripted hand instead of keys")
    ap.add_argument("--axes", nargs="*", default=list(AXES), choices=list(AXES))
    ap.add_argument("--hand-mps", type=float, default=0.3)
    args = ap.parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")

    base = Harness(ControlConfig(translate_frame="world"))
    postures: dict[str, dict[str, np.ndarray]] = {}
    for name in args.postures:
        postures[name] = {a: reach(base, a, initial_q(base, a), SCRIPTS[name]) for a in args.arms}
        for a in args.arms:
            tcp = base.tcp(a, postures[name][a])
            print(
                f"posture {name:20} {a}: tcp {np.round(tcp.position, 3)} "
                f"q(deg) {np.round(np.degrees(postures[name][a][:7]), 1)}"
            )
    over = config_overrides(args.config)
    for scale in args.scales:
        cfg = hw_equivalent(scale, over)
        h = Harness(
            cfg,
            scale=scale,
            streamer=args.streamer,
            cap=args.cap,
            pullback=not args.no_pullback,
            tracker=args.tracker,
        )
        print(
            f"\n=== hw@{scale:g} cap={args.cap}{' no-pullback' if args.no_pullback else ''}"
            f"{' STREAMER-EMU' if args.streamer else ' host-only'}"
            f"{' tracker hand ' + str(args.hand_mps) + ' m/s' if args.tracker else ''}"
            f" {' '.join(args.config)}: dq_max {cfg.dq_max_rad:.4f} rad/tick, "
            f"cart {cfg.jog.plan_cart_step_m:.4f} m/tick, linear {cfg.teleop.linear_mps:.3f} m/s, "
            f"target_rate {cfg.target_rate.v_mps:.3f} m/s, leash {cfg.leash.pos_m} m ==="
        )
        for pname in args.postures:
            print(f"--- {pname} ---")
            print(HDR + (HDR_STREAMER if args.streamer else ""))
            worst_off = worst_rot = 0.0
            for arm in args.arms:
                inputs = args.axes if args.tracker else args.keys
                for inp in inputs:
                    if args.tracker:
                        r = measure_hand(
                            h, arm, postures[pname][arm], inp, args.hand_mps, args.hold, args.settle
                        )
                    else:
                        r = measure_key(h, arm, postures[pname][arm], inp, args.hold, args.settle)
                    print(fmt(arm, r, args.streamer))
                    worst_off = max(worst_off, r["off_per_100"])
                    worst_rot = max(worst_rot, r["rot_deg"])
            print(f"worst: off/100 {worst_off:.2f} mm, rot {worst_rot:.3f} deg")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
