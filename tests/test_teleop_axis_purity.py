"""Keyboard teleop axis purity on the ``mavis_v2`` sim over a REAL server (04-runtime
§6 "Per-tick joint step cap"; operator report 2026-09-09: "when I press forward/back
with the keyboard the arm also drifts up/down; a single-axis key should move only
along that axis").

Both arms, from the operator's initial-condition posture, translate frame ``world``,
at the control config a HARDWARE bring-up hands the loop at speed scale 1.0 and 0.5:
host scaling + the driver ``ServoLimits`` caps (joint 0.6 rad/s -> ``dq_max`` 0.006 /
0.003 rad per tick; lever-weighted Cartesian step 4 / 2 mm per tick via
``jog.plan_cart_step_m``; linear 0.12 / 0.06 m/s). These are the HOST-SIDE caps only:
the sim's servo has no streamer, so the driver's ``_ServoStreamer`` clips (which the
host cap now makes inactive) are not exercised here - the closed-loop emulation of the
streamer is ``tools/axis_purity_measure.py --streamer``. Each of ``W S A D E Q`` is
held 2 s; the Manipulation / Perception Arm TCP (world frame, FK of the telemetry
joints) must move along the key's axis only: off-axis translation <= 3 % of the
on-axis travel and orientation drift <= 0.5 deg. Until 2026-09-09 the loop clipped
the IK step PER JOINT at ``dq_max``: whenever one joint saturated the others kept
their full step and the direction bent - 6.4 mm down + 1.7 deg on a 237 mm ``S`` at
100 %, 1.8 deg on ``Q`` at 50 % (from this posture; 15-39 mm down on ``W`` from the
posture one ``W`` further). The fix scales the whole step uniformly - by the larger of
the per-joint and the lever-weighted Cartesian ratio - and pulls the keyboard target
back along the driven axes to what the arm achieved (measurement history in
04-runtime §6). With the Cartesian bound the arm runs at the streamer's real capacity,
well below the 0.12 m/s the key requests, so the cap binds on nearly every tick.
Measured here 2026-09-09 (both arms, both speeds, six keys): worst 0.59 % off-axis and
0.051 deg; on-axis 44-182 mm per 2 s hold against 120 / 240 mm requested; ``dq_capped``
194-201 of 200 ticks; 0 IK slips.

Reverting ``ControlLoop._cap_joint_step`` to the per-joint ``np.clip`` fails this file
at 100 % (verified 2026-09-09, twice); the 50 % parametrizations are a regression guard
only - whether they fail with the old clamp is timing-dependent (2 of 4 and 3 of 4
parametrizations failed on two runs). The pure behaviours (exact uniform scaling, both
bounds, the pull-back geometry, ``clamp_ticks``) are pinned by the fast
``tests/test_joint_step_cap.py``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import httpx
import numpy as np
import pytest
from apollo_mavis_v2_core import ArmPosture, StateProfile, Twist, se3
from conftest import LiveServer, make_runtime_config
from test_e2e_teleop import PulsingCtl, Tele

from apollo_mavis_v2_runtime.config import ControlConfig
from apollo_mavis_v2_runtime.control.teleop import held_to_twist, twist_to_control_frame
from apollo_mavis_v2_runtime.profiles.seed_initial import DEFAULT_POSTURE_DEG
from apollo_mavis_v2_runtime.session.hardware import (
    ExecutorCaps,
    apply_executor_caps,
    apply_teleop_caps,
    scale_control_config,
)

pytest.importorskip("apollo_mavis_v2_sim")
pytestmark = pytest.mark.egl

ARMS = ("grip", "view")
KEYS = ("KeyW", "KeyS", "KeyA", "KeyD", "KeyE", "KeyQ")
HOLD_S = 2.0
SETTLE_S = 0.5
OFF_AXIS_MAX_FRACTION = 0.03  # off-axis translation vs on-axis travel
ROT_DRIFT_MAX_DEG = 0.5  # per 2 s hold
MIN_TRAVEL_FRACTION = 0.25  # the arm must at least move (joint-capped postures are slower)
# The Perception Arm's ``W`` from the initial posture runs its elbow (joint 4) into the
# -11 deg stop after ~190 mm (the tool is 36 cm on the operator's side of the base and
# the arm straightens as it comes back); past that point the sim's link2/link4 contact
# pushes the TCP up. That is the workspace boundary, not the step cap, so this one hold
# stays short of it. With the Cartesian bound the arm covers only ~90 mm in 1.5 s at
# 100 %, so the stop is out of reach anyway; the entry stays as a guard against a
# raised ServoLimits cap. Everything else holds the full 2 s.
SHORT_HOLDS: dict[tuple[str, str, float], float] = {("view", "KeyW", 1.0): 1.5}
# The driver's ``ServoLimits`` defaults at speed scale 1.0 (apollo_mavis_v2_hardware;
# 02-hardware / CLAUDE.md "Driver caps"): read from the [hardware] extra when it is
# installed, so a ServoLimits change moves this test with the real chain; the constants
# are the documented fallback for a checkout without the extra.
SERVO_JOINT_RADPS = 0.6
SERVO_CART_STEP_M = 0.004
SERVO_LEVER_ARM_M = (1.20, 1.20, 1.00, 0.75, 0.44, 0.30, 0.10)


def servo_defaults() -> tuple[float, float, tuple[float, ...]]:
    """``(min max_joint_vel, max_cart_step_m, lever_arm_m)`` of the driver's defaults."""
    try:
        from apollo_mavis_v2_hardware.config import ServoLimits  # [hardware] extra
    except Exception:  # noqa: BLE001 - optional extra absent
        return SERVO_JOINT_RADPS, SERVO_CART_STEP_M, SERVO_LEVER_ARM_M
    lim = ServoLimits()
    return (
        min(float(v) for v in lim.max_joint_vel),
        float(lim.max_cart_step_m),
        tuple(float(v) for v in lim.lever_arm_m),
    )


SPEC = {
    "mode": "teleop",
    "kind": "sim",
    "arms": list(ARMS),
    "frames": {a: f"arm_base:{a}" for a in ARMS},
    "sim_scene": "mavis_v2",
}


def hardware_equivalent_control(scale: float) -> ControlConfig:
    """The ``ControlConfig`` ``SessionManager.connect_hardware_rig`` hands the loop at
    ``speed_scale``: host scaling, then the executor + teleop caps of the servo stream
    (``dq_max`` = 0.6 rad/s * scale / 100 Hz, ``jog.plan_cart_step_m`` = 4 mm * scale
    with the driver's lever arms). A sim session applies neither, so the server is
    built with this config directly. Host-side caps only: the sim has no streamer."""
    joint_radps, cart_step_m, lever = servo_defaults()
    cfg = scale_control_config(ControlConfig(translate_frame="world"), scale)
    joint_step = joint_radps * scale / cfg.rate_hz
    caps = ExecutorCaps(
        slew_rad_per_tick=min(cfg.jog.slew_rad_per_tick, joint_step),
        cart_step_m=cart_step_m * scale,
        lever_arm_m=lever,
        source="servo",
        joint_step_rad=joint_step,
    )
    return apply_teleop_caps(apply_executor_caps(cfg, caps), caps)


def initial_profile() -> StateProfile:
    """The operator's default posture (``profiles/seed_initial``) as a sim profile. Joint 1
    is spelled +pi instead of -180 deg: the same posture, but the sim keyframe sits at +pi
    and the start_from plan would otherwise turn the base joint a full 2 pi."""
    arms = {}
    for arm_id, deg in DEFAULT_POSTURE_DEG.items():
        q = [math.radians(d) for d in deg]
        if q[0] < -math.pi + 1e-6:
            q[0] += 2.0 * math.pi
        arms[arm_id] = ArmPosture(q=q, rail_pos_m=None, gripper_open_frac=1.0)
    return StateProfile(name="axis purity start", workcell_kind="sim", arms=arms)


@dataclass
class Row:
    arm: str
    key: str
    on_mm: float
    off_mm: float
    rot_deg: float
    cap_ticks: int
    ik_slips: int

    @property
    def off_fraction(self) -> float:
        return self.off_mm / max(abs(self.on_mm), 1e-9)

    def line(self) -> str:
        return (
            f"{self.arm:5} {self.key:5} on {self.on_mm:7.1f} mm  off {self.off_mm:6.2f} mm "
            f"({self.off_fraction * 100:5.2f} %)  rot {self.rot_deg:6.3f} deg  "
            f"dq_capped {self.cap_ticks:3d}  ik_slips {self.ik_slips}"
        )


class Rig:
    """A running teleop session on ``mavis_v2`` at one speed scale + the helpers."""

    def __init__(self, server: LiveServer, scale: float) -> None:
        from apollo_mavis_v2_sim import REGISTRY

        from apollo_mavis_v2_runtime.control.fk import SceneKinematics

        self.server = server
        self.scale = scale
        self.api = httpx.Client(base_url=server.http, timeout=30.0)
        self.kin = SceneKinematics(REGISTRY.build("mavis_v2", None))
        profile = server.runtime.profile_store.save(initial_profile())
        spec = dict(SPEC, start_from=f"profile:{profile.profile_id}")
        r = self.api.post("/api/session", json=spec)
        assert r.status_code == 200, r.text
        deadline = time.monotonic() + 60.0
        while self.api.get("/api/session").json()["state"] != "running":
            assert time.monotonic() < deadline, "session never reached running"
            time.sleep(0.05)
        self.loop = server.runtime.manager.session.loop
        self.cfg: ControlConfig = self.loop.cfg
        self.ctl = PulsingCtl(server)
        self.tele = Tele(server)
        self.q0 = {a: self.q(a) for a in ARMS}  # the start_from posture, measured

    def close(self) -> None:
        self.ctl.close()
        self.tele.close()
        self.api.delete("/api/session")
        self.api.close()

    # -- observation -------------------------------------------------------------------------
    def q(self, arm: str) -> np.ndarray:
        msg = self.tele.latest()
        a = next(x for x in msg["arms"] if x["arm_id"] == arm)
        return np.asarray(list(a["q"]) + [a["rail_pos_m"]], dtype=np.float64)

    def tcp(self, arm: str):
        return self.kin.tcp_world(arm, self.q(arm))

    def key_direction(self, arm: str, key: str) -> np.ndarray:
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

    # -- actions -----------------------------------------------------------------------------
    def activate(self, arm: str) -> None:
        ack = self.ctl.action("switch_arm", {"arm_id": arm})
        assert ack["ok"], ack

    def reset_to_start(self, arm: str) -> None:
        """Jog back to the start posture (joint space, no IK) and wait for arrival."""
        goal = self.q0[arm]
        ack = self.ctl.action(
            "joint_target", {"arm_id": arm, "positions": [float(x) for x in goal], "mode": "jog"}
        )
        assert ack["ok"], ack
        deadline = time.monotonic() + 10.0
        while np.max(np.abs(self.q(arm)[:7] - goal[:7])) > 2e-3:
            assert time.monotonic() < deadline, f"{arm} never got back to the start posture"
            time.sleep(0.05)
        time.sleep(0.3)

    def hold_and_measure(self, arm: str, key: str) -> Row:
        hold_s = SHORT_HOLDS.get((arm, key, self.scale), HOLD_S)
        self.reset_to_start(arm)
        p0 = self.tcp(arm)
        d = self.key_direction(arm, key)
        cap0, slip0 = self.loop.clamp_ticks, self.loop.ik_slips
        self.ctl.hold([key], hold_s)
        time.sleep(SETTLE_S)
        p1 = self.tcp(arm)
        dp = p1.position - p0.position
        on = float(dp @ d)
        off = float(np.linalg.norm(dp - on * d))
        rot = math.degrees(se3.quat_geodesic(p0.orientation, p1.orientation))
        return Row(
            arm,
            key,
            on * 1e3,
            off * 1e3,
            rot,
            self.loop.clamp_ticks - cap0,
            self.loop.ik_slips - slip0,
        )


@pytest.fixture(scope="module", params=[1.0, 0.5], ids=["speed100", "speed50"])
def rig(request, tmp_path_factory):
    scale = request.param
    cfg = make_runtime_config(tmp_path_factory.mktemp("rt"), scene="mavis_v2")
    cfg = cfg.model_copy(update={"control": hardware_equivalent_control(scale)})
    server = LiveServer(cfg)
    r = Rig(server, scale)
    try:
        yield r
    finally:
        r.close()
        server.stop()


@pytest.mark.parametrize("arm", ARMS, ids=["manipulation_arm", "perception_arm"])
def test_single_axis_keys_move_along_their_axis_only(rig: Rig, arm: str):
    rig.activate(arm)
    rate = rig.cfg.teleop.linear_mps
    rows = [rig.hold_and_measure(arm, key) for key in KEYS]
    table = "\n".join(r.line() for r in rows)
    print(f"\n[axis purity] speed {rig.scale:.1f}, {arm}, dq_max {rig.cfg.dq_max_rad:.4f}\n{table}")
    for r in rows:
        hold_s = SHORT_HOLDS.get((arm, r.key, rig.scale), HOLD_S)
        assert r.on_mm >= MIN_TRAVEL_FRACTION * rate * hold_s * 1e3, f"barely moved:\n{table}"
        assert r.off_fraction <= OFF_AXIS_MAX_FRACTION, f"off-axis drift on {r.key}:\n{table}"
        assert r.rot_deg <= ROT_DRIFT_MAX_DEG, f"orientation drift on {r.key}:\n{table}"
    # the cap is exercised on the default teleop arm at both speeds: the assertion above
    # is about a saturated loop, not an idle one (Q from this posture caps 28-64 ticks)
    if arm == "grip":
        assert sum(r.cap_ticks for r in rows) > 0, f"dq cap never bound:\n{table}"
    assert rig.loop.ik_diverged == 0, "IK diverged during the holds"
