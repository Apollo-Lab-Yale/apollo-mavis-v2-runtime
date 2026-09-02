"""RuntimeConfig — YAML-loadable runtime configuration (04-runtime §14)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from apollo_xarm7_core import ConfigError, WorkcellConfig
from pydantic import BaseModel, Field, ValidationError, model_validator


class TeleopRates(BaseModel):
    """Held-key integration rates (04-runtime §6 defaults)."""

    linear_mps: float = 0.12
    angular_rps: float = 0.6
    rail_mps: float = 0.10
    gripper_frac_ps: float = 1.2


class LeashConfig(BaseModel):
    pos_m: float = 0.025
    rot_rad: float = 0.2


class TargetRateConfig(BaseModel):
    """Cartesian approach rate of the clutched tracker target toward the IK
    (04-runtime §6 "Target rate limit"): per tick the target handed to IK moves
    from the previous commanded target by at most ``v_mps·dt`` / ``w_radps·dt``.
    Keeps the single-step QP out of joint-velocity saturation, where it would
    trade position for orientation. Truncation is not slipped into the anchor.
    Both rates must be > 0: ``clamp_pose_to_leash`` with a zero cap freezes the
    clutched target and a negative cap steps AWAY from the hand."""

    v_mps: float = Field(default=1.0, gt=0.0)
    w_radps: float = Field(default=2.0, gt=0.0)


class JogConfig(BaseModel):
    slew_rad_per_tick: float = 0.02
    rail_m_per_tick: float = 0.002
    goto_threshold_rad: float = 0.15


class WatchdogConfig(BaseModel):
    stale_s: float = 0.2  # = SafetyConfig.input_deadman_s
    ramp_s: float = 0.1  # = SafetyConfig.input_ramp_s


class ControlConfig(BaseModel):
    rate_hz: float = 100.0
    teleop: TeleopRates = TeleopRates()
    leash: LeashConfig = LeashConfig()
    target_rate: TargetRateConfig = TargetRateConfig()
    dq_max_rad: float = 0.04  # per tick
    jog: JogConfig = JogConfig()
    watchdog: WatchdogConfig = WatchdogConfig()
    residual_max_pos_m: float = 0.01  # IK residual: freeze target back (glide)
    residual_max_rot_rad: float = 0.1


class VideoConfig(BaseModel):
    preview_fps: float = 15.0
    session_fps: float = 30.0
    jpeg_quality: int = 80


class ExtrinsicsTolerance(BaseModel):
    pos_m: float
    rot_rad: float


class RecorderConfig(BaseModel):
    """Episode recorder tuning (04-runtime §10/§14; 10-frames §7.5)."""

    fps: int = Field(default=25, ge=20, le=30)  # dataset fps; 20-30 band (binding)
    # rgb encoder; "auto" -> first hardware encoder that really OPENS with lerobot's
    # options (NVENC on the 4090s, needs bf=0), else libsvtav1. A resumed dataset
    # keeps its codec family (10-frames §7.5).
    vcodec: str = "auto"
    jpeg_quality: int = 80
    image_writer_threads: int = 4  # PNG fallback path only
    # Checkpoint-load extrinsics verification thresholds (10-frames §5.3; phase-08).
    extrinsics_warn: ExtrinsicsTolerance = ExtrinsicsTolerance(pos_m=0.003, rot_rad=0.010)
    extrinsics_max: ExtrinsicsTolerance = ExtrinsicsTolerance(pos_m=0.010, rot_rad=0.035)


class TrainerSettings(BaseModel):
    """AsyncTrainer process knobs (12-dagger §7 defaults)."""

    port: int = 5757  # ZMQ REP control endpoint, tcp://127.0.0.1
    device: str = "cuda:1"  # GPU 1 on the target box (render/inference own GPU 0)
    cuda_visible_devices: str | None = "1"
    min_new_labels: int = 100
    push_period_s: float = 5.0
    batch_size: int = 64
    lr: float = 1e-5


class DaggerConfig(BaseModel):
    """DAgger/inference session tuning (12-dagger §2/§6/§7)."""

    t_blend_s: float = Field(default=0.3, ge=0.2, le=0.5)
    policy_rate_hz: float = Field(default=15.0, ge=10.0, le=30.0)
    policy_device: str = "cuda:0"  # runtime-side inference; falls back to cpu
    slew_window_s: float = Field(default=0.4, ge=0.3, le=0.5)
    trainer: TrainerSettings = TrainerSettings()


ControllerInput = Literal[
    "trigger_click",
    "trackpad_left",
    "trackpad_right",
    "trackpad_up",
    "trackpad_down",
    "menu_click",
    "grip_click",
    "none",
]
"""Vive-controller inputs a teleop action may be bound to (13-tracker §1.1).

``trigger_click`` = trigger button (id 0) pressed (held-only: it registers no
press edge). ``trackpad_left`` / ``trackpad_right`` / ``trackpad_up`` /
``trackpad_down`` = trackpad button (id 1) pressed, classified ONCE at the press
edge from the pad position by the dominant axis (``|x|`` and ``|y|`` both below
``trackpad_deadzone`` => the click is ignored) and held until release.
``menu_click`` = menu button (id 6), ``grip_click`` = grip button (id 7).
``none`` = unbound. The system button (id 3) is deliberately not bindable:
menu + system is the dongle pairing combo.
"""

HELD_ONLY_INPUTS: frozenset[str] = frozenset({"trigger_click"})  # no press edge -> never discrete
# The controller_map actions by kind. ``devices/tracker.py`` keys its code /
# action tables by these same names and cross-checks them at import time, and
# ``ControllerMapConfig`` is checked below to declare exactly these fields.
CONTROLLER_HELD_ACTIONS: tuple[str, ...] = (
    "clutch",
    "gripper_open",
    "gripper_close",
    "rail_neg",
    "rail_pos",
)
CONTROLLER_DISCRETE_ACTIONS: tuple[str, ...] = ("arm_next", "arm_prev")


class ControllerMapConfig(BaseModel):
    """Controller input per teleop action (13-tracker §1.1).

    Held actions (``clutch`` / ``gripper_open`` / ``gripper_close`` /
    ``rail_neg`` / ``rail_pos``) inject the core keymap code of the action of
    the same name (``tracker_clutch`` for the clutch; looked up by action, never
    hard-coded) while the input is active; the rail codes drive the rail at the
    device source's scale exactly like the arrow keys. Discrete actions
    (``arm_next`` -> ``switch_arm``, ``arm_prev`` -> ``switch_arm_prev``) fire
    once per press edge of the bound input inside the control loop, subject to
    the same nacks as the WS actions; they accept any input except
    ``trigger_click`` (held-only). Every input may be bound to at most one
    action.
    """

    clutch: ControllerInput = "trigger_click"
    gripper_open: ControllerInput = "trackpad_up"
    gripper_close: ControllerInput = "trackpad_down"
    rail_neg: ControllerInput = "trackpad_left"
    rail_pos: ControllerInput = "trackpad_right"
    arm_next: ControllerInput = "menu_click"
    arm_prev: ControllerInput = "none"

    @model_validator(mode="after")
    def _check_bindings(self) -> ControllerMapConfig:
        for name in CONTROLLER_DISCRETE_ACTIONS:
            value = getattr(self, name)
            if value in HELD_ONLY_INPUTS:
                raise ValueError(
                    f"{name} is a discrete action and cannot be bound to the held-only input "
                    f"{value!r}; use trackpad_*, menu_click, grip_click or none"
                )
        bound = [v for v in self.model_dump().values() if v != "none"]
        dup = sorted({v for v in bound if bound.count(v) > 1})
        if dup:
            raise ValueError(f"controller input bound to more than one action: {dup}")
        return self


if set(ControllerMapConfig.model_fields) != set(CONTROLLER_HELD_ACTIONS) | set(
    CONTROLLER_DISCRETE_ACTIONS
):  # pragma: no cover - import-time invariant
    raise RuntimeError(
        "ControllerMapConfig fields must be exactly CONTROLLER_HELD_ACTIONS + "
        "CONTROLLER_DISCRETE_ACTIONS (13-tracker §1.1)"
    )


class TrackerFilterConfig(BaseModel):
    """One Euro pose filter on the aligned tracker pose (13-tracker §4 "Pose
    filter"); mirrors ``control.pose_filter.PoseFilterConfig``. ``enabled`` /
    ``min_cutoff_hz`` / ``beta`` are live-tunable via ``tracker_settings``."""

    enabled: bool = True
    min_cutoff_hz: float = Field(default=1.0, ge=0.05, le=50.0)  # cutoff at rest
    beta: float = Field(default=0.05, ge=0.0, le=5.0)  # speed coefficient
    d_cutoff_hz: float = Field(default=1.0, gt=0.0)  # velocity-estimate cutoff
    deadband_m: float = Field(default=0.002, ge=0.0)  # rest deadband, position
    deadband_rad: float = Field(default=0.005, ge=0.0)  # rest deadband, orientation


class TrackerConfig(BaseModel):
    """Vive-tracker teleop device + defaults (13-tracker §4).

    ``yaw_deg`` / ``pos_scale`` / ``follow_rotation`` are the process-lifetime
    defaults; the ``tracker_settings`` action mutates the live values.
    ``controller_map`` / ``trackpad_deadzone`` bind the paired controller's
    buttons to device-held key codes (clutch, gripper, rail) and to the
    discrete arm-switch actions (13-tracker §1.1).
    """

    backend: Literal["none", "fake", "libsurvive"] = "none"
    object_name: str = "WM0"  # libsurvive codename of the dongle-paired tracker
    libsurvive_args: list[str] = Field(default_factory=lambda: ["--lighthousecount", "2"])
    yaw_deg: float = 0.0  # lighthouse world -> MJCF world (both z-up; yaw only)
    pos_scale: float = Field(default=1.0, ge=0.1, le=3.0)
    follow_rotation: bool = True
    stale_s: float = 0.2  # sample older than this -> hold
    max_jump_m: float = 0.10  # consecutive-sample jump above this -> invalid sample
    controller_map: ControllerMapConfig = ControllerMapConfig()
    trackpad_deadzone: float = Field(default=0.3, ge=0.0, le=1.0)  # |x|,|y| <= dz: click ignored
    filter: TrackerFilterConfig = TrackerFilterConfig()


class RuntimeConfig(BaseModel):
    """Top-level runtime config; sane defaults for sim-only dev."""

    host: str = "127.0.0.1"
    port: int = 8765  # 8000 is commonly taken on dev boxes (gohttpserver on the lab machine)
    ui_dist: Path | None = None  # built SPA; None = API-only (Vite dev)
    workcells: dict[str, WorkcellConfig] = Field(default_factory=dict)  # "hardware"|"sim"
    profiles_dir: Path = Path("~/apollo/profiles")
    datasets_root: Path = Path("~/apollo/datasets")
    checkpoints_root: Path = Path("~/apollo/checkpoints")
    control: ControlConfig = ControlConfig()
    recorder: RecorderConfig = RecorderConfig()
    dagger: DaggerConfig = DaggerConfig()
    tracker: TrackerConfig = TrackerConfig()
    telemetry_hz: float = 25.0
    video: VideoConfig = VideoConfig()
    egl_device_id: int = 0

    def model_post_init(self, __context) -> None:
        for name in ("profiles_dir", "datasets_root", "checkpoints_root"):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser())
        if self.ui_dist is not None:
            object.__setattr__(self, "ui_dist", Path(self.ui_dist).expanduser())

    def workcell_config(self, kind: str) -> WorkcellConfig | None:
        return self.workcells.get(kind)


def load_runtime_config(path: str | Path | None = None) -> RuntimeConfig:
    """Load config from ``path``, ``$APOLLO_CONFIG``, or defaults (no file)."""
    if path is None:
        env = os.environ.get("APOLLO_CONFIG")
        if env:
            path = env
    if path is None:
        return RuntimeConfig()
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(str(e), path=str(p)) from e
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML: {e}", path=str(p)) from e
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError("expected a mapping at the document root", path=str(p), loc="/")
    try:
        return RuntimeConfig.model_validate(data)
    except ValidationError as e:
        first = e.errors()[0]
        loc = "/" + "/".join(str(part) for part in first["loc"])
        raise ConfigError(first["msg"], path=str(p), loc=loc) from e


__all__ = [
    "TeleopRates",
    "LeashConfig",
    "TargetRateConfig",
    "JogConfig",
    "WatchdogConfig",
    "ControlConfig",
    "TrainerSettings",
    "DaggerConfig",
    "ExtrinsicsTolerance",
    "RecorderConfig",
    "ControllerInput",
    "HELD_ONLY_INPUTS",
    "CONTROLLER_HELD_ACTIONS",
    "CONTROLLER_DISCRETE_ACTIONS",
    "ControllerMapConfig",
    "TrackerFilterConfig",
    "TrackerConfig",
    "VideoConfig",
    "RuntimeConfig",
    "load_runtime_config",
]
