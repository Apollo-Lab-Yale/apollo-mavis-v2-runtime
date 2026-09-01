"""RuntimeConfig — YAML-loadable runtime configuration (04-runtime §14)."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from apollo_xarm7_core import ConfigError, WorkcellConfig
from pydantic import BaseModel, Field, ValidationError


class TeleopRates(BaseModel):
    """Held-key integration rates (04-runtime §6 defaults)."""

    linear_mps: float = 0.12
    angular_rps: float = 0.6
    rail_mps: float = 0.10
    gripper_frac_ps: float = 1.2


class LeashConfig(BaseModel):
    pos_m: float = 0.025
    rot_rad: float = 0.2


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
    vcodec: str = "auto"  # rgb encoder; "auto" -> NVENC when available, else libsvtav1
    jpeg_quality: int = 80
    image_writer_threads: int = 4  # PNG fallback path only
    # Checkpoint-load extrinsics verification thresholds (10-frames §5.3; phase-08).
    extrinsics_warn: ExtrinsicsTolerance = ExtrinsicsTolerance(pos_m=0.003, rot_rad=0.010)
    extrinsics_max: ExtrinsicsTolerance = ExtrinsicsTolerance(pos_m=0.010, rot_rad=0.035)


class RuntimeConfig(BaseModel):
    """Top-level runtime config; sane defaults for sim-only dev."""

    host: str = "127.0.0.1"
    port: int = 8000
    ui_dist: Path | None = None  # built SPA; None = API-only (Vite dev)
    workcells: dict[str, WorkcellConfig] = Field(default_factory=dict)  # "hardware"|"sim"
    profiles_dir: Path = Path("~/apollo/profiles")
    datasets_root: Path = Path("~/apollo/datasets")
    checkpoints_root: Path = Path("~/apollo/checkpoints")
    control: ControlConfig = ControlConfig()
    recorder: RecorderConfig = RecorderConfig()
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
    "JogConfig",
    "WatchdogConfig",
    "ControlConfig",
    "ExtrinsicsTolerance",
    "RecorderConfig",
    "VideoConfig",
    "RuntimeConfig",
    "load_runtime_config",
]
