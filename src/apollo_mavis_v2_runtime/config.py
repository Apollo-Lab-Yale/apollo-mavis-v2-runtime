"""RuntimeConfig — YAML-loadable runtime configuration (04-runtime §14)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from apollo_mavis_v2_core import ConfigError, WorkcellConfig
from pydantic import BaseModel, Field, ValidationError, model_validator

# -- self-contained paths (04-runtime §14) --------------------------------------------
# Every filesystem path in the config resolves inside the workspace so a fresh
# ``git clone --recurse-submodules`` runs with no ``~/apollo`` (or any other
# machine-specific) dependency: the config anchors data at ``${APOLLO_HOME}/var/...``.
# ``${APOLLO_HOME}`` is the workspace root — ``$APOLLO_HOME`` when the launcher /
# systemd unit exports it, otherwise inferred from the config file's location (and,
# as a last resort, from this installed package's location). Absolute paths and ``~``
# still work verbatim, so the rendered ops config may pin FHS paths (/var/lib/...).
_WS_MARKERS = ("apollo-mavis-v2-core", "apollo-mavis-v2-runtime")


def _find_workspace_root(start: Path) -> Path | None:
    """Nearest ancestor of ``start`` holding the side-by-side sub-repos (the ws
    root). ``None`` when ``start`` is not inside such a checkout (installed
    wheel); ``$APOLLO_HOME`` must then be set explicitly."""
    try:
        start = start.resolve()
    except OSError:
        return None
    for d in (start, *start.parents):
        if all((d / m).is_dir() for m in _WS_MARKERS):
            return d
    return None


@lru_cache(maxsize=1)
def _package_workspace_root() -> Path | None:
    return _find_workspace_root(Path(__file__).parent)


def apollo_home() -> str | None:
    """Value ``${APOLLO_HOME}`` expands to: ``$APOLLO_HOME`` if exported, else the
    ws root inferred from this package's location. ``load_runtime_config`` also
    seeds ``$APOLLO_HOME`` from the config file so child processes inherit it."""
    env = os.environ.get("APOLLO_HOME")
    if env:
        return env
    root = _package_workspace_root()
    return str(root) if root is not None else None


def _resolve_path(value: str | Path) -> Path:
    """Expand ``${APOLLO_HOME}`` / other ``$VARS`` / ``~`` in a config path and
    anchor a still-relative result at the workspace root."""
    s = str(value)
    home = apollo_home()
    if "${APOLLO_HOME}" in s or "$APOLLO_HOME" in s:
        if not home:
            # Never fall through to a literal "${APOLLO_HOME}/var/..." directory
            # (2026-09-07): an installed wheel outside a checkout must be told
            # where the workspace is.
            raise ConfigError(
                f"path {s!r} uses ${{APOLLO_HOME}} but $APOLLO_HOME is not set and no "
                "workspace root (apollo-mavis-v2-core + apollo-mavis-v2-runtime side by "
                "side) was found above the config file or the installed package"
            )
        s = s.replace("${APOLLO_HOME}", home).replace("$APOLLO_HOME", home)
    p = Path(os.path.expandvars(s)).expanduser()
    if not p.is_absolute() and home:
        p = Path(home) / p
    return p


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
    # Hardware executor caps (phase-09d; set PROGRAMMATICALLY by the hardware bring-up
    # from the connected driver's ServoLimits, not meant for YAML): the PlanExecutor
    # also bounds sum|dq_j| * plan_lever_arm_m[j] per tick by plan_cart_step_m, the
    # lever-weighted Cartesian step the driver's servo streamer enforces, so the
    # commanded path never runs ahead of what the arm can follow along the validated
    # straight segment. None = joint slew only (sim).
    plan_cart_step_m: float | None = None
    plan_lever_arm_m: tuple[float, ...] | None = None


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
    # Rail in the differential IK? False (default, 2026-09-03): the IK never moves
    # the rail to reach a target; only rail inputs (arrow keys / controller
    # trackpad) do, and they slide the WHOLE arm (the world-frame target and the
    # tracker anchors ride along). True: the rail is an (expensive) IK dof.
    rail_in_ik: bool = False
    # Control-loop health line period, s (2026-09-07; LoggingConfig): one INFO line
    # with tick rate / overruns / clutch / tracker ages / gate / IK slip counters /
    # servo-stream stats. 0 disables it.
    health_log_every_s: float = Field(default=1.0, ge=0.0)


class VideoConfig(BaseModel):
    preview_fps: float = 15.0
    session_fps: float = 30.0
    jpeg_quality: int = 80


class MicrophoneConfig(BaseModel):
    """Runtime-owned microphone preview (phase-11; 04-runtime §13.3/§14).

    The RØDE NT-USB Mini on the Perception Arm (arm id ``view``) is captured
    THROUGH PulseAudio (never ``hw:``: Pulse owns the card, a direct open fails
    with EBUSY and stalls every other Pulse client). ``backend`` ``auto`` tries
    ``sounddevice`` (PortAudio's ALSA ``pulse`` plugin, source pinned via
    ``PULSE_SOURCE``), then a ``parec`` subprocess; ``fake`` synthesizes an
    amplitude-modulated sine for tests; ``none`` disables capture while the
    microphone stays listed. Frame rate = ``RuntimeConfig.telemetry_hz`` so
    every telemetry tick carries exactly one new frame (25 Hz -> 1920 samples
    = 64 bins x 30 samples).
    """

    enabled: bool = False  # list + capture the microphone (Hardware tab)
    mic_id: str = "mic_view"
    label: str = "Perception Arm microphone"  # user-facing; the id stays mic_view
    backend: Literal["auto", "sounddevice", "parec", "fake", "none"] = "auto"
    source_match: str = "NT-USB Mini"  # substring of the Pulse source name/description
    sample_rate: int = Field(default=48000, gt=0)
    bins: int = Field(default=64, ge=1, le=1024)  # envelope bins per frame (int8 min/max)
    stale_s: float = Field(default=0.5, gt=0.0)  # no frame for this long -> "stalled"


class HardwareProbeConfig(BaseModel):
    """Reachability probe for the configured hardware arms (phase-11): a
    background thread does a TCP connect-and-close on ``ip:port`` (xArm control
    port 502; never writes a byte) every ``period_s`` and publishes
    ``ArmStatusInfo.reachable`` / ``WorkcellStatus.hardware_ready``. Paused
    while a hardware session runs."""

    enabled: bool = True
    period_s: float = Field(default=2.0, gt=0.0)
    timeout_s: float = Field(default=1.0, gt=0.0)
    port: int = Field(default=502, ge=1, le=65535)


class HardwareMonitorConfig(BaseModel):
    """Read-only controller state monitor (phase-09a; 04-runtime §13.3
    ``hardware_monitor``): one ``ArmStateMonitor`` (hardware package) per
    configured hardware arm polls joint angles, flange pose, error/warn codes,
    linear-track and gripper registers at ``poll_hz`` and NEVER commands the
    box. A sample older than ``stale_s`` reports ``stale``; connect / read
    failures retry from ``reconnect_s`` with exponential backoff (cap 10 s).
    Paused (= connections RELEASED, two SDK clients on one box are unevidenced)
    while a hardware session owns the arms."""

    enabled: bool = True
    poll_hz: float = Field(default=10.0, gt=0.0)
    stale_s: float = Field(default=0.5, gt=0.0)
    reconnect_s: float = Field(default=2.0, gt=0.0)


RGB = tuple[int, int, int]


class TwinOverlayConfig(BaseModel):
    """Digital-twin alignment overlays (phase-09a; 04-runtime §13.4 ``*_align``).

    One ``<camera_id><stream_suffix>`` stream per hardware wrist camera: the
    real frame with the ``mavis_v2`` twin - posed from the monitor's joint
    angles / rail position and rendered from the SAME wrist camera with the
    D435i colour intrinsics (``CameraConfig.intrinsics``) - composited as a
    pale-yellow translucent silhouette (``alpha``, ``tint_rgb`` shaded by the
    twin's own luminance, 1 px ``edge_rgb`` outline). ``env_outline`` draws the
    table / obstacle edges in ``env_rgb`` as the base-placement cue; a stale /
    erroring monitor swaps the tint for ``stale_tint_rgb``.
    ``joint1_offset_rad`` is a diagnostic knob only (the identity joint
    convention is verified); ``rail_flip`` maps ``q_sim = 0.65 - q_track`` -
    since phase-09c it is a DEPRECATED ALIAS of ``hardware_session.rail_flip``
    (the overlay and the gate / sweep twins must agree; ``RuntimeConfig``
    unifies the two keys, either one set -> both true);
    ``rail_fallback_m`` is the rail position the twin assumes per arm while the
    track is not homed (its register is meaningless then) - the tile caption
    says so (``rail not homed - twin assumes X m``); the ``home_rail`` sweep and
    a hardware session's frozen (unselected) arm use the same fallback.
    ``principal_offset_px`` is a per-camera OVERLAY-ONLY principal-point nudge
    ``{camera_id: [du, dv]}`` added to that camera's rendered principal point
    (``cx += du``, ``cy += dv``): it aligns a wrist camera whose physical mount
    differs slightly from the shared ``xarm7_on_rail.xml`` ``wrist_cam`` pose
    (solved on the Manipulation Arm), WITHOUT touching the true factory
    ``CameraConfig.intrinsics`` that get baked into recordings. The residual is
    a UNIFORM, pose-/depth-independent image shift (measured 2026-09-06: the
    Perception Arm needed ``[21, 13]`` px, ~2 cm at the arm, while the
    Manipulation Arm needed 0) so a principal-point offset cancels it exactly at
    every posture; see 03-sim §4.3 "per-arm wrist camera overlay offset".
    """

    enabled: bool = True
    fps: float = Field(default=12.0, gt=0.0)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)  # robot tint opacity
    tint_rgb: RGB = (255, 235, 140)  # pale yellow, shaded by the twin's own luminance
    edge_rgb: RGB = (255, 220, 60)  # 1 px robot outline
    env_outline: bool = True  # table / obstacle edges as thin lines (alignment cue)
    env_rgb: RGB = (90, 200, 250)
    stale_tint_rgb: RGB = (170, 170, 170)  # monitor stale/error -> grey
    joint1_offset_rad: float = 0.0  # diagnostic knob; identity is verified
    rail_flip: bool = False  # DEPRECATED alias of hardware_session.rail_flip (kept one release)
    rail_fallback_m: dict[str, float] = Field(
        default_factory=lambda: {"grip": 0.65, "view": 0.0}
    )  # used while the track is not homed
    principal_offset_px: dict[str, tuple[float, float]] = Field(default_factory=dict)
    stream_suffix: str = "_align"


class HardwareSessionConfig(BaseModel):
    """Real-cell session defaults and rail conventions (phase-09c/09d; 04-runtime §5).

    ``default_speed_scale`` is what the Hardware tab pre-selects (first live
    runs at 10 %; the session body still decides). A hardware session always
    includes EVERY configured arm (phase-09d: ``SessionSpec.arms`` must equal
    the workcell's arms, else 409), so there is no arm pre-selection any more
    (the phase-09c ``default_arms`` key is gone; an old key in a YAML is
    ignored). ``rail_flip`` maps the track register onto the twin's rail slot
    as ``q_sim = 0.65 - q_track`` for BOTH the alignment overlay and the gate /
    ``home_rail`` sweep twins (one convention; verify with the overlay right
    after the first homing). ``home_rail_inflation_m`` / ``home_rail_step_m``
    tune the full-travel twin sweep that gates the operator-triggered
    ``home_rail`` maintenance op (D4: the guardrail's debug margin, 5 mm steps
    -> 131 checks) and the position-agnostic path check of its planned
    pre-positioning motion (phase-09d); the gate twin itself keeps
    ``safety.geom_inflation_m``. ``bringup_timeout_s`` bounds
    ``HardwareWorkcell.bring_up`` inside ``POST /api/session`` and inside the
    rail-homing job's connect.
    """

    # ARMING SWITCH (2026-09-05, after a test process reached a real control box): the
    # real xArm drivers are only ever CONNECTED (enable, servo stream, rail homing)
    # when this is true. The repo config keeps it false, so any Runtime built from it
    # (tests, dev instances, a forgotten render) refuses hardware sessions and
    # home_rail with 409 "hardware not armed". The lab render sets it true.
    armed: bool = False
    default_speed_scale: float = Field(default=0.1, gt=0.0, le=1.0)  # D2: 10 % first
    rail_flip: bool = False  # q_sim = 0.65 - q_track (overlay + gate + sweep twins)
    home_rail_inflation_m: float = Field(default=0.025, gt=0.0)  # D4 sweep margin (m)
    home_rail_step_m: float = Field(default=0.005, gt=0.0)  # D4 sweep step (m)
    bringup_timeout_s: float = Field(default=60.0, gt=0.0)  # HardwareWorkcell.bring_up budget


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

    # The reference map (13-tracker §1.1; the tests and design docs pin it). The
    # LAB config (configs/mavis_v2.yaml, 2026-09-07) departs from it: gripper on
    # the two plain buttons, arm switching on the pad - see the comment there.
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


class TrackerCalibrationConfig(BaseModel):
    """Tracker calibration wizard tuning (13-tracker §4 "Calibration modes";
    phase-10). Base-station validation acceptance comes from the 2026-09-03
    measurements (multi-spot calibration: std <= 0.1 mm, max step 0.1 mm;
    single-spot: std ~60 mm, steps up to 248 mm): **std < 5 mm and max step
    < 20 mm**. The yaw gesture checks reject legs shorter than
    ``yaw_min_leg_m`` and fits worse than ``yaw_max_residual_deg``."""

    min_scenes: int = 6  # GSS scenes (max over stations) before validation is allowed
    validation_seconds: float = 10.0  # stationary sample window measured
    validation_skip_seconds: float = 3.0  # convergence window dropped first
    validation_std_mm: float = 5.0  # per-axis position std threshold
    validation_step_mm: float = 20.0  # adjacent-sample max step threshold
    still_window_s: float = 0.5  # controller_still: sample window
    still_threshold_mm: float = 3.0  # controller_still: position std below this
    yaw_min_leg_m: float = 0.10  # each horizontal gesture leg at least this long
    yaw_max_residual_deg: float = 15.0  # mean leg misalignment after the fit
    yaw_capture_average_s: float = 0.3  # a click averages the raw positions of this window


class TrackerConfig(BaseModel):
    """Vive-tracker teleop device + defaults (13-tracker §4).

    ``yaw_deg`` / ``pos_scale`` / ``follow_rotation`` are the process-lifetime
    defaults; the ``tracker_settings`` action mutates the live values.
    ``controller_map`` / ``trackpad_deadzone`` bind the paired controller's
    buttons to device-held key codes (clutch, gripper, rail) and to the
    discrete arm-switch actions (13-tracker §1.1). ``libsurvive_config_path``
    is the file libsurvive reads/writes its lighthouse calibration from; the
    base-station wizard (``calibration``) never points libsurvive at it
    directly — it works on a temporary copy and replaces it only on install.
    """

    backend: Literal["none", "fake", "libsurvive"] = "none"
    object_name: str = "WM0"  # libsurvive codename of the dongle-paired tracker
    libsurvive_args: list[str] = Field(default_factory=lambda: ["--lighthousecount", "2"])
    libsurvive_config_path: Path = Path("${APOLLO_HOME}/var/libsurvive/config.json")
    yaw_deg: float = 0.0  # lighthouse world -> MJCF world (both z-up; yaw only)
    pos_scale: float = Field(default=1.0, ge=0.1, le=3.0)
    follow_rotation: bool = True
    stale_s: float = 0.2  # sample older than this -> hold
    max_jump_m: float = 0.10  # consecutive-sample jump above this -> invalid sample
    controller_map: ControllerMapConfig = ControllerMapConfig()
    trackpad_deadzone: float = Field(default=0.3, ge=0.0, le=1.0)  # |x|,|y| <= dz: click ignored
    filter: TrackerFilterConfig = TrackerFilterConfig()
    calibration: TrackerCalibrationConfig = TrackerCalibrationConfig()

    def model_post_init(self, __context) -> None:
        object.__setattr__(
            self, "libsurvive_config_path", _resolve_path(self.libsurvive_config_path)
        )


class LoggingConfig(BaseModel):
    """Process logging (2026-09-07; 04-runtime §14 "Logging").

    Until this existed the runtime had ONE ``basicConfig`` to stderr and the dev
    launcher appended that stream to an unrotated file (40 MB after two days, a
    quarter of it uvicorn access lines for three UI poll endpoints) with no
    level knob, so the first live teleop defects ("the arm froze", "it stops
    late") could not be read off a log. Now the runtime owns a
    :class:`logging.handlers.RotatingFileHandler` in ``dir`` (self-contained:
    ``${APOLLO_HOME}/var/logs`` by default, gitignored with the rest of ``var``;
    the ops render pins ``$DATA_ROOT/logs``), keeps stderr for journald / the
    launcher's raw capture, and the control loop writes one INFO health line per
    ``control.health_log_every_s`` (tick rate, overruns, clutch / tracker / gate
    state, IK slip counters, servo-stream stats) plus edge lines for every
    transition that matters to teleop (clutch, tracker fresh/stale, watchdog
    latch; gate block edges were already the supervisor's ``collision event``).
    ``level`` applies to both handlers; ``DEBUG`` adds per-event IK slips and
    driver events. ``access_log`` re-enables uvicorn's per-request lines.
    ``dir: null`` = stderr only (tests, embedding apps).
    """

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    dir: Path | None = Path("${APOLLO_HOME}/var/logs")
    file: str = "runtime.log"
    max_bytes: int = Field(default=20 * 1024 * 1024, gt=0)
    backup_count: int = Field(default=10, ge=0)
    access_log: bool = False


class RuntimeConfig(BaseModel):
    """Top-level runtime config; sane defaults for sim-only dev."""

    host: str = "127.0.0.1"
    port: int = 8765  # 8000 is commonly taken on dev boxes (gohttpserver on the lab machine)
    ui_dist: Path | None = None  # built SPA; None = API-only (Vite dev)
    workcells: dict[str, WorkcellConfig] = Field(default_factory=dict)  # "hardware"|"sim"
    profiles_dir: Path = Path("${APOLLO_HOME}/var/profiles")
    datasets_root: Path = Path("${APOLLO_HOME}/var/datasets")
    checkpoints_root: Path = Path("${APOLLO_HOME}/var/checkpoints")
    # Tracker calibration artefacts (phase-10): tracker_calibration.json (persisted
    # yaw / install state), temporary + installed libsurvive config copies.
    calibration_dir: Path = Path("${APOLLO_HOME}/var/calibration")
    control: ControlConfig = ControlConfig()
    recorder: RecorderConfig = RecorderConfig()
    dagger: DaggerConfig = DaggerConfig()
    tracker: TrackerConfig = TrackerConfig()
    microphone: MicrophoneConfig = MicrophoneConfig()  # phase-11 (04-runtime §14)
    hardware_probe: HardwareProbeConfig = HardwareProbeConfig()  # phase-11
    hardware_monitor: HardwareMonitorConfig = HardwareMonitorConfig()  # phase-09a
    twin_overlay: TwinOverlayConfig = TwinOverlayConfig()  # phase-09a
    hardware_session: HardwareSessionConfig = HardwareSessionConfig()  # phase-09c
    telemetry_hz: float = 25.0
    video: VideoConfig = VideoConfig()
    egl_device_id: int = 0
    logging: LoggingConfig = LoggingConfig()  # 2026-09-07

    def model_post_init(self, __context) -> None:
        for name in ("profiles_dir", "datasets_root", "checkpoints_root", "calibration_dir"):
            object.__setattr__(self, name, _resolve_path(getattr(self, name)))
        if self.ui_dist is not None:
            object.__setattr__(self, "ui_dist", _resolve_path(self.ui_dist))
        if self.logging.dir is not None:
            lg = self.logging.model_copy(update={"dir": _resolve_path(self.logging.dir)})
            object.__setattr__(self, "logging", lg)
        # phase-09c: ``twin_overlay.rail_flip`` is the deprecated alias of
        # ``hardware_session.rail_flip``; either key set -> one convention for the
        # overlay, the gate twin and the home_rail sweep.
        flip = bool(self.hardware_session.rail_flip or self.twin_overlay.rail_flip)
        if flip != self.hardware_session.rail_flip:
            hs = self.hardware_session.model_copy(update={"rail_flip": flip})
            object.__setattr__(self, "hardware_session", hs)
        if flip != self.twin_overlay.rail_flip:
            ov = self.twin_overlay.model_copy(update={"rail_flip": flip})
            object.__setattr__(self, "twin_overlay", ov)

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
    # Seed ${APOLLO_HOME} from the config file's own workspace so a config that uses
    # ${APOLLO_HOME}/var/... resolves without the launcher, and child processes
    # (dagger trainer, libsurvive) inherit the same anchor. An explicit env wins.
    if "APOLLO_HOME" not in os.environ:
        root = _find_workspace_root(p.parent)
        if root is not None:
            os.environ["APOLLO_HOME"] = str(root)
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
    "TrackerCalibrationConfig",
    "TrackerConfig",
    "VideoConfig",
    "MicrophoneConfig",
    "HardwareProbeConfig",
    "HardwareMonitorConfig",
    "TwinOverlayConfig",
    "HardwareSessionConfig",
    "LoggingConfig",
    "RuntimeConfig",
    "load_runtime_config",
]
