"""Read a saved episode back as a trajectory (2026-09-10; 04-runtime §10.8).

Operator request: a **Playback** button on every episode row of the Welcome page's
Datasets panel, opening a dialog with two actions — "return to this episode's initial
state" and "play back the whole episode" — the second disabled until the first has run
(05-ui §8.1 item 7).

What is replayed is ``observation.state``, i.e. what the arms MEASURED, not ``action``.
The action column is ``delta_ee`` by default (10-frames §6): a stream of TCP deltas that
only means anything against the exact state it was produced from, so integrating it
would drift. The measured joint trajectory is the ground truth of where the arms
actually went, and it is directly commandable.

Nothing here imports lerobot or torch — one ``pyarrow`` read of ``frames.parquet``, plus
``episode.json`` and the dataset ``manifest.json`` for the column layout. The per-dim
NAMES come from the dataset's own manifest (``features["observation.state"]["names"]``,
built by ``recorder/features.arm_state_names``), never from the current session's arm
set: an episode recorded with one arm, or before a track was fitted, has to read back
correctly on today's cell or be refused with a reason, never mis-sliced in silence.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apollo_mavis_v2_core.protocol import EpisodePlaybackArm

from .manifest import EPISODE_JSON, EPISODES_DIR, FRAMES_PARQUET, MANIFEST_FILENAME, read_json

logger = logging.getLogger(__name__)

#: Suffixes of the ``observation.state`` dim names this module needs (10-frames §6.1).
JOINT_SUFFIXES = tuple(f"joint{i}.pos" for i in range(1, 8))
GRIPPER_SUFFIX = "gripper.pos"
RAIL_SUFFIX = "rail.pos"


class PlaybackError(Exception):
    """Why this episode cannot be replayed; the text is operator-facing."""

    def __init__(self, detail: str, not_found: bool = False) -> None:
        super().__init__(detail)
        self.not_found = not_found


@dataclass(frozen=True)
class ArmColumns:
    """Where one arm's values sit inside a flat ``observation.state`` row."""

    arm_id: str
    joints: tuple[int, ...]  # exactly 7, in joint order
    gripper: int | None
    rail: int | None


def parse_state_names(names: list[str]) -> list[ArmColumns]:
    """Group ``["grip_joint1.pos", …, "view_rail.pos", …]`` into per-arm column maps.

    Arm order follows first appearance, which is the recording order the manifest was
    written in (Manipulation Arm first for this cell). A name that does not end in a
    suffix we care about — the measured TCP pose dims — is ignored.
    """
    joints: dict[str, dict[str, int]] = {}
    gripper: dict[str, int] = {}
    rail: dict[str, int] = {}
    order: list[str] = []

    def note(arm_id: str) -> None:
        if arm_id not in order:
            order.append(arm_id)

    for i, name in enumerate(names):
        for suffix in JOINT_SUFFIXES:
            if name.endswith(f"_{suffix}"):
                arm_id = name[: -len(suffix) - 1]
                joints.setdefault(arm_id, {})[suffix] = i
                note(arm_id)
                break
        else:
            if name.endswith(f"_{GRIPPER_SUFFIX}"):
                arm_id = name[: -len(GRIPPER_SUFFIX) - 1]
                gripper[arm_id] = i
                note(arm_id)
            elif name.endswith(f"_{RAIL_SUFFIX}"):
                arm_id = name[: -len(RAIL_SUFFIX) - 1]
                rail[arm_id] = i
                note(arm_id)

    out: list[ArmColumns] = []
    for arm_id in order:
        got = joints.get(arm_id, {})
        missing = [s for s in JOINT_SUFFIXES if s not in got]
        if missing:
            raise PlaybackError(
                f"episode's observation.state has no {missing[0]} column for arm {arm_id!r}"
            )
        out.append(
            ArmColumns(
                arm_id=arm_id,
                joints=tuple(got[s] for s in JOINT_SUFFIXES),
                gripper=gripper.get(arm_id),
                rail=rail.get(arm_id),
            )
        )
    if not out:
        raise PlaybackError("episode's observation.state names no arm joints")
    return out


@dataclass
class EpisodeTrajectory:
    """One saved episode, ready to command.

    ``rows`` is the raw ``observation.state`` matrix (``frames`` x ``dim``) and
    ``columns`` says where each arm's joints / gripper / rail live in a row. Both are
    kept rather than a pre-sliced per-arm dict so a caller can walk frames without
    rebuilding anything per tick.
    """

    repo_id: str
    episode_id: str
    directory: Path
    fps: float
    rows: list[list[float]]
    columns: list[ArmColumns]

    @property
    def frames(self) -> int:
        return len(self.rows)

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps if self.fps > 0 else 0.0

    @property
    def arm_ids(self) -> list[str]:
        return [c.arm_id for c in self.columns]

    def state_at(self, index: int) -> dict[str, EpisodePlaybackArm]:
        """Per-arm joints / rail / gripper at frame ``index``."""
        row = self.rows[index]
        return {
            c.arm_id: EpisodePlaybackArm(
                arm_id=c.arm_id,
                q=[float(row[i]) for i in c.joints],
                rail_pos_m=None if c.rail is None else float(row[c.rail]),
                gripper_open_frac=None if c.gripper is None else float(row[c.gripper]),
            )
            for c in self.columns
        }

    def initial_state(self) -> dict[str, EpisodePlaybackArm]:
        """Frame 0 — "this episode's initial state", the goto target."""
        return self.state_at(0)


def _episode_dir(ds_root: Path, episode_id: str) -> Path:
    directory = ds_root / EPISODES_DIR / episode_id
    if not (directory / EPISODE_JSON).exists():
        raise PlaybackError(f"unknown episode {episode_id!r}", not_found=True)
    return directory


def _state_names(ds_root: Path, directory: Path) -> list[str]:
    """The ``observation.state`` dim names, from the dataset manifest.

    Falls back to the episode's own ``features`` block if a manifest ever lacks one, so a
    directory copied out of its dataset still reads.
    """
    for source in (ds_root / MANIFEST_FILENAME, directory / EPISODE_JSON):
        blob = read_json(source) or {}
        feature = (blob.get("features") or {}).get("observation.state") or {}
        names = feature.get("names")
        if isinstance(names, list) and names:
            return [str(n) for n in names]
    raise PlaybackError(
        "dataset manifest has no observation.state column names - it was not written by "
        "this runtime, so playback cannot tell the arms' columns apart"
    )


def load_trajectory(repo_id: str, ds_root: Path, episode_id: str) -> EpisodeTrajectory:
    """Read one episode directory into an :class:`EpisodeTrajectory`.

    Raises :class:`PlaybackError` with an operator-facing reason for anything the caller
    should turn into a 404 / 409 rather than a 500: an unknown episode, a missing or
    unreadable parquet, a manifest without column names, an empty episode.
    """
    directory = _episode_dir(ds_root, episode_id)
    parquet = directory / FRAMES_PARQUET
    if not parquet.exists():
        raise PlaybackError(f"episode {episode_id!r} has no {FRAMES_PARQUET}")
    names = _state_names(ds_root, directory)
    columns = parse_state_names(names)
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(parquet, columns=["observation.state"])
    except Exception as e:  # noqa: BLE001 - a corrupt file is a refusal, never a 500
        raise PlaybackError(
            f"episode {episode_id!r}: {FRAMES_PARQUET} is unreadable: {e}"
        ) from None
    rows = [[float(v) for v in row] for row in table.column("observation.state").to_pylist()]
    if not rows:
        raise PlaybackError(f"episode {episode_id!r} has no frames")
    width = len(names)
    bad = next((i for i, r in enumerate(rows) if len(r) != width), None)
    if bad is not None:
        raise PlaybackError(
            f"episode {episode_id!r}: observation.state row {bad} has {len(rows[bad])} values "
            f"but the manifest names {width}"
        )
    meta: dict[str, Any] = read_json(directory / EPISODE_JSON) or {}
    fps = float(meta.get("fps") or 0) or 1.0
    return EpisodeTrajectory(
        repo_id=repo_id,
        episode_id=episode_id,
        directory=directory,
        fps=fps,
        rows=rows,
        columns=columns,
    )


@dataclass(frozen=True)
class ExecutorCaps:
    """The per-tick bounds :class:`~apollo_mavis_v2_runtime.control.joint_panel.PlanExecutor`
    applies, read off the session's ``JogConfig`` (the hardware bring-up lowers them to the
    driver's own ``ServoLimits``, so this is the REAL cell's rate, not the YAML default)."""

    slew_rad_per_tick: float
    rail_m_per_tick: float
    #: Hardware only: ``sum|dq_j| * lever_j`` per tick, the lever-weighted Cartesian step the
    #: driver's servo streamer enforces. None on sim.
    cart_step_m: float | None = None
    lever_arm_m: tuple[float, ...] | None = None

    @classmethod
    def from_jog(cls, jog) -> ExecutorCaps:
        lever = getattr(jog, "plan_lever_arm_m", None)
        cart = getattr(jog, "plan_cart_step_m", None)
        usable = cart is not None and lever is not None and float(cart) > 0.0
        return cls(
            slew_rad_per_tick=float(jog.slew_rad_per_tick),
            rail_m_per_tick=float(jog.rail_m_per_tick),
            cart_step_m=float(cart) if usable else None,
            lever_arm_m=tuple(float(v) for v in lever) if usable else None,
        )

    def ticks_for(self, a: list[float], b: list[float]) -> int:
        """How many executor ticks the straight segment ``a -> b`` costs (>= 1).

        Mirrors ``PlanExecutor.step``'s ``ratio`` exactly - joint slew, rail slew and the
        hardware Cartesian bound - because that is what decides whether the executor walks
        a waypoint in one tick or silently subdivides it. Playback depends on ONE tick per
        waypoint (see :func:`resample`), so this is the function that has to agree.
        """
        ratio = 0.0
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            limit = self.slew_rad_per_tick if i < 7 else self.rail_m_per_tick
            if limit > 0.0:
                ratio = max(ratio, abs(y - x) / limit)
        if self.cart_step_m is not None and self.lever_arm_m is not None:
            n = min(7, len(a), len(self.lever_arm_m))
            cart = sum(abs(b[i] - a[i]) * self.lever_arm_m[i] for i in range(n))
            ratio = max(ratio, cart / self.cart_step_m)
        return max(1, math.ceil(ratio - 1e-9))


@dataclass
class ResampledPlayback:
    """A recorded episode resampled onto the control loop's tick.

    ``waypoints[arm]`` all have the SAME length and each consecutive pair is within every
    executor cap, so the executor retires exactly one waypoint per tick for every arm and
    the arms stay in LOCKSTEP. ``gripper[arm][k]`` is the gripper opening recorded at
    waypoint ``k``.
    """

    waypoints: dict[str, list[list[float]]]
    gripper: dict[str, list[float]]
    fps: float
    #: Recorded frames per emitted waypoint, i.e. how much slower than real time this
    #: replay runs. 1.0 = real time.
    slowdown: float

    @property
    def length(self) -> int:
        return len(next(iter(self.waypoints.values()))) if self.waypoints else 0

    def postures(self):
        """Yield ``(index, {arm: q_full})`` for every waypoint — what the twin verifies."""
        for k in range(self.length):
            yield k, {arm: wps[k] for arm, wps in self.waypoints.items()}


def resample(
    traj: EpisodeTrajectory,
    caps: ExecutorCaps,
    *,
    loop_hz: float,
    arms: list[str] | None = None,
    rail_hold: dict[str, float] | None = None,
) -> ResampledPlayback:
    """Resample a recorded episode onto the loop tick, in lockstep across the arms.

    Two properties this has to preserve, and one it deliberately gives up:

    * **Relative timing between the arms.** The episode was recorded with both arms
      moving TOGETHER, and the twin only ever saw those simultaneous combinations. So one
      global time scale is applied to every arm - never a per-arm one - and the segment
      count is the MAX over all arms. Replaying such a path one arm at a time (the rule
      for twin-PLANNED motions, 2026-09-08) would be a different path through space that
      nothing has ever validated.
    * **Real time where the caps allow it.** ``loop_hz / fps`` ticks per recorded frame
      (4 at 100 Hz and 25 fps) is the real-time rate, and a frame whose motion needs more
      ticks than that gets them - so the replay is faithful where it can be and uniformly
      SLOWER where the recording was quicker than this session's caps (e.g. recorded at
      100 % speed, replayed at 10 %). It never runs faster than recorded.
    * **Absolute duration** is given up when the caps bind: ``slowdown`` says by how much.

    ``arms`` restricts the output to the session's arms; ``rail_hold`` supplies the rail
    slot for an arm whose recording has no rail column (its carriage is held there).
    """
    ids = [a for a in traj.arm_ids if arms is None or a in arms]
    if not ids:
        raise PlaybackError("the episode names none of this session's arms")
    frames = [traj.state_at(k) for k in range(traj.frames)]

    def full(arm_id: str, k: int) -> list[float]:
        st = frames[k][arm_id]
        rail = st.rail_pos_m
        if rail is None:
            rail = (rail_hold or {}).get(arm_id)
        return [*st.q] if rail is None else [*st.q, float(rail)]

    per_frame = max(1, round(loop_hz / traj.fps)) if traj.fps > 0 else 1
    waypoints: dict[str, list[list[float]]] = {a: [full(a, 0)] for a in ids}
    gripper: dict[str, list[float]] = {
        a: [_grip(frames[0][a])] for a in ids if frames[0][a].gripper_open_frac is not None
    }
    for k in range(traj.frames - 1):
        # ONE segment count for every arm: the max of the real-time rate and what any
        # arm's own motion needs under the caps.
        steps = per_frame
        for a in ids:
            steps = max(steps, caps.ticks_for(full(a, k), full(a, k + 1)))
        for step in range(1, steps + 1):
            t = step / steps
            for a in ids:
                start, end = full(a, k), full(a, k + 1)
                waypoints[a].append([s + (e - s) * t for s, e in zip(start, end, strict=True)])
                if a in gripper:
                    g0, g1 = _grip(frames[k][a]), _grip(frames[k + 1][a])
                    gripper[a].append(g0 + (g1 - g0) * t)
    emitted = len(waypoints[ids[0]])
    ideal = 1 + (traj.frames - 1) * per_frame
    return ResampledPlayback(
        waypoints=waypoints,
        gripper=gripper,
        fps=traj.fps,
        slowdown=emitted / ideal if ideal else 1.0,
    )


def _grip(state) -> float:
    frac = state.gripper_open_frac
    return 1.0 if frac is None else min(1.0, max(0.0, float(frac)))


__all__ = [
    "ArmColumns",
    "EpisodeTrajectory",
    "ExecutorCaps",
    "PlaybackError",
    "ResampledPlayback",
    "load_trajectory",
    "parse_state_names",
    "resample",
    "JOINT_SUFFIXES",
]
