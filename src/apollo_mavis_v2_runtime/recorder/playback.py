"""Read a saved episode back as a trajectory (2026-09-10; 04-runtime §10.8).

Operator request: a **Playback** button on every episode row of the Welcome page's
Datasets panel, opening a dialog with two actions — "return to this episode's initial
state" and "play back the whole episode" — the second disabled until the first has run
(05-ui §8.1 item 7).

Three replay SOURCES (2026-09-11; the dialog's ``source`` control):

* ``state`` — the MEASURED joint / rail / gripper trajectory (``observation.state``),
  resampled onto the loop tick and streamed through the plan executor
  (``ControlLoop._op_playback_path``). The default, and the only source verified
  posture by posture in the twin before anything moves.
* ``delta_ee`` — the recorded ``action`` column (per-frame TCP deltas of the COMMANDED
  pose, 10-frames §3.2) integrated by the executor path the policy uses
  (``dagger/step.policy_step`` -> ``ActionAnchor.apply_delta``), anchored on the last
  gated command and leashed to the measured pose.
* ``abs_ee`` — the ``action.abs_ee`` column (the commanded TCP at frame k+1, rotation
  as the first two columns of R) handed to the same path verbatim
  (``ActionAnchor.apply_absolute``, deadline interpolation toward each row).

The two action sources exist to measure the EXECUTOR: a recorded episode is the one
input whose intended path is known, so the terminal residual against the last measured
frame says how faithfully the delta / absolute paths follow a command stream. The old
note here — "integrating delta_ee would drift, so replay the state" — described the
executor that re-anchored to the MEASURED pose every tick; that executor is gone
(policy_runner.ActionAnchor, 2026-09-11) and the drift is now the thing to measure.

Nothing here imports lerobot or torch — one ``pyarrow`` read of ``frames.parquet``, plus
``episode.json`` and the dataset ``manifest.json`` for the column layout. The per-dim
NAMES come from the dataset's own manifest (``features["observation.state"]["names"]``,
``features["action"]["names"]``, ``features["action.abs_ee"]["names"]``, built by
``recorder/features``), never from the current session's arm set: an episode recorded
with one arm, or before a track was fitted, has to read back correctly on today's cell
or be refused with a reason, never mis-sliced in silence. An action column that is
absent, unnamed or malformed simply does not appear in ``EpisodeTrajectory.sources``
(the reason is kept per source) - the state replay never depends on it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from apollo_mavis_v2_core.protocol import EpisodePlaybackArm

from .features import arm_action_names
from .manifest import EPISODE_JSON, EPISODES_DIR, FRAMES_PARQUET, MANIFEST_FILENAME, read_json

logger = logging.getLogger(__name__)

#: Suffixes of the ``observation.state`` dim names this module needs (10-frames §6.1).
JOINT_SUFFIXES = tuple(f"joint{i}.pos" for i in range(1, 8))
GRIPPER_SUFFIX = "gripper.pos"
RAIL_SUFFIX = "rail.pos"
#: The measured TCP pose dims of ``observation.state`` (recording frame; 10-frames §6.1).
TCP_SUFFIXES = ("ee.x", "ee.y", "ee.z", "ee.qw", "ee.qx", "ee.qy", "ee.qz")

#: Replay source -> the parquet column / manifest feature it reads (10-frames §3.1).
ACTION_COLUMNS: dict[str, str] = {"delta_ee": "action", "abs_ee": "action.abs_ee"}
#: How the operator is told to get a missing ``action.abs_ee`` column.
BACKFILL_HINT = (
    "backfill it with `python -m apollo_mavis_v2_runtime.tools.backfill_abs_ee <repo_id>`"
)


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
    #: The measured TCP pose (x, y, z, qw, qx, qy, qz) in the recording frame, when the
    #: recording carries all seven (2026-09-11: the action replay's arrival yardstick).
    tcp: tuple[int, ...] | None = None


def parse_state_names(names: list[str]) -> list[ArmColumns]:
    """Group ``["grip_joint1.pos", …, "view_rail.pos", …]`` into per-arm column maps.

    Arm order follows first appearance, which is the recording order the manifest was
    written in (Manipulation Arm first for this cell). A name that does not end in a
    suffix we care about — the measured TCP pose dims — is ignored.
    """
    joints: dict[str, dict[str, int]] = {}
    gripper: dict[str, int] = {}
    rail: dict[str, int] = {}
    tcp: dict[str, dict[str, int]] = {}
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
            else:
                for suffix in TCP_SUFFIXES:
                    if name.endswith(f"_{suffix}"):
                        arm_id = name[: -len(suffix) - 1]
                        tcp.setdefault(arm_id, {})[suffix] = i  # attached to a JOINTED arm only
                        break

    out: list[ArmColumns] = []
    for arm_id in order:
        got = joints.get(arm_id, {})
        missing = [s for s in JOINT_SUFFIXES if s not in got]
        if missing:
            raise PlaybackError(
                f"episode's observation.state has no {missing[0]} column for arm {arm_id!r}"
            )
        pose = tcp.get(arm_id, {})
        out.append(
            ArmColumns(
                arm_id=arm_id,
                joints=tuple(got[s] for s in JOINT_SUFFIXES),
                gripper=gripper.get(arm_id),
                rail=rail.get(arm_id),
                tcp=tuple(pose[s] for s in TCP_SUFFIXES) if len(pose) == 7 else None,
            )
        )
    if not out:
        raise PlaybackError("episode's observation.state names no arm joints")
    return out


@dataclass(frozen=True)
class ActionBlock:
    """Where one arm's block sits inside a flat action row (``names[start:stop]`` ==
    ``arm_action_names(arm_id, has_rail, action_space)``)."""

    arm_id: str
    start: int
    stop: int
    has_rail: bool

    @property
    def width(self) -> int:
        return self.stop - self.start


def parse_action_names(names: list[str], action_space: str) -> list[ActionBlock]:
    """Group a manifest's action dim names into whole per-arm blocks.

    The grammar is ``recorder/features.arm_action_names``: ``<arm>_<dim>`` with the dims
    of ``action_space`` in order, the rail dim last and only for a railed arm. Arm order
    follows first appearance. Anything else - a dim out of order, a block that is not
    exactly one of the two layouts, a name without a known suffix - is a refusal: an
    action row is a command, so it is never sliced by guesswork.
    """
    if action_space not in ACTION_COLUMNS:
        raise PlaybackError(f"unknown action source {action_space!r}")
    dims = [n[1:] for n in arm_action_names("", True, action_space)]  # "_ee.dx" -> "ee.dx"
    runs: list[tuple[str, int, int]] = []  # (arm_id, start, stop)
    for i, name in enumerate(names):
        arm_id = None
        for dim in dims:
            if name.endswith(f"_{dim}") and len(name) > len(dim) + 1:
                arm_id = name[: -len(dim) - 1]
                break
        if arm_id is None:
            raise PlaybackError(
                f"action column name {name!r} is not a {action_space} dim of any arm"
            )
        if runs and runs[-1][0] == arm_id and runs[-1][2] == i:
            runs[-1] = (arm_id, runs[-1][1], i + 1)
        else:
            runs.append((arm_id, i, i + 1))
    out: list[ActionBlock] = []
    seen: set[str] = set()
    for arm_id, start, stop in runs:
        if arm_id in seen:
            raise PlaybackError(f"action column names arm {arm_id!r} in two separate blocks")
        seen.add(arm_id)
        got = names[start:stop]
        has_rail = None
        for rail in (True, False):
            if got == arm_action_names(arm_id, rail, action_space):
                has_rail = rail
                break
        if has_rail is None:
            raise PlaybackError(
                f"action block of arm {arm_id!r} is not the {action_space} layout: {got}"
            )
        out.append(ActionBlock(arm_id=arm_id, start=start, stop=stop, has_rail=has_rail))
    if not out:
        raise PlaybackError("action column names no arm")
    return out


@dataclass
class ActionColumn:
    """One recorded action column, ready for :class:`~..dagger.replay_source.
    ReplayActionSource`: ``rows`` is ``frames x D`` float32 in the DATASET's arm order,
    ``blocks`` says where each arm's block lies, ``frames_map`` the recording frame per
    arm (``features[...]["info"]["frames"]``; ``arm_base:<arm>`` for every dataset the
    cell records)."""

    source: str  # "delta_ee" | "abs_ee"
    column: str  # parquet column name
    names: list[str]
    blocks: list[ActionBlock]
    rows: np.ndarray
    frames_map: dict[str, str] = field(default_factory=dict)

    @property
    def arm_ids(self) -> list[str]:
        return [b.arm_id for b in self.blocks]

    def block(self, arm_id: str) -> ActionBlock | None:
        return next((b for b in self.blocks if b.arm_id == arm_id), None)


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
    #: Recorded action columns by replay source (``delta_ee`` -> ``action``, ``abs_ee``
    #: -> ``action.abs_ee``); a source that is absent or unreadable is not here.
    actions: dict[str, ActionColumn] = field(default_factory=dict)
    #: Why a source is NOT offered (``"episode has no action.abs_ee column - ..."``).
    action_problems: dict[str, str] = field(default_factory=dict)
    #: ``features["action"]["info"]["action_space"]`` of the recorded action column
    #: (``delta_ee`` for every dataset the cell records); None without one.
    action_space: str | None = None

    @property
    def frames(self) -> int:
        return len(self.rows)

    # -- replay sources (2026-09-11) --------------------------------------------------
    @property
    def sources(self) -> list[str]:
        """The replay sources this episode offers: ``state`` always, then ``delta_ee``
        / ``abs_ee`` when their column is present and well-formed."""
        return ["state"] + [s for s in ACTION_COLUMNS if s in self.actions]

    def action_column(self, source: str) -> ActionColumn:
        """The column behind ``source``; :class:`PlaybackError` (operator-facing, naming
        the backfill for ``abs_ee``) when the episode does not offer it."""
        col = self.actions.get(source)
        if col is not None:
            return col
        if source == "state" or source not in ACTION_COLUMNS:
            raise PlaybackError(f"{source!r} is not an action replay source")
        why = self.action_problems.get(source)
        if why is None:
            why = f"episode {self.episode_id!r} has no {ACTION_COLUMNS[source]} column"
            if source == "abs_ee":
                why += f" - {BACKFILL_HINT}"
        raise PlaybackError(why)

    def action_names(self, source: str) -> list[str]:
        """The flat dim names of ``source``'s rows (dataset arm order)."""
        return list(self.action_column(source).names)

    def action_rows(self, source: str) -> np.ndarray:
        """``frames x D`` float32; row k = the action recorded at frame k (a delta from
        the commanded pose at k to the one at k+1 / the commanded pose at k+1)."""
        return self.action_column(source).rows

    def action_block(self, source: str, arm_id: str, k: int) -> np.ndarray:
        """Arm ``arm_id``'s block of row ``k`` (a view into :meth:`action_rows`)."""
        col = self.action_column(source)
        b = col.block(arm_id)
        if b is None:
            raise PlaybackError(f"the {col.column} column names no arm {arm_id!r}")
        return col.rows[k, b.start : b.stop]

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

    def tcp_at(self, index: int, arm_id: str):
        """The MEASURED TCP pose recorded at frame ``index`` for ``arm_id`` (a core
        ``Pose`` in the recording frame), or None when the recording has no pose dims."""
        from apollo_mavis_v2_core import Pose, se3

        col = next((c for c in self.columns if c.arm_id == arm_id), None)
        if col is None or col.tcp is None:
            return None
        row = self.rows[index]
        vals = [float(row[i]) for i in col.tcp]
        return Pose(np.asarray(vals[:3]), se3.quat_normalize(np.asarray(vals[3:7])))


def _episode_dir(ds_root: Path, episode_id: str) -> Path:
    directory = ds_root / EPISODES_DIR / episode_id
    if not (directory / EPISODE_JSON).exists():
        raise PlaybackError(f"unknown episode {episode_id!r}", not_found=True)
    return directory


def _feature(ds_root: Path, directory: Path, key: str) -> dict[str, Any] | None:
    """The manifest's feature block for ``key`` (with ``names``), or None.

    Falls back to the episode's own ``features`` block if a manifest ever lacks one, so a
    directory copied out of its dataset still reads.
    """
    for source in (ds_root / MANIFEST_FILENAME, directory / EPISODE_JSON):
        blob = read_json(source) or {}
        feature = (blob.get("features") or {}).get(key) or {}
        names = feature.get("names")
        if isinstance(names, list) and names:
            return feature
    return None


def _state_names(ds_root: Path, directory: Path) -> list[str]:
    """The ``observation.state`` dim names, from the dataset manifest."""
    feature = _feature(ds_root, directory, "observation.state")
    if feature is not None:
        return [str(n) for n in feature["names"]]
    raise PlaybackError(
        "dataset manifest has no observation.state column names - it was not written by "
        "this runtime, so playback cannot tell the arms' columns apart"
    )


def _read_action_column(
    ds_root: Path, directory: Path, table, source: str, frames: int
) -> ActionColumn:
    """Parse one action column out of the already-read ``table``; :class:`PlaybackError`
    (the text kept as the source's ``action_problems`` entry) when it cannot be offered."""
    column = ACTION_COLUMNS[source]
    if column not in table.column_names:
        why = f"episode has no {column} column"
        if source == "abs_ee":
            why += f" - {BACKFILL_HINT}"
        raise PlaybackError(why)
    feature = _feature(ds_root, directory, column)
    if feature is None:
        raise PlaybackError(f"dataset manifest has no {column} column names")
    info = feature.get("info") or {}
    space = str(info.get("action_space") or source)
    if space != source:
        raise PlaybackError(f"the {column} column is recorded as {space!r}, not {source!r}")
    names = [str(n) for n in feature["names"]]
    blocks = parse_action_names(names, source)
    try:
        rows = np.asarray(table.column(column).to_pylist(), dtype=np.float32)
    except Exception as e:  # noqa: BLE001 - a malformed column is a per-source refusal
        raise PlaybackError(f"the {column} column is unreadable: {e}") from None
    if rows.ndim != 2 or rows.shape != (frames, len(names)):
        raise PlaybackError(
            f"the {column} column has shape {tuple(rows.shape)} but the manifest names "
            f"{len(names)} dims for {frames} frames"
        )
    frames_map = info.get("frames") if isinstance(info.get("frames"), dict) else {}
    return ActionColumn(
        source=source,
        column=column,
        names=names,
        blocks=blocks,
        rows=rows,
        frames_map={str(k): str(v) for k, v in (frames_map or {}).items()},
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

        present = set(pq.read_schema(parquet).names)
        wanted = ["observation.state"] + [c for c in ACTION_COLUMNS.values() if c in present]
        table = pq.read_table(parquet, columns=wanted)
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
    # The action columns are OPTIONAL: a problem with one is remembered per source and
    # surfaces only when that source is asked for; the state replay never depends on it.
    actions: dict[str, ActionColumn] = {}
    problems: dict[str, str] = {}
    for source in ACTION_COLUMNS:
        try:
            actions[source] = _read_action_column(ds_root, directory, table, source, len(rows))
        except PlaybackError as e:
            problems[source] = str(e)
            if ACTION_COLUMNS[source] in table.column_names:
                logger.warning("episode %s: %s source unavailable: %s", episode_id, source, e)
    action_feature = _feature(ds_root, directory, "action")
    action_space = None
    if action_feature is not None:
        action_space = str((action_feature.get("info") or {}).get("action_space") or "delta_ee")
    return EpisodeTrajectory(
        repo_id=repo_id,
        episode_id=episode_id,
        directory=directory,
        fps=fps,
        rows=rows,
        columns=columns,
        actions=actions,
        action_problems=problems,
        action_space=action_space,
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
    "ACTION_COLUMNS",
    "ActionBlock",
    "ActionColumn",
    "ArmColumns",
    "BACKFILL_HINT",
    "EpisodeTrajectory",
    "ExecutorCaps",
    "PlaybackError",
    "ResampledPlayback",
    "load_trajectory",
    "parse_action_names",
    "parse_state_names",
    "resample",
    "JOINT_SUFFIXES",
    "TCP_SUFFIXES",
]
