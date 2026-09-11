"""``python -m apollo_mavis_v2_runtime.tools.backfill_abs_ee <dataset dir> [--scene mavis_v2]
[--dry-run] [--force]`` — add the ``action.abs_ee`` column (10-frames §6, 2026-09-11) to
an episode-directory dataset recorded before the recorder wrote it, in place.

``<dataset dir>`` is the directory holding ``manifest.json`` + ``episodes/`` (a
``bc_demo/<name>`` root or an Online DAgger ``<session>/rollouts`` dir; the DAgger
extra columns are preserved untouched). Per episode:

* backups first, OUTSIDE the episode directory (its file set is pinned):
  ``<dataset>/backups/<UTC stamp>/<episode_id>/{frames.parquet, episode.json}`` and
  ``<dataset>/backups/<stamp>/manifest.json``;
* FK with the twin (``RecorderKinematics(REGISTRY.build(scene))``, ``MUJOCO_GL=egl``);
  the scene comes from the episode's session sidecar (``spec.sim_scene`` /
  ``spec.digital_twin_scene``), else ``--scene``;
* ``observation.state`` ``ee.*`` is RECOMPUTED for every row as
  ``convert_pose(arm, tcp_base(q_meas), base_world(q_meas))`` — for hardware episodes
  this replaces the SDK flange pose (and its wrong-composition quaternion) with the
  twin ``link_tcp`` pose the live recorder's deltas are built from; for sim episodes
  it is a no-op (max change logged, expected < 1e-5 m);
* ``action.abs_ee[k]`` integrates the stored deltas from the recomputed pose of row 0
  (the arm is at rest at the first kept frame, so ``cmd[0] == meas[0]``):
  ``p_{k+1} = p_k + dp_F[k]``, ``q_{k+1} = rotvec_to_quat(dr_F[k]) (x) q_k`` (10-frames
  §3.2 left composition, deltas already in the recording frame); row k gets
  ``(p_{k+1}, r6(q_{k+1}))``, the delta column's gripper dim, and ``rail_meas[0] +
  cumsum(rail.dpos)[..k]``. The terminal residual ``|p_N - p_meas[N-1]|`` is logged
  per arm and flagged above 20 mm;
* ``frames.parquet`` is rewritten like ``EpisodeDirRecorder._table`` (FixedSizeList
  float32, snappy, ONE row group, features first, extra columns preserved) via a temp
  file + ``os.replace``; ``episode.json`` gets recomputed ``observation.state`` stats,
  ``stats["action.abs_ee"]`` and the marker ``backfill.abs_ee`` (version 1);
* idempotent: an episode whose marker is version 1 and whose parquet has the column
  is skipped unless ``--force``.

After EVERY episode succeeded the manifest's ``features`` gain ``action.abs_ee`` (from
``build_features`` with the dataset's arms / frames) under ``MANIFEST_LOCK`` and the
export is marked stale — last, so a crashed run leaves ``dataset_incompatibility`` and
the export consistent. ``exports/``, ``trainer_spool/``, video and audio are never
touched. ``--dry-run`` computes and prints everything and writes nothing. Exit status
is non-zero on any failure. Run it with no session open on the dataset.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from apollo_mavis_v2_core import Pose, se3

from . import stats as _stats
from .features import (
    ABS_EE_KEY,
    ArmMeta,
    arm_action_names,
    arm_state_names,
    build_features,
)
from .frames import RecordingFrameConverter
from .manifest import (
    EPISODE_JSON,
    EPISODES_DIR,
    FRAMES_PARQUET,
    MANIFEST_FILENAME,
    MANIFEST_LOCK,
    SESSIONS_DIR,
    TMP_PREFIX,
    episode_dirs,
    mark_export_stale,
    read_json,
    read_manifest,
    utc_now_iso,
    write_json_atomic,
    write_manifest,
)

logger = logging.getLogger(__name__)

TOOL_NAME = "apollo_mavis_v2_runtime.tools.backfill_abs_ee"
BACKFILL_VERSION = 1
BACKUPS_DIR = "backups"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
RESIDUAL_FLAG_M = 0.020  # terminal residual above this is flagged (sample data: ~1-4 mm)
SIM_STATE_CHANGE_FLAG_M = 1e-4  # a sim episode's recomputed ee should not move


class BackfillError(RuntimeError):
    """A dataset-level refusal (no manifest, wrong layout, an open episode)."""


KinFactory = Callable[[str], Any]  # scene id -> object with base_world / tcp_base


# -- layout ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetLayout:
    """What the manifest says about the arms; validated against the recorded names."""

    arms: list[ArmMeta]
    frames: dict[str, str]
    robot_type: str
    abs_feature: dict[str, Any]  # the build_features spec for action.abs_ee

    @property
    def sim(self) -> bool:
        return self.robot_type.endswith("_mujoco")


def dataset_layout(manifest: dict[str, Any]) -> DatasetLayout:
    feats = manifest.get("features") or {}
    action = feats.get(ACTION_KEY)
    state = feats.get(STATE_KEY)
    if not action or not state:
        raise BackfillError("manifest declares no 'action' / 'observation.state' feature")
    info = action.get("info") or {}
    if info.get("action_space") != "delta_ee":
        raise BackfillError(
            f"the dataset's primary action_space is {info.get('action_space')!r}; the backfill "
            "integrates delta_ee actions only"
        )
    frames = info.get("frames") or {}
    rail_arms = set((info.get("rail") or {}).get("arms") or [])
    if not frames:
        raise BackfillError("manifest features.action.info.frames is empty")
    arms = [ArmMeta(str(a), a in rail_arms) for a in frames]
    expected = build_features(arms, frames, {}, "delta_ee")
    if list(action.get("names") or []) != expected[ACTION_KEY]["names"]:
        raise BackfillError(
            f"action names {action.get('names')!r} do not match the "
            f"{[a.arm_id for a in arms]} / rail {sorted(rail_arms)} layout"
        )
    if list(state.get("names") or []) != expected[STATE_KEY]["names"]:
        raise BackfillError("observation.state names do not match the manifest's arm layout")
    return DatasetLayout(arms, {k: str(v) for k, v in frames.items()},
                         str(manifest.get("robot_type") or ""), expected[ABS_EE_KEY])


def _block_offsets(arms: list[ArmMeta]) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Per-arm start offsets into observation.state / action / action.abs_ee."""
    s_off: dict[str, int] = {}
    a_off: dict[str, int] = {}
    b_off: dict[str, int] = {}
    s = a = b = 0
    for arm in arms:
        s_off[arm.arm_id], a_off[arm.arm_id], b_off[arm.arm_id] = s, a, b
        s += len(arm_state_names(arm.arm_id, arm.has_rail))
        a += len(arm_action_names(arm.arm_id, arm.has_rail, "delta_ee"))
        b += len(arm_action_names(arm.arm_id, arm.has_rail, "abs_ee"))
    return s_off, a_off, b_off


# -- the per-episode computation (pure) ----------------------------------------------------------
@dataclass
class EpisodeColumns:
    state: np.ndarray  # (N, Ds) float32, ee.* recomputed
    abs_ee: np.ndarray  # (N, Dabs) float32
    terminal_residual_m: dict[str, float]
    state_ee_max_change_m: dict[str, float]


def compute_columns(
    state: np.ndarray,
    action: np.ndarray,
    arms: list[ArmMeta],
    converter: RecordingFrameConverter,
    kin: Any,
) -> EpisodeColumns:
    """Recompute ``observation.state`` ee.* from FK and integrate the deltas into
    ``action.abs_ee`` (see the module docstring). Float64 throughout, cast once."""
    state = np.asarray(state)
    action = np.asarray(action)
    n = int(state.shape[0])
    if n == 0:
        raise ValueError("empty episode")
    if action.shape[0] != n:
        raise ValueError(f"action rows {action.shape[0]} != state rows {n}")
    s_off, a_off, b_off = _block_offsets(arms)
    n_abs = sum(len(arm_action_names(a.arm_id, a.has_rail, "abs_ee")) for a in arms)
    new_state = np.array(state, dtype=np.float64, copy=True)
    abs_ee = np.zeros((n, n_abs), dtype=np.float64)
    residual: dict[str, float] = {}
    max_change: dict[str, float] = {}
    for arm in arms:
        so, ao, bo = s_off[arm.arm_id], a_off[arm.arm_id], b_off[arm.arm_id]
        ee_o = so + (9 if arm.has_rail else 8)  # joints 7, gripper, [rail], then ee.*
        # (c) measured TCP in the recording frame, every row
        ee_meas = np.zeros((n, 7), dtype=np.float64)
        for k in range(n):
            joints = np.asarray(state[k, so : so + 7], dtype=np.float64)
            q = np.concatenate([joints, [float(state[k, so + 8])]]) if arm.has_rail else joints
            pose_f = converter.convert_pose(
                arm.arm_id, kin.tcp_base(arm.arm_id, q), kin.base_world(arm.arm_id, q)
            )
            ee_meas[k, :3] = pose_f.position
            ee_meas[k, 3:] = se3.quat_normalize(pose_f.orientation)  # canonical w >= 0
        old_ee = np.asarray(state[:, ee_o : ee_o + 7], dtype=np.float64)
        max_change[arm.arm_id] = float(
            np.max(np.linalg.norm(ee_meas[:, :3] - old_ee[:, :3], axis=1))
        )
        new_state[:, ee_o : ee_o + 7] = ee_meas
        # (d) integrate the deltas from the recomputed pose of row 0
        dp = np.asarray(action[:, ao : ao + 3], dtype=np.float64)
        dr = np.asarray(action[:, ao + 3 : ao + 6], dtype=np.float64)
        pos = ee_meas[0, :3] + np.cumsum(dp, axis=0)  # pos[k] = p_{k+1}
        q = ee_meas[0, 3:].copy()
        for k in range(n):
            q = se3.quat_mul(se3.rotvec_to_quat(dr[k]), q)
            q = se3.quat_normalize(q)
            abs_ee[k, bo : bo + 3] = pos[k]
            abs_ee[k, bo + 3 : bo + 9] = se3.quat_to_rot6d(q)
        abs_ee[:, bo + 9] = action[:, ao + 6]  # the delta column's gripper dim, verbatim
        if arm.has_rail:
            rail0 = float(state[0, so + 8])
            abs_ee[:, bo + 10] = rail0 + np.cumsum(np.asarray(action[:, ao + 7], dtype=np.float64))
        residual[arm.arm_id] = float(np.linalg.norm(pos[n - 1] - ee_meas[n - 1, :3]))
    return EpisodeColumns(
        state=new_state.astype(np.float32),
        abs_ee=abs_ee.astype(np.float32),
        terminal_residual_m=residual,
        state_ee_max_change_m=max_change,
    )


# -- parquet helpers -----------------------------------------------------------------------------
# pyarrow is sanctioned in the recorder package; this tool rewrites the recorder's own
# frames.parquet (10-frames §11.4) and is a CLI entry point the runtime never imports.
def _matrix(table, key: str, width: int) -> np.ndarray:
    import pyarrow as pa

    if key not in table.column_names:
        raise ValueError(f"frames.parquet lacks the {key!r} column")
    col = table.column(key).combine_chunks()
    if not (pa.types.is_fixed_size_list(col.type) or pa.types.is_list(col.type)):
        raise ValueError(f"column {key!r} is {col.type}, not a list column")
    flat = np.asarray(col.flatten().to_numpy(zero_copy_only=False))
    if len(col) and flat.shape[0] % len(col):
        raise ValueError(f"column {key!r}: ragged rows")
    arr = flat.reshape(len(col), -1) if len(col) else flat.reshape(0, width)
    if arr.shape[1] != width:
        raise ValueError(f"column {key!r} has {arr.shape[1]} dims, the manifest says {width}")
    return arr


def _fixed_list(arr: np.ndarray):
    import pyarrow as pa

    arr = np.ascontiguousarray(arr, dtype=np.float32)
    flat = pa.array(arr.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, int(arr.shape[1]))


def _rewritten_table(table, cols: EpisodeColumns):
    """Original columns in their order, ``observation.state`` replaced, ``action.abs_ee``
    inserted right after ``action`` (or replaced in place under --force)."""
    import pyarrow as pa

    names = list(table.column_names)
    arrays = {name: table.column(name) for name in names}
    arrays[STATE_KEY] = _fixed_list(cols.state)
    arrays[ABS_EE_KEY] = _fixed_list(cols.abs_ee)
    if ABS_EE_KEY not in names:
        names.insert(names.index(ACTION_KEY) + 1, ABS_EE_KEY)
    return pa.table({name: arrays[name] for name in names})


def _write_parquet_atomic(path: Path, table) -> None:
    import pyarrow.parquet as pq

    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    try:
        pq.write_table(table, tmp, compression="snappy", row_group_size=max(1, table.num_rows))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# -- scene / kinematics ----------------------------------------------------------------------------
def default_kin_factory(scene_id: str) -> Any:
    """``RecorderKinematics(REGISTRY.build(scene_id))`` — the FK the live recorder uses."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    from apollo_mavis_v2_sim import REGISTRY  # [sim] extra

    from .kinematics import RecorderKinematics

    return RecorderKinematics(REGISTRY.build(scene_id))


def _session_scene(root: Path, episode: dict[str, Any]) -> str | None:
    sid = episode.get("session_id")
    if not sid:
        return None
    doc = read_json(root / SESSIONS_DIR / f"session_{sid}.json")
    if not isinstance(doc, dict):
        return None
    spec = doc.get("spec") or {}
    return spec.get("sim_scene") or spec.get("digital_twin_scene") or None


def _camera_poses(episode: dict[str, Any]) -> dict[str, Pose]:
    out: dict[str, Pose] = {}
    for cam, block in (episode.get("extrinsics") or {}).items():
        pose = (block or {}).get("T_W_C") if isinstance(block, dict) else None
        if isinstance(pose, dict) and pose.get("position") and pose.get("orientation_wxyz"):
            out[str(cam)] = Pose(
                np.asarray(pose["position"], dtype=np.float64),
                np.asarray(pose["orientation_wxyz"], dtype=np.float64),
            )
    return out


# -- driver ----------------------------------------------------------------------------------------
@dataclass
class EpisodeReport:
    episode_id: str
    status: str  # backfilled | would-backfill | skipped | failed
    rows: int = 0
    scene: str | None = None
    terminal_residual_m: dict[str, float] = field(default_factory=dict)
    state_ee_max_change_m: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class BackfillResult:
    root: Path
    reports: list[EpisodeReport]
    manifest_patched: bool
    backup_dir: Path | None
    dry_run: bool

    @property
    def failures(self) -> list[EpisodeReport]:
        return [r for r in self.reports if r.status == "failed"]

    @property
    def ok(self) -> bool:
        return not self.failures


class _Backups:
    """Lazily created ``<dataset>/backups/<stamp>/`` (nothing on disk until a write)."""

    def __init__(self, root: Path, enabled: bool) -> None:
        self.root = Path(root)
        self.enabled = enabled
        self.dir: Path | None = None

    def _ensure(self) -> Path:
        if self.dir is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            base = self.root / BACKUPS_DIR
            cand = base / stamp
            k = 1
            while cand.exists():
                cand = base / f"{stamp}-{k}"
                k += 1
            cand.mkdir(parents=True)
            self.dir = cand
        return self.dir

    def episode(self, ep_dir: Path) -> None:
        if not self.enabled:
            return
        dst = self._ensure() / ep_dir.name
        dst.mkdir(parents=True, exist_ok=True)
        for name in (FRAMES_PARQUET, EPISODE_JSON):
            shutil.copy2(ep_dir / name, dst / name)

    def manifest(self, root: Path) -> None:
        if not self.enabled:
            return
        shutil.copy2(root / MANIFEST_FILENAME, self._ensure() / MANIFEST_FILENAME)


def _already_done(episode: dict[str, Any], column_names: list[str]) -> bool:
    marker = ((episode.get("backfill") or {}).get("abs_ee") or {})
    return int(marker.get("version", 0) or 0) == BACKFILL_VERSION and ABS_EE_KEY in column_names


def _process_episode(
    root: Path,
    ep_dir: Path,
    layout: DatasetLayout,
    *,
    scene_default: str,
    kins: dict[str, Any],
    kin_factory: KinFactory,
    backups: _Backups,
    dry_run: bool,
    force: bool,
) -> EpisodeReport:
    import pyarrow.parquet as pq

    report = EpisodeReport(ep_dir.name, "failed")
    episode = read_json(ep_dir / EPISODE_JSON)
    if not isinstance(episode, dict):
        report.error = "episode.json unreadable"
        return report
    parquet = ep_dir / FRAMES_PARQUET
    if not parquet.exists():
        report.error = "frames.parquet missing"
        return report
    table = pq.read_table(parquet)
    report.rows = int(table.num_rows)
    if not force and _already_done(episode, table.column_names):
        report.status = "skipped"
        return report
    scene_id = _session_scene(root, episode) or scene_default
    report.scene = scene_id
    if scene_id not in kins:
        kins[scene_id] = kin_factory(scene_id)
    kin = kins[scene_id]
    converter = RecordingFrameConverter(layout.frames, _camera_poses(episode))
    n_state = sum(len(arm_state_names(a.arm_id, a.has_rail)) for a in layout.arms)
    n_action = sum(len(arm_action_names(a.arm_id, a.has_rail, "delta_ee")) for a in layout.arms)
    state = _matrix(table, STATE_KEY, n_state)
    action = _matrix(table, ACTION_KEY, n_action)
    cols = compute_columns(state, action, layout.arms, converter, kin)
    report.terminal_residual_m = dict(cols.terminal_residual_m)
    report.state_ee_max_change_m = dict(cols.state_ee_max_change_m)
    for arm, res in cols.terminal_residual_m.items():
        if res > RESIDUAL_FLAG_M:
            report.flags.append(f"{arm}: terminal residual {res * 1e3:.1f} mm > 20 mm")
    if layout.sim:
        for arm, ch in cols.state_ee_max_change_m.items():
            if ch > SIM_STATE_CHANGE_FLAG_M:
                report.flags.append(f"{arm}: sim ee.* moved {ch * 1e3:.2f} mm on recompute")
    if dry_run:
        report.status = "would-backfill"
        return report
    # (a) backups OUTSIDE the episode directory, then (e) parquet, then (f) episode.json
    backups.episode(ep_dir)
    _write_parquet_atomic(parquet, _rewritten_table(table, cols))
    stats = dict(episode.get("stats") or {})
    fresh = _stats.to_jsonable(
        {STATE_KEY: _stats.feature_stats(cols.state), ABS_EE_KEY: _stats.feature_stats(cols.abs_ee)}
    )
    stats.update(fresh)
    episode["stats"] = stats
    backfill = dict(episode.get("backfill") or {})
    backfill["abs_ee"] = {
        "version": BACKFILL_VERSION,
        "at": utc_now_iso(),
        "tool": TOOL_NAME,
        "state_ee_recomputed": True,
        "fk_scene": scene_id,
        "terminal_residual_m": {k: float(v) for k, v in cols.terminal_residual_m.items()},
    }
    episode["backfill"] = backfill
    write_json_atomic(ep_dir / EPISODE_JSON, episode)
    report.status = "backfilled"
    return report


def _patch_manifest(root: Path, layout: DatasetLayout, backups: _Backups, *, force: bool) -> bool:
    """(g) add ``action.abs_ee`` to the manifest's features after ``action``; True if
    the manifest changed."""
    spec = dict(layout.abs_feature)
    if isinstance(spec.get("shape"), tuple):
        spec["shape"] = list(spec["shape"])
    with MANIFEST_LOCK:
        manifest = read_manifest(root)
        if manifest is None:
            raise BackfillError("manifest.json vanished during the run")
        feats: dict[str, Any] = dict(manifest.get("features") or {})
        if ABS_EE_KEY in feats and feats[ABS_EE_KEY] == spec and not force:
            return False
        patched: dict[str, Any] = {}
        for key, value in feats.items():
            if key == ABS_EE_KEY:
                continue
            patched[key] = value
            if key == ACTION_KEY:
                patched[ABS_EE_KEY] = spec
        if ABS_EE_KEY not in patched:  # no 'action' key ordering to hang on: append
            patched[ABS_EE_KEY] = spec
        backups.manifest(root)
        manifest["features"] = patched
        mark_export_stale(manifest)
        write_manifest(root, manifest)
        return True


def backfill_dataset(
    root: Path,
    *,
    scene: str = "mavis_v2",
    dry_run: bool = False,
    force: bool = False,
    kin_factory: KinFactory | None = None,
) -> BackfillResult:
    """Backfill every episode under ``<root>/episodes`` (see the module docstring).
    ``kin_factory(scene_id)`` builds the FK object (default: the twin)."""
    root = Path(root)
    manifest = read_manifest(root)
    if manifest is None:
        raise BackfillError(f"{root} holds no manifest.json (not an episode-directory dataset)")
    episodes = root / EPISODES_DIR
    open_eps = sorted(episodes.glob(f"{TMP_PREFIX}*")) if episodes.is_dir() else []
    if open_eps:
        raise BackfillError(
            f"{root} has an episode being recorded ({open_eps[0].name}); end the session first"
        )
    layout = dataset_layout(manifest)
    factory = kin_factory or default_kin_factory
    kins: dict[str, Any] = {}
    backups = _Backups(root, enabled=not dry_run)
    reports: list[EpisodeReport] = []
    for ep_dir in episode_dirs(root):
        try:
            rep = _process_episode(
                root, ep_dir, layout, scene_default=scene, kins=kins, kin_factory=factory,
                backups=backups, dry_run=dry_run, force=force,
            )
        except Exception as exc:  # noqa: BLE001 - one bad episode must not stop the report
            logger.exception("episode %s: backfill failed", ep_dir.name)
            rep = EpisodeReport(ep_dir.name, "failed", error=f"{type(exc).__name__}: {exc}")
        reports.append(rep)
    patched = False
    if all(r.status != "failed" for r in reports) and not dry_run:
        patched = _patch_manifest(root, layout, backups, force=force)
    return BackfillResult(root, reports, patched, backups.dir, dry_run)


# -- reporting / CLI -------------------------------------------------------------------------------
def _fmt_mm(values: dict[str, float]) -> str:
    return " ".join(f"{k}={v * 1e3:.1f}" for k, v in values.items()) or "-"


def format_report(result: BackfillResult) -> str:
    lines = [
        f"dataset {result.root}" + ("  (dry run: nothing written)" if result.dry_run else ""),
        f"{'episode':<32} {'rows':>6} {'status':<15} {'residual_mm':<24} "
        f"{'state_ee_change_mm':<24} flags",
    ]
    for r in result.reports:
        flags = "; ".join(r.flags) if r.flags else ""
        if r.error:
            flags = (flags + "; " if flags else "") + r.error
        lines.append(
            f"{r.episode_id:<32} {r.rows:>6} {r.status:<15} "
            f"{_fmt_mm(r.terminal_residual_m):<24} {_fmt_mm(r.state_ee_max_change_m):<24} {flags}"
        )
    counts: dict[str, int] = {}
    for r in result.reports:
        counts[r.status] = counts.get(r.status, 0) + 1
    summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "no episodes"
    lines.append(f"summary: {len(result.reports)} episodes ({summary})")
    flagged = sum(1 for r in result.reports if r.flags)
    if flagged:
        lines.append(f"flagged: {flagged} episode(s) - inspect before training on them")
    if result.dry_run:
        lines.append("manifest: unchanged (dry run)")
    elif result.failures:
        lines.append(
            "manifest: NOT patched (fix the failures and re-run; done episodes are skipped)"
        )
    else:
        lines.append(
            "manifest: action.abs_ee feature added, export marked stale"
            if result.manifest_patched else "manifest: already declares action.abs_ee"
        )
    if result.backup_dir is not None:
        lines.append(f"backups: {result.backup_dir}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=f"python -m {TOOL_NAME}",
        description="Add the action.abs_ee column to an episode-directory dataset in place.",
    )
    parser.add_argument("dataset", type=Path, help="directory holding manifest.json + episodes/")
    parser.add_argument(
        "--scene", default="mavis_v2",
        help="twin scene for FK when the session sidecar names none (default mavis_v2)",
    )
    parser.add_argument("--dry-run", action="store_true", help="compute and report, write nothing")
    parser.add_argument(
        "--force", action="store_true", help="redo episodes already carrying the version-1 marker"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )
    try:
        result = backfill_dataset(
            args.dataset, scene=args.scene, dry_run=args.dry_run, force=args.force
        )
    except BackfillError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(format_report(result))
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = [
    "BACKFILL_VERSION",
    "BACKUPS_DIR",
    "BackfillError",
    "BackfillResult",
    "EpisodeColumns",
    "EpisodeReport",
    "backfill_dataset",
    "compute_columns",
    "dataset_layout",
    "default_kin_factory",
    "format_report",
    "main",
]
