"""Dataset manifest + episode-directory layout helpers (10-frames §11; 04-runtime §10).

Torch-free and lerobot-free on purpose: the REST listing / deletion paths
(``DatasetStore``), the recorder and the export job all read the same
``manifest.json`` / ``episodes/<id>/episode.json`` files through this module.

Layout (10-frames §11.2)::

    <root>/manifest.json                  dataset-level, rebuildable (§11.5)
    <root>/sessions/session_<id>.json     §9 sidecar
    <root>/scenes/<sha256[:16]>.xml       §9 sidecar
    <root>/episodes/<episode_id>/         one directory per SAVED episode
    <root>/episodes/.tmp-<episode_id>/    the episode being recorded / a crash
    <root>/exports/lerobot_v3/            DERIVED (export_lerobot.py)
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# One process-wide lock for every manifest read-modify-write (the recorder's save,
# a REST delete's refresh, the export job's last_export record): three threads
# touch the same file and a lost update showed up as a manifest counting an episode
# a mid-session delete had removed. Re-entrant so a transaction can call the
# helpers that take it themselves.
MANIFEST_LOCK = threading.RLock()


@contextmanager
def manifest_lock():
    with MANIFEST_LOCK:
        yield

APOLLO_DATASET_LAYOUT = 1
MANIFEST_FILENAME = "manifest.json"
EPISODES_DIR = "episodes"
SESSIONS_DIR = "sessions"
SCENES_DIR = "scenes"
EXPORTS_DIR = "exports"
TMP_PREFIX = ".tmp-"
EPISODE_JSON = "episode.json"
FRAMES_PARQUET = "frames.parquet"
VIDEO_DIR = "video"
AUDIO_WAV = "audio.wav"
IMAGE_PREFIX = "observation.images."
LEGACY_INFO = Path("meta") / "info.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def ns_to_iso(wallclock_ns: int) -> str:
    return (
        datetime.fromtimestamp(int(wallclock_ns) / 1e9, tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def mint_episode_id(now: datetime | None = None) -> str:
    """``%Y%m%dT%H%M%S.fffZ-<6 hex>`` (10-frames §11.3): sorts in capture order,
    unique across hosts and clock corrections, never reused."""
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%S.%f")[:-3] + "Z-" + secrets.token_hex(3)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """``mkstemp`` in the target directory + ``os.replace``: a UNIQUE temp name per
    writer, so the recorder thread and a REST handler never collide on one
    ``<name>.tmp`` (the fixed-name variant lost writes)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# -- manifest -----------------------------------------------------------------------------
def manifest_path(root: Path) -> Path:
    return Path(root) / MANIFEST_FILENAME


def read_manifest(root: Path) -> dict[str, Any] | None:
    """The dataset manifest, or None when ``root`` is not an episode-directory
    dataset (a legacy v3 tree or nothing). Unknown layout majors are refused."""
    m = read_json(manifest_path(root))
    if m is None:
        return None
    major = int(m.get("apollo_dataset_layout", 0) or 0)
    if major != APOLLO_DATASET_LAYOUT:
        raise ValueError(
            f"{manifest_path(root)}: apollo_dataset_layout {major} is not the supported "
            f"{APOLLO_DATASET_LAYOUT}"
        )
    return m


def is_legacy_v3(root: Path) -> bool:
    """A phase-07 LeRobot v3 tree (``meta/info.json`` at the root, no manifest)."""
    root = Path(root)
    return (root / LEGACY_INFO).exists() and not manifest_path(root).exists()


def new_manifest(
    repo_id: str,
    fps: int,
    robot_type: str,
    features: dict[str, dict],
    video: dict[str, Any],
) -> dict[str, Any]:
    now = utc_now_iso()
    frames_map = (features.get("action") or {}).get("info", {}).get("frames") or {}
    return {
        "apollo_dataset_layout": APOLLO_DATASET_LAYOUT,
        "repo_id": repo_id,
        "created_at": now,
        "modified_at": now,
        "fps": int(fps),
        "robot_type": robot_type,
        "features": _jsonable_features(features),
        "arms": list(frames_map),
        "cameras": [k[len(IMAGE_PREFIX):] for k in features if k.startswith(IMAGE_PREFIX)],
        "video": dict(video),
        "episodes": 0,
        "frames": 0,
        "last_export": None,
    }


def _jsonable_features(features: dict[str, dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for key, spec in features.items():
        spec = dict(spec)
        if isinstance(spec.get("shape"), tuple):
            spec["shape"] = list(spec["shape"])
        out[key] = spec
    return out


def write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    with MANIFEST_LOCK:
        manifest["modified_at"] = utc_now_iso()
        write_json_atomic(manifest_path(root), manifest)


def episode_dirs(root: Path) -> list[Path]:
    """Saved episode directories in capture order (the id sorts by time)."""
    base = Path(root) / EPISODES_DIR
    if not base.is_dir():
        return []
    return sorted(
        p for p in base.iterdir()
        if p.is_dir() and not p.name.startswith(".") and (p / EPISODE_JSON).exists()
    )


def rebuild_counters(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Recount ``episodes`` / ``frames`` from the directories (10-frames §11.1:
    the manifest is derivable and is rebuilt whenever it could disagree)."""
    dirs = episode_dirs(root)
    frames = 0
    for d in dirs:
        ep = read_json(d / EPISODE_JSON) or {}
        frames += int(ep.get("length", 0) or 0)
    manifest["episodes"] = len(dirs)
    manifest["frames"] = frames
    return manifest


def mark_export_stale(manifest: dict[str, Any]) -> None:
    last = manifest.get("last_export")
    if isinstance(last, dict):
        last["stale"] = True


def refresh_manifest(root: Path) -> dict[str, Any] | None:
    """Rebuild the counters from the directories, mark the export stale when the
    episode count moved, and write. Returns the manifest (None = not a dataset)."""
    with MANIFEST_LOCK:
        manifest = read_manifest(root)
        if manifest is None:
            return None
        before = (int(manifest.get("episodes", 0) or 0), int(manifest.get("frames", 0) or 0))
        rebuild_counters(root, manifest)
        if (manifest["episodes"], manifest["frames"]) != before:
            mark_export_stale(manifest)
        write_manifest(root, manifest)
        return manifest


# -- crash sweep --------------------------------------------------------------------------
def sweep_incomplete_episodes(
    datasets_root: Path, keep: set[Path] | None = None, *, dataset_glob: str = "*/*"
) -> list[Path]:
    """Remove every ``<dataset>/episodes/.tmp-*`` directory under ``datasets_root``
    except the ones in ``keep`` (the running session's open episode), where
    ``dataset_glob`` selects the dataset directories below the root: ``*/*`` for
    the generic ``<root>/<ns>/<name>`` layout, ``*`` / ``*/<subdir>`` for a mapped
    namespace root (15-online-dagger §7, D5; ``DatasetStore.sweep`` walks every root).
    Called at every open — runtime start, ``DatasetStore`` scan (10-frames §11.6
    step 5). Returns the removed paths."""
    root = Path(datasets_root)
    if not root.exists():
        return []
    keep_resolved = {Path(p).resolve() for p in (keep or ())}
    removed: list[Path] = []
    for tmp in sorted(root.glob(f"{dataset_glob}/{EPISODES_DIR}/{TMP_PREFIX}*")):
        if not tmp.is_dir() or tmp.resolve() in keep_resolved:
            continue
        shutil.rmtree(tmp, ignore_errors=True)
        removed.append(tmp)
        logger.warning("swept incomplete episode directory %s", tmp)
    return removed


# -- compatibility --------------------------------------------------------------------------
_INFO_KEYS = ("apollo_schema", "action_space", "frames", "rail")


def _feature_signature(spec: dict) -> tuple:
    """dtype + shape + per-dim names + the convention info blocks (10-frames §8.3)."""
    names = spec.get("names")
    if isinstance(names, dict):
        names = tuple(sorted((k, tuple(v)) for k, v in names.items()))
    elif isinstance(names, list):
        names = tuple(names)
    info = spec.get("info") or {}
    info_sig = tuple(
        (k, json.dumps(info[k], sort_keys=True)) for k in _INFO_KEYS if k in info
    )
    return (str(spec.get("dtype")), tuple(spec.get("shape") or ()), names, info_sig)


def dataset_incompatibility(
    manifest: dict[str, Any] | None, features: dict[str, dict], fps: int, robot_type: str
) -> str | None:
    """Why a session with ``features`` / ``fps`` / ``robot_type`` may NOT record into
    the dataset described by ``manifest`` — None when it may (or when there is no
    dataset yet). Compares fps, robot_type, the feature set and each feature's
    signature INCLUDING the ``apollo_schema`` / ``action_space`` / ``frames`` /
    ``rail`` info blocks (10-frames §11.5)."""
    if manifest is None:
        return None
    stored_fps = int(manifest.get("fps", 0) or 0)
    if stored_fps != int(fps):
        return f"recorded at {stored_fps} fps, this session records at {fps} fps"
    stored_robot = manifest.get("robot_type")
    if stored_robot != robot_type:
        return f"robot_type {stored_robot!r} != {robot_type!r} (sim vs hardware, or arm count)"
    stored = manifest.get("features") or {}
    missing = sorted(set(stored) - set(features))
    extra = sorted(set(features) - set(stored))
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"dataset has {missing} which this session lacks")
        if extra:
            parts.append(f"this session adds {extra}")
        why = "feature set differs: " + "; ".join(parts)
        if not missing and extra == ["action.abs_ee"]:
            # A dataset recorded before 2026-09-11: the ONLY difference is the absolute
            # action column every session now writes - the backfill tool adds it in place.
            why += (
                " - the dataset predates the action.abs_ee column; run "
                "`python -m apollo_mavis_v2_runtime.tools.backfill_abs_ee <dataset dir>` "
                "to add it, then resume"
            )
        return why
    for key, spec in features.items():
        if _feature_signature(stored[key]) != _feature_signature(spec):
            return (
                f"feature {key!r} differs: dataset {_feature_signature(stored[key])!r} vs "
                f"session {_feature_signature(spec)!r}"
            )
    return None


__all__ = [
    "APOLLO_DATASET_LAYOUT",
    "AUDIO_WAV",
    "EPISODES_DIR",
    "EPISODE_JSON",
    "EXPORTS_DIR",
    "FRAMES_PARQUET",
    "IMAGE_PREFIX",
    "LEGACY_INFO",
    "MANIFEST_FILENAME",
    "MANIFEST_LOCK",
    "manifest_lock",
    "SCENES_DIR",
    "SESSIONS_DIR",
    "TMP_PREFIX",
    "VIDEO_DIR",
    "dataset_incompatibility",
    "episode_dirs",
    "is_legacy_v3",
    "manifest_path",
    "mark_export_stale",
    "mint_episode_id",
    "new_manifest",
    "ns_to_iso",
    "read_json",
    "read_manifest",
    "rebuild_counters",
    "refresh_manifest",
    "sweep_incomplete_episodes",
    "utc_now_iso",
    "write_json_atomic",
    "write_manifest",
]
