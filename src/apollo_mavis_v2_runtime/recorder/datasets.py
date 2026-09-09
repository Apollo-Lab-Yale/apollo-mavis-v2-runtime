"""DatasetStore — the recorded datasets under the generic ``datasets_root`` AND the
per-namespace roots (04-runtime §10.6; 10-frames §11; 15-online-dagger §7 / D5).

Repo ids keep the ``<ns>/<name>`` grammar. A namespace with a mapped root
(``RuntimeConfig.datasets.namespaces``) lives at ``<root>/<name>`` or
``<root>/<name>/<subdir>`` (``bc_demo/<name>`` -> ``~/data/bc_demo/<name>``,
``online_dagger/<s>`` -> ``~/data/online_dagger/<s>/rollouts``); every other namespace
lives at ``<datasets_root>/<ns>/<name>``. ``root_of`` is the ONE place that
spells a dataset directory — the recorder build, the export CLI and the REST all
go through it.

Reads ONLY ``manifest.json`` and the per-episode ``episode.json`` sidecars (plus
``meta/info.json`` to recognise a legacy phase-07 LeRobot v3 tree, listed
read-only): no lerobot, no torch, no parquet scan on the REST path. Deletion is
one ``rmtree`` of an episode directory (+ its ``trainer_spool`` row) followed by
a manifest refresh — no other episode is read, decoded or renumbered. The
LeRobot v3 export is a batch job on a ``dataset-export`` thread (one at a time);
its progress rides ``telemetry.datasets.export``.
"""

from __future__ import annotations

import logging
import shutil
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from apollo_mavis_v2_core.protocol import (
    DatasetExportInfo,
    DatasetExportTelemetry,
    DatasetInfo,
    DatasetLayoutInfo,
    DatasetNamespaceInfo,
    EpisodeInfo,
)

from .export_lerobot import EXPORT_FORMAT, ExportProgress, export_lerobot_v3
from .manifest import (
    AUDIO_WAV,
    EPISODE_JSON,
    EPISODES_DIR,
    IMAGE_PREFIX,
    LEGACY_INFO,
    SESSIONS_DIR,
    TMP_PREFIX,
    episode_dirs,
    is_legacy_v3,
    manifest_path,
    read_json,
    read_manifest,
    refresh_manifest,
    sweep_incomplete_episodes,
)

logger = logging.getLogger(__name__)

DEFAULT_NAMESPACE = "apollo"
LEGACY_READ_ONLY = "legacy LeRobot v3 dataset - read-only"


class DatasetError(Exception):
    """Refused dataset op (REST 409 / 404 by ``not_found``)."""

    def __init__(self, detail: str, not_found: bool = False) -> None:
        super().__init__(detail)
        self.not_found = not_found


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class NamespaceRoot:
    """One mapped namespace (15-online-dagger §7, D5): its datasets live at
    ``root/<name>`` or, with ``subdir``, at ``root/<name>/<subdir>``."""

    root: Path
    subdir: str | None = None

    @property
    def dataset_glob(self) -> str:
        """Glob, relative to ``root``, that selects every dataset directory."""
        return "*" if self.subdir is None else f"*/{self.subdir}"

    def dataset_dir(self, name: str) -> Path:
        d = self.root / name
        return d if self.subdir is None else d / self.subdir

    def name_of(self, ds_root: Path) -> str:
        """The dataset name of a directory ``dataset_glob`` matched."""
        return ds_root.name if self.subdir is None else ds_root.parent.name


GENERIC_DATASET_GLOB = "*/*"  # <datasets_root>/<ns>/<name>


def _namespace_roots(namespaces: Mapping[str, Any] | None) -> dict[str, NamespaceRoot]:
    """Accepts ``DatasetNamespaceConfig``-like objects (``.root`` / ``.subdir``),
    :class:`NamespaceRoot`, ``{"root": .., "subdir": ..}`` mappings, ``(root, subdir)``
    pairs and bare paths; the store stays importable without ``config.py``."""
    out: dict[str, NamespaceRoot] = {}
    for ns, spec in (namespaces or {}).items():
        if isinstance(spec, NamespaceRoot):
            out[ns] = spec
        elif isinstance(spec, Mapping):
            out[ns] = NamespaceRoot(Path(spec["root"]).expanduser(), spec.get("subdir"))
        elif isinstance(spec, (str, Path)):
            out[ns] = NamespaceRoot(Path(spec).expanduser(), None)
        elif isinstance(spec, tuple):
            root, subdir = spec
            out[ns] = NamespaceRoot(Path(root).expanduser(), subdir)
        else:
            out[ns] = NamespaceRoot(Path(spec.root).expanduser(), getattr(spec, "subdir", None))
    return out


class DatasetStore:
    """Datasets below the generic ``root`` (``<namespace>/<name>/manifest.json``) plus
    every mapped namespace root (``namespaces``); ``resolve()`` prefixes a bare name
    with ``default_namespace``."""

    def __init__(
        self,
        root: Path,
        *,
        default_namespace: str = DEFAULT_NAMESPACE,
        namespaces: Mapping[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.default_namespace = default_namespace
        self.namespaces: dict[str, NamespaceRoot] = _namespace_roots(namespaces)
        # Set by the SessionManager: the repo the running session records into and
        # the id of the episode being recorded right now (both None when idle).
        self.in_use_repo: Callable[[], str | None] = lambda: None
        self.open_episode: Callable[[], str | None] = lambda: None
        # Why a SAVED episode of ``repo_id`` may not be deleted right now (a 409 reason),
        # else None: the running Online DAgger session's rollouts (its counters, its
        # session.json rows and the trainer's buffer are never told about a deletion).
        self.episode_delete_refusal: Callable[[str], str | None] = lambda repo_id: None
        self._export_lock = threading.Lock()
        self._export_thread: threading.Thread | None = None
        self._export_progress: ExportProgress | None = None
        self._exporting: str | None = None

    # -- naming -----------------------------------------------------------------------
    @property
    def namespace(self) -> str:
        """Alias of ``default_namespace`` (pre-2026-09-08 attribute name)."""
        return self.default_namespace

    def resolve(self, dataset: str) -> str:
        """``"pick_cube"`` -> ``"<default_namespace>/pick_cube"``; a namespaced id
        passes through."""
        return dataset if "/" in dataset else f"{self.default_namespace}/{dataset}"

    @staticmethod
    def split(repo_id: str) -> tuple[str, str]:
        """``"ns/name"`` -> ``("ns", "name")``."""
        ns, _, name = repo_id.partition("/")
        return ns, name

    def root_of(self, repo_id: str) -> Path:
        """The dataset directory: ``<mapped root>/<name>[/<subdir>]`` for a mapped
        namespace, else ``<datasets_root>/<ns>/<name>`` (a bare name resolves first)."""
        repo_id = self.resolve(repo_id)
        ns, name = self.split(repo_id)
        mapped = self.namespaces.get(ns)
        if mapped is not None:
            return mapped.dataset_dir(name)
        return self.root / ns / name

    def layout(self) -> DatasetLayoutInfo:
        """``GET /api/datasets/layout`` (15-online-dagger §7): the roots, for the UI."""
        return DatasetLayoutInfo(
            default_namespace=self.default_namespace,
            generic_root=str(self.root),
            namespaces={
                ns: DatasetNamespaceInfo(root=str(m.root), subdir=m.subdir)
                for ns, m in self.namespaces.items()
            },
        )

    def search_roots(self) -> list[tuple[Path, str]]:
        """``(base, dataset_glob)`` pairs covering every place a dataset may live:
        the generic root (``*/*``) and each mapped root (``*`` / ``*/<subdir>``)."""
        out: list[tuple[Path, str]] = [(self.root, GENERIC_DATASET_GLOB)]
        out.extend((m.root, m.dataset_glob) for m in self.namespaces.values())
        return out

    def layout_of(self, repo_id: str) -> str | None:
        """``episode_dirs`` | ``lerobot_v3`` (legacy) | None (no dataset)."""
        ds_root = self.root_of(repo_id)
        if manifest_path(ds_root).exists():
            return "episode_dirs"
        if is_legacy_v3(ds_root):
            return "lerobot_v3"
        return None

    def exists(self, repo_id: str) -> bool:
        return self.layout_of(repo_id) is not None

    # -- export job state ----------------------------------------------------------------
    @property
    def exporting(self) -> str | None:
        """repo_id of the export running right now, else None."""
        with self._export_lock:
            if self._export_thread is not None and not self._export_thread.is_alive():
                self._export_thread = None
                self._exporting = None
            return self._exporting

    def export_telemetry(self) -> DatasetExportTelemetry | None:
        """The running / last export's progress (``telemetry.datasets.export``)."""
        progress = self._export_progress
        return progress.telemetry() if progress is not None else None

    # -- listing ------------------------------------------------------------------------
    def _dataset_dirs(self) -> list[tuple[str, Path]]:
        """``(repo_id, dataset dir)`` for every manifest / legacy ``meta/info.json``
        under the generic root and the mapped roots, de-duplicated by repo id. A
        generic-root directory of a MAPPED namespace is skipped: ``root_of`` could
        never address it (said once per listing at DEBUG)."""
        found: dict[str, Path] = {}
        places: list[tuple[str | None, Path, str]] = [(None, self.root, GENERIC_DATASET_GLOB)]
        places.extend((ns, m.root, m.dataset_glob) for ns, m in self.namespaces.items())
        for mapped_ns, base, pattern in places:
            if not base.exists():
                continue
            hits = list(base.glob(f"{pattern}/manifest.json")) + list(
                base.glob(f"{pattern}/{LEGACY_INFO.as_posix()}")
            )
            for path in hits:
                ds_root = path.parent if path.name == "manifest.json" else path.parent.parent
                if mapped_ns is None:
                    ns, name = ds_root.parent.name, ds_root.name
                    if ns in self.namespaces:
                        logger.debug(
                            "dataset %s/%s under the generic root %s is shadowed by the "
                            "mapped namespace root %s - not listed",
                            ns, name, self.root, self.namespaces[ns].root,
                        )
                        continue
                else:
                    ns, name = mapped_ns, self.namespaces[mapped_ns].name_of(ds_root)
                found.setdefault(f"{ns}/{name}", ds_root)
        return list(found.items())

    def list(self) -> list[DatasetInfo]:
        """Every dataset (manifest or legacy ``meta/info.json``) under the generic root
        and every mapped root, newest first."""
        out: list[DatasetInfo] = []
        self.sweep()
        for repo_id, _ds_root in self._dataset_dirs():
            try:
                info = self.describe(repo_id)
            except Exception as e:  # noqa: BLE001 - one bad manifest must not 500 the listing
                logger.warning("dataset %s skipped from the listing: %s", repo_id, e)
                continue
            if info is not None:
                out.append(info)
        out.sort(key=lambda d: d.modified_at, reverse=True)
        return out

    def describe(self, repo_id: str) -> DatasetInfo | None:
        layout = self.layout_of(repo_id)
        if layout is None:
            return None
        ds_root = self.root_of(repo_id)
        in_use = self.in_use_repo() == repo_id
        if layout == "lerobot_v3":
            return self._describe_legacy(repo_id, ds_root)
        manifest = read_manifest(ds_root)  # raises ValueError on an unsupported layout major
        if manifest is None:
            raise ValueError(f"{manifest_path(ds_root)} is not valid JSON")
        features = manifest.get("features") or {}
        cameras = list(
            manifest.get("cameras")
            or [k[len(IMAGE_PREFIX):] for k in features if k.startswith(IMAGE_PREFIX)]
        )
        session = self._latest_session_sidecar(ds_root) or {}
        spec = session.get("spec") or {}
        workcell = session.get("workcell") or {}
        kind = workcell.get("kind")
        return DatasetInfo(
            repo_id=repo_id,
            root=str(ds_root),
            namespace=self.split(repo_id)[0],
            path=str(ds_root),
            layout="episode_dirs",
            total_episodes=int(manifest.get("episodes", 0) or 0),
            total_frames=int(manifest.get("frames", 0) or 0),
            fps=int(manifest.get("fps", 0) or 0),
            robot_type=manifest.get("robot_type"),
            kind=kind if kind in ("hardware", "sim") else None,
            task=spec.get("task"),
            cameras=cameras,
            arms=list(manifest.get("arms") or workcell.get("arm_ids") or spec.get("arms") or []),
            modified_at=str(
                manifest.get("modified_at") or _iso(manifest_path(ds_root).stat().st_mtime)
            ),
            in_use=in_use,
            export=self._export_info(repo_id, manifest),
        )

    def _export_info(self, repo_id: str, manifest: dict[str, Any]) -> DatasetExportInfo:
        last = manifest.get("last_export")
        if self.exporting == repo_id:
            state = "running"
        elif not isinstance(last, dict):
            return DatasetExportInfo(state="none")
        elif last.get("error"):
            state = "failed"
        elif last.get("stale"):
            state = "stale"
        else:
            state = "fresh"
        last = last if isinstance(last, dict) else {}
        return DatasetExportInfo(
            state=state,  # type: ignore[arg-type]
            format=str(last.get("format") or EXPORT_FORMAT),
            path=last.get("path"),
            at=last.get("at"),
            episodes=int(last.get("episodes", 0) or 0),
            detail=str(last.get("error") or ""),
        )

    def _describe_legacy(self, repo_id: str, ds_root: Path) -> DatasetInfo | None:
        info = read_json(ds_root / LEGACY_INFO)
        if info is None:
            return None
        features = info.get("features") or {}
        cameras = [k[len(IMAGE_PREFIX):] for k in features if k.startswith(IMAGE_PREFIX)]
        session = self._latest_session_sidecar(ds_root, legacy=True) or {}
        spec = session.get("spec") or {}
        workcell = session.get("workcell") or {}
        kind = workcell.get("kind")
        return DatasetInfo(
            repo_id=repo_id,
            root=str(ds_root),
            namespace=self.split(repo_id)[0],
            path=str(ds_root),
            layout="lerobot_v3",
            total_episodes=int(info.get("total_episodes", 0) or 0),
            total_frames=int(info.get("total_frames", 0) or 0),
            fps=int(info.get("fps", 0) or 0),
            robot_type=info.get("robot_type"),
            kind=kind if kind in ("hardware", "sim") else None,
            task=spec.get("task"),
            cameras=cameras,
            arms=list(workcell.get("arm_ids") or spec.get("arms") or []),
            modified_at=_iso((ds_root / LEGACY_INFO).stat().st_mtime),
            in_use=False,
            export=DatasetExportInfo(
                state="fresh", path=".", episodes=int(info.get("total_episodes", 0) or 0)
            ),
        )

    @staticmethod
    def _latest_session_sidecar(ds_root: Path, legacy: bool = False) -> dict[str, Any] | None:
        base = ds_root / "meta" / "apollo" if legacy else ds_root / SESSIONS_DIR
        if not base.exists():
            return None
        newest: tuple[float, Path] | None = None
        for p in base.glob("session_*.json"):
            m = p.stat().st_mtime
            if newest is None or m > newest[0]:
                newest = (m, p)
        return read_json(newest[1]) if newest else None

    # -- episodes -----------------------------------------------------------------------
    def episodes(self, repo_id: str) -> list[EpisodeInfo]:
        layout = self.layout_of(repo_id)
        if layout is None:
            raise DatasetError(f"unknown dataset {repo_id!r}", not_found=True)
        if layout == "lerobot_v3":
            return []  # legacy trees are listed at dataset level only
        ds_root = self.root_of(repo_id)
        open_id = self.open_episode() if self.in_use_repo() == repo_id else None
        out: list[EpisodeInfo] = []
        for idx, d in enumerate(episode_dirs(ds_root)):
            ep = read_json(d / EPISODE_JSON) or {}
            tasks = ep.get("tasks") or []
            length = int(ep.get("length", 0) or 0)
            fps = float(ep.get("fps") or 0) or 1.0
            audio = ep.get("audio")
            out.append(
                EpisodeInfo(
                    episode_id=d.name,
                    index=idx,
                    frames=length,
                    duration_s=float(ep.get("duration_s") or length / fps),
                    task=str(tasks[0]) if tasks else None,
                    session_id=ep.get("session_id"),
                    recorded_at=ep.get("recorded_at"),
                    frames_dropped=int(ep.get("frames_dropped", 0) or 0),
                    audio=bool(
                        isinstance(audio, dict)
                        and (d / str(audio.get("path") or AUDIO_WAV)).exists()
                    ),
                    export_ok=bool(ep.get("export_ok", True)),
                    export_note=ep.get("export_note"),
                    open=False,
                )
            )
        if open_id is not None:
            out.append(
                EpisodeInfo(
                    episode_id=open_id, index=len(out), frames=0, duration_s=0.0, open=True,
                )
            )
        return out

    # -- deletion -------------------------------------------------------------------------
    def delete_episode(self, repo_id: str, episode_id: str) -> None:
        """``rmtree(episodes/<id>)`` (+ its trainer spool row), refresh the manifest,
        mark the export stale (10-frames §11.7). 404 unknown dataset / id; 409 legacy,
        the open episode, or the running Online DAgger session's rollouts
        (``episode_delete_refusal``)."""
        layout = self.layout_of(repo_id)
        if layout is None:
            raise DatasetError(f"unknown dataset {repo_id!r}", not_found=True)
        if layout == "lerobot_v3":
            raise DatasetError(LEGACY_READ_ONLY)
        if self.in_use_repo() == repo_id and self.open_episode() == episode_id:
            raise DatasetError("episode is being recorded")
        refusal = self.episode_delete_refusal(repo_id)
        if refusal:
            raise DatasetError(refusal)
        if self.exporting == repo_id:
            raise DatasetError(f"dataset {repo_id!r} is being exported - retry in a moment")
        ds_root = self.root_of(repo_id)
        target = ds_root / EPISODES_DIR / episode_id
        if episode_id.startswith(".") or "/" in episode_id or not target.is_dir():
            raise DatasetError(f"no episode {episode_id!r} in {repo_id!r}", not_found=True)
        shutil.rmtree(target)
        spool = ds_root / "trainer_spool" / f"ep_{episode_id}.parquet"
        spool.unlink(missing_ok=True)
        refresh_manifest(ds_root)
        logger.info("dataset %s: deleted episode %s", repo_id, episode_id)

    def delete_dataset(self, repo_id: str) -> None:
        """The whole tree (10-frames §11.7); 409 while a session records into it or
        an export runs; 404 unknown."""
        layout = self.layout_of(repo_id)
        if layout is None:
            raise DatasetError(f"unknown dataset {repo_id!r}", not_found=True)
        if layout == "lerobot_v3":
            raise DatasetError(LEGACY_READ_ONLY)
        if self.in_use_repo() == repo_id:
            raise DatasetError(f"dataset {repo_id!r} is in use by the running session")
        if self.exporting == repo_id:
            raise DatasetError(f"dataset {repo_id!r} is being exported - retry in a moment")
        ds_root = self.root_of(repo_id)
        shutil.rmtree(ds_root)
        if self.split(repo_id)[0] not in self.namespaces:
            self._prune_namespace(ds_root.parent)  # an emptied <datasets_root>/<ns> goes too
        # a mapped root is the operator's folder (and, with a subdir, the Online DAgger
        # session directory next to session.json + the trainer's files): never removed here
        logger.info("dataset %s deleted", repo_id)

    # -- export ---------------------------------------------------------------------------
    def export(
        self,
        repo_id: str,
        fmt: str = EXPORT_FORMAT,
        out: str | Path | None = None,
        *,
        video_file_mb: int = 200,
        data_file_mb: int = 100,
        validate: bool = True,
    ) -> str:
        """Start the export job (10-frames §11.8) on the ``dataset-export`` thread;
        returns the ISO start time. 404 unknown; 409 legacy / in use / another export
        running / an unsupported format."""
        if fmt != EXPORT_FORMAT:
            raise DatasetError(f"unsupported export format {fmt!r} (only {EXPORT_FORMAT})")
        layout = self.layout_of(repo_id)
        if layout is None:
            raise DatasetError(f"unknown dataset {repo_id!r}", not_found=True)
        if layout == "lerobot_v3":
            raise DatasetError(LEGACY_READ_ONLY)
        if self.in_use_repo() == repo_id:
            raise DatasetError(
                f"dataset {repo_id!r} is in use by the running session - end the session first"
            )
        with self._export_lock:
            if self._export_thread is not None and self._export_thread.is_alive():
                raise DatasetError(
                    f"an export of {self._exporting!r} is already running - retry when it is done"
                )
            progress = ExportProgress(repo_id)
            self._export_progress = progress
            self._exporting = repo_id
            root = self.root_of(repo_id)
            out_path = Path(out) if out else None

            def work() -> None:
                try:
                    export_lerobot_v3(
                        root, repo_id, out_path, video_file_mb=video_file_mb,
                        data_file_mb=data_file_mb, progress=progress, validate=validate,
                    )
                except Exception:  # noqa: BLE001 - recorded in progress + manifest
                    pass
                finally:
                    with self._export_lock:
                        self._exporting = None

            self._export_thread = threading.Thread(target=work, name="dataset-export", daemon=True)
            self._export_thread.start()
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def wait_export(self, timeout: float | None = None) -> bool:
        """Join the running export (tests / CLI); True when none runs afterwards."""
        t = self._export_thread
        if t is None:
            return True
        t.join(timeout)
        return not t.is_alive()

    # -- housekeeping -----------------------------------------------------------------------
    def sweep(self) -> list[Path]:
        """Remove crashed ``episodes/.tmp-*`` directories of every dataset under the
        generic root AND every mapped root (10-frames §11.6 step 5), except the
        running session's (its open episode owns one). Called at ``Runtime.start``
        and before every listing; returns the removed paths."""
        in_use = self.in_use_repo()
        keep: set[Path] = set()
        if in_use is not None:
            keep = set((self.root_of(in_use) / EPISODES_DIR).glob(f"{TMP_PREFIX}*"))
        removed: list[Path] = []
        for base, pattern in self.search_roots():
            removed.extend(sweep_incomplete_episodes(base, keep, dataset_glob=pattern))
        return removed


    @staticmethod
    def _prune_namespace(ns_dir: Path) -> None:
        try:
            if ns_dir.exists() and not any(ns_dir.iterdir()):
                ns_dir.rmdir()
        except OSError:
            pass


__all__ = [
    "DEFAULT_NAMESPACE",
    "GENERIC_DATASET_GLOB",
    "LEGACY_READ_ONLY",
    "DatasetError",
    "DatasetStore",
    "NamespaceRoot",
]
