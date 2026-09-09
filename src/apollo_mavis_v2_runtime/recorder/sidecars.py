"""Apollo sidecar metadata at the dataset root (10-frames §9, §11.2).

Since 2026-09-07 the sidecars live NEXT TO the data they describe in the
episode-directory store: ``<root>/sessions/session_<id>.json`` and
``<root>/scenes/<sha256[:16]>.xml`` (the per-episode ``episode.json`` is written
by the recorder inside the episode directory). The LeRobot v3 export carries
them into ``meta/apollo/`` exactly as phase-07 laid them out. Writes are atomic
(``.tmp`` + ``os.replace``, same discipline as ``ProfileStore``).
"""

from __future__ import annotations

import hashlib
import json
import os
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from apollo_mavis_v2_core import Pose


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True))


def pose_json(pose: Pose) -> dict[str, list[float]]:
    return {
        "position": [float(x) for x in pose.position],
        "orientation_wxyz": [float(x) for x in pose.orientation],
    }


def software_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for key, dist in (
        ("apollo_core", "apollo-mavis-v2-core"),
        ("apollo_runtime", "apollo-mavis-v2-runtime"),
        ("apollo_sim", "apollo-mavis-v2-sim"),
        ("lerobot", "lerobot"),
        ("mujoco", "mujoco"),
        ("mink", "mink"),
    ):
        try:
            out[key] = importlib_metadata.version(dist)
        except importlib_metadata.PackageNotFoundError:
            out[key] = None
    return out


class SidecarWriter:
    """Owns ``<root>/sessions/`` and ``<root>/scenes/`` for one dataset root."""

    def __init__(self, dataset_root: Path) -> None:
        self.base = Path(dataset_root)

    # -- scenes -------------------------------------------------------------------
    def archive_scene_xml(self, xml: str) -> str:
        """Dedupe-archive the composed MJCF; returns the full sha256 hex."""
        sha = hashlib.sha256(xml.encode("utf-8")).hexdigest()
        path = self.base / "scenes" / f"{sha[:16]}.xml"
        if not path.exists():
            _write_text_atomic(path, xml)
        return sha

    # -- session ------------------------------------------------------------------
    def write_session(
        self,
        session_id: str,
        mode: str,
        spec: dict[str, Any],
        workcell: dict[str, Any],
        safety: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> Path:
        payload: dict[str, Any] = {
            "session_id": session_id,
            "mode": mode,
            "spec": spec,
            "workcell": workcell,
            "software": software_versions(),
            "safety": safety,
        }
        if extra:
            payload.update(extra)
        path = self.base / "sessions" / f"session_{session_id}.json"
        _write_json_atomic(path, payload)
        return path


__all__ = ["SidecarWriter", "software_versions", "pose_json"]
