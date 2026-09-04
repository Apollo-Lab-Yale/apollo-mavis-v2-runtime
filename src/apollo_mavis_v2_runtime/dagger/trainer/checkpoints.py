"""CheckpointStore — versioned checkpoint dirs + atomic pointers (12-dagger §7).

Layout::

    checkpoints/{run_id}/
      v{n:06d}/{state_dict.pt, trainer_state.pt, manifest.json}
      LATEST            # "v000012\\n", os.replace-atomic; sanity_ok only
      LAST_KNOWN_GOOD   # maintained by the RUNTIME (§8/§12)
      doubt.jsonl

Filesystem-only (no torch) so the runtime reloader can import it cheaply.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from pathlib import Path

from apollo_mavis_v2_core.dagger import CheckpointInfo

STATE_DICT = "state_dict.pt"
TRAINER_STATE = "trainer_state.pt"
MANIFEST = "manifest.json"
LATEST = "LATEST"
LAST_KNOWN_GOOD = "LAST_KNOWN_GOOD"

_VERSION_RE = re.compile(r"^v(\d{6})$")


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class CheckpointStore:
    """One run's checkpoint directory; shared by trainer (write) + runtime (read)."""

    def __init__(self, checkpoints_root: Path | str, run_id: str) -> None:
        self.run_id = run_id
        self.root = Path(checkpoints_root) / run_id
        self.root.mkdir(parents=True, exist_ok=True)

    # -- paths --------------------------------------------------------------------
    def version_dir(self, version: int) -> Path:
        return self.root / f"v{version:06d}"

    def state_dict_path(self, version: int) -> Path:
        return self.version_dir(version) / STATE_DICT

    def trainer_state_path(self, version: int) -> Path:
        return self.version_dir(version) / TRAINER_STATE

    # -- manifests ------------------------------------------------------------------
    def write_manifest(self, info: CheckpointInfo) -> None:
        d = self.version_dir(info.version)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / (MANIFEST + ".tmp")
        tmp.write_text(json.dumps(dataclasses.asdict(info), indent=2), encoding="utf-8")
        os.replace(tmp, d / MANIFEST)

    def read_manifest(self, version: int) -> CheckpointInfo | None:
        path = self.version_dir(version) / MANIFEST
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return CheckpointInfo(**data)
        except (OSError, json.JSONDecodeError, TypeError):
            return None

    def verify(self, info: CheckpointInfo) -> bool:
        """sha256(state_dict.pt) matches the manifest (pre-swap check, §8)."""
        path = self.state_dict_path(info.version)
        try:
            return sha256_file(path) == info.sha256
        except OSError:
            return False

    def versions(self) -> list[int]:
        out = []
        if not self.root.exists():
            return out
        for entry in self.root.iterdir():
            m = _VERSION_RE.match(entry.name)
            if m and entry.is_dir():
                out.append(int(m.group(1)))
        return sorted(out)

    # -- pointers ----------------------------------------------------------------
    def _read_pointer(self, name: str) -> int | None:
        try:
            text = (self.root / name).read_text(encoding="utf-8").strip()
        except OSError:
            return None
        m = _VERSION_RE.match(text)
        return int(m.group(1)) if m else None

    def _write_pointer(self, name: str, version: int) -> None:
        tmp = self.root / (name + ".tmp")
        tmp.write_text(f"v{version:06d}\n", encoding="utf-8")
        os.replace(tmp, self.root / name)

    def latest(self) -> int | None:
        return self._read_pointer(LATEST)

    def advance_latest(self, version: int) -> None:
        self._write_pointer(LATEST, version)

    def last_known_good(self) -> int | None:
        return self._read_pointer(LAST_KNOWN_GOOD)

    def set_last_known_good(self, version: int) -> None:
        self._write_pointer(LAST_KNOWN_GOOD, version)

    # -- doubt log (HG-DAgger Eq. 4 input; logged, not gating in v1) ---------------
    def append_doubt(self, payload: dict) -> None:
        with open(self.root / "doubt.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")


__all__ = [
    "CheckpointStore",
    "sha256_file",
    "STATE_DICT",
    "TRAINER_STATE",
    "MANIFEST",
    "LATEST",
    "LAST_KNOWN_GOOD",
]
