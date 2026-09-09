"""Online DAgger runtime package data (15-online-dagger §9, D8): the agentic SKILL a
policy repo's coding harness installs to set up the trainer role against this runtime.

The skill directory (``SKILL.md`` + ``references/``) ships INSIDE this package and is
mirrored byte-for-byte into
``apollo-mavis-v2-policy-node/skills/mavis-online-dagger-trainer/``;
``RuntimeConfig.online_dagger.skill_dir`` may point at another directory. REST serves it
as ``GET /api/online_dagger/skill`` (the markdown) and ``GET /api/online_dagger/skill.tgz``
(the whole directory as a gzip tarball rooted at ``mavis-online-dagger-trainer/`` so
``curl ... | tar xz -C ~/.claude/skills/`` installs it in place).

The rollout-level shell lives in :mod:`apollo_mavis_v2_runtime.dagger.online_dagger`.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

SKILL_NAME = "mavis-online-dagger-trainer"
SKILL_FILE = "SKILL.md"
PACKAGED_SKILL_DIR = Path(__file__).resolve().parent / "skill"


def skill_dir(override: Path | str | None = None) -> Path:
    """The skill directory: ``override`` (``RuntimeConfig.online_dagger.skill_dir``) or the
    packaged one."""
    return Path(override) if override is not None else PACKAGED_SKILL_DIR


def skill_markdown(override: Path | str | None = None) -> str:
    """``SKILL.md`` text (starts with the ``---`` frontmatter)."""
    return (skill_dir(override) / SKILL_FILE).read_text(encoding="utf-8")


def skill_tarball(override: Path | str | None = None) -> bytes:
    """gzip tar of the skill directory rooted at ``mavis-online-dagger-trainer/``
    (deterministic member order; mtimes as on disk)."""
    root = skill_dir(override)
    if not (root / SKILL_FILE).is_file():
        raise FileNotFoundError(f"{root / SKILL_FILE} (no skill directory there)")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            tar.add(path, arcname=f"{SKILL_NAME}/{path.relative_to(root).as_posix()}")
    return buf.getvalue()


__all__ = [
    "PACKAGED_SKILL_DIR",
    "SKILL_FILE",
    "SKILL_NAME",
    "skill_dir",
    "skill_markdown",
    "skill_tarball",
]
