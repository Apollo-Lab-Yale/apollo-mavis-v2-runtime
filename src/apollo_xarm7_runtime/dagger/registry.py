"""Checkpoint inventory + session policy resolution (04-runtime §13.1; 12-dagger §9).

``GET /api/policies`` rows come from scanning ``checkpoints_root`` for
manifests; promotion is explicit — a ``PROMOTED`` pointer file at the root
naming one deploy checkpoint id. Inference sessions load ONLY the promoted
deploy checkpoint; DAgger defaults to the newest sanity_ok online version.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from apollo_xarm7_core.dagger import CheckpointInfo
from apollo_xarm7_core.protocol import PolicyInfo

from ..errors import SessionError
from .trainer.checkpoints import MANIFEST, STATE_DICT

logger = logging.getLogger(__name__)

PROMOTED_FILE = "PROMOTED"
_ONLINE_RE = re.compile(r"^v(\d{6})$")
_DEPLOY_RE = re.compile(r"^v(\d{3})$")


@dataclass(frozen=True)
class ResolvedPolicy:
    policy_id: str
    info: CheckpointInfo
    state_dict_path: Path


def _read_manifest(d: Path) -> CheckpointInfo | None:
    try:
        return CheckpointInfo(**json.loads((d / MANIFEST).read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _iter_checkpoints(root: Path):
    """Yield (policy_id, dir, info, is_deploy) for every readable manifest."""
    if not root.exists():
        return
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for d in sorted(run_dir.iterdir()):
            if d.is_dir() and _ONLINE_RE.match(d.name):
                info = _read_manifest(d)
                if info is not None:
                    yield f"{run_dir.name}/{d.name}", d, info, False
        deploy = run_dir / "deploy"
        if deploy.is_dir():
            for d in sorted(deploy.iterdir()):
                if d.is_dir() and _DEPLOY_RE.match(d.name):
                    info = _read_manifest(d)
                    if info is not None:
                        yield f"{run_dir.name}/deploy/{d.name}", d, info, True


def promoted_id(root: Path) -> str | None:
    try:
        text = (Path(root) / PROMOTED_FILE).read_text(encoding="utf-8").strip()
        return text or None
    except OSError:
        return None


def scan_policies(root: Path) -> list[PolicyInfo]:
    root = Path(root)
    promoted = promoted_id(root)
    out: list[PolicyInfo] = []
    for policy_id, d, info, is_deploy in _iter_checkpoints(root):
        out.append(PolicyInfo(
            policy_id=policy_id,
            path=str(d),
            action_space=info.action_space,  # type: ignore[arg-type]
            action_frame=info.action_frame,
            policy_version=info.version,
            promoted=is_deploy and policy_id == promoted,
        ))
    return out


def resolve_policy(root: Path, mode: str, policy: str | None) -> ResolvedPolicy:
    """Session policy lookup; raises ``SessionError`` (-> 409) per 12-dagger §9."""
    root = Path(root)
    entries = {pid: (d, info, dep) for pid, d, info, dep in _iter_checkpoints(root)}
    if mode == "inference":
        pid = policy or promoted_id(root)
        if pid is None:
            raise SessionError("no promoted deploy checkpoint (12-dagger §9)")
        entry = entries.get(pid)
        if entry is None:
            raise SessionError(f"unknown policy {pid!r}")
        d, info, is_deploy = entry
        if not is_deploy or pid != promoted_id(root):
            raise SessionError(
                f"inference sessions load only the promoted deploy checkpoint, not {pid!r}"
            )
        return ResolvedPolicy(pid, info, d / STATE_DICT)
    # dagger: explicit id, else the newest sanity_ok checkpoint (any run)
    if policy is not None:
        entry = entries.get(policy)
        if entry is None:
            raise SessionError(f"unknown policy {policy!r}")
        d, info, _ = entry
        if not info.sanity_ok:
            raise SessionError(f"policy {policy!r} failed its sanity gate")
        return ResolvedPolicy(policy, info, d / STATE_DICT)
    best: tuple[str, Path, CheckpointInfo] | None = None
    for pid, (d, info, _dep) in entries.items():
        if not info.sanity_ok:
            continue
        if best is None or info.created_wallclock_ns > best[2].created_wallclock_ns:
            best = (pid, d, info)
    if best is None:
        raise SessionError("no policy checkpoints available (seed one first)")
    pid, d, info = best
    return ResolvedPolicy(pid, info, d / STATE_DICT)


__all__ = ["scan_policies", "resolve_policy", "promoted_id", "ResolvedPolicy",
           "PROMOTED_FILE"]
