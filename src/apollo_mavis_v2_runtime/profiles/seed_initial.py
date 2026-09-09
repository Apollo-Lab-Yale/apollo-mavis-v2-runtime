"""Seed the MAVIS v2 default posture as an initial-condition profile.

    uv run python -m apollo_mavis_v2_runtime.profiles.seed_initial --kind hardware
    uv run python -m apollo_mavis_v2_runtime.profiles.seed_initial --kind sim
    ... --dry-run          # print what it would write, touch nothing
    ... --config <path>    # a rendered config (defaults to the repo config)

The postures are the OPERATOR'S, given in DEGREES on 2026-09-08 (04-runtime §10.5):

    Manipulation Arm (grip)  [-180, -12, -20,  30, -5, 35, -8.9]
    Perception Arm   (view)  [   0, 0.8,   0, 28.9,  0, 28.2,  0]

This is what the ``R`` key and the Cockpit's "End session" return to. It writes ONE
profile per workcell kind and designates it the initial condition — the store keeps at
most one per kind, so re-running it replaces the designation rather than accumulating.

``rail_pos_m`` is deliberately LEFT UNSET (``None`` = "keep the carriage"): the
operator asked for the joints first and called the carriage optional, and the twin's
travel limit is a hard end stop that nothing should be commanded onto blind. Save a
profile from a live session (Cockpit: "save current state as profile" with "use as
initial condition") when you do want the carriages homed too — the return then runs
the carriage as a second, separately planned and gated phase.

NOT run automatically: writing an initial condition changes what a session's
return-to-start motion aims at, so it stays an explicit operator action.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from apollo_mavis_v2_core import ArmPosture, ProfileStore, StateProfile

PROFILE_NAME = "default posture (2026-09-08)"
NOTES = (
    "Operator-given default posture, 2026-09-08: Manipulation Arm "
    "[-180, -12, -20, 30, -5, 35, -8.9] deg, Perception Arm "
    "[0, 0.8, 0, 28.9, 0, 28.2, 0] deg. Carriages unset (kept where they are). "
    "Seeded by apollo_mavis_v2_runtime.profiles.seed_initial."
)

# Degrees, as the operator reads them off the controller / UFACTORY Studio.
DEFAULT_POSTURE_DEG: dict[str, list[float]] = {
    "grip": [-180.0, -12.0, -20.0, 30.0, -5.0, 35.0, -8.9],
    "view": [0.0, 0.8, 0.0, 28.9, 0.0, 28.2, 0.0],
}
# Gripper open on arrival (the Perception Arm has none; the field is ignored there).
GRIPPER_OPEN_FRAC = 1.0


def default_profile(kind: str) -> StateProfile:
    """The profile this module writes for ``kind`` (no id / timestamp yet)."""
    return StateProfile(
        name=PROFILE_NAME,
        notes=NOTES,
        workcell_kind=kind,  # type: ignore[arg-type]
        arms={
            arm_id: ArmPosture(
                q=[math.radians(d) for d in deg],
                rail_pos_m=None,  # keep the carriage (see the module docstring)
                gripper_open_frac=GRIPPER_OPEN_FRAC,
            )
            for arm_id, deg in DEFAULT_POSTURE_DEG.items()
        },
    )


def seed(store: ProfileStore, kind: str, *, dry_run: bool = False) -> StateProfile:
    """Write (or overwrite) this kind's default-posture profile and designate it.

    Re-running is idempotent: the profile is matched by NAME within the kind, so the
    same file is rewritten instead of a second one appearing.
    """
    profile = default_profile(kind)
    existing = next(
        (p for p in store.list() if p.name == PROFILE_NAME and p.workcell_kind == kind),
        None,
    )
    if existing is not None:
        profile = profile.model_copy(
            update={"profile_id": existing.profile_id, "created_at": existing.created_at}
        )
    if dry_run:
        return profile
    saved = store.save(profile)
    store.set_initial(saved.profile_id)
    return store.get(saved.profile_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m apollo_mavis_v2_runtime.profiles.seed_initial",
        description="Seed the operator's default posture as an initial-condition profile.",
    )
    parser.add_argument(
        "--kind",
        action="append",
        choices=["hardware", "sim"],
        help="workcell kind (repeatable; default: both)",
    )
    parser.add_argument("--config", type=Path, default=None, help="runtime config YAML")
    parser.add_argument(
        "--dry-run", action="store_true", help="print what would be written, touch nothing"
    )
    args = parser.parse_args(argv)

    from ..config import load_runtime_config

    cfg = load_runtime_config(args.config)
    store = ProfileStore(cfg.profiles_dir)
    for kind in args.kind or ["hardware", "sim"]:
        profile = seed(store, kind, dry_run=args.dry_run)
        verb = "would write" if args.dry_run else "wrote"
        print(f"{verb} {kind} initial condition {profile.profile_id or '<new>'} in {store.root}")
        for arm_id, posture in sorted(profile.arms.items()):
            deg = [round(math.degrees(v), 3) for v in posture.q]
            rail = "keep" if posture.rail_pos_m is None else f"{posture.rail_pos_m:.3f} m"
            print(f"  {arm_id}: {deg} deg, rail {rail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
