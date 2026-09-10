"""Seed the "Kitchen Interaction" posture as a (non-initial-condition) profile.

    uv run python -m apollo_mavis_v2_runtime.profiles.seed_kitchen --kind hardware
    uv run python -m apollo_mavis_v2_runtime.profiles.seed_kitchen --kind sim
    ... --dry-run          # print what it would write, touch nothing
    ... --config <path>    # a rendered config (defaults to the repo config)

This is the posture the ``mavis_v2_kitchen`` twin was PREPARED AND MEASURED at
(03-sim §4.4 and the header of ``assets/scenes/mavis_v2_kitchen.yaml``): the
Perception Arm frames the fridge / range / counter with its wrist D435i, and the
Manipulation Arm is parked at the cell's factory-zero posture at the far rail end.
It was that scene's keyframe; the operator asked on 2026-09-09 for it to become a
profile of its own after the GELLO mode it was originally built for was cut, so it
can be reached with "Go to profile" like any other posture.

    Perception Arm (view)     [2.646, -1.598, 0.018, 1.637, 0.25, 2.007, 0.029] rad
                              = [151.605, -91.559, 1.031, 93.793, 14.324, 114.993, 1.662] deg
    Manipulation Arm (grip)   [pi, 0, 0, 0, 0, 0, 0] rad — the xArm7 factory zero
                              with joint 1 = pi, i.e. the cell's own initial state

Unlike ``seed_initial`` this profile PINS BOTH RAILS (view 0.0, grip 0.65). It has to:
every kitchen number in 03-sim §4.4 was deprojected from a frame taken with the
Perception Arm's carriage at its zero end, so the appliance geometry only lines up
with the real cameras from THIS carriage position. Both values are the rails' homed
end stops, reached by ``home_rail`` itself, and a goto still goes through the twin
planner and the gate in two separately planned phases (joints, then carriages).

It is deliberately NOT designated the initial condition — that stays the operator's
default posture from ``seed_initial``, which is what ``R`` and "End session" return to.

Verified 2026-09-09 on both scenes (``mavis_v2_kitchen`` and ``mavis_v2``), microphone
on and off, at the cell's raised gate shell ``geom_inflation_m = 0.025``: collision
free, tightest monitored pair ``table <-> grip_*_finger_pad_2`` at 114.7 mm.

NOT run automatically: writing profiles is an explicit operator action.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from apollo_mavis_v2_core import ArmPosture, ProfileStore, StateProfile

PROFILE_NAME = "Kitchen Interaction"
NOTES = (
    "The posture the mavis_v2_kitchen twin was measured at (2026-09-09; 03-sim §4.4): "
    "Perception Arm framing the fridge / range / counter with its wrist D435i at "
    "[2.646, -1.598, 0.018, 1.637, 0.25, 2.007, 0.029] rad, carriage at its zero end; "
    "Manipulation Arm parked at the cell's factory-zero initial state, carriage at 0.65. "
    "The carriages are PINNED because the kitchen geometry was deprojected from a frame "
    "taken at exactly this Perception Arm carriage position. Not an initial condition. "
    "Seeded by apollo_mavis_v2_runtime.profiles.seed_kitchen."
)

# Radians, as measured / as the kitchen scene's keyframe carries them.
KITCHEN_POSTURE_RAD: dict[str, list[float]] = {
    "view": [2.646, -1.598, 0.018, 1.637, 0.25, 2.007, 0.029],
    "grip": [math.pi, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}
# Carriage per arm (m). Pinned, unlike seed_initial — see the module docstring.
KITCHEN_RAIL_M: dict[str, float] = {"view": 0.0, "grip": 0.65}
# Gripper open on arrival (the Perception Arm has none; the field is ignored there).
GRIPPER_OPEN_FRAC = 1.0


def kitchen_profile(kind: str) -> StateProfile:
    """The profile this module writes for ``kind`` (no id / timestamp yet)."""
    return StateProfile(
        name=PROFILE_NAME,
        notes=NOTES,
        workcell_kind=kind,  # type: ignore[arg-type]
        arms={
            arm_id: ArmPosture(
                q=list(q),
                rail_pos_m=KITCHEN_RAIL_M[arm_id],
                gripper_open_frac=GRIPPER_OPEN_FRAC,
            )
            for arm_id, q in KITCHEN_POSTURE_RAD.items()
        },
    )


def seed(store: ProfileStore, kind: str, *, dry_run: bool = False) -> StateProfile:
    """Write (or overwrite) this kind's Kitchen Interaction profile.

    Re-running is idempotent: the profile is matched by NAME within the kind, so the
    same file is rewritten instead of a second one appearing. The initial-condition
    designation is never touched.
    """
    profile = kitchen_profile(kind)
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
    return store.get(saved.profile_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m apollo_mavis_v2_runtime.profiles.seed_kitchen",
        description='Seed the "Kitchen Interaction" posture as a profile.',
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
        print(f'{verb} {kind} profile "{profile.name}" {profile.profile_id or "<new>"}')
        for arm_id, posture in sorted(profile.arms.items()):
            deg = [round(math.degrees(v), 3) for v in posture.q]
            rail = "keep" if posture.rail_pos_m is None else f"{posture.rail_pos_m:.3f} m"
            print(f"  {arm_id}: {deg} deg, rail {rail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
