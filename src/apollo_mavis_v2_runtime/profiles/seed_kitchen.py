"""Seed the "Kitchen Interaction" posture as a (non-initial-condition) profile.

    uv run python -m apollo_mavis_v2_runtime.profiles.seed_kitchen --kind hardware
    uv run python -m apollo_mavis_v2_runtime.profiles.seed_kitchen --kind sim
    ... --dry-run          # print what it would write, touch nothing
    ... --config <path>    # a rendered config (defaults to the repo config)

The point of this profile is to put the PERCEPTION ARM where the ``mavis_v2_kitchen``
twin was measured from (03-sim §4.4 and the header of
``assets/scenes/mavis_v2_kitchen.yaml``): framing the fridge / range / counter with its
wrist D435i, carriage pinned at its zero end. Every kitchen number in §4.4 was
deprojected from a frame taken there, so the appliance geometry only lines up with the
real cameras from THAT carriage position — hence the pin. The operator asked on
2026-09-09 for the posture to become a profile of its own, reachable with "Go to
profile", after the GELLO mode it was originally built for was cut.

    Perception Arm (view)     [2.646, -1.598, 0.018, 1.637, 0.25, 2.007, 0.029] rad
                              = [151.605, -91.559, 1.031, 93.793, 14.324, 114.993, 1.662] deg
                              carriage PINNED at 0.0
    Manipulation Arm (grip)   exactly its entry in the DEFAULT posture (``seed_initial``),
                              carriage left unset ("keep it where it is")

**The Manipulation Arm is the operator's default posture, not the scene keyframe**
(operator request 2026-09-10). Until then this profile carried the kitchen scene's own
keyframe for the grip arm — the xArm7 factory zero, ``[pi, 0, 0, 0, 0, 0, 0]``, with the
carriage pinned at 0.65. Both of those cost real motion for nothing:

* joint 1 of the factory zero is ``+pi`` while the default posture's is ``-180 deg`` =
  ``-pi``. Same physical orientation, DIFFERENT joint value, and the planner walks
  straight lines in joint space — so every goto between the default posture and this
  profile rotated joint 1 a full **360 deg** before teleop could start. That is the
  "turn a full circle" the operator reported.
* the 0.65 pin added an up-to-0.65 m carriage traverse whose only purpose was parking
  the arm at the far end, out of the Perception Arm's way.

Neither is needed: the grip arm has nothing to do with the kitchen measurement, so
parking it at the posture it is ALREADY in makes a goto from the default posture move
the Perception Arm and NOTHING else. The Manipulation Arm's entry is therefore DERIVED
from ``seed_initial.default_profile`` rather than copied, so the two can never drift
apart.

It is deliberately NOT designated the initial condition — that stays the operator's
default posture from ``seed_initial``, which is what ``R`` and "End session" return to.

Verified 2026-09-10 in both twins (``mavis_v2_kitchen`` and ``mavis_v2``), microphone on
and off, at the cell's raised gate shell ``geom_inflation_m = 0.025``: collision free at
EVERY grip carriage position from 0.000 to 0.650 m (which is why the pin could go),
tightest monitored pair ``obstacle <-> grip_rail_platform`` at 75.3 mm.

NOT run automatically: writing profiles is an explicit operator action. Re-run it after
this change — a store seeded before 2026-09-10 still holds the old numbers.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from apollo_mavis_v2_core import ArmPosture, ProfileStore, StateProfile

from .seed_initial import DEFAULT_POSTURE_DEG, GRIPPER_OPEN_FRAC, default_profile

PROFILE_NAME = "Kitchen Interaction"
NOTES = (
    "Where the mavis_v2_kitchen twin was measured FROM (2026-09-09; 03-sim §4.4): "
    "Perception Arm framing the fridge / range / counter with its wrist D435i at "
    "[2.646, -1.598, 0.018, 1.637, 0.25, 2.007, 0.029] rad, carriage PINNED at its zero "
    "end because the kitchen geometry was deprojected from a frame taken there. The "
    "Manipulation Arm is its entry in the DEFAULT posture with the carriage left unset "
    "(operator request 2026-09-10), so a goto from the default posture moves the "
    "Perception Arm and nothing else - the old factory-zero entry differed from the "
    "default only in the SIGN of joint 1 and cost a 360 deg rotation for nothing. "
    "Not an initial condition. Seeded by apollo_mavis_v2_runtime.profiles.seed_kitchen."
)

# The ONE arm whose kitchen posture differs from the default (radians, as measured).
VIEW_POSTURE_RAD: list[float] = [2.646, -1.598, 0.018, 1.637, 0.25, 2.007, 0.029]
# The Perception Arm's carriage is PINNED: the kitchen numbers only line up from here.
VIEW_RAIL_M = 0.0

# The whole posture, for tests and tools that want it in one place. The Manipulation Arm
# is DERIVED from the default posture (never copied) so the two cannot drift apart, and
# its carriage is None = "keep it where it is", exactly as in seed_initial.
KITCHEN_POSTURE_RAD: dict[str, list[float]] = {
    "view": list(VIEW_POSTURE_RAD),
    "grip": [math.radians(d) for d in DEFAULT_POSTURE_DEG["grip"]],
}
KITCHEN_RAIL_M: dict[str, float | None] = {"view": VIEW_RAIL_M, "grip": None}


def kitchen_profile(kind: str) -> StateProfile:
    """The profile this module writes for ``kind`` (no id / timestamp yet).

    Built by taking the DEFAULT posture and replacing the Perception Arm's entry: that
    is the invariant the operator asked for on 2026-09-10 — only the Perception Arm may
    differ from the default — expressed as code rather than as two lists to keep in
    step. Any arm ``seed_initial`` gains in future is inherited unchanged.
    """
    arms = dict(default_profile(kind).arms)
    arms["view"] = ArmPosture(
        q=list(VIEW_POSTURE_RAD),
        rail_pos_m=VIEW_RAIL_M,
        gripper_open_frac=GRIPPER_OPEN_FRAC,
    )
    return StateProfile(
        name=PROFILE_NAME,
        notes=NOTES,
        workcell_kind=kind,  # type: ignore[arg-type]
        arms=arms,
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
