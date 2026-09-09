"""StateProfile service — core ProfileStore re-export + wrappers (04-runtime §9)."""

from __future__ import annotations

from collections.abc import Mapping

from apollo_mavis_v2_core import ArmPosture, ArmState, ProfileStore, StateProfile

INITIAL_PROFILE_NAME = "initial"


def _postures(states: Mapping[str, ArmState], arms: list[str]) -> dict[str, ArmPosture]:
    out: dict[str, ArmPosture] = {}
    for arm_id in arms:
        st = states[arm_id]
        out[arm_id] = ArmPosture(
            q=[float(x) for x in st.q[:7]],
            rail_pos_m=st.rail_pos_m,
            gripper_open_frac=float(st.gripper.open_frac),
        )
    return out


def save_from_states(
    store: ProfileStore,
    states: Mapping[str, ArmState],
    arms: list[str],
    kind: str,
    name: str,
    notes: str = "",
) -> StateProfile:
    """Snapshot measured q/rail/gripper for the session arms into a profile."""
    profile = StateProfile(
        profile_id="",
        name=name,
        notes=notes,
        workcell_kind=kind,  # type: ignore[arg-type]
        arms=_postures(states, arms),
        created_at="",
    )
    return store.save(profile)


def save_from_snapshot(store: ProfileStore, snap, name: str, notes: str = "") -> StateProfile:
    """04-runtime §9 wrapper over a :class:`StateSnapshot`."""
    kind = snap.session_extra.get("kind", "sim")
    return save_from_states(store, snap.arms, list(snap.arms), kind, name, notes)


def save_initial_overwrite(
    store: ProfileStore,
    states: Mapping[str, ArmState],
    arms: list[str],
    kind: str,
) -> StateProfile:
    """Save the current state as the ``"initial"`` profile (OVERWRITE the
    existing same-name profile of this kind) and designate it initial."""
    existing = next(
        (p for p in store.list() if p.name == INITIAL_PROFILE_NAME and p.workcell_kind == kind),
        None,
    )
    profile = StateProfile(
        profile_id=existing.profile_id if existing else "",
        name=INITIAL_PROFILE_NAME,
        notes="auto-saved by set_initial_condition",
        workcell_kind=kind,  # type: ignore[arg-type]
        arms=_postures(states, arms),
        created_at=existing.created_at if existing else "",
        is_initial_condition=existing.is_initial_condition if existing else False,
    )
    profile = store.save(profile)
    store.set_initial(profile.profile_id)
    return store.get(profile.profile_id)


__all__ = [
    "ProfileStore",
    "StateProfile",
    "INITIAL_PROFILE_NAME",
    "save_from_states",
    "save_from_snapshot",
    "save_initial_overwrite",
]
