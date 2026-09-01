"""REST /api routes (04-runtime §13.1). Response models come from core.protocol."""

from __future__ import annotations

from apollo_xarm7_core import ProfileError, ProfileNotFoundError, StateProfile
from apollo_xarm7_core.protocol import (
    KEYMAP,
    KeymapEntry,
    ProfileInfo,
    SceneInfo,
    SessionInfo,
    SessionSpec,
    WorkcellStatus,
)
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

import apollo_xarm7_runtime

from ..errors import SessionError, SessionNotFoundError

router = APIRouter()


def _runtime(request: Request):
    return request.app.state.runtime


@router.get("/health")
def health(request: Request) -> dict:
    return {
        "status": "ok",
        "epoch": _runtime(request).epoch,
        "version": apollo_xarm7_runtime.__version__,
    }


@router.get("/workcell")
def workcell(request: Request) -> WorkcellStatus:
    return _runtime(request).manager.workcell_status()


@router.get("/cameras")
def cameras(request: Request) -> list:
    return _runtime(request).manager.camera_infos()


@router.get("/scenes")
def scenes(request: Request, kind: str = "sim") -> list[SceneInfo]:
    if kind not in ("sim", "twin"):
        raise HTTPException(422, "kind must be sim|twin")
    return _runtime(request).manager.scene_infos(kind)


@router.get("/keymap")
def keymap() -> list[KeymapEntry]:
    return list(KEYMAP)


@router.get("/policies")
def policies() -> list:
    return []  # checkpoint registry lands with phase-08


@router.get("/episodes")
def episodes(request: Request) -> dict:
    return {"repo_id": None, "total_episodes": 0, "total_frames": 0}


# -- profiles (management CRUD only; save/set-initial ride /ws/control) ----------
@router.get("/profiles")
def list_profiles(request: Request) -> list[ProfileInfo]:
    return [
        ProfileInfo(
            profile_id=p.profile_id, name=p.name, arms=sorted(p.arms),
            notes=p.notes, created_at=p.created_at,
            is_initial_condition=p.is_initial_condition,
        )
        for p in _runtime(request).profile_store.list()
    ]


@router.get("/profiles/{profile_id}")
def get_profile(request: Request, profile_id: str) -> StateProfile:
    try:
        return _runtime(request).profile_store.get(profile_id)
    except ProfileNotFoundError:
        raise HTTPException(404, f"unknown profile {profile_id!r}") from None


class ProfilePatch(BaseModel):
    name: str | None = None
    notes: str | None = None


@router.patch("/profiles/{profile_id}")
def patch_profile(request: Request, profile_id: str, patch: ProfilePatch) -> ProfileInfo:
    store = _runtime(request).profile_store
    try:
        current = store.get(profile_id)
        p = store.rename(
            profile_id,
            patch.name if patch.name is not None else current.name,
            patch.notes,
        )
    except ProfileNotFoundError:
        raise HTTPException(404, f"unknown profile {profile_id!r}") from None
    return ProfileInfo(
        profile_id=p.profile_id, name=p.name, arms=sorted(p.arms), notes=p.notes,
        created_at=p.created_at, is_initial_condition=p.is_initial_condition,
    )


@router.delete("/profiles/{profile_id}", status_code=204)
def delete_profile(request: Request, profile_id: str) -> Response:
    store = _runtime(request).profile_store
    try:
        store.delete(profile_id)
    except ProfileNotFoundError:
        raise HTTPException(404, f"unknown profile {profile_id!r}") from None
    except ProfileError as e:  # designated initial condition
        raise HTTPException(409, str(e)) from None
    return Response(status_code=204)


# -- session lifecycle ---------------------------------------------------------------
@router.get("/session")
def get_session(request: Request) -> SessionInfo:
    try:
        return _runtime(request).manager.info()
    except SessionNotFoundError:
        raise HTTPException(404, "no active session") from None


@router.post("/session")
def post_session(request: Request, spec: SessionSpec) -> SessionInfo:
    try:
        return _runtime(request).manager.create(spec)
    except SessionError as e:
        raise HTTPException(409, str(e)) from None


@router.delete("/session", status_code=204)
def delete_session(request: Request) -> Response:
    _runtime(request).manager.teardown()  # idempotent
    return Response(status_code=204)


__all__ = ["router"]
