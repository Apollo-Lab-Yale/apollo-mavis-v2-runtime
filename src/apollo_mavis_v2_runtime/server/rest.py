"""REST /api routes (04-runtime §13.1). Response models come from core.protocol."""

from __future__ import annotations

import logging

from apollo_mavis_v2_core import ProfileError, ProfileNotFoundError, StateProfile
from apollo_mavis_v2_core.protocol import (
    KEYMAP,
    ArmMaintenanceRequest,
    ArmMaintenanceResult,
    DatasetExportRequest,
    DatasetInfo,
    DatasetLayoutInfo,
    DoraInfo,
    EpisodeInfo,
    EpisodePlaybackInfo,
    EpisodePlaybackRequest,
    KeymapEntry,
    MicrophoneInfo,
    OnlineDaggerSessionInfo,
    ProfileInfo,
    ReturnHomeResult,
    SceneInfo,
    SessionInfo,
    SessionSpec,
    TrackerCalibrationCommand,
    TrackerCalibrationStatus,
    WorkcellStatus,
)
from fastapi import APIRouter, HTTPException, Path, Request, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

import apollo_mavis_v2_runtime

from ..devices.tracker_calibration import CalibrationError
from ..errors import (
    MaintenanceUnavailableError,
    SafetyConfigError,
    SessionError,
    SessionNotFoundError,
)
from ..recorder.datasets import DatasetError
from ..recorder.playback import PlaybackError

logger = logging.getLogger(__name__)

router = APIRouter()


def _runtime(request: Request):
    return request.app.state.runtime


@router.get("/health")
def health(request: Request) -> dict:
    return {
        "status": "ok",
        "epoch": _runtime(request).epoch,
        "version": apollo_mavis_v2_runtime.__version__,
    }


@router.get("/workcell")
def workcell(request: Request, kind: str | None = None) -> WorkcellStatus:
    """``?kind=hardware|sim`` selects the workcell described (phase-11 Welcome
    tabs); omitted = the session's kind or sim (legacy behaviour)."""
    if kind is not None and kind not in ("hardware", "sim"):
        raise HTTPException(422, "kind must be hardware|sim")
    return _runtime(request).manager.workcell_status(kind)


@router.get("/cameras")
def cameras(request: Request) -> list:
    return _runtime(request).manager.camera_infos()


@router.get("/microphones")
def microphones(request: Request) -> list[MicrophoneInfo]:
    """Configured microphones with their live status (phase-11; the Hardware
    tab renders the MicTile from this list and reads levels from telemetry)."""
    return _runtime(request).microphone_infos()


@router.get("/scenes")
def scenes(request: Request, kind: str = "sim") -> list[SceneInfo]:
    if kind not in ("sim", "twin"):
        raise HTTPException(422, "kind must be sim|twin")
    return _runtime(request).manager.scene_infos(kind)


@router.get("/keymap")
def keymap() -> list[KeymapEntry]:
    return list(KEYMAP)


@router.get("/policies")
def policies(request: Request) -> list:
    from ..dagger.registry import scan_policies

    return scan_policies(_runtime(request).cfg.checkpoints_root)


@router.get("/episodes", deprecated=True)
def episodes(request: Request) -> dict:
    """DEPRECATED (2026-09-07): the running session's counters ride
    ``telemetry.episode``; kept one release as an alias (``repo_id: null``
    without a collect / DAgger session)."""
    status = _runtime(request).manager.episode_status()
    if status is None:
        return {"repo_id": None, "total_episodes": 0, "total_frames": 0}
    return {
        "repo_id": status.repo_id,
        "total_episodes": status.total_episodes,
        "total_frames": status.total_frames,
    }


# -- datasets (2026-09-07; 04-runtime §10.6 / §13.1; 10-frames §11) ----------------------------
# Every route reads manifest.json / episode.json only (no lerobot import); the export
# is a batch job (202) whose progress rides telemetry.datasets.export. Episode ids are
# 10-frames §11.3 (`20260907T141203.512Z-3f9a1c`): the `.` and `Z` travel verbatim.
_NAME = r"^[A-Za-z0-9][A-Za-z0-9_\-]*$"
_EPISODE_ID = r"^[0-9TZ.\-a-f]+$"


def _dataset_error(e: DatasetError) -> HTTPException:
    return HTTPException(404 if e.not_found else 409, str(e))


@router.get("/datasets")
def list_datasets(request: Request) -> list[DatasetInfo]:
    return _runtime(request).manager.dataset_store.list()


# Declared BEFORE the parametrised /datasets/{ns}/{name} routes on purpose (2026-09-08;
# 15-online-dagger §7): a literal segment must never be read as a namespace.
@router.get("/datasets/layout")
def dataset_layout(request: Request) -> DatasetLayoutInfo:
    """Where datasets live (``RuntimeConfig.datasets``): the default namespace, the
    generic root and every mapped namespace root, so the UI shows the real folder in
    its previews and never hard-codes a namespace."""
    return _runtime(request).manager.dataset_store.layout()


@router.get("/datasets/{ns}/{name}")
def get_dataset(
    request: Request, ns: str = Path(pattern=_NAME), name: str = Path(pattern=_NAME)
) -> DatasetInfo:
    repo_id = f"{ns}/{name}"
    try:
        info = _runtime(request).manager.dataset_store.describe(repo_id)
    except Exception as e:  # noqa: BLE001 - a corrupt manifest is a 409 detail, never a 500
        raise HTTPException(409, f"dataset {repo_id!r} is unreadable: {e}") from None
    if info is None:
        raise HTTPException(404, f"unknown dataset {repo_id!r}")
    return info


@router.get("/datasets/{ns}/{name}/episodes")
def list_episodes(
    request: Request, ns: str = Path(pattern=_NAME), name: str = Path(pattern=_NAME)
) -> list[EpisodeInfo]:
    try:
        return _runtime(request).manager.dataset_store.episodes(f"{ns}/{name}")
    except DatasetError as e:
        raise _dataset_error(e) from None


@router.get("/datasets/{ns}/{name}/episodes/{episode_id}/playback")
def episode_playback_info(
    request: Request,
    ns: str = Path(pattern=_NAME),
    name: str = Path(pattern=_NAME),
    episode_id: str = Path(pattern=_EPISODE_ID),
) -> EpisodePlaybackInfo:
    """What a playback of this episode would do (2026-09-10; 04-runtime §10.8, 05-ui §8.1
    item 7): frame count, fps, duration and the per-arm INITIAL state, plus ``playable`` /
    ``reason``.

    Session-less by design — the Welcome page's Playback dialog opens and explains itself
    before anything moves, so a refusal is a sentence in the dialog rather than a 409 the
    operator has to interpret. 404 for an unknown dataset / episode; 409 for a legacy tree
    or an unreadable ``frames.parquet``.
    """
    try:
        return _runtime(request).manager.episode_playback_info(f"{ns}/{name}", episode_id)
    except DatasetError as e:
        raise _dataset_error(e) from None
    except PlaybackError as e:
        raise HTTPException(404 if e.not_found else 409, str(e)) from None


@router.delete("/datasets/{ns}/{name}/episodes/{episode_id}", status_code=204)
def delete_episode(
    request: Request,
    ns: str = Path(pattern=_NAME),
    name: str = Path(pattern=_NAME),
    episode_id: str = Path(pattern=_EPISODE_ID),
) -> Response:
    """Removes ONE episode directory (10-frames §11.7): 404 unknown, 409 for the
    open episode / a legacy tree; allowed while a session records into the dataset."""
    who = request.client.host if request.client is not None else "unknown"
    try:
        _runtime(request).manager.dataset_store.delete_episode(f"{ns}/{name}", episode_id)
    except DatasetError as e:
        logger.info(
            "delete episode %s of %s/%s from %s: refused - %s", episode_id, ns, name, who, e
        )
        raise _dataset_error(e) from None
    logger.info("delete episode %s of %s/%s from %s: ok", episode_id, ns, name, who)
    return Response(status_code=204)


@router.delete("/datasets/{ns}/{name}", status_code=204)
def delete_dataset(
    request: Request, ns: str = Path(pattern=_NAME), name: str = Path(pattern=_NAME)
) -> Response:
    """The whole tree; 409 while a session records into it or an export runs."""
    who = request.client.host if request.client is not None else "unknown"
    try:
        _runtime(request).manager.dataset_store.delete_dataset(f"{ns}/{name}")
    except DatasetError as e:
        logger.info("delete dataset %s/%s from %s: refused - %s", ns, name, who, e)
        raise _dataset_error(e) from None
    logger.info("delete dataset %s/%s from %s: ok", ns, name, who)
    return Response(status_code=204)


@router.post("/datasets/{ns}/{name}/export", status_code=202)
def export_dataset(
    request: Request,
    body: DatasetExportRequest,
    ns: str = Path(pattern=_NAME),
    name: str = Path(pattern=_NAME),
) -> dict:
    """Starts the LeRobot v3 export job (10-frames §11.8) -> 202 ``{repo_id, format,
    started_at}``; progress on ``telemetry.datasets.export``. 409 while a session
    records into the dataset, while another export runs or for a legacy tree."""
    rt = _runtime(request)
    repo_id = f"{ns}/{name}"
    try:
        started = rt.manager.dataset_store.export(
            repo_id,
            body.format,
            body.out,
            video_file_mb=rt.cfg.recorder.export.video_file_mb,
            data_file_mb=rt.cfg.recorder.export.data_file_mb,
        )
    except DatasetError as e:
        raise _dataset_error(e) from None
    logger.info("export %s of %s started", body.format, repo_id)
    return {"repo_id": repo_id, "format": body.format, "started_at": started}


# -- profiles (management CRUD only; save/set-initial ride /ws/control) ----------
@router.get("/profiles")
def list_profiles(request: Request) -> list[ProfileInfo]:
    return [
        ProfileInfo(
            profile_id=p.profile_id,
            name=p.name,
            arms=sorted(p.arms),
            notes=p.notes,
            created_at=p.created_at,
            is_initial_condition=p.is_initial_condition,
            workcell_kind=p.workcell_kind,
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
        profile_id=p.profile_id,
        name=p.name,
        arms=sorted(p.arms),
        notes=p.notes,
        created_at=p.created_at,
        is_initial_condition=p.is_initial_condition,
        workcell_kind=p.workcell_kind,
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
    rt = _runtime(request)
    # Orphaned-session watch (2026-09-09; 04-runtime §13.2): a session that has just
    # been created has not had time to open its /ws/control socket, and a hardware
    # bring-up can take seconds before the Cockpit mounts. Stamp the countdown before
    # the session exists so the grace period starts from the operator's click.
    rt.orphan_watch.note_activity("POST /api/session")
    if rt.tracker_calibration.active:  # reader restarts / trigger clicks must not hit a session
        raise HTTPException(409, "tracker calibration in progress")
    # phase-09d: fail fast (before manager._lock, which a rail-homing job holds for up to
    # its whole connect - monitor join + bring-up) while a job / home_rail request is
    # live; create() re-checks under the lock for the race-free 409.
    busy_arm = rt.rail_homing.active_arm
    if busy_arm is not None:
        from ..session.hardware import arm_label

        raise HTTPException(
            409, f"rail homing in progress on the {arm_label(busy_arm)} - wait for it to finish"
        )
    try:
        info = rt.manager.create(spec)
    except (SessionError, SafetyConfigError) as e:
        # SessionError: the sim / hardware refusal matrices (04-runtime §5, §13.1);
        # SafetyConfigError: a hardware session that would run without a twin-backed
        # SafetyGate (11-safety §4) - never a 500, the operator reads the detail.
        raise HTTPException(409, str(e)) from None
    # The calibration's own guard reads manager.session_active, which create() raises
    # under its lock; a calibration that started between the check above and that
    # moment passed both guards, so re-check and give way to it (13-tracker §4).
    if rt.tracker_calibration.active:
        rt.manager.teardown()
        raise HTTPException(409, "tracker calibration in progress")
    return info


@router.post("/session/return_home")
def post_session_return_home(request: Request) -> ReturnHomeResult:
    """Walk the workcell back to its designated initial-condition profile and answer
    with where the arms ended up (04-runtime §10.5; 2026-09-08 operator request).

    SYNCHRONOUS: the Cockpit's "End session" runs this and waits before it issues the
    DELETE, so the arms are folded back before the drivers hand them over. Two
    twin-planned, gated phases — joints first with the carriages held, then the
    carriages — and any operator input cancels the motion. Never an error status for an
    operational refusal (no session, no initial condition, an open episode, a faulted
    arm, an unplannable path): the answer carries ``ok: false`` plus an operator-facing
    ``detail`` that the UI shows in a dialog. The ``reset_to_initial`` key (``R``) fires
    the SAME motion over /ws/control, fire-and-forget.
    """
    rt = _runtime(request)
    # Orphaned-session watch (2026-09-09; 04-runtime §13.2): this motion is operator
    # activity and its two gated phases can outlast a short grace period, so it stamps
    # the countdown both before and after — the watch must never release the arms in
    # the middle of walking them home.
    rt.orphan_watch.note_activity("POST /api/session/return_home")
    try:
        return rt.manager.return_to_initial()
    finally:
        rt.orphan_watch.note_activity("return_home finished")


@router.post("/session/playback")
def post_session_playback(request: Request, body: EpisodePlaybackRequest) -> ReturnHomeResult:
    """Episode playback motions (2026-09-10; operator request, 04-runtime §10.8).

    ``goto_initial`` walks the arms to the episode's FIRST recorded frame and answers when
    they are there — SYNCHRONOUS like ``return_home``, because the dialog only enables
    **Playback** once this succeeded (replaying a trajectory from the wrong place is how an
    arm gets driven into something). It is an ordinary twin-planned, gated, interruptible
    profile motion: two separately planned phases (joints with the carriages held, then the
    carriages), one arm at a time in the planner's ``arm_order``, cancelled by any operator
    input.

    ``play`` replays the whole episode's MEASURED trajectory, also synchronously (a 37 s
    episode is a 37 s request): resampled onto the loop tick with one global time scale so
    the arms keep their recorded relative timing, every posture re-verified in the twin
    before anything is sent, then streamed through the gate. It refuses when an arm is not
    at frame 0 — with the distance, rather than quietly re-placing it.

    ``stop`` cancels a replay in flight and is the operator's only stop button here: the
    Welcome page never opens ``/ws/control``, so there is no key to cancel with.
    Idempotent.

    Never an error status for an operational refusal — the answer carries ``ok: false``
    plus a ``detail`` the dialog shows.
    """
    rt = _runtime(request)
    # The Welcome page never opens /ws/control, so a playback is the only operator
    # activity the orphaned-session watch would see. Stamp it (04-runtime §13.2).
    rt.orphan_watch.note_activity(f"POST /api/session/playback {body.action}")
    try:
        if body.action == "goto_initial":
            return rt.manager.episode_playback_goto_initial(body.repo_id, body.episode_id)
        if body.action == "play":
            return rt.manager.episode_playback_play(body.repo_id, body.episode_id)
        return rt.manager.episode_playback_stop()
    finally:
        rt.orphan_watch.note_activity("playback finished")


@router.delete("/session", status_code=204)
def delete_session(request: Request) -> Response:
    _runtime(request).manager.teardown()  # idempotent
    return Response(status_code=204)


# -- tracker calibration (13-tracker §4; phase-10) ---------------------------------------
# Session-less device management rides REST (binding supplement to 04-runtime §13.1:
# /ws/control nacks actions without a session and AckMsg carries no payload);
# progress is broadcast on telemetry as ``tracker.calibration``.
@router.get("/tracker/calibration")
def get_tracker_calibration(request: Request) -> TrackerCalibrationStatus:
    return _runtime(request).tracker_calibration.status()


@router.post("/tracker/calibration")
def post_tracker_calibration(
    request: Request, cmd: TrackerCalibrationCommand
) -> TrackerCalibrationStatus:
    try:
        return _runtime(request).tracker_calibration.command(cmd)
    except CalibrationError as e:  # illegal transition / precondition
        raise HTTPException(409, str(e)) from None


# -- arm maintenance (phase-09b/09c/09d; 04-runtime §13.1 / §15) ---------------------------
# Session-less device management rides REST (addendum above). Three of the four ops
# produce no motion: clear_errors = clean_error + clean_warn (monitor path, no enable);
# apply_backstops = the ArmConfig safety parameters (monitor path; 409 in a session);
# recover = the driver's user recovery incl. enable + servo mode + re-seed from the
# measured position (session path; 409 without one). home_rail (phase-09c) is THE
# ONE op that moves a mechanical part - the linear-track carriage drives to the
# zero end - so it is session-less only (409 "end the session first") and twin-gated:
# dry_run = the sweep verdict + pre_position plan only (zero writes); a sweep-clear
# posture homes synchronously here (<= 45 s, D3; status "done", 200); a posture that
# first needs the planned pre-positioning motion (phase-09d) starts a RailHomingJob
# and answers **202** with status "accepted" + job_id (progress on
# telemetry.hardware_monitor.arms[].maintenance, the final result at GET .../last);
# no rail-safe plan = status "refused" (ok false, 200). While a job runs every op is
# 409 "rail homing in progress". 200 whether or not ``ok`` otherwise.
@router.post("/hardware/arms/{arm_id}/maintenance")
def post_arm_maintenance(
    request: Request, arm_id: str, body: ArmMaintenanceRequest, response: Response
) -> ArmMaintenanceResult:
    rt = _runtime(request)
    who = request.client.host if request.client is not None else "unknown"
    label = f"{body.op}{' (dry run)' if body.dry_run else ''}"
    try:
        result = rt.arm_maintenance(arm_id, body.op, dry_run=body.dry_run)
    except KeyError:
        raise HTTPException(404, f"unknown hardware arm {arm_id!r}") from None
    except MaintenanceUnavailableError as e:
        logger.info("maintenance %s on arm %s from %s: refused - %s", label, arm_id, who, e)
        raise HTTPException(409, str(e)) from None
    if result.status == "accepted":
        response.status_code = 202
    logger.info(
        "maintenance %s on arm %s from %s via %s: %s - %s",
        label,
        arm_id,
        who,
        result.path,
        result.status if result.status != "done" else ("ok" if result.ok else "FAILED"),
        result.detail,
    )
    return result


@router.get("/hardware/arms/{arm_id}/maintenance/last")
def get_arm_maintenance_last(request: Request, arm_id: str) -> ArmMaintenanceResult:
    """phase-09d: the final ``ArmMaintenanceResult`` of the last ``home_rail`` on
    this arm (the same ``job_id`` as the 202 that started it); 404 until one exists."""
    try:
        result = _runtime(request).last_maintenance(arm_id)
    except KeyError:
        raise HTTPException(404, f"unknown hardware arm {arm_id!r}") from None
    if result is None:
        raise HTTPException(404, f"no maintenance result for {arm_id!r} yet")
    return result


# -- Online DAgger (phase-14; 15-online-dagger §7, §9) ------------------------------------------
# Session-less. The skill is package data served as markdown and as a gzip tarball rooted
# at mavis-online-dagger-trainer/ (install: curl -s .../skill.tgz | tar xz -C
# ~/.claude/skills/); the sessions listing reads every <online_dagger root>/*/session.json
# (the launch sheet's resume pill). The trainer contract itself rides the dora bus.
@router.get("/online_dagger/skill", response_class=PlainTextResponse)
def get_online_dagger_skill(request: Request) -> PlainTextResponse:
    from ..online_dagger import skill_markdown

    try:
        text = skill_markdown(_runtime(request).cfg.online_dagger.skill_dir)
    except OSError as e:
        raise HTTPException(404, f"Online DAgger skill not found: {e}") from None
    return PlainTextResponse(text, media_type="text/markdown; charset=utf-8")


@router.get("/online_dagger/skill.tgz")
def get_online_dagger_skill_tgz(request: Request) -> Response:
    from ..online_dagger import SKILL_NAME, skill_tarball

    try:
        data = skill_tarball(_runtime(request).cfg.online_dagger.skill_dir)
    except OSError as e:
        raise HTTPException(404, f"Online DAgger skill not found: {e}") from None
    return Response(
        content=data,
        media_type="application/gzip",
        headers={"Content-Disposition": f'attachment; filename="{SKILL_NAME}.tgz"'},
    )


@router.get("/online_dagger/sessions")
def list_online_dagger_sessions(request: Request) -> list[OnlineDaggerSessionInfo]:
    return _runtime(request).manager.online_dagger_sessions()


# -- external interface over dora (phase-12; 14-dora §2.6, §9) -----------------------------
# Connection facts for foreign clients (same-host policy nodes, remote viewers). The auth
# token is NEVER served here (read it from <var_dir>/.dora-token on the lab host). ``join``
# is the explicit rescan a remote operator triggers after starting their daemon: 202 +
# the current facts (idempotent while the daemon is already registered), 404 when the
# machine is not in ``dora.machines``.
@router.get("/dora")
def get_dora(request: Request) -> DoraInfo:
    return _runtime(request).dora_info()


@router.post("/dora/machines/{machine_id}/join", status_code=202)
def post_dora_join(request: Request, machine_id: str, response: Response) -> DoraInfo:
    rt = _runtime(request)
    if not rt.dora_join(machine_id):
        raise HTTPException(404, f"machine {machine_id!r} is not in dora.machines")
    response.status_code = 202
    return rt.dora_info()


__all__ = ["router"]
