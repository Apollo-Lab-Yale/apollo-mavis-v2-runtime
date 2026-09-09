"""GELLO launch check + preview (16-gello §5.1 / §5.4; ``POST /api/gello/preview``).

One :class:`GelloPreviewService` per runtime keeps a kitchen ``DigitalTwin`` per
``(kind, scene)`` — built lazily with the SAME ``SceneOverrides`` the sessions use
(microphones per kind; the hardware base poses), the same inflation / allowed pairs, and
the D7 grasp whitelist (the handles against the Manipulation Arm's gripper) — and runs
the launch check of §5.1 on it, session-less:

1. the leader: a fresh, valid, calibrated sample (else ``no_leader`` / ``not_calibrated``);
2. ``q_goal["grip"] = unwrap(sample.q, q_meas)`` + the CURRENT rail (the monitor sample on
   hardware, the launched scene's keyframe in sim — what ``SimWorkcell.start`` resets to)
   -> the joint-limit table less 0.5° (``joint_limit``);
3. ``q_goal["view"] = gello.view_posture_rad + [gello.view_rail_m]``;
4. ``DigitalTwin.check_config_violations`` at the full goal (both arms at their goals) ->
   the violating pairs, tightest first (``collision``), else ``clear``.

:meth:`GelloPreviewService.evaluate` is that check (the manager's ``_check_gello`` runs
it before any side effect and turns the verdict into the §9.2 409s);
:meth:`GelloPreviewService.preview` adds the PNG: ``cam_kitchen`` (``cam_front`` when the
scene lacks it) rendered on a DEDICATED render thread that owns its ``mujoco.Renderer``
instances (GL contexts are thread-affine; ``MUJOCO_GL=egl`` as everywhere in the runtime)
from a PRIVATE, un-inflated model (``spec.compile()`` of the same built scene: the twin's
inflation lives in ``geom_gap`` and is invisible anyway) with the colliding pairs' geoms
tinted red — ``geom_matid = -1`` first, since ``geom_rgba`` is ignored while a material is
assigned (textured tag plates). The endpoint never moves anything and works with no
session and no viewpoint node; ``preview`` is serialised (the twin's ``MjData`` is not
thread-safe) and the sheet polls it at 2 Hz.
"""

from __future__ import annotations

import base64
import logging
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from apollo_mavis_v2_core.protocol import GelloPairInfo, GelloPreviewRequest, GelloPreviewResult

from ..config import RuntimeConfig
from ..errors import GelloUnavailableError
from . import FOLLOWER_ARM_ID, VIEW_ARM_ID
from .engage import (
    JointLimitViolation,
    clip_to_joint_limits,
    joint_limit_violation,
    unwrap_to_reference,
)

logger = logging.getLogger(__name__)

PREVIEW_CAMERA = "cam_kitchen"
FALLBACK_CAMERAS = ("cam_front", "cam_top")
DEFAULT_IMAGE_SIZE = (640, 480)  # (width, height)
HIGHLIGHT_RGBA = (1.0, 0.16, 0.10, 1.0)
RENDER_TIMEOUT_S = 5.0
STOP_JOIN_TIMEOUT_S = 5.0


def collision_text(pairs: list[tuple[tuple[str, str], float]]) -> str:
    """``"<a> / <b> at <mm> mm[, …]"`` — every pair, tightest first (16-gello §5.1 item 4)."""
    return ", ".join(f"{a} / {b} at {d * 1e3:.1f} mm" for (a, b), d in pairs)


def launch_collision_409(pairs: list[tuple[tuple[str, str], float]]) -> str:
    """The launch refusal (16-gello D4 / §9.2)."""
    return f"GELLO posture collides: {collision_text(pairs)} - move GELLO and retry"


def launch_joint_limit_409(v: JointLimitViolation) -> str:
    return f"GELLO posture outside the Manipulation Arm's joint limits ({v.describe()})"


@dataclass
class PreviewTwin:
    """One cached kitchen twin per (kind, scene): the gate-grade ``DigitalTwin`` for the
    check and a private un-inflated model for the render thread."""

    kind: str
    scene_id: str
    twin: Any  # DigitalTwin
    render_model: Any  # mujoco.MjModel (private copy of the scene: never inflated)
    camera: str  # the camera the PNG is rendered from
    key_qpos: np.ndarray
    label_geoms: dict[str, np.ndarray] = field(default_factory=dict)  # label -> render geom ids

    def keyframe_q(self, arm_id: str) -> np.ndarray:
        return np.array(self.key_qpos[self.twin.addr[arm_id].qpos_adr], dtype=np.float64)

    def qpos_for(self, q_goal: dict[str, Any]) -> np.ndarray:
        qpos = np.array(self.key_qpos, dtype=np.float64)
        for arm_id, q in q_goal.items():
            adr = self.twin.addr[arm_id].qpos_adr
            qpos[adr] = np.asarray(q, dtype=np.float64)[: len(adr)]
        return qpos


@dataclass(frozen=True)
class Evaluation:
    """The §5.1 verdict without the picture."""

    status: str  # GelloPreviewStatus
    detail: str
    pairs: list[tuple[tuple[str, str], float]]
    q_goal: dict[str, list[float]]
    leader_q: list[float] | None
    joint_limit: JointLimitViolation | None = None

    @property
    def ok(self) -> bool:
        return self.status == "clear"


class _RenderWorker:
    """The render thread: owns every ``mujoco.Renderer`` (created and closed in-thread)."""

    def __init__(self) -> None:
        self._jobs: queue.Queue[tuple[Callable[[], Any], Future] | None] = queue.Queue()
        self._renderers: dict[tuple[int, int, int], Any] = {}
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def submit(self, fn: Callable[[], Any]) -> Future:
        fut: Future = Future()
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run, name="gello-preview-render", daemon=True
                )
                self._thread.start()
        self._jobs.put((fn, fut))
        return fut

    def renderer(self, model: Any, height: int, width: int) -> Any:
        """Render-thread only."""
        import mujoco

        key = (id(model), int(height), int(width))
        r = self._renderers.get(key)
        if r is None:
            r = mujoco.Renderer(model, height=int(height), width=int(width))
            self._renderers[key] = r
        return r

    def _run(self) -> None:
        try:
            while True:
                item = self._jobs.get()
                if item is None:
                    return
                fn, fut = item
                try:
                    fut.set_result(fn())
                except BaseException as e:  # noqa: BLE001 - reported to the caller
                    fut.set_exception(e)
        finally:
            for r in self._renderers.values():
                try:
                    r.close()
                except Exception:  # noqa: BLE001
                    logger.warning("gello preview renderer close failed", exc_info=True)
            self._renderers.clear()

    def close(self, timeout: float = STOP_JOIN_TIMEOUT_S) -> None:
        with self._lock:
            t = self._thread
            self._thread = None
        if t is not None and t.is_alive():
            self._jobs.put(None)
            t.join(timeout=timeout)


class GelloPreviewService:
    """Session-less GELLO launch check + preview render (module docstring).

    ``hardware_arm_q()`` -> ``(q8 | None, why)``: the Manipulation Arm's CURRENT joints +
    rail from the read-only monitor (the Runtime wires it); sim uses the launched scene's
    keyframe. ``render`` False skips the picture (unit tests without a GL context).
    """

    def __init__(
        self,
        cfg: RuntimeConfig,
        reader: Any,  # GelloReader-like: status(now), fresh_sample(now)
        *,
        hardware_arm_q: Callable[[], tuple[np.ndarray | None, str]] | None = None,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        render: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self.reader = reader
        self.hardware_arm_q = hardware_arm_q
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.render_enabled = bool(render)
        self._clock = clock
        self._twins: dict[tuple[str, str], PreviewTwin] = {}
        self._lock = threading.RLock()
        self._worker = _RenderWorker()
        self.previews = 0

    # -- twins ---------------------------------------------------------------------------------
    def twin_for(self, kind: str, scene_id: str, wc: Any) -> PreviewTwin:
        """The cached kitchen twin of ``(kind, scene_id)`` (built on first use, ~0.5 s);
        raises the sim's scene errors when it cannot be built."""
        key = (kind, scene_id)
        with self._lock:
            pt = self._twins.get(key)
            if pt is not None:
                return pt
            pt = self._build(kind, scene_id, wc)
            self._twins[key] = pt
            return pt

    def _build(self, kind: str, scene_id: str, wc: Any) -> PreviewTwin:
        import mujoco
        from apollo_mavis_v2_sim import REGISTRY, DigitalTwin, SceneOverrides

        from ..streams.twin_overlay import base_pose_overrides

        known = {a.id for a in REGISTRY.descriptor(scene_id).arms}
        mics = {a.id: bool(a.microphone) for a in wc.arms if a.id in known}
        base = (
            {k: v for k, v in base_pose_overrides(wc).items() if k in known}
            if kind == "hardware"
            else {}
        )
        built = REGISTRY.build(scene_id, SceneOverrides(microphones=mics, base_pose=base))
        missing = [a for a in (FOLLOWER_ARM_ID, VIEW_ARM_ID) if a not in built.meta.arm_ids]
        if missing:
            raise ValueError(
                f"scene {scene_id!r} lacks the arms {missing} a GELLO session needs "
                f"(has {list(built.meta.arm_ids)})"
            )
        safety = wc.safety
        # the render model FIRST: DigitalTwin takes ownership of built.model (inflation)
        spec = built.spec
        w, h = self.image_size
        spec.visual.global_.offwidth = max(int(spec.visual.global_.offwidth), w)
        spec.visual.global_.offheight = max(int(spec.visual.global_.offheight), h)
        render_model = spec.compile()
        twin = DigitalTwin(
            built,
            inflation_m=safety.geom_inflation_m,
            allowed_pairs_extra=safety.allowed_pairs_extra,
            hysteresis_m=safety.hysteresis_m,
        )
        graspable = list(getattr(built.meta, "graspable", ()) or ())
        if graspable and twin.addr[FOLLOWER_ARM_ID].has_gripper:
            twin.set_grasp_whitelist(FOLLOWER_ARM_ID, graspable)  # 16-gello D7
        camera = PREVIEW_CAMERA
        cams = tuple(built.meta.cameras)
        if camera not in cams:
            camera = next((c for c in FALLBACK_CAMERAS if c in cams), cams[0] if cams else "")
        key_qpos = np.array(
            twin.model.key_qpos[0] if twin.model.nkey > 0 else twin.model.qpos0, dtype=np.float64
        )
        label_geoms: dict[str, np.ndarray] = {}
        for label, gids in twin._geoms_of_label.items():
            ids = []
            for g in gids:
                name = mujoco.mj_id2name(twin.model, mujoco.mjtObj.mjOBJ_GEOM, int(g))
                rid = (
                    mujoco.mj_name2id(render_model, mujoco.mjtObj.mjOBJ_GEOM, name) if name else -1
                )
                ids.append(int(rid) if rid >= 0 else int(g))
            label_geoms[label] = np.asarray(ids, dtype=np.intp)
        logger.info(
            "gello preview twin %r (%s): %d monitored pairs, camera %s, graspable %s",
            scene_id,
            kind,
            len(twin.monitored_pairs),
            camera or "free",
            graspable or "none",
        )
        return PreviewTwin(kind, scene_id, twin, render_model, camera, key_qpos, label_geoms)

    # -- the §5.1 check ------------------------------------------------------------------------
    def evaluate(
        self,
        kind: str,
        scene_id: str,
        wc: Any,
        *,
        sample: Any = None,
        q_meas: np.ndarray | None = None,
        now: float | None = None,
    ) -> tuple[Evaluation, PreviewTwin | None]:
        """The launch check (module docstring). ``sample`` / ``q_meas`` default to the
        reader's fresh sample and the kind's current Manipulation Arm posture; the
        ``PreviewTwin`` is returned for the render (None on ``scene_error``)."""
        now = self._clock() if now is None else now
        g = self.cfg.gello
        view_goal = [float(x) for x in g.view_posture_rad] + [float(g.view_rail_m)]
        try:
            pt = self.twin_for(kind, scene_id, wc)
        except Exception as e:  # noqa: BLE001 - TwinAuditError / SceneNotFoundError / compile
            return (
                Evaluation(
                    "scene_error",
                    f"twin scene {scene_id!r} unavailable: {type(e).__name__}: {e}",
                    [],
                    {VIEW_ARM_ID: view_goal},
                    None,
                ),
                None,
            )
        st = self.reader.status(now)
        if not st.calibrated:
            return (
                Evaluation(
                    "not_calibrated",
                    "GELLO joint offsets unknown - POST /api/gello/calibrate {op: match_arm} "
                    f"first (leader {st.status}{': ' + st.detail if st.detail else ''})",
                    [],
                    {VIEW_ARM_ID: view_goal},
                    None,
                ),
                pt,
            )
        if sample is None:
            sample = self.reader.fresh_sample(now)
        if sample is None or not getattr(sample, "valid", False):
            return (
                Evaluation(
                    "no_leader",
                    f"no fresh GELLO leader sample (status {st.status}"
                    f"{': ' + st.detail if st.detail else ''})",
                    [],
                    {VIEW_ARM_ID: view_goal},
                    None,
                ),
                pt,
            )
        if q_meas is None:
            if kind == "hardware":
                q_meas, why = (
                    self.hardware_arm_q() if self.hardware_arm_q is not None else (None, "")
                )
                if q_meas is None:
                    return (
                        Evaluation(
                            "no_workcell",
                            why or "no monitor sample of the Manipulation Arm",
                            [],
                            {VIEW_ARM_ID: view_goal},
                            None,
                        ),
                        pt,
                    )
            else:
                q_meas = pt.keyframe_q(FOLLOWER_ARM_ID)
        q_meas = np.asarray(q_meas, dtype=np.float64)
        q_unwrapped, _k = unwrap_to_reference(sample.q, q_meas[:7])
        rail = [float(q_meas[7])] if q_meas.shape[0] > 7 else []
        leader_q = [float(x) for x in q_unwrapped]
        violation = joint_limit_violation(q_unwrapped)
        grip_goal = [float(x) for x in clip_to_joint_limits(q_unwrapped)] + rail
        q_goal = {FOLLOWER_ARM_ID: grip_goal, VIEW_ARM_ID: view_goal}
        if violation is not None:
            return (
                Evaluation(
                    "joint_limit",
                    launch_joint_limit_409(violation),
                    [],
                    q_goal,
                    leader_q,
                    violation,
                ),
                pt,
            )
        with self._lock:
            raw = pt.twin.check_config_violations(pt.qpos_for(q_goal))
        by_pair: dict[tuple[str, str], float] = {}
        for pair, dist in raw:
            key = tuple(sorted(pair))
            by_pair[key] = min(float(dist), by_pair.get(key, np.inf))
        pairs = sorted(by_pair.items(), key=lambda kv: kv[1])
        if pairs:
            ev = Evaluation(
                "collision", f"collides: {collision_text(pairs)}", pairs, q_goal, leader_q
            )
            return ev, pt
        return Evaluation("clear", "clear", [], q_goal, leader_q), pt

    # -- the endpoint ---------------------------------------------------------------------------
    def preview(self, req: GelloPreviewRequest) -> GelloPreviewResult:
        """``POST /api/gello/preview``: the check + the PNG. Raises
        :class:`GelloUnavailableError` (409) only when the runtime has no workcell of
        ``req.kind``; every other outcome is a 200 with the status."""
        wc = self.cfg.workcell_config(req.kind)
        if wc is None:
            raise GelloUnavailableError(f"no {req.kind} workcell is configured")
        scene_id = req.scene or self.cfg.gello.scene_id
        ev, pt = self.evaluate(req.kind, scene_id, wc)
        image: str | None = None
        camera = pt.camera if pt is not None else PREVIEW_CAMERA
        detail = ev.detail
        if pt is not None and self.render_enabled and ev.q_goal.get(FOLLOWER_ARM_ID) is not None:
            try:
                image = self._render_png_b64(pt, ev)
            except Exception as e:  # noqa: BLE001 - the verdict is the payload, the PNG extra
                logger.warning("gello preview render failed: %r", e)
                detail = f"{detail} (render failed: {type(e).__name__})"
        self.previews += 1
        return GelloPreviewResult(
            status=ev.status,  # type: ignore[arg-type]
            ok=ev.ok,
            detail=detail,
            pairs=[GelloPairInfo(a=a, b=b, dist_m=float(d)) for (a, b), d in ev.pairs],
            q_goal=ev.q_goal,
            leader_q=ev.leader_q,
            image_png_b64=image,
            camera=camera or PREVIEW_CAMERA,
        )

    def _render_png_b64(self, pt: PreviewTwin, ev: Evaluation) -> str | None:
        qpos = pt.qpos_for(ev.q_goal)
        highlight: list[int] = []
        for (a, b), _d in ev.pairs:
            for label in (a, b):
                highlight.extend(int(g) for g in pt.label_geoms.get(label, ()))
        ids = np.asarray(sorted(set(highlight)), dtype=np.intp)
        w, h = self.image_size
        fut = self._worker.submit(lambda: self._render(pt, qpos, ids, h, w))
        png = fut.result(timeout=RENDER_TIMEOUT_S)
        return base64.b64encode(png).decode("ascii") if png else None

    def _render(self, pt: PreviewTwin, qpos: np.ndarray, ids: np.ndarray, h: int, w: int) -> bytes:
        """Render-thread only: pose, tint, render, encode."""
        import cv2
        import mujoco

        model = pt.render_model
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        saved_mat = model.geom_matid[ids].copy() if ids.size else None
        saved_rgba = model.geom_rgba[ids].copy() if ids.size else None
        if ids.size:
            model.geom_matid[ids] = -1  # rgba is ignored while a material is assigned
            model.geom_rgba[ids] = np.asarray(HIGHLIGHT_RGBA, dtype=np.float32)
        try:
            renderer = self._worker.renderer(model, h, w)
            cam = pt.camera if pt.camera else -1
            renderer.update_scene(data, camera=cam)
            rgb = renderer.render()
        finally:
            if ids.size:
                model.geom_matid[ids] = saved_mat
                model.geom_rgba[ids] = saved_rgba
        ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("PNG encode failed")
        return buf.tobytes()

    def close(self) -> None:
        self._worker.close()


__all__ = [
    "DEFAULT_IMAGE_SIZE",
    "FALLBACK_CAMERAS",
    "HIGHLIGHT_RGBA",
    "PREVIEW_CAMERA",
    "Evaluation",
    "GelloPreviewService",
    "PreviewTwin",
    "collision_text",
    "launch_collision_409",
    "launch_joint_limit_409",
]
