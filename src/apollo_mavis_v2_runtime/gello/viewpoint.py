"""``ViewpointSource`` — the Perception Arm's external viewpoint node in a gello session
(16-gello D5 / §7).

The viewpoint node rides the EXISTING external policy path (14-dora §5 / §6: the
``ExternalPolicyHub`` caches the node's ``policy_spec`` heartbeat and routes its
``policy_action`` to the attached ``ExternalPolicySource``) scoped to ONE arm: the
session announces ``external_arms: ["view"]`` and a view-only ``action_names`` layout
(``view_ee.dx … view_gripper.pos, view_rail.dpos``), and this object decides, every tick,
whether an :class:`~apollo_mavis_v2_runtime.dora_bridge.policy_source.ExternalPolicySource`
built over ``arms_meta = [("view", has_rail)]`` is live:

* ``poll(now, allow_attach)``: with a FRESH hub spec that is COMPATIBLE (``action_space ==
  "delta_ee"``, ``action_frame`` == the session's view frame, ``action_names`` == the view
  layout, ``state_names`` a subset of the view state layout) and no live source, build one
  and ``start()`` it (``policy_reset{session_start}``) — but only while ``allow_attach``
  (the loop passes "session RUNNING and no planned motion owns the Perception Arm": during
  BRINGUP / the launch motion / ``R`` / Go to profile / the exit return the announce and
  ``obs_state`` flow, nothing a node sends is counted late or advances a watermark, and
  the first attach happens once the arm has reached its posture). A stale (``>
  spec_stale_s``) or incompatible spec, or ``allow_attach`` false, ``stop()``s a live
  source (``policy_reset{session_stop}``) and the arm holds. Incompatible specs are
  logged once per ``policy_id`` and shown in the panel (``detail``).
* ``latest()`` / ``staleness_scale()`` / ``period`` forward to the live source (``None``
  / ``0.0`` / the default period without one); ``pause()`` / ``resume()`` (the loop's NaN
  three-strike, cleared by the operator's Resume) and ``drop_and_requery()`` too. While
  paused ``telemetry()`` says ``paused: True`` with the reason as ``detail`` (``poll()``
  does not overwrite it with "attached"), and a detach drops the pause so a restarted node
  starts clean (2026-09-09 review: the pause was invisible on the wire and survived a
  node restart).
* ``viewpoint == "hold"``: the loop never polls; the object only reports telemetry.

``compatibility(ann, view_frame, has_rail)`` is the shared check the launch uses for
``viewpoint: external`` (409 ``no external viewpoint node attached (…)`` / ``viewpoint node
action_names … != view layout …``, 16-gello §5.1 item 5). Everything here runs on the
control thread except ``stop()`` at teardown; the ``ExternalPolicySource`` itself is
thread-safe (its bus-thread writes are reference swaps under a lock).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from apollo_mavis_v2_core.protocol.external import PolicySpecAnnounce

from ..recorder.features import arm_action_names, arm_state_names
from . import VIEW_ARM_ID

logger = logging.getLogger(__name__)

NO_HUB_DETAIL = "no dora bridge (dora.enabled is false) - holding the GELLO posture"
HOLD_DETAIL = "holding the GELLO posture"
WAITING_DETAIL = "waiting for a viewpoint node (no fresh policy_spec)"
PAUSED_DETAIL = "paused after 3 NaN actions - press Resume"


def view_action_names(has_rail: bool, arm_id: str = VIEW_ARM_ID) -> list[str]:
    """The view-only ``delta_ee`` action layout a gello session announces (16-gello D5)."""
    return arm_action_names(arm_id, has_rail, "delta_ee")


def view_state_names(has_rail: bool, arm_id: str = VIEW_ARM_ID) -> list[str]:
    return arm_state_names(arm_id, has_rail)


def compatibility(
    ann: PolicySpecAnnounce, view_frame: str, has_rail: bool, arm_id: str = VIEW_ARM_ID
) -> str | None:
    """None when the announced spec may drive the Perception Arm of a gello session,
    else the operator-facing reason (16-gello §7)."""
    s = ann.spec
    if s.action_space != "delta_ee":
        return f"viewpoint node action_space {s.action_space!r} != 'delta_ee'"
    if s.action_frame != view_frame:
        return (
            f"viewpoint node action_frame {s.action_frame!r} != the session's view frame "
            f"{view_frame!r}"
        )
    want = view_action_names(has_rail, arm_id)
    if list(s.action_names) != want:
        return f"viewpoint node action_names {list(s.action_names)} != view layout {want}"
    allowed = set(view_state_names(has_rail, arm_id))
    extra = sorted(set(s.state_names) - allowed)
    if extra:
        return f"viewpoint node state_names outside the view state layout: {extra}"
    return None


class ViewpointSource:
    """The Perception Arm's viewpoint node in a gello session (module docstring)."""

    def __init__(
        self,
        hub: Any,  # ExternalPolicyHub | None (None = no dora bridge: never attaches)
        publisher: Any,  # SnapshotPublisher (observation_t_mono / publish_event) | None
        *,
        session_id: str,
        view_frame: str,
        mode: str,  # GelloViewpointMode: auto | external | hold
        cfg: Any,  # DoraPolicyConfig (spec_stale_s, max_obs_age_s)
        rate_hz_default: float = 15.0,
        chunk_dt_s: float | None = None,
        has_rail: bool = True,
        arm_id: str = VIEW_ARM_ID,
        clock: Callable[[], float] = time.monotonic,
        source_factory: Callable[..., Any] | None = None,  # tests: replaces ExternalPolicySource
    ) -> None:
        self.hub = hub
        self.publisher = publisher
        self.session_id = session_id
        self.view_frame = view_frame
        self.mode = mode
        self.cfg = cfg
        self.rate_hz_default = float(rate_hz_default)
        self.chunk_dt_s = chunk_dt_s
        self.has_rail = bool(has_rail)
        self.arm_id = arm_id
        self._clock = clock
        self._factory = source_factory
        self.arms_meta: list[tuple[str, bool]] = [(arm_id, self.has_rail)]
        self._source: Any = None
        self._policy_id: str | None = None
        self._detail: str = HOLD_DETAIL if mode == "hold" else WAITING_DETAIL
        self._incompatible_logged: set[str] = set()
        self._paused = False
        self.attaches = 0
        self.detaches = 0

    # -- readouts -----------------------------------------------------------------------------
    @property
    def attached(self) -> bool:
        return self._source is not None

    @property
    def source(self) -> Any:
        return self._source

    @property
    def policy_id(self) -> str | None:
        return self._policy_id if self._source is not None else None

    @property
    def detail(self) -> str:
        return self._detail

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def period(self) -> float:
        src = self._source
        return float(src.period) if src is not None else 1.0 / self.rate_hz_default

    def telemetry(self) -> dict[str, Any]:
        """``GelloTelemetry.viewpoint`` (16-gello §8.3; ``paused`` since the 2026-09-09
        review)."""
        return {
            "mode": self.mode,
            "attached": self.attached,
            "policy_id": self.policy_id,
            "detail": self._detail,
            "paused": self._paused,
        }

    def current_version(self) -> int | None:
        src = self._source
        return int(src.current_version()) if src is not None else None

    # -- the tick -----------------------------------------------------------------------------
    def poll(self, now: float, allow_attach: bool = True) -> None:
        """Attach / detach against the hub's cached spec (module docstring)."""
        if self.mode == "hold":
            self._detail = HOLD_DETAIL
            return
        hub = self.hub
        if hub is None:
            self._detail = NO_HUB_DETAIL
            return
        ann = hub.spec(now)
        bridge_attached = bool(getattr(getattr(hub, "bridge", None), "attached", True))
        if ann is None or not bridge_attached:
            if self._source is not None:
                self._detach("policy_spec heartbeat stale" if ann is None else "bridge detached")
            self._detail = WAITING_DETAIL
            return
        why = compatibility(ann, self.view_frame, self.has_rail, self.arm_id)
        if why is not None:
            if ann.policy_id not in self._incompatible_logged:
                self._incompatible_logged.add(ann.policy_id)
                logger.warning("viewpoint node %r ignored: %s", ann.policy_id, why)
            if self._source is not None:
                self._detach(why)
            self._detail = f"node {ann.policy_id!r} ignored: {why}"
            return
        if not allow_attach:
            if self._source is not None:
                self._detach("planned motion owns the Perception Arm")
            self._detail = f"node {ann.policy_id!r} compatible - attaches once the arm is free"
            return
        if self._source is None:
            self._attach(ann)
        if not self._paused:  # a NaN pause keeps its reason on the wire until Resume
            self._detail = f"external node {ann.policy_id!r} attached"

    def _attach(self, ann: PolicySpecAnnounce) -> None:
        from ..dora_bridge.policy_source import ExternalPolicySource, spec_from_announce

        factory = self._factory or ExternalPolicySource
        src = factory(
            self.hub,
            self.publisher,
            session_id=self.session_id,
            spec=spec_from_announce(ann),
            policy_id=ann.policy_id,
            arms_meta=list(self.arms_meta),
            rate_hz=float(ann.rate_hz) if ann.rate_hz else self.rate_hz_default,
            chunk_dt_s=ann.chunk_dt_s,
            cfg=self.cfg,
        )
        src.start()  # publishes policy_reset{session_start}
        if self._paused:
            src.pause()
        self._source = src
        self._policy_id = ann.policy_id
        self.attaches += 1
        logger.info("gello viewpoint: node %r attached (view block only)", ann.policy_id)

    def _detach(self, why: str) -> None:
        src = self._source
        self._source = None
        self._paused = False  # a re-attached / restarted node starts clean
        self.detaches += 1
        if src is not None:
            try:
                src.stop()  # publishes policy_reset{session_stop}
            except Exception:  # noqa: BLE001 - the bus never breaks the tick
                logger.exception("gello viewpoint: source stop failed")
        logger.info("gello viewpoint: node %r detached (%s) - holding", self._policy_id, why)

    # -- PolicySource-like surface (the loop reads these) -------------------------------------
    def latest(self) -> tuple[Any, float]:
        src = self._source
        if src is None:
            return None, -1e9
        return src.latest()

    def staleness_scale(self, now: float) -> float:
        src = self._source
        return float(src.staleness_scale(now)) if src is not None else 0.0

    def drop_and_requery(self, reason: str = "handback") -> None:
        src = self._source
        if src is not None:
            src.drop_and_requery(reason)

    def pause(self, detail: str = PAUSED_DETAIL) -> None:
        """The loop's NaN three-strike: hold until the operator's Resume (16-gello §6.3);
        ``detail`` is what the panel shows meanwhile."""
        self._paused = True
        self._detail = detail
        src = self._source
        if src is not None:
            src.pause()

    def resume(self) -> None:
        self._paused = False
        src = self._source
        if src is not None:
            src.resume()
            self._detail = f"external node {self._policy_id!r} attached"

    def stop(self) -> None:
        """Teardown: detach a live source (``policy_reset{session_stop}``)."""
        if self._source is not None:
            self._detach("session teardown")


__all__ = [
    "HOLD_DETAIL",
    "NO_HUB_DETAIL",
    "PAUSED_DETAIL",
    "WAITING_DETAIL",
    "ViewpointSource",
    "compatibility",
    "view_action_names",
    "view_state_names",
]
