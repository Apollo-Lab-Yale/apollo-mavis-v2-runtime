"""Orphaned-session watch (2026-09-09 evening; 04-runtime §13.2 "orphaned session").

A session nobody can drive is a hazard, not a feature: on 2026-09-09 a hardware
Teleop session started at 22:51 was still ``running`` at 23:16 because the Cockpit
tab had been closed with the browser's Back button instead of **End session**. Both
control boxes stayed enabled with the arms holding their posture, the read-only
hardware monitor stayed paused (so every Welcome-page arm gate read ``monitor_off``
and all four launch cards were disabled), and nothing on the machine would ever have
released them.

So the runtime ends such a session ITSELF. The one liveness signal is the
**controller** ``/ws/control`` connection (04-runtime §13.2: the first connection is
the controller, later ones are read-only observers): that socket is what carries the
25 Hz key heartbeat, the actions and the deadman, so a session without it cannot be
driven by anybody. When it has been absent for ``control.orphan_session_grace_s``
— and no session-scoped REST call has stamped :meth:`note_activity` meanwhile — the
watch calls :meth:`SessionManager.teardown`.

Three properties worth stating out loud:

* **The end produces NO motion.** It is exactly the ``DELETE /api/session`` path:
  the drivers hand the arms back stopped with the brakes engaged, where they stand,
  and the tracks keep their homed state. It deliberately does NOT run the Cockpit's
  return-to-initial-condition first — that is a twin-planned motion, and with nobody
  in the room and the twin not yet modelling the furniture (03-sim §4.5) an
  unattended replan is the last thing we want.
* **A reload is not an orphan.** The grace period is what separates "the tab was
  closed for good" from "the operator pressed F5": a reloaded Cockpit re-opens
  ``/ws/control`` within a second and the countdown resets.
* **An open episode is discarded**, because ``teardown()`` discards it (04-runtime
  §10.4). An episode being recorded by nobody is not data worth keeping the arms
  live for; the notice and the log line both say it happened.

What ended the session is published as :class:`SessionAutoEndNotice` on
``telemetry.session.auto_ended`` and survives until the next session starts, so the
Welcome page can tell the operator that the arms were released while nobody was
watching — otherwise the cell would silently be in a different state than the person
who walks back to it expects.

Observers do not count as attendance on purpose: an observer tab cannot send keys or
actions, so a session with only observers is just as undrivable as one with no
sockets at all.

**A session that has NEVER had a controller is never ended.** The watch fires only on
a session whose controller connected and then went away — which is exactly the case
that bit us, and it keeps a deliberately REST-driven session (a test harness, a
script, a policy bring-up that has not opened its socket yet) safe from a background
thread tearing it down. The cost is one narrow gap: a ``POST /api/session`` whose
browser died before the Cockpit ever mounted stays up. That one is visible instead —
``telemetry.session.mode`` now tells the Welcome page a session is running, so the
page offers its Cockpit route (and thus **End session**) rather than four disabled
launch cards.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from apollo_mavis_v2_core.protocol import SessionAutoEndNotice

from .types import SessionState

logger = logging.getLogger(__name__)

#: How often the watch thread re-evaluates. Far finer than any sane grace period, so
#: the reason text can quote the configured grace rather than the measured age.
POLL_S = 0.5


def _fmt_s(seconds: float) -> str:
    """``30.0`` -> ``"30 s"``, ``7.5`` -> ``"7.5 s"`` (operator-facing text)."""
    return f"{seconds:.0f} s" if float(seconds).is_integer() else f"{seconds:g} s"


@dataclass(frozen=True)
class _Watched:
    """Identity of the session the watch is currently counting down on."""

    session_id: str
    mode: str
    kind: str


class OrphanSessionWatch:
    """Ends a session with no controller ``/ws/control`` connection (§13.2).

    ``manager`` is duck-typed to the slice used here (``session``, ``state``,
    ``teardown()``, ``open_episode()``) so tests can script it without building a
    workcell. ``attended`` is read live — the Runtime passes
    ``lambda: self.controller_connected``, the same flag ``server/ws_control``
    maintains, so there is no second copy of the connection state to keep in sync.

    ``grace_s <= 0`` disables the watch completely: :meth:`start` spawns no thread
    and :meth:`poll` never ends anything (the runtime's own test suites and any
    REST-only driver keep working).
    """

    def __init__(
        self,
        manager,
        grace_s: float,
        attended: Callable[[], bool],
        *,
        poll_s: float = POLL_S,
        monotonic: Callable[[], float] = time.monotonic,
        now_iso: Callable[[], str] | None = None,
    ) -> None:
        self.manager = manager
        self.grace_s = float(grace_s)
        self._attended = attended
        self._poll_s = float(poll_s)
        self._mono = monotonic
        self._now_iso = now_iso or (lambda: datetime.now(timezone.utc).isoformat())
        self._lock = threading.Lock()
        self._watched: _Watched | None = None
        self._attended_at: float = monotonic()
        self._had_controller = False  # per watched session; see the module docstring
        self._notice: SessionAutoEndNotice | None = None
        self._ended = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # -- state the rest of the runtime reads -----------------------------------
    @property
    def notice(self) -> SessionAutoEndNotice | None:
        """``telemetry.session.auto_ended``: why the LAST session ended without an
        operator click; ``None`` when the last one ended by DELETE or none ran."""
        with self._lock:
            return self._notice

    @property
    def ended_count(self) -> int:
        """How many sessions this process has ended as orphans (tests / logs)."""
        with self._lock:
            return self._ended

    def note_activity(self, what: str = "") -> None:
        """Reset the countdown: an operator just did something session-scoped over
        REST rather than over ``/ws/control``.

        Stamped by ``POST /api/session`` (a session that has just been created has
        not had time to open its control socket) and by
        ``POST /api/session/return_home``, whose synchronous motion can outlast a short
        grace period. Harmless to call without a session.

        This resets the countdown; it does NOT mark the session as having HAD a
        controller (:meth:`note_controller_connected` does), so a REST-only session
        stays exempt however often it is stamped.
        """
        with self._lock:
            self._attended_at = self._mono()
        if what:
            logger.debug("orphan watch: countdown reset by %s", what)

    def note_controller_connected(self) -> None:
        """``server/ws_control`` accepted a CONTROLLER connection (never an observer).

        Both arms the countdown and records that this session has been driven from a
        Cockpit, which is the precondition for ever ending it. Without this the watch
        would miss a controller that connected and dropped again inside one poll period
        (:data:`POLL_S`) — rare, but it would leave a session that really was abandoned
        exempt for the rest of its life.
        """
        with self._lock:
            self._attended_at = self._mono()
            self._had_controller = True

    # -- the watch itself ------------------------------------------------------
    def poll(self) -> SessionAutoEndNotice | None:
        """One evaluation; returns the notice iff this call ended a session.

        Called from the watch thread, and directly from the tests with an injected
        clock. The teardown runs OUTSIDE ``_lock`` because it takes the session
        manager's own lock and can block for the length of a driver hand-back.
        """
        now = self._mono()
        with self._lock:
            session = getattr(self.manager, "session", None)
            if session is None:
                # No session (or a hardware bring-up whose session object does not
                # exist yet): nothing to orphan, and the next one starts its own
                # countdown from scratch. This gap between two sessions is also the ONE
                # place `_had_controller` is cleared — clearing it on a session-id
                # change instead would race the Cockpit, whose control socket opens
                # while `session` is still None during a hardware bring-up.
                self._watched = None
                self._attended_at = now
                if not self._attended():
                    self._had_controller = False
                return None
            watched = _Watched(
                session_id=session.session_id,
                mode=session.spec.mode,
                kind=session.spec.kind,
            )
            if self._watched is None or self._watched.session_id != watched.session_id:
                # A new session: it clears the previous notice (the operator's own
                # launch is the acknowledgement) and starts its grace period now,
                # even if it was created by something other than POST /api/session.
                self._watched = watched
                self._attended_at = now
                self._notice = None
                return None
            if self._attended():
                self._attended_at = now
                self._had_controller = True
                return None
            if self.grace_s <= 0.0:
                return None
            if not self._had_controller:
                return None  # never driven from a Cockpit: not an orphan (module docstring)
            if getattr(self.manager, "state", None) is SessionState.TEARDOWN:
                return None  # a DELETE is already walking it down; don't double-report
            unattended = now - self._attended_at
            if unattended < self.grace_s:
                return None
            open_episode = None
            try:
                open_episode = self.manager.open_episode()
            except Exception:  # noqa: BLE001 - never let bookkeeping block the release
                logger.exception("orphan watch: open_episode() failed")
            reason = (
                f"no controller connected for {_fmt_s(self.grace_s)} - the Cockpit was "
                "closed or reloaded for good, so the runtime ended the session; the arms "
                "were stopped and braked where they stood (no motion)"
            )
            if open_episode:
                reason += f", and the episode being recorded ({open_episode}) was discarded"
            notice = SessionAutoEndNotice(
                session_id=watched.session_id,
                mode=watched.mode,
                kind=watched.kind,
                ended_at=self._now_iso(),
                reason=reason,
            )
        logger.warning(
            "orphaned session %s (%s, %s): %s",
            watched.session_id,
            watched.mode,
            watched.kind,
            reason,
        )
        self.manager.teardown()
        with self._lock:
            self._notice = notice
            self._watched = None
            self._had_controller = False
            self._attended_at = self._mono()
            self._ended += 1
        return notice

    # -- lifecycle (Runtime.start / Runtime.stop) ------------------------------
    def start(self) -> None:
        """Spawn the watch thread; a no-op when disabled or already running.

        A dedicated thread on purpose: the orphan case is "every browser tab is
        gone", so the watch must not be driven by the telemetry socket that a
        closed tab takes with it.
        """
        if self.grace_s <= 0.0:
            logger.info("orphaned-session watch disabled (control.orphan_session_grace_s = 0)")
            return
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="orphan-watch", daemon=True)
        self._thread.start()
        logger.info(
            "orphaned-session watch armed: a session with no controller /ws/control "
            "connection for %s is ended (no motion)",
            _fmt_s(self.grace_s),
        )

    def stop(self) -> None:
        """Stop the thread. Called FIRST in ``Runtime.stop()`` so the shutdown's own
        ``manager.teardown()`` can never race a watch-initiated one."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.wait(self._poll_s):
            try:
                self.poll()
            except Exception:  # noqa: BLE001 - a watch crash must not silence the watch
                logger.exception("orphaned-session watch poll failed")


__all__ = ["OrphanSessionWatch", "POLL_S"]
