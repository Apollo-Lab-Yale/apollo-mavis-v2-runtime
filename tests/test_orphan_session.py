"""Orphaned-session watch (2026-09-09 evening; 04-runtime §13.2, session/orphan.py).

The incident these tests pin: a hardware Teleop session started at 22:51 was still
``running`` at 23:16 because the Cockpit tab had been closed with the browser's Back
button instead of **End session**. Both control boxes stayed enabled with the arms
holding their posture and nothing on the machine would ever have released them.

Everything here drives :meth:`OrphanSessionWatch.poll` with an INJECTED clock against
a scripted manager, so the suite is instant and never builds a workcell. One test at
the end runs the real thread with a millisecond grace period to prove
``start()``/``stop()`` are wired.
"""

from __future__ import annotations

import threading
import time

import pytest
from apollo_mavis_v2_core.protocol import SessionAutoEndNotice

from apollo_mavis_v2_runtime.session.orphan import OrphanSessionWatch
from apollo_mavis_v2_runtime.session.types import SessionState

GRACE = 30.0


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


class FakeSpec:
    def __init__(self, mode: str = "teleop", kind: str = "hardware") -> None:
        self.mode = mode
        self.kind = kind


class FakeSession:
    def __init__(self, session_id: str = "s1", mode: str = "teleop", kind: str = "hardware"):
        self.session_id = session_id
        self.spec = FakeSpec(mode, kind)


class FakeManager:
    """The slice of :class:`SessionManager` the watch touches."""

    def __init__(self, session: FakeSession | None = None) -> None:
        self.session = session
        self.state = SessionState.RUNNING
        self.teardowns: list[str | None] = []
        self.episode: str | None = None

    def teardown(self) -> None:
        self.teardowns.append(self.session.session_id if self.session else None)
        self.session = None
        self.state = SessionState.IDLE

    def open_episode(self) -> str | None:
        return self.episode


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class Attendance:
    """Stands in for ``Runtime.controller_connected`` (maintained by ws_control)."""

    def __init__(self, connected: bool = False) -> None:
        self.connected = connected

    def __call__(self) -> bool:
        return self.connected


def make_watch(
    manager: FakeManager,
    attendance: Attendance,
    grace_s: float = GRACE,
    clock: Clock | None = None,
) -> tuple[OrphanSessionWatch, Clock]:
    clock = clock or Clock()
    watch = OrphanSessionWatch(
        manager,
        grace_s,
        attendance,
        monotonic=clock,
        now_iso=lambda: "2026-09-09T23:16:00+00:00",
    )
    return watch, clock


def settle(watch: OrphanSessionWatch, clock: Clock, attendance: Attendance) -> None:
    """Two polls with the controller attending: the watch adopts the session and
    records that it HAS been driven (the precondition for ever ending it)."""
    was = attendance.connected
    attendance.connected = True
    watch.poll()  # adopts the session
    watch.poll()  # sees the controller
    attendance.connected = was
    clock.advance(0.0)


# -- the case that bit us -------------------------------------------------------------
def test_ends_a_session_whose_controller_went_away():
    mgr = FakeManager(FakeSession("abc123", "teleop", "hardware"))
    att = Attendance()
    watch, clock = make_watch(mgr, att)
    settle(watch, clock, att)

    clock.advance(GRACE - 0.5)
    assert watch.poll() is None, "still inside the grace period"
    assert mgr.teardowns == []

    clock.advance(1.0)
    notice = watch.poll()
    assert isinstance(notice, SessionAutoEndNotice)
    assert mgr.teardowns == ["abc123"], "ended through the no-motion teardown, exactly once"
    assert (notice.session_id, notice.mode, notice.kind) == ("abc123", "teleop", "hardware")
    assert notice.ended_at == "2026-09-09T23:16:00+00:00"
    # The operator reads this verbatim on the Welcome page: it must say what happened to
    # the arms, because the person walking back to the cell needs to know.
    assert notice.reason.startswith("no controller connected for 30 s")
    assert "no motion" in notice.reason
    assert watch.notice == notice
    assert watch.ended_count == 1


def test_the_notice_survives_until_the_next_session_starts():
    mgr = FakeManager(FakeSession("abc123"))
    att = Attendance()
    watch, clock = make_watch(mgr, att)
    settle(watch, clock, att)
    clock.advance(GRACE)
    notice = watch.poll()
    assert notice is not None

    for _ in range(5):  # no session: the notice is what the Welcome page still shows
        clock.advance(GRACE)
        assert watch.poll() is None
        assert watch.notice == notice

    mgr.session = FakeSession("def456")  # the operator launches again = acknowledgement
    assert watch.poll() is None
    assert watch.notice is None


def test_one_teardown_only_even_if_polling_continues():
    mgr = FakeManager(FakeSession("abc123"))
    att = Attendance()
    watch, clock = make_watch(mgr, att)
    settle(watch, clock, att)
    clock.advance(GRACE)
    assert watch.poll() is not None
    for _ in range(10):
        clock.advance(GRACE)
        assert watch.poll() is None
    assert mgr.teardowns == ["abc123"]
    assert watch.ended_count == 1


# -- what must NOT be ended -----------------------------------------------------------
def test_a_session_no_controller_ever_drove_is_never_ended():
    """A REST-driven session (a test harness, a script, a policy bring-up whose socket
    is not open yet) is not an orphan — nobody walked away from it."""
    mgr = FakeManager(FakeSession("rest-only"))
    att = Attendance(connected=False)
    watch, clock = make_watch(mgr, att)
    for _ in range(20):
        clock.advance(GRACE)
        assert watch.poll() is None
    assert mgr.teardowns == []
    assert watch.notice is None


def test_a_reload_inside_the_grace_period_is_not_an_orphan():
    """F5 in the Cockpit drops and re-opens /ws/control within a second. The grace
    period exists exactly to tell that apart from a tab closed for good."""
    mgr = FakeManager(FakeSession("abc123"))
    att = Attendance()
    watch, clock = make_watch(mgr, att)
    settle(watch, clock, att)

    clock.advance(GRACE * 0.9)
    assert watch.poll() is None
    att.connected = True  # the reloaded Cockpit is back
    assert watch.poll() is None
    att.connected = False
    clock.advance(GRACE * 0.9)  # the countdown restarted from the reconnect
    assert watch.poll() is None
    assert mgr.teardowns == []

    clock.advance(GRACE)
    assert watch.poll() is not None


def test_an_attending_controller_never_expires():
    mgr = FakeManager(FakeSession("abc123"))
    att = Attendance(connected=True)
    watch, clock = make_watch(mgr, att)
    for _ in range(20):
        clock.advance(GRACE * 3)
        assert watch.poll() is None
    assert mgr.teardowns == []


def test_note_activity_postpones_the_countdown():
    """``POST /api/session`` and ``POST /api/session/return_home`` stamp the watch: a
    synchronous return walking both arms home must never be interrupted by it."""
    mgr = FakeManager(FakeSession("abc123"))
    att = Attendance()
    watch, clock = make_watch(mgr, att)
    settle(watch, clock, att)
    for _ in range(5):
        clock.advance(GRACE * 0.9)
        watch.note_activity("POST /api/session/return_home")
        assert watch.poll() is None
    assert mgr.teardowns == []
    clock.advance(GRACE)
    assert watch.poll() is not None


def test_note_activity_alone_never_arms_the_watch():
    """Stamping is not attendance: a REST-only session stays exempt however many
    session calls it makes."""
    mgr = FakeManager(FakeSession("rest-only"))
    att = Attendance()
    watch, clock = make_watch(mgr, att)
    watch.poll()
    for _ in range(5):
        watch.note_activity("POST /api/session/return_home")
        clock.advance(GRACE * 2)
        assert watch.poll() is None
    assert mgr.teardowns == []


def test_grace_zero_disables_the_watch():
    mgr = FakeManager(FakeSession("abc123"))
    att = Attendance()
    watch, clock = make_watch(mgr, att, grace_s=0.0)
    settle(watch, clock, att)
    for _ in range(10):
        clock.advance(3600.0)
        assert watch.poll() is None
    assert mgr.teardowns == []
    before = threading.active_count()
    watch.start()  # spawns nothing
    assert threading.active_count() == before
    watch.stop()  # idempotent without a thread


def test_a_teardown_already_in_flight_is_not_reported():
    """A DELETE that is walking the session down sets TEARDOWN before it clears
    ``session``; the watch must not claim that end as its own."""
    mgr = FakeManager(FakeSession("abc123"))
    att = Attendance()
    watch, clock = make_watch(mgr, att)
    settle(watch, clock, att)
    mgr.state = SessionState.TEARDOWN
    clock.advance(GRACE * 3)
    assert watch.poll() is None
    assert mgr.teardowns == []
    assert watch.notice is None


def test_the_controller_flag_does_not_leak_into_the_next_session():
    """Session A was driven from a Cockpit and ended normally; a REST-only session B
    must not inherit A's attendance and be torn down."""
    mgr = FakeManager(FakeSession("driven"))
    att = Attendance()
    watch, clock = make_watch(mgr, att)
    settle(watch, clock, att)
    mgr.session = None  # DELETE /api/session
    assert watch.poll() is None  # the gap between sessions clears the flag

    mgr.session = FakeSession("rest-only")
    for _ in range(10):
        clock.advance(GRACE)
        assert watch.poll() is None
    assert mgr.teardowns == []


def test_a_sub_poll_controller_visit_still_arms_the_watch():
    """A controller that connects and drops again between two polls would otherwise
    leave an abandoned session exempt for the rest of its life, so ``ws_control``
    records the attendance itself instead of relying on the watch sampling the flag."""
    mgr = FakeManager(FakeSession("abc123"))
    att = Attendance()
    watch, clock = make_watch(mgr, att)
    watch.poll()  # adopts the session; no controller seen yet
    watch.note_controller_connected()  # ws_control accepted a controller...
    clock.advance(GRACE + 1.0)  # ...which was already gone before the next poll
    notice = watch.poll()
    assert notice is not None
    assert mgr.teardowns == ["abc123"]


# -- an open episode is discarded, and the notice says so -----------------------------
def test_an_open_episode_is_named_in_the_reason():
    mgr = FakeManager(FakeSession("abc123", "collect", "hardware"))
    mgr.episode = "20260909T231600.000Z-a1b2c3"
    att = Attendance()
    watch, clock = make_watch(mgr, att)
    settle(watch, clock, att)
    clock.advance(GRACE)
    notice = watch.poll()
    assert notice is not None
    assert "20260909T231600.000Z-a1b2c3" in notice.reason
    assert "discarded" in notice.reason


def test_a_broken_open_episode_lookup_never_blocks_the_release():
    mgr = FakeManager(FakeSession("abc123", "collect"))
    mgr.open_episode = lambda: (_ for _ in ()).throw(RuntimeError("recorder gone"))  # type: ignore[method-assign]
    att = Attendance()
    watch, clock = make_watch(mgr, att)
    settle(watch, clock, att)
    clock.advance(GRACE)
    notice = watch.poll()
    assert notice is not None and mgr.teardowns == ["abc123"]


# -- the real thread -----------------------------------------------------------------
def test_the_watch_thread_ends_the_session(monkeypatch):
    """start()/stop() wiring with a millisecond grace period (no injected clock)."""
    mgr = FakeManager(FakeSession("abc123"))
    att = Attendance(connected=True)
    watch = OrphanSessionWatch(mgr, 0.05, att, poll_s=0.01)
    watch.start()
    try:
        deadline = 0
        while mgr.teardowns == [] and deadline < 100:  # the controller is still attending
            deadline += 1
            _sleep(0.005)
        assert mgr.teardowns == [], "an attending controller keeps the session alive"
        att.connected = False
        for _ in range(200):
            if mgr.teardowns:
                break
            _sleep(0.005)
        assert mgr.teardowns == ["abc123"]
        assert watch.notice is not None
    finally:
        watch.stop()
    assert not any(t.name == "orphan-watch" for t in threading.enumerate())


def test_stop_is_idempotent():
    watch = OrphanSessionWatch(FakeManager(), 1.0, Attendance(), poll_s=0.01)
    watch.start()
    watch.stop()
    watch.stop()


def test_the_thread_keeps_watching_after_a_failed_poll():
    """A watch that died on one bad poll would silently stop protecting the cell, so
    the thread swallows the exception and comes back next period."""

    class Broken:
        def __init__(self) -> None:
            self.calls = 0

        @property
        def session(self):
            self.calls += 1
            raise RuntimeError("manager exploded")

    mgr = Broken()
    watch = OrphanSessionWatch(mgr, 1.0, Attendance(), poll_s=0.01)
    with pytest.raises(RuntimeError):
        watch.poll()  # poll() itself propagates to its caller...
    watch.start()  # ...while the thread logs and keeps going
    try:
        _sleep(0.08)
        assert mgr.calls > 2
        assert any(t.name == "orphan-watch" for t in threading.enumerate())
    finally:
        watch.stop()


# -- the whole wiring, through the real server ----------------------------------------
def test_a_closed_cockpit_tab_ends_the_session_end_to_end(tmp_path):
    """ws_control -> Runtime.controller_connected -> the watch thread -> the no-motion
    teardown -> telemetry.session.auto_ended, over ``create_app`` with a sim workcell.

    This is the test that would have caught the 2026-09-09 incident: closing the
    Cockpit's socket is all it takes, and the session must be gone shortly after.
    """
    from conftest import make_runtime_config
    from starlette.testclient import TestClient

    from apollo_mavis_v2_runtime.runtime import Runtime
    from apollo_mavis_v2_runtime.server.app import create_app

    cfg = make_runtime_config(tmp_path)
    # The suite-wide default is 0 (disabled, see conftest); this module wants it armed.
    cfg.control = cfg.control.model_copy(update={"orphan_session_grace_s": 0.3})
    spec = {
        "mode": "teleop",
        "kind": "sim",
        "arms": ["arm0"],
        "frames": {"arm0": "arm_base:arm0"},
        "sim_scene": "single_rail",
    }
    with TestClient(create_app(Runtime(cfg))) as client:
        assert client.post("/api/session", json=spec).status_code == 200
        session_id = client.get("/api/session").json()["session_id"]

        # The Cockpit is open: its control socket is the controller.
        with client.websocket_connect("/ws/control") as ws:
            assert ws.receive_json()["role"] == "controller"
            _sleep(1.0)  # comfortably past the grace period
            assert client.get("/api/session").status_code == 200, "attended: still running"

        # The tab is closed (the browser's Back button, not "End session").
        for _ in range(100):
            if client.get("/api/session").status_code == 404:
                break
            _sleep(0.05)
        assert client.get("/api/session").status_code == 404, "the orphan was not released"

        with client.websocket_connect("/ws/telemetry") as ws:
            block = ws.receive_json()["session"]
        assert block["state"] == "idle"
        assert block["session_id"] is None
        auto = block["auto_ended"]
        assert auto is not None, "the Welcome page must be able to say why the arms were freed"
        assert auto["session_id"] == session_id
        assert (auto["mode"], auto["kind"]) == ("teleop", "sim")
        assert auto["reason"].startswith("no controller connected for 0.3 s")
        assert auto["ended_at"]
