"""IdleArmReader + sources (14-dora §4.2 "Arm states without a session", §8)."""

from __future__ import annotations

import time

import numpy as np
import pytest
from apollo_mavis_v2_core import ArmState, CommandError, GripperState, Pose

from apollo_mavis_v2_runtime.bus import RuntimeBus
from apollo_mavis_v2_runtime.dora_bridge.idle_state import (
    IDLE_TICK,
    DriverIdleSource,
    IdleArmReader,
    MonitorIdleSource,
    SimIdleSource,
)


class FakeReadonlyDriver:
    """The ``XArmDriver.connect(readonly=True)`` surface with a call log."""

    def __init__(self, arm_id: str, log: list[str], *, fail_connect: int = 0) -> None:
        self.arm_id = arm_id
        self.log = log
        self.fail_connect = fail_connect
        self.q = np.array([3.14159, 0, 0, 0, 0, 0, 0, 0.3])
        self.stale = False
        self.connected = False

    def connect(self, readonly: bool = False) -> None:
        self.log.append(f"{self.arm_id}.connect(readonly={readonly})")
        if self.fail_connect > 0:
            self.fail_connect -= 1
            raise ConnectionError("box off")
        self.connected = True

    def disconnect(self) -> None:
        self.log.append(f"{self.arm_id}.disconnect")
        self.connected = False

    def get_state(self) -> ArmState:
        self.log.append(f"{self.arm_id}.get_state")
        return ArmState(
            arm_id=self.arm_id,
            q=self.q,
            dq=np.zeros(8),
            ee_pose=Pose.identity(),
            gripper=GripperState(open_frac=0.7),
            rail_pos_m=float(self.q[7]),
            error_code=0,
            warn_code=0,
            mode=0,
            state=2,
            stale=self.stale,
            t_mono=time.monotonic(),
            wallclock_ns=time.time_ns(),
        )

    def command_joints(self, q):
        raise CommandError("read-only")


def wait(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.005)
    return False


def test_driver_source_reads_only_and_pauses_release_connections():
    log: list[str] = []
    drivers = {a: FakeReadonlyDriver(a, log) for a in ("grip", "view")}
    src = DriverIdleSource(
        lambda a: drivers[a], ["grip", "view"], None, {"grip": True, "view": True}, ["grip"]
    )
    bus = RuntimeBus()
    reader = IdleArmReader(bus, src, hz=50.0, kind="hardware")
    reader.start()
    try:
        assert wait(lambda: reader.snapshots >= 3)
        assert reader.status == "running"
        got = bus.snapshot.get()
        assert got is not None
        snap = got[0]
        assert snap.tick == IDLE_TICK and snap.session_extra == {
            "source": "idle",
            "kind": "hardware",
        }
        assert set(snap.arms) == {"grip", "view"} and snap.arms["grip"].gripper.open_frac == 0.7
        assert snap.arms["view"].gripper.open_frac == 1.0  # camera-only arm: no gripper read
        assert all(c.endswith(("connect(readonly=True)", "get_state")) for c in log)
        assert all("command" not in c and "set_" not in c for c in log)
        # pause = release every connection; no snapshot while paused
        reader.pause()
        assert all(not d.connected for d in drivers.values()) and reader.status == "paused"
        n = reader.snapshots
        time.sleep(0.1)
        assert reader.snapshots == n
        reader.resume()
        assert wait(lambda: reader.snapshots > n, 1.0)  # <= 1 s
        assert all(d.connected for d in drivers.values())
        # a stale report freezes q and marks the arm stale
        drivers["view"].stale = True
        assert wait(lambda: reader.status == "stale")
        assert bus.snapshot.get()[0].arms["view"].stale is True
        drivers["view"].q = drivers["view"].q + 1.0  # the frozen q must not follow
        time.sleep(0.1)
        assert bus.snapshot.get()[0].arms["view"].q[0] == pytest.approx(3.14159)
        drivers["view"].stale = False
        assert wait(lambda: reader.status == "running")
    finally:
        reader.stop()
    assert log[-1].endswith("disconnect") and all(not d.connected for d in drivers.values())


def test_driver_source_reconnects_with_backoff_after_a_lost_box():
    log: list[str] = []
    drv = FakeReadonlyDriver("grip", log, fail_connect=2)
    src = DriverIdleSource(lambda a: drv, ["grip"], None, {"grip": True}, ["grip"])
    src.connect()
    assert src.states() == {} and "retry in" in src.detail and src.connect_attempts == 1
    src._retry_at["grip"] = 0.0  # fast-forward the backoff (1 -> 2 s)
    src.states()
    assert src.connect_attempts == 2 and drv.connected is False
    src._retry_at["grip"] = 0.0
    states = src.states()
    assert drv.connected and "grip" in states and src.connect_attempts == 3
    src.disconnect()
    assert not drv.connected


def test_sim_and_monitor_sources_build_arm_states():
    q = {"grip": np.array([3.14, 0, 0, 0, 0, 0, 0, 0.65]), "view": np.zeros(8)}
    sim = SimIdleSource(lambda: q, None, {"grip": True, "view": True})
    st = sim.states()
    assert st["grip"].rail_pos_m == 0.65 and st["grip"].stale is False and st["grip"].dof == 8

    class Sample:
        def __init__(self, q7, rail, t):
            self.q = q7
            self.rail_pos_m = rail
            self.gripper_open_frac = 0.4
            self.error_code = 0
            self.warn_code = 0
            self.t_mono = t

    class Monitor:
        def __init__(self):
            self.samples = {"grip": Sample([3.14] + [0.0] * 6, None, time.monotonic())}
            self.status = "running"

        def snapshot(self):
            return self.samples

        def status_of(self, arm_id):
            return (self.status, "")

    mon = Monitor()
    src = MonitorIdleSource(mon, None, {"grip": True, "view": True}, {"grip": 0.65, "view": 0.0})
    st = src.states()
    assert set(st) == {"grip"}  # no view sample yet -> absent
    assert st["grip"].rail_pos_m == 0.65  # unhomed rail -> fallback
    assert st["grip"].gripper.open_frac == 0.4 and st["grip"].stale is False
    mon.status = "paused"
    assert src.states()["grip"].stale is True
    mon.status = "running"
    mon.samples["grip"].rail_pos_m = 0.1
    src_flip = MonitorIdleSource(mon, None, {"grip": True}, {"grip": 0.65}, rail_flip=True)
    assert src_flip.states()["grip"].rail_pos_m == pytest.approx(0.55)
    old = Sample([0.0] * 7, 0.2, time.monotonic() - 5.0)
    mon.samples["grip"] = old
    assert src.states()["grip"].stale is True  # sample older than 1 s freezes as stale
