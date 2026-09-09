"""GelloReader (phase-15; 16-gello §4 / D2): backends none / fake / dynamixel (the real
backend driven by an injected fake ``dynamixel_sdk`` module: baud scan, GroupSyncRead
parsing incl. negative ticks, calibration mapping, jump rejection, restart with backoff),
port resolution by USB serial against a tmp sysfs tree, the FTDI latency warning, and the
import confinement of ``dynamixel_sdk`` / ``serial`` (mirrors test_tracker_device.py)."""

from __future__ import annotations

import ast
import json
import logging
import math
import subprocess
import sys
import time
import types
from collections import deque
from pathlib import Path

import numpy as np
import pytest
from apollo_mavis_v2_core import LatestSlot
from apollo_mavis_v2_core.protocol import GelloInfo, GelloTelemetry

import apollo_mavis_v2_runtime
from apollo_mavis_v2_runtime.config import GELLO_BAUD_SCAN, GelloConfig, RuntimeConfig
from apollo_mavis_v2_runtime.devices.gello import (
    ADDR_PRESENT_POSITION,
    FAKE_DEFAULT_GRIPPER,
    FAKE_DEFAULT_Q,
    LEN_PRESENT_POSITION,
    PING_TIMEOUT_S,
    READ_FAILS_BEFORE_RESTART,
    GelloPortError,
    GelloReader,
    GelloSample,
    device_telemetry_fields,
    ftdi_latency_timer_ms,
    resolve_port,
    ticks_to_rad,
    tty_nodes_by_usb_serial,
)
from apollo_mavis_v2_runtime.gello.calibration import GelloCalibration, GelloCalibrationStore

SRC = Path(apollo_mavis_v2_runtime.__file__).parent
PI = math.pi
RAD_PER_TICK = 2 * PI / 4096


def _wait(pred, timeout_s: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.005)
    return pred()


# -- a fake dynamixel_sdk module (no write methods exist: a torque / goal write would raise) -----
class FakeBus:
    """What the servos "are": the bus rate they answer at, their ids and Present Position
    ticks (signed; the fake GroupSyncRead returns the uint32 wire value)."""

    def __init__(self, baud: int = 2_000_000, ids=(1, 2, 3, 4, 5, 6, 7, 8)):
        self.baud = baud
        self.ids = list(ids)
        self.ticks: dict[int, int] = dict.fromkeys(self.ids, 0)
        self.results: deque[int] = deque()  # scripted txRxPacket results (default success)
        self.fail_always = False
        self.reads = 0
        self.pings: list[tuple[int, int]] = []  # (baud, id) per ping
        self.ping_delay_s = 0.0  # what one unanswered ping costs (the SDK's packet timeout)
        self.ports: list[FakePort] = []


class FakePort:
    def __init__(self, bus: FakeBus, name: str, *, open_ok: bool = True):
        self.bus, self.name, self.open_ok = bus, name, open_ok
        self.opened = self.closed = False
        self.baud: int | None = None
        self.baud_calls: list[int] = []
        bus.ports.append(self)

    def openPort(self) -> bool:  # noqa: N802 - SDK spelling
        self.opened = self.open_ok
        return self.open_ok

    def setBaudRate(self, baud: int) -> bool:  # noqa: N802
        self.baud = baud
        self.baud_calls.append(baud)
        return True

    def closePort(self) -> None:  # noqa: N802
        self.closed = True


class FakePacketHandler:
    def __init__(self, bus: FakeBus, protocol: float):
        assert protocol == 2.0
        self.bus = bus

    def broadcastPing(self, port: FakePort):  # noqa: N802 - kept for the SDK surface; unused
        if port.baud == self.bus.baud:
            return {i: [1020, 45] for i in self.bus.ids}, 0
        return {}, -3001

    def ping(self, port: FakePort, dxl_id: int):
        """``(model_number, comm_result, error)`` like protocol 2.0; an unanswered ping
        costs ``bus.ping_delay_s`` (the SDK waits its packet timeout)."""
        self.bus.pings.append((port.baud, dxl_id))
        if port.baud == self.bus.baud and dxl_id in self.bus.ids:
            return 1020, 0, 0
        if self.bus.ping_delay_s:
            time.sleep(self.bus.ping_delay_s)
        return 0, -3001, 0

    def getTxRxResult(self, result: int) -> str:  # noqa: N802
        return f"[TxRxResult] {result}"


class FakeGroupSyncRead:
    def __init__(self, bus: FakeBus, port: FakePort, ph: FakePacketHandler, addr: int, length: int):
        assert (addr, length) == (ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
        self.bus, self.port, self.ids = bus, port, []

    def addParam(self, dxl_id: int) -> bool:  # noqa: N802
        self.ids.append(dxl_id)
        return True

    def txRxPacket(self) -> int:  # noqa: N802
        self.bus.reads += 1
        if self.bus.fail_always:
            return -3001
        return self.bus.results.popleft() if self.bus.results else 0

    def isAvailable(self, dxl_id: int, addr: int, length: int) -> bool:  # noqa: N802
        return dxl_id in self.bus.ticks and self.port.baud == self.bus.baud

    def getData(self, dxl_id: int, addr: int, length: int) -> int:  # noqa: N802
        return int(self.bus.ticks[dxl_id]) & 0xFFFFFFFF  # the uint32 wire value


def fake_sdk(bus: FakeBus, *, open_ok: bool = True) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        PortHandler=lambda name: FakePort(bus, name, open_ok=open_ok),
        PacketHandler=lambda proto: FakePacketHandler(bus, proto),
        GroupSyncRead=lambda port, ph, addr, length: FakeGroupSyncRead(bus, port, ph, addr, length),
        COMM_SUCCESS=0,
        COMM_RX_TIMEOUT=-3001,
    )


def _reader(cfg: GelloConfig, bus: FakeBus, tmp_path, **kw) -> tuple[GelloReader, LatestSlot]:
    slot: LatestSlot = LatestSlot()
    kw.setdefault("sysfs_root", tmp_path / "no-sysfs")
    return GelloReader(cfg, slot, import_dynamixel=lambda: fake_sdk(bus), **kw), slot


# -- config -------------------------------------------------------------------------------
def test_config_defaults_match_the_contract():
    g = GelloConfig()
    assert (g.backend, g.port, g.usb_serial, g.baud) == ("none", "/dev/ttyUSB0", None, None)
    assert g.joint_ids == [1, 2, 3, 4, 5, 6, 7] and g.gripper_id == 8
    assert g.joint_signs == [1] * 7 and g.joint_offsets_rad is None
    assert (g.poll_hz, g.stale_s, g.max_jump_rad) == (100.0, 0.2, 0.5)
    assert (g.engage_tol_rad, g.leash_rad, g.gripper_quantum) == (0.10, 0.80, 0.01)
    assert g.max_joint_vel_rad_s == 0.6 and g.view_rail_m == 0.0
    assert g.view_posture_rad == [2.646, -1.598, 0.018, 1.637, 0.25, 2.007, 0.029]
    assert g.scene_id == "mavis_v2_kitchen"
    assert g.calibration_path.is_absolute() and g.calibration_path.name == "gello_calibration.json"
    assert GELLO_BAUD_SCAN == (57_600, 1_000_000, 2_000_000, 3_000_000, 4_000_000)
    assert RuntimeConfig().gello == g and RuntimeConfig().twin_overlay.scene is None
    for bad in (
        {"joint_ids": [1, 2, 3]},
        {"joint_ids": [1, 1, 2, 3, 4, 5, 6]},
        {"joint_signs": [1, 1, 1, 1, 1, 1, 2]},
        {"joint_signs": [1] * 6},
        {"joint_offsets_rad": [0.0] * 6},
        {"view_posture_rad": [0.0] * 8},
        {"gripper_id": 3},
        {"poll_hz": 0},
    ):
        with pytest.raises(ValueError):
            GelloConfig(**bad)
    assert GelloConfig(joint_signs=[-1, 1, -1, 1, 1, 1, -1], joint_offsets_rad=[0.0] * 7)
    assert GelloConfig(gripper_id=None).gripper_id is None


def test_ticks_to_rad_handles_multi_turn_two_s_complement():
    assert ticks_to_rad(0) == 0.0
    assert ticks_to_rad(2048) == pytest.approx(PI)
    assert ticks_to_rad(4096) == pytest.approx(2 * PI)
    assert ticks_to_rad(0xFFFFFFFF) == pytest.approx(-RAD_PER_TICK)  # -1 tick
    assert ticks_to_rad(0x8000_0000) == pytest.approx(-(2**31) * RAD_PER_TICK)
    assert ticks_to_rad(-100 & 0xFFFFFFFF) == pytest.approx(-100 * RAD_PER_TICK)


# -- backend none / fake ------------------------------------------------------------------
def test_none_backend_reports_no_backend_and_never_threads():
    reader = GelloReader(GelloConfig(backend="none"), LatestSlot())
    reader.start()
    st = reader.status(0.0)
    assert st.backend == "none" and st.status == "no_backend" and "none" in st.detail
    assert st.last is None and st.seq == 0 and st.age_s is None and st.port == ""
    assert st.calibrated is False and st.joint_offsets_rad is None and st.joint_signs == (1,) * 7
    assert reader._thread is None
    with pytest.raises(RuntimeError):
        reader.fake_set([0.0] * 7)
    reader.stop()


def test_fake_backend_publishes_the_keyframe_and_fake_set_moves_it():
    slot: LatestSlot = LatestSlot()
    t = [100.0]
    reader = GelloReader(GelloConfig(backend="fake"), slot, clock=lambda: t[0])
    reader.start()
    try:
        assert _wait(lambda: reader.status(t[0]).seq >= 30, 4.0)
        st = reader.status(t[0])
        assert st.backend == "fake" and st.status == "connected" and st.port == "fake"
        assert st.baud is None and st.age_s == 0.0
        assert st.calibrated and st.joint_offsets_rad == (0.0,) * 7
        assert st.calibration_source == "fake"
        sample, _ = slot.get()
        assert isinstance(sample, GelloSample) and sample.valid
        assert np.allclose(sample.q_raw, FAKE_DEFAULT_Q) and np.allclose(sample.q, FAKE_DEFAULT_Q)
        assert sample.gripper_frac == FAKE_DEFAULT_GRIPPER == sample.gripper_raw
        # fake_set publishes at once (and the loop keeps publishing the new posture)
        q1 = [3.0, 0.1, -0.2, 0.3, 0.0, 0.5, -1.0]
        out = reader.fake_set(q1, 0.25)
        assert out is not None and np.allclose(out.q, q1) and out.gripper_frac == 0.25
        assert np.allclose(slot.get()[0].q, q1)
        seq = out.seq
        assert _wait(lambda: reader.status(t[0]).seq > seq + 5)
        assert np.allclose(slot.get()[0].q, q1) and slot.get()[0].gripper_frac == 0.25
        reader.fake_set(q1)  # gripper unchanged
        assert slot.get()[0].gripper_frac == 0.25
        # the fake applies NO calibration: signs / offsets are echoed as identity
        with pytest.raises(ValueError):
            reader.fake_set([0.0] * 6)
        with pytest.raises(ValueError):
            reader.fake_set(q1, 1.5)
        # rate over the window is the real 100 Hz loop (the frozen clock stamps rx only)
        t[0] += 1.0
        st = reader.status(t[0])
        assert st.status == "stale" and st.age_s == pytest.approx(1.0, abs=0.05)
        assert reader.fresh_sample(t[0]) is None
        t[0] -= 1.0
    finally:
        reader.stop()
    assert reader._thread is None


def test_fake_backend_ignores_calibration_but_echoes_the_gripper_endpoints(tmp_path):
    store = GelloCalibrationStore(tmp_path / "cal.json")
    store.save(GelloCalibration((PI / 2,) * 7, gripper_open_rad=2.0, gripper_closed_rad=0.5))
    reader = GelloReader(GelloConfig(backend="fake"), LatestSlot(), calibration_store=store)
    reader.start()
    try:
        assert _wait(lambda: reader.status().seq >= 3)
        st = reader.status()
        assert np.allclose(st.last.q, FAKE_DEFAULT_Q)  # identity mapping
        assert st.joint_offsets_rad == (0.0,) * 7 and st.calibration_source == "fake"
        assert (st.gripper_open_rad, st.gripper_closed_rad) == (2.0, 0.5)
        store.clear()
        reader.reload_calibration()
        assert reader.status().gripper_open_rad is None
    finally:
        reader.stop()


# -- dynamixel backend --------------------------------------------------------------------
def test_dynamixel_backend_without_the_sdk_is_no_backend(monkeypatch):
    monkeypatch.setitem(sys.modules, "dynamixel_sdk", None)  # import -> ImportError
    reader = GelloReader(GelloConfig(backend="dynamixel"), LatestSlot())
    reader.start()
    try:
        assert _wait(lambda: reader.status().status != "starting")
        st = reader.status()
        assert st.status == "no_backend" and "dynamixel_sdk" in st.detail and "[gello]" in st.detail
        assert reader.restarts == 0
    finally:
        reader.stop()


def test_dynamixel_backend_scans_the_baud_parses_ticks_and_maps_the_calibration(tmp_path):
    bus = FakeBus(baud=2_000_000)
    bus.ticks.update({1: 2048, 2: -100, 8: 1000})
    store = GelloCalibrationStore(tmp_path / "cal.json")
    store.save(
        GelloCalibration(
            (PI / 2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            gripper_open_rad=1000 * RAD_PER_TICK,
            gripper_closed_rad=0.0,
        )
    )
    cfg = GelloConfig(backend="dynamixel", joint_signs=[1, -1, 1, 1, 1, 1, 1], port="/dev/ttyFAKE")
    reader, slot = _reader(cfg, bus, tmp_path, calibration_store=store)
    reader.start()
    try:
        assert _wait(lambda: reader.status().status == "connected")
        st = reader.status()
        assert st.port == "/dev/ttyFAKE" and st.baud == 2_000_000
        assert bus.ports[0].baud_calls == [57_600, 1_000_000, 2_000_000]  # stops at the answer
        # one ping per CONFIGURED id per rate, never the SDK's 252-id broadcast (2026-09-09)
        assert [i for b, i in bus.pings if b == 57_600] == [1, 2, 3, 4, 5, 6, 7, 8]
        assert {b for b, _ in bus.pings} == {57_600, 1_000_000, 2_000_000}
        assert st.calibrated and st.calibration_source == "file"
        assert st.joint_offsets_rad == (PI / 2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        s = slot.get()[0]
        assert s.valid and s.q_raw[0] == pytest.approx(PI)
        assert s.q_raw[1] == pytest.approx(-100 * RAD_PER_TICK)  # negative multi-turn ticks
        assert s.q[0] == pytest.approx(PI - PI / 2)  # sign * (raw - offset)
        assert s.q[1] == pytest.approx(+100 * RAD_PER_TICK)  # sign -1
        assert s.gripper_raw == pytest.approx(1000 * RAD_PER_TICK) and s.gripper_frac == 1.0
        assert _wait(lambda: bus.reads >= 5 and reader.status().rate_hz > 50.0, 3.0)
        # every configured servo (7 joints + the gripper) rides ONE sync read per sample
        assert reader.status().seq <= bus.reads and set(cfg.joint_ids + [8]) == set(bus.ticks)
        bus.ticks[8] = 500
        assert _wait(lambda: slot.get()[0].gripper_frac == pytest.approx(0.5))
        # the two wire models accept the status verbatim
        fields = device_telemetry_fields(reader.status())
        GelloTelemetry(**fields)
        GelloInfo(
            **fields, scene_id="mavis_v2_kitchen", scene_label="k", view_posture_rad=[0.0] * 7,
            view_rail_m=0.0, calibration_path="x", hardware_admitted=True,
        )
    finally:
        assert reader.stop()
    assert bus.ports[0].closed


def test_dynamixel_uncalibrated_samples_are_invalid_and_jumps_are_rejected(tmp_path):
    bus = FakeBus(baud=1_000_000)
    cfg = GelloConfig(backend="dynamixel", baud=1_000_000, gripper_id=None)
    reader, slot = _reader(cfg, bus, tmp_path)  # no store, no config offsets
    reader.start()
    try:
        assert _wait(lambda: reader.status().status == "connected")
        assert bus.ports[0].baud_calls == [1_000_000]  # a fixed baud is not scanned
        st = reader.status()
        s = slot.get()[0]
        assert not s.valid and not s.jump and not st.calibrated and "uncalibrated" in st.detail
        assert s.gripper_frac is None and s.gripper_raw is None
        assert reader.fresh_sample() is None and st.invalid_samples >= 1
        # 2026-09-09 review: the calibration ops read the UNCALIBRATED sample - fresh, no jump
        got = reader.fresh_sample(require_calibrated=False)
        assert got is not None and not got.valid and not got.jump
        assert reader.fresh_sample(time.monotonic() + 5.0, require_calibrated=False) is None
    finally:
        reader.stop()
    # config offsets -> calibrated, valid; a jump above max_jump_rad -> ONE invalid sample
    cfg2 = cfg.model_copy(update={"joint_offsets_rad": [0.0] * 7})
    bus2 = FakeBus(baud=1_000_000)
    reader, slot = _reader(cfg2, bus2, tmp_path)
    reader.start()
    try:
        assert _wait(lambda: reader.status().status == "connected" and slot.get()[0].valid)
        assert reader.status().calibration_source == "config"
        bus2.ticks[3] = 1000  # 1.53 rad > 0.5 rad since the previous sample
        assert _wait(lambda: slot.get()[0].q_raw[2] > 1.0)
        first = slot.get()[0]
        assert not first.valid and first.jump and "jump" in reader.status().detail
        assert _wait(lambda: slot.get()[0].seq > first.seq + 2)
        assert slot.get()[0].valid and reader.status().detail == ""  # the next sample is fine
        assert reader.status().invalid_samples >= 1
    finally:
        reader.stop()


def test_dynamixel_no_servo_answering_is_an_error_that_restarts_with_backoff(tmp_path):
    bus = FakeBus(baud=9_999)  # answers at a rate nobody scans
    reader, _ = _reader(GelloConfig(backend="dynamixel"), bus, tmp_path)
    reader.start()
    try:
        assert _wait(lambda: reader.status().status == "error")
        st = reader.status()
        assert "no servo answered" in st.detail and "57600" in st.detail and "4000000" in st.detail
        assert "ids [1, 2, 3, 4, 5, 6, 7, 8]" in st.detail
        assert bus.ports[0].closed and bus.ports[0].baud_calls == list(GELLO_BAUD_SCAN)
        assert _wait(lambda: reader.restarts >= 1 and len(bus.ports) >= 2, 3.0)  # 0.5 s backoff
    finally:
        reader.stop()


def test_stop_interrupts_the_baud_scan_promptly_and_closes_the_port(tmp_path):
    """2026-09-09 review: the scan used to run every rate's 0.8-1.4 s busy-polling broadcast
    ping to the end, ignoring ``stop()``; now the stop flag is checked before every rate
    and every per-id ping, and ``stop()``'s join budget covers a whole worst-case scan."""
    bus = FakeBus(baud=9_999)  # nothing answers: the worst case (servos unpowered)
    bus.ping_delay_s = 0.05  # 8 ids x 5 rates x 50 ms = 2 s for one full scan
    reader, _ = _reader(GelloConfig(backend="dynamixel"), bus, tmp_path)
    assert reader.scan_budget_s() >= len(GELLO_BAUD_SCAN) * 8 * PING_TIMEOUT_S
    reader.start()
    try:
        assert _wait(lambda: reader.status().detail.startswith("pinging servos"), 2.0)
        t0 = time.monotonic()
        assert reader.stop()
        took = time.monotonic() - t0
    finally:
        reader.stop()
    assert took < 0.5, took  # one ping (50 ms here), not the remaining scan
    assert len(bus.ports[0].baud_calls) < len(GELLO_BAUD_SCAN)  # the scan was cut short
    assert bus.ports[0].closed  # the port is never abandoned open
    assert reader.status().status == "error" and "stopped during the baud scan" in (
        reader.status().detail
    )
    assert reader.restarts == 0  # no restart cycle after a stop


def test_fresh_sample_require_calibrated_switch(tmp_path):
    """``fresh_sample(require_calibrated=False)`` hands out a fresh UNCALIBRATED sample (the
    calibration ops' input) but never a stale or a jump-flagged one."""
    bus = FakeBus(baud=1_000_000)
    cfg = GelloConfig(backend="dynamixel", baud=1_000_000, gripper_id=None)
    t = [100.0]
    reader, slot = _reader(cfg, bus, tmp_path, clock=lambda: t[0])
    reader.start()
    try:
        assert _wait(lambda: reader.status(t[0]).status == "connected")
        s = slot.get()[0]
        assert not s.valid and not s.jump
        assert reader.fresh_sample(t[0]) is None  # the default still wants a calibration
        got = reader.fresh_sample(t[0], require_calibrated=False)
        assert got is not None and not got.valid and not got.jump
        assert reader.fresh_sample(t[0] + 1.0, require_calibrated=False) is None  # stale
        bus.ticks[2] = 2000  # a 3 rad jump on joint 2 -> the next sample is jump-flagged
        seen: list[GelloSample] = []
        assert _wait(lambda: bool(seen.append(slot.get()[0]) or seen[-1].jump))
        jump_sample = seen[-1]
        # the jump sample is the newest one for one poll period only: stop the thread and
        # pin it as the newest reading for the assertion
        assert reader.stop()
        reader._last = jump_sample
        assert jump_sample.jump and not jump_sample.valid
        assert reader.fresh_sample(t[0], require_calibrated=False) is None
        assert reader.fresh_sample(t[0]) is None
    finally:
        reader.stop()


def test_dynamixel_read_failures_close_and_reopen_the_port(tmp_path):
    bus = FakeBus(baud=57_600)
    bus.fail_always = True
    reader, _ = _reader(GelloConfig(backend="dynamixel", baud=57_600), bus, tmp_path)
    reader.start()
    try:
        assert _wait(lambda: reader.status().status == "error", 4.0)
        assert "sync read failed" in reader.status().detail
        assert bus.reads >= READ_FAILS_BEFORE_RESTART and bus.ports[0].closed
    finally:
        reader.stop()


def test_dynamixel_port_open_failure_is_an_error(tmp_path):
    bus = FakeBus()
    slot: LatestSlot = LatestSlot()
    reader = GelloReader(
        GelloConfig(backend="dynamixel", port="/dev/ttyNOPE"),
        slot,
        import_dynamixel=lambda: fake_sdk(bus, open_ok=False),
        sysfs_root=tmp_path,
    )
    reader.start()
    try:
        assert _wait(lambda: reader.status().status == "error")
        assert "cannot open /dev/ttyNOPE" in reader.status().detail
    finally:
        reader.stop()


# -- port by USB serial (sysfs) -------------------------------------------------------------
def _sysfs(tmp_path, serials: dict[str, str], latency: dict[str, str] | None = None) -> Path:
    """``<root>/ttyUSBn/device -> devices/usb1/1-<n>/1-<n>:1.0/ttyUSBn`` like ftdi_sio."""
    root = tmp_path / "sys" / "class" / "tty"
    root.mkdir(parents=True)
    for i, (node, serial) in enumerate(serials.items()):
        usb = tmp_path / "sys" / "devices" / "usb1" / f"1-{i + 3}"
        port_dev = usb / f"1-{i + 3}:1.0" / node
        port_dev.mkdir(parents=True)
        (usb / "serial").write_text(serial + "\n")
        (usb / "idVendor").write_text("0403\n")
        if latency and node in latency:
            (port_dev / "latency_timer").write_text(latency[node] + "\n")
        (root / node).mkdir()
        (root / node / "device").symlink_to(port_dev)
    (root / "ttyS0").mkdir()  # a non-USB uart: never matched
    return root


def test_port_resolution_by_usb_serial_walks_the_sysfs_tree(tmp_path):
    root = _sysfs(tmp_path, {"ttyUSB0": "A1B2C3", "ttyUSB1": "FTAKROCJ"}, {"ttyUSB1": "16"})
    assert tty_nodes_by_usb_serial("FTAKROCJ", root) == ["/dev/ttyUSB1"]
    assert tty_nodes_by_usb_serial("A1B2C3", root) == ["/dev/ttyUSB0"]
    assert tty_nodes_by_usb_serial("nope", root) == []
    assert tty_nodes_by_usb_serial("FTAKROCJ", tmp_path / "absent") == []
    cfg = GelloConfig(backend="dynamixel", usb_serial="FTAKROCJ", port="/dev/ttyUSB0")
    assert resolve_port(cfg, root) == "/dev/ttyUSB1"  # the serial wins over `port`
    assert resolve_port(GelloConfig(port="/dev/ttyUSB7"), root) == "/dev/ttyUSB7"
    with pytest.raises(GelloPortError, match="FTAKROCJ"):
        resolve_port(cfg, tmp_path / "absent")
    # latency_timer sits on the usb-serial port device (the `device` link target)
    assert ftdi_latency_timer_ms("/dev/ttyUSB1", root) == 16
    assert ftdi_latency_timer_ms("/dev/ttyUSB0", root) is None
    assert ftdi_latency_timer_ms("/dev/ttyS0", root) is None


def test_dynamixel_backend_uses_the_serial_resolved_port_and_warns_about_latency(tmp_path, caplog):
    root = _sysfs(tmp_path, {"ttyUSB0": "OTHER", "ttyUSB1": "FTAKROCJ"}, {"ttyUSB1": "16"})
    bus = FakeBus(baud=57_600)
    reader, _ = _reader(
        GelloConfig(backend="dynamixel", usb_serial="FTAKROCJ"), bus, tmp_path, sysfs_root=root
    )
    with caplog.at_level(logging.WARNING, logger="apollo_mavis_v2_runtime.devices.gello"):
        reader.start()
        try:
            assert _wait(lambda: reader.status().status == "connected")
            assert reader.status().port == "/dev/ttyUSB1" and bus.ports[0].name == "/dev/ttyUSB1"
        finally:
            reader.stop()
    assert any("latency_timer is 16 ms" in r.getMessage() for r in caplog.records)


def test_dynamixel_missing_adapter_by_serial_is_an_error(tmp_path):
    root = _sysfs(tmp_path, {"ttyUSB0": "OTHER"})
    bus = FakeBus()
    reader, _ = _reader(
        GelloConfig(backend="dynamixel", usb_serial="FTAKROCJ"), bus, tmp_path, sysfs_root=root
    )
    reader.start()
    try:
        assert _wait(lambda: reader.status().status == "error")
        assert "FTAKROCJ" in reader.status().detail and bus.ports == []  # nothing opened
    finally:
        reader.stop()


def test_restart_reopens_the_port():
    bus = FakeBus(baud=57_600)
    reader, _ = _reader(GelloConfig(backend="dynamixel", baud=57_600), bus, Path("/nonexistent"))
    reader.start()
    try:
        assert _wait(lambda: reader.status().status == "connected")
        reader.restart()
        assert _wait(lambda: len(bus.ports) == 2 and reader.status().status == "connected")
        assert bus.ports[0].closed and bus.ports[1].opened
    finally:
        reader.stop()


# -- import confinement ---------------------------------------------------------------------
def _import_names(tree: ast.AST) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_dynamixel_and_serial_imports_are_confined_to_devices_gello():
    offenders = {}
    for path in SRC.rglob("*.py"):
        mods = _import_names(ast.parse(path.read_text(encoding="utf-8"))) & {
            "dynamixel_sdk", "serial",
        }
        if mods:
            offenders[str(path.relative_to(SRC))] = sorted(mods)
    assert offenders == {"devices/gello.py": ["dynamixel_sdk"]}, offenders
    # ... and lazily: no module-level import there
    tree = ast.parse((SRC / "devices" / "gello.py").read_text(encoding="utf-8"))
    for node in tree.body:
        assert not (
            isinstance(node, ast.Import | ast.ImportFrom)
            and "dynamixel_sdk" in _import_names(ast.Module(body=[node], type_ignores=[]))
        )


def test_reader_never_writes_to_the_servos():
    """GELLO stays passive (16-gello §4): no torque / goal / EEPROM write, reboot or reset
    call exists in the module — only reads, pings and port management."""
    tree = ast.parse((SRC / "devices" / "gello.py").read_text(encoding="utf-8"))
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    forbidden = {
        a
        for a in attrs
        if a.startswith(("write", "syncWrite", "bulkWrite", "regWrite", "GroupSyncWrite",
                         "GroupBulkWrite"))
        or a in {"reboot", "factoryReset", "clearMultiTurn", "action"}
    }
    assert not forbidden, forbidden


def test_importing_the_runtime_never_imports_dynamixel_or_serial():
    child = (
        "import sys, json\n"
        "import apollo_mavis_v2_runtime.runtime, apollo_mavis_v2_runtime.devices.gello\n"
        "import apollo_mavis_v2_runtime.gello.engage, apollo_mavis_v2_runtime.gello.calibration\n"
        "print(json.dumps({'dxl': 'dynamixel_sdk' in sys.modules,"
        " 'serial': 'serial' in sys.modules}))\n"
    )
    out = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip().splitlines()[-1]) == {"dxl": False, "serial": False}
