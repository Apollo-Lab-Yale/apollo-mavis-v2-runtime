"""MicrophoneReader (phase-11): backends none/fake/parec-stub, Pulse source matching,
frame analysis, status machine, wire conversions and the sounddevice import
confinement (mirrors test_tracker_device.py)."""

from __future__ import annotations

import ast
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from apollo_mavis_v2_core import LatestSlot
from apollo_mavis_v2_core.protocol import MicrophoneInfo, MicrophoneTelemetry

import apollo_mavis_v2_runtime
from apollo_mavis_v2_runtime.config import MicrophoneConfig, RuntimeConfig
from apollo_mavis_v2_runtime.devices import microphone as micmod
from apollo_mavis_v2_runtime.devices.microphone import (
    CLIP_DBFS,
    FAKE_AMP_MAX,
    FAKE_AMP_MIN,
    MicrophoneReader,
    PulseSource,
    frame_stats,
    match_source,
    parec_argv,
    to_info,
    to_telemetry,
)

SRC = Path(apollo_mavis_v2_runtime.__file__).parent

RODE_NAME = "alsa_input.usb-R__DE_Microphones_R__DE_NT-USB_Mini_750BFEE8-00.mono-fallback"
RODE = PulseSource(
    index=64, name=RODE_NAME, description="(null)",
    properties={"device.class": "sound", "device.vendor.id": "19f7", "device.product.id": "0015"},
)
RODE_MONITOR = PulseSource(
    index=63,
    name="alsa_output.usb-R__DE_Microphones_R__DE_NT-USB_Mini_750BFEE8-00.analog-stereo.monitor",
    description="(null)", properties={"device.class": "monitor"},
)
OTHER = PulseSource(
    index=4, name="alsa_input.usb-Generic_USB_Audio-00.HiFi__hw_Audio_2__source",
    description="USB Audio Microphone", properties={"device.class": "sound"},
)


def _wait(pred, timeout_s: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


# -- config -------------------------------------------------------------------------------
def test_config_defaults_match_contract():
    cfg = MicrophoneConfig()
    assert (cfg.enabled, cfg.mic_id, cfg.label) == (False, "mic_view", "Perception Arm microphone")
    assert (cfg.backend, cfg.source_match) == ("auto", "NT-USB Mini")
    assert (cfg.sample_rate, cfg.bins, cfg.stale_s) == (48000, 64, 0.5)
    rt = RuntimeConfig()
    assert rt.microphone == MicrophoneConfig()
    probe = rt.hardware_probe
    assert (probe.enabled, probe.period_s, probe.timeout_s, probe.port) == (True, 2.0, 1.0, 502)


def test_frame_length_is_aligned_to_telemetry_hz():
    cfg = MicrophoneConfig(enabled=True, backend="fake")
    r = MicrophoneReader(cfg, LatestSlot(), frame_hz=25)
    assert r.frame_len == 1920 and r.frame_len % r.cfg.bins == 0  # 64 x 30 exact bins
    assert MicrophoneReader(cfg, LatestSlot(), frame_hz=30).frame_len == 1600


# -- frame analysis -------------------------------------------------------------------------
def test_frame_stats_sine_levels_envelope_and_clipping():
    n, bins = 1920, 64
    t = np.arange(n) / 48000.0
    x = (0.5 * np.sin(2 * math.pi * 1000.0 * t)).astype(np.float32)
    peak_dbfs, rms_dbfs, clipping, env_min, env_max = frame_stats(x, bins)
    assert abs(peak_dbfs - 20 * math.log10(0.5)) < 0.05
    assert abs(rms_dbfs - 20 * math.log10(0.5 / math.sqrt(2))) < 0.1
    assert clipping is False
    assert env_min.shape == (bins,) and env_max.shape == (bins,)
    assert env_min.dtype.kind == "i" and env_max.dtype.kind == "i"
    assert env_min.min() >= -127 and env_max.max() <= 127
    assert np.all(env_max >= env_min)
    assert env_max.max() == 64 and env_min.min() == -64  # 0.5 * 127 rounded, time-ordered bins
    # Full-scale -> clipping; int8 saturates at +-127.
    loud = np.clip(x * 3.0, -1.0, 1.0)
    p, _, clip, lo, hi = frame_stats(loud, bins)
    assert p >= CLIP_DBFS and clip is True and hi.max() == 127 and lo.min() == -127
    # Silence -> floor, no clipping, flat envelope; short frames are sampled.
    p0, r0, c0, lo0, hi0 = frame_stats(np.zeros(n, dtype=np.float32), bins)
    assert p0 == r0 == micmod.DBFS_FLOOR and c0 is False and not lo0.any() and not hi0.any()
    assert frame_stats(np.ones(10, dtype=np.float32), bins)[4].shape == (bins,)


# -- Pulse source matching ------------------------------------------------------------------
def test_match_source_by_name_ignores_monitors_and_case():
    srcs = [OTHER, RODE_MONITOR, RODE]
    assert match_source(srcs, "NT-USB Mini") is RODE  # spaces vs underscores/dashes
    assert match_source(srcs, "nt_usb_mini") is RODE
    assert match_source(srcs, "750BFEE8") is RODE  # serial in the name
    assert match_source(srcs, "USB Audio Microphone") is OTHER  # description match
    assert match_source([RODE_MONITOR, OTHER], "NT-USB Mini") is None  # monitor never matches
    assert match_source(srcs, "") is None


def test_parec_argv_is_the_pulse_route_command():
    argv = parec_argv(RODE_NAME, 48000)
    assert argv[:3] == ["parec", "-d", RODE_NAME]
    assert "--format=s16le" in argv and "--rate=48000" in argv and "--channels=1" in argv
    assert "--raw" in argv and "--latency-msec=50" in argv


def test_list_pulse_sources_parses_pactl_json_and_short(monkeypatch):
    import subprocess

    class Done:
        def __init__(self, rc, out, err=""):
            self.returncode, self.stdout, self.stderr = rc, out, err

    rows = [
        {"index": 64, "name": RODE_NAME, "description": "(null)",
         "properties": {"device.class": "sound"}},
        {"index": 63, "name": RODE_MONITOR.name, "description": None,
         "properties": {"device.class": "monitor"}},
    ]
    import json

    def fake_run(argv, **kw):
        if argv[:2] == ["pactl", "-f"]:
            return Done(0, json.dumps(rows), "Invalid non-ASCII character\n")
        raise AssertionError(argv)

    monkeypatch.setattr(subprocess, "run", fake_run)
    got = micmod.list_pulse_sources()
    assert [s.index for s in got] == [64, 63] and got[0].description == "(null)"
    assert got[1].is_monitor and not got[0].is_monitor

    def fake_run_short(argv, **kw):
        if argv[:2] == ["pactl", "-f"]:
            return Done(1, "", "unknown option")
        return Done(0, f"64\t{RODE_NAME}\tmodule-alsa-card.c\ts24le 1ch 48000Hz\tSUSPENDED\n")

    monkeypatch.setattr(subprocess, "run", fake_run_short)
    got = micmod.list_pulse_sources()
    assert len(got) == 1 and got[0].name == RODE_NAME and got[0].index == 64

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Done(1, "", "Connection refused"))
    with pytest.raises(OSError):
        micmod.list_pulse_sources()


# -- backend none / disabled -------------------------------------------------------------
def test_disabled_and_none_backends_report_no_backend_and_never_thread():
    for cfg, word in (
        (MicrophoneConfig(enabled=False, backend="fake"), "enabled"),
        (MicrophoneConfig(enabled=True, backend="none"), "none"),
    ):
        reader = MicrophoneReader(cfg, LatestSlot())
        reader.start()
        st = reader.status(0.0)
        assert st.status == "no_backend" and st.kind == "none" and word in st.detail
        assert st.seq == 0 and st.age_s is None and st.last is None and st.source is None
        assert reader._thread is None and not reader.active
        info = to_info(st)
        assert isinstance(info, MicrophoneInfo)
        assert info.live is False and info.kind == "none" and info.source is None
        tele = to_telemetry(st)
        assert isinstance(tele, MicrophoneTelemetry)
        assert tele.status == "no_backend" and tele.env_min == [] and tele.rms_dbfs is None
        reader.stop()


# -- fake backend ---------------------------------------------------------------------------
def test_fake_backend_frames_at_frame_hz_then_stalled_after_stop():
    slot: LatestSlot = LatestSlot()
    cfg = MicrophoneConfig(enabled=True, backend="fake", bins=64, stale_s=0.5)
    reader = MicrophoneReader(cfg, slot, frame_hz=25.0)
    reader.start()
    try:
        assert _wait(lambda: reader.status().seq >= 20, 4.0)
        st = reader.status()
        assert st.status == "live" and st.kind == "fake" and st.backend == "fake"
        assert st.source is None and st.sample_rate == 48000 and st.channels == 1
        assert 15.0 <= st.rate_hz <= 35.0, st.rate_hz  # ~25 Hz
        assert st.age_s is not None and st.age_s < 0.2
        frame = slot.get()[0]
        assert frame.seq == st.seq and frame is st.last
        assert len(frame.env_min) == 64 and len(frame.env_max) == 64
        assert all(-127 <= v <= 127 for v in frame.env_min + frame.env_max)
        assert all(hi >= lo for lo, hi in zip(frame.env_min, frame.env_max, strict=True))
        # RMS of an AM sine stays inside [amp_min, amp_max] / sqrt(2) (+-1 dB slack).
        lo = 20 * math.log10(FAKE_AMP_MIN / math.sqrt(2)) - 1.0
        hi = 20 * math.log10(FAKE_AMP_MAX / math.sqrt(2)) + 1.0
        assert lo <= frame.rms_dbfs <= hi, frame.rms_dbfs
        assert frame.peak_dbfs >= frame.rms_dbfs and not frame.clipping
        seq0 = st.seq
        assert _wait(lambda: reader.status().seq > seq0, 1.0)  # seq keeps advancing
        tele = to_telemetry(reader.status())
        assert tele.status == "live" and tele.seq > seq0 and len(tele.env_max) == 64
        assert all(isinstance(v, int) for v in tele.env_min + tele.env_max)
        assert to_info(reader.status()).live is True
    finally:
        reader.stop()
    assert reader._thread is None
    later = reader.status(time.monotonic() + 1.0)
    assert later.status == "stalled" and later.rate_hz == 0.0
    assert to_info(later).live is False and to_telemetry(later).status == "stalled"


def test_fake_backend_start_is_idempotent_and_restartable():
    reader = MicrophoneReader(MicrophoneConfig(enabled=True, backend="fake"), LatestSlot())
    reader.start()
    t1 = reader._thread
    reader.start()  # no second thread
    assert reader._thread is t1
    reader.stop()
    reader.start()
    assert reader._thread is not None and reader._thread is not t1
    reader.stop()


# -- Pulse route without hardware: absent / parec stub / missing backends ----------------
def test_pulse_route_reports_absent_when_no_source_matches():
    reader = MicrophoneReader(
        MicrophoneConfig(enabled=True, backend="auto"), LatestSlot(),
        source_lister=lambda: [OTHER, RODE_MONITOR],
    )
    reader.start()
    try:
        assert _wait(lambda: reader.status().status == "absent")
        st = reader.status()
        assert st.kind == "pulse" and st.source is None and "NT-USB Mini" in st.detail
        info = to_info(st)
        assert info.live is False and info.kind == "pulse" and info.status == "absent"
    finally:
        reader.stop()


def test_pactl_failure_is_error_not_crash():
    def boom():
        raise OSError("pactl: Connection refused")

    reader = MicrophoneReader(
        MicrophoneConfig(enabled=True, backend="auto"), LatestSlot(), source_lister=boom
    )
    reader.start()
    try:
        assert _wait(lambda: reader.status().status == "error")
        assert "pactl" in reader.status().detail
    finally:
        reader.stop()


def _fake_parec(tmp_path: Path) -> Path:
    """A ``parec`` stand-in: streams a 0.25 amplitude 1 kHz s16le mono sine at
    48 kHz pace to stdout, ignoring its arguments."""
    script = tmp_path / "parec"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import math, struct, sys, time\n"
        "sr, n, i = 48000, 480, 0\n"
        "out = sys.stdout.buffer\n"
        "t0 = time.monotonic()\n"
        "while True:\n"
        "    buf = bytearray()\n"
        "    for k in range(n):\n"
        "        s = 0.25 * math.sin(2 * math.pi * 1000.0 * (i + k) / sr)\n"
        "        buf += struct.pack('<h', int(s * 32767))\n"
        "    i += n\n"
        "    try:\n"
        "        out.write(buf); out.flush()\n"
        "    except BrokenPipeError:\n"
        "        break\n"
        "    delay = t0 + i / sr - time.monotonic()\n"
        "    if delay > 0: time.sleep(delay)\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def test_parec_route_delivers_frames_with_pulse_kind_and_source(tmp_path):
    script = _fake_parec(tmp_path)
    seen: list[list[str]] = []

    def argv(source: str, sr: int) -> list[str]:
        cmd = [str(script)] + parec_argv(source, sr)[1:]
        seen.append(cmd)
        return cmd

    reader = MicrophoneReader(
        MicrophoneConfig(enabled=True, backend="parec", stale_s=0.5), LatestSlot(),
        frame_hz=25.0, source_lister=lambda: [OTHER, RODE], parec_command=argv,
    )
    reader.start()
    try:
        assert _wait(lambda: reader.status().seq >= 10, 6.0), reader.status()
        st = reader.status()
        assert st.status == "live" and st.kind == "pulse" and st.source == RODE_NAME
        assert seen and seen[0][1:3] == ["-d", RODE_NAME] and "--format=s16le" in seen[0]
        assert abs(st.last.rms_dbfs - 20 * math.log10(0.25 / math.sqrt(2))) < 0.5
        assert len(st.last.env_min) == 64 and not st.last.clipping
        assert to_info(st).live is True and to_info(st).source == RODE_NAME
    finally:
        reader.stop()
    assert reader._thread is None


def test_parec_route_unplug_flips_to_absent(tmp_path):
    script = _fake_parec(tmp_path)
    present = {"on": True}
    reader = MicrophoneReader(
        MicrophoneConfig(enabled=True, backend="parec"), LatestSlot(), frame_hz=25.0,
        source_lister=lambda: [RODE] if present["on"] else [],
        parec_command=lambda src, sr: [str(script)] + parec_argv(src, sr)[1:],
    )
    reader.start()
    try:
        assert _wait(lambda: reader.status().status == "live", 6.0)
        present["on"] = False  # 1 Hz presence probe notices within ~1 s
        assert _wait(lambda: reader.status().status == "absent", 4.0), reader.status()
        assert reader.status().source is None
        present["on"] = True
        assert _wait(lambda: reader.status().status == "live", 6.0), reader.status()
    finally:
        reader.stop()


def test_sounddevice_missing_explicit_backend_is_no_backend(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)  # import -> ImportError
    reader = MicrophoneReader(
        MicrophoneConfig(enabled=True, backend="sounddevice"), LatestSlot(),
        source_lister=lambda: [RODE],
    )
    reader.start()
    try:
        assert _wait(lambda: reader.status().status == "no_backend")
        st = reader.status()
        assert "sounddevice" in st.detail and st.kind == "none"
        assert to_info(st).kind == "none" and to_info(st).live is False
    finally:
        reader.stop()


def test_auto_without_sounddevice_falls_back_to_parec(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    script = _fake_parec(tmp_path)
    reader = MicrophoneReader(
        MicrophoneConfig(enabled=True, backend="auto"), LatestSlot(), frame_hz=25.0,
        source_lister=lambda: [RODE],
        parec_command=lambda src, sr: [str(script)] + parec_argv(src, sr)[1:],
    )
    reader.start()
    try:
        assert _wait(lambda: reader.status().status == "live", 6.0), reader.status()
        assert reader.status().kind == "pulse"
    finally:
        reader.stop()


def test_auto_without_any_route_is_no_backend(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    missing = tmp_path / "definitely-not-parec"
    reader = MicrophoneReader(
        MicrophoneConfig(enabled=True, backend="auto"), LatestSlot(),
        source_lister=lambda: [RODE],
        parec_command=lambda src, sr: [str(missing), "-d", src],
    )
    reader.start()
    try:
        assert _wait(lambda: reader.status().status == "no_backend")
        detail = reader.status().detail
        assert "sounddevice" in detail and "not found" in detail
    finally:
        reader.stop()


def test_open_failure_is_error_with_retry(tmp_path):
    calls = {"n": 0}

    def argv(src, sr):
        calls["n"] += 1
        return ["/bin/false"]  # present, exits immediately -> read fails -> error + backoff

    reader = MicrophoneReader(
        MicrophoneConfig(enabled=True, backend="parec"), LatestSlot(),
        source_lister=lambda: [RODE], parec_command=argv,
    )
    reader.start()
    try:
        assert _wait(lambda: reader.status().status == "error", 4.0), reader.status()
        assert _wait(lambda: calls["n"] >= 2, 4.0)  # re-opened after the backoff
    finally:
        reader.stop()


# -- import confinement -------------------------------------------------------------------
def test_sounddevice_import_confined_to_devices_microphone():
    offenders = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(n.split(".")[0] == "sounddevice" for n in names):
                offenders.append(str(path.relative_to(SRC)))
    assert offenders == ["devices/microphone.py"], offenders


def test_sounddevice_import_in_microphone_module_is_lazy():
    """The import lives inside a function (module import never touches PortAudio)."""
    tree = ast.parse((SRC / "devices" / "microphone.py").read_text(encoding="utf-8"))
    top_level = [
        n for n in tree.body
        if isinstance(n, ast.Import | ast.ImportFrom)
        and any(
            (a.name if isinstance(n, ast.Import) else (n.module or "")).split(".")[0]
            == "sounddevice"
            for a in (n.names if isinstance(n, ast.Import) else [None])
        )
    ]
    assert top_level == []


# -- real microphone (Pulse route), opt-in ----------------------------------------------------
@pytest.mark.skipif(
    os.environ.get("APOLLO_MIC_HW") != "1",
    reason="real RØDE NT-USB Mini via PulseAudio; set APOLLO_MIC_HW=1",
)
def test_real_microphone_pulse_route_delivers_frames():
    reader = MicrophoneReader(MicrophoneConfig(enabled=True, backend="auto"), LatestSlot())
    reader.start()
    try:
        assert _wait(lambda: reader.status().status in ("live", "absent", "no_backend"), 5.0)
        st = reader.status()
        if st.status == "live":
            assert st.kind == "pulse" and st.source and st.seq > 0
    finally:
        reader.stop()
