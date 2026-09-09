"""EpisodeAudioSink over a stub microphone reader (10-frames §11.4; 04-runtime §10.5):
blocks captured between begin() and finish() land in ``<tmp>/audio.wav`` with the
alignment block; abort leaves nothing; a silent reader yields ``None``."""

from __future__ import annotations

import wave

import numpy as np

from apollo_mavis_v2_runtime.recorder.audio import EpisodeAudioSink


class StubReader:
    """The slice of MicrophoneReader the sink uses: cfg.sample_rate, add/remove_sink, status()."""

    class _Cfg:
        sample_rate = 48000
        backend = "fake"

    def __init__(self) -> None:
        self.cfg = self._Cfg()
        self.sinks = []
        self.overruns = 0
        self.active = True

    def add_sink(self, fn):
        self.sinks.append(fn)

    def remove_sink(self, fn):
        self.sinks = [s for s in self.sinks if s is not fn]

    def status(self, now=None):
        class _St:
            overruns = self.overruns
            status = "live"

        return _St()

    def push(self, samples, rx_mono):
        for fn in list(self.sinks):
            fn(samples, rx_mono)


def test_finish_writes_wav_and_alignment_block(tmp_path):
    reader = StubReader()
    clock = {"t": 100.0}
    sink = EpisodeAudioSink(reader, clock=lambda: clock["t"])
    sink.begin()
    assert reader.sinks == [sink._sink_fn]  # one stable object: remove_sink is by identity
    block = np.linspace(-0.5, 0.5, 1920, dtype=np.float32)
    reader.push(block, 100.04)
    reader.push(block, 100.08)
    reader.overruns = 1
    meta = sink.finish(tmp_path / "audio.wav", tmp_path)
    assert reader.sinks == []  # detached
    assert meta["path"] == "audio.wav" and meta["format"] == "wav"
    assert meta["sample_format"] == "pcm_s16le" and meta["channels"] == 1
    assert meta["sample_rate"] == 48000 and meta["samples"] == 3840 and meta["blocks"] == 2
    assert meta["duration_s"] == 3840 / 48000
    assert meta["t0_mono"] == 100.0 and meta["audio_start_mono"] == 100.04 - 1920 / 48000
    assert meta["overruns_delta"] == 1 and isinstance(meta["t0_wallclock_ns"], int)
    with wave.open(str(tmp_path / "audio.wav"), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 48000)
        assert w.getnframes() == 3840
        pcm = np.frombuffer(w.readframes(3840), dtype="<i2")
    assert pcm.min() < -16000 and pcm.max() > 16000


def test_blocks_outside_an_episode_are_ignored_and_abort_leaves_nothing(tmp_path):
    reader = StubReader()
    sink = EpisodeAudioSink(reader)
    reader.push(np.zeros(10, np.float32), 1.0)  # before begin: no-op
    sink.begin()
    reader.push(np.zeros(10, np.float32), 1.0)
    sink.abort()
    assert reader.sinks == [] and not list(tmp_path.iterdir())
    sink.begin()
    assert sink.finish(tmp_path / "audio.wav", tmp_path) is None  # nothing arrived
    assert not (tmp_path / "audio.wav").exists()
