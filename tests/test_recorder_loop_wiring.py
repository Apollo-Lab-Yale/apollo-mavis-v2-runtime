"""ControlLoop <-> recorder wiring: episode ops route to the recorder, status
flows to the snapshot, joint_target is nacked while recording (04-runtime §10)."""

from __future__ import annotations

from apollo_mavis_v2_core import Command
from apollo_mavis_v2_core.protocol import EpisodeStatus
from conftest import run_ticks


class FakeRecorderThread:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.state = "idle"

    def request(self, op):
        self.calls.append(op)
        if op == "new":
            self.state = "recording"
            return True, "episode 0"
        if op == "save":
            if self.state != "recording":
                return False, self.state
            self.state = "saving"
            return True, "saving"
        if op == "discard":
            return (self.state == "recording"), "discarding"
        return False, "?"

    def status(self):
        return EpisodeStatus(state=self.state, index=0, frames=3, duration_s=0.12)


def submit(bus, op, **args):
    return bus.commands.submit(Command(op=op, args=args))


def test_episode_ops_route_and_status_flows(fake_loop):
    cell, bus, loop = fake_loop
    rec = FakeRecorderThread()
    loop.recorder = rec

    f = submit(bus, "episode_new")
    snap = run_ticks_snap(loop, cell)
    assert f.result(0).ok and rec.calls == ["new"]
    assert snap.episode.state == "recording"
    assert loop.episode_state == "recording"

    # joint_target nacked with detail "recording" while recording (§7.3)
    fj = bus.commands.submit(
        Command(op="joint_target",
                args={"arm_id": "arm0", "positions": [0.0] * 8, "mode": "jog"})
    )
    run_ticks(loop, cell, 1)
    res = fj.result(0)
    assert not res.ok and res.detail == "recording"

    fs = submit(bus, "episode_save")
    run_ticks(loop, cell, 1)
    assert fs.result(0).ok and rec.calls[-1] == "save"


def test_episode_ops_nacked_without_recorder(fake_loop):
    cell, bus, loop = fake_loop
    assert loop.recorder is None
    f = submit(bus, "episode_new")
    run_ticks(loop, cell, 1)
    res = f.result(0)
    assert not res.ok and "recorder" in res.detail


def run_ticks_snap(loop, cell):
    run_ticks(loop, cell, 1)
    return loop.bus.snapshot.get()[0]
