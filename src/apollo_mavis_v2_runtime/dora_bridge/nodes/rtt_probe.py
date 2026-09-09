"""``rtt_probe`` — two-hop RTT instrument (14-dora §0, §10, §12).

Plays the RUNTIME's role on a private dataflow (attaches as ``mavis_runtime``
while no runtime process is attached): publishes ``obs_state`` at ``--hz``,
runs against a ``fake_policy --mode echo`` attached as ``policy``, and measures
``policy_action`` receive time minus the ``obs_state`` publish time for the
echoed ``observation_id`` (both on this host's monotonic clock) plus seq gaps
and lost round trips. Writes a JSON report. numpy + pyarrow + dora only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from . import node_env_defaults


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rtt_probe")
    p.add_argument("--daemon-port", type=int, required=True)
    p.add_argument("--node-id", default="mavis_runtime")
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--hz", type=float, default=30.0)
    p.add_argument("--dim", type=int, default=16)
    p.add_argument("--action-dim", type=int, default=8)
    p.add_argument("--session-id", default="rtt")
    p.add_argument("--out", required=True)
    p.add_argument("--settle-s", type=float, default=3.0, help="wait for the policy's spec")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    node_env_defaults()
    import pyarrow as pa  # noqa: TID251 - node process
    from dora import Node  # noqa: TID251 - node process

    node = Node(a.node_id, daemon_port=a.daemon_port)
    names = [
        f"grip_{d}" for d in ("dx", "dy", "dz", "drx", "dry", "drz", "gripper.pos", "rail.dpos")
    ][: a.action_dim]
    session = {
        "mavis_schema": 1,
        "epoch": "rtt",
        "session_id": a.session_id,
        "state": "running",
        "spec": None,
        "kind": "sim",
        "arm_ids": ["grip"],
        "has_rail": {"grip": True},
        "frames": {"grip": "arm_base:grip"},
        "action_space": "delta_ee",
        "action_names": names,
        "state_names": [f"grip_s{i}" for i in range(a.dim)],
        "camera_ids": [],
        "cameras": {},
        "policy_source": "external",
    }
    seq = 0

    def send(oid: str, arr, meta: dict) -> None:
        nonlocal seq
        seq += 1
        base = {
            "mavis_schema": 1,
            "epoch": "rtt",
            "session_id": a.session_id,
            "seq": seq,
            "t_mono": time.monotonic(),
            "wallclock_ns": time.time_ns(),
        }
        base.update(meta)
        node.send_output(oid, arr, base)

    # settle: announce the session until a spec arrives (or settle_s passes)
    spec_seen = False
    t_end = time.monotonic() + a.settle_s
    while time.monotonic() < t_end and not spec_seen:
        send("session", pa.array([json.dumps(session)], type=pa.string()), {"state": "running"})
        ev = node.next(timeout=0.5)
        if ev and ev.get("type") == "INPUT" and ev.get("id") == "policy_spec":
            spec_seen = True
    sent: dict[int, float] = {}
    rtts: list[float] = []
    lost = 0
    last_action_seq: int | None = None
    action_gaps = 0
    period = 1.0 / a.hz
    next_t = time.monotonic()
    state = np.zeros(a.dim, dtype=np.float32)
    for i in range(1, a.count + 1):
        now = time.monotonic()
        if now < next_t:
            time.sleep(next_t - now)
        next_t += period
        t_send = time.monotonic()
        sent[i] = t_send
        send(
            "obs_state",
            pa.array(state, type=pa.float32()),
            {
                "observation_id": i,
                "tick": i,
                "state_names": session["state_names"],
                "arm_ids": ["grip"],
                "frames": ["arm_base:grip"],
                "has_rail": [1],
                "image_camera_ids": [],
                "image_seq": [],
                "engaged_arm": "",
                "episode_state": "recording",
                "quat_order": "wxyz",
            },
        )
        # drain replies until the next send is due (a timeout returns an ERROR event, not None)
        while time.monotonic() < next_t:
            ev = node.next(timeout=max(0.001, next_t - time.monotonic()))
            if ev is None:
                break
            if ev.get("type") == "STOP":
                break
            if ev.get("type") != "INPUT" or ev.get("id") != "policy_action":
                continue
            m = ev.get("metadata") or {}
            oid = int(m.get("observation_id", -1))
            s = int(m.get("seq", 0))
            if last_action_seq is not None and s != last_action_seq + 1:
                action_gaps += 1
            last_action_seq = s
            t0 = sent.pop(oid, None)
            if t0 is not None:
                rtts.append((time.monotonic() - t0) * 1e3)
    # tail: give in-flight replies 0.5 s
    t_end = time.monotonic() + 0.5
    while time.monotonic() < t_end and sent:
        ev = node.next(timeout=0.1)
        if ev and ev.get("type") == "INPUT" and ev.get("id") == "policy_action":
            m = ev.get("metadata") or {}
            t0 = sent.pop(int(m.get("observation_id", -1)), None)
            if t0 is not None:
                rtts.append((time.monotonic() - t0) * 1e3)
    lost = len(sent)
    arr = np.asarray(rtts)
    report = {
        "count": a.count,
        "hz": a.hz,
        "spec_seen": spec_seen,
        "replies": int(arr.size),
        "lost": lost,
        "action_seq_gaps": action_gaps,
        "rtt_ms": None
        if arr.size == 0
        else {
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(arr.max()),
            "min": float(arr.min()),
            "mean": float(arr.mean()),
        },
    }
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
