"""``viewer_probe`` — the fixed-viewpoint acceptance instrument (14-dora §7, §10).

Attaches as ``viewer`` (or ``--node-id viewer_<machine>`` on a remote daemon),
consumes ``cam_<camera>`` (+ ``_depth``) and ``arm_state`` for ``--duration-s``
and writes a JSON report: per stream count, rate, seq gaps, one-hop latency
(``time.monotonic() - t_mono``, same host clock), pose-metadata completeness on
every non-prewarm camera frame (``camera_pose_world`` 7, ``tcp_pose_world`` 7,
``q`` 8, ``intrinsics`` 4, ``pose_source``), the last ``camera_pose_world`` /
``q`` / ``pose_source`` seen and the per-frame inter-arrival maximum. numpy +
pyarrow + dora only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from . import node_env_defaults


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="viewer_probe")
    p.add_argument("--daemon-port", type=int, required=True)
    p.add_argument("--node-id", default="viewer")
    p.add_argument("--camera", default="view_wrist_cam", help="comma list of camera ids")
    p.add_argument("--duration-s", type=float, default=10.0)
    p.add_argument("--out", required=True)
    p.add_argument("--first-timeout-s", type=float, default=10.0)
    p.add_argument("--warmup-s", type=float, default=1.0, help="discard stats before this")
    p.add_argument("--dump-frames", action="store_true", help="per-frame rows of the first camera")
    p.add_argument("--stop-on-pose-change", action="store_true", help="exit once pose_source flips")
    return p


class _Stream:
    def __init__(self) -> None:
        self.n = 0
        self.gaps = 0
        self.gap_list: list[list[int]] = []
        self.last_seq: int | None = None
        self.lat_ms: list[float] = []
        self.arrivals: list[float] = []
        self.bytes = 0
        self.frame_age_ms: list[float] = []  # cameras: now - capture time (encoder poll + hop)

    def note(self, seq: int, t_mono: float, nbytes: int, now: float) -> None:
        if self.last_seq is not None and seq != self.last_seq + 1:
            self.gaps += 1
            if len(self.gap_list) < 20:
                self.gap_list.append([self.last_seq, seq])
        self.last_seq = seq
        self.n += 1
        self.lat_ms.append((now - t_mono) * 1e3)
        self.arrivals.append(now)
        self.bytes += nbytes

    def summary(self) -> dict:
        lat = np.asarray(self.lat_ms)
        arr = np.asarray(self.arrivals)
        dt = np.diff(arr) if arr.size > 1 else np.zeros(0)
        return {
            "n": self.n,
            "gaps": self.gaps,
            "gap_list": self.gap_list,
            "rate_hz": float((self.n - 1) / (arr[-1] - arr[0]))
            if self.n > 1 and arr[-1] > arr[0]
            else None,
            "bytes_total": self.bytes,
            "lat_ms": None
            if lat.size == 0
            else {
                "p50": float(np.percentile(lat, 50)),
                "p90": float(np.percentile(lat, 90)),
                "p99": float(np.percentile(lat, 99)),
                "max": float(lat.max()),
                "mean": float(lat.mean()),
            },
            "interarrival_ms": None
            if dt.size == 0
            else {
                "p50": float(np.percentile(dt, 50) * 1e3),
                "max": float(dt.max() * 1e3),
            },
            "frame_age_ms": None
            if not self.frame_age_ms
            else {
                "p50": float(np.percentile(self.frame_age_ms, 50)),
                "p99": float(np.percentile(self.frame_age_ms, 99)),
                "max": float(max(self.frame_age_ms)),
            },
        }


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    node_env_defaults()
    from dora import Node  # noqa: TID251 - node process

    cams = [c.strip() for c in a.camera.split(",") if c.strip()]
    cam_ids = {f"cam_{c}" for c in cams}
    depth_ids = {f"cam_{c}_depth" for c in cams}
    # Pay pyarrow's lazy set-up BEFORE the first steady-state frame: the first
    # ``Array.to_numpy()`` of a process costs ~190 ms (measured 2026-09-08, "body" stall in
    # ``slow_iterations``) and every camera input is ``queue_size: 1``, so a stall that long
    # discards the 2-3 frames behind it (dora: "Discarding event ... due to queue size
    # limit") - the one seq gap the LAN acceptance runs kept showing 0.1-0.4 s after warm-up.
    import pyarrow as pa  # noqa: TID251 - node process

    pa.array([0.0]).to_numpy()
    t_attach0 = time.monotonic()
    node = Node(a.node_id, daemon_port=a.daemon_port)
    attach_s = time.monotonic() - t_attach0
    streams = {"arm_state": _Stream(), "session": _Stream()}
    for c in cams:
        streams[f"cam_{c}"] = _Stream()
        streams[f"cam_{c}_depth"] = _Stream()
    pose_missing: list[dict] = []
    pose_ok = 0
    last_pose: dict | None = None
    prewarm = 0
    sessions: list[dict] = []
    arm_meta: dict | None = None
    errors: list[str] = []
    first_t: float | None = None
    deadline = None
    first_deadline = time.monotonic() + a.first_timeout_s
    pose_sources: list[str] = []
    frames: list[dict] = []
    first_cam = f"cam_{cams[0]}"
    slow: list[dict] = []  # iterations > 20 ms: where a consumer stall comes from
    prev: tuple[float, str, object] | None = None  # (now, iid, seq) of the previous event
    while True:
        t0 = time.monotonic()
        if prev is not None and t0 - prev[0] > 0.02 and len(slow) < 40:
            slow.append(
                {
                    "where": "body",
                    "ms": round((t0 - prev[0]) * 1e3, 1),
                    "iid": prev[1],
                    "seq": prev[2],
                    "wall": time.time(),
                    "t_rel": round(t0 - first_t, 3) if first_t is not None else None,
                }
            )
        ev = node.next(timeout=1.0)
        now = time.monotonic()
        dt_next = now - t0
        if first_t is None and now > first_deadline:
            errors.append("no input before first_timeout_s")
            break
        if deadline is not None and now > deadline:
            break
        if ev is None:
            errors.append("next() returned None")
            break
        kind = ev.get("type")
        if kind == "ERROR":
            text = str(ev.get("error") or "")
            if "Receiver timed out" not in text:
                errors.append(text[:200])
            continue
        if kind == "STOP":
            errors.append("STOP")
            break
        if kind != "INPUT":
            continue
        iid = ev["id"]
        meta = ev.get("metadata") or {}
        prev = (now, iid, meta.get("seq"))
        if dt_next > 0.02 and first_t is not None and len(slow) < 40:
            slow.append(
                {
                    "where": "next",
                    "ms": round(dt_next * 1e3, 1),
                    "iid": iid,
                    "seq": meta.get("seq"),
                    "wall": time.time(),
                    "t_rel": round(now - first_t, 3),
                }
            )
        if first_t is None:
            first_t = now
            deadline = now + a.duration_s
        st = streams.get(iid)
        if st is None:
            continue
        if meta.get("prewarm"):
            prewarm += 1
            continue
        if now - first_t < a.warmup_s:
            _ = ev["value"]  # zenoh / SHM warm-up: touch the payload, keep it out of the stats
            continue
        value = ev["value"]
        nbytes = int(getattr(value, "nbytes", 0))
        st.note(
            int(meta.get("seq", 0)),
            float(meta.get("send_t_mono", meta.get("t_mono", now))),
            nbytes,
            now,
        )
        if iid in cam_ids or iid in depth_ids:
            st.frame_age_ms.append((now - float(meta.get("frame_t_mono", now))) * 1e3)
            need = {"camera_pose_world": 7, "tcp_pose_world": 7, "q": 8, "intrinsics": 4}
            bad = {
                k: len(meta.get(k) or []) for k, n in need.items() if len(meta.get(k) or []) != n
            }
            if "pose_source" not in meta:
                bad["pose_source"] = 0
            if bad:
                if len(pose_missing) < 20:
                    pose_missing.append({"stream": iid, "seq": meta.get("seq"), "bad": bad})
            else:
                pose_ok += 1
                last_pose = {
                    "camera_pose_world": meta["camera_pose_world"],
                    "tcp_pose_world": meta["tcp_pose_world"],
                    "q": meta["q"],
                    "pose_source": meta["pose_source"],
                    "frame_seq": meta.get("frame_seq"),
                    "session_id": meta.get("session_id"),
                    "t_mono": meta.get("t_mono"),
                    "width": meta.get("width"),
                    "height": meta.get("height"),
                    "encoding": meta.get("encoding"),
                }
                src = str(meta["pose_source"])
                if a.dump_frames and iid == first_cam:
                    frames.append(
                        {
                            "t": now,
                            "wall": time.time(),  # correlate with dora's own log timestamps
                            "seq": meta.get("seq"),
                            "frame_seq": meta.get("frame_seq"),
                            "pose_source": src,
                            "session_id": meta.get("session_id"),
                            "camera_pose_world": meta["camera_pose_world"],
                            "q": meta["q"],
                        }
                    )
                if not pose_sources or pose_sources[-1] != src:
                    pose_sources.append(src)
                    if a.stop_on_pose_change and len(pose_sources) > 1:
                        deadline = now + 1.0
        if iid == "arm_state":
            arm_meta = {
                k: meta.get(k)
                for k in ("arm_ids", "source", "session_id", "tick", "stale", "layout")
            }
            vals = value.to_numpy(zero_copy_only=False)
            arm_meta["n_values"] = int(vals.shape[0])
            n_arms = len(meta.get("arm_ids") or [])
            if n_arms:
                blocks = vals.reshape(n_arms, -1)
                arm_meta["max_abs_dq"] = float(np.max(np.abs(blocks[:, 8:15])))
                arm_meta["q"] = blocks[:, :8].tolist()
        elif iid == "session":
            try:
                sessions.append(json.loads(value[0].as_py()))
                del sessions[:-5]
            except Exception:  # noqa: BLE001
                pass
    report = {
        "node_id": a.node_id,
        "daemon_port": a.daemon_port,
        "attach_s": attach_s,
        "duration_s": a.duration_s,
        "streams": {k: v.summary() for k, v in streams.items()},
        "pose_ok_frames": pose_ok,
        "pose_missing": pose_missing,
        "prewarm_frames": prewarm,
        "slow_iterations": slow,
        "last_pose": last_pose,
        "pose_sources": pose_sources,
        "arm_state": arm_meta,
        "sessions": [
            {"session_id": s.get("session_id"), "state": s.get("state")} for s in sessions
        ],
        "errors": errors,
        "frames": frames,
    }
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(
        json.dumps(
            {k: report[k] for k in ("pose_ok_frames", "prewarm_frames", "pose_sources", "errors")}
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
