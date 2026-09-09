"""Tier 4 (14-dora §7, §10): the fixed viewpoint without a command surface.

A ``viewer_probe`` consumes ``cam_view_wrist_cam`` across: no session (idle pose) -> a
teleop session that moves the Perception Arm (``view``) -> ``DELETE /api/session``. It
asserts the pose metadata on every frame, the ``loop`` -> ``idle`` flip of ``pose_source``,
that the post-session pose equals the last in-session pose, that frames keep flowing across
TEARDOWN (gap <= 2 frame periods) and that the parked arm then stays put.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import httpx
import numpy as np
import pytest
from conftest import LiveServer
from test_e2e_teleop import Ctl

from dora_bridge.harness import dora_runtime_config, node_env, requires_dora, wait_until

pytestmark = [pytest.mark.dora, pytest.mark.egl, requires_dora]

MOD = "apollo_mavis_v2_runtime.dora_bridge.nodes"
SPEC = {
    "mode": "teleop",
    "kind": "sim",
    "arms": ["view", "grip"],
    "frames": {"view": "arm_base:view", "grip": "arm_base:grip"},
    "sim_scene": "mavis_v2",
}


def test_park_the_perception_arm_then_end_the_session(tmp_path):
    cfg = dora_runtime_config(tmp_path)
    srv = LiveServer(cfg)
    try:
        with httpx.Client(base_url=srv.http, timeout=60) as api:
            info = wait_until(
                lambda: (
                    (i := api.get("/api/dora").json()) and (i if i["state"] == "attached" else None)
                ),
                10.0,
                "attached",
            )
            out = tmp_path / "viewer.json"
            probe = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    f"{MOD}.viewer_probe",
                    "--daemon-port",
                    str(info["daemon_port"]),
                    "--camera",
                    "view_wrist_cam",
                    "--duration-s",
                    "24",
                    "--warmup-s",
                    "0.5",
                    "--dump-frames",
                    "--out",
                    str(out),
                ],
                env=node_env(info["bind_host"], info["zenoh_port"]),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            time.sleep(4.0)  # idle frames first
            # -- the session: drive the Perception Arm to a new pose ------------------------------
            r = api.post("/api/session", json=SPEC)
            assert r.status_code == 200, r.text
            wait_until(
                lambda: api.get("/api/session").json()["state"] == "running", 20.0, "running"
            )
            ctl = Ctl(srv)
            try:
                # the default active arm is the Manipulation Arm (grip); Tab to the Perception Arm
                ack = ctl.action("switch_arm")
                assert ack["ok"], ack
                ctl.hold(["KeyW", "KeyE"], 2.0)  # forward + up for 2 s
                ctl.keys([])
                time.sleep(2.0)  # zero-twist ramp, then the sim PD settles below 1e-6 rad
                snap = srv.runtime.bus.snapshot.get()[0]
                q_last = np.asarray(snap.arms["view"].q, dtype=np.float64)
                t_delete = time.monotonic()
            finally:
                ctl.close()
            assert api.delete("/api/session").status_code == 204
            t_after = time.monotonic()
            probe.wait(timeout=60)
            assert probe.returncode == 0
        r = json.loads(out.read_text())
        assert r["errors"] == [], r["errors"]
        assert r["pose_missing"] == [], r["pose_missing"][:3]
        assert r["pose_sources"] == ["idle", "loop", "idle"], r["pose_sources"]
        frames = r["frames"]
        assert len(frames) > 60
        loop_frames = [f for f in frames if f["pose_source"] == "loop"]
        after = [f for f in frames if f["pose_source"] == "idle" and f["t"] > loop_frames[-1]["t"]]
        assert loop_frames and len(after) >= 30
        # frames kept flowing across TEARDOWN: no inter-arrival above 2 frame periods (15 fps)
        # around the loop -> idle switch (the session cameras end, the previews take over)
        window = [f for f in frames if loop_frames[-1]["t"] - 1.0 <= f["t"] <= after[0]["t"] + 1.0]
        gaps = np.diff([f["t"] for f in window])
        assert gaps.max() <= 2.0 / cfg.video.preview_fps + 0.02, (
            f"max gap {gaps.max() * 1e3:.0f} ms"
        )
        # the parked pose: the first idle frames repeat the last in-session pose exactly
        last_loop = np.asarray(loop_frames[-1]["camera_pose_world"])
        first_idle = np.asarray(after[0]["camera_pose_world"])
        # the parked joints are captured AFTER the loop thread stopped (its final snapshot is
        # what the last frames were stamped with): measured 0.000 mm (2026-09-08); an earlier
        # capture before loop.stop() left 4.6e-5 m of sim PD creep between the two samples
        delta = float(np.max(np.abs(first_idle - last_loop)))
        assert delta <= 1e-6, (delta, last_loop, first_idle)
        print(f"\nparked pose vs last in-session frame: {delta * 1e3:.3f} mm")
        assert np.max(np.abs(np.asarray(after[0]["q"]) - q_last)) <= 1e-6
        assert after[0]["session_id"] == "" and loop_frames[0]["session_id"] != ""
        # ... and stays put for the rest of the run
        poses = np.asarray([f["camera_pose_world"] for f in after])
        assert np.max(np.abs(poses - poses[0])) <= 1e-9
        assert r["arm_state"]["source"] == "idle" and r["arm_state"]["max_abs_dq"] == 0.0
        # the arm actually moved during the session (the parked pose is a NEW viewpoint)
        idle_before = [
            f for f in frames if f["pose_source"] == "idle" and f["t"] < loop_frames[0]["t"]
        ]
        assert idle_before
        moved = np.max(
            np.abs(np.asarray(idle_before[-1]["camera_pose_world"][:3]) - first_idle[:3])
        )
        assert moved > 0.01, f"the Perception Arm's camera moved only {moved * 1e3:.1f} mm"
        assert t_after - t_delete < 10.0
    finally:
        srv.stop()
