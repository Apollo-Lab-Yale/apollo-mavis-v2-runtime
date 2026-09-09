"""Fixed-viewpoint consumer in ~30 lines (14-dora §7): the starting point for another
robot's code that wants the Perception Arm's camera + pose.

    eval "$(python -m apollo_mavis_v2_runtime.dora_bridge.nodes.env)"   # or GET /api/dora
    python examples/viewer_node.py [daemon_port]
"""

import os
import sys

import numpy as np
import pyarrow as pa
from dora import Node

# Pay pyarrow's numpy set-up BEFORE attaching: the first `to_numpy()` of a process costs ~190 ms
# and every camera input is `queue_size: 1`, so paying it on the first real frame drops the 2-3
# frames behind it (14-dora §16.1).
pa.array([0.0]).to_numpy()

os.environ.setdefault("DORA_ZENOH_MULTICAST", "off")  # 14-dora §9 node hygiene
os.environ.setdefault("DORA_ZENOH_LISTEN", "tcp/127.0.0.1:0")
port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("DORA_DAEMON_PORT", "53391"))
node = Node("viewer", daemon_port=port)  # the placeholder declared by the runtime's dataflow
for ev in node:
    if ev["type"] == "STOP":
        break
    if ev["type"] != "INPUT" or ev["id"] != "cam_view_wrist_cam":
        continue
    m = ev["metadata"]
    if m.get("prewarm"):
        continue
    img = ev["value"].to_numpy(zero_copy_only=False).reshape(m["height"], m["width"], 3)
    T_W_C = np.asarray(m["camera_pose_world"])  # [x, y, z, qw, qx, qy, qz], OpenCV convention
    K = m.get("intrinsics")  # [fx, fy, cx, cy]
    print(
        f"frame {m['frame_seq']} {img.shape} pose_source={m['pose_source']} cam@{T_W_C[:3]} K={K}"
    )
