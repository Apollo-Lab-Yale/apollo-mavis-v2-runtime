"""`python -m apollo_xarm7_runtime --config <runtime.yaml>` (04-runtime §13.5).

Sets ``MUJOCO_GL=egl`` (and the EGL device) BEFORE any mujoco import, then
runs a single uvicorn worker with permessage-deflate disabled (100 Hz
control channel; compression only adds CPU and buffering).
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="apollo_xarm7_runtime")
    parser.add_argument("--config", default=None, help="runtime YAML (or $APOLLO_CONFIG)")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)

    # MUST precede any mujoco import / GL init (04-runtime §13.5).
    os.environ.setdefault("MUJOCO_GL", "egl")

    from .config import load_runtime_config

    cfg = load_runtime_config(args.config)
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", str(cfg.egl_device_id))
    if args.host:
        cfg = cfg.model_copy(update={"host": args.host})
    if args.port:
        cfg = cfg.model_copy(update={"port": args.port})

    import uvicorn

    from .runtime import Runtime
    from .server.app import create_app

    app = create_app(Runtime(cfg))
    uvicorn.run(
        app,
        host=cfg.host,
        port=cfg.port,
        ws_per_message_deflate=False,  # binding: no deflate on the control channel
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
