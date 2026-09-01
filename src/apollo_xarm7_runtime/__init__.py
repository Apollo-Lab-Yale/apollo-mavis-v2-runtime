"""apollo-xarm7-runtime: session engine, control loop, and server (04-runtime).

Composes a workcell (hardware or sim), runs the 100 Hz control loop, owns
safety supervision, and hosts the single FastAPI app (REST + WS + video +
SPA) on one port. ``MUJOCO_GL=egl`` must be exported by the entrypoint
before any mujoco import (see ``__main__.py``).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
