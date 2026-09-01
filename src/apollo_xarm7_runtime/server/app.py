"""FastAPI app factory (04-runtime §13.5). SPA static mount comes LAST."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ..runtime import Runtime
from . import rest, ws_control, ws_telemetry, ws_video


def create_app(runtime: Runtime) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime.start()
        try:
            yield
        finally:
            runtime.stop()

    app = FastAPI(title="apollo-xarm7-runtime", lifespan=lifespan)
    app.state.runtime = runtime
    app.include_router(rest.router, prefix="/api")
    app.add_api_websocket_route("/ws/control", ws_control.endpoint)
    app.add_api_websocket_route("/ws/telemetry", ws_telemetry.endpoint)
    app.add_api_websocket_route("/ws/video/{stream_id}", ws_video.endpoint)
    app.add_route("/video/{stream_id}.mjpg", ws_video.mjpeg, methods=["GET"])
    ui_dist = runtime.cfg.ui_dist  # packaged SPA or path from config
    if ui_dist and ui_dist.exists():  # mount LAST: after every API route
        app.mount("/", StaticFiles(directory=ui_dist, html=True), name="spa")
    return app


__all__ = ["create_app"]
