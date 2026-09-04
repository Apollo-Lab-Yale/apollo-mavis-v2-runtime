"""FrameSources for the reserved "sim"/"twin" render streams (04-runtime §13.4).

The render thread (sim's RenderService, hosted by the runtime) owns every
``mujoco.Renderer``; this adapter only registers a stream and polls its
depth-1 frame slot. Exists only while a session runs.
"""

from __future__ import annotations

from apollo_mavis_v2_core import CameraFrame


class RenderStreamSource:
    """Adapter: RenderService stream -> VideoHub FrameSource."""

    def __init__(
        self,
        render_service,
        stream_id: str,
        source: str,  # registered render source: "sim" | "twin"
        camera: str | int | None = None,  # None = free camera
        resolution: tuple[int, int] = (640, 480),
        fps: float = 30.0,
    ) -> None:
        self._service = render_service
        self._stream_id = stream_id
        self._spec_kwargs = dict(
            stream_id=stream_id,
            source=source,
            camera=camera,
            width=resolution[0],
            height=resolution[1],
            fps=fps,
        )
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        from apollo_mavis_v2_sim import StreamSpec  # sim extra only

        self._service.add_stream(StreamSpec(**self._spec_kwargs))
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._service.remove_stream(self._stream_id)
        self._started = False

    def latest(self) -> CameraFrame | None:
        return self._service.latest(self._stream_id)


__all__ = ["RenderStreamSource"]
