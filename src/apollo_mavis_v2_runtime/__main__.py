"""`python -m apollo_mavis_v2_runtime --config <runtime.yaml>` (04-runtime §13.5).

Sets ``MUJOCO_GL=egl`` (and the EGL device) BEFORE any mujoco import, then
runs a single uvicorn worker with permessage-deflate disabled (100 Hz
control channel; compression only adds CPU and buffering).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import LoggingConfig

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
# Third-party loggers that flood at DEBUG without saying anything about the cell;
# they are held at INFO whatever ``logging.level`` says.
_QUIET_AT_DEBUG = ("asyncio", "PIL", "urllib3", "httpcore", "httpx", "websockets", "watchfiles")


def configure_logging(level: int = logging.INFO) -> bool:
    """INFO to stderr with a sane format unless the root logger is already
    configured (embedding apps / tests keep their own handlers). Returns True
    when this call installed the handler."""
    root = logging.getLogger()
    if root.handlers:
        return False
    logging.basicConfig(level=level, format=LOG_FORMAT, stream=sys.stderr)
    return True


def configure_file_logging(cfg: LoggingConfig) -> RotatingFileHandler | None:
    """Apply :class:`LoggingConfig` (2026-09-07): set the root level and add a
    :class:`RotatingFileHandler` at ``cfg.dir / cfg.file`` (``None`` when
    ``cfg.dir`` is null or a file handler is already installed). The stderr
    handler from :func:`configure_logging` stays, so journald / the launcher's
    raw capture keep seeing the same lines. Returns the handler it added.
    Third-party loggers that flood at DEBUG are pinned at INFO."""
    root = logging.getLogger()
    root.setLevel(cfg.level)
    for name in _QUIET_AT_DEBUG:
        logging.getLogger(name).setLevel(max(logging.INFO, root.level))
    if cfg.dir is None or any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        return None
    cfg.dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        cfg.dir / cfg.file,
        maxBytes=cfg.max_bytes,
        backupCount=cfg.backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(handler)
    return handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="apollo_mavis_v2_runtime")
    parser.add_argument("--config", default=None, help="runtime YAML (or $APOLLO_CONFIG)")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)
    configure_logging()  # tracker/reader warnings must reach stderr (13-tracker §4)

    # MUST precede any mujoco import / GL init (04-runtime §13.5).
    os.environ.setdefault("MUJOCO_GL", "egl")

    from .config import load_runtime_config

    cfg = load_runtime_config(args.config)
    handler = configure_file_logging(cfg.logging)
    logging.getLogger(__name__).info(
        "runtime starting: config %s, log level %s, file %s (rotate at %d MB x %d)",
        args.config or os.environ.get("APOLLO_CONFIG") or "<defaults>",
        cfg.logging.level,
        handler.baseFilename if handler is not None else "<stderr only>",
        cfg.logging.max_bytes // (1024 * 1024),
        cfg.logging.backup_count,
    )
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
        # log_config=None: uvicorn installs NO handlers of its own, so its lines
        # propagate to the root handlers above (stderr + rotating file) instead of
        # a private stderr handler the file never sees. The per-request access log
        # (three UI poll endpoints at 25 Hz were a quarter of the old file) is off
        # unless ``logging.access_log`` asks for it.
        log_config=None,
        log_level=cfg.logging.level.lower(),
        access_log=cfg.logging.access_log,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
