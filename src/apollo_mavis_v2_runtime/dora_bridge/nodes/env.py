"""``env`` — print the ``export DORA_*`` lines a same-host client needs (14-dora §2.6, §9).

Reads ``GET /api/dora`` from a running runtime (``--url``, default
``http://127.0.0.1:8765``) and prints::

    export DORA_ZENOH_CONNECT=tcp/<bind_host>:<zenoh_port>
    export DORA_ZENOH_MULTICAST=off
    export DORA_ZENOH_LISTEN=tcp/127.0.0.1:0
    export DORA_DAEMON_PORT=<daemon_port>
    export DORA_COORDINATOR_ADDR=<bind_host>   DORA_COORDINATOR_PORT=<P>   (remote daemons)
    export DORA_AUTH_TOKEN=<token>             (only when --var-dir holds a readable .dora-token)

The token is never part of the REST response (§9); it is read from the
runtime's ``var_dir`` on the lab host (``--var-dir``, default
``${APOLLO_HOME}/var/dora``) and handed to remote operators out of band.
Usage: ``eval "$(python -m apollo_mavis_v2_runtime.dora_bridge.nodes.env)"``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dora env")
    p.add_argument("--url", default=os.environ.get("APOLLO_RUNTIME_URL", "http://127.0.0.1:8765"))
    p.add_argument("--var-dir", default=None, help="runtime dora var_dir (for the auth token)")
    p.add_argument("--json", action="store_true", help="print the raw GET /api/dora document")
    return p


def fetch_info(url: str) -> dict:
    with urllib.request.urlopen(f"{url.rstrip('/')}/api/dora", timeout=5.0) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def default_var_dir() -> Path | None:
    home = os.environ.get("APOLLO_HOME")
    if home:
        return Path(home) / "var" / "dora"
    try:
        from ...config import apollo_home

        root = apollo_home()
    except Exception:  # noqa: BLE001
        return None
    return Path(root) / "var" / "dora" if root else None


def export_lines(info: dict, var_dir: Path | None) -> list[str]:
    host = info.get("bind_host", "127.0.0.1")
    lines = [
        "export DORA_ZENOH_CONNECT="
        + str(info.get("zenoh_connect") or f"tcp/{host}:{info.get('zenoh_port', 7447)}"),
        "export DORA_ZENOH_MULTICAST=off",
        "export DORA_ZENOH_LISTEN=tcp/127.0.0.1:0",
        f"export DORA_DAEMON_PORT={info.get('daemon_port', 53391)}",
        f"export DORA_COORDINATOR_ADDR={info.get('coordinator_addr', host)}",
        f"export DORA_COORDINATOR_PORT={info.get('coordinator_port', 6113)}",
    ]
    if info.get("auth") and var_dir is not None:
        token_path = var_dir / ".dora-token"
        try:
            token = token_path.read_text(encoding="utf-8").strip()
        except OSError:
            lines.append(
                f"# auth is on but {token_path} is not readable here: get DORA_AUTH_TOKEN "
                "from the lab operator"
            )
        else:
            if token:
                lines.append(f"export DORA_AUTH_TOKEN={token}")
    lines.append(
        f"# node id for a same-host policy: policy ; state: {info.get('state')} ; "
        f"dataflow: {info.get('dataflow_id')}"
    )
    return lines


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        info = fetch_info(a.url)
    except Exception as exc:  # noqa: BLE001
        print(f"# cannot reach {a.url}/api/dora: {exc}", file=sys.stderr)
        return 1
    if a.json:
        print(json.dumps(info, indent=2))
        return 0
    var_dir = Path(a.var_dir) if a.var_dir else default_var_dir()
    print("\n".join(export_lines(info, var_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
