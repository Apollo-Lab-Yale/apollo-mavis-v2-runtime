"""``probe`` — the ONLY spawned node of the runtime dataflow (14-dora §2.3, §8).

Keeps the dataflow ``Running`` (the daemon finishes a dataflow when every
spawned node has exited) and is the liveness canary: 1 Hz ``heartbeat``
(``Int64[1]`` = seq) driven by ``dora/timer/hz/1``. Spawned by the daemon, so
``Node()`` reads its id from the environment. Exits on ``STOP`` / ``None``.
"""

from __future__ import annotations

import sys
import time

from . import node_env_defaults


def main() -> int:
    node_env_defaults()
    import pyarrow as pa  # noqa: TID251 - node process
    from dora import Node  # noqa: TID251 - node process

    node = Node()
    seq = 0
    while True:
        ev = node.next(timeout=5.0)
        if ev is None:
            return 0
        kind = ev.get("type")
        if kind == "INPUT" and ev.get("id") == "tick":
            seq += 1
            node.send_output(
                "heartbeat",
                pa.array([seq], type=pa.int64()),
                {"t_mono": time.monotonic(), "wallclock_ns": time.time_ns(), "seq": seq},
            )
        elif kind == "STOP":
            return 0
        elif kind == "ERROR":
            text = str(ev.get("error") or "")
            if "Receiver timed out" in text:
                continue
            if "daemon channel broken" in text or "fatal" in text:
                return 1


if __name__ == "__main__":
    sys.exit(main())
