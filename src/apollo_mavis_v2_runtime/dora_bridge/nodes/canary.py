"""Start-barrier canary (14-dora §16.1 join protocol).

dora 1.0.1 holds EVERY node of a multi-machine dataflow - spawned and dynamic alike - until
each remote dynamic placeholder has been attached, and a ``Node()`` attach made while that
barrier is closed blocks forever HOLDING THE GIL. The bridge therefore never attaches
``mavis_runtime`` after a restart with remote placeholders until this process, attached as
the lab-side dynamic node ``canary``, has come back: it returns 0 the moment the barrier is
open (its ``Node()`` returns) and simply gets killed by the bridge if the window expires.
A hang here costs a subprocess, never the runtime.
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--daemon-port", type=int, required=True)
    p.add_argument("--node-id", default="canary")
    a = p.parse_args(argv)
    os.environ.setdefault("RUST_LOG", "error")
    from dora import Node  # noqa: TID251 - node process

    node = Node(a.node_id, daemon_port=a.daemon_port)  # blocks while the barrier is closed
    del node
    return 0


if __name__ == "__main__":
    sys.exit(main())
