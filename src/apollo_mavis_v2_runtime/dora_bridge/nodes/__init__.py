"""Helper dora nodes (14-dora §2.3, §10): ``probe`` (the one spawned node),
``fake_policy`` (CI stand-in for the policy repo), ``viewer_probe`` /
``rtt_probe`` (acceptance instruments) and ``env`` (shell export helper).
Every node sets ``DORA_ZENOH_MULTICAST=off`` / ``DORA_ZENOH_LISTEN=tcp/127.0.0.1:0``
before creating its ``Node`` (§9 hygiene)."""

from __future__ import annotations

import os


def node_env_defaults() -> None:
    """§9: never open multicast / per-NIC UDP / a wildcard TCP listener from a node."""
    os.environ.setdefault("RUST_LOG", "error")
    os.environ.setdefault("DORA_ZENOH_MULTICAST", "off")
    os.environ.setdefault("DORA_ZENOH_LISTEN", "tcp/127.0.0.1:0")


__all__ = ["node_env_defaults"]
