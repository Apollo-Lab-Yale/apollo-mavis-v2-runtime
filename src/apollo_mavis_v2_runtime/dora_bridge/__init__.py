"""External interface over dora-rs 1.0 (phase-12; 14-dora-interface.md).

This package is the ONLY place in the runtime that imports ``dora`` and
``pyarrow`` (ruff ``TID251`` bans both names everywhere else; the AST scan in
``tests/dora/test_import_confinement.py`` enforces it). Both imports are LAZY:
importing this package never imports dora, so a runtime without the ``[dora]``
extra starts and serves exactly as before (``telemetry.external.state ==
"disabled"``).

Layout (14-dora §2):

- ``netaddr``        resolve / vet ``dora.bind_host`` (IPv4 literal or interface name;
                     never 0.0.0.0, never a control-box subnet).
- ``codec``          pure pyarrow encode/decode of every payload shape (unit-testable
                     without a daemon).
- ``dataflow``       render the dataflow YAML (``render_dataflow``).
- ``control_plane``  the runtime-owned PRIVATE coordinator + daemon (spawn, status,
                     start/stop the dataflow, registered-machine rescan, reap).
- ``bridge``         ``DoraBridge``: state machine + the single ``dora-bus`` thread that
                     owns the ``Node`` handle, outbound slots, inbound validation.
- ``publishers``     ``SnapshotPublisher`` (``dora-publisher`` thread), ``CameraTap``,
                     ``MicPublisher``, ``PoseStamper`` (per-frame camera pose metadata).
- ``idle_state``     ``IdleArmReader``: arm states between sessions (read-only).
- ``policy_source``  ``ExternalPolicySource``: the ``PolicySource`` fed by ``policy_action``.
- ``nodes``          probe / fake_policy / viewer_probe / rtt_probe / env helper nodes.

The 100 Hz control thread NEVER calls into this package (14-dora §2.5): it only
reads the plain Python object ``ExternalPolicySource`` the bus thread fills.
"""

from __future__ import annotations

__all__ = [
    "bridge",
    "codec",
    "control_plane",
    "dataflow",
    "idle_state",
    "netaddr",
    "policy_source",
    "publishers",
]
