"""AsyncTrainer process package (12-dagger §7).

Run as a separate OS process: ``python -m apollo_mavis_v2_runtime.dagger.trainer
--config <json> [--resume]``. Crash-isolated from the servo loop; channels are
the dataset spool dir (read-only), the checkpoint dir, and a ZMQ REP control
socket. ``checkpoints`` is also imported by the runtime's reloader.
"""
