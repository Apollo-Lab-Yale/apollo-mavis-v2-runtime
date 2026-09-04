# apollo-mavis-v2-runtime

Session engine, 100 Hz control loop, safety supervisor, recorder and FastAPI
server for the MAVIS v2 cell (two xArm7 arms on linear tracks: the
**Manipulation Arm** `grip` with xArm Gripper G2 + wrist camera — the default
teleop arm — and the **Perception Arm** `view` with wrist RealSense D435 +
microphone), in simulation (`apollo-mavis-v2-sim`) or on the real hardware
(`apollo-mavis-v2-hardware`; control boxes 192.168.1.201 / 192.168.2.219).
Run: `uv run python -m apollo_mavis_v2_runtime --config <yaml>`; design in
`docs/design/04-runtime.md`.

## DAgger / online fine-tuning notes (phase-08)

- **Between-session hygiene (recommended).** Online fine-tuning drifts; the
  50/50 aggregate mixing, sanity gate and `LAST_KNOWN_GOOD` rollback only
  bound it. The promoted deployment artifact should come from round-based
  retraining **from the full aggregate** between sessions
  (`apollo-dagger-retrain`, 12-dagger §9 — offline CLI, TODO), never from an
  online-fine-tuned checkpoint. HG-DAgger's per-round full retrain is the
  safe fallback.
- **Label-quality risk.** Keyboard corrections are lower-quality expert
  labels than a spacemouse or leader arm (HG-DAgger label-quality argument):
  discrete axis-aligned twists, reflexive first reactions. `T_blend`
  transition exclusion trims the reflexive prefix, but expect noisier labels
  than kinesthetic teaching; prefer short, deliberate takeovers.
- **Inference mode never records.** The takeover there is a safety escape;
  no recorder object exists in that session (structural, not a flag).
