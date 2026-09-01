# apollo-xarm7-runtime

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
