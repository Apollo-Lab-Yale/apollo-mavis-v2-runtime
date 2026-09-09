# PRO-DAgger on the Online DAgger shell (reference implementation)

The runtime is algorithm-agnostic; PRO-DAgger is ONE trainer you can put on its hooks. The
policy-node repo ships it as `mavis_policy_node.pro_dagger` (the reference implementation
on top of the generic `mavis_policy_node.online_dagger` loop); this note explains what it
does with each hook so you can adapt it or write your own variant. Naming rule of the
method: "PGrad", "projected gradient", "reference gradient" - never "A-GEM".

## 1. The idea in five lines

1. An **offline pool** of expert (state, action-chunk) pairs from a demonstration dataset
   provides a **reference gradient** `g_ref` (the direction that keeps the offline behaviour).
2. Every kept rollout contributes its **expert frames** (`control_mode == 1`: the operator
   corrected the novice) to an **online buffer**.
3. Every `R` kept rollouts = one **iteration**: train on the online buffer; before each
   optimiser step **project** the gradient `g` so it never points against `g_ref`
   (`g <- g - (g·g_ref / g_ref·g_ref) g_ref` when `g·g_ref < 0`), with an EMA of `g_ref`
   over the iteration.
4. Swap the weights into the acting policy, bump the version, roll out again.
5. Watch `proj_rate` (fraction of projected steps, healthy ~0.28-0.45) and the std ratio
   `scale_check_ratio` between online and offline targets (O(1)).

The shell gives you every ingredient: rollouts arrive as `on_episode_saved` (count them),
their expert frames are labelled in the parquet, the operator's take-overs are the
corrections, `train_if_due` is where the iteration runs, `swap_weights` installs it.

## 2. Hook mapping

| hook | PRO-DAgger reference (`mavis_policy_node.pro_dagger.trainer.ProDaggerTrainer`) |
|---|---|
| `on_session(session)` | reset the online buffer, the iteration counter and the EMA; load the offline dataset from YOUR config (`offline_dataset` - a repo id or a directory), build the offline pool (every `offline_stride`-th frame, normalised chunks) and `g_ref_0`; cache them under `<session_dir>/trainer/` keyed by (policy id, weights sha, dataset `modified_at`, episode count, stride) so a resume is instant. On a resume also re-read the rollouts on disk into the online buffer. The node reports `preparing` while this runs, `ready` when it returns |
| `on_episode_saved(rollout)` | read `rollout.spool_path`, keep `control_mode == 1` rows, chunk (`chunk_horizon` or the model's), apply `chunk_stride`, normalise exactly as the novice was trained, append to the online buffer (`replay_buffer: true`: it accumulates across iterations, FIFO `max_demos`, 0 = unbounded; `false`: dropped after each iteration); `pending += 1`. Refuses (`RuntimeError`) while the last `on_session` did not complete |
| `on_episode_discarded(episode_id)` | nothing landed (we read on save); a variant that pre-loads on `gate` events would drop the id here |
| `on_gate(event)` | ignored (a variant could weight the frames right after a take-over) |
| `on_train_now()` | `pending = R` (close the iteration early) |
| `train_if_due()` | `None` unless `pending >= R` (or Train now) and the buffer is non-empty; else `pending = 0`, run the PGrad iteration (§3), return `TrainResult(metrics={loss, proj_rate, n_proj, train_steps, n_samples, replay_size, train_buf, wall_s, scale_check_ratio, iteration}, detail="iteration k: ...")`; with `replay_buffer: false` the buffer is cleared afterwards. Refuses (`RuntimeError`) while the last `on_session` did not complete |
| `swap_weights()` | install the trained copy into the acting module under the policy's lock, bump `spec.version`, return it |
| `status_metrics()` | `iteration`, `pending`, `buffer`, `replay_size` + the last iteration's `loss` / `proj_rate` (counters and an atomically-replaced dict: safe on the publishing thread) |

Defaults (operator decision 2026-09-08, §0 item 7 of the design): the offline pool provides
the reference gradient ONLY (`freeze_offline_gref: true` - past interventions never join
`g_ref`'s pool), the online buffer accumulates EVERY expert intervention and is trained on
at every iteration (`replay_buffer: true`, `max_demos: 0`). The old "current iteration only,
then drop" behaviour is gone.

## 3. The iteration (`train_if_due`)

```python
def train_if_due(self):
    if self.pending < self.cfg.rollouts_per_iteration or not self.online:
        return None
    self.pending = 0
    self.iteration += 1
    self.model.load_state_dict(self.policy.model.state_dict())      # train a COPY of the acting weights
    ref_batches = self.reference_batches()                           # ONE bounded draw per pool state (ReferenceSubset)
    g_ref = self.g_ref_0 if self.iteration == 1 else None            # the cached round-1 reference
    sc = pgrad.scale_check(self.online_targets(), self.offline_targets())
    t0, last, n_proj, steps = time.monotonic(), None, 0, 0
    for ev in pgrad.pgrad_iteration(self.params, self.opt, fired=self.online, collate=self.collate,
                                    loss_fn=self.loss, replay_batches=ref_batches, rd=self.iteration,
                                    hparams=self.cfg.as_dict(), initial_gref=g_ref):
        last, n_proj, steps = ev["loss"], ev["n_proj"], ev["step"]
        self.progress(ev["step"] / ev["n_steps"], loss=last, proj_rate=n_proj / ev["step"], epoch=ev["epoch"])
    if self.cfg.save_checkpoints:
        torch.save(self.model.state_dict(), self.trainer_dir / "checkpoints" / f"iter_{self.iteration:03d}.pt")
    if not self.cfg.replay_buffer:
        self.online = []                                             # current iteration only
    return TrainResult(metrics={"loss": last, "proj_rate": n_proj / max(1, steps), "n_proj": n_proj,
                                "train_steps": steps, "n_samples": len(fired),
                                "replay_size": len(self.reference_pool()), "train_buf": len(self.online),
                                "wall_s": time.monotonic() - t0, "iteration": self.iteration,
                                "scale_check_ratio": sc["std_ratio"]},
                       detail=f"iteration {self.iteration}: ...")
```

`pgrad.pgrad_iteration` yields once per optimiser step: it draws a reference batch, computes
`g_ref` (EMA `gref_ema_beta`, default 0.9, seeded with `initial_gref` in round 1), computes
the loss gradient on the online batch, projects when `g·g_ref < 0`, clips
(`grad_clip`) and steps. `train_mode: "epoch"` runs `n_epochs` over the buffer; `"steps"`
caps at `steps_per_iteration`.

## 4. Configuration (yours, not the runtime's)

`--trainer-config trainer.yaml` (or `PRO_DAGGER_*` environment variables):

| key | default | meaning |
|---|---|---|
| `offline_dataset` | (required) | a directory, or a repo id `<ns>/<name>` resolved on the LOCAL filesystem as `<datasets_home>/<ns>/<name>` (a bare `<name>` means `bc_demo/<name>`); no runtime call is involved |
| `datasets_home` | `"~/data"` | where repo ids resolve (the lab's per-namespace roots) |
| `rollouts_per_iteration` | `5` | R |
| `train_mode` | `"epoch"` | `"epoch"` or `"steps"` |
| `n_epochs` | `8` | epochs over the online buffer per iteration |
| `steps_per_iteration` | `200` | the cap in steps mode |
| `steps_per_batch` | `10` | steps mode: the data-scaled step count is `min(steps_per_iteration, max(10, steps_per_batch * n_batches))` |
| `lr` | `1e-4` | AdamW lr (real-robot preset: `2e-6`) |
| `batch_size` | `8` | (real-robot preset: 16) |
| `replay_buffer` | `true` | the online buffer accumulates every expert intervention across iterations; `false` = train each iteration on its own interventions only, then drop them |
| `max_demos` | `0` | FIFO cap of the online buffer; 0 = unbounded |
| `use_pgrad` | `true` | projection on / off (off = plain AdamW on the online buffer) |
| `gref_ema_beta` | `0.9` | per-step EMA of the reference gradient |
| `max_ref_batches` | `32` | reference subset = `max_ref_batches * batch_size` pairs |
| `grad_clip` | `1.0` | `clip_grad_norm_` |
| `freeze_offline_gref` | `true` | the `g_ref` pool stays offline-only |
| `offline_stride` | `5` | every k-th offline frame seeds the pool |
| `chunk_stride` | `3` | keep every 3rd chunk per intervention window |
| `chunk_horizon` | `null` | null = the policy's own horizon |
| `seed` | `0` | |
| `save_checkpoints` | `true` | write the trained copy to `<session_dir>/trainer/checkpoints/iter_<k>.pt` after every iteration |

Run it: `mavis-policy-node --loader entrypoint --entrypoint my_pkg.policy:make_policy
--online-dagger mavis_policy_node.pro_dagger:make_trainer --trainer-config trainer.yaml ...`.
The Cockpit shows your `metrics` verbatim (`loss` sparkline, `proj_rate`, `iteration`).

## 5. Reading the two health signals

- **`proj_rate`** over an iteration: ~0.28-0.45 is the healthy band. Near 0 = the online
  gradient never conflicts with the offline behaviour (the interventions teach nothing new,
  or `g_ref` is degenerate: check the reference batches). Near 1 = every step is projected:
  the online targets contradict the offline pool systematically - usually a normalisation or
  frame mismatch between the demonstrations and the rollouts.
- **`scale_check_ratio`** (`pgrad.scale_check`): std of the online targets over the offline
  ones. O(1) is right; 0.1 or 10 means the rollouts were recorded in another frame /
  unit than the demonstrations (check `SessionAnnounce.frames` against the dataset's).

## 6. HG-DAgger in five lines (the trivial variant)

No offline pool, no projection: train on the accumulated expert frames after every kept
rollout.

```python
def on_episode_saved(self, rollout): self.buffer += expert_pairs(rollout.spool_path); self.pending += 1
def on_train_now(self): self.pending = max(self.pending, 1)
def train_if_due(self):
    if not self.pending: return None
    self.pending = 0; return TrainResult(metrics={"loss": fit(self.model, self.buffer, self.epochs)})
```

`swap_weights` / `status_metrics` / `on_session` as in the SKILL.md example. That is the
whole difference between the variants as far as the runtime is concerned: which frames you
keep, when you train, how you step.
