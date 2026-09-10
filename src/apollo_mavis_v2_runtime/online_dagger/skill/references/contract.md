# Online DAgger trainer contract (wire spellings, messages, refusals, dataset layout)

Authority: `apollo_mavis_v2_core/protocol/external.py` (spellings), design notes
`15-online-dagger.md` (this role), `14-dora-interface.md` (the bus), `10-frames-and-data.md`
§11 (the dataset layout). The policy-node package copies every spelling into
`mavis_policy_node/contract.py` and pins it with `tests/golden/contract_golden.json`; the
runtime repo tests against the same golden file. `mavis_schema` is `1` everywhere; every
addition below is additive.

## 1. Node, streams, metadata

| item | spelling |
|---|---|
| your node id (dynamic placeholder) | `policy` (`Node("policy", daemon_port=<GET /api/dora daemon_port>)`) |
| the runtime's node id | `mavis_runtime` |
| dataflow name | `mavis_v2` |
| inputs your node receives | `obs_state`, `session`, `policy_reset`, `events`, `cam_<camera_id>`, `cam_<camera_id>_depth` |
| outputs your node may send | `action`, `spec`, `status`, **`trainer_status`** (`POLICY_OUTPUTS`, in this order) |
| runtime inputs (its side of the same wires) | `tick`, `probe_heartbeat`, `policy_action`, `policy_spec`, `policy_status`, **`policy_trainer_status`** (dora source `policy/trainer_status`, `queue_size: 8`) |
| metadata on every message your node sends | `mavis_schema` (1), `session_id` (echo of the current announce; `""` before one), `client` (your process id string), `seq` (ONE monotonic counter per client over every output), `t_mono`, `wallclock_ns`, `epoch` when known |
| metadata values | `bool` / `int` / `float` / `str` / `list[int\|float\|str]` only |
| JSON payloads | one `Utf8[1]` Arrow scalar holding the JSON object |
| capability | `PolicySpecAnnounce.capabilities` contains `"online_dagger"` |

`mavis-policy-node` produces all of this for you; the table is what you see on the wire
(and in the runtime's `var/logs/runtime.log`).

## 2. Messages

### 2.1 `PolicySpecAnnounce` (your `spec` output, 1 Hz heartbeat + after every `session`)

Fields in order: `mavis_schema, policy_id, policy_version, node_version, spec, rate_hz,
chunk_len, chunk_dt_s, loader, device, supports_reload, health, detail, uptime_s,
acts_total, last_compute_ms, extrinsics_sha, capabilities`. `spec` is
`{action_space, action_frame, action_names, state_names, camera_keys, version}`.
`policy_version` == `spec.version`; **bump it in `swap_weights()`** - the runtime's acting
version follows the ANNOUNCED version (this heartbeat or an action's `policy_version`
metadata), never your `trainer_status.policy_version` claim; a change shows as "swapped".

### 2.2 `SessionAnnounce` (the runtime's `session` output, 1 Hz + on change)

Fields in order: `mavis_schema, epoch, session_id, state, spec, kind, arm_ids, has_rail,
frames, action_space, action_names, state_names, camera_ids, cameras, policy_source,
dataset_root, run_id, deprecated_keys, online_dagger`.

`state` is the runtime's session state (`idle`, `bringup`, `start_from`, `running`,
`recovering`, `fault`, `teardown`); the trainer loop acts on **`running`**. `spec` is the
whole `SessionSpec` the operator launched; for Online DAgger `spec.mode == "dagger"`,
`spec.policy_source == "external"`, and `spec.online_dagger` is
`{session_name, resume, pause_while_training, wait_for_trainer_ready}` - nothing else
(no dataset, no hyper-parameters: those are yours). `online_dagger` (the
`OnlineDaggerAnnounce`, null for other sessions), fields in order:

| field | meaning |
|---|---|
| `session_name` | slug the operator typed |
| `session_dir` | `$HOME/data/online_dagger/<session_name>/` (runtime-owned `session.json` inside) |
| `rollouts_dir` | `<session_dir>/rollouts` - a standard episode-directory dataset (repo id `online_dagger/<session_name>`) |

Anything else you write (checkpoints, caches, logs) is your business; recommended
`<session_dir>/trainer/`. The runtime never reads it.

### 2.3 `TrainerStatusAnnounce` (your `trainer_status` output; runtime input `policy_trainer_status`)

Fields in order (the loop fills them; `progress(...)` / `status_metrics()` in your hooks
update the meters):

| field | type / default | meaning |
|---|---|---|
| `mavis_schema` | int = 1 | |
| `trainer_id` | str (required) | e.g. `my-policy-repo/online_dagger` (the node uses `<policy_id>/online_dagger` or your trainer's `trainer_id` attribute) |
| `node_version` | str (required) | `mavis_policy_node.__version__` |
| `state` | `idle` / `preparing` / `training` / `ready` / `error` | §3 |
| `session_id` | str or null | echo of the served `SessionAnnounce.session_id` - in EVERY status, heartbeats included. The runtime drives its phase ONLY from statuses whose `session_id` echoes its current session: `null` counts as "trainer alive" but drives nothing; another session's id is ignored outright (§3) |
| `policy_version` | int = 0 | version the acting policy runs after the last swap (informational; the runtime follows the announced spec) |
| `progress` | float 0..1 = 0 | of the current `preparing` / `training` step (the TRAINING pill's bar) |
| `metrics` | dict[str, float] = {} | free-form FINITE scalars (`loss`, `proj_rate`, `buffer`, ...); the Cockpit lists them verbatim, `loss` gets the sparkline |
| `detail` | str = "" | free text (the error message in `error`; shown in the refusals) |
| `uptime_s` | float = 0 | node uptime |

Validated on your side by `mavis_policy_node.contract.validate_trainer_status()` and on the
runtime's side by pydantic: **`inf` / `nan` anywhere is a rejected message** (the runtime
drops it and counts a dropped input) - report a diverged run as `state: "error"` + `detail`.

### 2.4 `EventEnvelope` (the runtime's `events` output)

`{mavis_schema, kind, t_mono, wallclock_ns, session_id, epoch, payload}`; `kind` also rides
the metadata. Kinds (`EVENT_KINDS`, in order): `collision, gate, episode_saved,
episode_discarded, policy_anomaly, policy_swap, policy_version_changed, reset_watermark,
session_error, train_now`. Payloads the trainer reads:

- `gate`: `{arm_id, mode, seq, source, episode_id}` on every take-over / hand-back instant
  of the session - `mode` is the NEW mode (`policy`, `takeover_transition`, `human`),
  `source` who caused it (`keyboard` = Space, `action` = the Take over / Hand back buttons,
  `auto_advance` = the blend into `human`, `episode_reset` = the boundary reset),
  `episode_id` the open episode or null. Informational (segment interventions live).
- `episode_saved`: `{episode_index, summary, dataset_root, spool_path, run_id, online_dagger:
  {episode_id, rollouts_saved, actor_counts: {novice, expert}, policy_version, spool_path}}`
  - **a kept rollout**. `rollouts_saved` is the session's running count (continues across a
  resume); `summary` carries `n_frames`, `n_expert_frames`, `n_novice_frames`,
  `n_label_frames` and the take-over segments.
- `episode_discarded`: `{episode_index, episode_id, reason}` - the operator (or the
  runtime, e.g. an all-filtered save or teardown) discarded the rollout. **Nothing is on
  disk**: the runtime removed the episode's temporary directory. Drop anything you landed
  for `episode_id` (a trainer that only reads on `episode_saved` has nothing to do).
- `train_now`: `{rollouts_saved, requested_by: "operator"}` - the Cockpit's **Train now**
  button. Honour it (`on_train_now`) or ignore it; the runtime does not care.

Events whose envelope `session_id` is not the served session, or that arrive before any
Online DAgger announce, are ignored by the loop.

### 2.5 `PolicyResetMsg` (the runtime's `policy_reset` output)

`{mavis_schema, reason, after_observation_id, session_id, t_mono}`; `reason` in
`handback, episode_boundary, session_start, anomaly, session_stop`. The runtime spells
`episode_boundary` at every episode boundary (save or discard) and `handback` when the
operator hands control back inside an episode. The node calls `policy.reset()` and ignores
observations at or below the watermark.

## 3. The trainer state machine and the runtime's phases

```
trainer_status.state            runtime OnlineDaggerCoordinator.phase       episode_new is ...
--------------------            -------------------------------------       ------------------
(no fresh status)               (unchanged)                                 refused: "no Online DAgger trainer attached"
idle / preparing (before ready) waiting_trainer                             refused: "waiting for the trainer to report ready (<detail>)"
ready                           rollout                                     admitted
training                        training  (pause_while_training on)         refused: "training in progress (<detail>)"
training                        rollout   (pause_while_training off)        admitted; act() keeps running while you train
idle / preparing (after ready)  rollout                                     admitted
error                           error                                       refused: "trainer error: <detail>"; your next non-error status clears it
(status stale > spec_stale_s)   phase unchanged, trainer_alive false        refused: "no Online DAgger trainer attached"; banner in the UI
```

How `mavis-policy-node` leaves `error` (the runtime just follows the next status): a failed
`on_session` is sticky - the trainer never prepared, so that session's events are dropped
without reaching your hooks and only a NEW session id (which re-runs `on_session`) leaves
`error`; after any other hook failure your hooks keep being called, but `error` clears only
when a later `train_if_due` returns a `TrainResult` and the weights were swapped (a hook
merely accepting an event, or a poll returning `None`, clears nothing) - or a new session id.

With `wait_for_trainer_ready: false` the session starts in `rollout` and only `training`
(pause on) / `error` ever refuse. `<detail>` is your `detail`, else `trainer <state>` /
the training `progress` in percent; before you echo the session id it reads
`the trainer has not picked up this session yet`, before any status `no trainer status yet`.

**Which status counts.** The runtime drives its phase ONLY from trainer statuses whose
`session_id` echoes its current session, so a trainer MUST echo the served
`SessionAnnounce.session_id` in every `trainer_status` it publishes - the first `idle` after
the announce, every progress update and every 1 Hz heartbeat. Three guards before the state
machine sees a status (`OnlineDaggerCoordinator.on_trainer_status`):

- a status older than the newest one already seen is dropped (the attach replay racing a
  fresh heartbeat), so the trainer view never runs backwards;
- a status for ANOTHER session (a non-null `session_id` that is not the runtime's: the
  previous session's tail) is ignored outright - it is not this session's trainer, so it
  neither refreshes `trainer_alive` nor moves anything;
- a status with `session_id: null` (the trainer is up, serving nobody yet) counts as alive
  and is shown verbatim, but drives NO transition.

A trainer that never echoes the id therefore leaves the session in `waiting_trainer` forever.
`mavis-policy-node` echoes it for you (the loop stamps the served id on every status from
the announce until the session ends; the final `idle` carries `null`); a training job that
finishes after the served session changed is discarded, never reported under the new id.

Order of things in one rollout: the operator presses `N` (`episode_new`; admitted per the
table) -> the policy drives, the operator takes over (`Space`, or the **Take over** button)
and hands back (`Space` again / **Hand back**) as often as needed - every instant is an
`events.gate` -> `Enter` saves (`events.episode_saved`, then the arms return to the start
profile) or `Backspace` discards (`events.episode_discarded`, nothing on disk). YOU decide
when to train: the node reports `training` while `train_if_due` runs (the runtime refuses
`N` with pause on), calls `swap_weights`, and reports `ready` with the new version; the spec
heartbeat carries it and the runtime shows "swapped". **Never swap weights mid-episode**:
the loop only calls `swap_weights` right after `train_if_due`, and with
`pause_while_training: true` (default) no episode is open then. With it `false` the runtime
keeps rolling out - and calling `act()` - while you train: train a copy of the acting module
and install it in `swap_weights` under a lock `act()` also holds, so no half-trained weights
ever act.

Resume: the operator can relaunch a session with the same name (`online_dagger.resume:
true`); you receive a new `session_id` with the same directories, `on_session` runs again
(rebuild your state from `rollouts_dir` or your own files) and `rollouts_saved` continues
from where `session.json` left it. Transient session states of the SAME id (`bringup` /
`start_from` before `running`, `recovering` / `fault` after it) do not end the session for
the loop; `idle` / `teardown`, another id or a vanished `online_dagger` block do.

## 4. `OnlineDaggerConfig` - the operator's block

Received as `SessionAnnounce.spec.online_dagger`. It carries the SHELL's settings only:

| key | default | meaning |
|---|---|---|
| `session_name` | (required) | slug; the directory under `~/data/online_dagger/` |
| `resume` | `false` | continue an existing session dir (409 "already exists" otherwise) |
| `pause_while_training` | `true` | the runtime refuses `episode_new` while you report `training`; `false` = rollouts and `act()` continue while you train (train a copy, swap under a lock - §3) |
| `wait_for_trainer_ready` | `true` | the runtime refuses `episode_new` until you have reported `ready` once for this session |

Everything about the algorithm (offline dataset, R, epochs, lr, buffers, projection ...) is
your trainer's own configuration (`--trainer-config` / env). The runtime never forwards
hyper-parameters; an unknown key in this block is a 422 at launch.

## 5. Dataset layout you read

The rollouts (`rollouts_dir`) and any offline dataset you configure yourself are the same
layout:

```
<dataset>/manifest.json                      apollo_dataset_layout 1, repo_id, fps, robot_type,
                                             features {name: {dtype, shape, names, info}}, arms, cameras,
                                             episodes, frames, modified_at
<dataset>/episodes/<episode_id>/episode.json sidecar (length, task, stats, online_dagger block for rollouts:
                                             {session_name, rollouts_saved, policy_version, actor_counts})
<dataset>/episodes/<episode_id>/frames.parquet
<dataset>/episodes/<episode_id>/video/<camera_id>.mp4   one per recorded camera (pts = frame index, 1/fps)
<dataset>/episodes/<episode_id>/audio.wav               optional
<dataset>/trainer_spool/ep_<episode_id>.parquet         DAgger only: the non-video columns again
<dataset>/episodes/.tmp-<id>/                            an episode being recorded - ignore (a discard removes it)
```

Episode ids are capture-time stamps (`20260907T141203.512Z-3f9a1c`), sorted = capture order,
never reused. `frames.parquet` columns (one row per KEPT frame - the runtime's idle-frame
filter drops hesitation frames at record time, human frames only):

| column | type | notes |
|---|---|---|
| `action` | fixed list float32[D] | the executed command (post-gate) in the recording frame; `manifest.features.action.names` |
| `observation.state` | fixed list float32[S] | `manifest.features["observation.state"].names` |
| `intervention` | bool | human engaged |
| `action_source` | int8 | 0 policy, 3 takeover |
| `wallclock_ns` | int64 | |
| `control_mode` | int8 (DAgger only) | 0 policy, 1 human, 2 takeover_transition |
| `policy_action` | fixed list float32[D] (DAgger only) | counterfactual; NaN while the human drives |
| `policy_version` | int32 (DAgger only) | |
| `actor` | int8 (DAgger only) | 0 novice, 1 expert (= `control_mode != 0`) |
| `timestamp` | float32 | `frame_index / fps` |
| `frame_index` | int64 | |
| `task` | string | |

The spool has the same columns as plain (non-fixed) lists, minus `timestamp` / `frame_index`
/ `task`. **Training labels are `control_mode == 1`** (transition frames excluded); `actor`
is the operator-readable key. Offline demonstration datasets (recorded with Data
Collection) have no `actor` / `control_mode`: every frame is expert.
`mavis_policy_node.online_dagger.datasets` reads all of this (`list_episodes`,
`read_episode_frames`, `expert_mask`, `chunk_windows`, `decode_video_frames`).

## 6. Refusals you will see and what they mean

| where | text | cause |
|---|---|---|
| `POST /api/session` 409 | `no external policy attached (...)` | no `spec` heartbeat from your node within 3 s / bridge not attached |
| 409 | `no Online DAgger trainer attached (the policy node does not report the online_dagger capability)` | node running without `--online-dagger` |
| 409 | `policy/dataset frame mismatch` | `spec.action_names` / `action_frame` / `state_names` do not match the session |
| 409 | `hardware sessions support teleop and data collection only` | Online DAgger is sim-only until the operator admits it on hardware |
| 409 | `Online DAgger session '<s>' already exists - resume it or pick another name` | `resume: false` on an existing name |
| 409 | `Online DAgger session '<s>' not found` | `resume: true` on a name that has no directory |
| 409 | `Online DAgger session '<s>': session.json is unreadable - fix or remove it` | a corrupt record is never overwritten |
| `episode_new` refused | `no Online DAgger trainer attached` | no fresh `trainer_status` (> 3 s) |
| `episode_new` refused | `waiting for the trainer to report ready (<detail>)` | you have not reported `ready` for this `session_id` yet |
| `episode_new` refused | `training in progress (<detail>)` | your state is `training` (pause on) |
| `episode_new` refused | `trainer error: <detail>` | your state is `error` |
| `train_now` refused | `save or discard the episode first` / `no Online DAgger trainer attached` | an episode is open / no fresh status |
| `takeover` / `handback` | ack `already taken over` / `policy already driving` | idempotent no-ops (not errors) |
| node log | `the running dataflow does not declare policy/trainer_status` | runtime older than phase-14; trainer status is dropped |
