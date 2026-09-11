---
name: mavis-online-dagger-trainer
description: Set up an Online DAgger trainer (any DAgger variant) for a policy repo that drives the MAVIS v2 cell over dora
---

# MAVIS Online DAgger trainer

You are working in a **policy repository**: a Python project that owns a robot policy
(a model that maps an observation to an action chunk) and its training code. The MAVIS v2
runtime (Apollo Lab's dual-arm cell: two xArm7 on linear tracks, sim or real) is the
**rollout-level shell**: it performs rollouts with your policy, lets a human operator take
over and hand back, records every step with a **novice / expert** label, saves the kept
rollouts into a dataset and reports what your trainer says. It knows nothing about any
DAgger algorithm - **you** decide when to train, on what, and how (HG-DAgger, PRO-DAgger,
DRIFT-DAgger, plain aggregation ...). You implement two things in this repo - a `Policy`
and an `OnlineDaggerTrainer` - and run them with the `mavis-policy-node` process, which
speaks the wire protocol for you. Nothing in this skill requires reading the MAVIS source.

Read `references/contract.md` (every message, refusal and the dataset layout) as you go;
`references/pro-dagger-example.md` implements PRO-DAgger on top of the generic hooks and
shows HG-DAgger as the five-line trivial case.

## 1. Roles

```
 operator ──keys/Vive/UI──▶ mavis runtime                              your policy node
                            ├─ takeover gate + takeover/handback API     ├─ inference (obs_state → action)
                            ├─ recorder → rollouts/ (actor column)       ├─ trainer (ANY DAgger variant)
                            ├─ events: gate, episode_saved,      events▶│   counts rollouts, decides when
                            │          episode_discarded, train_now      │   to train, swaps its weights
                            └─ telemetry ◀──────────────── trainer_status (state, progress, metrics)
```

The runtime never tells you "train now" on its own schedule. It publishes what happened
(`episode_saved`, `episode_discarded`, `gate`) and forwards the operator's **Train now**
button (`train_now`); you keep your own count and train when your algorithm says so. Two
generic gates protect the cell from a training process: while you report `training` the
runtime refuses new rollouts (`pause_while_training`, default on), and until you have
reported `ready` once for the session it refuses them too (`wait_for_trainer_ready`,
default on).

## 2. Prerequisites

- Linux, Python >= 3.11, a GPU if the model needs one. The node runs **on the lab machine**
  (the runtime's `policy` placeholder is deployed there only; a remote policy node is not
  supported in v1).
- dora-rs **1.0.1** - installed as a dependency of the package below. Never `dora up`,
  never `pkill -f dora`, never bind `0.0.0.0`.
- Install the package into the policy repo's environment:

  ```bash
  pip install "mavis-policy-node[torch]"        # + [video] if the model uses images; + [pro_dagger] for the reference implementation
  # or from the repo checkout: pip install -e "<path>/apollo-mavis-v2-policy-node[torch]"
  ```

- Connection facts come from the running runtime, not from a config file:

  ```bash
  curl -s http://<lab-host>:8765/api/dora
  # {"enabled": true, "state": "attached", "bind_host": "...", "daemon_port": 53391,
  #  "zenoh_port": 7447, "zenoh_connect": "tcp/<bind_host>:7447", "dataflow_name": "mavis_v2", ...}
  export DORA_ZENOH_CONNECT=tcp/<bind_host>:<zenoh_port>
  export DORA_ZENOH_MULTICAST=off
  export DORA_ZENOH_LISTEN=tcp/127.0.0.1:0
  ```

  `--daemon-port` below is `daemon_port`. No coordinator address, no auth token.

- Whatever your algorithm needs besides the rollouts (an offline demonstration dataset, a
  pre-trained novice checkpoint, hyper-parameters) is **your configuration**: the runtime's
  launch sheet has no dataset picker and no hyper-parameter fields. Pass it with
  `--trainer-config <yaml|json>` or environment variables of your own.

## 3. Implement the `Policy`

The node's README documents it; the essentials:

```python
import threading
import numpy as np
from mavis_policy_node.types import Observation, PolicyOutput, PolicySpec

class MyPolicy:
    def __init__(self, model, spec: PolicySpec, device): ...
    spec: PolicySpec            # action_space "delta_ee" | "abs_ee", action_frame "arm_base:grip", action_names,
                                # state_names, camera_keys, version (int - bump on every swap)
    policy_id = "my-policy"     # optional; the node's spec.policy_id / trainer_id prefix (default: the class name)
    def reset(self) -> None: ...                       # drop chunks / history
    def act(self, obs: Observation) -> PolicyOutput: ...  # obs.state float32 (len(state_names),), obs.images[cam] (H,W,3) uint8
    def act_chunk(self, obs) -> np.ndarray | None: ... # optional: (K, D) rows
    def load_weights(self, path: str) -> None: ...

    # not part of the node's protocol - conventions between your policy and your trainer:
    model: "torch.nn.Module"    # the ACTING module act() runs
    lock = threading.Lock()     # act() holds it around the forward pass; swap_weights() takes it too

def make_policy(path: str | None, device: str) -> MyPolicy: ...
```

`action_names` / `state_names` must match the session exactly (the runtime refuses
`policy/dataset frame mismatch` otherwise); they are the dataset's `manifest.json`
`features["action"]["names"]` / `features["observation.state"]["names"]`. Actions are in
the session's recording frame, deltas per `chunk_dt_s`, gripper absolute, metres / radians.
Only `spec`, `reset`, `act`, `load_weights` (and the optional `act_chunk` / `on_session` /
`policy_id`) are read by the node; `model` and `lock` are conventions between your policy
and your trainer.

## 4. Implement the `OnlineDaggerTrainer`

Eight hooks (`mavis_policy_node.online_dagger.protocol.OnlineDaggerTrainer`). All of them
except `status_metrics` run on ONE worker thread, in order (the session hook, then each event
followed by `train_if_due`, plus a 1 Hz `train_if_due` poll while `ready`), so no two of them
ever overlap and inference is never blocked. `status_metrics()` is the exception: it is called
on every status publish (>= 1 Hz, from the node thread) and MAY overlap a running
`train_if_due` - it must be cheap and read only atomically-replaced values.

```python
from mavis_policy_node.online_dagger.protocol import GateEvent, Rollout, SessionInfo, TrainResult

class MyTrainer:
    def __init__(self, policy: MyPolicy, config: dict, device): ...
    def on_session(self, session: SessionInfo) -> None: ...            # a new Online DAgger session (dirs, spec, session_id)
    def on_episode_saved(self, rollout: Rollout) -> None: ...         # a KEPT rollout: episode_id, rollouts_saved, actor_counts, spool_path, policy_version
    def on_episode_discarded(self, episode_id: str) -> None: ...      # drop anything you landed for that id (nothing is on disk)
    def on_gate(self, event: GateEvent) -> None: ...                  # take-over / hand-back instants (informational)
    def on_train_now(self) -> None: ...                               # the operator pressed Train now (honour or ignore)
    def train_if_due(self) -> TrainResult | None: ...                 # None = nothing to do; else train and return metrics
    def swap_weights(self) -> int: ...                                # install the trained weights, bump spec.version, return it
    def status_metrics(self) -> dict[str, float]: ...                 # free-form finite scalars for the Cockpit (loss, ...); publishing thread

def make_trainer(policy, config: dict, device) -> MyTrainer: ...
```

- `on_session` arrives on the `running` announce that carries the `online_dagger` block
  (§2.2 of the contract): `session.session_id` (echo it - the node does), `session.session_dir`
  (`~/data/online_dagger/<name>/`), `session.rollouts_dir` (a standard episode-directory
  dataset with the `actor` column), `session.spec` (the operator's `SessionSpec`; its
  `online_dagger` block has ONLY `session_name`, `resume`, `pause_while_training`,
  `wait_for_trainer_ready`). Prepare whatever your algorithm needs (load the offline pool,
  compute a reference gradient, ...) - the node reports `preparing` while the hook runs and
  `ready` when it returns, which is what admits the first rollout. A resumed session is a
  NEW session id over the same directories: rebuild your state from the rollouts on disk (or
  from your own files under `<session_dir>/trainer/`).
- `on_episode_saved` is where you count. `rollout.rollouts_saved` is the runtime's kept
  count (continues across a resume); `rollout.spool_path` is the non-video parquet of the
  episode (`<rollouts_dir>/trainer_spool/ep_<episode_id>.parquet`; image policies decode
  `episodes/<episode_id>/video/<cam>.mp4`); `rollout.actor_counts == {novice, expert}`.
  Training labels are `control_mode == 1` (`actor` is the readable key).
- `train_if_due` is polled after every event and at 1 Hz. Return `None` to do nothing;
  otherwise train and return a `TrainResult(metrics={...}, detail="...")`. The node reports
  `training` with your `progress(fraction, **metrics)` calls while it runs, then calls
  `swap_weights` and reports `ready` with the new `policy_version`. Decide "due" however
  your algorithm does: every `R` kept rollouts (PRO-DAgger), every rollout (HG-DAgger),
  on `on_train_now` only, on a wall-clock budget ...
- `swap_weights` installs the trained weights into the acting policy object (the one
  `act()` runs on - the same process, so `load_state_dict` from your training copy is
  enough), bumps `policy.spec.version` and returns it. **A half-trained module must never
  act.** With `pause_while_training: true` the runtime refuses rollouts while you train, so
  training the acting module in place is a valid shortcut - valid ONLY then. With it
  `false` `act()` runs concurrently while you train: train a copy and install it under a
  lock that `act()` also holds (the example does).
- `status_metrics` fills `trainer_status.metrics`: any finite floats; a `loss` key gets the
  Cockpit sparkline. It runs on the publishing thread, possibly while `train_if_due` is in
  the middle of a step: return numbers you replace atomically (`self.last_loss`, a fresh
  dict), never iterate a list or dict `train_if_due` mutates. Never put `inf` / `nan` there -
  report a diverged run as an exception from `train_if_due`: the node publishes
  `state: "error"` with the message and the runtime refuses rollouts. The error stands until
  a later `train_if_due` returns a `TrainResult` (the hooks keep being called meanwhile; a
  hook merely accepting an event clears nothing) or a new session id starts over. An
  exception from `on_session` is sticky: the trainer never prepared, so the node drops that
  session's events and only a new session id (which re-runs `on_session`) leaves `error`.

### Minimal example: HG-DAgger (train on every kept rollout's expert frames)

```python
# my_pkg/online_dagger.py
from __future__ import annotations
import copy
import torch
from mavis_policy_node.online_dagger import datasets as ds
from mavis_policy_node.online_dagger.protocol import TrainResult

class MyTrainer:
    def __init__(self, policy, config, device):
        self.policy, self.device = policy, device or "cpu"
        self.model = copy.deepcopy(policy.model).to(self.device)   # train a copy, swap under the lock
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=float(config.get("lr", 1e-4)))
        self.epochs = int(config.get("epochs", 2))
        self.buffer, self.pending, self.last_loss = [], 0, None

    def on_session(self, session):            # rebuild from disk on a resume
        self.buffer, self.pending = [], 0
        for eid in ds.list_episodes(session.rollouts_dir):
            frames = ds.read_episode_frames(ds.episode_dir(session.rollouts_dir, eid))
            self.buffer += ds.expert_pairs(frames, self.policy.norm)

    def on_episode_saved(self, rollout):
        frames = ds.read_episode_frames(rollout.spool_path)
        self.buffer += ds.expert_pairs(frames, self.policy.norm)    # keep control_mode == 1 rows
        self.pending += 1

    def on_episode_discarded(self, episode_id): pass                # nothing landed (we read on save)
    def on_gate(self, event): pass
    def on_train_now(self): self.pending = max(self.pending, 1)

    def train_if_due(self):
        if self.pending == 0 or not self.buffer:
            return None
        self.pending = 0
        self.model.load_state_dict(self.policy.model.state_dict())
        for _ in range(self.epochs):
            for x, y in ds.batches(self.buffer, 64, self.device):
                loss = torch.nn.functional.mse_loss(self.model(x).view(y.shape), y)
                self.opt.zero_grad(); loss.backward(); self.opt.step()
                self.last_loss = float(loss)
        return TrainResult(metrics={"loss": self.last_loss, "buffer": float(len(self.buffer))})

    def swap_weights(self) -> int:
        with self.policy.lock:
            self.policy.model.load_state_dict(self.model.state_dict())
            self.policy.spec = self.policy.spec.bump()
        return self.policy.spec.version

    def status_metrics(self):
        return {"loss": self.last_loss} if self.last_loss is not None else {}

def make_trainer(policy, config, device): return MyTrainer(policy, config, device)
```

PRO-DAgger (projected reference gradient, offline pool, EMA) on the same hooks:
`references/pro-dagger-example.md` and the shipped `mavis_policy_node.pro_dagger`.

## 5. Run the node

```bash
mavis-policy-node --loader entrypoint --entrypoint my_pkg.policy:make_policy \
    --online-dagger my_pkg.online_dagger:make_trainer --trainer-config trainer.yaml \
    --path ckpt/ --device cuda:0 --daemon-port <daemon_port>
```

With `--online-dagger` the node's `spec.capabilities` lists `online_dagger` (the launcher
checks it before **Start**), forwards `session` / `events` to the trainer loop and publishes
`trainer_status` (1 Hz + on change). Every status - the first `idle` after an announce,
progress, heartbeats - echoes the served `SessionAnnounce.session_id`: the runtime drives
its phase ONLY from statuses that echo its current session (`session_id: null` counts as
"trainer alive" but drives nothing; another session's id is ignored), so a trainer that
speaks the wire itself MUST echo it in every heartbeat or the session never leaves
`waiting_trainer`. The state machine and the runtime's reactions are in
`references/contract.md` §3.

## 6. Test before touching the cell

1. **Selftest (no dora, no runtime):**
   `mavis-policy-node --loader entrypoint --entrypoint my_pkg.policy:make_policy --online-dagger my_pkg.online_dagger:make_trainer --selftest online-dagger`
   drives your trainer with a synthetic session (two kept rollouts with a take-over each,
   one discard, one `train_now`) and prints the status sequence; exit code 0 means
   `idle -> preparing -> ready -> training -> ready` with a version bump. If your model's
   action layout differs from the synthetic one (one arm with a rail, 8 action dims), make
   the trainer tolerate it or run the selftest with the fake policy (`--loader fake`) first
   to see the expected output.
2. **Private control plane:** `tests/test_node_e2e.py::test_online_dagger_trainer_end_to_end`
   in a git checkout of `apollo-mavis-v2-policy-node`
   (<https://github.com/Apollo-Lab-Yale/apollo-mavis-v2-policy-node>; the tests ship in the
   checkout and the sdist, NOT in the wheel a `pip install` gives you) shows how to spawn a
   private `dora coordinator` + `dora daemon` on free loopback ports, attach as
   `mavis_runtime`, send the announce and events and read `trainer_status`. Copy its
   `RuntimeStub` to test your trainer against realistic messages without the runtime.
3. **Against the runtime in sim:** start the runtime (sim workcell, `dora.enabled: true`),
   run the node, open the UI's **Online DAgger** card. Before launch the "Connect a trainer"
   view shows what the node advertises: the external policy attached and the `online_dagger`
   capability (no `trainer_status` is published before the first announce, so the trainer
   pill shows only the capability until a session starts). Type a session
   name, **Start Online DAgger**: the Cockpit's phase pill shows `WAITING FOR TRAINER` (with
   your `preparing` detail) until you report `ready`, then `ROLLOUT` and the first `N` is
   admitted. Drive two rollouts - `Space` or the **Take over** / **Hand back** buttons - and
   watch the panel go `TRAINING` with your metrics -> the "swapped" flash on the new version.

## 7. Acceptance checklist

- [ ] `mavis-policy-node ... --selftest online-dagger` exits 0.
- [ ] Against the runtime: `GET /api/dora` shows the policy attached; the Online DAgger sheet
      shows the `online_dagger` capability and your trainer pill. After **Start** the Cockpit
      shows `WAITING FOR TRAINER` - `N` is refused meanwhile with "waiting for the trainer to
      report ready" - and admits the first rollout once your `trainer_status` reports `ready`
      for this `session_id`.
- [ ] A rollout saves with `actor` counts (`episode_saved.online_dagger.actor_counts` has both
      `novice` and `expert` > 0 after a take-over) and `rollouts/episodes/<id>/frames.parquet`
      has `control_mode` and `actor`; `rollouts_saved` counts up by one per kept rollout.
- [ ] When your algorithm decides to train, `trainer_status` goes `training` (progress /
      metrics updating at >= 1 Hz; the runtime refuses `N` with "training in progress") ->
      `ready` with a bumped `policy_version`.
- [ ] `spec.version` / `policy_version` bumped; the runtime's telemetry shows the new acting
      version ("swapped") and admits the next rollout.
- [ ] A discarded rollout (`Backspace`) leaves NO directory under `rollouts/episodes/` and
      your trainer never trained on it (`on_episode_discarded` dropped what it landed).
- [ ] **Train now** in the Cockpit reaches `on_train_now` (`events.train_now`); honouring it
      is your call.
- [ ] Killing and restarting the node mid-session re-attaches; a resumed session
      (`resume: true`, same name) continues `rollouts_saved` and your `on_session` rebuilds
      your state from `rollouts_dir`.
