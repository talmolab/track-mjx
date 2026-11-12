# DMPO

track-mjx ships a JAX port of Distributional MPO (DMPO; Abdolmaleki et al.,
2018), the algorithm of DeepMind acme's `agents/tf/dmpo`. The policy and
critic train on a single device: a batched MJX rollout alternates with K
on-device SGD steps over a flashbax replay. The package lives in
`track_mjx/agent/dmpo/`, and its configs in `track_mjx/config/dmpo/`.

## Install

Install track-mjx as described in [Installation](index.md). The high-level
entries load a frozen imitation prior trained with SCAMPER. SCAMPER is pinned
by commit in `requirements-dmpo-prior.txt`, and installed without its
dependencies so it cannot change the pins of this environment:

```bash
pip install --no-deps -r requirements-dmpo-prior.txt
```

The vnl-playground environments the entries train on must be installed at a
version that ships `vnl_playground.tasks.wrappers_truncation` (and, for the
kl-anchor entry, `wrappers_kl_anchor`).

## Environment variables

Configs never name a machine path. Two roots are read from the environment:

| Variable | Meaning | Default |
|---|---|---|
| `DMPO_PRIOR_ROOT` | Directory holding the imitation prior checkpoints (`<root>/<prior id>`, listed in `config/dmpo/priors/MANIFEST.md`). | none; required by every config that loads a prior |
| `DMPO_CKPT_ROOT` | Root of run directories and of graft or warm-start sources. | `dmpo_checkpoints` (relative to the working directory) |

## Launch

Three entry points, each a hydra app over `track_mjx/config/dmpo`:

| Entry | Use | Config groups |
|---|---|---|
| `scripts/train_dmpo_highlvl_kl_anchor.py` | Frozen prior and decoder inside the policy, an optional KL anchor to the prior; feed-forward, GRU and slow-core heads; grafts | `run_gap/`, `maze/`, `full_maze_memory/`, `spout_maze/` |
| `scripts/train_dmpo_highlvl.py` | Prior and decoder transfer without the anchor | `run_gap/velocity_only` and similar |
| `scripts/train_dmpo.py` | Motion-capture imitation | `rodent-imitation`, `rodent-imitation-intention` |

```bash
export DMPO_PRIOR_ROOT=/path/to/priors
export DMPO_CKPT_ROOT=/path/to/checkpoints
python scripts/train_dmpo_highlvl_kl_anchor.py --config-name=run_gap/<arm>
```

Overrides use hydra syntax, e.g. `++train_config.num_envs=1024`. A run writes
`DMPONetwork_<env step>/` checkpoints under its `checkpoint_dir` and keeps
every one of them (`checkpoint_max_to_keep` absent or `null`). Launching again
with the same `checkpoint_dir` resumes from the newest checkpoint.

## Startup markers

Read these log lines before committing a long run to the GPU:

- `kl-anchor loss term: ON|OFF (alpha=..., beta_linear=...)`: whether the
  anchor arrays are stored in the replay.
- `Compressed replay schema: ...` and `Replay transition size: ... GB`: the
  replay layout and its memory; `Replay add: flashbax scatter|in-place slice`:
  the replay write (`train_config.replay_inplace_add`).
- `Seeded DMPO normalizer with imit proprio stats: ...`: the frozen prior
  sees normalized proprioception from step 0 (grafted arms).
- `DMPO WARM START: ...`, `GRAFT ok ...` and `GRAFT invariance:
  policy_max_abs_diff=0 critic_max_abs_diff=0`: a warm start or graft
  copied what it should, and the grafted networks equal the source at
  step 0.
- `<label> r_anchor=... action_mse=...` (startup probe): the frozen prior and
  decoder are unchanged; `r_anchor` near 1.
- `RECURRENT policy head ACTIVE: ...`, `PRIVILEGED critic input ACTIVE: ...`,
  `FROM-SCRATCH arm ACTIVE: ...`: the head and critic inputs in use; with a
  privileged critic, `critic_priv width verified: ...` gives the shape and
  dtype the env writes.
- `Checkpoint retention: ALL (keeping every step)`.
- `Restored DMPO checkpoint: sgd_step=... env_steps=...` on a resume.
- `replay warm-up complete at ... env steps`, then one
  `chunk env_steps=... policy_loss=... critic_loss=...` line per chunk, with
  finite losses.

`train_config.log_every_steps` is accepted for compatibility and ignored: the
loop logs once per chunk (`iters_per_chunk` rollout and update iterations) and
evaluates every `eval_every_steps`.

Each eval logs an `eval env_steps=...` line. On the spout maze it also logs a
`SPOUT_EVAL env_steps=...` line, and on the full maze a `MAZE_EVAL
env_steps=...` line: per-env first-episode treats, length, fall, all-treats
and timeout fractions, and visited cells. The eval runs on the training
`num_envs` unless `eval_render_config.num_envs` sets a smaller batch, which
lets a long eval fit beside a large replay.

## Alignment with acme

Six `DMPOConfig` fields select between acme's semantics and the port's
historical ones. Their defaults are acme's:

| Field | Default (acme) | Legacy value | What it decides |
|---|---|---|---|
| `critic_target_mode` | `sample_mixture` | `mean_action` | Bootstrap distribution: the mean of the target critic's distributions over N target-policy samples, or one evaluation at the target policy's mean action. |
| `bootstrap_on_truncation` | `true` | `false` | A time limit is a bootstrap point, not a terminal. Needs the env composed with `wrap_for_dmpo_training` and a `truncation` replay field. |
| `critic_action_input` | `executed` | `raw` | The critic sees the tanh-bound action the env executed, or the stored pre-tanh sample. |
| `action_penalty_formula` | `acme` | `vnl_ray` | MO-MPO out-of-bound cost: `-‖a − clip(a, −1, 1)‖`, or `-‖a‖`. |
| `policy_loss_state` | `auto` (`next` on the feed-forward learner, `current` on the recurrent one) | `current` | The MPO policy loss is evaluated at the n-step arrival state, sharing its samples with the critic target, or at the state the action was taken in. |
| `policy_grad_clip_scope` | `global` | `per_block` | One global-norm clip over all trainable policy parameters, or one per parameter block. |

The preset `config/dmpo/legacy_pre_alignment.yaml` sets all six to their
legacy values and reproduces the pre-alignment learner bit for bit (CPU,
pinned by `tests/agent/dmpo/test_legacy_equivalence.py`). Add it to an arm's
defaults after `/base` and before `_self_` to reproduce a run made before the
fields existed:

```yaml
defaults:
  - /base
  - /legacy_pre_alignment
  - _self_
```

With `policy_loss_state: next` and a loss-side KL anchor, the anchor is read
at the arrival state, so the replay must hold it there: use
`sequence_length: n_step + 1` with `store_next_observation: false`.
`learner.arrival_anchor_missing` is the check, and
`tests/scripts/test_arrival_anchor_layout.py` runs it on every shipped arm.

### Checkpoints record the fields

Every checkpoint's saved config records the six fields in effect. A checkpoint
saved before the fields existed reads as the legacy preset. Continuing from a
checkpoint trained with different values would switch the algorithm partway
through a run, so the entries refuse:

- a graft (`transfer.graft_ff_checkpoint`) unless
  `transfer.graft_allow_algorithm_mismatch: true`;
- a resume (`checkpoint_dir` already holds a checkpoint) unless
  `train_config.allow_alignment_change_on_resume: true`.

Both overrides log a WARNING listing the differing fields. A graft between a
feed-forward and a recurrent learner (for example a feed-forward maze prior
into a slow-core arm) does not compare `policy_loss_state`, which resolves to
`next` on one and `current` on the other; the other five fields are still
compared. On a resume across a change of `policy_grad_clip_scope` the policy
optimizer state is re-initialized, since its structure depends on that field.
`transfer.warm_start_dmpo_checkpoint` is not checked: it copies the policy
into a fresh critic and fresh optimizers by design.

### Checkpoints trained before the alignment

Run such a checkpoint with the code and the environment it was trained with.
The `legacy_pre_alignment` preset reproduces the old learner, but not the old
rendering: the MJX renderer of MuJoCo 3.11 changed the vision observations,
so a vision policy trained on an older stack does not transfer to this one.
Graft or warm-start only from checkpoints trained on this code.

### Parity test

`tests/agent/dmpo/test_acme_parity.py` replays four learner steps of acme's TF
`DistributionalMPOLearner` (stored in `tests/agent/dmpo/acme_tf_reference/`)
through `sgd_step` on the default fields and matches every loss, MPO
statistic and online, target and dual variable to `rtol=1e-4`. The
reference's README describes how to regenerate it.

## Optional features

Every option below is off by default, and a config without it trains as
before.

### LSTM slow core

`network_config.policy_head_slow_core` is a recurrent core that updates every
`tick_stride` env steps beside the feed-forward head, so a feed-forward
checkpoint grafts into it with zero-filled new rows:

```yaml
network_config:
  policy_head_slow_core:
    hidden_size: 512
    tick_stride: 10
    cell: lstm        # gru (default) or lstm
    forget_bias: 1.0  # LSTM forget-gate bias; default 0, lstm only
```

The LSTM stores its cell and hidden state, `2 * hidden_size` wide, in the
replay at `store_dtype` (float16 by default). The recurrent learner logs
`rnn/lstm_c_absmax`, the largest absolute cell value in the sampled batch, to
watch the float16 headroom.

### In-place replay add

`train_config.replay_inplace_add: true` writes each rollout of
`unroll_length` steps into the replay as one in-place slice instead of
flashbax's scatter, which saves a copy of every replay leaf on each add. The
per-env depth, `max_replay_size // num_envs`, must be a multiple of
`unroll_length`; the entry refuses other sizes at startup. The replay holds
the same values, but the compiled training step differs, so a run is not
bit-identical to one with the scatter add (agreement is at float32
rounding).

### Full-maze critic vector

On `RodentMazeForageVision`, `env_config.maze_critic_priv: true` makes the
env write `info["critic_priv"]` for the critic alone: the visited-cell map,
the collected-treat mask, the treats' planar vectors in the torso's yaw frame,
the torso position and heading, and the time in the episode. Its width is
`n_free_cells + 3 * n_treats + 5`, 264 on the default maze. The arm must set
`network_config.critic_privileged_dim` to that width. It must also set
`env_config.episode_length`, since the time entry is normalised by the env's
horizon, which must be the one the training wrapper truncates at. The entry
refuses either mismatch. `env_config.maze_critic_priv_dtype:
float16` halves the vector's replay storage. The env also reports
`maze/visited_cells`, which the `MAZE_EVAL` line averages. The policy's
observation is unchanged.

### Contact termination

`env_config.termination_criteria.contact_termination` ends a maze episode as
soon as a walker geom outside `allowed_bodies` touches the floor, with no
patience, as in dm_control:

```yaml
env_config:
  termination_criteria:
    contact_termination:
      allowed_bodies: [foot_L, foot_R, hand_L, hand_R, vertebra_C1]  # default
      margin: -0.002  # metres; default 0. Negative tolerates that much penetration
```

`contact_termination: {}` takes the defaults. The `MAZE_EVAL` fall fraction
counts these episodes as falls.

## Known differences from acme

These are kept on purpose.

| Id | Difference | Why |
|---|---|---|
| D-uniform | Windows are sampled uniformly from stored sequences. acme's n-step adder also writes the short transitions (s₀→s₁) … (s₀→sₙ) at every episode start, which puts min(n, T) items on s₀. | An artifact of the adder, not part of the algorithm; uniform sampling is the textbook choice. |
| D-trunc | A time-limit window bootstraps one step early, from s_{T−1}, not from s_T. | s_T is not observable after the auto-reset without a second render. |
| D-term | With `policy_loss_state: next`, a window whose episode terminates inside it evaluates the policy at s_k, the last stored state of that episode, not at s_T. Its bootstrap coefficient is 0 either way. | Same cause as D-trunc. |
| D-tanh | The policy is a Gaussian over pre-tanh actions, squashed by tanh at the env. acme's is an unsquashed Gaussian whose samples the env clips. | The frozen imitation decoder was trained under a tanh-squashed Gaussian. |
| D-norm | A running observation normalizer. On grafted arms its proprioception part is seeded from the imitation checkpoint and keeps updating. acme has none. | The frozen prior and decoder need normalized proprioception. |
| D-infra | A batched MJX rollout alternates with K on-device SGD steps over a flashbax replay; the behavior policy is the online policy. acme runs distributed actors and a reverb replay with a rate limiter. | Infrastructure. The replay ratio is a config choice (`sgd_steps_per_rollout`, `samples_per_insert`). |
| D-hparams | Arm configs differ from acme's defaults (dual learning rate, KL bounds, initial duals, target periods 101/107, discount, n, atoms). | Configs are run choices, not the algorithm. |

## Experiment workflow

- Start new experiment work on its own branch off the DMPO integration
  branch, kept local or private.
- Merge it back once its CPU tests and the GPU smokes
  (`tests/scripts/test_train_dmpo_*_smoke.py`) pass.
- Before every push of the integration branch, re-check the diff for internal
  references.
- Every working recipe that later work builds on gets a baseline note with
  its exact commits, config and checkpoints, so a run can fall back to it.
