---
title: LatentMimic + DMPO JAX port — Collaborator Handoff
author: Scott Yang (scyang@salk.edu)
date: 2026-05-12
---

# LatentMimic + DMPO JAX port — Collaborator Handoff

This doc covers three "clean" branches that carve out the LatentMimic and DMPO
JAX/linen port work from `scott_claude/latent-mimic` (track-mjx) and
`scott_claude/gap-jump-trial-elife` (vnl-playground), where it had grown
intertwined with unrelated work (gap-jump-trial-elife, RNN/Vision, binocular
overlap sweeps, etc.).

The original messy branches are still intact and tagged
(`scott_claude/dmpo-latent-mimic-snapshot-2026-05-12` on each).

## Source paper

Wang et al., *LatentMimic: Terrain-Adaptive Locomotion via Latent Space
Imitation*, arXiv:2604.16440v1, 7 Apr 2026. PDF is checked in at
`ClaudeCode_PromptHistory/2026-04-25-1-latent-mimic/Wang et al. - 2026 - LatentMimic ....pdf`.

Three phases per the paper:
1. **Phase 1**: pretrain motion autoencoder (encoder + decoder) and motion
   predictor on rat mocap reference clips. Frozen weights feed Phase 2.
2. **Phase 2**: PPO with `r_mimic` reward = KL(z_sim || z_target) — pin the
   policy's encoded motion to the prior-encoded reference motion.
3. **Phase 3**: terrain-adaptive RL with `r_anchor` style anchor (policy
   KL-anchored to a frozen prior+decoder) on out-of-distribution terrain.

---

## What's in which branch

| Repo | Branch | What's in it | Commits |
|---|---|---|---|
| track-mjx | `scott_claude/dmpo-jax-port` | JAX/linen DMPO port + KL-anchor extensions + track-mjx imitation entries | 4 |
| track-mjx | `scott_claude/latent-mimic-handoff` | Phase 1 pretraining + Phase 2 latent KL RL | 3 |
| vnl-playground | `scott_claude/dmpo-jax-port` | DMPO training entries + KL-anchor wrapper + sweep scripts | 3 |

Both track-mjx branches are independent off `origin/main`. The
vnl-playground branch depends on the track-mjx DMPO-port branch being checked
out (it imports `track_mjx.agent.dmpo.*`).

### track-mjx `scott_claude/dmpo-jax-port`

```
8e15214 feat(dmpo): track-mjx DMPO training entries + configs
15894c0 feat(dmpo): KL-anchor extensions for LatentMimic Phase 3
9a1812e feat(dmpo): JAX/linen DMPO port
345fe1e chore: foundational dict-observation + binocular vision infra
```

- **9a1812e** is the core port: linen GaussianPolicyHead + CategoricalCriticHead
  (C51), MPO dual loss + categorical Bellman, flashbax n-step replay, fused
  rollout+replay+scan-K-SGD jit step, train_dmpo entry with wandb resume +
  orbax checkpoints + eval rollout. Numerical parity test vs vnl-ray TF MPO
  at 1e-4 tolerance.
- **15894c0** adds the KL-anchor branch in `learner.py` (post-MPO `-α * E[KL
  to prior]`), `kl_anchor_utils.pretanh_gaussian_kl`, `networks_kl_anchor`
  with prior+decoder warm-start, `normalizer_seeding`, asymmetric-LR
  optimizer. Opt-in via `cfg.kl_anchor_alpha > 0`. Defaults disable it.
- **8e15214** is the track-mjx-side imitation entries (`track_mjx.train_dmpo`,
  `track_mjx.train_dmpo_imitation`) and their YAML configs.
- **345fe1e** brings forward Scott-branch infrastructure (dict-obs schema,
  binocular vision encoders) that the DMPO port consumes but that hasn't
  merged to main yet. Should drop out cleanly once those upstream pieces land.

### track-mjx `scott_claude/latent-mimic-handoff`

```
32df62a feat(latent_ppo): LatentMimic Phase 2 - latent KL r_mimic training
f34dd86 feat(latent_ppo): LatentMimic Phase 1 - latent prior pretraining
6b15366 chore: foundational track-mjx infrastructure for LatentMimic
```

- **f34dd86 (Phase 1)**: `track_mjx/agent/latent_ppo/` autoencoder +
  predictor with conv1d encoder torso, sigma cap, dead-dim mask. Pretrain
  Hydra entry, warmup-hold-cosine LR, best-on-val orbax checkpoint, wandb
  recon figures. Configs `latent_mimic_pretrain*.yaml` (v9..v15 are iter
  history; v15 is most recent).
- **32df62a (Phase 2)**: `prior_module.FrozenLatentPrior` loads the Phase 1
  checkpoint; `LatentMimicEnvWrapper` computes `z_sim` / `z_target` and emits
  `r_mimic`; `term_curriculum` implements paper Eq. 10 0.5→2π rad ramp;
  brax PPO policy/value factory + `track_mjx.train` top-level dispatcher
  via `network_registry`. Configs `latent_mimic_phase2*.yaml` (v9d is the
  most recent variant).
- **6b15366** brings the same kind of Scott-branch infrastructure forward
  (broader scope than the DMPO-port's prep — also includes refreshed
  recurrent_ppo, wandb_logging, config/utils, walker_registry).

`latent_ppo` is fully isolated from `dmpo` (no imports between them).

### vnl-playground `scott_claude/dmpo-jax-port`

```
0fb55f3 chore: bring in updated tasks/wrappers.py for PriorHighLevelWrapper
fdbe3dc feat(dmpo): vnl-playground KL-anchor (LatentMimic Phase 3) entry
0f10df3 feat(dmpo): vnl-playground DMPO training entries
```

- **0f10df3**: `train_dmpo.py` (downstream tasks), `train_highlvl_dmpo.py`
  (prior+decoder warm-start consumer of `PriorHighLevelWrapper`),
  `naccdmax_patch.py` (Warp broadphase contact budget — required on
  5090+Warp), DMPO YAMLs in `config/rodent_run_gap_dmpo/` (velocity_only,
  vision, vision_scratch_position), smoke test.
- **fdbe3dc**: `train_highlvl_dmpo_kl_anchor.py` (Phase 3 trainer),
  `tasks/wrappers_kl_anchor.KLAnchorPriorDecoderWrapper` (runs frozen
  prior+decoder per step, exposes `(mu_imit, log_std_imit)` in
  `state.info`), `tasks/prior_utils.py` (SCAMPER prior loader +
  pre-tanh logits helper), `velocity_only_kl_anchor.yaml` config,
  autonomous sweep scripts, smoke tests.
- **0fb55f3**: adds `PriorHighLevelWrapper` to `tasks/wrappers.py` (without
  this, train_highlvl_dmpo's import fails).

---

## Environment & setup

```bash
# Activate the project venv (per project CLAUDE.md).
source /home/talmolab/Desktop/SalkResearch/mimic-mjx/bin/activate

# Use uv pip, not bare pip.
uv pip install -e /path/to/track-mjx
uv pip install -e /path/to/vnl-playground
# SCAMPER is needed at runtime for the Phase 3 (kl_anchor) path
# (vnl_playground/tasks/prior_utils.py imports scamper.agent.{imitation,mlp_prior,observation_utils}).
```

After checkout:

```bash
cd track-mjx && git checkout scott_claude/dmpo-jax-port              # for DMPO + kl_anchor
# OR
cd track-mjx && git checkout scott_claude/latent-mimic-handoff       # for Phase 1 + Phase 2

cd vnl-playground && git checkout scott_claude/dmpo-jax-port          # only when running DMPO consumers
```

---

## How to run each piece

### Phase 1 — pretrain the latent prior on rat mocap

`track-mjx scott_claude/latent-mimic-handoff` checked out.

```bash
python -m track_mjx.agent.latent_ppo.pretrain --config-name=latent_mimic_pretrain_v15
```

Reference clips: `vnl_playground/tasks/rodent/reference_data/reference_clips.h5`
(DVC-tracked; `dvc pull` first if missing).

Outputs `mimic_prior_*.msgpack` orbax checkpoint with frozen encoder /
predictor / normalizer that Phase 2 consumes.

### Phase 2 — RL with r_mimic (latent KL)

`track-mjx scott_claude/latent-mimic-handoff` checked out. Edit
`track_mjx/config/latent_mimic_phase2_v9d.yaml`'s `prior_checkpoint_path` to
point at the Phase 1 checkpoint.

```bash
python -m track_mjx.agent.latent_ppo.train_phase2 --config-name=latent_mimic_phase2_v9d
```

### DMPO without LatentMimic (vanilla port)

`track-mjx scott_claude/dmpo-jax-port` checked out.

```bash
# Imitation, in-repo task.
python -m track_mjx.train_dmpo --config-name=rodent-dmpo-imitation

# Or via the intention-network entry.
python -m track_mjx.train_dmpo_imitation --config-name=rodent-dmpo-imitation-intention
```

### DMPO on vnl-playground downstream tasks (run-gap etc.)

`track-mjx scott_claude/dmpo-jax-port` AND `vnl-playground scott_claude/dmpo-jax-port`
checked out.

```bash
# Downstream task — pure DMPO.
python -m vnl_playground.train_dmpo --config-name=velocity_only

# Prior+decoder warm-start (no anchor).
python -m vnl_playground.train_highlvl_dmpo \
    --config-path=vnl_playground/config/rodent_run_gap_dmpo --config-name=velocity_only
```

### Phase 3 / KL-anchor — DMPO pinned to a frozen prior+decoder

Same checkouts as above.

```bash
python -m vnl_playground.train_highlvl_dmpo_kl_anchor \
    --config-path=vnl_playground/config/rodent_run_gap_dmpo \
    --config-name=velocity_only_kl_anchor
```

Tunable knobs (in YAML):
- `kl_anchor_alpha`: weight on the anchor loss term (0 disables it).
- `kl_anchor_w`: weight on the anchor "KL-reward" before annealing.
- `kl_anchor_w_floor`, `kl_anchor_decay_sgd_steps`: linear w-decay schedule.
  Latest result (2026-05-06): w-decay at 600M sustained higher variance
  (4.0 MB band) vs static (collapsed to 2.75 MB at 471M).

The autonomous sweep loops live at `scripts/run_dmpo_kl_anchor_loop.py` and
`scripts/run_dmpo_kl_anchor_fast_sweep.py`.

---

## Known WIP / gotchas

- **w-decay tuning is open.** Empirical sweep is in progress (see autonomous
  sweep scripts). Static `kl_anchor_w` collapsed the policy stddev around
  471M steps; w-decay run at 600M held a wider stddev band.
- **`action_penalization` must be False** for the KL-anchor flow. The vnl-ray
  formula penalizes all action magnitudes which fights the prior; with
  `action_penalization=False`, `init_log_alpha_stddev=1`, `eps_stddev=1e-3`,
  the policy explores.
- **5090 memory ceilings:** MLP runs at 8192 envs are fine; RNN runs at 8192
  envs OOM — use 4096 envs for RNN variants. KL-anchor (Phase 3) currently
  uses MLP.
- **`naconmax`** must be ≥ 65536 for the kl_anchor wrapper (otherwise
  contact-count overflow on dense gap scenes). Already set in the YAML.
- **The `chore: foundational ... infra` commits on each branch are not the
  feature itself.** They are Scott-branch upstream infrastructure (dict-obs
  schema migration, refreshed vision encoders, etc.) that the feature
  depends on but that hasn't formally merged to main yet. When that upstream
  work merges, those commits should be dropped via interactive rebase.
- **vnl-playground task registry on main is incomplete.** The handoff branch
  only updates `tasks/wrappers.py`, not `tasks/__init__.py`. End-to-end
  training of the run-gap downstream task will require the corresponding
  `run_gap_vision`, `gap_jump_trial`, etc. task modules — those live on
  `scott_claude/gap-jump-trial-elife` and are unrelated to latent-mimic. If
  the collaborator needs them, easiest path is to merge that branch (or its
  subset) into the DMPO-port branch.

---

## Planning + experiment-log pointers

The plan-driven workflow under
`/home/talmolab/Desktop/SalkResearch/ClaudeCode_PromptHistory/` captures the
design history. Most-relevant dirs:

- `2026-04-25-1-latent-mimic/plan.md` — full LatentMimic implementation plan,
  including paper-faithful decisions made on 2026-04-26.
- `2026-04-28-2-kl-anchor-task-transfer/` — initial kl-anchor design.
- `2026-05-04-1-dmpo-train-dmpo/`, `2026-05-04-2-dmpo-gpu-saturation/`,
  `2026-05-04-3-dmpo-script-split/` — DMPO JAX port iterations.
- `2026-05-05-1-dmpo-highlvl-prior-decoder/` — prior+decoder warm-start.
- `2026-05-05-2-dmpo-imitation-intention/` — intention-encoder DMPO entry.
- `2026-05-05-2-dmpo-kl-anchor-b-aggressive/` — B-aggressive policy/critic
  shape for kl-anchor.
- `2026-05-06-1-dmpo-kl-anchor-warmstart-fix/` — warm-start invariant fix
  (r_anchor=0.926 live as of 2026-05-06).
- `2026-05-06-2-dmpo-kl-in-loss-port/` — SCAMPER-style KL-in-loss formula.

Recent wandb runs are tagged with `kl_a_v1..v6` (Phase B sweep) and the
`prior_decoder_*` family.

---

## Provenance / recovery

If anything in a clean branch turns out to be wrong, the original commits
are still on:

- `scott_claude/latent-mimic` (track-mjx) — tagged
  `scott_claude/dmpo-latent-mimic-snapshot-2026-05-12`
- `scott_claude/gap-jump-trial-elife` (vnl-playground) — tagged
  `scott_claude/dmpo-latent-mimic-snapshot-2026-05-12`

A local `tmp/dmpo-jax-port-before-prep` (track-mjx) tag also records the
pre-restructure state of the DMPO-port branch.

---

## Questions

Scott — scyang@salk.edu
