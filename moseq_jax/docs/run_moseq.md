# Running the MoSeq Decoder Pipeline (Pipeline A)

This guide covers both single-run training and full hyperparameter sweeps.

## Overview

Pipeline A trains an **RNN decoder** policy where behavioral codes come from
**Keypoint-MoSeq (KPMS)** instead of the VQ-VAE encoder. The GRU-based
recurrent decoder maintains hidden state across timesteps, enabling temporally
coherent control within a syllable. An optional continuous encoder (z_e) acts
as a training scaffold that can be annealed to zero.

```
KPMS Sweep → Code Generation → Decoder RL Training
(jax x64)    (jax x64)         (jax float32)
```

Each stage MUST run in a **separate process** because `keypoint_moseq` requires
`jax_enable_x64=True`, which is process-global and irreversible.

## Prerequisites

```bash
# Install keypoint_moseq in the track-mjx venv
.venv/bin/pip install keypoint-moseq[cuda] jax-moseq[cuda]
```

---

## Quick Start: Full Pipeline Sweep

The `run_pipeline.sh` script chains all 3 stages together for hyperparameter
tuning across both KPMS and decoder settings.

```bash
cd moseq_jax
bash run_pipeline.sh
```

This reads `configs/pipeline_sweep.yaml` and:

1. Fits KPMS models for every `(num_states × kappa)` combination (2 seeds each)
2. Generates a separate code set (`.npz`) per KPMS setting (best seed)
3. Trains a decoder for every `(code_set × latent_dim × kl_weight)` combination

### Pipeline Config (`configs/pipeline_sweep.yaml`)

```yaml
sweep:
  num_states: [10, 20, 30]            # KPMS vocabulary size
  kappa: [1.0e+2, 5.0e+2, 1.0e+3, 5.0e+3, 1.0e+4]  # syllable stickiness
  model_type: ["arhmm"]               # ARHMM only (SLDS 26x worse MSE)
  seeds_per_setting: 2                # for EML model selection

decoder:
  continuous_latent_dims: [2, 4]      # encoder bottleneck size
  kl_weights: [0.01, 0.1]            # KL regularization
  use_rnn_decoder: true               # GRU-based recurrent decoder (default)
  rnn_hidden_sizes: [256]             # GRU hidden dimensions
  use_continuous_encoder: true        # reference trajectory -> z_e
  z_e_anneal: false                   # gradually reduce z_e contribution
  num_timesteps: 500_000_000          # training budget per run
```

**Grid size**: 3 × 5 = 15 KPMS settings → 15 code sets → 15 × 2 × 2 = **60 decoder runs**.

### Pipeline Options

```bash
# Skip KPMS (reuse existing sweep results)
bash run_pipeline.sh --skip-kpms

# Skip KPMS + codegen (only re-run decoders with new decoder HPs)
bash run_pipeline.sh --skip-kpms --skip-codegen

# Use a custom config
bash run_pipeline.sh --config configs/my_sweep.yaml

# Preview what would run (no execution)
bash run_pipeline.sh --dry-run
```

### Restarting After Failure

The script runs decoder jobs sequentially and reports failures at the end.
To re-run only the decoder stage (e.g., after fixing a config issue):

```bash
bash run_pipeline.sh --skip-kpms --skip-codegen
```

Already-completed WandB runs will appear as separate entries in the same group.
Use `run_name` to identify each combination.

---

## Manual Steps (Single Run)

### Step 1: KPMS Hyperparameter Sweep

```bash
cd moseq_jax
python -m sweep.run_sweep
```

Runs a grid search defined in `configs/kpms_sweep.yaml`. Results saved to
`outputs/kpms_sweep/sweep_results.json`.

Custom config:
```bash
python -m sweep.run_sweep --config path/to/custom_sweep.yaml
```

### Step 2: Generate Codes

**Single best model** (original behavior):
```bash
python -m codegen.generate_codes
```

**Per-setting codes** (for pipeline sweep):
```bash
python -m codegen.generate_all_codes \
    --sweep-results outputs/pipeline_sweep/kpms/sweep_results.json \
    --balanced-split ../data/rodent/rodent_balanced_splits.json \
    --output-dir outputs/pipeline_sweep/codes
```

This creates one `.npz` per `(num_states, kappa)` setting plus a `manifest.json`
index.

### Step 3: Decoder RL Training

```bash
python train_moseq_decoder.py \
    kpms_config.codes_path=outputs/kpms_sweep/best_codes.npz
```

By default this uses the **RNN decoder** (`use_rnn_decoder: true` in
`moseq_decoder.yaml`).

#### Config Overrides

```bash
# RNN decoder is the default — override hidden sizes or disable
python train_moseq_decoder.py \
    network_config.rnn_hidden_sizes="[512]"

# Fall back to feedforward MLP decoder
python train_moseq_decoder.py \
    network_config.use_rnn_decoder=false

# Continuous encoder + z_e annealing
python train_moseq_decoder.py \
    network_config.use_continuous_encoder=true \
    network_config.continuous_latent_dim=4 \
    network_config.kl_weight=0.1 \
    network_config.z_e_anneal=true

# Shorter run for testing
python train_moseq_decoder.py \
    train_setup.train_config.num_timesteps=5_000_000

# Random codes (smoke test, no KPMS needed)
python train_moseq_decoder.py kpms_config.codes_path=null
```

---

## KPMS Hyperparameter Guide

Results from the initial sweep (Mar 2026, 48 fits) inform the recommended
search ranges.

### Parameters That Matter

| Parameter | Effect | Recommended Range |
|-----------|--------|-------------------|
| `num_states` | Vocabulary size. 50 states collapses to ~10 effective codes. | 10–30 |
| `kappa` | Self-transition stickiness. Higher = longer syllables, fewer transitions. | 1e2–1e4 |

### Parameters to Fix

| Parameter | Value | Reason |
|-----------|-------|--------|
| `model_type` | `arhmm` | SLDS has 26× worse reconstruction MSE |
| `latent_dim` | `10` | PCA dim; all ARHMM fits have identical MSE at fixed latent_dim |
| `ar_iters` | `50` | Sufficient for convergence |

### Key Findings

- **All ARHMM fits have identical reconstruction MSE** (0.0027) because ARHMM
  reconstruction depends only on PCA (latent_dim), not on HMM parameters.
- **kappa controls syllable duration**: 1e3 → ~50 frames, 1e4 → ~75 frames,
  1e5 → ~110 frames. At 250 frames/clip, high kappa means very few transitions.
- **50 states is misleading**: The best s50 model has 9 codes covering 98% of
  usage; 40+ codes are near-dead. Use 10–30 for meaningful vocabularies.
- **Model selection** uses: lowest MSE → highest EML → highest usage ratio.
  Within ARHMM (identical MSE), this reduces to EML → usage.

---

## Decoder Hyperparameter Guide

| Parameter | Effect | Recommended Range |
|-----------|--------|-------------------|
| `use_rnn_decoder` | Use GRU recurrent decoder (default `true`). | `true` / `false` |
| `rnn_hidden_sizes` | GRU hidden dimensions per layer. | `[256]` or `[256, 256]` |
| `continuous_latent_dim` | Encoder bottleneck. Smaller = more compressed. | 2–8 |
| `kl_weight` | KL penalty on continuous latent. Higher = more regularized. | 0.01–0.5 |
| `code_embed_dim` | Discrete code embedding size. | 8–32 |
| `z_e_anneal` | Gradually reduce z_e scale to 0 during training. | `true` / `false` |
| `z_e_anneal_start_frac` | Fraction of training when annealing starts. | 0.3 |
| `z_e_anneal_end_frac` | Fraction of training when z_e reaches 0. | 0.7 |

---

## Architecture

### RNN Decoder Policy (default, `use_rnn_decoder: true`)

```
obs["imitation_target"]
    → MLP encoder → (mean, logvar) → reparameterize → z_e   (optional)

obs["kpms_code"] (float)
    → round → Embed(num_codes, embed_dim) → code_emb

obs["proprioception"]
    → flatten → proprio

concat(code_emb, z_e * z_e_scale, proprio)
    → GRU(hidden_state) → new_hidden_state
    → Dense → action_params

Hidden state is carried across timesteps within an episode and reset on
episode boundaries. During training, `apply_sequence` uses `jax.lax.scan`
to replay the full unroll with hidden state threading.

z_e_scale can be annealed from 1.0 → 0.0 during training to gradually
remove the encoder scaffold, forcing the decoder to rely on code_emb +
hidden state alone.
```

### Feedforward Decoder Policy (`use_rnn_decoder: false`)

```
concat(code_emb, z_e, proprio)
    → Dense → SiLU → LayerNorm  ×4
    → Dense → action_params
```

### Value Network

```
obs["imitation_target"] + obs["proprioception"]
    → normalize → flatten → flat_obs

obs["kpms_code"]
    → round → Embed(num_codes, embed_dim) → code_emb

concat(flat_obs, code_emb)
    → MLP → scalar value
```

### Loss

Standard PPO (clipped surrogate + value + entropy) + KL on continuous latent.
No VQ-VAE auxiliary losses. For the RNN decoder, the loss recomputes the forward
pass via `apply_sequence` (scan) to thread hidden state, with stored PRNG keys
for deterministic z_e replay.

---

## WandB Metrics

| Metric | Description |
|--------|-------------|
| `moseq/perplexity` | Code usage entropy (higher = more uniform) |
| `moseq/codes_used` | Number of active codes |
| `moseq/eval_transition_rate` | Fraction of steps with code changes |
| `moseq/code_sequence` | Timeline visualization of code usage |
| `kl_loss` | KL divergence of continuous latent |
| `hidden_state_norm` | Mean GRU hidden state norm (RNN decoder only) |
| `z_e_scale` | Current z_e multiplier (tracks annealing progress) |
| `z_e_norm` | Mean L2 norm of continuous latent |

---

## Pipeline Outputs

```
outputs/pipeline_sweep/
├── kpms/                          # Stage 1: KPMS sweep
│   ├── sweep_results.json         # All fit results + best per setting
│   ├── s10_k1e+02_l10_arhmm/     # Per-setting fit directories
│   │   ├── seed0/
│   │   └── seed1/
│   ├── s10_k5e+02_l10_arhmm/
│   │   └── ...
│   └── ...
├── codes/                         # Stage 2: per-setting codes
│   ├── manifest.json              # Index of all code sets
│   ├── s10_k1e+02_arhmm.npz
│   ├── s10_k5e+02_arhmm.npz
│   └── ...
└── (decoder checkpoints go to model_checkpoints/ per moseq_decoder.yaml)
```

## File Structure

```
moseq_jax/
├── run_pipeline.sh                # Full pipeline orchestration script
├── configs/
│   ├── kpms_sweep.yaml            # KPMS-only sweep config
│   ├── moseq_decoder.yaml         # Decoder RL training config
│   └── pipeline_sweep.yaml        # Full pipeline sweep config
├── kpms/
│   ├── config.py                  # KPMSHyperparams dataclass
│   ├── keypoint_loader.py         # qpos → keypoints via FK
│   └── fit_kpms.py                # Single KPMS fit
├── sweep/
│   └── run_sweep.py               # KPMS grid search
├── codegen/
│   ├── generate_codes.py          # Codes from global best model
│   └── generate_all_codes.py      # Codes per (num_states, kappa) setting
├── moseq_env_wrapper.py           # MoSeqImitation env subclass
├── moseq_decoder_network.py       # Encoder-decoder Flax module
├── moseq_ppo_networks.py          # Network factories + inference fns
├── moseq_ppo.py                   # PPO train wrapper (monkey-patching)
├── moseq_losses.py                # PPO loss + KL loss
├── train_moseq_decoder.py         # Decoder RL entry point
└── docs/
    └── run_moseq.md               # This file
```
