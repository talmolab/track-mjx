# Running the MoSeq Decoder Pipeline (Pipeline A)

This guide covers the end-to-end KPMS decoder-only RL training pipeline.

## Overview

Pipeline A trains a **decoder-only** policy where behavioral codes come from
**Keypoint-MoSeq (KPMS)** instead of the VQ-VAE encoder. This enables comparing
a pre-computed tokenizer (KPMS) against the learned tokenizer (VQ-VAE) for
motor control.

```
KPMS Sweep → Code Generation → Decoder RL Training
(separate process)  (separate process)   (main process)
```

## Prerequisites

```bash
# Install keypoint_moseq in the track-mjx venv
.venv/bin/pip install keypoint-moseq[cuda] jax-moseq[cuda]
```

## Step 1: KPMS Hyperparameter Sweep

**IMPORTANT**: This step MUST run in a separate process because
`keypoint_moseq` requires `jax_enable_x64=True`, which is process-global.

```bash
cd moseq_jax
python -m sweep.run_sweep
```

This runs a grid search over:
- `num_states`: [10, 20, 30, 50]
- `kappa`: [1e3, 1e4, 1e5, 1e6]
- `latent_dim`: [5, 10]
- `model_type`: ["arhmm", "slds"]
- 3 seeds per setting

Results are saved to `moseq_jax/outputs/kpms_sweep/sweep_results.json`.

To use a custom config:
```bash
python -m sweep.run_sweep --config path/to/custom_sweep.yaml
```

## Step 2: Generate Codes

Extract syllable codes from the best model (also requires x64 process):

```bash
cd moseq_jax
python -m codegen.generate_codes \
    --sweep-results outputs/kpms_sweep/sweep_results.json \
    --balanced-split ../data/rodent/rodent_balanced_splits.json \
    --output outputs/kpms_sweep/best_codes.npz \
    --wandb-project moseq_experiments
```

Output `.npz` contains:
- `train_codes`: shape `[N_train, 250]` int32
- `test_codes`: shape `[N_test, 250]` int32
- `train_indices`, `test_indices`: original clip indices

## Step 3: Decoder RL Training

```bash
cd moseq_jax
python train_moseq_decoder.py \
    kpms_config.codes_path=outputs/kpms_sweep/best_codes.npz
```

### Smoke Test (Random Codes)

To test the pipeline without running the KPMS sweep first:

```bash
python train_moseq_decoder.py \
    kpms_config.codes_path=null \
    train_setup.train_config.num_timesteps=5_000_000
```

This generates random codes and trains for 5M steps.

### Key Config Overrides

```bash
# Change number of codes
python train_moseq_decoder.py network_config.num_codes=64

# Change embedding dimension
python train_moseq_decoder.py network_config.code_embed_dim=32

# Change decoder architecture
python train_moseq_decoder.py \
    "network_config.decoder_layer_sizes=[1024,1024,512,512]"
```

## Architecture

### Network

```
obs["kpms_code"] (float, shape [1])
    → round to int → Embed(num_codes, embed_dim) → code_emb

obs["proprioception"] (normalized)
    → flatten → proprio

concat(code_emb, proprio)
    → Dense → SiLU → LayerNorm  ×4
    → Dense → action_params
```

### Value Network

```
obs["imitation_target"] + obs["proprioception"]
    → normalize → flatten → flat_obs

obs["kpms_code"]
    → round to int → Embed(num_codes, embed_dim) → code_emb

concat(flat_obs, code_emb)
    → MLP → scalar value
```

### Loss

Standard PPO (clipped surrogate + value + entropy). No VQ-VAE auxiliary
losses.

## WandB Metrics

| Metric | Description |
|--------|-------------|
| `moseq/perplexity` | Code usage entropy (higher = more uniform) |
| `moseq/codes_used` | Number of active codes |
| `moseq/eval_transition_rate` | Fraction of steps with code changes |
| `moseq/code_sequence` | Timeline visualization of code usage |

## File Structure

```
moseq_jax/
├── configs/
│   ├── kpms_sweep.yaml        # KPMS grid search config
│   └── moseq_decoder.yaml     # RL training config
├── kpms/
│   ├── config.py              # KPMSHyperparams dataclass
│   ├── keypoint_loader.py     # qpos → keypoints via FK
│   └── fit_kpms.py            # Single KPMS fit
├── sweep/
│   └── run_sweep.py           # Grid search
├── codegen/
│   └── generate_codes.py      # Extract codes from best model
├── moseq_env_wrapper.py       # MoSeqCodeWrapper
├── moseq_decoder_network.py   # Decoder Flax module
├── moseq_ppo_networks.py      # Network factories + inference fns
├── moseq_ppo.py               # PPO train wrapper (monkey-patching)
├── moseq_losses.py            # Standard PPO loss
├── train_moseq_decoder.py     # RL training entry point
└── docs/
    └── run_moseq.md           # This file
```
