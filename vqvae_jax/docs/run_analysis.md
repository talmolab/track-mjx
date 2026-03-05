# Analysis & Ablation Pipelines

Both pipelines log to the same WandB project (`vqvae-eval`). Set `wandb.enabled=false` to disable, or use `WANDB_MODE=offline` for offline logging.

## Analysis

Analyzes VQ-VAE code semantics from pre-computed rollout H5 files: global transition matrix, t-SNE trajectory visualization, pose gallery, and kinematic profiles.

### 1. Generate rollout data

```bash
cd vqvae_jax
python -m inference.run_inference checkpoint.path=/path/to/checkpoint inference.data_split=test
```

This writes an H5 file to `./outputs/` (default: `rollout_rvq_32_test.h5`).

### 2. Run analysis

```bash
python -m analysis.code_analysis
```

Key overrides:
- `data.analysis_split=train` to analyze train split instead of test
- `pose_gallery.top_k_codes=8` to change number of codes in pose gallery
- `tsne_trajectory.enabled=false` to skip t-SNE visualization
- `kinematic_profile.enabled=false` to skip kinematic profiles

## Code Ablation

Tests what each D0 code does by mutating the codebook at inference time. Two experiments: `code_injection` (force a specific D0 code, zero D1) and `d0_only` (natural D0 via encoder, zero D1). Runs from two starting poses (lowest/highest torso z).

```bash
cd vqvae_jax
python -m ablation.run_ablation
```

Key overrides:
- `ablation.top_k=8` to inject more codes
- `ablation.experiments=[code_injection]` to run only code injection
- `ablation.max_steps=1000` to extend rollout length
- `ablation.data_split=train` to use train clips for starting poses
