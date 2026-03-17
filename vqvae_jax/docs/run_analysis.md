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

## Code Usability by Posture

For each starting posture (low/high torso z-height), independently classifies codes into preferred/medium/not-preferred by within-pose frequency terciles from inference H5 data, then compares them via decoder-only injection with tabbed HTML and quantitative metrics.

```bash
cd vqvae_jax
python -m ablation.run_code_usability
```

Key overrides:
- `usability.max_steps_per_code=300` to extend injection rollout length
- `usability.z_split=0.05` to use an explicit z-height threshold instead of quartile
- `render.enabled=false` to skip video rendering (metrics only)
- `data.h5_path=./outputs/rollout_train.h5` to analyze train split

WandB panels:
- `code_usability/{low,high}_height/viewer` — tabbed HTML (preferred/medium/not-preferred)
- `code_usability/metrics/cross_pose_scatter` — joint velocity scatter
- `code_usability/metrics/bars_{pose}` — preferred vs not-preferred bar charts
- `code_usability/metrics/activity_heatmap` — code x pose heatmap with per-pose rank
, 
## HMM Prior

Fits a discrete HMM on D0 code sequences and generates free-loop behavior using only the decoder (no encoder). Validates temporal structure in the learned code space.

```bash
cd vqvae_jax
python -m hmm_prior.run_hmm_prior
```

Key overrides:
- `hmm.num_states_sweep=[8,16]` to change sweep range
- `free_loop.commitment_horizon=5` to change code holding duration
- `free_loop.max_steps=2000` for longer rollouts

## Divergent Futures

Demonstrates that D0 codes encode categorically different motor plans (not just quality modulators). Finds clips with similar initial rearing poses but divergent D0 code futures from inference H5, then runs 3 decoder-only conditions from the same starting state: correct codes (A), random step-excluded codes (B), and random trajectory-excluded codes (C). Renders overlaid ghost-body videos and quantifies trajectory divergence.

### Prerequisites

Generate rollout H5 data first (see [Analysis](#analysis) step 1).

### Run

```bash
cd vqvae_jax
python -m ablation.run_divergent_futures
```

Key overrides:
- `experiment.K=5` to compare more trajectories (default: 3)
- `experiment.max_steps=1000` for longer rollouts
- `experiment.pose_selection.z_percentile=80` to relax rearing threshold
- `experiment.pose_selection.joint_distance_threshold=2.0` to allow more pose variation
- `experiment.pose_selection.min_code_divergence=0.5` to relax code divergence filter
- `render.camera=close_profile` for a tighter camera angle (default: `top`)
- `render.enabled=false` to skip video rendering (metrics only)

### Outputs

All saved to `./outputs/divergent_futures/` (configurable via `output.base_dir`):
- `divergent_futures.html` — single-page summary with embedded videos, divergence plot, and metrics table
- `reference_trajectories.mp4` — overlaid ghost video of the selected H5 reference clips
- `condition_a_correct.mp4` / `condition_b_step-excluded.mp4` / `condition_c_traj-excluded.mp4` — ghost videos per condition
- `divergence_curves.png` — mean pairwise joint L2 over time for all 3 conditions
- `divergent_futures_summary.json` — machine-readable metrics

WandB panels (under `divergent/`):
- `divergent/reference_trajectories` — reference video
- `divergent/a_correct`, `b_step-excluded`, `c_traj-excluded` — condition videos
- `divergent/divergence_curves` — divergence plot
- `divergent/summary` — full HTML report
