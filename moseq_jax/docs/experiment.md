# MoSeq Experiments

## Overview

Three experiment scripts evaluate the KPMS Code2Act decoder:

| Script | Config | Claims |
|--------|--------|--------|
| `run_inference.py` | `inference.yaml` | 2.1–2.4: Trajectory matching, reward decomposition, transition analysis, K-body videos, transition matrices |
| `run_code_sequence.py` | `code_sequence_exp.yaml` | 3.1–3.3 + 5: Temporal order of codes, killer demo (instructional/discriminative codes) |
| `run_code_generation.py` | `code_generation_exp.yaml` | 4: Generative models (empirical TM, dynamax HMM, ARHMM L1/L2) + free-loop rollouts |

All experiments log to WandB progressively (plots and videos as they complete).

## Prerequisites

- Trained MoSeq checkpoint (default: `model_checkpoints/260326_014031_608396/`)
- KPMS codes at `outputs/kpms_sweep/best_codes.npz`
- Balanced splits at `data/rodent/rodent_balanced_splits.json`
- Python packages: `jax`, `flax`, `orbax-checkpoint`, `wandb`, `dynamax`, `imageio`, `matplotlib`

## Running Experiments

All commands run from the `moseq_jax/` directory:

```bash
cd moseq_jax
```

### Experiment 1: Inference

```bash
# Full run (all test clips, both modes)
python -m experiments.run_inference

# Override checkpoint
python -m experiments.run_inference checkpoint.path=/path/to/checkpoint

# Run offline (no WandB upload)
WANDB_MODE=offline python -m experiments.run_inference

# Fewer clips for testing
python -m experiments.run_inference inference.splits=["test"] rendering.K=3
```

**Outputs:**
- `outputs/moseq_inference/{split}_{mode}.npz` — saved rollout data
- Reward decomposition plots (coarse vs fine, full vs code-only)
- Transition window analysis around code boundaries
- K-body ghost videos + solo body videos
- Transition matrix heatmaps

### Experiment 2: Code Sequence

```bash
# Full run
python -m experiments.run_code_sequence

# Quick test (fewer clips)
python -m experiments.run_code_sequence temporal_order.K=3 killer_demo.K=3
```

**Outputs:**
- Divergence curves (correct vs shuffled-step vs shuffled-trajectory)
- K-body ghost videos per condition
- Killer demo: K bodies at same start, different code sequences
- Root displacement plots per behaviour

### Experiment 3: Code Generation

```bash
# Full run (all 4 methods)
python -m experiments.run_code_generation

# Single method
python -m experiments.run_code_generation generation.methods=["transition_matrix"]

# Fewer sequences for testing
python -m experiments.run_code_generation generation.num_sequences=3
```

**Outputs:**
- Transition matrix heatmaps per generative method
- Solo body videos from generated code sequences
- Survival comparison bar chart

## Architecture Support

All experiments work with both:
- **Feedforward (MLP)** decoder: set `network_config.use_rnn_decoder=false`
- **Recurrent (GRU)** decoder: set `network_config.use_rnn_decoder=true`

The scripts auto-detect from the checkpoint config.

## Configuration

Each experiment's YAML config contains:
- `checkpoint`: path and step
- Experiment-specific parameters (K, max_steps, conditions, etc.)
- `rendering`: camera, resolution, fps
- `wandb`: project, entity
- `env_config`: physics settings (DR disabled for deterministic inference)
- `walker_config`: rodent body model

## WandB Logging

All experiments log progressively. Key sections:

- `inference/` — reward plots, transition matrices, videos
- `temporal_order/` — divergence curves, condition ghost videos
- `killer_demo/` — per-behaviour ghost videos, displacement plots
- `code_gen/` — per-method transition matrices, solo videos, survival
