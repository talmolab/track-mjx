# MoSeq Experiments

## Overview

Five experiment scripts evaluate the KPMS Code2Act decoder against a Mimic-MJX oracle baseline:

| # | Script | Config | Purpose |
|---|--------|--------|---------|
| 1 | `run_inference.py` | `inference.yaml` | Trajectory matching, reward decomposition, K-body videos, transition matrices |
| 2 | `run_code_sequence.py` | `code_sequence_exp.yaml` | Temporal order of codes, killer demo (instructional/discriminative codes) |
| 3 | `run_code_generation.py` | `code_generation_exp.yaml` | Generative models (empirical TM, dynamax HMM, ARHMM L1/L2) + free-loop rollouts |
| 4 | `run_generalization.py` | `generalization.yaml` | Generalization to unseen continuous data via KPMS re-inference |
| 5 | `run_roundtrip.py` | `roundtrip.yaml` | Round-trip code consistency (qpos -> FK -> KPMS -> codes, compare to original) |

All experiments save outputs (plots, videos, npz data) to disk under `outputs/`.

## Prerequisites

- Trained MoSeq decoder checkpoint (default: `model_checkpoints/260407_031233_484020/`)
  - This checkpoint uses `use_pretrained_decoder=true` with a Mimic-MJX IntentionNetwork decoder
- Mimic-MJX oracle checkpoint (default: `trained_mimic_ckpts/260405_235031_234849/`)
- KPMS codes at `outputs/kpms_sweep/best_codes.npz`
- Balanced splits at `data/rodent/rodent_balanced_splits.json`
- Python packages: `jax`, `flax`, `orbax-checkpoint`, `dynamax`, `imageio`, `matplotlib`
- For generalization/roundtrip: `keypoint_moseq`, unseen H5 data, stac-mjx rodent XML

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

# Fewer clips for testing
python -m experiments.run_inference inference.splits=["test"] rendering.K=3
```

**Outputs** (`outputs/moseq_inference/`):
- `{split}_{mode}.npz` — rollout data (qpos, rewards, codes, decomposed rewards)
- Reward decomposition plots (coarse vs fine)
- K-body ghost videos + solo body videos

### Experiment 2: Code Sequence

```bash
# Full run
python -m experiments.run_code_sequence

# Quick test (fewer clips)
python -m experiments.run_code_sequence temporal_order.K=3 killer_demo.K=3
```

**Outputs** (`outputs/moseq_code_sequence/`):
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

**Outputs** (`outputs/moseq_code_generation/`):
- Transition matrix heatmaps per generative method
- Solo body videos from generated code sequences
- Survival comparison bar chart
- Mimic-MJX oracle survival baseline

### Experiment 4: Generalization

```bash
# Full run (20 segments x 1000 frames from unseen data)
python -m experiments.run_generalization

# Fewer segments
python -m experiments.run_generalization new_data.n_segments=5
```

Samples segments from a new unsegmented recording, runs KPMS inference to
extract syllable codes, then evaluates Code2Act vs Mimic-MJX on those segments.

**Outputs** (`outputs/moseq_generalization/`):
- `generalization_{mode}.npz` — rollout data per mode
- `generalization_codes.npz` — KPMS codes extracted from new data
- Reward decomposition plot
- Mean reward comparison bar chart
- Solo videos per segment per mode

### Experiment 5: Round-Trip

```bash
python -m experiments.run_roundtrip
```

Tests whether Code2Act behaviour, re-encoded by the same KPMS model, recovers
the original input codes. Three conditions: reference (ceiling), Mimic-MJX
(oracle), Code2Act. Runs on two datasets: 250-frame test set and 1000-frame
generalization set.

**Prerequisites**: Requires output from Experiments 1 and 4 (the npz files
with rollout qpos for each mode).

**Outputs** (`outputs/moseq_roundtrip/`):
- Confusion matrices per condition per dataset
- Accuracy comparison bar chart (PNG + SVG)
- `roundtrip_summary.json` — frame-level accuracy per condition

## Architecture Support

All experiments work with both:
- **Feedforward (MLP)** decoder: `network_config.use_rnn_decoder=false`
- **Recurrent (GRU)** decoder: `network_config.use_rnn_decoder=true`
- **Pretrained decoder** (from Mimic-MJX): `network_config.use_pretrained_decoder=true`

The scripts auto-detect architecture from the checkpoint config.

## Configuration

Each experiment's YAML config contains:
- `checkpoint`: Code2Act decoder path and step
- `mimic_checkpoint`: Mimic-MJX oracle path and step (Experiments 1-4)
- Experiment-specific parameters (K, max_steps, conditions, etc.)
- `rendering`: camera, resolution, fps
- `env_config`: physics settings (domain randomization disabled for deterministic inference)
- `walker_config`: rodent body model
