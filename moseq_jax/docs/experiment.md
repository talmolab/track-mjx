# MoSeq Experiments

## Overview

Eleven experiment scripts evaluate the KPMS Code2Act decoder:

| # | Script | Config | Purpose |
|---|--------|--------|---------|
| 1 | `run_inference.py` | `inference.yaml` | Trajectory matching, reward decomposition, K-body videos, transition matrices |
| 2 | `run_code_sequence.py` | `code_sequence_exp.yaml` | Temporal order of codes, killer demo (instructional/discriminative codes) |
| 3 | `run_code_generation.py` | `code_generation_exp.yaml` | Generative models (empirical TM, dynamax HMM, ARHMM L2) + free-loop rollouts |
| 4 | `run_generalization.py` | `generalization.yaml` | Generalization to unseen continuous data via KPMS re-inference |
| 5 | `run_roundtrip.py` | `roundtrip.yaml` | Round-trip code consistency (qpos -> FK -> KPMS -> codes, compare to original) |
| 6 | `run_single_code.py` | `single_code_exp.yaml` | Single-code sustain grid — each code held for K frames, 2 body poses, 5×10 grid video |
| 7 | `run_behavior_parade.py` | `behavior_parade_exp.yaml` | Behavior transition parade — 10 bodies, top-down view, walk→groom→rear sequence |
| 8 | `run_syllable_viz.py` | `syllable_viz_exp.yaml` | KPMS syllable 3D visualization — interactive Plotly trajectories, frequency/duration stats, dendrogram |
| 9 | `run_inception_distance.py` | `inception_distance_exp.yaml` | FID/KID inception distance — generative model distribution quality vs real mocap |
| 10 | `run_gait_dynamics.py` | `gait_dynamics_exp.yaml` | Gait dynamics PSD — Code2Act vs Mimic-MJX walking frequency comparison |
| 11 | `run_hidden_dynamics.py` | `hidden_dynamics_exp.yaml` | RNN hidden state dynamics — PCA/t-SNE/UMAP of GRU hidden states per behavior |

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

### Experiment 6: Single-Code Sustain Grid

```bash
python -m experiments.run_single_code

# Override sustain duration:
python -m experiments.run_single_code sustain_frames=200
```

For each of the 50 KPMS codes (ordered most→least popular), holds the code
constant for K frames (default 100) with two body instantiations (low-z and
high-z starting pose). Outputs a 5×10 grid video showing all codes.

**Outputs** (`outputs/moseq_single_code/`):
- `code_frequency.png` — code distribution histogram
- `bout_stats.json` — per-code bout duration statistics
- `single_code_grid.mp4` — 5-column grid video of all codes

### Experiment 7: Behavior Transition Parade

```bash
python -m experiments.run_behavior_parade

# Fewer bodies:
python -m experiments.run_behavior_parade num_bodies=5
```

10 bodies spaced on the x-axis, viewed from a top-down camera. All bodies
receive the same code sequence: walk → groom → rear. Behavior-representative
codes are automatically selected from real data via kinematic criteria
(XY displacement for walk, Z rise for rear, XYZ stillness for groom).

**Outputs** (`outputs/moseq_behavior_parade/`):
- `behavior_parade.mp4` — top-down parade video with behavior labels
- `code_selection.json` — selected codes and kinematic rationale
- `code_selection.png` — kinematic analysis plot with highlighted codes

### Experiment 8: KPMS Syllable 3D Visualization

```bash
python -m experiments.run_syllable_viz

# Skip dendrogram (faster):
python -m experiments.run_syllable_viz dendrogram.enabled=false
```

Visualizes the KPMS syllable decomposition on the original reference data
using keypoint_moseq's built-in 3D visualization tools. **No decoder needed.**

Loads 3D keypoints via forward kinematics from the reference clips, then
feeds them to keypoint_moseq's trajectory plot, frequency, duration, and
dendrogram functions with the existing KPMS model results.

**Prerequisites**: KPMS model checkpoint at
`outputs/kpms_sweep/s50_k1e+04_l10_arhmm/seed1/` (produced by the KPMS
sweep pipeline).

**Outputs** (`outputs/moseq_syllable_viz/`):
- `trajectory_plots/all_trajectories.html` — interactive 3D Plotly visualization
- `trajectory_plots/*.xy.pdf`, `*.xz.pdf` — 2D projection trajectory plots per syllable
- `trajectory_plots/*.xy.gif`, `*.xz.gif` — animated trajectory GIFs per syllable
- `syllable_frequencies.pdf/.png` — syllable frequency histogram
- `duration_distribution.pdf/.png` — syllable duration distribution
- `similarity_dendrogram.pdf/.png` — hierarchical clustering of syllable trajectories
- `summary.json` — experiment metadata

### Experiment 9: Inception Distance (FID/KID)

```bash
# Full run (default: ARHMM L2, 3 VAE seeds)
python -m experiments.run_inception_distance

# Multiple methods
python -m experiments.run_inception_distance \
  inception_distance.methods='["arhmm_level2","uniform_random","decoder_original_codes"]'

# Quick smoke test
python -m experiments.run_inception_distance \
  inception_distance.methods='["uniform_random"]' \
  inception_distance.num_clips=10 \
  inception_distance.vae.num_epochs=5 \
  inception_distance.vae.seeds='[0]'
```

Trains a VAE feature extractor on real mocap qpos, then computes FID and KID
between real motion capture and decoder rollouts driven by generative code
models. Measures how well the full generative pipeline (code model + decoder)
reproduces the distribution of natural mouse behavior.

Adapts the evaluation methodology from SCAMPER (Aidan's prior network
evaluation pipeline). VAE weights are cached after first training; subsequent
runs with the same config skip training and load directly.

**Available methods**: `arhmm_level2`, `arhmm_level1`, `hmm_dynamax`,
`transition_matrix`, `decoder_original_codes` (ceiling), `uniform_random`.

**Outputs** (`outputs/moseq_inception_distance/`):
- `results.json` — per-seed + aggregated FID/KID metrics
- `results.csv` — summary table
- `fid_barplot.png` — FID comparison with split baseline
- `kid_barplot.png` — KID comparison with split baseline
- `rollouts_{method}.npz` — raw rollout qpos per method
- `vae_cache/` — cached VAE weights (reused across runs)

### Experiment 10: Gait Dynamics PSD

```bash
python -m experiments.run_gait_dynamics

# Fewer clips for quick test
python -m experiments.run_gait_dynamics gait_dynamics.k_clips=3

# Looser walking filter
python -m experiments.run_gait_dynamics gait_dynamics.min_xy_speed=0.01
```

Selects K clips that are purely walking (XY speed always above threshold),
runs both Code2Act and Mimic-MJX on the same clips, and compares the power
spectral density (PSD) of hind limb joint angles. If the dominant gait
frequencies match, the decoder preserves the temporal dynamics of locomotion.

**Outputs** (`outputs/moseq_gait_dynamics/`):
- `psd_overlay.{png,pdf}` — per-joint PSD curves: Mimic (solid) vs Code2Act (dashed)
- `dominant_frequencies.{png,pdf}` — grouped bar chart of peak gait frequencies
- `trajectory_overlay.{png,pdf}` — time-domain joint angle overlay for best walking clip
- `summary.json` — per-clip dominant frequencies and metadata

### Experiment 11: RNN Hidden State Dynamics

```bash
python -m experiments.run_hidden_dynamics

# Fewer clips per behavior
python -m experiments.run_hidden_dynamics K=5

# Longer rollouts
python -m experiments.run_hidden_dynamics max_steps=500
```

Records the GRU hidden state at every timestep for K clips per behavior
category (walk, groom, rear). The resulting hidden state trajectories are
then visualized with PCA, t-SNE, and UMAP to show that the RNN learns
distinct dynamical regimes for different behavioral syllables.

The experiment selects clips via balanced splits, feeds each clip's own
KPMS code sequence, and collects the last GRU layer's hidden state
(256-dim) at each of the `max_steps` timesteps. All trajectories are
uniform length (the rollout does not break on done).

**Requires**: RNN decoder checkpoint (`use_rnn_decoder=true`).

**Outputs** (`outputs/moseq_hidden_dynamics/`):
- `data/hidden_dynamics.npz` — per-behavior hidden states `[K, T, 256]`, codes, survivals
- Also copied to `figures/data/hidden_dynamics.npz` for the figure script

**Figure script 1 — 3D PCA** (`figures/plot_hidden_dynamics.py`):

```bash
cd moseq_jax/figures
python plot_hidden_dynamics.py
```

Produces 2 figures (3 panels each: Walk | Groom | Rear, shared PCA axes):
- `hidden_scatter_pca3d.{pdf,png,svg}` — 3D point cloud per behavior
- `hidden_trajectories_pca3d.{pdf,png,svg}` — 3D connected paths colored by time (light=early, dark=late)

PCA is fitted on all behaviors jointly so panels share the same coordinate
system. Each panel shows one behavior to reveal which regions it occupies.

**Figure script 2 — DSA** (`figures/plot_hidden_dsa.py`):

```bash
cd moseq_jax/figures
python plot_hidden_dsa.py                 # compute + plot
python plot_hidden_dsa.py --replot        # replot from cached distances
python plot_hidden_dsa.py --n_delays 10 --rank 15   # custom DSA params
```

Dynamical Similarity Analysis (Ostrow et al. 2023): treats each clip as a
separate dynamical system (30 total), fits delay-embedded DMD, and compares
pairwise. Shows that clips with the same behavior code have similar dynamics
(low within-behavior distance) while clips with different codes have
dissimilar dynamics (high between-behavior distance).

Uses the `dsa-metric` package if installed (`pip install dsa-metric`),
otherwise falls back to a lightweight eigenvalue-spectrum comparison.
Adapted from `mech-complexity/dsa.py` (Charles Xu).

Produces 2 figures:
- `dsa_heatmap.{pdf,png,svg}` — 30×30 pairwise distance heatmap (block-diagonal structure)
- `dsa_within_between.{pdf,png,svg}` — within-behavior vs between-behavior distance bar chart

Cached distance matrix saved to `figures/data/dsa_distances.npz` for fast replotting.

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
