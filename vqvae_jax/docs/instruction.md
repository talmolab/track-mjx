# VQ-VAE Motion Imitation - Instructions

Complete guide for training, analysis, and generation with VQ-VAE motion imitation models.

---

## Table of Contents

1. [Directory Structure](#directory-structure)
2. [Environment Setup](#environment-setup)
3. [Training](#training)
4. [Analysis](#analysis)
5. [Random Walk Generation](#random-walk-generation)
6. [Rendering Options](#rendering-options)
7. [Troubleshooting](#troubleshooting)

---

## Directory Structure

```
vqvae_jax/
├── configs/                        # Configuration files
│   ├── vqvae_minimal.yaml         # Training config
│   ├── analysis_config.yaml       # Unified analysis config
│   └── random_walk_config.yaml      # Generation config
├── analysis/                       # Analysis pipeline
│   ├── analyze.py                 # Central analysis entry point
│   ├── random_walk.py             # Random walk generation
│   ├── checkpoint_utils.py        # Checkpoint loading utilities
│   ├── code_sequences.py          # Transition analysis utilities
│   ├── rendering.py               # Video rendering (Nature style)
│   └── visualization.py           # Static visualizations
├── checkpoints/                    # Saved model checkpoints
└── outputs/                        # Analysis and generation outputs
    ├── analysis/                   # Analysis outputs
    │   ├── transitions/           # Transition matrices
    │   ├── renders/               # Rendered videos
    │   ├── transition_graphs/     # Transition visualizations
    │   └── visualizations/        # Codebook visualizations
    └── generation/                 # Generation outputs
```

---

## Environment Setup

### Required Environment Variables

```bash
# Set rendering backend (required for headless rendering)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

### Working Directory

All commands should be run from the `vqvae_jax` directory:

```bash
cd /home/jovyan/vast/kaiwen/track-mjx/vqvae_jax
```

---

## Training

### Basic Training

```bash
# Train with default config
python train_vqvae.py --config-name vqvae_minimal
```

### Training with Parameter Overrides

```bash
# Override specific parameters
python train_vqvae.py --config-name vqvae_minimal \
    train_setup.train_config.num_envs=8192 \
    network_config.num_codes=128 \
    network_config.latent_dim=64
```

### Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_codes` | 64 | Codebook size (number of motor primitives) |
| `latent_dim` | 32 | Dimension of latent embeddings |
| `commitment_cost` | 0.1 | Beta for commitment loss |
| `num_envs` | 4096 | Number of parallel environments |
| `learning_rate` | 1e-4 | Learning rate |

---

## Analysis

The unified analysis pipeline handles:
- Building transition matrices from clip inference
- Rendering clips with code transition bars (Nature paper style)
- Creating transition graphs and matrix visualizations
- Generating codebook visualizations

### Running Full Analysis

```bash
# Run all analysis modules (transitions + rendering + visualizations)
python -m analysis.analyze

# With custom config
python -m analysis.analyze --config configs/analysis_config.yaml
```

### Running Specific Analysis Modes

```bash
# Build transition matrix only
python -m analysis.analyze --mode transitions

# Render clips only (requires transitions to be built first)
python -m analysis.analyze --mode render

# Create visualizations only
python -m analysis.analyze --mode visualize
```

### Analysis Command-Line Options

```bash
python -m analysis.analyze \
    --checkpoint /path/to/checkpoint \  # Override checkpoint path
    --step 1000000 \                    # Specific checkpoint step
    --num-clips 50 \                    # Number of clips for transitions
    --output-dir outputs/custom \       # Custom output directory
    --no-render                         # Skip video rendering
```

### Analysis Outputs

After running full analysis:

```
outputs/analysis/
├── transitions/
│   ├── transition_probs.npy      # Probability matrix [num_codes, num_codes]
│   ├── transition_counts.npy     # Count matrix
│   ├── usage_histogram.npy       # Code usage counts
│   └── transition_metrics.json   # Summary statistics
├── renders/
│   ├── clip_0.mp4                # Individual clip videos
│   ├── clip_1.mp4
│   ├── ...
│   └── clips_grid.mp4            # Grid montage (5x5 max)
├── transition_graphs/
│   ├── transition_counts.png     # Transition count heatmap
│   ├── transition_probs.png      # Probability heatmap
│   └── transition_graph.png      # Network graph visualization
└── visualizations/
    ├── codebook_2d.png           # PCA/UMAP projection
    ├── usage_histogram.png       # Code usage histogram
    └── codebook_usage.png        # Codebook colored by usage
```

---

## Random Walk Generation

Random walk generation creates novel motion by:
1. Random walk on learned transition probabilities to sample code sequence
2. Free-running decoder execution with real proprioceptive feedback

### Prerequisites

Run analysis first to build transition matrix:

```bash
python -m analysis.analyze --mode transitions
```

### Running Generation

```bash
# Basic generation with default settings
python -m analysis.random_walk generate

# With custom temperature (lower = more deterministic)
python -m analysis.random_walk generate --temperature 0.5

# With different strategy
python -m analysis.random_walk generate --strategy greedy
python -m analysis.random_walk generate --strategy nucleus

# Longer generation horizon
python -m analysis.random_walk generate --horizon 1000

# Skip rendering
python -m analysis.random_walk generate --no-render
```

### Comparing Strategies

```bash
# Run comparison across different sampling strategies
python -m analysis.random_walk compare

# With more trials
python -m analysis.random_walk compare --num-trials 20

# Shorter horizon for faster comparison
python -m analysis.random_walk compare --horizon 200
```

### Sampling Strategies

| Strategy | Description |
|----------|-------------|
| `temperature` | Scale probabilities by 1/T. T=1 balanced, T<1 deterministic, T>1 random |
| `nucleus` | Sample from smallest set with cumulative probability >= top_p |
| `greedy` | Always pick highest probability transition |

### Generation Outputs

```
outputs/generation/
├── generation_metrics.json       # Full generation metrics
├── random_walk_generation.mp4   # Rendered video with code bars
└── comparison/
    └── comparison_results.json  # Strategy comparison results
```

---

## Rendering Options

### Camera Names

| Camera | Description |
|--------|-------------|
| `close_profile-ghost` | Side view with ghost reference |
| `close_profile-rodent` | Side view of agent only |
| `back-ghost` | Rear view with ghost |
| `top-ghost` | Top-down view with ghost |
| `egocentric-rodent` | First-person view |

### Video Settings

Override in config or via rendering functions:

```yaml
render:
  camera: close_profile-ghost
  cell_width: 320
  cell_height: 240
  fps: 50
  code_bar:
    height: 40
    show_playhead: true
    show_code_label: true
```

### Grid Rendering (Nature Paper Style)

The analysis pipeline creates grid montages with:
- Maximum 5x5 clips per video
- Code transition bars beneath each clip
- Professional white background with padding
- Clip labels in corner

---