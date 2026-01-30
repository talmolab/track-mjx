"""VQ-VAE Analysis Pipeline for Motion Imitation Models.

This package provides per-clip analysis of trained VQ-VAE models:

- **code_analysis.py**: Main entry point for per-clip analysis pipeline
  - Runs analysis on individual clips
  - Generates interactive HTML viewer
  - Renders videos with code/community overlays

- **per_clip_analysis.py**: Core per-clip analysis module
  - Per-clip transition matrices and probability heatmaps
  - Per-clip community detection via spectral clustering
  - Per-clip transition graphs with community-colored nodes
  - Dual timeline visualization (code + community colors)
  - Video rendering with code/community bars

- **inference_cache.py**: Data structures for inference results
- **checkpoint_utils.py**: Checkpoint loading utilities
- **rendering.py**: Video rendering utilities and colormaps

Usage:
    # First, generate rollout data using the inference module
    python -m inference.run_inference checkpoint.path=/path/to/checkpoint

    # Then run per-clip analysis pipeline on the H5 data
    cd vqvae_jax
    python -m analysis.code_analysis

    # Import modules directly
    from analysis.checkpoint_utils import load_vq_checkpoint
    from analysis.inference_cache import InferenceResult
    from analysis.per_clip_analysis import run_per_clip_analysis
"""

__all__ = [
    "checkpoint_utils",
    "code_analysis",
    "inference_cache",
    "per_clip_analysis",
    "rendering",
]
