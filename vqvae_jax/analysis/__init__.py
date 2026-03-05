"""VQ-VAE Analysis Pipeline for Motion Imitation Models.

This package provides analysis of trained VQ-VAE models:

- **code_analysis.py**: Main entry point for the analysis pipeline

- **transition_context_analysis.py**: Global transition matrix, stationary
  distribution, code popularity metrics, and pose gallery rendering

- **tsne_trajectory_analysis.py**: t-SNE skill-space trajectory visualization
  - Select high-movement clips by root XYZ displacement
  - Embed k-transition sequences (concatenated codebook vectors) via t-SNE
  - Synchronized HTML viewer: animated t-SNE canvas + video playback

- **utils.py**: Shared utilities (identify_null_code, CodeRun, extract_code_runs)
- **inference_cache.py**: Data structures for inference results
- **checkpoint_utils.py**: Checkpoint loading utilities
- **rendering.py**: Video rendering utilities and colormaps

Usage:
    # First, generate rollout data using the inference module
    python -m inference.run_inference checkpoint.path=/path/to/checkpoint

    # Then run analysis pipeline on the H5 data
    cd vqvae_jax
    python -m analysis.code_analysis

    # Import modules directly
    from analysis.checkpoint_utils import load_vq_checkpoint
    from analysis.inference_cache import InferenceResult
"""

__all__ = [
    "checkpoint_utils",
    "code_analysis",
    "inference_cache",
    "rendering",
    "transition_context_analysis",
    "tsne_trajectory_analysis",
    "utils",
]
