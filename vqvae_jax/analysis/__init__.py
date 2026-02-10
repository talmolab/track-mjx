"""VQ-VAE Analysis Pipeline for Motion Imitation Models.

This package provides analysis of trained VQ-VAE models:

- **code_analysis.py**: Main entry point for the analysis pipeline

- **per_clip_analysis.py**: Per-clip analysis module
  - Per-clip transition matrices and probability heatmaps
  - Video rendering with code timeline bars

- **transition_context_analysis.py**: Cross-clip code consistency analysis
  - Compare predecessor/successor patterns for top K codes
  - Measure functional consistency across different clips
  - Render transition videos (predecessor → code → successor)

- **mutual_information.py**: MI analysis between codes and kinematic features
  - KSG estimator for mutual information computation
  - Extended feature extraction (limb activities, posture PCA, heading)
  - MI ranking, feature-code heatmap, and code-feature scatter plots

- **compositional_transition_analysis.py**: Compositional code transition analysis
  - Determinism testing (same pose + same codes → similar trajectories?)
  - Compositional decomposition of k-transition sequences
  - Interactive HTML tree viewer with decomposition and W2 scores

- **tsne_trajectory_analysis.py**: t-SNE skill-space trajectory visualization
  - Select high-movement clips by root XYZ displacement
  - Embed k-transition sequences (concatenated codebook vectors) via t-SNE
  - Synchronized HTML viewer: animated t-SNE canvas + video playback

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
    from analysis.per_clip_analysis import run_per_clip_analysis
    from analysis.transition_context_analysis import run_transition_context_analysis
    from analysis.mutual_information import run_mutual_information_analysis
"""

__all__ = [
    "checkpoint_utils",
    "code_analysis",
    "compositional_transition_analysis",
    "inference_cache",
    "mutual_information",
    "per_clip_analysis",
    "rendering",
    "transition_context_analysis",
    "tsne_trajectory_analysis",
]
