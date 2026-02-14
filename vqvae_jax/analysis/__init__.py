"""VQ-VAE Analysis Pipeline for Motion Imitation Models.

This package provides analysis of trained VQ-VAE models:

- **code_analysis.py**: Main entry point for the analysis pipeline

- **transition_context_analysis.py**: Cross-clip code consistency analysis
  - Compare predecessor/successor patterns for top K codes
  - Measure functional consistency across different clips
  - Render transition videos (predecessor → code → successor)

- **rvq_analysis.py**: Multi-depth RVQ analysis
  - Parent-child heatmap (L0 x L1 joint distribution)
  - Intra-parent diversity (L1 entropy conditioned on L0)
  - Hierarchical transition rates (L1 transitions within L0 segments)

- **correction_analysis.py**: Correction semantics analysis
  - Burst detection (contiguous non-null code runs)
  - Kinematic deltas (joint angle/velocity changes during bursts)
  - Burst statistics (duration, inter-burst intervals, co-occurrence)
  - Correction PCA (latent-space correction vectors)

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
    from analysis.transition_context_analysis import run_transition_context_analysis
"""

__all__ = [
    "checkpoint_utils",
    "code_analysis",
    "compositional_transition_analysis",
    "correction_analysis",
    "inference_cache",
    "rendering",
    "rvq_analysis",
    "transition_context_analysis",
    "tsne_trajectory_analysis",
]
