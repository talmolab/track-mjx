"""VQ-VAE Analysis Pipeline for Motion Imitation Models.

This package provides a unified pipeline for analyzing trained VQ-VAE models:

- **code_analysis.py**: Main entry point for code semantic analysis
  - Transition matrix and chain analysis
  - Code segment extraction and duration statistics
  - Kinematic feature correlation
  - DTW-based segment alignment

- **inference_cache.py**: Caching utilities for inference results
  - InferenceResult dataclass
  - Cache key computation and NPZ storage

- **transition_analysis.py**: Transition matrix analysis
  - Build transition matrices from code sequences
  - Find transition chains (A->B->C patterns)
  - Classify code roles (entry, exit, hub, steady-state)
  - Visualize transition graphs

- **segment_analysis.py**: Code segment analysis
  - Extract contiguous segments per code
  - Compute duration statistics
  - Render per-code segment grid videos

- **kinematic_analysis.py**: Kinematic feature analysis
  - Extract features (velocity, body height, etc.)
  - Compute per-code kinematic profiles
  - Plot heatmaps and clusters

- **alignment_analysis.py**: DTW alignment analysis
  - Compute DTW distance between segments
  - Align segments to longest reference
  - Render aligned comparison videos

- **noise_analysis.py**: Prior noise injection analysis
  - Test noise effects on code switching
  - Analyze behavioral diversity

- **checkpoint_utils.py**: Checkpoint loading utilities
- **rendering.py**: Video rendering with overlays
- **visualization.py**: Matplotlib plotting utilities

Usage:
    # Run code analysis pipeline
    cd vqvae_jax
    python -m analysis.code_analysis

    # Run noise analysis (distillation experiments)
    python -m analysis.noise_analysis

    # Import modules directly
    from analysis.checkpoint_utils import load_vq_checkpoint
    from analysis.code_analysis import run_or_load_inference
    from analysis.transition_analysis import compute_transition_matrix
"""

__all__ = [
    "alignment_analysis",
    "checkpoint_utils",
    "code_analysis",
    "inference_cache",
    "kinematic_analysis",
    "noise_analysis",
    "rendering",
    "segment_analysis",
    "transition_analysis",
    "visualization",
]
