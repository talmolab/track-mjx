"""VQ-VAE Analysis Pipeline for Motion Imitation Models.

This package provides a unified pipeline for analyzing trained VQ-VAE models:

- **analyze.py**: Central entry point for all analysis tasks
  - Build transition matrices from clip inference
  - Render clips with code transition bars (Nature paper style)
  - Generate transition graphs and matrices
  - Create codebook visualizations

- **random_walk.py**: Random walk motion generation
  - Random walk on learned transition probabilities
  - Free-running decoder execution with proprioceptive feedback

Usage:
    # Run full analysis pipeline
    cd scratch/vqvae_jax
    python -m analysis.analyze

    # Run specific analysis modes
    python -m analysis.analyze --mode transitions
    python -m analysis.analyze --mode render
    python -m analysis.analyze --mode visualize

    # Random walk generation (requires transition matrix from analysis)
    python -m analysis.random_walk generate

    # Import modules directly
    from analysis.checkpoint_utils import load_vq_checkpoint
    from analysis.analyze import run_analysis_pipeline
    from analysis.random_walk import run_random_walk_generation
"""

__all__ = [
    "analyze",
    "checkpoint_utils",
    "code_sequences",
    "random_walk",
    "rendering",
    "visualization",
]
