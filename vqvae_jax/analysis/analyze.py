#!/usr/bin/env python
"""Unified VQ-VAE Analysis Pipeline.

Central entry point for all VQ-VAE analysis tasks:
- Building transition matrices from clip inference
- Rendering clips with code transition bars (Nature paper style)
- Generating transition graphs and matrices
- Creating codebook visualizations

Usage:
    # Run full analysis (all enabled modules)
    python -m analysis.analyze

    # Run specific analysis modes
    python -m analysis.analyze --mode transitions   # Build transition matrix
    python -m analysis.analyze --mode render        # Render clips
    python -m analysis.analyze --mode visualize     # Create visualizations
    python -m analysis.analyze --mode all           # Run everything

    # Override config
    python -m analysis.analyze --checkpoint /path/to/ckpt
    python -m analysis.analyze --config custom_config.yaml
"""

from __future__ import annotations

import os

# Must set rendering backend BEFORE importing MuJoCo or JAX
os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM", "egl")

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
import numpy as np
import yaml
from ml_collections import config_dict
from omegaconf import DictConfig, OmegaConf


# =============================================================================
# CONFIGURATION
# =============================================================================


def load_config(config_path: str | Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_default_config_path() -> Path:
    """Get default config path."""
    return Path(__file__).parent.parent / "configs" / "analysis_config.yaml"


def setup_logging(log_level: str, log_file: Path | None = None) -> None:
    """Configure logging with console and optional file output."""
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


# =============================================================================
# ENVIRONMENT CREATION
# =============================================================================


def create_environment(cfg: DictConfig) -> Any:
    """Create VNL imitation environment from config."""
    from vnl_playground.tasks.rodent import imitation
    from vnl_playground.tasks.rodent import wrappers as vnl_wrappers

    env_cfg = cfg.env_config
    env_cfg_ml = config_dict.ConfigDict(OmegaConf.to_container(env_cfg, resolve=True))
    return vnl_wrappers.FlattenObsWrapper(imitation.Imitation(config=env_cfg_ml))


# =============================================================================
# TRANSITION MATRIX BUILDING
# =============================================================================


def build_transition_matrix(
    config: dict,
    env: Any = None,
    inference_fn: Any = None,
    num_codes: int = 64,
) -> dict[str, Any]:
    """Build transition matrix by running inference on multiple clips.

    This runs the trained policy on reference clips and records code indices,
    then computes transition counts and probabilities.

    Args:
        config: Full configuration dict.
        env: Pre-created environment (optional).
        inference_fn: Pre-created inference function (optional).
        num_codes: Number of codes in codebook.

    Returns:
        Dictionary with transition data and clip results.
    """
    from analysis.checkpoint_utils import load_vq_checkpoint, load_vq_inference_fn

    trans_cfg = config["transitions"]
    output_dir = Path(config["output"]["base_dir"]) / trans_cfg["output_subdir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint if needed
    if env is None or inference_fn is None:
        logging.info(f"Loading checkpoint from {config['checkpoint']['path']}")
        ckpt = load_vq_checkpoint(
            config["checkpoint"]["path"],
            step=config["checkpoint"]["step"],
        )
        cfg = ckpt["cfg"]
        policy_params = ckpt["policy"]
        num_codes = cfg.network_config.num_codes

        if inference_fn is None:
            inference_fn = load_vq_inference_fn(cfg, policy_params, deterministic=True)
        if env is None:
            env = create_environment(cfg)

    num_clips = trans_cfg["num_clips"]
    seed = trans_cfg["seed"]

    logging.info("=" * 60)
    logging.info(f"Building Transition Matrix from {num_clips} clips")
    logging.info("=" * 60)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    all_indices = []
    all_rewards = []
    clip_results = {}

    for clip_idx in range(num_clips):
        if clip_idx % 10 == 0:
            logging.info(f"  Processing clip {clip_idx}/{num_clips}")

        rng = jax.random.PRNGKey(seed + clip_idx)
        reset_rng, step_rng = jax.random.split(rng)

        try:
            state = jit_reset(reset_rng)
        except Exception as e:
            logging.warning(f"  Failed to reset clip {clip_idx}: {e}")
            continue

        clip_indices = []
        clip_rewards = []
        states = [state]

        max_steps = 1000
        for _ in range(max_steps):
            step_rng, action_rng = jax.random.split(step_rng)
            action, extras = inference_fn(state.obs, action_rng)

            idx = int(extras["indices"])
            clip_indices.append(idx)

            state = jit_step(state, action)
            states.append(state)
            clip_rewards.append(float(state.reward))

            if state.done:
                break

        all_indices.extend(clip_indices)
        all_rewards.extend(clip_rewards)

        clip_results[clip_idx] = {
            "num_steps": len(clip_indices),
            "indices": clip_indices,
            "states": states,
            "mean_reward": float(np.mean(clip_rewards)) if clip_rewards else 0.0,
        }

    logging.info(f"Collected {len(all_indices)} total timesteps from {len(clip_results)} clips")

    # Build transition matrix
    trans_counts = np.zeros((num_codes, num_codes), dtype=np.int32)
    for i in range(len(all_indices) - 1):
        trans_counts[all_indices[i], all_indices[i + 1]] += 1

    # Normalize to probabilities
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    trans_probs = np.where(
        row_sums > 0,
        trans_counts / (row_sums + 1e-10),
        np.ones((num_codes, num_codes)) / num_codes,
    )

    # Compute metrics
    histogram = np.bincount(all_indices, minlength=num_codes)
    codes_used = int(np.sum(histogram > 0))
    utilization = codes_used / num_codes

    probs = histogram / (histogram.sum() + 1e-10)
    entropy = -np.sum(np.where(probs > 0, probs * np.log(probs + 1e-10), 0))
    perplexity = float(np.exp(entropy))

    # Save outputs
    np.save(output_dir / "transition_probs.npy", trans_probs)
    np.save(output_dir / "transition_counts.npy", trans_counts)
    np.save(output_dir / "usage_histogram.npy", histogram)

    metrics = {
        "num_clips": len(clip_results),
        "total_steps": len(all_indices),
        "unique_codes": codes_used,
        "utilization": utilization,
        "perplexity": perplexity,
        "mean_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
    }

    with open(output_dir / "transition_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logging.info(f"  Unique codes: {codes_used}/{num_codes} ({utilization:.1%})")
    logging.info(f"  Perplexity: {perplexity:.2f}")
    logging.info(f"  Saved to {output_dir}")

    return {
        "trans_probs": trans_probs,
        "trans_counts": trans_counts,
        "histogram": histogram,
        "metrics": metrics,
        "clip_results": clip_results,
        "output_dir": str(output_dir),
        "env": env,
        "num_codes": num_codes,
    }


# =============================================================================
# CLIP RENDERING
# =============================================================================


def render_transition_clips(
    config: dict,
    clip_results: dict[int, dict],
    env: Any,
    num_codes: int,
) -> dict[str, Any]:
    """Render all clips used for transition matrix with code bars.

    Args:
        config: Full configuration dict.
        clip_results: Dict mapping clip_idx to rollout data.
        env: Environment for rendering.
        num_codes: Number of codes in codebook.

    Returns:
        Dictionary with paths to rendered videos.
    """
    from analysis.rendering import render_clips_grid, render_rollout_to_video

    render_cfg = config["render"]
    output_dir = Path(config["output"]["base_dir"]) / render_cfg["output_subdir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 60)
    logging.info(f"Rendering {len(clip_results)} clips")
    logging.info("=" * 60)

    # Prepare clip data for grid rendering
    clip_data = []
    for clip_idx in sorted(clip_results.keys()):
        result = clip_results[clip_idx]
        clip_data.append({
            "states": result["states"],
            "indices": np.array(result["indices"]),
            "clip_idx": clip_idx,
        })

    paths = {"individual": [], "grids": []}

    # Render individual clips if requested
    if not render_cfg.get("render_all_transition_clips", True):
        # Only render a subset
        subset = clip_data[:min(10, len(clip_data))]
        for clip in subset:
            video_path = output_dir / f"clip_{clip['clip_idx']}.mp4"
            render_rollout_to_video(
                env=env,
                rollout_states=clip["states"],
                output_path=video_path,
                camera=render_cfg["camera"],
                width=render_cfg["cell_width"],
                height=render_cfg["cell_height"],
                fps=render_cfg["fps"],
                indices=clip["indices"],
                num_codes=num_codes,
                clip_idx=clip["clip_idx"],
                code_bar_height=render_cfg["code_bar"]["height"],
            )
            paths["individual"].append(str(video_path))
    else:
        # Render all clips
        for clip in clip_data:
            video_path = output_dir / f"clip_{clip['clip_idx']}.mp4"
            render_rollout_to_video(
                env=env,
                rollout_states=clip["states"],
                output_path=video_path,
                camera=render_cfg["camera"],
                width=render_cfg["cell_width"],
                height=render_cfg["cell_height"],
                fps=render_cfg["fps"],
                indices=clip["indices"],
                num_codes=num_codes,
                clip_idx=clip["clip_idx"],
                code_bar_height=render_cfg["code_bar"]["height"],
            )
            paths["individual"].append(str(video_path))

    # Create grid montages (Nature paper style)
    if render_cfg.get("create_grid", True) and len(clip_data) > 1:
        logging.info("Creating grid montages...")
        grid_paths = render_clips_grid(
            env=env,
            clip_data=clip_data,
            output_path=output_dir / "clips_grid.mp4",
            max_rows=render_cfg.get("max_grid_rows", 5),
            max_cols=render_cfg.get("max_grid_cols", 5),
            camera=render_cfg["camera"],
            cell_width=render_cfg["cell_width"],
            cell_height=render_cfg["cell_height"],
            fps=render_cfg["fps"],
            num_codes=num_codes,
            code_bar_height=render_cfg["code_bar"]["height"],
            padding=render_cfg.get("grid_padding", 4),
            bg_color=tuple(render_cfg.get("grid_bg_color", [255, 255, 255])),
        )
        paths["grids"] = grid_paths

    logging.info(f"  Rendered {len(paths['individual'])} individual clips")
    logging.info(f"  Created {len(paths['grids'])} grid montages")

    return paths


# =============================================================================
# TRANSITION VISUALIZATION
# =============================================================================


def create_transition_visualizations(
    config: dict,
    trans_probs: np.ndarray,
    trans_counts: np.ndarray,
    histogram: np.ndarray,
) -> dict[str, str]:
    """Create transition matrix and graph visualizations.

    Args:
        config: Full configuration dict.
        trans_probs: Transition probability matrix.
        trans_counts: Transition count matrix.
        histogram: Usage histogram.

    Returns:
        Dictionary mapping visualization names to file paths.
    """
    from analysis.code_sequences import (
        get_bidirectional_pairs,
        get_hub_codes,
        get_likely_chains,
    )
    from analysis.visualization import plot_transition_matrix

    viz_cfg = config["transition_viz"]
    output_dir = Path(config["output"]["base_dir"]) / viz_cfg["output_subdir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    logging.info("=" * 60)
    logging.info("Creating Transition Visualizations")
    logging.info("=" * 60)

    # Transition matrix heatmaps
    if viz_cfg["matrix"]["enabled"]:
        if viz_cfg["matrix"]["show_counts"]:
            paths["transition_counts"] = plot_transition_matrix(
                trans_counts,
                output_dir / "transition_counts.png",
                title="Code Transition Counts",
                figsize=tuple(viz_cfg["matrix"]["figsize"]),
                cmap=viz_cfg["matrix"]["cmap"],
                log_scale=viz_cfg["matrix"]["log_scale"],
            )
            logging.info(f"  Saved transition counts: {paths['transition_counts']}")

        if viz_cfg["matrix"]["show_probs"]:
            paths["transition_probs"] = plot_transition_matrix(
                trans_probs,
                output_dir / "transition_probs.png",
                title="Code Transition Probabilities",
                figsize=tuple(viz_cfg["matrix"]["figsize"]),
                cmap=viz_cfg["matrix"]["cmap"],
                log_scale=False,
            )
            logging.info(f"  Saved transition probs: {paths['transition_probs']}")

    # Transition graph (network visualization)
    if viz_cfg["graph"]["enabled"]:
        graph_path = _create_transition_graph(
            trans_probs=trans_probs,
            trans_counts=trans_counts,
            histogram=histogram,
            output_path=output_dir / "transition_graph.png",
            config=viz_cfg["graph"],
        )
        paths["transition_graph"] = graph_path
        logging.info(f"  Saved transition graph: {graph_path}")

    # Print summary statistics
    if viz_cfg["summary"]["enabled"]:
        _print_transition_summary(
            trans_probs=trans_probs,
            trans_counts=trans_counts,
            show_bidirectional=viz_cfg["summary"]["show_bidirectional"],
            show_chains=viz_cfg["summary"]["show_chains"],
        )

    return paths


def _create_transition_graph(
    trans_probs: np.ndarray,
    trans_counts: np.ndarray,
    histogram: np.ndarray,
    output_path: Path,
    config: dict,
) -> str:
    """Create network graph visualization of transitions.

    Args:
        trans_probs: Transition probability matrix.
        trans_counts: Transition count matrix.
        histogram: Usage histogram.
        output_path: Path to save figure.
        config: Graph configuration.

    Returns:
        Path to saved figure.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    try:
        import networkx as nx
    except ImportError:
        logging.warning("networkx not installed, skipping graph visualization")
        return ""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_codes = trans_probs.shape[0]
    min_weight = config.get("min_edge_weight", 0.01)

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes for codes with transitions
    active_codes = set()
    for i in range(num_codes):
        if histogram[i] > 0:
            active_codes.add(i)

    for code in active_codes:
        G.add_node(code, weight=histogram[code])

    # Add edges
    max_prob = trans_probs.max()
    for i in active_codes:
        for j in active_codes:
            if trans_probs[i, j] > min_weight * max_prob:
                G.add_edge(i, j, weight=trans_probs[i, j])

    if len(G.nodes()) == 0:
        logging.warning("No active codes found for graph visualization")
        return ""

    # Create figure
    fig, ax = plt.subplots(figsize=tuple(config.get("figsize", [14, 14])))

    # Layout
    layout_name = config.get("layout", "spring")
    if layout_name == "spring":
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif layout_name == "circular":
        pos = nx.circular_layout(G)
    elif layout_name == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Node sizes based on usage
    node_sizes = [G.nodes[n]["weight"] * config.get("node_size_scale", 100)
                  for n in G.nodes()]
    node_sizes = [max(s, 100) for s in node_sizes]  # Minimum size

    # Edge widths based on probability
    edge_widths = [G.edges[e]["weight"] * config.get("edge_width_scale", 3.0)
                   for e in G.edges()]

    # Draw
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=list(G.nodes()),
        cmap=plt.cm.viridis,
        alpha=0.8,
    )
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths,
        alpha=0.5,
        edge_color="gray",
        arrows=True,
        arrowsize=10,
        connectionstyle="arc3,rad=0.1",
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=8,
        font_weight="bold",
    )

    ax.set_title(f"Code Transition Graph ({len(G.nodes())} codes, {len(G.edges())} edges)")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(output_path)


def _print_transition_summary(
    trans_probs: np.ndarray,
    trans_counts: np.ndarray,
    show_bidirectional: bool = True,
    show_chains: bool = True,
) -> None:
    """Print a summary of transition statistics."""
    from analysis.code_sequences import (
        get_bidirectional_pairs,
        get_hub_codes,
        get_likely_chains,
    )

    logging.info("")
    logging.info("=" * 60)
    logging.info("TRANSITION SUMMARY")
    logging.info("=" * 60)

    total_trans = int(trans_counts.sum())
    codes_with_trans = int((trans_counts.sum(axis=1) > 0).sum())
    logging.info(f"Total transitions: {total_trans}")
    logging.info(f"Codes with outgoing transitions: {codes_with_trans}")

    logging.info("\n--- Hub Codes (most connections) ---")
    hubs = get_hub_codes(trans_counts, top_k=5)
    logging.info("Code   In  Out Total")
    for code, in_deg, out_deg, total in hubs:
        logging.info(f"  {code:3d}   {in_deg:2d}   {out_deg:2d}   {total:3d}")

    if show_bidirectional:
        logging.info("\n--- Bidirectional Pairs (A<->B) ---")
        pairs = get_bidirectional_pairs(trans_counts, min_count=3)[:5]
        for a, b, ab, ba, total in pairs:
            logging.info(f"  Code {a:2d} <-> {b:2d}: {ab} + {ba} = {total}")

    if show_chains:
        logging.info("\n--- Top Transition Chains (A->B->C) ---")
        chains = get_likely_chains(trans_probs, trans_counts)[:5]
        for a, b, c, pab, pbc, pchain in chains:
            logging.info(f"  {a:2d} -> {b:2d} -> {c:2d}: {pchain:.1%}")


# =============================================================================
# CODEBOOK VISUALIZATION
# =============================================================================


def create_codebook_visualizations(
    config: dict,
    codebook: np.ndarray,
    histogram: np.ndarray | None = None,
) -> dict[str, str]:
    """Create codebook visualizations.

    Args:
        config: Full configuration dict.
        codebook: Codebook embeddings [num_codes, latent_dim].
        histogram: Optional usage histogram.

    Returns:
        Dictionary mapping visualization names to file paths.
    """
    from analysis.visualization import (
        plot_code_histogram,
        plot_codebook_2d,
        plot_codebook_with_usage,
        project_codebook_2d,
    )

    viz_cfg = config["visualization"]
    output_dir = Path(config["output"]["base_dir"]) / viz_cfg["output_subdir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    logging.info("=" * 60)
    logging.info("Creating Codebook Visualizations")
    logging.info("=" * 60)

    # Project to 2D
    proj_cfg = viz_cfg["projection"]
    codebook_2d = project_codebook_2d(codebook, method=proj_cfg["method"])

    # Basic codebook plot
    paths["codebook_2d"] = plot_codebook_2d(
        codebook_2d,
        output_dir / "codebook_2d.png",
        title=f"Codebook ({len(codebook)} codes, {proj_cfg['method'].upper()})",
        figsize=tuple(proj_cfg["figsize"]),
        show_labels=proj_cfg["show_labels"],
        point_size=proj_cfg["point_size"],
    )
    logging.info(f"  Saved codebook 2D: {paths['codebook_2d']}")

    # Usage histogram
    if histogram is not None and viz_cfg["histogram"]["enabled"]:
        hist_cfg = viz_cfg["histogram"]
        paths["usage_histogram"] = plot_code_histogram(
            histogram,
            output_dir / "usage_histogram.png",
            figsize=tuple(hist_cfg["figsize"]),
            highlight_threshold=hist_cfg.get("highlight_threshold"),
        )
        logging.info(f"  Saved usage histogram: {paths['usage_histogram']}")

    # Codebook with usage
    if histogram is not None and viz_cfg["codebook_usage"]["enabled"]:
        usage_cfg = viz_cfg["codebook_usage"]
        paths["codebook_usage"] = plot_codebook_with_usage(
            codebook_2d,
            histogram,
            output_dir / "codebook_usage.png",
            cmap=usage_cfg["cmap"],
        )
        logging.info(f"  Saved codebook usage: {paths['codebook_usage']}")

    return paths


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def run_analysis_pipeline(
    config: dict,
    mode: str = "all",
) -> dict[str, Any]:
    """Run the unified analysis pipeline.

    Args:
        config: Full configuration dict.
        mode: Analysis mode ("all", "transitions", "render", "visualize").

    Returns:
        Dictionary with all analysis results.
    """
    from analysis.checkpoint_utils import get_codebook, load_vq_checkpoint

    results = {}

    # Load checkpoint once for all analyses
    logging.info(f"Loading checkpoint from {config['checkpoint']['path']}")
    ckpt = load_vq_checkpoint(
        config["checkpoint"]["path"],
        step=config["checkpoint"]["step"],
    )
    cfg = ckpt["cfg"]
    policy_params = ckpt["policy"]
    loaded_step = ckpt["step"]

    num_codes = cfg.network_config.num_codes
    codebook = np.array(get_codebook(policy_params))

    results["checkpoint_step"] = loaded_step
    results["num_codes"] = num_codes

    # Create environment and inference function
    env = create_environment(cfg)

    from analysis.checkpoint_utils import load_vq_inference_fn
    inference_fn = load_vq_inference_fn(cfg, policy_params, deterministic=True)

    # Stage 1: Build transition matrix
    if mode in ["all", "transitions"] and config["transitions"]["enabled"]:
        trans_results = build_transition_matrix(
            config=config,
            env=env,
            inference_fn=inference_fn,
            num_codes=num_codes,
        )
        results["transitions"] = trans_results

        # Stage 2: Render clips (uses transition results)
        if mode in ["all", "render"] and config["render"]["enabled"]:
            render_results = render_transition_clips(
                config=config,
                clip_results=trans_results["clip_results"],
                env=env,
                num_codes=num_codes,
            )
            results["renders"] = render_results

        # Stage 3: Create transition visualizations
        if mode in ["all", "visualize"] and config["transition_viz"]["enabled"]:
            trans_viz_results = create_transition_visualizations(
                config=config,
                trans_probs=trans_results["trans_probs"],
                trans_counts=trans_results["trans_counts"],
                histogram=trans_results["histogram"],
            )
            results["transition_viz"] = trans_viz_results

    # Stage 4: Codebook visualizations (can run independently)
    if mode in ["all", "visualize"] and config["visualization"]["enabled"]:
        histogram = None
        if "transitions" in results:
            histogram = results["transitions"]["histogram"]

        cb_viz_results = create_codebook_visualizations(
            config=config,
            codebook=codebook,
            histogram=histogram,
        )
        results["codebook_viz"] = cb_viz_results

    return results


# =============================================================================
# CLI
# =============================================================================


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="VQ-VAE Unified Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full analysis
    python -m analysis.analyze

    # Run specific modes
    python -m analysis.analyze --mode transitions
    python -m analysis.analyze --mode render
    python -m analysis.analyze --mode visualize

    # Override config
    python -m analysis.analyze --checkpoint /path/to/checkpoint
    python -m analysis.analyze --num-clips 50
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(get_default_config_path()),
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override checkpoint path from config",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Override checkpoint step (None = latest)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "transitions", "render", "visualize"],
        default="all",
        help="Which analysis to run",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--num-clips",
        type=int,
        default=None,
        help="Override number of clips for transition building",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip rendering even if enabled in config",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Apply command-line overrides
    if args.checkpoint:
        config["checkpoint"]["path"] = args.checkpoint
    if args.step is not None:
        config["checkpoint"]["step"] = args.step
    if args.output_dir:
        config["output"]["base_dir"] = args.output_dir
    if args.num_clips:
        config["transitions"]["num_clips"] = args.num_clips
    if args.no_render:
        config["render"]["enabled"] = False

    # Setup logging
    output_dir = Path(config["output"]["base_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = None
    if config["logging"]["log_to_file"]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = output_dir / f"{config['logging']['log_filename']}_{timestamp}"

    setup_logging(config["logging"]["level"], log_file)

    logging.info("=" * 60)
    logging.info("VQ-VAE Unified Analysis Pipeline")
    logging.info("=" * 60)
    logging.info(f"Checkpoint: {config['checkpoint']['path']}")
    logging.info(f"Output: {config['output']['base_dir']}")
    logging.info(f"Mode: {args.mode}")

    try:
        results = run_analysis_pipeline(config=config, mode=args.mode)

        logging.info("")
        logging.info("=" * 60)
        logging.info("Analysis Complete")
        logging.info("=" * 60)

        if "transitions" in results:
            metrics = results["transitions"]["metrics"]
            logging.info(f"  Codes used: {metrics['unique_codes']}/{results['num_codes']}")
            logging.info(f"  Perplexity: {metrics['perplexity']:.2f}")

        if "renders" in results:
            logging.info(f"  Individual videos: {len(results['renders']['individual'])}")
            logging.info(f"  Grid montages: {len(results['renders']['grids'])}")

        return results

    except Exception as e:
        logging.exception(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
