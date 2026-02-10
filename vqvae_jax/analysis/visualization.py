"""Visualization utilities for VQ-VAE codebook analysis.

This module provides functions for visualizing codebook embeddings,
code usage patterns, and trajectories through the latent space.
"""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def project_codebook_2d(
    codebook: np.ndarray,
    method: str = "pca",
) -> np.ndarray:
    """Project codebook embeddings to 2D for visualization.

    Args:
        codebook: Codebook embeddings, shape [num_codes, latent_dim].
        method: Projection method ("pca" or "umap").

    Returns:
        2D coordinates, shape [num_codes, 2].
    """
    if method == "pca":
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        return pca.fit_transform(codebook)
    elif method == "umap":
        try:
            import umap

            reducer = umap.UMAP(n_components=2, random_state=42)
            return reducer.fit_transform(codebook)
        except ImportError:
            raise ImportError(
                "UMAP not installed. Install with: pip install umap-learn"
            )
    else:
        raise ValueError(f"Unknown projection method: {method}")


def plot_codebook_2d(
    codebook_2d: np.ndarray,
    output_path: str | Path,
    title: str = "Codebook Embeddings (2D)",
    figsize: tuple[int, int] = (10, 10),
    show_labels: bool = True,
    label_fontsize: int = 8,
    point_size: int = 100,
    cmap: str = "viridis",
) -> str:
    """Plot 2D codebook embeddings with code labels.

    Args:
        codebook_2d: 2D coordinates, shape [num_codes, 2].
        output_path: Path to save figure.
        title: Figure title.
        figsize: Figure size.
        show_labels: Whether to show code index labels.
        label_fontsize: Font size for labels.
        point_size: Size of scatter points.
        cmap: Colormap name.

    Returns:
        Path to saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    num_codes = len(codebook_2d)
    colors = np.arange(num_codes)

    scatter = ax.scatter(
        codebook_2d[:, 0],
        codebook_2d[:, 1],
        c=colors,
        cmap=cmap,
        s=point_size,
        alpha=0.7,
        edgecolors="white",
        linewidth=0.5,
    )

    if show_labels:
        for i, (x, y) in enumerate(codebook_2d):
            ax.annotate(
                str(i),
                (x, y),
                fontsize=label_fontsize,
                ha="center",
                va="center",
            )

    plt.colorbar(scatter, ax=ax, label="Code Index")
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.axis("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(output_path)


def plot_code_histogram(
    histogram: np.ndarray | list,
    output_path: str | Path,
    title: str = "Code Usage Histogram",
    figsize: tuple[int, int] = (12, 4),
    highlight_threshold: int | None = None,
) -> str:
    """Plot histogram of code usage.

    Args:
        histogram: Usage count per code.
        output_path: Path to save figure.
        title: Figure title.
        figsize: Figure size.
        highlight_threshold: Highlight codes used more than this.

    Returns:
        Path to saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    histogram = np.array(histogram)
    num_codes = len(histogram)

    fig, ax = plt.subplots(figsize=figsize)

    colors = ["steelblue"] * num_codes
    if highlight_threshold is not None:
        colors = [
            "orange" if h > highlight_threshold else "steelblue" for h in histogram
        ]

    ax.bar(range(num_codes), histogram, color=colors, alpha=0.8)

    ax.set_xlabel("Code Index")
    ax.set_ylabel("Usage Count")
    ax.set_title(title)
    ax.set_xlim(-0.5, num_codes - 0.5)

    # Add statistics
    used_codes = np.sum(histogram > 0)
    total_uses = np.sum(histogram)
    stats_text = f"Used: {used_codes}/{num_codes} codes ({used_codes/num_codes:.1%})\n"
    stats_text += f"Total: {total_uses} uses"
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(output_path)


def plot_codebook_with_usage(
    codebook_2d: np.ndarray,
    usage_counts: np.ndarray | list,
    output_path: str | Path,
    title: str = "Codebook Usage",
    figsize: tuple[int, int] = (10, 10),
    show_labels: bool = True,
    cmap: str = "YlOrRd",
) -> str:
    """Plot codebook with usage frequency as color intensity.

    Args:
        codebook_2d: 2D coordinates, shape [num_codes, 2].
        usage_counts: Usage count per code.
        output_path: Path to save figure.
        title: Figure title.
        figsize: Figure size.
        show_labels: Whether to show code labels.
        cmap: Colormap for usage intensity.

    Returns:
        Path to saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    usage_counts = np.array(usage_counts)
    num_codes = len(codebook_2d)

    fig, ax = plt.subplots(figsize=figsize)

    # Normalize usage for coloring
    norm = Normalize(vmin=0, vmax=max(usage_counts.max(), 1))

    # Size proportional to usage (but with minimum)
    min_size = 50
    max_size = 300
    sizes = min_size + (usage_counts / (usage_counts.max() + 1)) * (max_size - min_size)

    scatter = ax.scatter(
        codebook_2d[:, 0],
        codebook_2d[:, 1],
        c=usage_counts,
        s=sizes,
        cmap=cmap,
        norm=norm,
        alpha=0.7,
        edgecolors="gray",
        linewidth=0.5,
    )

    if show_labels:
        for i, (x, y) in enumerate(codebook_2d):
            if usage_counts[i] > 0:
                ax.annotate(
                    str(i),
                    (x, y),
                    fontsize=7,
                    ha="center",
                    va="center",
                    alpha=0.8,
                )

    plt.colorbar(scatter, ax=ax, label="Usage Count")
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.axis("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(output_path)


def plot_trajectory_on_codebook(
    codebook_2d: np.ndarray,
    trajectory: list[int] | np.ndarray,
    output_path: str | Path,
    title: str = "Code Trajectory",
    figsize: tuple[int, int] = (10, 10),
    show_background: bool = True,
    line_alpha: float = 0.5,
    line_width: float = 1.0,
) -> str:
    """Plot trajectory through codebook space.

    Args:
        codebook_2d: 2D coordinates, shape [num_codes, 2].
        trajectory: Sequence of code indices.
        output_path: Path to save figure.
        title: Figure title.
        figsize: Figure size.
        show_background: Whether to show all codes in background.
        line_alpha: Transparency of trajectory line.
        line_width: Width of trajectory line.

    Returns:
        Path to saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trajectory = np.array(trajectory)

    fig, ax = plt.subplots(figsize=figsize)

    # Background codes
    if show_background:
        ax.scatter(
            codebook_2d[:, 0],
            codebook_2d[:, 1],
            c="lightgray",
            s=80,
            alpha=0.5,
            edgecolors="gray",
            linewidth=0.5,
        )
        for i, (x, y) in enumerate(codebook_2d):
            ax.annotate(str(i), (x, y), fontsize=6, ha="center", va="center", alpha=0.3)

    # Trajectory
    traj_coords = codebook_2d[trajectory]

    # Draw line
    ax.plot(
        traj_coords[:, 0],
        traj_coords[:, 1],
        "b-",
        alpha=line_alpha,
        linewidth=line_width,
        zorder=2,
    )

    # Color points by time
    colors = np.linspace(0, 1, len(trajectory))
    scatter = ax.scatter(
        traj_coords[:, 0],
        traj_coords[:, 1],
        c=colors,
        cmap="viridis",
        s=30,
        alpha=0.8,
        zorder=3,
    )

    # Mark start and end
    ax.scatter(
        traj_coords[0, 0],
        traj_coords[0, 1],
        c="green",
        s=200,
        marker="^",
        zorder=4,
        label="Start",
    )
    ax.scatter(
        traj_coords[-1, 0],
        traj_coords[-1, 1],
        c="red",
        s=200,
        marker="s",
        zorder=4,
        label="End",
    )

    plt.colorbar(scatter, ax=ax, label="Time")
    ax.legend()
    ax.set_title(f"{title} ({len(trajectory)} steps)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.axis("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(output_path)


def plot_transition_matrix(
    transitions: np.ndarray | list,
    output_path: str | Path,
    title: str = "Code Transition Matrix",
    figsize: tuple[int, int] = (12, 10),
    cmap: str = "Blues",
    log_scale: bool = True,
) -> str:
    """Plot code transition matrix as heatmap.

    Args:
        transitions: Transition counts, shape [num_codes, num_codes].
        output_path: Path to save figure.
        title: Figure title.
        figsize: Figure size.
        cmap: Colormap name.
        log_scale: Whether to use log scale for colors.

    Returns:
        Path to saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    transitions = np.array(transitions)
    num_codes = len(transitions)

    fig, ax = plt.subplots(figsize=figsize)

    if log_scale:
        # Add 1 to avoid log(0)
        plot_data = np.log10(transitions + 1)
        label = "log10(count + 1)"
    else:
        plot_data = transitions
        label = "count"

    im = ax.imshow(plot_data, cmap=cmap, aspect="auto")

    plt.colorbar(im, ax=ax, label=label)
    ax.set_title(title)
    ax.set_xlabel("To Code")
    ax.set_ylabel("From Code")

    # Show fewer ticks for large codebooks
    if num_codes > 20:
        tick_step = num_codes // 10
        ticks = np.arange(0, num_codes, tick_step)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(output_path)


def plot_stability_summary(
    per_code_stability: dict[int, dict[str, Any]],
    output_path: str | Path,
    title: str = "Code Stability Summary",
    figsize: tuple[int, int] = (14, 5),
) -> str:
    """Plot stability metrics across all codes.

    Args:
        per_code_stability: Dict mapping code_idx to stability metrics.
        output_path: Path to save figure.
        title: Figure title.
        figsize: Figure size.

    Returns:
        Path to saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    codes = sorted(per_code_stability.keys())
    survival_steps = [per_code_stability[c]["survival_steps"] for c in codes]
    mean_rewards = [per_code_stability[c]["mean_reward"] for c in codes]
    fallen = [per_code_stability[c]["fallen"] for c in codes]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Survival steps
    colors = ["red" if f else "steelblue" for f in fallen]
    axes[0].bar(codes, survival_steps, color=colors, alpha=0.7)
    axes[0].set_xlabel("Code Index")
    axes[0].set_ylabel("Survival Steps")
    axes[0].set_title("Survival by Code (red=fallen)")

    # Mean reward
    axes[1].bar(codes, mean_rewards, color="steelblue", alpha=0.7)
    axes[1].set_xlabel("Code Index")
    axes[1].set_ylabel("Mean Reward")
    axes[1].set_title("Mean Reward by Code")

    # Survival vs Reward scatter
    axes[2].scatter(survival_steps, mean_rewards, c=fallen, cmap="RdYlGn_r", alpha=0.7)
    axes[2].set_xlabel("Survival Steps")
    axes[2].set_ylabel("Mean Reward")
    axes[2].set_title("Survival vs Reward")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(output_path)


def create_analysis_report(
    output_dir: str | Path,
    codebook: np.ndarray,
    usage_histogram: np.ndarray | list | None = None,
    transitions: np.ndarray | list | None = None,
    per_code_stability: dict[int, dict[str, Any]] | None = None,
    projection_method: str = "pca",
) -> dict[str, str]:
    """Generate a comprehensive visual analysis report.

    Args:
        output_dir: Directory to save all visualizations.
        codebook: Codebook embeddings.
        usage_histogram: Optional usage counts per code.
        transitions: Optional transition matrix.
        per_code_stability: Optional stability metrics per code.
        projection_method: Method for 2D projection.

    Returns:
        Dictionary mapping visualization names to file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Project codebook to 2D
    codebook_2d = project_codebook_2d(codebook, method=projection_method)

    # Base codebook visualization
    paths["codebook_2d"] = plot_codebook_2d(
        codebook_2d,
        output_dir / "codebook_2d.png",
        title=f"Codebook ({len(codebook)} codes, {codebook.shape[1]}D → 2D via {projection_method.upper()})",
    )

    # Usage histogram
    if usage_histogram is not None:
        paths["usage_histogram"] = plot_code_histogram(
            usage_histogram,
            output_dir / "usage_histogram.png",
        )
        paths["codebook_usage"] = plot_codebook_with_usage(
            codebook_2d,
            usage_histogram,
            output_dir / "codebook_usage.png",
        )

    # Transitions
    if transitions is not None:
        paths["transitions"] = plot_transition_matrix(
            transitions,
            output_dir / "transitions.png",
        )

    # Stability
    if per_code_stability is not None:
        paths["stability"] = plot_stability_summary(
            per_code_stability,
            output_dir / "stability_summary.png",
        )

    return paths
