"""Visualization utilities for VQ-VAE codebook analysis.

This module provides functions for visualizing codebook embeddings,
code usage patterns, trajectories through the latent space, and
community structure analysis.
"""

from pathlib import Path
from typing import Any, TYPE_CHECKING

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Wedge

if TYPE_CHECKING:
    from .community_analysis import CommunityStructure


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
            raise ImportError("UMAP not installed. Install with: pip install umap-learn")
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
        edgecolors='white',
        linewidth=0.5,
    )

    if show_labels:
        for i, (x, y) in enumerate(codebook_2d):
            ax.annotate(
                str(i),
                (x, y),
                fontsize=label_fontsize,
                ha='center',
                va='center',
            )

    plt.colorbar(scatter, ax=ax, label='Code Index')
    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.axis('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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

    colors = ['steelblue'] * num_codes
    if highlight_threshold is not None:
        colors = ['orange' if h > highlight_threshold else 'steelblue' for h in histogram]

    ax.bar(range(num_codes), histogram, color=colors, alpha=0.8)

    ax.set_xlabel('Code Index')
    ax.set_ylabel('Usage Count')
    ax.set_title(title)
    ax.set_xlim(-0.5, num_codes - 0.5)

    # Add statistics
    used_codes = np.sum(histogram > 0)
    total_uses = np.sum(histogram)
    stats_text = f'Used: {used_codes}/{num_codes} codes ({used_codes/num_codes:.1%})\n'
    stats_text += f'Total: {total_uses} uses'
    ax.text(
        0.98, 0.98, stats_text,
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
        edgecolors='gray',
        linewidth=0.5,
    )

    if show_labels:
        for i, (x, y) in enumerate(codebook_2d):
            if usage_counts[i] > 0:
                ax.annotate(
                    str(i),
                    (x, y),
                    fontsize=7,
                    ha='center',
                    va='center',
                    alpha=0.8,
                )

    plt.colorbar(scatter, ax=ax, label='Usage Count')
    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.axis('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
            c='lightgray',
            s=80,
            alpha=0.5,
            edgecolors='gray',
            linewidth=0.5,
        )
        for i, (x, y) in enumerate(codebook_2d):
            ax.annotate(str(i), (x, y), fontsize=6, ha='center', va='center', alpha=0.3)

    # Trajectory
    traj_coords = codebook_2d[trajectory]

    # Draw line
    ax.plot(
        traj_coords[:, 0],
        traj_coords[:, 1],
        'b-',
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
        cmap='viridis',
        s=30,
        alpha=0.8,
        zorder=3,
    )

    # Mark start and end
    ax.scatter(
        traj_coords[0, 0], traj_coords[0, 1],
        c='green', s=200, marker='^', zorder=4, label='Start'
    )
    ax.scatter(
        traj_coords[-1, 0], traj_coords[-1, 1],
        c='red', s=200, marker='s', zorder=4, label='End'
    )

    plt.colorbar(scatter, ax=ax, label='Time')
    ax.legend()
    ax.set_title(f"{title} ({len(trajectory)} steps)")
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.axis('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
        label = 'log10(count + 1)'
    else:
        plot_data = transitions
        label = 'count'

    im = ax.imshow(plot_data, cmap=cmap, aspect='auto')

    plt.colorbar(im, ax=ax, label=label)
    ax.set_title(title)
    ax.set_xlabel('To Code')
    ax.set_ylabel('From Code')

    # Show fewer ticks for large codebooks
    if num_codes > 20:
        tick_step = num_codes // 10
        ticks = np.arange(0, num_codes, tick_step)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
    colors = ['red' if f else 'steelblue' for f in fallen]
    axes[0].bar(codes, survival_steps, color=colors, alpha=0.7)
    axes[0].set_xlabel('Code Index')
    axes[0].set_ylabel('Survival Steps')
    axes[0].set_title('Survival by Code (red=fallen)')

    # Mean reward
    axes[1].bar(codes, mean_rewards, color='steelblue', alpha=0.7)
    axes[1].set_xlabel('Code Index')
    axes[1].set_ylabel('Mean Reward')
    axes[1].set_title('Mean Reward by Code')

    # Survival vs Reward scatter
    axes[2].scatter(survival_steps, mean_rewards, c=fallen, cmap='RdYlGn_r', alpha=0.7)
    axes[2].set_xlabel('Survival Steps')
    axes[2].set_ylabel('Mean Reward')
    axes[2].set_title('Survival vs Reward')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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


# =============================================================================
# COMMUNITY VISUALIZATION
# =============================================================================


# Community color palette (8 distinct colors)
COMMUNITY_COLORS = np.array([
    [66, 133, 244],    # Blue
    [234, 67, 53],     # Red
    [251, 188, 5],     # Yellow
    [52, 168, 83],     # Green
    [155, 89, 182],    # Purple
    [26, 188, 156],    # Teal
    [241, 196, 15],    # Gold
    [230, 126, 34],    # Orange
], dtype=np.uint8)


def get_community_colormap(n_communities: int) -> np.ndarray:
    """Get a colormap for community visualization.

    Args:
        n_communities: Number of communities.

    Returns:
        Array of RGB colors, shape [n_communities, 3], values 0-255.
    """
    if n_communities <= len(COMMUNITY_COLORS):
        return COMMUNITY_COLORS[:n_communities]

    # Generate additional colors via HSV
    colors = list(COMMUNITY_COLORS)
    for i in range(len(COMMUNITY_COLORS), n_communities):
        hue = (i - len(COMMUNITY_COLORS)) / (n_communities - len(COMMUNITY_COLORS))
        # Convert HSV to RGB
        c = plt.cm.hsv(hue)
        colors.append([int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)])

    return np.array(colors, dtype=np.uint8)


def plot_community_transition_graph(
    structure: "CommunityStructure",
    output_path: str | Path,
    figsize: tuple[int, int] = (12, 10),
    min_edge_prob: float = 0.05,
) -> str:
    """Plot coarsened community-level transition graph.

    Args:
        structure: CommunityStructure from community analysis.
        output_path: Path to save figure.
        figsize: Figure size.
        min_edge_prob: Minimum probability to draw an edge.

    Returns:
        Path to saved figure.
    """
    try:
        import networkx as nx
    except ImportError:
        return ""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_communities = structure.n_communities
    colors = get_community_colormap(n_communities) / 255.0

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes
    for comm in structure.communities:
        G.add_node(
            comm.id,
            label=f"C{comm.id}\n({len(comm.code_indices)} codes)",
            size=len(comm.code_indices),
        )

    # Add edges from coarsened transition matrix
    coarsened = structure.coarsened_transitions
    for i in range(n_communities):
        for j in range(n_communities):
            prob = coarsened[i, j]
            if prob >= min_edge_prob:
                G.add_edge(i, j, weight=prob)

    fig, ax = plt.subplots(figsize=figsize)

    # Layout
    pos = nx.kamada_kawai_layout(G)

    # Node sizes based on number of codes
    node_sizes = [300 + 50 * G.nodes[n]["size"] for n in G.nodes()]
    node_colors = [colors[n] for n in G.nodes()]

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.9,
        edgecolors="black",
        linewidths=2,
    )

    # Draw edges with varying width
    edges = G.edges()
    if edges:
        weights = [G[u][v]["weight"] * 5 for u, v in edges]
        edge_colors = [colors[u] for u, v in edges]

        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=edges,
            width=weights,
            alpha=0.6,
            edge_color=edge_colors,
            arrows=True,
            arrowsize=20,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1",
        )

    # Draw labels
    labels = {n: G.nodes[n]["label"] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=9, font_weight="bold")

    ax.set_title(
        f"Community Transition Graph\n"
        f"({n_communities} communities, modularity={structure.modularity:.3f})",
        fontsize=12, fontweight="bold"
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    return str(output_path)


def plot_community_composition(
    structure: "CommunityStructure",
    output_path: str | Path,
    figsize: tuple[int, int] = (14, 6),
) -> str:
    """Plot composition of each community (codes within each).

    Args:
        structure: CommunityStructure from community analysis.
        output_path: Path to save figure.
        figsize: Figure size.

    Returns:
        Path to saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_communities = structure.n_communities
    colors = get_community_colormap(n_communities) / 255.0

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Stacked bar showing codes per community
    ax = axes[0]
    community_ids = [comm.id for comm in structure.communities]
    core_counts = [len(comm.core_codes) for comm in structure.communities]
    boundary_counts = [len(comm.boundary_codes) for comm in structure.communities]

    x = np.arange(n_communities)
    width = 0.6

    ax.bar(x, core_counts, width, label="Core codes", color=[colors[i] for i in community_ids], alpha=0.9)
    ax.bar(x, boundary_counts, width, bottom=core_counts,
           label="Boundary codes", color=[colors[i] for i in community_ids], alpha=0.5,
           edgecolor="black", linewidth=1)

    ax.set_xlabel("Community")
    ax.set_ylabel("Number of Codes")
    ax.set_title("Community Composition")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{i}" for i in community_ids])
    ax.legend()

    # Right: Pie chart of community sizes
    ax = axes[1]
    sizes = [len(comm.code_indices) for comm in structure.communities]
    labels = [f"C{comm.id} ({len(comm.code_indices)})" for comm in structure.communities]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels,
        colors=[colors[i] for i in community_ids],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )
    ax.set_title("Community Size Distribution")

    plt.suptitle(
        f"Community Structure: {n_communities} communities, "
        f"{len(structure.overlapping_codes)} overlapping codes",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(output_path)


def _draw_pie_marker(ax, x, y, membership_dict: dict[int, float], colors: np.ndarray, size: float = 0.05):
    """Draw a pie chart as a node marker showing multi-community membership.

    Args:
        ax: Matplotlib axis.
        x, y: Position.
        membership_dict: Dict of {community_id: membership_probability}.
        colors: Community colormap [n_communities, 3].
        size: Size of the pie.
    """
    # Sort by community ID for consistent ordering
    sorted_items = sorted(membership_dict.items())
    communities = [c for c, _ in sorted_items]
    probs = [p for _, p in sorted_items]

    # Normalize to sum to 1
    total = sum(probs)
    probs = [p / total for p in probs]

    # Draw pie wedges
    start_angle = 90  # Start from top
    for comm_id, prob in zip(communities, probs):
        angle = prob * 360
        color = colors[comm_id] / 255.0
        wedge = Wedge(
            (x, y), size,
            start_angle - angle, start_angle,
            facecolor=color,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.add_patch(wedge)
        start_angle -= angle


def plot_community_transition_matrix(
    structure: "CommunityStructure",
    community_id: int,
    trans_probs: np.ndarray,
    output_path: str | Path,
    figsize: tuple[int, int] | None = None,
) -> str:
    """Plot transition matrix heatmap for codes within a single community.

    Args:
        structure: CommunityStructure from community analysis.
        community_id: ID of community to visualize.
        trans_probs: Full code-level transition probability matrix.
        output_path: Path to save figure.
        figsize: Figure size (auto-calculated if None).

    Returns:
        Path to saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    community = structure.communities[community_id]
    code_indices = sorted(community.code_indices)

    if len(code_indices) == 0:
        return ""

    # Extract sub-matrix for this community
    n_codes = len(code_indices)
    sub_matrix = np.zeros((n_codes, n_codes))
    for i, code_i in enumerate(code_indices):
        for j, code_j in enumerate(code_indices):
            sub_matrix[i, j] = trans_probs[code_i, code_j]

    # Auto-calculate figure size based on number of codes
    if figsize is None:
        size = max(6, min(20, n_codes * 0.4))
        figsize = (size, size)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(sub_matrix, cmap="Blues", aspect="equal", vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Transition Probability", fontsize=10)

    # Set ticks and labels
    ax.set_xticks(range(n_codes))
    ax.set_yticks(range(n_codes))
    ax.set_xticklabels(code_indices, fontsize=8, rotation=90)
    ax.set_yticklabels(code_indices, fontsize=8)

    ax.set_xlabel("To Code", fontsize=11)
    ax.set_ylabel("From Code", fontsize=11)

    ax.set_title(
        f"Community {community_id} Transition Matrix\n"
        f"({len(community.core_codes)} core, {len(community.boundary_codes)} boundary, "
        f"{n_codes} total codes)",
        fontsize=12, fontweight="bold"
    )

    # Add grid
    ax.set_xticks(np.arange(-0.5, n_codes, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_codes, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    return str(output_path)


def plot_overlap_summary(
    structure: "CommunityStructure",
    output_path: str | Path,
    figsize: tuple[int, int] = (14, 6),
) -> str:
    """Plot summary of overlapping codes across community pairs.

    Args:
        structure: CommunityStructure from community analysis.
        output_path: Path to save figure.
        figsize: Figure size.

    Returns:
        Path to saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_communities = structure.n_communities
    colors = get_community_colormap(n_communities) / 255.0

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Heatmap of overlapping codes between community pairs
    ax = axes[0]

    # Count codes shared between each pair
    overlap_matrix = np.zeros((n_communities, n_communities))
    for code, communities in structure.overlap_stats.items():
        comm_ids = list(communities.keys())
        for i, c1 in enumerate(comm_ids):
            for c2 in comm_ids[i+1:]:
                overlap_matrix[c1, c2] += 1
                overlap_matrix[c2, c1] += 1

    im = ax.imshow(overlap_matrix, cmap="YlOrRd")
    ax.set_xticks(range(n_communities))
    ax.set_yticks(range(n_communities))
    ax.set_xticklabels([f"C{i}" for i in range(n_communities)])
    ax.set_yticklabels([f"C{i}" for i in range(n_communities)])
    ax.set_xlabel("Community")
    ax.set_ylabel("Community")
    ax.set_title("Shared Boundary Codes")
    plt.colorbar(im, ax=ax, label="# shared codes")

    # Annotate cells
    for i in range(n_communities):
        for j in range(n_communities):
            if overlap_matrix[i, j] > 0:
                ax.text(j, i, int(overlap_matrix[i, j]),
                        ha="center", va="center", fontsize=8, fontweight="bold")

    # Right: Bar chart of overlapping codes per community
    ax = axes[1]

    # Count boundary codes per community
    boundary_counts = {comm.id: len(comm.boundary_codes) for comm in structure.communities}
    core_counts = {comm.id: len(comm.core_codes) for comm in structure.communities}

    x = np.arange(n_communities)
    width = 0.35

    ax.bar(x - width/2, [core_counts[i] for i in range(n_communities)],
           width, label="Core", color="steelblue", alpha=0.8)
    ax.bar(x + width/2, [boundary_counts[i] for i in range(n_communities)],
           width, label="Boundary", color="coral", alpha=0.8)

    ax.set_xlabel("Community")
    ax.set_ylabel("Number of Codes")
    ax.set_title("Core vs Boundary Codes")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{i}" for i in range(n_communities)])
    ax.legend()

    # Add overall statistics
    total_codes = sum(len(c.code_indices) for c in structure.communities)
    n_overlapping = len(structure.overlapping_codes)
    overlap_fraction = n_overlapping / total_codes if total_codes > 0 else 0

    fig.suptitle(
        f"Overlap Analysis: {n_overlapping} overlapping codes ({overlap_fraction:.1%} of total)",
        fontsize=12, fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(output_path)


def create_community_visualizations(
    structure: "CommunityStructure",
    trans_probs: np.ndarray,
    output_dir: str | Path,
) -> dict[str, str]:
    """Generate all community visualizations.

    Args:
        structure: CommunityStructure from community analysis.
        trans_probs: Code-level transition probability matrix.
        output_dir: Directory to save outputs.

    Returns:
        Dictionary mapping output names to file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Community transition graph
    paths["community_graph"] = plot_community_transition_graph(
        structure, output_dir / "community_transition_graph.png"
    )

    # Community composition
    paths["community_composition"] = plot_community_composition(
        structure, output_dir / "community_composition.png"
    )

    # Overlap summary
    paths["overlap_summary"] = plot_overlap_summary(
        structure, output_dir / "overlap_summary.png"
    )

    # Per-community transition matrices
    for comm in structure.communities:
        path = plot_community_transition_matrix(
            structure, comm.id, trans_probs,
            output_dir / f"community_{comm.id}_transitions.png"
        )
        if path:
            paths[f"community_{comm.id}_transitions"] = path

    return paths
