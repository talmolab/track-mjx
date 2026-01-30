"""Transition matrix and chain analysis for VQ-VAE codes.

This module provides functions for analyzing code transition patterns,
identifying transition chains, and classifying code roles (entry, exit, hub).
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .inference_cache import InferenceResult


@dataclass
class TransitionChain:
    """A sequence of three consecutive code transitions (A -> B -> C).

    Attributes:
        codes: Tuple of three code indices (a, b, c).
        prob_ab: Probability of transition from A to B.
        prob_bc: Probability of transition from B to C.
        chain_prob: Joint probability of the chain (prob_ab * prob_bc).
        count: Number of times this chain was observed.
    """

    codes: tuple[int, int, int]
    prob_ab: float
    prob_bc: float
    chain_prob: float
    count: int


@dataclass
class CodeRole:
    """Classification of a code's role in the transition graph.

    Attributes:
        code_idx: The code index.
        in_degree: Number of unique incoming transitions.
        out_degree: Number of unique outgoing transitions.
        self_loop_prob: Probability of staying in this code.
        is_entry: True if this code has many incoming transitions.
        is_exit: True if this code has many outgoing transitions but few incoming.
        is_hub: True if this code has high connectivity in both directions.
        is_steady_state: True if this code has high self-loop probability.
    """

    code_idx: int
    in_degree: int
    out_degree: int
    self_loop_prob: float
    is_entry: bool
    is_exit: bool
    is_hub: bool
    is_steady_state: bool


def compute_transition_matrix(
    results: Sequence[InferenceResult],
    num_codes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute transition counts and probabilities from inference results.

    Args:
        results: List of InferenceResult with code_indices.
        num_codes: Total number of codes in the codebook.

    Returns:
        Tuple of (transition_counts, transition_probs).
        - transition_counts: Shape [num_codes, num_codes], counts[i, j] = # of i->j.
        - transition_probs: Row-normalized probabilities.
    """
    trans_counts = np.zeros((num_codes, num_codes), dtype=np.int32)

    for result in results:
        indices = result.code_indices
        if len(indices) < 2:
            continue
        for i in range(len(indices) - 1):
            from_code = int(indices[i])
            to_code = int(indices[i + 1])
            if 0 <= from_code < num_codes and 0 <= to_code < num_codes:
                trans_counts[from_code, to_code] += 1

    # Normalize to probabilities
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    trans_probs = np.where(row_sums > 0, trans_counts / row_sums, 0.0)

    return trans_counts, trans_probs


def find_transition_chains(
    trans_probs: np.ndarray,
    trans_counts: np.ndarray,
    min_chain_prob: float = 0.01,
    top_k: int = 20,
) -> list[TransitionChain]:
    """Find the most common A -> B -> C transition chains.

    Args:
        trans_probs: Transition probability matrix [num_codes, num_codes].
        trans_counts: Transition count matrix [num_codes, num_codes].
        min_chain_prob: Minimum chain probability threshold.
        top_k: Return only the top K chains by probability.

    Returns:
        List of TransitionChain objects sorted by chain probability.
    """
    num_codes = trans_probs.shape[0]
    chains = []

    for a in range(num_codes):
        for b in range(num_codes):
            prob_ab = trans_probs[a, b]
            if prob_ab < 0.01:  # Skip very low probability first transitions
                continue

            for c in range(num_codes):
                prob_bc = trans_probs[b, c]
                chain_prob = prob_ab * prob_bc

                if chain_prob >= min_chain_prob:
                    # Count occurrences of this chain
                    count = min(trans_counts[a, b], trans_counts[b, c])

                    chains.append(
                        TransitionChain(
                            codes=(a, b, c),
                            prob_ab=float(prob_ab),
                            prob_bc=float(prob_bc),
                            chain_prob=float(chain_prob),
                            count=int(count),
                        )
                    )

    # Sort by chain probability descending
    chains.sort(key=lambda x: x.chain_prob, reverse=True)
    return chains[:top_k]


def classify_code_roles(
    trans_probs: np.ndarray,
    trans_counts: np.ndarray,
    in_degree_threshold: float = 0.7,
    out_degree_threshold: float = 0.7,
    hub_threshold: float = 0.5,
    steady_state_threshold: float = 0.5,
) -> list[CodeRole]:
    """Classify each code's role based on transition patterns.

    Args:
        trans_probs: Transition probability matrix [num_codes, num_codes].
        trans_counts: Transition count matrix [num_codes, num_codes].
        in_degree_threshold: Percentile threshold for high in-degree.
        out_degree_threshold: Percentile threshold for high out-degree.
        hub_threshold: Percentile threshold for hub classification.
        steady_state_threshold: Self-loop probability threshold for steady state.

    Returns:
        List of CodeRole for each code.
    """
    num_codes = trans_probs.shape[0]

    # Compute degrees (number of unique transitions with non-zero probability)
    in_degrees = np.sum(trans_probs > 0.01, axis=0)
    out_degrees = np.sum(trans_probs > 0.01, axis=1)
    self_loop_probs = np.diag(trans_probs)

    # Compute thresholds based on percentiles
    in_threshold = np.percentile(in_degrees, in_degree_threshold * 100)
    out_threshold = np.percentile(out_degrees, out_degree_threshold * 100)
    hub_in = np.percentile(in_degrees, hub_threshold * 100)
    hub_out = np.percentile(out_degrees, hub_threshold * 100)

    roles = []
    for code_idx in range(num_codes):
        in_deg = int(in_degrees[code_idx])
        out_deg = int(out_degrees[code_idx])
        self_loop = float(self_loop_probs[code_idx])

        # Classification logic (convert to native bool for JSON serialization)
        is_entry = bool(in_deg >= in_threshold and out_deg < out_threshold)
        is_exit = bool(out_deg >= out_threshold and in_deg < in_threshold)
        is_hub = bool(in_deg >= hub_in and out_deg >= hub_out)
        is_steady_state = bool(self_loop >= steady_state_threshold)

        roles.append(
            CodeRole(
                code_idx=code_idx,
                in_degree=in_deg,
                out_degree=out_deg,
                self_loop_prob=self_loop,
                is_entry=is_entry,
                is_exit=is_exit,
                is_hub=is_hub,
                is_steady_state=is_steady_state,
            )
        )

    return roles


def classify_kinematic_type(
    linear_vel: float,
    angular_vel: float,
    joint_vel: float,
) -> str:
    """Classify a code into kinematic type based on velocity features.

    Args:
        linear_vel: Mean linear velocity.
        angular_vel: Mean angular velocity.
        joint_vel: Mean joint velocity.

    Returns:
        One of: "resting", "transitional", "locomotion"
    """
    # Thresholds based on observed distributions
    if linear_vel < 0.12 and angular_vel < 2.0 and joint_vel < 0.5:
        return "resting"
    elif linear_vel > 0.25 or (angular_vel > 3.5 and joint_vel > 0.7):
        return "locomotion"
    else:
        return "transitional"


def visualize_enhanced_transition_graph(
    trans_probs: np.ndarray,
    trans_counts: np.ndarray,
    roles: list[CodeRole],
    kinematic_profiles: list[dict] | None,
    duration_stats: list[dict] | None,
    output_path: str | Path,
    min_edge_prob: float = 0.02,
    figsize: tuple[int, int] = (16, 14),
) -> str:
    """Visualize enhanced transition graph with kinematic and duration info.

    Creates a comprehensive visualization showing:
    - Node colors based on kinematic type (resting/transitional/locomotion)
    - Node size based on total frame count
    - Node border based on role (hub/entry/exit)
    - Edge thickness based on transition probability
    - Labels showing code index and key stats

    Args:
        trans_probs: Transition probability matrix.
        trans_counts: Transition count matrix.
        roles: List of CodeRole for each code.
        kinematic_profiles: List of kinematic profile dicts (from kinematic_analysis).
        duration_stats: List of duration stat dicts (from segment_analysis).
        output_path: Path to save the figure.
        min_edge_prob: Minimum probability to draw an edge (excluding self-loops).
        figsize: Figure size.

    Returns:
        Path to the saved figure.
    """
    try:
        import networkx as nx
    except ImportError:
        logging.warning("NetworkX not installed, skipping enhanced graph visualization")
        return ""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_codes = trans_probs.shape[0]

    # Build kinematic type mapping
    kin_types = {}
    if kinematic_profiles:
        for profile in kinematic_profiles:
            code_idx = profile["code_idx"]
            kin_type = classify_kinematic_type(
                profile.get("linear_velocity_mean", 0),
                profile.get("angular_velocity_mean", 0),
                profile.get("joint_velocities_mean", 0),
            )
            kin_types[code_idx] = kin_type
    else:
        # Default all to transitional if no kinematic data
        kin_types = {i: "transitional" for i in range(num_codes)}

    # Build frame count mapping for node sizes
    frame_counts = {}
    if kinematic_profiles:
        for profile in kinematic_profiles:
            frame_counts[profile["code_idx"]] = profile.get("n_frames", 100)
    else:
        frame_counts = {i: 100 for i in range(num_codes)}

    # Build duration mapping for labels
    mean_durations = {}
    if duration_stats:
        for stat in duration_stats:
            mean_durations[stat["code_idx"]] = stat.get("mean", 0)
    else:
        mean_durations = {i: 0 for i in range(num_codes)}

    # Create directed graph
    G = nx.DiGraph()

    # Color scheme by kinematic type
    type_colors = {
        "resting": "#4CAF50",      # Green
        "transitional": "#FF9800",  # Orange
        "locomotion": "#2196F3",    # Blue
    }

    # Border colors by role
    role_borders = {
        "hub": "#E91E63",      # Pink (thick border)
        "entry": "#9C27B0",    # Purple
        "exit": "#00BCD4",     # Cyan
        "normal": "#666666",   # Gray
    }

    # Add nodes
    node_colors = []
    node_borders = []
    node_sizes = []
    node_border_widths = []

    for role in roles:
        code_idx = role.code_idx
        G.add_node(code_idx)

        # Node color by kinematic type
        kin_type = kin_types.get(code_idx, "transitional")
        node_colors.append(type_colors[kin_type])

        # Node border by role
        if role.is_hub:
            node_borders.append(role_borders["hub"])
            node_border_widths.append(4)
        elif role.is_entry:
            node_borders.append(role_borders["entry"])
            node_border_widths.append(3)
        elif role.is_exit:
            node_borders.append(role_borders["exit"])
            node_border_widths.append(3)
        else:
            node_borders.append(role_borders["normal"])
            node_border_widths.append(1)

        # Node size by frame count (log scale for better visibility)
        frames = frame_counts.get(code_idx, 100)
        size = 400 + 300 * np.log1p(frames / 100)
        node_sizes.append(size)

    # Add edges (excluding self-loops for clarity, but include them in analysis)
    edges_to_draw = []
    edge_weights = []
    edge_colors = []

    for i in range(num_codes):
        for j in range(num_codes):
            prob = trans_probs[i, j]
            if i == j:
                # Self-loops handled separately
                continue
            if prob >= min_edge_prob:
                edges_to_draw.append((i, j))
                edge_weights.append(prob)
                # Color edges by source kinematic type
                src_type = kin_types.get(i, "transitional")
                edge_colors.append(type_colors[src_type])

    for u, v in edges_to_draw:
        G.add_edge(u, v, weight=trans_probs[u, v])

    if len(edges_to_draw) == 0:
        logging.warning("No edges above threshold, skipping graph visualization")
        return ""

    fig, ax = plt.subplots(figsize=figsize)

    # Use kamada_kawai layout for better separation
    pos = nx.kamada_kawai_layout(G)

    # Draw edges first (below nodes)
    edge_widths = [w * 8 for w in edge_weights]  # Scale for visibility
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges_to_draw,
        width=edge_widths,
        alpha=0.4,
        edge_color=edge_colors,
        arrows=True,
        arrowsize=20,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.1",
        ax=ax,
    )

    # Draw self-loop indicators (arcs)
    for role in roles:
        code_idx = role.code_idx
        self_prob = trans_probs[code_idx, code_idx]
        if self_prob > 0.5:
            x, y = pos[code_idx]
            # Draw a small arc above the node to indicate self-loop
            arc_size = 0.08 * (self_prob - 0.5) / 0.5  # Scale by probability
            circle = plt.Circle(
                (x, y + 0.12),
                arc_size,
                fill=False,
                color=type_colors[kin_types.get(code_idx, "transitional")],
                linewidth=2,
                alpha=0.7,
            )
            ax.add_patch(circle)

    # Draw nodes with borders
    for i, node in enumerate(G.nodes()):
        x, y = pos[node]
        # Outer circle (border)
        outer = plt.Circle(
            (x, y),
            0.06 + node_border_widths[i] * 0.003,
            color=node_borders[i],
            zorder=2,
        )
        ax.add_patch(outer)
        # Inner circle (fill)
        inner = plt.Circle(
            (x, y),
            0.055,
            color=node_colors[i],
            zorder=3,
        )
        ax.add_patch(inner)

    # Draw labels with stats
    for node in G.nodes():
        x, y = pos[node]
        role = roles[node]
        kin_type = kin_types.get(node, "?")[0].upper()  # R/T/L
        self_prob = trans_probs[node, node]
        dur = mean_durations.get(node, 0)

        # Main label (code index)
        ax.text(
            x, y, str(node),
            fontsize=11, fontweight="bold",
            ha="center", va="center",
            color="white", zorder=4,
        )
        # Stats label below
        stats_text = f"{self_prob*100:.0f}%"
        if dur > 0:
            stats_text += f"\n{dur:.0f}f"
        ax.text(
            x, y - 0.11, stats_text,
            fontsize=7, ha="center", va="top",
            color="#333333", zorder=4,
        )

    # Create legend
    legend_elements = [
        # Kinematic types
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=type_colors["resting"],
                   markersize=12, label="Resting (low velocity)"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=type_colors["transitional"],
                   markersize=12, label="Transitional (moderate)"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=type_colors["locomotion"],
                   markersize=12, label="Locomotion (high velocity)"),
        plt.Line2D([0], [0], color="w", label=""),  # Spacer
        # Roles (border colors)
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="white", markeredgecolor=role_borders["hub"],
                   markeredgewidth=3, markersize=12, label="Hub (high connectivity)"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="white", markeredgecolor=role_borders["entry"],
                   markeredgewidth=3, markersize=12, label="Entry (many inputs)"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="white", markeredgecolor=role_borders["exit"],
                   markeredgewidth=3, markersize=12, label="Exit (many outputs)"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=9,
        framealpha=0.9,
    )

    # Add title and annotations
    ax.set_title(
        "VQ-VAE Code Transition Graph\n"
        "(node color = kinematic type, border = role, "
        "labels show self-loop % and mean duration)",
        fontsize=12,
        fontweight="bold",
    )

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    logging.info(f"Saved enhanced transition graph to {output_path}")
    return str(output_path)


def visualize_transition_graph(
    trans_probs: np.ndarray,
    roles: list[CodeRole],
    output_path: str | Path,
    min_edge_prob: float = 0.05,
    figsize: tuple[int, int] = (12, 12),
) -> str:
    """Visualize the transition graph with role-based coloring.

    Args:
        trans_probs: Transition probability matrix.
        roles: List of CodeRole for node coloring.
        output_path: Path to save the figure.
        min_edge_prob: Minimum probability to draw an edge.
        figsize: Figure size.

    Returns:
        Path to the saved figure.
    """
    try:
        import networkx as nx
    except ImportError:
        logging.warning("NetworkX not installed, skipping graph visualization")
        return ""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_codes = trans_probs.shape[0]

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes with role-based colors
    node_colors = []
    for role in roles:
        G.add_node(role.code_idx)
        if role.is_hub:
            node_colors.append("#FF6B6B")  # Red for hubs
        elif role.is_entry:
            node_colors.append("#4ECDC4")  # Teal for entry
        elif role.is_exit:
            node_colors.append("#45B7D1")  # Blue for exit
        elif role.is_steady_state:
            node_colors.append("#96CEB4")  # Green for steady state
        else:
            node_colors.append("#CCCCCC")  # Gray for others

    # Add edges with weights
    edge_weights = []
    for i in range(num_codes):
        for j in range(num_codes):
            if trans_probs[i, j] >= min_edge_prob:
                G.add_edge(i, j, weight=trans_probs[i, j])
                edge_weights.append(trans_probs[i, j])

    if len(edge_weights) == 0:
        logging.warning("No edges above threshold, skipping graph visualization")
        return ""

    fig, ax = plt.subplots(figsize=figsize)

    # Use spring layout for positioning
    pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42)

    # Draw nodes
    node_sizes = [300 + 100 * roles[n].in_degree for n in G.nodes()]
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        ax=ax,
    )

    # Draw edges with varying width based on probability
    edges = G.edges()
    weights = [G[u][v]["weight"] * 3 for u, v in edges]
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges,
        width=weights,
        alpha=0.5,
        edge_color="gray",
        arrows=True,
        arrowsize=15,
        ax=ax,
    )

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF6B6B",
                   markersize=10, label="Hub"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#4ECDC4",
                   markersize=10, label="Entry"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#45B7D1",
                   markersize=10, label="Exit"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#96CEB4",
                   markersize=10, label="Steady State"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#CCCCCC",
                   markersize=10, label="Other"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    ax.set_title("Code Transition Graph", fontsize=14)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(output_path)


def save_transition_analysis(
    output_dir: str | Path,
    trans_counts: np.ndarray,
    trans_probs: np.ndarray,
    chains: list[TransitionChain],
    roles: list[CodeRole],
) -> dict[str, str]:
    """Save all transition analysis results.

    Args:
        output_dir: Directory to save outputs.
        trans_counts: Transition count matrix.
        trans_probs: Transition probability matrix.
        chains: List of TransitionChain objects.
        roles: List of CodeRole objects.

    Returns:
        Dictionary mapping output names to file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Save transition matrix as NPZ
    np.savez(
        output_dir / "transition_matrix.npz",
        counts=trans_counts,
        probs=trans_probs,
    )
    paths["matrix"] = str(output_dir / "transition_matrix.npz")

    # Save chains as JSON
    chains_data = [asdict(c) for c in chains]
    with open(output_dir / "chains.json", "w") as f:
        json.dump(chains_data, f, indent=2)
    paths["chains"] = str(output_dir / "chains.json")

    # Save roles as JSON
    roles_data = [asdict(r) for r in roles]
    with open(output_dir / "code_roles.json", "w") as f:
        json.dump(roles_data, f, indent=2)
    paths["roles"] = str(output_dir / "code_roles.json")

    logging.info(f"Saved transition analysis to {output_dir}")
    return paths


def run_transition_analysis(
    results: Sequence[InferenceResult],
    num_codes: int,
    output_dir: str | Path,
    min_chain_prob: float = 0.01,
    top_k_chains: int = 20,
    min_edge_prob: float = 0.05,
) -> dict[str, str]:
    """Run complete transition analysis pipeline.

    Args:
        results: List of InferenceResult with code_indices.
        num_codes: Total number of codes.
        output_dir: Directory to save outputs.
        min_chain_prob: Minimum probability for chain detection.
        top_k_chains: Number of top chains to return.
        min_edge_prob: Minimum probability to draw edges in graph.

    Returns:
        Dictionary mapping output names to file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Computing transition matrix...")
    trans_counts, trans_probs = compute_transition_matrix(results, num_codes)

    logging.info("Finding transition chains...")
    chains = find_transition_chains(
        trans_probs, trans_counts, min_chain_prob, top_k_chains
    )
    logging.info(f"Found {len(chains)} chains above threshold")

    logging.info("Classifying code roles...")
    roles = classify_code_roles(trans_probs, trans_counts)

    # Count roles
    n_hubs = sum(1 for r in roles if r.is_hub)
    n_entry = sum(1 for r in roles if r.is_entry)
    n_exit = sum(1 for r in roles if r.is_exit)
    n_steady = sum(1 for r in roles if r.is_steady_state)
    logging.info(
        f"Roles: {n_hubs} hubs, {n_entry} entry, {n_exit} exit, {n_steady} steady"
    )

    # Save results
    paths = save_transition_analysis(
        output_dir, trans_counts, trans_probs, chains, roles
    )

    # Visualize graph
    graph_path = visualize_transition_graph(
        trans_probs, roles, output_dir / "transition_graph.png", min_edge_prob
    )
    if graph_path:
        paths["graph"] = graph_path

    # Plot transition matrix
    from .visualization import plot_transition_matrix

    matrix_plot = plot_transition_matrix(
        trans_probs, output_dir / "transition_matrix.png"
    )
    paths["matrix_plot"] = matrix_plot

    return paths
