"""RVQ-aware analysis for multi-depth residual vector quantization.

Provides three analyses gated by the availability of ``rvq_indices``:

1. **Parent-Child Heatmap**: Joint distribution of L0 and L1 code usage.
2. **Intra-Parent Diversity**: Entropy of L1 child codes conditioned on L0 parent.
3. **Hierarchical Transitions**: L1 transition rates conditioned on L0 stability.

All functions accept ``list[InferenceResult]`` and return matplotlib figures.
"""

import logging
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .inference_cache import InferenceResult


# =============================================================================
# 3a. Parent-Child Heatmap
# =============================================================================


def compute_parent_child_heatmap(
    results: Sequence[InferenceResult],
    num_codes: int,
) -> tuple[plt.Figure, np.ndarray]:
    """Compute and plot the joint distribution of L0 and L1 codes.

    Args:
        results: Inference results with ``rvq_indices`` populated.
        num_codes: Number of codes per depth level.

    Returns:
        Tuple of (figure, joint_counts) where joint_counts has shape
        ``[num_codes, num_codes]`` with ``[l0, l1]`` = count.
    """
    joint_counts = np.zeros((num_codes, num_codes), dtype=np.int64)

    for r in results:
        if r.rvq_indices is None or len(r.rvq_indices) < 2:
            continue
        l0 = r.rvq_indices[0]
        l1 = r.rvq_indices[1]
        T = min(len(l0), len(l1))
        for t in range(T):
            joint_counts[int(l0[t]), int(l1[t])] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        np.log1p(joint_counts),
        aspect="auto",
        cmap="viridis",
        origin="lower",
    )
    ax.set_xlabel("L1 Code Index")
    ax.set_ylabel("L0 Code Index")
    ax.set_title("Parent-Child Code Usage (log scale)")
    plt.colorbar(im, ax=ax, label="log(count + 1)")
    plt.tight_layout()

    return fig, joint_counts


# =============================================================================
# 3b. Intra-Parent Diversity
# =============================================================================


def compute_intra_parent_diversity(
    joint_counts: np.ndarray,
) -> tuple[plt.Figure, np.ndarray]:
    """Compute and plot entropy of L1 children conditioned on each L0 parent.

    Args:
        joint_counts: Joint count array of shape ``[num_codes_l0, num_codes_l1]``
            from :func:`compute_parent_child_heatmap`.

    Returns:
        Tuple of (figure, per_parent_entropy) where per_parent_entropy has
        shape ``[num_codes_l0]``.
    """
    num_parents = joint_counts.shape[0]
    entropies = np.zeros(num_parents)

    for p in range(num_parents):
        row = joint_counts[p]
        total = row.sum()
        if total == 0:
            continue
        probs = row / total
        probs = probs[probs > 0]
        entropies[p] = -np.sum(probs * np.log2(probs))

    active = joint_counts.sum(axis=1) > 0
    mean_entropy = entropies[active].mean() if active.any() else 0.0

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["steelblue" if a else "lightgray" for a in active]
    ax.bar(range(num_parents), entropies, color=colors, edgecolor="none")
    ax.axhline(
        mean_entropy,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"Mean = {mean_entropy:.2f}",
    )
    ax.set_xlabel("L0 Parent Code")
    ax.set_ylabel("Entropy (bits)")
    ax.set_title("Intra-Parent L1 Diversity")
    ax.legend()
    ax.set_xlim(-0.5, num_parents - 0.5)
    plt.tight_layout()

    return fig, entropies


# =============================================================================
# 3c. Hierarchical Transition Analysis
# =============================================================================


def compute_hierarchical_transitions(
    results: Sequence[InferenceResult],
    num_codes: int,
) -> tuple[plt.Figure, dict[str, float], np.ndarray]:
    """Compute L1 transition rates conditioned on L0 stability.

    Key metric: L1 transition rate when L0 stays the same. If this is
    high, L1 captures fine-grained temporal structure within L0 segments.

    Args:
        results: Inference results with ``rvq_indices`` populated.
        num_codes: Number of codes per depth level.

    Returns:
        Tuple of (figure, rates_dict, within_parent_transitions) where:
        - rates_dict has keys: l0_transition_rate, l1_transition_rate,
          l1_transition_rate_within_l0.
        - within_parent_transitions has shape ``[num_codes, num_codes]`` with
          ``[l1_from, l1_to]`` = count, accumulated only when L0 stays same.
    """
    l0_transitions = 0
    l0_total = 0
    l1_transitions = 0
    l1_total = 0
    l1_transitions_within_l0 = 0
    l1_total_within_l0 = 0

    within_parent_trans = np.zeros((num_codes, num_codes), dtype=np.int64)

    for r in results:
        if r.rvq_indices is None or len(r.rvq_indices) < 2:
            continue
        l0 = r.rvq_indices[0]
        l1 = r.rvq_indices[1]
        T = min(len(l0), len(l1))

        for t in range(1, T):
            l0_prev, l0_curr = int(l0[t - 1]), int(l0[t])
            l1_prev, l1_curr = int(l1[t - 1]), int(l1[t])

            # L0 transitions
            l0_total += 1
            if l0_curr != l0_prev:
                l0_transitions += 1

            # L1 transitions (unconditional)
            l1_total += 1
            if l1_curr != l1_prev:
                l1_transitions += 1

            # L1 transitions within same L0 parent
            if l0_curr == l0_prev:
                l1_total_within_l0 += 1
                if l1_curr != l1_prev:
                    l1_transitions_within_l0 += 1
                    within_parent_trans[l1_prev, l1_curr] += 1

    rates = {
        "l0_transition_rate": l0_transitions / max(l0_total, 1),
        "l1_transition_rate": l1_transitions / max(l1_total, 1),
        "l1_transition_rate_within_l0": (
            l1_transitions_within_l0 / max(l1_total_within_l0, 1)
        ),
    }

    # Plot: bar chart of rates + within-parent transition heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: bar chart of rates
    rate_names = [
        "L0 trans. rate",
        "L1 trans. rate\n(unconditional)",
        "L1 trans. rate\n(within L0)",
    ]
    rate_values = [
        rates["l0_transition_rate"],
        rates["l1_transition_rate"],
        rates["l1_transition_rate_within_l0"],
    ]
    bar_colors = ["#2196F3", "#FF9800", "#4CAF50"]
    axes[0].bar(rate_names, rate_values, color=bar_colors, edgecolor="none")
    axes[0].set_ylabel("Transition Rate")
    axes[0].set_title("Hierarchical Transition Rates")
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(rate_values):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

    # Right: within-parent L1 transition matrix
    row_sums = within_parent_trans.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        trans_prob = np.where(row_sums > 0, within_parent_trans / row_sums, 0)
    im = axes[1].imshow(trans_prob, aspect="auto", cmap="Blues", origin="lower")
    axes[1].set_xlabel("L1 To")
    axes[1].set_ylabel("L1 From")
    axes[1].set_title("Within-Parent L1 Transitions (P)")
    plt.colorbar(im, ax=axes[1], label="P(L1_to | L1_from, L0 same)")

    plt.tight_layout()
    return fig, rates, within_parent_trans


# =============================================================================
# Pipeline entry point
# =============================================================================


def run_rvq_analysis(
    results: Sequence[InferenceResult],
    num_codes: int,
    output_dir: Path,
    cfg: dict | None = None,
) -> dict[str, str]:
    """Run all RVQ-specific analyses.

    Skips entirely if no result has ``rvq_indices`` with depth >= 2.

    Args:
        results: Inference results.
        num_codes: Number of codes per depth level.
        output_dir: Directory to save figures.
        cfg: Configuration dict with optional keys:
            parent_child_heatmap, intra_parent_diversity,
            hierarchical_transitions (all default True).

    Returns:
        Mapping from figure name to file path. Empty if skipped.
    """
    cfg = cfg or {}

    # Check if any result has multi-depth indices
    has_rvq = any(
        r.rvq_indices is not None and len(r.rvq_indices) >= 2 for r in results
    )
    if not has_rvq:
        logging.info("  No multi-depth RVQ indices found, skipping RVQ analysis")
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    # 3a. Parent-Child Heatmap
    if cfg.get("parent_child_heatmap", True):
        logging.info("  Computing parent-child heatmap...")
        heatmap_fig, joint_counts = compute_parent_child_heatmap(results, num_codes)
        heatmap_path = output_dir / "parent_child_heatmap.png"
        heatmap_fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
        plt.close(heatmap_fig)
        paths["parent_child_heatmap"] = str(heatmap_path)

        # 3b. Intra-Parent Diversity (depends on joint_counts from 3a)
        if cfg.get("intra_parent_diversity", True):
            logging.info("  Computing intra-parent diversity...")
            div_fig, entropies = compute_intra_parent_diversity(joint_counts)
            div_path = output_dir / "intra_parent_diversity.png"
            div_fig.savefig(div_path, dpi=150, bbox_inches="tight")
            plt.close(div_fig)
            paths["intra_parent_diversity"] = str(div_path)
            active = joint_counts.sum(axis=1) > 0
            logging.info(
                f"    Mean intra-parent entropy: "
                f"{entropies[active].mean():.2f} bits"
                if active.any()
                else "    No active parents"
            )

    # 3c. Hierarchical Transitions
    if cfg.get("hierarchical_transitions", True):
        logging.info("  Computing hierarchical transitions...")
        trans_fig, rates, within_trans = compute_hierarchical_transitions(
            results, num_codes
        )
        trans_path = output_dir / "hierarchical_transitions.png"
        trans_fig.savefig(trans_path, dpi=150, bbox_inches="tight")
        plt.close(trans_fig)
        paths["hierarchical_transitions"] = str(trans_path)
        logging.info(
            f"    L0 rate: {rates['l0_transition_rate']:.3f}, "
            f"L1 rate: {rates['l1_transition_rate']:.3f}, "
            f"L1 within L0: {rates['l1_transition_rate_within_l0']:.3f}"
        )

    logging.info(f"  RVQ analysis complete: {len(paths)} figures saved")
    return paths
