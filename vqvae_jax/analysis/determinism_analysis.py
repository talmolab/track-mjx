"""Code determinism analysis for VQ-VAE.

Empirically tests whether the VQ-VAE encoder is deterministic with respect
to body state: for a given qpos, the same code should be selected regardless
of which clip the frame comes from.

Produces a single figure (``code_determinism.png``) showing code agreement
rate vs qpos L2 distance for cross-clip frame pairs.
"""

import logging
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .inference_cache import InferenceResult


def _plot_determinism_curve(
    bin_centers: np.ndarray,
    agreement_rates: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    counts: np.ndarray,
    chance_level: float,
    determinism_score: float,
    robustness_radius: float,
    output_path: Path,
) -> str:
    """Plot code agreement rate vs qpos distance.

    Args:
        bin_centers: Center of each distance bin.
        agreement_rates: Agreement rate per bin.
        ci_lower: Lower bound of 95% CI per bin.
        ci_upper: Upper bound of 95% CI per bin.
        counts: Number of pairs per bin.
        chance_level: Expected agreement rate by chance (1/num_codes).
        determinism_score: Agreement rate at smallest distance bin.
        robustness_radius: Distance where agreement drops below 50%.
        output_path: Path to save the figure.

    Returns:
        String path to the saved figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Main curve
    mask = counts > 0
    ax.plot(
        bin_centers[mask],
        agreement_rates[mask],
        color="#2196F3",
        linewidth=2,
        label="Agreement rate",
    )
    ax.fill_between(
        bin_centers[mask],
        ci_lower[mask],
        ci_upper[mask],
        color="#2196F3",
        alpha=0.2,
        label="95% CI",
    )

    # Chance level
    ax.axhline(
        y=chance_level,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"Chance level (1/{int(1/chance_level)})",
    )

    # Robustness radius marker
    if np.isfinite(robustness_radius):
        ax.axvline(
            x=robustness_radius,
            color="#FF9800",
            linestyle=":",
            linewidth=1.5,
            label=f"50% threshold (r={robustness_radius:.2f})",
        )

    # Annotations
    ax.annotate(
        f"Determinism score: {determinism_score:.1%}",
        xy=(0.02, 0.95),
        xycoords="axes fraction",
        fontsize=11,
        fontweight="bold",
        va="top",
    )
    radius_text = (
        f"{robustness_radius:.2f}" if np.isfinite(robustness_radius) else "N/A"
    )
    ax.annotate(
        f"Robustness radius: {radius_text}",
        xy=(0.02, 0.89),
        xycoords="axes fraction",
        fontsize=11,
        va="top",
    )

    ax.set_xlabel("Qpos L2 Distance", fontsize=12)
    ax.set_ylabel("Code Agreement Rate", fontsize=12)
    ax.set_title("Code Determinism: Agreement Rate vs Qpos Distance", fontsize=14)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(output_path)


def run_determinism_analysis(
    results: Sequence[InferenceResult],
    num_codes: int,
    output_dir: Path,
    cfg: dict | None = None,
) -> dict[str, str]:
    """Analyze VQ-VAE code determinism across clips.

    Samples random cross-clip frame pairs, bins them by qpos L2 distance,
    and computes code agreement rate per bin. A deterministic encoder should
    show high agreement for frames with similar qpos, decaying with distance.

    Args:
        results: Sequence of InferenceResult from different clips.
        num_codes: Total number of VQ-VAE codes.
        output_dir: Directory to save output figure.
        cfg: Optional config dict with keys:
            - n_sample_pairs: Number of random pairs to sample (default 1M).
            - n_bins: Number of distance bins (default 50).

    Returns:
        Dict mapping figure name to file path.
    """
    if cfg is None:
        cfg = {}

    n_sample_pairs = cfg.get("n_sample_pairs", 1_000_000)
    n_bins = cfg.get("n_bins", 50)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Collect all (qpos, code, clip_idx) tuples ---
    all_qpos_list = []
    all_codes_list = []
    all_clips_list = []

    for result in results:
        n_frames = len(result.code_indices)
        all_qpos_list.append(result.qpos[:n_frames])
        all_codes_list.append(result.code_indices[:n_frames])
        all_clips_list.append(np.full(n_frames, result.clip_idx, dtype=np.int32))

    all_qpos = np.concatenate(all_qpos_list, axis=0)
    all_codes = np.concatenate(all_codes_list, axis=0)
    all_clips = np.concatenate(all_clips_list, axis=0)
    n_total = len(all_codes)

    logging.info(f"  Total frames: {n_total}")
    logging.info(f"  Unique clips: {len(np.unique(all_clips))}")

    # --- Step 2: Sample random cross-clip pairs ---
    rng = np.random.default_rng(42)
    idx_i = rng.integers(0, n_total, size=n_sample_pairs)
    idx_j = rng.integers(0, n_total, size=n_sample_pairs)

    # Filter to cross-clip pairs
    cross_clip_mask = all_clips[idx_i] != all_clips[idx_j]
    idx_i = idx_i[cross_clip_mask]
    idx_j = idx_j[cross_clip_mask]
    n_cross = len(idx_i)

    logging.info(f"  Cross-clip pairs: {n_cross} / {n_sample_pairs} sampled")

    if n_cross == 0:
        logging.warning("  No cross-clip pairs found. Skipping determinism analysis.")
        return {}

    # --- Step 3: Compute distance and agreement ---
    dist = np.linalg.norm(all_qpos[idx_i] - all_qpos[idx_j], axis=1)
    agree = all_codes[idx_i] == all_codes[idx_j]

    # --- Step 4: Bin by distance ---
    bin_edges = np.histogram_bin_edges(dist, bins=n_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_indices = np.digitize(dist, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    agreement_rates = np.zeros(n_bins)
    ci_lower = np.zeros(n_bins)
    ci_upper = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=np.int64)

    for b in range(n_bins):
        mask = bin_indices == b
        n_b = mask.sum()
        counts[b] = n_b
        if n_b > 0:
            rate = agree[mask].mean()
            agreement_rates[b] = rate
            # Binomial 95% CI
            se = np.sqrt(rate * (1 - rate) / n_b) if n_b > 1 else 0.0
            ci_lower[b] = max(0.0, rate - 1.96 * se)
            ci_upper[b] = min(1.0, rate + 1.96 * se)

    # --- Step 5: Compute summary stats ---
    chance_level = 1.0 / num_codes

    # Determinism score: agreement at smallest bin with data
    valid_bins = counts > 0
    if valid_bins.any():
        first_valid = np.argmax(valid_bins)
        determinism_score = agreement_rates[first_valid]
    else:
        determinism_score = 0.0

    # Robustness radius: distance where agreement drops below 50%
    below_50 = (agreement_rates < 0.5) & valid_bins
    if below_50.any():
        robustness_radius = bin_centers[np.argmax(below_50)]
    else:
        robustness_radius = float("inf")

    logging.info(f"  Determinism score: {determinism_score:.1%}")
    logging.info(
        f"  Robustness radius: " f"{robustness_radius:.2f}"
        if np.isfinite(robustness_radius)
        else "N/A"
    )
    logging.info(f"  Chance level: {chance_level:.4f}")

    # --- Step 6: Plot ---
    output_path = output_dir / "code_determinism.png"
    fig_path = _plot_determinism_curve(
        bin_centers=bin_centers,
        agreement_rates=agreement_rates,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        counts=counts,
        chance_level=chance_level,
        determinism_score=determinism_score,
        robustness_radius=robustness_radius,
        output_path=output_path,
    )

    return {"code_determinism": fig_path}
