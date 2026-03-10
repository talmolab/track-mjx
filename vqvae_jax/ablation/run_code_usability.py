"""VQ-VAE Code Usability by Posture.

For each starting posture (low-z, high-z), independently classify codes
into preferred / medium / not-preferred based on within-pose frequency
terciles, then:

1. Visually compare categories via decoder-only injection with tabbed HTML
2. Quantitatively measure whether preferred codes produce different
   decoder activity levels than not-preferred codes

Usage:
    cd vqvae_jax
    WANDB_MODE=offline python -m ablation.run_code_usability
"""

import os

# Must set rendering backend before importing MuJoCo
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import base64
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent
VQVAE_DIR = SCRIPT_DIR.parent
REPO_ROOT = VQVAE_DIR.parent
sys.path.insert(0, str(VQVAE_DIR))
sys.path.insert(0, str(REPO_ROOT))

import hydra
import jax
import numpy as np
from absl import logging
from omegaconf import DictConfig
from scipy import stats
from vnl_playground.tasks.rodent import imitation
from vnl_playground.tasks.rodent.reference_clips import ReferenceClips

from track_mjx.config import utils as config_utils

from analysis.checkpoint_utils import get_all_codebooks, load_vq_checkpoint
from analysis.code_analysis import load_rollouts_from_h5
from analysis.inference_cache import InferenceResult
from analysis.rendering import render_rollout_to_video

from .run_ablation import (
    _encode_file_b64,
    make_decoder_only_step_fn,
    plot_code_histogram,
    run_decoder_only_rollout,
    select_starting_clips,
    subset_clips,
)


# =============================================================================
# PER-POSE INDEPENDENT CLASSIFICATION
# =============================================================================


@dataclass
class PoseCodeRanking:
    """Per-pose code ranking based on within-pose frequency terciles.

    Attributes:
        preferred: Codes in the top tercile of frame frequency for this pose.
        medium: Codes in the middle tercile.
        not_preferred: Codes in the bottom tercile (including zero-count).
        frame_counts: Raw frame counts per code in this pose's rollouts.
        frame_fracs: Normalized frame fractions per code.
        rank_order: Code indices sorted by frequency (most frequent first).
    """

    preferred: set[int] = field(default_factory=set)
    medium: set[int] = field(default_factory=set)
    not_preferred: set[int] = field(default_factory=set)
    frame_counts: np.ndarray = field(default_factory=lambda: np.array([]))
    frame_fracs: np.ndarray = field(default_factory=lambda: np.array([]))
    rank_order: np.ndarray = field(default_factory=lambda: np.array([]))


def classify_codes_per_pose(
    rollouts: list[InferenceResult],
    num_codes: int,
    z_low_max: float,
    z_high_min: float,
) -> dict[str, PoseCodeRanking]:
    """Classify codes independently for each pose using frequency terciles.

    For each posture class (low, high), counts how many frames each code
    appears in, then splits codes into terciles: top 1/3 = preferred,
    middle 1/3 = medium, bottom 1/3 = not_preferred.

    Args:
        rollouts: Inference results with code_indices and qpos.
        num_codes: Total number of discrete codes.
        z_low_max: Maximum z-height for "low" posture class.
        z_high_min: Minimum z-height for "high" posture class.

    Returns:
        Dict mapping pose name ("low_height", "high_height") to its
        independent PoseCodeRanking.
    """
    pose_counts: dict[str, np.ndarray] = {
        "low_height": np.zeros(num_codes, dtype=float),
        "high_height": np.zeros(num_codes, dtype=float),
    }
    n_low = 0
    n_high = 0
    n_excluded = 0

    for r in rollouts:
        starting_z = float(r.qpos[0, 2])
        counts = np.bincount(r.code_indices.astype(int), minlength=num_codes)
        if starting_z <= z_low_max:
            pose_counts["low_height"] += counts[:num_codes]
            n_low += 1
        elif starting_z >= z_high_min:
            pose_counts["high_height"] += counts[:num_codes]
            n_high += 1
        else:
            n_excluded += 1

    logging.info(
        f"  Posture split: {n_low} low (z<={z_low_max:.4f}), "
        f"{n_high} high (z>={z_high_min:.4f}), "
        f"{n_excluded} excluded"
    )

    results: dict[str, PoseCodeRanking] = {}
    for pose_name, counts in pose_counts.items():
        total = max(counts.sum(), 1.0)
        fracs = counts / total

        # Rank codes by frequency (descending)
        rank_order = np.argsort(counts)[::-1]

        # Split into terciles
        tercile_size = num_codes // 3
        remainder = num_codes % 3

        # Distribute remainder: preferred gets +1 if remainder >= 1,
        # medium gets +1 if remainder >= 2
        n_preferred = tercile_size + (1 if remainder >= 1 else 0)
        n_medium = tercile_size + (1 if remainder >= 2 else 0)
        # n_not_preferred = num_codes - n_preferred - n_medium

        preferred = set(int(c) for c in rank_order[:n_preferred])
        medium = set(int(c) for c in rank_order[n_preferred:n_preferred + n_medium])
        not_preferred = set(
            int(c) for c in rank_order[n_preferred + n_medium:]
        )

        results[pose_name] = PoseCodeRanking(
            preferred=preferred,
            medium=medium,
            not_preferred=not_preferred,
            frame_counts=counts,
            frame_fracs=fracs,
            rank_order=rank_order,
        )

    return results


# =============================================================================
# ACTIVITY METRICS
# =============================================================================


def compute_code_activity(result: InferenceResult) -> dict[str, float]:
    """Compute activity metrics for a single decoder-only rollout.

    Args:
        result: Rollout result from decoder-only injection.

    Returns:
        Dict with "joint_velocity", "survival", and "displacement".
    """
    # Joint velocity magnitude — exclude root (qvel[:, 6:])
    if len(result.qvel) > 0 and result.qvel.shape[1] > 6:
        joint_vel = np.mean(np.linalg.norm(result.qvel[:, 6:], axis=1))
    else:
        joint_vel = 0.0

    # Survival length
    survival = len(result.rewards)

    # Root displacement
    if len(result.qpos) >= 2:
        displacement = float(
            np.linalg.norm(result.qpos[-1, :2] - result.qpos[0, :2])
        )
    else:
        displacement = 0.0

    return {
        "joint_velocity": float(joint_vel),
        "survival": float(survival),
        "displacement": displacement,
    }


# =============================================================================
# PLOTTING
# =============================================================================

# Color scheme for per-pose categories.
_RANK_COLORS = {
    "preferred": "#4CAF50",
    "medium": "#FFC107",
    "not_preferred": "#F44336",
}

_RANK_LABELS = {
    "preferred": "Preferred",
    "medium": "Medium",
    "not_preferred": "Not Preferred",
}

_RANK_KEYS = ["preferred", "medium", "not_preferred"]


def _code_rank(code: int, ranking: PoseCodeRanking) -> str:
    """Determine the rank category of a code within a pose."""
    if code in ranking.preferred:
        return "preferred"
    elif code in ranking.not_preferred:
        return "not_preferred"
    else:
        return "medium"


def plot_frequency_bars_per_pose(
    ranking: PoseCodeRanking,
    num_codes: int,
    pose_name: str,
    output_path: Path,
) -> str:
    """Bar chart of frame frequency per code, colored by tercile rank.

    Args:
        ranking: Per-pose code ranking.
        num_codes: Total number of codes.
        pose_name: Pose label (e.g. "low_height").
        output_path: Path to save PNG.

    Returns:
        Path to saved figure.
    """
    from matplotlib.patches import Patch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort by frame fraction (descending)
    sort_idx = ranking.rank_order

    colors = [_RANK_COLORS[_code_rank(int(c), ranking)] for c in sort_idx]

    fig, ax = plt.subplots(figsize=(max(8, num_codes * 0.3), 4))
    ax.bar(
        range(num_codes),
        ranking.frame_fracs[sort_idx],
        color=colors,
        edgecolor="none",
    )
    ax.set_xticks(range(num_codes))
    ax.set_xticklabels(
        [str(int(c)) for c in sort_idx], fontsize=7, rotation=45
    )
    ax.set_xlabel("Code Index (sorted by frequency)", fontsize=9)
    ax.set_ylabel("Frame Fraction", fontsize=9)
    ax.set_title(
        f"Code Frequency — {pose_name}", fontsize=10, fontweight="bold"
    )

    legend_handles = [
        Patch(facecolor=_RANK_COLORS[k], label=_RANK_LABELS[k])
        for k in _RANK_KEYS
    ]
    ax.legend(handles=legend_handles, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_usability_scatter(
    activities_low: dict[int, dict[str, float]],
    activities_high: dict[int, dict[str, float]],
    rankings: dict[str, PoseCodeRanking],
    output_path: Path,
) -> str:
    """Scatter plot: joint velocity at low pose vs high pose per code.

    Points colored by combined rank across both poses:
    - "Both Preferred": preferred in both
    - "Both Not Preferred": not_preferred in both
    - "Mixed": different rank in each pose

    Args:
        activities_low: Per-code activity dict from low-height injection.
        activities_high: Per-code activity dict from high-height injection.
        rankings: Per-pose rankings.
        output_path: Path to save PNG.

    Returns:
        Path to saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))

    codes = sorted(activities_low.keys())
    low_r = rankings["low_height"]
    high_r = rankings["high_height"]

    combo_colors = {
        "both_preferred": "#4CAF50",
        "both_not_preferred": "#F44336",
        "mixed": "#9E9E9E",
    }
    combo_labels = {
        "both_preferred": "Preferred in Both",
        "both_not_preferred": "Not Preferred in Both",
        "mixed": "Mixed / Medium",
    }

    groups: dict[str, tuple[list, list, list]] = {
        k: ([], [], []) for k in combo_colors
    }
    for c in codes:
        lr = _code_rank(c, low_r)
        hr = _code_rank(c, high_r)
        if lr == "preferred" and hr == "preferred":
            key = "both_preferred"
        elif lr == "not_preferred" and hr == "not_preferred":
            key = "both_not_preferred"
        else:
            key = "mixed"
        groups[key][0].append(activities_low[c]["joint_velocity"])
        groups[key][1].append(activities_high[c]["joint_velocity"])
        groups[key][2].append(str(c))

    for key, (xs, ys, lbls) in groups.items():
        if not xs:
            continue
        ax.scatter(
            xs,
            ys,
            c=combo_colors[key],
            label=f"{combo_labels[key]} ({len(xs)})",
            s=50,
            alpha=0.8,
            edgecolors="k",
            linewidths=0.5,
        )
        for x, y, lbl in zip(xs, ys, lbls):
            ax.annotate(
                lbl,
                (x, y),
                fontsize=6,
                ha="center",
                va="bottom",
                xytext=(0, 4),
                textcoords="offset points",
            )

    # Diagonal reference line
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "--", color="#aaa", linewidth=0.8, zorder=0)

    ax.set_xlabel("Joint Velocity (Low Height Pose)", fontsize=10)
    ax.set_ylabel("Joint Velocity (High Height Pose)", fontsize=10)
    ax.set_title("Code Activity: Low vs High Starting Pose", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_usability_bars(
    activities: dict[int, dict[str, float]],
    ranking: PoseCodeRanking,
    pose_name: str,
    output_path: Path,
) -> str:
    """Bar chart comparing mean activity across preferred/medium/not_preferred.

    Args:
        activities: Per-code activity metrics.
        ranking: Per-pose code ranking.
        pose_name: Label for the pose (e.g. "low_height").
        output_path: Path to save PNG.

    Returns:
        Path to saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = ["joint_velocity", "survival", "displacement"]
    labels = ["Joint Velocity", "Survival", "Displacement"]

    # Group activities by rank category
    grouped: dict[str, dict[str, list[float]]] = {
        k: {m: [] for m in metrics} for k in _RANK_KEYS
    }
    for c, act in activities.items():
        rank = _code_rank(c, ranking)
        for m in metrics:
            grouped[rank][m].append(act[m])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    x_pos = np.arange(len(_RANK_KEYS))
    bar_labels = [
        f"{_RANK_LABELS[k]}\n(n={len(grouped[k][metrics[0]])})"
        for k in _RANK_KEYS
    ]
    colors = [_RANK_COLORS[k] for k in _RANK_KEYS]

    for ax, metric, label in zip(axes, metrics, labels):
        means = []
        stds = []
        for k in _RANK_KEYS:
            vals = grouped[k][metric]
            arr = np.array(vals) if vals else np.array([0.0])
            means.append(float(np.mean(arr)))
            stds.append(float(np.std(arr)))

        ax.bar(
            x_pos,
            means,
            yerr=stds,
            color=colors,
            capsize=5,
            edgecolor="k",
            linewidth=0.5,
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bar_labels, fontsize=7)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Code Activity by Rank — {pose_name}",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_activity_heatmap(
    activities_low: dict[int, dict[str, float]],
    activities_high: dict[int, dict[str, float]],
    rankings: dict[str, PoseCodeRanking],
    num_codes: int,
    output_path: Path,
) -> str:
    """Heatmap of displacement per code x pose with per-pose rank indicators.

    Each pose column is sorted by that pose's own frequency ranking.
    Side bars show the rank category for each pose independently.

    Args:
        activities_low: Per-code activity from low-height injection.
        activities_high: Per-code activity from high-height injection.
        rankings: Per-pose rankings.
        num_codes: Total number of codes.
        output_path: Path to save PNG.

    Returns:
        Path to saved figure.
    """
    from matplotlib.patches import Patch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build displacement matrix sorted by average rank across poses
    low_r = rankings["low_height"]
    high_r = rankings["high_height"]

    # Sort by average rank position across both poses
    low_rank_pos = {int(c): i for i, c in enumerate(low_r.rank_order)}
    high_rank_pos = {int(c): i for i, c in enumerate(high_r.rank_order)}
    avg_rank = [
        (low_rank_pos.get(c, num_codes) + high_rank_pos.get(c, num_codes)) / 2
        for c in range(num_codes)
    ]
    sort_idx = np.argsort(avg_rank)

    matrix = np.zeros((num_codes, 2))
    for c in range(num_codes):
        matrix[c, 0] = activities_low.get(c, {}).get("displacement", 0.0)
        matrix[c, 1] = activities_high.get(c, {}).get("displacement", 0.0)

    fig, (ax_low_rank, ax_main, ax_high_rank) = plt.subplots(
        1, 3, figsize=(8, max(6, num_codes * 0.25)),
        gridspec_kw={"width_ratios": [0.5, 3, 0.5]},
    )

    im = ax_main.imshow(
        matrix[sort_idx],
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    ax_main.set_xticks([0, 1])
    ax_main.set_xticklabels(["Low Height", "High Height"], fontsize=9)
    ax_main.set_yticks(range(num_codes))
    ax_main.set_yticklabels([str(int(c)) for c in sort_idx], fontsize=7)
    ax_main.set_title("Root Displacement", fontsize=10)
    fig.colorbar(im, ax=ax_main, label="Displacement", shrink=0.6)

    # Left bar: low-height rank
    low_colors = [
        _RANK_COLORS[_code_rank(int(c), low_r)] for c in sort_idx
    ]
    ax_low_rank.barh(
        range(num_codes), [1] * num_codes,
        color=low_colors, edgecolor="none", height=0.8,
    )
    ax_low_rank.set_xlim(0, 1)
    ax_low_rank.set_yticks([])
    ax_low_rank.set_xticks([])
    ax_low_rank.set_title("Low\nRank", fontsize=8)
    ax_low_rank.invert_yaxis()

    # Right bar: high-height rank
    high_colors = [
        _RANK_COLORS[_code_rank(int(c), high_r)] for c in sort_idx
    ]
    ax_high_rank.barh(
        range(num_codes), [1] * num_codes,
        color=high_colors, edgecolor="none", height=0.8,
    )
    ax_high_rank.set_xlim(0, 1)
    ax_high_rank.set_yticks([])
    ax_high_rank.set_xticks([])
    ax_high_rank.set_title("High\nRank", fontsize=8)
    ax_high_rank.invert_yaxis()

    # Legend
    legend_handles = [
        Patch(facecolor=_RANK_COLORS[k], label=_RANK_LABELS[k])
        for k in _RANK_KEYS
    ]
    fig.legend(
        handles=legend_handles, loc="lower center",
        ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


# =============================================================================
# HTML BUILDER FOR FREQUENCY CATEGORIES
# =============================================================================


def build_usability_html(
    ranking: PoseCodeRanking,
    per_code_videos: dict[int, str],
    histogram_path: str | None,
    title: str,
) -> str:
    """Build tabbed HTML with preferred / medium / not-preferred tabs.

    Args:
        ranking: Per-pose code ranking for this specific pose.
        per_code_videos: Mapping from code index to video file path.
        histogram_path: Path to frequency bar chart PNG for this pose.
        title: Page title.

    Returns:
        HTML string.
    """
    categories = {
        "preferred": sorted(ranking.preferred),
        "medium": sorted(ranking.medium),
        "not_preferred": sorted(ranking.not_preferred),
    }

    tab_labels = {
        "preferred": f"Preferred ({len(ranking.preferred)})",
        "medium": f"Medium ({len(ranking.medium)})",
        "not_preferred": f"Not Preferred ({len(ranking.not_preferred)})",
    }

    # Pre-encode all videos
    video_data: dict[int, str] = {}
    for code_idx, path in per_code_videos.items():
        video_data[code_idx] = _encode_file_b64(path, "video/mp4")
    hist_b64 = ""
    if histogram_path and Path(histogram_path).exists():
        hist_b64 = _encode_file_b64(histogram_path, "image/png")

    tab_buttons = []
    tab_contents = []
    for i, (cat, codes) in enumerate(categories.items()):
        label = tab_labels[cat]
        color = _RANK_COLORS[cat]
        active = " active" if i == 0 else ""
        display = "flex" if i == 0 else "none"

        tab_buttons.append(
            f'<button class="tab-btn{active}" '
            f'onclick="showTab(\'{cat}\')" '
            f'style="border-bottom: 3px solid {color}">'
            f"{label}</button>"
        )

        grid_items = []
        for code_idx in codes:
            vid_src = video_data.get(code_idx, "")
            if not vid_src:
                continue
            frac = ranking.frame_fracs[code_idx]
            grid_items.append(
                f'<div class="vid-cell">'
                f'<video src="{vid_src}" width="200" autoplay loop muted></video>'
                f'<div class="vid-label">Code {code_idx} '
                f'(frac={frac:.4f})</div>'
                f"</div>"
            )

        # Show histogram at the bottom of each tab
        hist_img = ""
        if hist_b64:
            hist_img = (
                f'<img src="{hist_b64}" '
                f'style="max-width:100%; margin-top:12px;" />'
            )

        tab_contents.append(
            f'<div class="tab-content" id="tab-{cat}" '
            f'style="display:{display}; flex-wrap:wrap; gap:8px;">'
            f'{"".join(grid_items)}'
            f'<div style="width:100%">{hist_img}</div>'
            f"</div>"
        )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: sans-serif; margin: 16px; background: #fafafa; }}
h2 {{ margin-bottom: 8px; }}
.tab-bar {{ display: flex; gap: 4px; margin-bottom: 12px; }}
.tab-btn {{ padding: 8px 16px; cursor: pointer; background: #eee;
            border: none; border-radius: 4px 4px 0 0; font-size: 13px; }}
.tab-btn.active {{ background: #fff; font-weight: bold; }}
.vid-cell {{ text-align: center; }}
.vid-label {{ font-size: 11px; margin-top: 2px; }}
</style>
<script>
function showTab(cat) {{
  document.querySelectorAll('.tab-content').forEach(
    el => el.style.display = 'none');
  document.querySelectorAll('.tab-btn').forEach(
    el => el.classList.remove('active'));
  document.getElementById('tab-' + cat).style.display = 'flex';
  event.target.classList.add('active');
}}
</script>
</head><body>
<h2>{title}</h2>
<div class="tab-bar">{"".join(tab_buttons)}</div>
{"".join(tab_contents)}
</body></html>"""
    return html


# =============================================================================
# MAIN PIPELINE
# =============================================================================


@hydra.main(
    version_base=None, config_path="../configs", config_name="code_usability"
)
def main(cfg: DictConfig):
    """Run VQ-VAE code usability analysis."""
    logging.set_verbosity(logging.INFO)

    print("=" * 60)
    print("VQ-VAE Code Usability by Posture")
    print("=" * 60)

    output_dir = Path(cfg.output.base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Step 1: Load H5 data and classify clips by starting z-height
    # ================================================================
    logging.info("\nLoading H5 rollout data...")
    h5_path = cfg.data.h5_path
    if not Path(h5_path).exists():
        raise FileNotFoundError(
            f"H5 file not found: {h5_path}\n"
            "Generate rollout data first with:\n"
            "  python -m inference.run_inference checkpoint.path=/path/to/checkpoint"
        )

    rollouts, h5_metadata = load_rollouts_from_h5(h5_path)
    logging.info(f"  Loaded {len(rollouts)} rollouts")

    # Load checkpoint for codebook info and decoder
    logging.info("\nLoading checkpoint...")
    ckpt = load_vq_checkpoint(cfg.checkpoint.path, step=cfg.checkpoint.step)
    vq_cfg = ckpt["cfg"]
    policy_params = ckpt["policy"]

    codebooks = get_all_codebooks(policy_params)
    num_codes = codebooks[0].shape[0]
    logging.info(f"  {num_codes} codes, {len(codebooks)} depth(s)")

    # Compute z-height distribution and thresholds
    z_heights = np.array([float(r.qpos[0, 2]) for r in rollouts])
    pcts = np.percentile(z_heights, [0, 10, 25, 50, 75, 90, 100])
    logging.info(
        f"  Z-height distribution: "
        f"min={pcts[0]:.4f} p10={pcts[1]:.4f} p25={pcts[2]:.4f} "
        f"median={pcts[3]:.4f} p75={pcts[4]:.4f} p90={pcts[5]:.4f} "
        f"max={pcts[6]:.4f}"
    )

    z_split_cfg = cfg.usability.get("z_split", "quartile")
    if z_split_cfg == "median":
        z_low_max = float(np.median(z_heights))
        z_high_min = z_low_max
    elif z_split_cfg == "quartile":
        z_low_max = float(np.percentile(z_heights, 25))
        z_high_min = float(np.percentile(z_heights, 75))
    else:
        # Expect [low_max, high_min] list
        try:
            z_low_max, z_high_min = float(z_split_cfg[0]), float(z_split_cfg[1])
        except (TypeError, IndexError):
            z_low_max = float(z_split_cfg)
            z_high_min = z_low_max

    logging.info(
        f"  Z split mode: {z_split_cfg} -> "
        f"low_max={z_low_max:.4f}, high_min={z_high_min:.4f}"
    )

    # ================================================================
    # Step 2: Per-pose independent code ranking
    # ================================================================
    logging.info("\nClassifying codes independently per pose...")
    rankings = classify_codes_per_pose(
        rollouts, num_codes, z_low_max, z_high_min
    )

    for pose_name, ranking in rankings.items():
        logging.info(f"\n  {pose_name}:")
        logging.info(
            f"    Preferred ({len(ranking.preferred)}): "
            f"{sorted(ranking.preferred)}"
        )
        logging.info(
            f"    Medium ({len(ranking.medium)}): "
            f"{sorted(ranking.medium)}"
        )
        logging.info(
            f"    Not Preferred ({len(ranking.not_preferred)}): "
            f"{sorted(ranking.not_preferred)}"
        )
        # Log per-code frequencies
        for c in ranking.rank_order:
            logging.info(
                f"      Code {int(c):2d}: "
                f"frac={ranking.frame_fracs[int(c)]:.4f}  "
                f"count={ranking.frame_counts[int(c)]:.0f}"
            )

    # ================================================================
    # Step 3: Decoder-only injection for ALL codes from BOTH poses
    # ================================================================
    logging.info("\nSetting up decoder-only injection...")

    # Load reference clips and select starting poses
    (_, cfg_dict, env_cfg_ml) = config_utils.prepare_config(cfg)

    reference_clips = ReferenceClips(
        data_path=vq_cfg.env_config.reference_data_path,
        n_frames_per_clip=vq_cfg.env_config.clip_length,
        keep_clips_idx=vq_cfg.env_config.get("keep_clips_idx", None),
    )
    train_ratio = float(vq_cfg.train_setup.get("train_subset_ratio", 1.0))
    train_seed = int(vq_cfg.train_setup.train_config.get("seed", 0))
    key_split, _ = jax.random.split(jax.random.PRNGKey(train_seed))
    _, test_clips = reference_clips.split(
        train_ratio=train_ratio, seed=key_split
    )

    starting_clips = select_starting_clips(test_clips)

    # Create per-pose single-clip environments
    pose_envs: dict[str, Any] = {}
    for pose_name, clip_idx in starting_clips.items():
        single = subset_clips(test_clips, clip_idx)
        pose_envs[pose_name] = imitation.Imitation(
            config=env_cfg_ml, clips=single
        )
        logging.info(f"  Created env for {pose_name} (clip {clip_idx})")

    # Build decoder-only step function
    decode_step, _ = make_decoder_only_step_fn(vq_cfg, policy_params)
    jit_decode = jax.jit(decode_step)

    max_steps = cfg.usability.max_steps_per_code
    seed = cfg.usability.seed
    render_enabled = cfg.render.get("enabled", True)
    env_suffix = "-rodent"
    camera_name = f"{cfg.render.camera}{env_suffix}"

    # Run injection rollouts for all codes from each pose
    all_activities: dict[str, dict[int, dict[str, float]]] = {}
    all_videos: dict[str, dict[int, str]] = {}

    for pose_name, env in pose_envs.items():
        logging.info(f"\n  === Injecting all codes from {pose_name} ===")
        activities: dict[int, dict[str, float]] = {}
        videos: dict[int, str] = {}

        for code_idx in range(num_codes):
            logging.info(f"    Code {code_idx}/{num_codes}...")
            results = run_decoder_only_rollout(
                env=env,
                jit_decode=jit_decode,
                code_idx=code_idx,
                num_repeats=1,
                max_steps=max_steps,
                seed=seed,
                num_render=1 if render_enabled else 0,
            )
            r = results[0]

            # Compute activity metrics
            activities[code_idx] = compute_code_activity(r)

            # Render video
            if render_enabled and r.states:
                vid_path = (
                    output_dir / f"inject_{pose_name}_c{code_idx}.mp4"
                )
                render_rollout_to_video(
                    env=env,
                    rollout_states=r.states,
                    output_path=vid_path,
                    camera=camera_name,
                    width=cfg.render.width,
                    height=cfg.render.height,
                    fps=cfg.render.fps,
                    indices=r.code_indices,
                    num_codes=num_codes,
                    d0_label=f"D0:{code_idx}",
                )
                videos[code_idx] = str(vid_path)

        all_activities[pose_name] = activities
        all_videos[pose_name] = videos

    # ================================================================
    # Step 4: Visual output — tabbed HTML per pose
    # ================================================================
    logging.info("\nBuilding visual outputs...")

    # Init WandB
    wandb_enabled = False
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("enabled", False):
        try:
            import wandb

            run_name = (
                f"code_usability_{datetime.now().strftime('%y%m%d_%H%M%S')}"
            )
            wandb.init(
                project=wandb_cfg.get("project", "vqvae-eval"),
                entity=wandb_cfg.get("entity"),
                name=run_name,
                config={
                    "checkpoint_path": str(cfg.checkpoint.path),
                    "h5_path": str(cfg.data.h5_path),
                    "z_low_max": z_low_max,
                    "z_high_min": z_high_min,
                    "num_codes": num_codes,
                    "max_steps": max_steps,
                },
            )
            wandb_enabled = True
        except Exception as e:
            logging.warning(f"Failed to init WandB: {e}")

    wandb_items: dict[str, Any] = {}

    for pose_name in pose_envs:
        ranking = rankings[pose_name]

        # Build per-pose frequency bar chart
        freq_bar_path = plot_frequency_bars_per_pose(
            ranking=ranking,
            num_codes=num_codes,
            pose_name=pose_name,
            output_path=output_dir / f"freq_bars_{pose_name}.png",
        )
        logging.info(f"  Frequency bars ({pose_name}): {freq_bar_path}")

        # Build tabbed HTML
        if render_enabled and all_videos.get(pose_name):
            html = build_usability_html(
                ranking=ranking,
                per_code_videos=all_videos[pose_name],
                histogram_path=freq_bar_path,
                title=f"Code Usability — {pose_name}",
            )
            html_path = output_dir / f"usability_{pose_name}.html"
            with open(html_path, "w") as f:
                f.write(html)
            logging.info(f"  Saved HTML: {html_path}")

            if wandb_enabled:
                import wandb

                wandb_items[
                    f"code_usability/{pose_name}/viewer"
                ] = wandb.Html(html)

    # ================================================================
    # Step 5: Quantitative metrics
    # ================================================================
    logging.info("\nComputing quantitative metrics...")

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # --- Cross-pose scatter plot ---
    scatter_path = plot_usability_scatter(
        activities_low=all_activities["low_height"],
        activities_high=all_activities["high_height"],
        rankings=rankings,
        output_path=metrics_dir / "cross_pose_scatter.png",
    )
    logging.info(f"  Scatter plot: {scatter_path}")

    # --- Bar charts per pose ---
    bar_paths: dict[str, str] = {}
    for pose_name in pose_envs:
        bar_path = plot_usability_bars(
            activities=all_activities[pose_name],
            ranking=rankings[pose_name],
            pose_name=pose_name,
            output_path=metrics_dir / f"bars_{pose_name}.png",
        )
        bar_paths[pose_name] = bar_path
        logging.info(f"  Bar chart ({pose_name}): {bar_path}")

    # --- Activity heatmap ---
    heatmap_path = plot_activity_heatmap(
        activities_low=all_activities["low_height"],
        activities_high=all_activities["high_height"],
        rankings=rankings,
        num_codes=num_codes,
        output_path=metrics_dir / "activity_heatmap.png",
    )
    logging.info(f"  Heatmap: {heatmap_path}")

    # --- Statistical tests (preferred vs not_preferred, per pose) ---
    stat_results: dict[str, Any] = {}
    for pose_name in pose_envs:
        activities = all_activities[pose_name]
        ranking = rankings[pose_name]

        pref_jv = [
            activities[c]["joint_velocity"]
            for c in ranking.preferred
            if c in activities
        ]
        not_pref_jv = [
            activities[c]["joint_velocity"]
            for c in ranking.not_preferred
            if c in activities
        ]
        medium_jv = [
            activities[c]["joint_velocity"]
            for c in ranking.medium
            if c in activities
        ]

        pose_stats: dict[str, Any] = {
            "preferred_mean_jv": (
                float(np.mean(pref_jv)) if pref_jv else None
            ),
            "preferred_std_jv": (
                float(np.std(pref_jv)) if pref_jv else None
            ),
            "medium_mean_jv": (
                float(np.mean(medium_jv)) if medium_jv else None
            ),
            "not_preferred_mean_jv": (
                float(np.mean(not_pref_jv)) if not_pref_jv else None
            ),
            "not_preferred_std_jv": (
                float(np.std(not_pref_jv)) if not_pref_jv else None
            ),
            "n_preferred": len(pref_jv),
            "n_medium": len(medium_jv),
            "n_not_preferred": len(not_pref_jv),
        }

        if len(pref_jv) >= 2 and len(not_pref_jv) >= 2:
            u_stat, p_value = stats.mannwhitneyu(
                pref_jv, not_pref_jv, alternative="two-sided"
            )
            pose_stats["mann_whitney_U"] = float(u_stat)
            pose_stats["p_value"] = float(p_value)
            logging.info(
                f"  {pose_name}: preferred_jv={np.mean(pref_jv):.3f} vs "
                f"not_preferred_jv={np.mean(not_pref_jv):.3f} "
                f"(U={u_stat:.1f}, p={p_value:.4f})"
            )
        else:
            pose_stats["note"] = (
                "Insufficient samples for statistical test "
                f"(pref={len(pref_jv)}, not_pref={len(not_pref_jv)})"
            )

        stat_results[pose_name] = pose_stats

    # --- Save full activity table as JSON ---
    activity_table: dict[str, Any] = {}
    for pose_name in pose_envs:
        ranking = rankings[pose_name]
        pose_table = {}
        for c in range(num_codes):
            act = all_activities[pose_name].get(c, {})
            rank = _code_rank(c, ranking)
            pose_table[str(c)] = {
                "joint_velocity": act.get("joint_velocity", 0.0),
                "survival": act.get("survival", 0.0),
                "displacement": act.get("displacement", 0.0),
                "rank": rank,
                "frame_frac": float(ranking.frame_fracs[c]),
                "frame_count": float(ranking.frame_counts[c]),
            }
        activity_table[pose_name] = pose_table

    summary = {
        "z_low_max": z_low_max,
        "z_high_min": z_high_min,
        "num_rollouts": len(rollouts),
        "rankings": {
            pose_name: {
                "preferred": sorted(rankings[pose_name].preferred),
                "medium": sorted(rankings[pose_name].medium),
                "not_preferred": sorted(rankings[pose_name].not_preferred),
            }
            for pose_name in rankings
        },
        "statistical_tests": stat_results,
        "activity_table": activity_table,
    }

    summary_path = metrics_dir / "usability_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logging.info(f"  Summary JSON: {summary_path}")

    # --- Log metrics plots to WandB ---
    if wandb_enabled:
        import wandb

        if scatter_path:
            wandb_items["code_usability/metrics/cross_pose_scatter"] = (
                wandb.Image(scatter_path)
            )
        for pose_name, bp in bar_paths.items():
            if bp:
                wandb_items[f"code_usability/metrics/bars_{pose_name}"] = (
                    wandb.Image(bp)
                )
        if heatmap_path:
            wandb_items["code_usability/metrics/activity_heatmap"] = (
                wandb.Image(heatmap_path)
            )

    # Single WandB log call
    if wandb_enabled:
        import wandb

        if wandb_items and wandb.run is not None:
            wandb.log(wandb_items)
        wandb.finish()

    # ================================================================
    # Done
    # ================================================================
    print("\n" + "=" * 60)
    print(f"Code usability analysis complete! Results saved to {output_dir}")
    print("=" * 60)
    print(f"\nMetrics: {metrics_dir}")
    print(f"Summary: {summary_path}")
    for pose_name in pose_envs:
        html_path = output_dir / f"usability_{pose_name}.html"
        if html_path.exists():
            print(f"HTML viewer ({pose_name}): {html_path}")


if __name__ == "__main__":
    main()
