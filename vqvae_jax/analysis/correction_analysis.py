"""Correction semantics analysis for VQ-VAE codebooks.

Characterizes the codebook as a sparse correction signal where a dominant
"null code" (~80% of frames) means "no correction" and non-null codes fire
in bursts at behavioral transitions.

Analyses:
1. **Kinematic Deltas**: Per-code mean joint angle/velocity change during bursts.
2. **Burst Statistics**: Duration, inter-burst intervals, code co-occurrence.
3. **Correction PCA**: PCA of latent correction vectors (z_q - z_null).

All functions accept ``Sequence[InferenceResult]`` and return figure paths.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .inference_cache import InferenceResult


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class Burst:
    """A contiguous run of non-null codes within a single clip.

    Attributes:
        clip_idx: Source clip index.
        start: Start frame (inclusive).
        end: End frame (exclusive).
        codes: Code indices during the burst, shape [end - start].
        primary_code: Modal L0 code in the burst.
    """

    clip_idx: int
    start: int
    end: int
    codes: np.ndarray = field(repr=False)
    primary_code: int = 0


# =============================================================================
# Utility functions
# =============================================================================


def identify_null_code(results: Sequence[InferenceResult]) -> int:
    """Identify the null (most frequent) code across all results.

    Args:
        results: Inference results with code_indices populated.

    Returns:
        Index of the most frequent code.
    """
    all_codes = np.concatenate([r.code_indices for r in results])
    return int(np.argmax(np.bincount(all_codes)))


def extract_bursts(
    results: Sequence[InferenceResult],
    null_code: int,
) -> list[Burst]:
    """Extract contiguous non-null code runs (bursts) from all clips.

    A burst is a maximal contiguous sequence of frames where
    ``code_indices[t] != null_code``.

    Args:
        results: Inference results.
        null_code: The null code index to exclude.

    Returns:
        List of Burst objects sorted by (clip_idx, start).
    """
    bursts: list[Burst] = []

    for r in results:
        codes = r.code_indices
        mask = codes != null_code  # True where non-null

        if not mask.any():
            continue

        # Find burst boundaries using diff on the boolean mask
        # Pad with False at both ends to detect edges
        padded = np.concatenate([[False], mask, [False]])
        diffs = np.diff(padded.astype(np.int8))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        for s, e in zip(starts, ends):
            burst_codes = codes[s:e]
            # Primary code = mode (most frequent code in burst)
            primary = int(np.argmax(np.bincount(burst_codes)))
            bursts.append(
                Burst(
                    clip_idx=r.clip_idx,
                    start=int(s),
                    end=int(e),
                    codes=burst_codes,
                    primary_code=primary,
                )
            )

    return bursts


# =============================================================================
# Analysis 2a: Kinematic Deltas
# =============================================================================


def compute_correction_deltas(
    results: Sequence[InferenceResult],
    bursts: list[Burst],
    num_codes: int,
    joint_names: list[str] | None = None,
) -> dict[str, str | plt.Figure]:
    """Compute per-code kinematic deltas during bursts.

    For each burst, computes ``delta_qpos = qpos[end-1, 7:] - qpos[start, 7:]``
    (joint angles only, skipping root pos + quat). Groups by primary_code
    and computes mean/std across joints.

    Args:
        results: Inference results with qpos and qvel.
        bursts: Extracted bursts.
        num_codes: Total number of codes.
        joint_names: Optional joint names for axis labels.

    Returns:
        Dict with figure objects keyed by name.
    """
    # Build lookup from clip_idx to result
    result_map = {r.clip_idx: r for r in results}

    # Collect deltas per primary code
    qpos_deltas: dict[int, list[np.ndarray]] = {}
    qvel_deltas: dict[int, list[np.ndarray]] = {}

    for burst in bursts:
        r = result_map.get(burst.clip_idx)
        if r is None:
            continue
        if burst.end - 1 >= r.qpos.shape[0] or burst.start >= r.qpos.shape[0]:
            continue

        # Joint angles: skip root pos (3) + root quat (4) = index 7+
        delta_q = r.qpos[burst.end - 1, 7:] - r.qpos[burst.start, 7:]
        qpos_deltas.setdefault(burst.primary_code, []).append(delta_q)

        # Joint velocities: skip root lin (3) + root ang (3) = index 6+
        if r.qvel is not None:
            delta_v = r.qvel[burst.end - 1, 6:] - r.qvel[burst.start, 6:]
            qvel_deltas.setdefault(burst.primary_code, []).append(delta_v)

    if not qpos_deltas:
        logging.warning("  No bursts with valid qpos data for delta computation")
        return {}

    # Find active codes (codes that have at least one burst)
    active_codes = sorted(qpos_deltas.keys())
    n_all = len(next(iter(qpos_deltas.values()))[0])

    # Build mean/std arrays over all DOFs
    mean_all = np.zeros((len(active_codes), n_all))
    std_all = np.zeros((len(active_codes), n_all))
    for i, code in enumerate(active_codes):
        arr = np.array(qpos_deltas[code])
        mean_all[i] = arr.mean(axis=0)
        std_all[i] = arr.std(axis=0)

    # Restrict to named joints for readable heatmaps.
    # MuJoCo qpos may have more DOFs than named joints (ball joint quaternions).
    if joint_names:
        n_joints = min(len(joint_names), n_all)
        labels = list(joint_names[:n_joints])
        mean_deltas = mean_all[:, :n_joints]
        std_deltas = std_all[:, :n_joints]
    else:
        n_joints = n_all
        labels = [f"q{i}" for i in range(n_joints)]
        mean_deltas = mean_all
        std_deltas = std_all

    figs: dict[str, plt.Figure] = {}

    # --- Qpos delta heatmap ---
    fig, ax = plt.subplots(figsize=(20, max(8, len(active_codes) * 0.4)))
    vmax = max(abs(mean_deltas.min()), abs(mean_deltas.max()), 1e-6)
    im = ax.imshow(mean_deltas, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(active_codes)))
    ax.set_yticklabels([f"Code {c}" for c in active_codes], fontsize=9)
    ax.set_xticks(range(n_joints))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_title("Mean Joint Angle Delta per Code (qpos)", fontsize=13, pad=10)
    plt.colorbar(im, ax=ax, label="Mean delta (rad)", shrink=0.8)
    fig.subplots_adjust(bottom=0.22)
    figs["correction_delta_heatmap"] = fig

    # --- Qvel delta heatmap ---
    if qvel_deltas:
        active_vel_codes = sorted(qvel_deltas.keys())
        n_vel_all = len(next(iter(qvel_deltas.values()))[0])
        mean_vel_all = np.zeros((len(active_vel_codes), n_vel_all))
        for i, code in enumerate(active_vel_codes):
            arr = np.array(qvel_deltas[code])
            mean_vel_all[i] = arr.mean(axis=0)

        # Restrict to named joints
        n_vel = min(n_joints, n_vel_all)
        mean_vel = mean_vel_all[:, :n_vel]

        fig_v, ax_v = plt.subplots(
            figsize=(20, max(8, len(active_vel_codes) * 0.4))
        )
        vmax_v = max(abs(mean_vel.min()), abs(mean_vel.max()), 1e-6)
        im_v = ax_v.imshow(
            mean_vel, aspect="auto", cmap="RdBu_r", vmin=-vmax_v, vmax=vmax_v
        )
        ax_v.set_yticks(range(len(active_vel_codes)))
        ax_v.set_yticklabels([f"Code {c}" for c in active_vel_codes], fontsize=9)
        ax_v.set_xticks(range(n_vel))
        ax_v.set_xticklabels(labels[:n_vel], rotation=45, ha="right", fontsize=9)
        ax_v.set_title("Mean Joint Velocity Delta per Code (qvel)", fontsize=13, pad=10)
        plt.colorbar(im_v, ax=ax_v, label="Mean delta (rad/s)", shrink=0.8)
        fig_v.subplots_adjust(bottom=0.22)
        figs["correction_delta_qvel_heatmap"] = fig_v

    # --- Consistency (coefficient of variation) ---
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(
            np.abs(mean_deltas) > 1e-8,
            std_deltas / np.abs(mean_deltas),
            np.nan,
        )
    # Mean CV per code (across joints, ignoring NaN)
    mean_cv_per_code = np.nanmean(cv, axis=1)

    fig_c, ax_c = plt.subplots(figsize=(max(6, len(active_codes) * 0.3), 4))
    colors = plt.cm.viridis(np.linspace(0, 1, len(active_codes)))
    ax_c.bar(
        range(len(active_codes)),
        mean_cv_per_code,
        color=colors,
        edgecolor="none",
    )
    ax_c.set_xticks(range(len(active_codes)))
    ax_c.set_xticklabels([f"{c}" for c in active_codes], fontsize=7)
    ax_c.set_xlabel("Code Index")
    ax_c.set_ylabel("Mean CV (std / |mean|)")
    ax_c.set_title("Correction Consistency per Code (lower = more consistent)")
    plt.tight_layout()
    figs["correction_consistency"] = fig_c

    return figs


# =============================================================================
# Analysis 2b: Burst Statistics
# =============================================================================


def compute_burst_statistics(
    bursts: list[Burst],
    num_codes: int,
) -> dict[str, plt.Figure | dict]:
    """Compute burst duration, inter-burst intervals, and code co-occurrence.

    Args:
        bursts: Extracted bursts.
        num_codes: Total number of codes.

    Returns:
        Dict with figure objects and stats dict.
    """
    if not bursts:
        logging.warning("  No bursts found for burst statistics")
        return {}

    figs: dict[str, plt.Figure | dict] = {}

    durations = np.array([b.end - b.start for b in bursts])

    # --- Duration distribution ---
    fig_d, ax_d = plt.subplots(figsize=(8, 4))
    max_dur = int(durations.max())
    bins = np.arange(1, min(max_dur + 2, 51))
    ax_d.hist(durations, bins=bins, color="steelblue", edgecolor="white", alpha=0.8)
    ax_d.axvline(
        np.median(durations),
        color="red",
        linestyle="--",
        label=f"Median = {np.median(durations):.0f}",
    )
    ax_d.set_xlabel("Burst Duration (frames)")
    ax_d.set_ylabel("Count")
    ax_d.set_title(f"Burst Duration Distribution (n={len(bursts)})")
    ax_d.legend()
    plt.tight_layout()
    figs["burst_duration_distribution"] = fig_d

    # --- Inter-burst intervals (per clip) ---
    clip_bursts: dict[int, list[Burst]] = {}
    for b in bursts:
        clip_bursts.setdefault(b.clip_idx, []).append(b)

    intervals: list[int] = []
    for clip_idx, cbs in clip_bursts.items():
        sorted_cbs = sorted(cbs, key=lambda x: x.start)
        for i in range(1, len(sorted_cbs)):
            gap = sorted_cbs[i].start - sorted_cbs[i - 1].end
            if gap >= 0:
                intervals.append(gap)

    if intervals:
        intervals_arr = np.array(intervals)
        fig_i, ax_i = plt.subplots(figsize=(8, 4))
        bins_i = np.arange(0, min(int(intervals_arr.max()) + 2, 101))
        ax_i.hist(
            intervals_arr, bins=bins_i, color="coral", edgecolor="white", alpha=0.8
        )
        ax_i.axvline(
            np.median(intervals_arr),
            color="red",
            linestyle="--",
            label=f"Median = {np.median(intervals_arr):.0f}",
        )
        ax_i.set_xlabel("Inter-Burst Interval (frames)")
        ax_i.set_ylabel("Count")
        ax_i.set_title(f"Inter-Burst Interval Distribution (n={len(intervals)})")
        ax_i.legend()
        plt.tight_layout()
        figs["inter_burst_interval"] = fig_i

    # --- Code co-occurrence within bursts ---
    cooccur = np.zeros((num_codes, num_codes), dtype=np.int64)
    for b in bursts:
        unique_codes = np.unique(b.codes)
        for i, c1 in enumerate(unique_codes):
            for c2 in unique_codes[i:]:
                cooccur[c1, c2] += 1
                if c1 != c2:
                    cooccur[c2, c1] += 1

    # Show only active codes
    active_mask = cooccur.sum(axis=0) > 0
    active_idx = np.where(active_mask)[0]
    if len(active_idx) > 1:
        sub_cooccur = cooccur[np.ix_(active_idx, active_idx)]
        fig_co, ax_co = plt.subplots(
            figsize=(max(6, len(active_idx) * 0.35), max(5, len(active_idx) * 0.35))
        )
        im_co = ax_co.imshow(
            np.log1p(sub_cooccur), aspect="auto", cmap="YlOrRd", origin="lower"
        )
        ax_co.set_xticks(range(len(active_idx)))
        ax_co.set_xticklabels(active_idx, fontsize=6, rotation=90)
        ax_co.set_yticks(range(len(active_idx)))
        ax_co.set_yticklabels(active_idx, fontsize=6)
        ax_co.set_xlabel("Code Index")
        ax_co.set_ylabel("Code Index")
        ax_co.set_title("Code Co-occurrence in Bursts (log scale)")
        plt.colorbar(im_co, ax=ax_co, label="log(count + 1)")
        plt.tight_layout()
        figs["code_cooccurrence_matrix"] = fig_co

    # --- Stats JSON data ---
    per_code_durations: dict[str, list[int]] = {}
    for b in bursts:
        key = str(b.primary_code)
        per_code_durations.setdefault(key, []).append(b.end - b.start)

    stats = {
        "total_bursts": len(bursts),
        "mean_duration": float(durations.mean()),
        "median_duration": float(np.median(durations)),
        "max_duration": int(durations.max()),
        "mean_inter_burst_interval": (float(np.mean(intervals)) if intervals else None),
        "num_active_codes": len(per_code_durations),
        "per_code_burst_count": {k: len(v) for k, v in per_code_durations.items()},
        "per_code_mean_duration": {
            k: float(np.mean(v)) for k, v in per_code_durations.items()
        },
    }
    figs["burst_stats"] = stats

    return figs


# =============================================================================
# Analysis 2c: Correction PCA
# =============================================================================


def compute_correction_pca(
    results: Sequence[InferenceResult],
    bursts: list[Burst],
    codebooks: list[np.ndarray],
    null_code: int,
) -> dict[str, plt.Figure]:
    """PCA of latent correction vectors (z_q - z_null).

    For depth >= 2, z_null uses the most common L1 code when L0 is null.
    Correction vectors are computed for each non-null timestep.

    Args:
        results: Inference results.
        bursts: Extracted bursts.
        codebooks: List of codebook arrays, one per depth level.
        null_code: The null (most frequent) L0 code index.

    Returns:
        Dict of figure objects keyed by name.
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        logging.warning("  sklearn not available, skipping correction PCA")
        return {}

    if not codebooks or len(codebooks) == 0:
        return {}

    result_map = {r.clip_idx: r for r in results}
    depth = len(codebooks)

    # Determine z_null
    cb0 = codebooks[0]
    z_null = cb0[null_code].copy()

    if depth >= 2:
        cb1 = codebooks[1]
        # Find most common L1 when L0 == null_code
        l1_counts = np.zeros(cb1.shape[0], dtype=np.int64)
        for r in results:
            if r.rvq_indices is None or len(r.rvq_indices) < 2:
                continue
            l0 = r.rvq_indices[0]
            l1 = r.rvq_indices[1]
            null_mask = l0 == null_code
            for idx in l1[null_mask]:
                l1_counts[int(idx)] += 1
        if l1_counts.sum() > 0:
            null_l1 = int(np.argmax(l1_counts))
            z_null = z_null + cb1[null_l1]

    # Collect correction vectors and metadata for non-null timesteps in bursts
    corrections: list[np.ndarray] = []
    code_labels: list[int] = []
    delta_norms: list[float] = []
    burst_durations: list[int] = []

    for burst in bursts:
        r = result_map.get(burst.clip_idx)
        if r is None:
            continue

        duration = burst.end - burst.start
        # Compute qpos delta norm for this burst
        if burst.end - 1 < r.qpos.shape[0] and burst.start < r.qpos.shape[0]:
            d_norm = float(
                np.linalg.norm(r.qpos[burst.end - 1, 7:] - r.qpos[burst.start, 7:])
            )
        else:
            d_norm = 0.0

        for t in range(burst.start, burst.end):
            if t >= len(r.code_indices):
                break
            l0 = int(r.code_indices[t])
            z_q = cb0[l0].copy()

            if depth >= 2 and r.rvq_indices is not None and len(r.rvq_indices) >= 2:
                l1 = int(r.rvq_indices[1][t]) if t < len(r.rvq_indices[1]) else 0
                z_q = z_q + codebooks[1][l1]

            corrections.append(z_q - z_null)
            code_labels.append(l0)
            delta_norms.append(d_norm)
            burst_durations.append(duration)

    if len(corrections) < 3:
        logging.warning("  Too few correction vectors for PCA (need >= 3)")
        return {}

    X = np.array(corrections)
    code_labels_arr = np.array(code_labels)
    delta_norms_arr = np.array(delta_norms)
    burst_durations_arr = np.array(burst_durations)

    # PCA
    n_components = min(2, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    if n_components < 2:
        logging.warning("  PCA needs at least 2 components, skipping")
        return {}

    var_explained = pca.explained_variance_ratio_

    figs: dict[str, plt.Figure] = {}

    # --- PCA colored by L0 code ---
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    unique_codes = np.unique(code_labels_arr)
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(unique_codes), 1)))
    for i, code in enumerate(unique_codes):
        mask = code_labels_arr == code
        ax1.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=[cmap[i % len(cmap)]],
            label=f"Code {code}",
            s=8,
            alpha=0.6,
        )
    ax1.set_xlabel(f"PC1 ({var_explained[0]:.1%} var)")
    ax1.set_ylabel(f"PC2 ({var_explained[1]:.1%} var)")
    ax1.set_title("Correction PCA (colored by L0 code)")
    if len(unique_codes) <= 20:
        ax1.legend(fontsize=6, markerscale=2, ncol=2)
    plt.tight_layout()
    figs["correction_pca"] = fig1

    # --- PCA colored by delta norm ---
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sc2 = ax2.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=delta_norms_arr,
        cmap="plasma",
        s=8,
        alpha=0.6,
    )
    ax2.set_xlabel(f"PC1 ({var_explained[0]:.1%} var)")
    ax2.set_ylabel(f"PC2 ({var_explained[1]:.1%} var)")
    ax2.set_title("Correction PCA (colored by ||delta_qpos||)")
    plt.colorbar(sc2, ax=ax2, label="||delta_qpos||")
    plt.tight_layout()
    figs["correction_pca_by_delta"] = fig2

    # --- PCA colored by burst duration ---
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sc3 = ax3.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=burst_durations_arr,
        cmap="viridis",
        s=8,
        alpha=0.6,
    )
    ax3.set_xlabel(f"PC1 ({var_explained[0]:.1%} var)")
    ax3.set_ylabel(f"PC2 ({var_explained[1]:.1%} var)")
    ax3.set_title("Correction PCA (colored by burst duration)")
    plt.colorbar(sc3, ax=ax3, label="Burst duration (frames)")
    plt.tight_layout()
    figs["correction_pca_by_duration"] = fig3

    return figs


# =============================================================================
# Pipeline entry point
# =============================================================================


def run_correction_analysis(
    results: Sequence[InferenceResult],
    num_codes: int,
    output_dir: Path,
    codebooks: list[np.ndarray] | None = None,
    joint_names: list[str] | None = None,
    cfg: dict | None = None,
) -> dict[str, str]:
    """Run correction semantics analysis pipeline.

    Args:
        results: Inference results.
        num_codes: Number of codes in the codebook.
        output_dir: Directory to save figures and JSON.
        codebooks: List of codebook arrays (one per RVQ depth).
            Required for correction PCA.
        joint_names: Optional joint names for axis labels.
        cfg: Configuration dict with optional keys:
            kinematic_deltas (default True), burst_statistics (default True),
            correction_pca (default True).

    Returns:
        Mapping from figure name to file path.
    """
    cfg = cfg or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    # Identify null code and extract bursts
    null_code = identify_null_code(results)
    bursts = extract_bursts(results, null_code)

    total_frames = sum(len(r.code_indices) for r in results)
    null_frames = sum(np.sum(r.code_indices == null_code) for r in results)
    null_pct = null_frames / max(total_frames, 1) * 100

    logging.info(
        f"  Null code: {null_code} ({null_pct:.1f}% of frames), "
        f"{len(bursts)} bursts extracted"
    )

    # 2a. Kinematic Deltas
    if cfg.get("kinematic_deltas", True):
        logging.info("  Computing kinematic deltas...")
        delta_figs = compute_correction_deltas(results, bursts, num_codes, joint_names)
        for name, fig in delta_figs.items():
            fig_path = output_dir / f"{name}.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            paths[name] = str(fig_path)

    # 2b. Burst Statistics
    if cfg.get("burst_statistics", True):
        logging.info("  Computing burst statistics...")
        burst_results = compute_burst_statistics(bursts, num_codes)
        for name, obj in burst_results.items():
            if name == "burst_stats":
                # Save JSON
                json_path = output_dir / "burst_stats.json"
                with open(json_path, "w") as f:
                    json.dump(obj, f, indent=2, default=str)
                paths["burst_stats_json"] = str(json_path)
            elif isinstance(obj, plt.Figure):
                fig_path = output_dir / f"{name}.png"
                obj.savefig(fig_path, dpi=150, bbox_inches="tight")
                plt.close(obj)
                paths[name] = str(fig_path)

    # 2c. Correction PCA
    if cfg.get("correction_pca", True) and codebooks:
        logging.info("  Computing correction PCA...")
        pca_figs = compute_correction_pca(results, bursts, codebooks, null_code)
        for name, fig in pca_figs.items():
            fig_path = output_dir / f"{name}.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            paths[name] = str(fig_path)

    logging.info(f"  Correction analysis complete: {len(paths)} outputs saved")
    return paths
