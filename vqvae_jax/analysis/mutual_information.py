"""Mutual information analysis for VQ-VAE codes.

Quantifies what behavioral features the codebook encodes by computing
mutual information between code assignments and kinematic/postural features.

Produces three WandB panels under ``mutual_information/``:
- ``mi_ranking``: Horizontal bar chart of MI(code; feature).
- ``feature_code_heatmap``: Z-scored mean feature per code, ordered by MI.
- ``code_feature_scatter``: Codes in top-2 MI feature space.
"""

import logging
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif

from .inference_cache import InferenceResult
from .kinematic_analysis import extract_kinematic_features


# ── Joint group detection ────────────────────────────────────────────────────


def _derive_joint_groups(joint_names: list[str]) -> dict[str, list[int]]:
    """Derive limb joint groups from joint names.

    Maps joint names to qvel indices. Joint velocities start at qvel[6:]
    so joint index ``i`` in the joint list maps to qvel index ``6 + i``.

    Args:
        joint_names: Ordered list of joint names from walker config.

    Returns:
        Mapping from limb name to list of qvel indices.
    """
    prefixes = {
        "hindlimb_L": ["hip_L", "knee_L", "ankle_L", "toe_L"],
        "hindlimb_R": ["hip_R", "knee_R", "ankle_R", "toe_R"],
        "forelimb_L": [
            "scapula_L",
            "shoulder_L",
            "shoulder_sup_L",
            "elbow_L",
            "wrist_L",
        ],
        "forelimb_R": [
            "scapula_R",
            "shoulder_R",
            "shoulder_sup_R",
            "elbow_R",
            "wrist_R",
            "finger_R",
        ],
    }

    groups: dict[str, list[int]] = {}
    for limb, pfx_list in prefixes.items():
        indices = []
        for i, name in enumerate(joint_names):
            if any(name.startswith(p) for p in pfx_list):
                indices.append(6 + i)  # offset by root DOFs
        if indices:
            groups[limb] = indices

    return groups


# ── Feature extraction ───────────────────────────────────────────────────────


def extract_extended_features(
    results: Sequence[InferenceResult],
    joint_groups: dict[str, list[int]],
    n_posture_pcs: int = 3,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract extended kinematic features aligned with code assignments.

    Args:
        results: Inference results with qpos/qvel and code_indices.
        joint_groups: Limb name to qvel indices mapping.
        n_posture_pcs: Number of posture PCA components.

    Returns:
        Tuple of (features [N, F], codes [N], feature_names [F]).
    """
    all_features: list[list[float]] = []
    all_codes: list[int] = []
    all_joint_angles: list[np.ndarray] = []

    for result in results:
        if result.qpos is None or result.qvel is None:
            continue

        kin = extract_kinematic_features(result.qpos, result.qvel)
        T = len(result.code_indices)

        for t in range(T):
            qvel_t = result.qvel[t]

            row = [
                float(kin.linear_velocity[t]),
                float(kin.angular_velocity[t]),
                float(kin.body_height[t]),
                float(kin.joint_velocities[t]),
                (
                    float(np.arctan2(qvel_t[1], qvel_t[0]))
                    if qvel_t.shape[0] >= 2
                    else 0.0
                ),
            ]

            # Limb activities
            for limb in ["hindlimb_L", "hindlimb_R", "forelimb_L", "forelimb_R"]:
                if limb in joint_groups:
                    idx = joint_groups[limb]
                    valid = [j for j in idx if j < qvel_t.shape[0]]
                    row.append(float(np.mean(np.abs(qvel_t[valid]))) if valid else 0.0)
                else:
                    row.append(0.0)

            # Acceleration (norm of velocity difference)
            if t > 0 and result.qvel.shape[1] >= 3:
                acc = np.linalg.norm(result.qvel[t, :3] - result.qvel[t - 1, :3])
            else:
                acc = 0.0
            row.append(float(acc))

            all_features.append(row)
            all_codes.append(int(result.code_indices[t]))
            all_joint_angles.append(result.qpos[t, 7:])

    if not all_features:
        return np.empty((0, 0)), np.empty(0, dtype=int), []

    features = np.array(all_features, dtype=np.float64)
    codes = np.array(all_codes, dtype=int)

    base_names = [
        "linear_velocity",
        "angular_velocity",
        "body_height",
        "joint_velocities",
        "heading_direction",
        "hindlimb_L_activity",
        "hindlimb_R_activity",
        "forelimb_L_activity",
        "forelimb_R_activity",
        "acceleration",
    ]

    # Posture PCA
    joint_angles = np.array(all_joint_angles, dtype=np.float64)
    n_pcs = min(n_posture_pcs, joint_angles.shape[1], joint_angles.shape[0])
    if n_pcs > 0:
        pca = PCA(n_components=n_pcs)
        posture_pcs = pca.fit_transform(joint_angles)
        features = np.hstack([features, posture_pcs])
        base_names.extend([f"posture_PC{i + 1}" for i in range(n_pcs)])

    return features, codes, base_names


# ── MI computation ───────────────────────────────────────────────────────────


def compute_mutual_information(
    features: np.ndarray,
    codes: np.ndarray,
    n_neighbors: int = 5,
) -> np.ndarray:
    """Compute mutual information between features and code assignments.

    Uses the KSG estimator via sklearn's ``mutual_info_classif``.

    Args:
        features: Feature matrix, shape [N, F].
        codes: Code assignments, shape [N].
        n_neighbors: Number of neighbours for the KSG estimator.

    Returns:
        MI scores in nats, shape [F].
    """
    return mutual_info_classif(
        features,
        codes,
        discrete_features=False,
        n_neighbors=n_neighbors,
        random_state=42,
    )


# ── Cross-depth MI ────────────────────────────────────────────────────────────


def _extract_leaf_codes(
    results: Sequence[InferenceResult],
    num_codes: int,
) -> np.ndarray | None:
    """Extract composite leaf codes (L0 * K + L1) for each frame.

    Args:
        results: Inference results with ``rvq_indices``.
        num_codes: Number of codes per depth level.

    Returns:
        Array of composite leaf codes, shape [N], or None if no RVQ data.
    """
    leaf_codes: list[int] = []
    for r in results:
        if r.rvq_indices is None or len(r.rvq_indices) < 2:
            return None
        l0 = r.rvq_indices[0]
        l1 = r.rvq_indices[1]
        T = min(len(l0), len(l1))
        for t in range(T):
            leaf_codes.append(int(l0[t]) * num_codes + int(l1[t]))
    return np.array(leaf_codes, dtype=int) if leaf_codes else None


def _make_null_leaf_codes(
    l0_codes: np.ndarray,
    num_codes: int,
    seed: int = 0,
) -> np.ndarray:
    """Create null-baseline leaf codes by randomly assigning L1 within each L0.

    Preserves L0 structure and cardinality (K^2 labels) but destroys any
    real L1 structure, providing a baseline for the MI cardinality effect.

    Args:
        l0_codes: L0 code assignments, shape [N].
        num_codes: Number of codes per depth level.
        seed: Random seed for reproducibility.

    Returns:
        Null leaf codes, shape [N].
    """
    rng = np.random.RandomState(seed)
    random_l1 = rng.randint(0, num_codes, size=len(l0_codes))
    return l0_codes * num_codes + random_l1


def compute_cross_depth_mi(
    features: np.ndarray,
    l0_codes: np.ndarray,
    leaf_codes: np.ndarray,
    feature_names: list[str],
    output_path: Path,
    n_neighbors: int = 5,
    num_codes: int = 32,
    n_null_shuffles: int = 3,
) -> str:
    """Compute MI(L0; features) vs MI(leaf; features) with null baseline.

    Compares three quantities per feature:
    - MI(L0; feature): information from L0 alone
    - MI(null_leaf; feature): null baseline with random L1 within each L0
      (same K^2 cardinality, no real L1 structure)
    - MI(leaf; feature): real composite label

    The gain above null = MI(leaf) - MI(null_leaf) isolates the true
    contribution of L1, controlling for the cardinality inflation.

    Args:
        features: Feature matrix, shape [N, F].
        l0_codes: L0 code assignments, shape [N].
        leaf_codes: Composite (L0*K + L1) code assignments, shape [N].
        feature_names: Names of each feature dimension.
        output_path: Path to save the figure.
        n_neighbors: KSG estimator neighbours.
        num_codes: Number of codes per depth level (for null baseline).
        n_null_shuffles: Number of random shuffles to average for null.

    Returns:
        Path to the saved figure.
    """
    logging.info("    Computing MI(L0; features)...")
    mi_l0 = compute_mutual_information(features, l0_codes, n_neighbors) / np.log(2)

    logging.info("    Computing MI(leaf; features)...")
    mi_leaf = compute_mutual_information(features, leaf_codes, n_neighbors) / np.log(2)

    # Null baseline: average over multiple random L1 shuffles
    logging.info(f"    Computing null baseline ({n_null_shuffles} shuffles)...")
    mi_null_runs = []
    for s in range(n_null_shuffles):
        null_codes = _make_null_leaf_codes(l0_codes, num_codes, seed=s)
        mi_null_s = compute_mutual_information(
            features, null_codes, n_neighbors
        ) / np.log(2)
        mi_null_runs.append(mi_null_s)
    mi_null = np.mean(mi_null_runs, axis=0)

    # Sort by gain above null
    gain_above_null = mi_leaf - mi_null
    order = np.argsort(gain_above_null)[::-1]

    fig, ax = plt.subplots(figsize=(10, max(5, 0.6 * len(feature_names))))
    y_pos = np.arange(len(feature_names))
    bar_height = 0.25

    sorted_names = [feature_names[i] for i in order]
    ax.barh(
        y_pos + bar_height,
        mi_l0[order],
        bar_height,
        label="MI(L0; feature)",
        color="#2196F3",
        alpha=0.8,
    )
    ax.barh(
        y_pos,
        mi_null[order],
        bar_height,
        label="MI(null leaf; feature)",
        color="#9E9E9E",
        alpha=0.7,
    )
    ax.barh(
        y_pos - bar_height,
        mi_leaf[order],
        bar_height,
        label="MI(real leaf; feature)",
        color="#FF9800",
        alpha=0.8,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel("Mutual Information (bits)")
    ax.set_title("Cross-Depth MI: L0 vs Null Leaf vs Real Leaf")
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


# ── Plotting ─────────────────────────────────────────────────────────────────


def _plot_mi_ranking(
    mi_bits: np.ndarray,
    feature_names: list[str],
    output_path: Path,
) -> str:
    """Horizontal bar chart of MI(code; feature), sorted descending."""
    order = np.argsort(mi_bits)
    sorted_mi = mi_bits[order]
    sorted_names = [feature_names[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(feature_names))))
    bars = ax.barh(
        range(len(sorted_mi)), sorted_mi, color="steelblue", edgecolor="white"
    )
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel("Mutual Information (bits)")
    ax.set_title("MI(code; feature)")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _plot_feature_code_heatmap(
    features: np.ndarray,
    codes: np.ndarray,
    mi_bits: np.ndarray,
    feature_names: list[str],
    num_codes: int,
    output_path: Path,
    min_mi_threshold: float = 0.05,
    top_k_features: int = 8,
    community_labels: np.ndarray | None = None,
) -> str:
    """Heatmap of z-scored mean feature value per code, rows by MI ranking."""
    # Filter features above threshold, then take top-k
    above_thresh = np.where(mi_bits > min_mi_threshold)[0]
    if len(above_thresh) == 0:
        above_thresh = np.argsort(mi_bits)[::-1][:top_k_features]
    else:
        rank = np.argsort(mi_bits[above_thresh])[::-1][:top_k_features]
        above_thresh = above_thresh[rank]

    selected_names = [feature_names[i] for i in above_thresh]

    # Mean feature per code
    data = np.zeros((len(above_thresh), num_codes))
    for ci in range(num_codes):
        mask = codes == ci
        if mask.sum() > 0:
            data[:, ci] = features[mask][:, above_thresh].mean(axis=0)

    # Z-score rows
    data_z = scipy_stats.zscore(data, axis=1)
    data_z = np.nan_to_num(data_z)

    # Order codes by community if available
    if community_labels is not None and len(community_labels) == num_codes:
        code_order = np.argsort(community_labels)
    else:
        code_order = np.arange(num_codes)

    fig, ax = plt.subplots(
        figsize=(max(8, num_codes * 0.15), max(4, len(selected_names) * 0.5))
    )
    im = ax.imshow(data_z[:, code_order], aspect="auto", cmap="RdBu_r")
    ax.set_yticks(range(len(selected_names)))
    ax.set_yticklabels(selected_names)
    ax.set_xlabel("Code Index (ordered by community)")
    ax.set_title("Mean Feature per Code (z-scored, ordered by MI)")
    plt.colorbar(im, ax=ax, label="Z-score")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _plot_code_feature_scatter(
    features: np.ndarray,
    codes: np.ndarray,
    mi_bits: np.ndarray,
    feature_names: list[str],
    num_codes: int,
    output_path: Path,
    community_labels: np.ndarray | None = None,
) -> str:
    """Scatter of codes in top-2 MI feature space."""
    top2 = np.argsort(mi_bits)[::-1][:2]
    if len(top2) < 2:
        return ""

    # Compute mean feature value per code
    code_means = np.zeros((num_codes, 2))
    code_counts = np.zeros(num_codes)
    for ci in range(num_codes):
        mask = codes == ci
        n = mask.sum()
        code_counts[ci] = n
        if n > 0:
            code_means[ci, 0] = features[mask, top2[0]].mean()
            code_means[ci, 1] = features[mask, top2[1]].mean()

    active = code_counts > 0
    if active.sum() < 2:
        return ""

    fig, ax = plt.subplots(figsize=(8, 7))

    sizes = np.clip(code_counts[active] / code_counts[active].max() * 300, 20, 300)

    if community_labels is not None and len(community_labels) == num_codes:
        colors = community_labels[active].astype(float)
        scatter = ax.scatter(
            code_means[active, 0],
            code_means[active, 1],
            c=colors,
            s=sizes,
            cmap="tab10",
            alpha=0.8,
            edgecolors="white",
            linewidth=0.5,
        )
        plt.colorbar(scatter, ax=ax, label="Community")
    else:
        ax.scatter(
            code_means[active, 0],
            code_means[active, 1],
            s=sizes,
            c="steelblue",
            alpha=0.8,
            edgecolors="white",
            linewidth=0.5,
        )

    # Label points
    active_indices = np.where(active)[0]
    for idx, (x, y) in zip(active_indices, code_means[active]):
        ax.annotate(str(idx), (x, y), fontsize=7, ha="center", va="center")

    ax.set_xlabel(feature_names[top2[0]])
    ax.set_ylabel(feature_names[top2[1]])
    ax.set_title("Codes in Top-2 MI Feature Space")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


# ── Pipeline entry point ─────────────────────────────────────────────────────


def run_mutual_information_analysis(
    results: Sequence[InferenceResult],
    num_codes: int,
    output_dir: Path,
    joint_names: list[str] | None = None,
    community_labels: np.ndarray | None = None,
    cfg: dict | None = None,
) -> dict[str, str]:
    """Run mutual information analysis and produce figures.

    Args:
        results: Inference results with qpos/qvel and code_indices.
        num_codes: Number of codes in the codebook.
        output_dir: Directory to save figures.
        joint_names: Ordered joint names from walker config.
        community_labels: Community assignment per code, shape [num_codes].
        cfg: Configuration dict with MI parameters.

    Returns:
        Mapping from figure name to file path.
    """
    cfg = cfg or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_neighbors = cfg.get("n_neighbors", 5)
    min_mi_threshold = cfg.get("min_mi_threshold", 0.05)
    top_k_features = cfg.get("top_k_features", 8)
    n_posture_pcs = cfg.get("posture_pcs", 3)

    # Derive joint groups
    joint_groups: dict[str, list[int]] = {}
    if joint_names:
        joint_groups = _derive_joint_groups(joint_names)
        logging.info(f"  Derived joint groups: {list(joint_groups.keys())}")

    # Extract features
    logging.info("  Extracting extended kinematic features...")
    features, codes, feature_names = extract_extended_features(
        results, joint_groups, n_posture_pcs=n_posture_pcs
    )

    if features.shape[0] == 0:
        logging.warning("  No features extracted, skipping MI analysis")
        return {}

    logging.info(f"  Features: {features.shape}, Codes: {codes.shape}")
    logging.info(f"  Feature names: {feature_names}")

    # Compute MI
    logging.info("  Computing mutual information (KSG estimator)...")
    mi_nats = compute_mutual_information(features, codes, n_neighbors=n_neighbors)
    mi_bits = mi_nats / np.log(2)

    for name, mi_val in sorted(
        zip(feature_names, mi_bits), key=lambda x: x[1], reverse=True
    ):
        logging.info(f"    {name}: {mi_val:.4f} bits")

    # Generate figures
    paths: dict[str, str] = {}

    mi_rank_path = _plot_mi_ranking(
        mi_bits, feature_names, output_dir / "mi_ranking.png"
    )
    paths["mi_ranking"] = mi_rank_path

    heatmap_path = _plot_feature_code_heatmap(
        features,
        codes,
        mi_bits,
        feature_names,
        num_codes,
        output_dir / "feature_code_heatmap.png",
        min_mi_threshold=min_mi_threshold,
        top_k_features=top_k_features,
        community_labels=community_labels,
    )
    paths["feature_code_heatmap"] = heatmap_path

    scatter_path = _plot_code_feature_scatter(
        features,
        codes,
        mi_bits,
        feature_names,
        num_codes,
        output_dir / "code_feature_scatter.png",
        community_labels=community_labels,
    )
    if scatter_path:
        paths["code_feature_scatter"] = scatter_path

    # Cross-depth MI (only if RVQ depth >= 2)
    leaf_codes = _extract_leaf_codes(results, num_codes)
    if leaf_codes is not None and len(leaf_codes) == len(codes):
        logging.info("  Computing cross-depth MI (L0 vs leaf)...")
        cross_path = compute_cross_depth_mi(
            features,
            codes,
            leaf_codes,
            feature_names,
            output_dir / "cross_depth_mi.png",
            n_neighbors=n_neighbors,
            num_codes=num_codes,
        )
        paths["cross_depth_mi"] = cross_path

    logging.info(f"  MI analysis complete: {len(paths)} figures saved")
    return paths
