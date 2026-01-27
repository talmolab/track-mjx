"""Kinematic analysis for VQ-VAE codes.

This module provides functions for extracting kinematic features from
motion data and correlating them with code activations.
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
from scipy import stats as scipy_stats

from .inference_cache import InferenceResult


@dataclass
class KinematicFeatures:
    """Kinematic features extracted from motion data.

    Attributes:
        linear_velocity: Root linear velocity magnitude, shape [T].
        angular_velocity: Root angular velocity magnitude, shape [T].
        body_height: Root z position (body height), shape [T].
        joint_velocities: Mean absolute joint velocities, shape [T].
    """

    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    body_height: np.ndarray
    joint_velocities: np.ndarray


def extract_kinematic_features(
    qpos: np.ndarray,
    qvel: np.ndarray,
    dt: float = 0.02,
) -> KinematicFeatures:
    """Extract kinematic features from position and velocity data.

    Assumes rodent model layout:
    - qpos[0:3]: root position (x, y, z)
    - qpos[3:7]: root quaternion
    - qpos[7:]: joint angles
    - qvel[0:3]: root linear velocity
    - qvel[3:6]: root angular velocity
    - qvel[6:]: joint velocities

    Args:
        qpos: Generalized positions, shape [T, nq].
        qvel: Generalized velocities, shape [T, nv].
        dt: Timestep for velocity computation.

    Returns:
        KinematicFeatures with extracted features.
    """
    T = qpos.shape[0]

    # Linear velocity from qvel (root linear velocity)
    if qvel.shape[1] >= 3:
        linear_velocity = np.linalg.norm(qvel[:, :3], axis=1)
    else:
        # Fallback: compute from position differences
        dpos = np.diff(qpos[:, :3], axis=0) / dt
        linear_velocity = np.zeros(T)
        linear_velocity[1:] = np.linalg.norm(dpos, axis=1)

    # Angular velocity from qvel
    if qvel.shape[1] >= 6:
        angular_velocity = np.linalg.norm(qvel[:, 3:6], axis=1)
    else:
        angular_velocity = np.zeros(T)

    # Body height (z position)
    body_height = qpos[:, 2]

    # Joint velocities (mean absolute velocity across joints)
    if qvel.shape[1] > 6:
        joint_velocities = np.mean(np.abs(qvel[:, 6:]), axis=1)
    else:
        joint_velocities = np.zeros(T)

    return KinematicFeatures(
        linear_velocity=linear_velocity,
        angular_velocity=angular_velocity,
        body_height=body_height,
        joint_velocities=joint_velocities,
    )


@dataclass
class KinematicProfile:
    """Kinematic profile for a single code.

    Attributes:
        code_idx: The code index.
        n_frames: Number of frames with this code active.
        linear_velocity_mean: Mean linear velocity.
        linear_velocity_std: Std of linear velocity.
        angular_velocity_mean: Mean angular velocity.
        angular_velocity_std: Std of angular velocity.
        body_height_mean: Mean body height.
        body_height_std: Std of body height.
        joint_velocities_mean: Mean joint velocities.
        joint_velocities_std: Std of joint velocities.
    """

    code_idx: int
    n_frames: int
    linear_velocity_mean: float
    linear_velocity_std: float
    angular_velocity_mean: float
    angular_velocity_std: float
    body_height_mean: float
    body_height_std: float
    joint_velocities_mean: float
    joint_velocities_std: float


def compute_kinematic_profiles(
    results: Sequence[InferenceResult],
    num_codes: int,
    dt: float = 0.02,
) -> list[KinematicProfile]:
    """Compute kinematic profiles for each code.

    Args:
        results: List of InferenceResult with qpos/qvel.
        num_codes: Total number of codes.
        dt: Timestep for feature extraction.

    Returns:
        List of KinematicProfile for each code.
    """
    # Aggregate features by code
    features_by_code: dict[int, dict[str, list]] = {
        i: {
            "linear_velocity": [],
            "angular_velocity": [],
            "body_height": [],
            "joint_velocities": [],
        }
        for i in range(num_codes)
    }

    for result in results:
        if result.qpos is None or result.qvel is None:
            continue

        features = extract_kinematic_features(result.qpos, result.qvel, dt)

        for t, code_idx in enumerate(result.code_indices):
            code_idx = int(code_idx)
            if 0 <= code_idx < num_codes:
                features_by_code[code_idx]["linear_velocity"].append(
                    features.linear_velocity[t]
                )
                features_by_code[code_idx]["angular_velocity"].append(
                    features.angular_velocity[t]
                )
                features_by_code[code_idx]["body_height"].append(
                    features.body_height[t]
                )
                features_by_code[code_idx]["joint_velocities"].append(
                    features.joint_velocities[t]
                )

    # Compute statistics
    profiles = []
    for code_idx in range(num_codes):
        feats = features_by_code[code_idx]
        n_frames = len(feats["linear_velocity"])

        if n_frames == 0:
            profiles.append(
                KinematicProfile(
                    code_idx=code_idx,
                    n_frames=0,
                    linear_velocity_mean=0.0,
                    linear_velocity_std=0.0,
                    angular_velocity_mean=0.0,
                    angular_velocity_std=0.0,
                    body_height_mean=0.0,
                    body_height_std=0.0,
                    joint_velocities_mean=0.0,
                    joint_velocities_std=0.0,
                )
            )
        else:
            profiles.append(
                KinematicProfile(
                    code_idx=code_idx,
                    n_frames=n_frames,
                    linear_velocity_mean=float(np.mean(feats["linear_velocity"])),
                    linear_velocity_std=float(np.std(feats["linear_velocity"])),
                    angular_velocity_mean=float(np.mean(feats["angular_velocity"])),
                    angular_velocity_std=float(np.std(feats["angular_velocity"])),
                    body_height_mean=float(np.mean(feats["body_height"])),
                    body_height_std=float(np.std(feats["body_height"])),
                    joint_velocities_mean=float(np.mean(feats["joint_velocities"])),
                    joint_velocities_std=float(np.std(feats["joint_velocities"])),
                )
            )

    return profiles


def plot_kinematic_heatmap(
    profiles: list[KinematicProfile],
    output_path: str | Path,
    figsize: tuple[int, int] = (12, 6),
) -> str:
    """Plot a heatmap of kinematic features vs codes.

    Args:
        profiles: List of KinematicProfile for each code.
        output_path: Path to save the figure.
        figsize: Figure size.

    Returns:
        Path to saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data matrix
    feature_names = [
        "Linear Velocity",
        "Angular Velocity",
        "Body Height",
        "Joint Velocities",
    ]
    num_codes = len(profiles)

    # Build matrix: [n_features, n_codes]
    data = np.zeros((4, num_codes))
    for i, profile in enumerate(profiles):
        data[0, i] = profile.linear_velocity_mean
        data[1, i] = profile.angular_velocity_mean
        data[2, i] = profile.body_height_mean
        data[3, i] = profile.joint_velocities_mean

    # Z-score normalize rows for better visualization
    data_zscore = scipy_stats.zscore(data, axis=1)
    data_zscore = np.nan_to_num(data_zscore)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(data_zscore, aspect="auto", cmap="RdBu_r")
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.set_xlabel("Code Index")
    ax.set_title("Kinematic Features by Code (z-scored)")

    plt.colorbar(im, ax=ax, label="Z-score")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(output_path)


def plot_kinematic_clusters(
    profiles: list[KinematicProfile],
    output_path: str | Path,
    method: str = "pca",
    figsize: tuple[int, int] = (10, 10),
) -> str:
    """Plot codes in 2D kinematic feature space.

    Args:
        profiles: List of KinematicProfile for each code.
        output_path: Path to save the figure.
        method: Dimensionality reduction method ("pca" or "umap").
        figsize: Figure size.

    Returns:
        Path to saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build feature matrix
    num_codes = len(profiles)
    features = np.zeros((num_codes, 4))

    for i, profile in enumerate(profiles):
        features[i, 0] = profile.linear_velocity_mean
        features[i, 1] = profile.angular_velocity_mean
        features[i, 2] = profile.body_height_mean
        features[i, 3] = profile.joint_velocities_mean

    # Filter out codes with no data
    valid_mask = np.any(features != 0, axis=1)
    if np.sum(valid_mask) < 3:
        logging.warning("Not enough valid codes for clustering visualization")
        return ""

    valid_features = features[valid_mask]
    valid_indices = np.where(valid_mask)[0]

    # Dimensionality reduction
    if method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2)
        coords_2d = reducer.fit_transform(valid_features)
    elif method == "umap":
        try:
            import umap

            reducer = umap.UMAP(n_components=2, random_state=42)
            coords_2d = reducer.fit_transform(valid_features)
        except ImportError:
            logging.warning("UMAP not installed, falling back to PCA")
            from sklearn.decomposition import PCA

            reducer = PCA(n_components=2)
            coords_2d = reducer.fit_transform(valid_features)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    fig, ax = plt.subplots(figsize=figsize)

    # Color by linear velocity (or another feature)
    colors = valid_features[:, 0]  # Linear velocity
    scatter = ax.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=colors,
        cmap="viridis",
        s=100,
        alpha=0.8,
        edgecolors="white",
        linewidth=0.5,
    )

    # Add code labels
    for i, (x, y) in enumerate(coords_2d):
        ax.annotate(
            str(valid_indices[i]),
            (x, y),
            fontsize=8,
            ha="center",
            va="center",
        )

    plt.colorbar(scatter, ax=ax, label="Linear Velocity")
    ax.set_title(f"Codes in Kinematic Space ({method.upper()})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.axis("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(output_path)


def plot_kinematic_bars(
    profiles: list[KinematicProfile],
    output_path: str | Path,
    figsize: tuple[int, int] = (14, 8),
) -> str:
    """Plot bar charts of kinematic features by code.

    Args:
        profiles: List of KinematicProfile for each code.
        output_path: Path to save the figure.
        figsize: Figure size.

    Returns:
        Path to saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    codes = [p.code_idx for p in profiles]

    features = [
        ("Linear Velocity", "linear_velocity_mean", "linear_velocity_std"),
        ("Angular Velocity", "angular_velocity_mean", "angular_velocity_std"),
        ("Body Height", "body_height_mean", "body_height_std"),
        ("Joint Velocities", "joint_velocities_mean", "joint_velocities_std"),
    ]

    for ax, (name, mean_attr, std_attr) in zip(axes, features):
        means = [getattr(p, mean_attr) for p in profiles]
        stds = [getattr(p, std_attr) for p in profiles]

        ax.bar(codes, means, yerr=stds, capsize=2, alpha=0.7, color="steelblue")
        ax.set_xlabel("Code Index")
        ax.set_ylabel(name)
        ax.set_title(f"{name} by Code")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(output_path)


def save_kinematic_analysis(
    output_dir: str | Path,
    profiles: list[KinematicProfile],
) -> dict[str, str]:
    """Save kinematic analysis results.

    Args:
        output_dir: Directory to save outputs.
        profiles: List of KinematicProfile.

    Returns:
        Dictionary mapping output names to file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Save profiles as JSON
    profiles_data = [asdict(p) for p in profiles]
    with open(output_dir / "kinematic_profiles.json", "w") as f:
        json.dump(profiles_data, f, indent=2)
    paths["profiles"] = str(output_dir / "kinematic_profiles.json")

    logging.info(f"Saved kinematic analysis to {output_dir}")
    return paths


def run_kinematic_analysis(
    results: Sequence[InferenceResult],
    num_codes: int,
    output_dir: str | Path,
    clustering_method: str = "pca",
    dt: float = 0.02,
) -> dict[str, str]:
    """Run complete kinematic analysis pipeline.

    Args:
        results: List of InferenceResult with qpos/qvel.
        num_codes: Total number of codes.
        output_dir: Directory to save outputs.
        clustering_method: Method for clustering visualization ("pca" or "umap").
        dt: Timestep for feature extraction.

    Returns:
        Dictionary mapping output names to file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Computing kinematic profiles...")
    profiles = compute_kinematic_profiles(results, num_codes, dt)

    # Count codes with data
    n_valid = sum(1 for p in profiles if p.n_frames > 0)
    logging.info(f"Computed profiles for {n_valid}/{num_codes} codes with data")

    # Save data
    paths = save_kinematic_analysis(output_dir, profiles)

    # Plot heatmap
    heatmap_path = plot_kinematic_heatmap(
        profiles, output_dir / "kinematic_heatmap.png"
    )
    paths["heatmap"] = heatmap_path

    # Plot clusters
    cluster_path = plot_kinematic_clusters(
        profiles, output_dir / "kinematic_clusters.png", method=clustering_method
    )
    if cluster_path:
        paths["clusters"] = cluster_path

    # Plot bar charts
    bars_path = plot_kinematic_bars(profiles, output_dir / "kinematic_bars.png")
    paths["bars"] = bars_path

    return paths
