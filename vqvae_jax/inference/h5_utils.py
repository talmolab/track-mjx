"""HDF5 utilities for storing and loading VQ-VAE rollout data.

This module provides data structures and I/O functions for storing
rollout results in HDF5 format, enabling separation between inference
and analysis pipelines.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass
class RolloutData:
    """Data from a single rollout for analysis.

    Attributes:
        clip_idx: Index of the reference clip.
        code_indices: Discrete code indices per frame, shape [T] (primary L0).
        qpos: Generalized positions per frame, shape [T, nq].
        qvel: Generalized velocities per frame, shape [T, nv].
        rewards: Reward values per frame, shape [T].
        z_e: Optional encoder outputs before quantization, shape [T, latent_dim].
        rvq_indices: Optional per-depth indices for RVQ, tuple of D arrays
            each shape [T]. None for depth=1 models.
    """

    clip_idx: int
    code_indices: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    rewards: np.ndarray
    z_e: np.ndarray | None = None
    rvq_indices: tuple[np.ndarray, ...] | None = None


def save_rollout_h5(
    path: str | Path,
    rollouts: list[RolloutData],
    metadata: dict[str, Any],
) -> None:
    """Save rollout data to HDF5 file.

    File structure:
        /metadata (attrs): checkpoint_path, step, num_clips, seed, etc.
        /clip_0/code_indices: [T]
        /clip_0/qpos: [T, nq]
        /clip_0/qvel: [T, nv]
        /clip_0/rewards: [T]
        /clip_0/z_e: [T, latent_dim] (optional)
        /clip_0/rvq_indices/depth_0: [T] (optional, for RVQ depth>1)
        /clip_0/rvq_indices/depth_1: [T] (optional)
        /clip_1/...
        ...

    Args:
        path: Path to save the HDF5 file.
        rollouts: List of RolloutData objects.
        metadata: Dictionary of metadata (checkpoint path, step, seed, etc.).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        # Store metadata as root attributes
        meta_group = f.create_group("metadata")
        for key, value in metadata.items():
            if value is None:
                meta_group.attrs[key] = "null"
            elif isinstance(value, (str, int, float, bool)):
                meta_group.attrs[key] = value
            else:
                # Convert complex types to string
                meta_group.attrs[key] = str(value)

        meta_group.attrs["num_rollouts"] = len(rollouts)

        # Store each rollout
        for i, rollout in enumerate(rollouts):
            group = f.create_group(f"clip_{i}")
            group.attrs["clip_idx"] = rollout.clip_idx

            group.create_dataset(
                "code_indices",
                data=rollout.code_indices,
                compression="gzip",
                compression_opts=4,
            )
            group.create_dataset(
                "qpos",
                data=rollout.qpos,
                compression="gzip",
                compression_opts=4,
            )
            group.create_dataset(
                "qvel",
                data=rollout.qvel,
                compression="gzip",
                compression_opts=4,
            )
            group.create_dataset(
                "rewards",
                data=rollout.rewards,
                compression="gzip",
                compression_opts=4,
            )

            if rollout.z_e is not None:
                group.create_dataset(
                    "z_e",
                    data=rollout.z_e,
                    compression="gzip",
                    compression_opts=4,
                )

            if rollout.rvq_indices is not None:
                rvq_group = group.create_group("rvq_indices")
                for d, idx_d in enumerate(rollout.rvq_indices):
                    rvq_group.create_dataset(
                        f"depth_{d}",
                        data=idx_d,
                        compression="gzip",
                        compression_opts=4,
                    )


def load_rollout_h5(
    path: str | Path,
) -> tuple[list[RolloutData], dict[str, Any]]:
    """Load rollout data from HDF5 file.

    Args:
        path: Path to the HDF5 file.

    Returns:
        Tuple of (rollouts, metadata).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Rollout file not found: {path}")

    with h5py.File(path, "r") as f:
        # Load metadata
        if "metadata" not in f:
            raise ValueError("Missing metadata group in rollout file")

        meta_group = f["metadata"]
        metadata = {}
        for key in meta_group.attrs:
            value = meta_group.attrs[key]
            if value == "null":
                metadata[key] = None
            else:
                metadata[key] = value

        num_rollouts = meta_group.attrs.get("num_rollouts", 0)

        # Load rollouts
        rollouts = []
        for i in range(num_rollouts):
            group_name = f"clip_{i}"
            if group_name not in f:
                raise ValueError(f"Missing group {group_name} in rollout file")

            group = f[group_name]
            clip_idx = group.attrs.get("clip_idx", i)

            z_e = None
            if "z_e" in group:
                z_e = np.array(group["z_e"])

            rvq_indices = None
            if "rvq_indices" in group:
                rvq_group = group["rvq_indices"]
                depth_arrays = []
                d = 0
                while f"depth_{d}" in rvq_group:
                    depth_arrays.append(np.array(rvq_group[f"depth_{d}"]))
                    d += 1
                if depth_arrays:
                    rvq_indices = tuple(depth_arrays)

            rollout = RolloutData(
                clip_idx=int(clip_idx),
                code_indices=np.array(group["code_indices"]),
                qpos=np.array(group["qpos"]),
                qvel=np.array(group["qvel"]),
                rewards=np.array(group["rewards"]),
                z_e=z_e,
                rvq_indices=rvq_indices,
            )
            rollouts.append(rollout)

    return rollouts, metadata


def get_rollout_summary(path: str | Path) -> dict[str, Any]:
    """Get summary information about a rollout file without loading all data.

    Args:
        path: Path to the HDF5 file.

    Returns:
        Dictionary with summary statistics.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Rollout file not found: {path}")

    with h5py.File(path, "r") as f:
        meta_group = f["metadata"]
        num_rollouts = meta_group.attrs.get("num_rollouts", 0)

        # Get metadata
        metadata = {}
        for key in meta_group.attrs:
            value = meta_group.attrs[key]
            if value == "null":
                metadata[key] = None
            else:
                metadata[key] = value

        # Compute summary stats from first rollout
        total_frames = 0
        qpos_dim = 0
        qvel_dim = 0
        has_z_e = False
        latent_dim = 0

        for i in range(num_rollouts):
            group = f[f"clip_{i}"]
            total_frames += len(group["code_indices"])

            if i == 0:
                qpos_dim = group["qpos"].shape[1] if len(group["qpos"].shape) > 1 else 0
                qvel_dim = group["qvel"].shape[1] if len(group["qvel"].shape) > 1 else 0
                if "z_e" in group:
                    has_z_e = True
                    latent_dim = (
                        group["z_e"].shape[1] if len(group["z_e"].shape) > 1 else 0
                    )

    return {
        "num_rollouts": num_rollouts,
        "total_frames": total_frames,
        "qpos_dim": qpos_dim,
        "qvel_dim": qvel_dim,
        "has_z_e": has_z_e,
        "latent_dim": latent_dim,
        "metadata": metadata,
    }
