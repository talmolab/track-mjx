"""Inference result caching for VQ-VAE analysis.

This module provides data structures and caching utilities for storing
and retrieving VQ-VAE inference results to avoid redundant computation.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class InferenceResult:
    """Result from running VQ-VAE inference on a clip.

    Attributes:
        clip_idx: Index of the reference clip.
        code_indices: Discrete code indices per frame, shape [T].
        qpos: Generalized positions per frame, shape [T, nq].
        qvel: Generalized velocities per frame, shape [T, nv].
        rewards: Reward values per frame, shape [T].
        states: Optional list of environment states for rendering.
    """

    clip_idx: int
    code_indices: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    rewards: np.ndarray
    states: list[Any] | None = None


def compute_cache_key(
    checkpoint_path: str,
    step: int | None,
    num_clips: int,
    seed: int,
    use_stickiness: bool = False,
) -> str:
    """Compute a unique cache key for inference results.

    Args:
        checkpoint_path: Path to the VQ-VAE checkpoint.
        step: Checkpoint step, or None for latest.
        num_clips: Number of clips to run inference on.
        seed: Random seed for reproducibility.
        use_stickiness: Whether stickiness bias is applied during inference.

    Returns:
        A hexadecimal hash string identifying these parameters.
    """
    key_data = {
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "step": step,
        "num_clips": num_clips,
        "seed": seed,
        "use_stickiness": use_stickiness,
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def get_cache_path(cache_dir: str | Path, cache_key: str) -> Path:
    """Get the file path for a cached inference result.

    Args:
        cache_dir: Directory to store cache files.
        cache_key: Unique cache key from compute_cache_key.

    Returns:
        Path to the cache file.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"inference_{cache_key}.npz"


def save_inference_cache(
    cache_path: str | Path,
    results: list[InferenceResult],
    metadata: dict[str, Any],
) -> None:
    """Save inference results to an NPZ cache file.

    Args:
        cache_path: Path to save the cache file.
        results: List of InferenceResult objects to cache.
        metadata: Additional metadata to store (checkpoint info, etc.).
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare arrays for NPZ storage
    # We store each result's arrays with indexed keys
    arrays = {}
    clip_indices = []

    for i, result in enumerate(results):
        clip_indices.append(result.clip_idx)
        arrays[f"code_indices_{i}"] = result.code_indices
        arrays[f"qpos_{i}"] = result.qpos
        arrays[f"qvel_{i}"] = result.qvel
        arrays[f"rewards_{i}"] = result.rewards

    arrays["clip_indices"] = np.array(clip_indices)
    arrays["num_results"] = np.array(len(results))

    # Store metadata as JSON string
    arrays["metadata"] = np.array(json.dumps(metadata))

    np.savez_compressed(str(cache_path), **arrays)
    logging.info(f"Saved inference cache to {cache_path}")


def load_inference_cache(
    cache_path: str | Path,
    expected_metadata: dict[str, Any] | None = None,
) -> tuple[list[InferenceResult], dict[str, Any]] | None:
    """Load inference results from an NPZ cache file.

    Args:
        cache_path: Path to the cache file.
        expected_metadata: If provided, validate that cached metadata matches.
            Returns None if validation fails.

    Returns:
        Tuple of (results, metadata) if cache is valid, None otherwise.
    """
    cache_path = Path(cache_path)

    if not cache_path.exists():
        logging.info(f"Cache not found: {cache_path}")
        return None

    try:
        data = np.load(str(cache_path), allow_pickle=True)

        # Load metadata
        metadata = json.loads(str(data["metadata"]))

        # Validate metadata if expected
        if expected_metadata is not None:
            for key, expected_value in expected_metadata.items():
                if key not in metadata:
                    logging.warning(f"Cache missing metadata key: {key}")
                    return None
                if metadata[key] != expected_value:
                    logging.warning(
                        f"Cache metadata mismatch for {key}: "
                        f"expected {expected_value}, got {metadata[key]}"
                    )
                    return None

        # Load results
        num_results = int(data["num_results"])
        clip_indices = data["clip_indices"]

        results = []
        for i in range(num_results):
            result = InferenceResult(
                clip_idx=int(clip_indices[i]),
                code_indices=data[f"code_indices_{i}"],
                qpos=data[f"qpos_{i}"],
                qvel=data[f"qvel_{i}"],
                rewards=data[f"rewards_{i}"],
                states=None,  # States are not cached (too large)
            )
            results.append(result)

        logging.info(f"Loaded {len(results)} results from cache: {cache_path}")
        return results, metadata

    except Exception as e:
        logging.warning(f"Failed to load cache {cache_path}: {e}")
        return None


def is_cache_valid(
    cache_dir: str | Path,
    checkpoint_path: str,
    step: int | None,
    num_clips: int,
    seed: int,
) -> bool:
    """Check if a valid cache exists for the given parameters.

    Args:
        cache_dir: Directory containing cache files.
        checkpoint_path: Path to the VQ-VAE checkpoint.
        step: Checkpoint step.
        num_clips: Number of clips.
        seed: Random seed.

    Returns:
        True if valid cache exists, False otherwise.
    """
    cache_key = compute_cache_key(checkpoint_path, step, num_clips, seed)
    cache_path = get_cache_path(cache_dir, cache_key)
    return cache_path.exists()
