"""Data loading utilities for discriminator training.

Handles loading H5 datasets created by collect_rollout_dataset.py and
creating train/test splits with proper handling of single-dataset mode.
"""

from dataclasses import dataclass
from typing import Dict, Generator, Tuple

import h5py
import numpy as np


@dataclass
class MotionClipDataset:
    """Container for motion clip data with train/test splits.

    Attributes:
        train_real: Training real samples of shape (N, num_steps, qpos_dim).
        train_fake: Training fake samples of shape (N, num_steps, qpos_dim).
        test_real: Test real samples of shape (M, num_steps, qpos_dim).
        test_fake: Test fake samples of shape (M, num_steps, qpos_dim).
        metadata: Dictionary with dataset metadata.
    """

    train_real: np.ndarray
    train_fake: np.ndarray
    test_real: np.ndarray
    test_fake: np.ndarray
    metadata: Dict


def load_h5_dataset(h5_path: str, dataset_name: str) -> np.ndarray:
    """Load a single dataset from H5 file.

    Args:
        h5_path: Path to H5 file created by collect_rollout_dataset.py.
        dataset_name: Name of dataset to load (e.g., 'original_qpos',
            'encoder_decoder_qpos', 'prior_logvar_-4_qpos', etc.).

    Returns:
        NumPy array of shape (num_clips, num_steps, qpos_dim).

    Raises:
        KeyError: If dataset_name not found in H5 file.
    """
    with h5py.File(h5_path, "r") as f:
        if dataset_name not in f:
            available = list(f.keys())
            raise KeyError(
                f"Dataset '{dataset_name}' not found in {h5_path}. "
                f"Available datasets: {available}"
            )
        data = f[dataset_name][:]
    return data


def load_h5_metadata(h5_path: str) -> Dict:
    """Load metadata attributes from H5 file.

    Args:
        h5_path: Path to H5 file.

    Returns:
        Dictionary with metadata (num_clips, num_steps, qpos_dim, etc.).
    """
    with h5py.File(h5_path, "r") as f:
        metadata = dict(f.attrs)
    return metadata


def create_train_test_split(
    real_data: np.ndarray,
    fake_data: np.ndarray,
    train_ratio: float = 0.8,
    seed: int = 42,
    single_dataset_mode: bool = False,
) -> MotionClipDataset:
    """Create train/test splits for real and fake data.

    Handles two modes:

    1. Normal mode (different datasets for real/fake):
       Each dataset is shuffled and split independently into train/test
       according to train_ratio.

    2. Single dataset mode (same dataset for both real/fake):
       First the dataset is shuffled and split in half to create
       real_half and fake_half. Then each half is split into train/test
       according to train_ratio. This creates a baseline where the
       network should not be able to learn meaningful discrimination.

    Args:
        real_data: Real samples array of shape (N, num_steps, qpos_dim).
        fake_data: Fake samples array of shape (N, num_steps, qpos_dim).
        train_ratio: Fraction of data to use for training (default: 0.8).
        seed: Random seed for reproducible splits.
        single_dataset_mode: If True, real_data and fake_data are from
            the same source and should be split in half first.

    Returns:
        MotionClipDataset with train/test splits for both real and fake.
    """
    rng = np.random.default_rng(seed)

    if single_dataset_mode:
        # Same dataset for both real and fake
        # First split in half, then each half into train/test
        data = real_data
        n_samples = len(data)
        indices = rng.permutation(n_samples)

        # Split into two halves
        mid = n_samples // 2
        real_indices = indices[:mid]
        fake_indices = indices[mid:]

        real_half = data[real_indices]
        fake_half = data[fake_indices]

        # Split each half into train/test
        real_n_train = int(len(real_half) * train_ratio)
        fake_n_train = int(len(fake_half) * train_ratio)

        train_real = real_half[:real_n_train]
        test_real = real_half[real_n_train:]
        train_fake = fake_half[:fake_n_train]
        test_fake = fake_half[fake_n_train:]

    else:
        # Different datasets - split each independently
        # Shuffle and split real data
        real_indices = rng.permutation(len(real_data))
        real_data_shuffled = real_data[real_indices]
        n_train_real = int(len(real_data_shuffled) * train_ratio)
        train_real = real_data_shuffled[:n_train_real]
        test_real = real_data_shuffled[n_train_real:]

        # Shuffle and split fake data (with different seed offset for independence)
        rng_fake = np.random.default_rng(seed + 1)
        fake_indices = rng_fake.permutation(len(fake_data))
        fake_data_shuffled = fake_data[fake_indices]
        n_train_fake = int(len(fake_data_shuffled) * train_ratio)
        train_fake = fake_data_shuffled[:n_train_fake]
        test_fake = fake_data_shuffled[n_train_fake:]

    metadata = {
        "single_dataset_mode": single_dataset_mode,
        "train_ratio": train_ratio,
        "seed": seed,
        "num_train_real": len(train_real),
        "num_train_fake": len(train_fake),
        "num_test_real": len(test_real),
        "num_test_fake": len(test_fake),
    }

    return MotionClipDataset(
        train_real=train_real,
        train_fake=train_fake,
        test_real=test_real,
        test_fake=test_fake,
        metadata=metadata,
    )


def create_batches(
    real_data: np.ndarray,
    fake_data: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool = True,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Create balanced batches of real and fake samples.

    Each batch contains batch_size//2 real samples and batch_size//2
    fake samples, combined and shuffled together.

    Args:
        real_data: Real samples of shape (N, num_steps, qpos_dim).
        fake_data: Fake samples of shape (M, num_steps, qpos_dim).
        batch_size: Total batch size (must be even).
        rng: NumPy random generator for shuffling.
        shuffle: Whether to shuffle data before batching.

    Yields:
        Tuples of (batch_data, batch_labels) where:
            - batch_data has shape (batch_size, num_steps, qpos_dim)
            - batch_labels has shape (batch_size,) with 1.0 for real, 0.0 for fake
    """
    half_batch = batch_size // 2

    if shuffle:
        real_indices = rng.permutation(len(real_data))
        fake_indices = rng.permutation(len(fake_data))
    else:
        real_indices = np.arange(len(real_data))
        fake_indices = np.arange(len(fake_data))

    # Number of complete batches we can form
    n_batches = min(len(real_data), len(fake_data)) // half_batch

    for i in range(n_batches):
        start = i * half_batch
        end = start + half_batch

        real_batch = real_data[real_indices[start:end]]
        fake_batch = fake_data[fake_indices[start:end]]

        # Combine real and fake samples
        batch_data = np.concatenate([real_batch, fake_batch], axis=0)
        batch_labels = np.concatenate(
            [
                np.ones(half_batch, dtype=np.float32),
                np.zeros(half_batch, dtype=np.float32),
            ]
        )

        # Shuffle within batch
        perm = rng.permutation(batch_size)
        batch_data = batch_data[perm]
        batch_labels = batch_labels[perm]

        yield batch_data, batch_labels
