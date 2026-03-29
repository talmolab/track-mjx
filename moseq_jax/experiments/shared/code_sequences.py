"""Code sequence generation for temporal-order and control experiments."""

import numpy as np


def make_correct_sequences(
    codes: np.ndarray,
    clip_indices: list[int],
    max_steps: int,
) -> list[np.ndarray]:
    """Extract correct KPMS code sequences, padded to *max_steps*.

    Args:
        codes: Full code array ``[n_clips, n_frames]``.
        clip_indices: Which clips to extract.
        max_steps: Target sequence length (pad with last code).

    Returns:
        List of ``K`` arrays each of shape ``[max_steps]``.
    """
    sequences: list[np.ndarray] = []
    for ci in clip_indices:
        seq = codes[ci]  # [n_frames]
        if len(seq) >= max_steps:
            sequences.append(seq[:max_steps].copy())
        else:
            pad = np.full(max_steps - len(seq), seq[-1], dtype=seq.dtype)
            sequences.append(np.concatenate([seq, pad]))
    return sequences


def make_shuffled_step_sequences(
    k: int,
    max_steps: int,
    num_codes: int,
    seed: int = 42,
) -> list[np.ndarray]:
    """Uniform-random code per timestep for *k* sequences.

    Args:
        k: Number of sequences.
        max_steps: Sequence length.
        num_codes: Number of distinct codes (draw from ``[0, num_codes)``).
        seed: Random seed.

    Returns:
        List of ``k`` arrays each of shape ``[max_steps]``.
    """
    rng = np.random.RandomState(seed)
    return [rng.randint(0, num_codes, size=max_steps) for _ in range(k)]


def make_shuffled_trajectory_sequences(
    correct_sequences: list[np.ndarray],
    seed: int = 42,
) -> list[np.ndarray]:
    """Random permutation of each correct sequence (preserves histogram).

    Args:
        correct_sequences: List of correct code arrays.
        seed: Random seed.

    Returns:
        List of shuffled arrays, same lengths as inputs.
    """
    rng = np.random.RandomState(seed)
    shuffled: list[np.ndarray] = []
    for seq in correct_sequences:
        s = seq.copy()
        rng.shuffle(s)
        shuffled.append(s)
    return shuffled
