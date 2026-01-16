"""Transition analysis utilities for VQ-VAE codebooks.

Provides functions for analyzing code transition patterns from learned
transition matrices.
"""

import numpy as np


def get_hub_codes(
    trans_counts: np.ndarray,
    top_k: int = 10,
) -> list[tuple[int, int, int, int]]:
    """Find hub codes with most connections.

    Args:
        trans_counts: Transition count matrix.
        top_k: Number of top codes to return.

    Returns:
        List of (code, in_degree, out_degree, total_degree).
    """
    out_degree = (trans_counts > 0).sum(axis=1)
    in_degree = (trans_counts > 0).sum(axis=0)
    total_degree = out_degree + in_degree

    top_indices = np.argsort(-total_degree)[:top_k]
    return [
        (int(c), int(in_degree[c]), int(out_degree[c]), int(total_degree[c]))
        for c in top_indices
    ]


def get_bidirectional_pairs(
    trans_counts: np.ndarray,
    min_count: int = 2,
) -> list[tuple[int, int, int, int, int]]:
    """Find bidirectional code pairs (A<->B).

    Args:
        trans_counts: Transition count matrix.
        min_count: Minimum transitions in each direction.

    Returns:
        List of (code_a, code_b, count_a_to_b, count_b_to_a, total).
    """
    num_codes = trans_counts.shape[0]
    pairs = []

    for i in range(num_codes):
        for j in range(i + 1, num_codes):
            if trans_counts[i, j] >= min_count and trans_counts[j, i] >= min_count:
                total = trans_counts[i, j] + trans_counts[j, i]
                pairs.append((i, j, int(trans_counts[i, j]), int(trans_counts[j, i]), total))

    pairs.sort(key=lambda x: -x[4])
    return pairs


def get_likely_chains(
    trans_probs: np.ndarray,
    trans_counts: np.ndarray,
    min_samples: int = 5,
    min_chain_prob: float = 0.01,
) -> list[tuple[int, int, int, float, float, float]]:
    """Find likely transition chains A->B->C.

    Args:
        trans_probs: Transition probability matrix.
        trans_counts: Transition count matrix.
        min_samples: Minimum samples for a code to be considered.
        min_chain_prob: Minimum chain probability.

    Returns:
        List of (a, b, c, p_ab, p_bc, p_chain).
    """
    num_codes = trans_probs.shape[0]
    row_sums = trans_counts.sum(axis=1)

    chains = []
    for a in range(num_codes):
        if row_sums[a] < min_samples:
            continue
        for b in range(num_codes):
            if a == b or trans_probs[a, b] < 0.05:
                continue
            if row_sums[b] < min_samples:
                continue
            for c in range(num_codes):
                if b == c or trans_probs[b, c] < 0.05:
                    continue
                chain_prob = trans_probs[a, b] * trans_probs[b, c]
                if chain_prob >= min_chain_prob:
                    chains.append((a, b, c, trans_probs[a, b], trans_probs[b, c], chain_prob))

    chains.sort(key=lambda x: -x[5])
    return chains
