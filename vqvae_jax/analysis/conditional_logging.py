"""Conditional transition analysis for wandb logging during training.

This module provides functions for pose-conditioned community detection,
where rollouts are grouped by starting pose similarity before analyzing
transition patterns.
"""

import numpy as np

from .community_analysis import discover_communities


def find_matching_rollouts_by_starting_pose(
    all_rollout_qpos: list[np.ndarray],
    reference_qpos_0: np.ndarray,
    threshold: float = 0.05,
) -> tuple[list[int], np.ndarray]:
    """Find rollouts with starting pose (qpos[0, 7:]) close to reference.

    Compares joint angles only, excluding the root 7 DOF (position + quaternion)
    to find rollouts that started in similar poses.

    Args:
        all_rollout_qpos: List of qpos arrays, each [T, nq].
        reference_qpos_0: Reference first frame qpos [nq].
        threshold: Mean absolute difference threshold for joint angles.

    Returns:
        matched_indices: List of rollout indices that match.
        distances: Array of distances for all rollouts.
    """
    # Exclude root 7 DOF (position + quaternion)
    ref_joints = reference_qpos_0[7:]

    distances = []
    matched = []
    for i, qpos in enumerate(all_rollout_qpos):
        rollout_joints = qpos[0, 7:]  # First frame, joint angles only
        dist = np.mean(np.abs(rollout_joints - ref_joints))
        distances.append(dist)
        if dist < threshold:
            matched.append(i)

    return matched, np.array(distances)


def compute_conditional_transition_matrix(
    all_rollout_indices: list[np.ndarray],
    matched_indices: list[int],
    num_codes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute transition matrix from matched rollouts only.

    Args:
        all_rollout_indices: List of code index arrays, each [T].
        matched_indices: Indices of matched rollouts.
        num_codes: Total number of codes.

    Returns:
        trans_counts: [num_codes, num_codes] transition counts.
        trans_probs: [num_codes, num_codes] row-normalized probabilities.
    """
    counts = np.zeros((num_codes, num_codes), dtype=np.int64)

    for rollout_idx in matched_indices:
        indices = all_rollout_indices[rollout_idx]
        for t in range(len(indices) - 1):
            counts[int(indices[t]), int(indices[t + 1])] += 1

    # Row-normalize
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.where(row_sums > 0, counts / row_sums, 0.0)

    return counts, probs


def detect_communities_from_transitions(
    trans_probs: np.ndarray,
) -> tuple[np.ndarray, int, dict[int, int]]:
    """Run spectral clustering on transition matrix.

    Args:
        trans_probs: Transition probability matrix [num_codes, num_codes].

    Returns:
        labels: [num_codes] community assignment.
        n_communities: Number of communities detected.
        code_to_community: Dict mapping code_idx -> community_id.
    """
    labels, _, _ = discover_communities(trans_probs)
    n_communities = len(np.unique(labels))
    code_to_community = {i: int(labels[i]) for i in range(len(labels))}

    return labels, n_communities, code_to_community
