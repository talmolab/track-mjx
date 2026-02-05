"""Community analysis for VQ-VAE codes using spectral clustering.

This module provides functions for discovering communities of codes based
on transition patterns, computing soft membership probabilities, and
identifying overlapping codes that bridge multiple communities.
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.special import softmax

from .inference_cache import InferenceResult
from .transition_analysis import compute_transition_matrix


@dataclass
class Community:
    """A community of related codes discovered via spectral clustering.

    Attributes:
        id: Community identifier.
        code_indices: All codes with membership > threshold.
        core_codes: Codes with high internal connectivity, single community.
        boundary_codes: Codes with significant membership in 2+ communities.
        internal_transition_rate: Rate of transitions within community.
    """

    id: int
    code_indices: list[int]
    core_codes: list[int]
    boundary_codes: list[int]
    internal_transition_rate: float


@dataclass
class CommunityStructure:
    """Complete community structure analysis results.

    Attributes:
        communities: List of Community objects.
        code_to_community: Primary community for each code (argmax).
        code_membership_probs: Soft membership probabilities [num_codes, n_communities].
        overlapping_codes: Codes with significant membership in 2+ communities.
        overlap_stats: For overlapping codes, dict of {code: {community: prob}}.
        coarsened_transitions: Community-level transition matrix [n_comm, n_comm].
        modularity: Newman modularity score of the clustering.
        n_communities: Number of communities discovered.
    """

    communities: list[Community]
    code_to_community: dict[int, int]
    code_membership_probs: np.ndarray
    overlapping_codes: list[int]
    overlap_stats: dict[int, dict[int, float]]
    coarsened_transitions: np.ndarray
    modularity: float
    n_communities: int


def compute_eigengap_heuristic(
    eigenvalues: np.ndarray,
    max_k: int = 8,
    min_gap_ratio: float = 1.5,
) -> int:
    """Determine optimal number of clusters using eigengap heuristic.

    Uses relative gap ratios rather than absolute gaps to be more robust
    to nearly-uniform eigenvalue distributions (e.g., from sparse transition
    matrices with high self-loop probabilities).

    Args:
        eigenvalues: Sorted eigenvalues from spectral decomposition.
        max_k: Maximum number of clusters to consider (default 8).
        min_gap_ratio: Minimum ratio of gap to median gap to be significant.

    Returns:
        Optimal number of clusters.
    """
    # Use eigenvalues in ascending order
    eigenvalues = np.sort(eigenvalues)[:max_k + 5]  # Get a few extra for context

    if len(eigenvalues) < 3:
        return 2

    # Compute gaps between consecutive eigenvalues
    gaps = np.diff(eigenvalues)

    if len(gaps) < 2:
        return 2

    # Start from index 1 to ensure at least 2 clusters
    # Look for gaps that are significantly larger than the median gap
    start_idx = 1
    end_idx = min(len(gaps), max_k)

    if end_idx <= start_idx:
        return 2

    relevant_gaps = gaps[start_idx:end_idx]
    median_gap = np.median(relevant_gaps)

    # Find gaps that are significantly larger than median
    if median_gap > 0:
        gap_ratios = relevant_gaps / median_gap
        # Find first gap that exceeds the threshold (indicates cluster boundary)
        significant_gaps = np.where(gap_ratios >= min_gap_ratio)[0]

        if len(significant_gaps) > 0:
            # Take the first significant gap (prefer fewer clusters)
            best_k = significant_gaps[0] + start_idx + 1
        else:
            # No significant gap found - default to small number of clusters
            # This happens when transition matrix is very sparse/diagonal
            best_k = min(3, max_k)
    else:
        # All gaps are zero (degenerate case)
        best_k = 2

    # Ensure reasonable bounds
    return max(2, min(best_k, max_k))


def discover_communities(
    trans_probs: np.ndarray,
    n_communities: int | None = None,
    min_communities: int = 2,
    max_communities: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discover communities via spectral clustering on transition matrix.

    Args:
        trans_probs: Transition probability matrix [num_codes, num_codes].
        n_communities: Number of communities (None = auto-detect via eigengap).
        min_communities: Minimum communities if auto-detecting.
        max_communities: Maximum communities if auto-detecting.

    Returns:
        Tuple of (labels, embedding, centroids):
        - labels: Community assignment for each code [num_codes].
        - embedding: Spectral embedding [num_codes, n_communities].
        - centroids: Cluster centroids [n_communities, n_communities].
    """
    from sklearn.cluster import KMeans

    num_codes = trans_probs.shape[0]

    # Symmetrize transition matrix for spectral analysis
    # Use geometric mean: sqrt(P[i,j] * P[j,i])
    sym_trans = np.sqrt(trans_probs * trans_probs.T + 1e-10)

    # Compute normalized graph Laplacian
    degree = np.sum(sym_trans, axis=1)
    degree_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0)
    D_inv_sqrt = np.diag(degree_inv_sqrt)
    L_norm = np.eye(num_codes) - D_inv_sqrt @ sym_trans @ D_inv_sqrt

    # Compute eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(L_norm)

    # Determine number of communities
    if n_communities is None:
        n_communities = compute_eigengap_heuristic(eigenvalues, max_communities)
        n_communities = max(min_communities, n_communities)
        logging.info(f"Auto-detected {n_communities} communities via eigengap")

    n_communities = min(n_communities, num_codes)

    # Use first k eigenvectors (smallest eigenvalues) as embedding
    embedding = eigenvectors[:, :n_communities]

    # Normalize rows for k-means
    row_norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding_normalized = np.where(row_norms > 0, embedding / row_norms, embedding)

    # Run k-means clustering
    kmeans = KMeans(n_clusters=n_communities, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embedding_normalized)
    centroids = kmeans.cluster_centers_

    return labels, embedding_normalized, centroids


def compute_soft_membership(
    embedding: np.ndarray,
    centroids: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """Compute soft membership probabilities via distance-based softmax.

    Args:
        embedding: Spectral embedding [num_codes, n_dim].
        centroids: Cluster centroids [n_communities, n_dim].
        temperature: Softmax temperature (lower = sharper).

    Returns:
        Membership probabilities [num_codes, n_communities].
    """
    # Compute squared distances to each centroid
    # distances[i, j] = ||embedding[i] - centroids[j]||^2
    distances_sq = np.sum(
        (embedding[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
        axis=2,
    )

    # Convert to similarities (negative distances)
    similarities = -distances_sq / temperature

    # Apply softmax to get probabilities
    membership_probs = softmax(similarities, axis=1)

    return membership_probs


def identify_overlapping_codes(
    membership_probs: np.ndarray,
    threshold: float = 0.2,
) -> tuple[list[int], dict[int, dict[int, float]]]:
    """Identify codes with significant membership in multiple communities.

    Args:
        membership_probs: Soft membership [num_codes, n_communities].
        threshold: Minimum membership to be considered significant.

    Returns:
        Tuple of (overlapping_codes, overlap_stats):
        - overlapping_codes: List of code indices with multi-community membership.
        - overlap_stats: Dict mapping code -> {community: membership_prob}.
    """
    overlapping_codes = []
    overlap_stats = {}

    for code_idx in range(membership_probs.shape[0]):
        probs = membership_probs[code_idx]
        significant_communities = np.where(probs >= threshold)[0]

        if len(significant_communities) >= 2:
            overlapping_codes.append(code_idx)
            overlap_stats[code_idx] = {
                int(comm): float(probs[comm]) for comm in significant_communities
            }

    return overlapping_codes, overlap_stats


def compute_modularity(
    trans_probs: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute Newman modularity of the clustering.

    Args:
        trans_probs: Transition probability matrix [num_codes, num_codes].
        labels: Community labels for each code [num_codes].

    Returns:
        Modularity score in range [-0.5, 1].
    """
    num_codes = trans_probs.shape[0]

    # Convert to counts (unnormalized)
    # Use symmetric version
    adj = (trans_probs + trans_probs.T) / 2

    # Total edge weight
    m = np.sum(adj) / 2

    if m == 0:
        return 0.0

    # Degree of each node
    k = np.sum(adj, axis=1)

    # Compute modularity
    Q = 0.0
    for i in range(num_codes):
        for j in range(num_codes):
            if labels[i] == labels[j]:
                Q += adj[i, j] - (k[i] * k[j]) / (2 * m)

    return float(Q / (2 * m))


def compute_coarsened_transitions(
    trans_probs: np.ndarray,
    membership_probs: np.ndarray,
    n_communities: int,
) -> np.ndarray:
    """Compute community-level transition matrix weighted by soft membership.

    Args:
        trans_probs: Code-level transition probabilities [num_codes, num_codes].
        membership_probs: Soft membership [num_codes, n_communities].
        n_communities: Number of communities.

    Returns:
        Coarsened transition matrix [n_communities, n_communities].
    """
    # Weight transitions by membership probabilities
    # coarsened[c1, c2] = sum over codes i,j of P(c1|i) * trans[i,j] * P(c2|j)
    coarsened = membership_probs.T @ trans_probs @ membership_probs

    # Normalize rows to get probabilities
    row_sums = coarsened.sum(axis=1, keepdims=True)
    coarsened = np.where(row_sums > 0, coarsened / row_sums, 0.0)

    return coarsened


def build_community_structure(
    trans_probs: np.ndarray,
    trans_counts: np.ndarray,
    labels: np.ndarray,
    membership_probs: np.ndarray,
    overlap_threshold: float = 0.2,
) -> CommunityStructure:
    """Build complete community structure from clustering results.

    Args:
        trans_probs: Transition probability matrix.
        trans_counts: Transition count matrix.
        labels: Hard community assignments [num_codes].
        membership_probs: Soft membership probabilities [num_codes, n_communities].
        overlap_threshold: Threshold for identifying overlapping codes.

    Returns:
        CommunityStructure with all analysis results.
    """
    num_codes = trans_probs.shape[0]
    n_communities = membership_probs.shape[1]

    # Build code_to_community (hard assignment)
    code_to_community = {int(i): int(labels[i]) for i in range(num_codes)}

    # Identify overlapping codes
    overlapping_codes, overlap_stats = identify_overlapping_codes(
        membership_probs, overlap_threshold
    )

    # Build communities
    communities = []
    for comm_id in range(n_communities):
        # Codes with this as primary community
        code_indices = [i for i in range(num_codes) if labels[i] == comm_id]

        # Core vs boundary codes
        core_codes = [c for c in code_indices if c not in overlapping_codes]
        boundary_codes = [c for c in code_indices if c in overlapping_codes]

        # Internal transition rate
        internal_transitions = 0
        total_transitions = 0
        for i in code_indices:
            for j in range(num_codes):
                count = trans_counts[i, j]
                total_transitions += count
                if j in code_indices:
                    internal_transitions += count

        internal_rate = (
            internal_transitions / total_transitions if total_transitions > 0 else 0.0
        )

        community = Community(
            id=comm_id,
            code_indices=code_indices,
            core_codes=core_codes,
            boundary_codes=boundary_codes,
            internal_transition_rate=float(internal_rate),
        )
        communities.append(community)

    # Compute coarsened transitions
    coarsened = compute_coarsened_transitions(trans_probs, membership_probs, n_communities)

    # Compute modularity
    modularity = compute_modularity(trans_probs, labels)

    return CommunityStructure(
        communities=communities,
        code_to_community=code_to_community,
        code_membership_probs=membership_probs,
        overlapping_codes=overlapping_codes,
        overlap_stats=overlap_stats,
        coarsened_transitions=coarsened,
        modularity=modularity,
        n_communities=n_communities,
    )


def run_community_analysis(
    results: Sequence[InferenceResult],
    num_codes: int,
    output_dir: str | Path,
    n_communities: int | None = None,
    overlap_threshold: float = 0.2,
) -> tuple[CommunityStructure, dict[str, str]]:
    """Run complete community analysis pipeline.

    Args:
        results: List of InferenceResult with code_indices.
        num_codes: Total number of codes.
        output_dir: Directory to save outputs.
        n_communities: Number of communities (None = auto-detect).
        overlap_threshold: Threshold for identifying overlapping codes.

    Returns:
        Tuple of (CommunityStructure, paths_dict).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Computing transition matrix...")
    trans_counts, trans_probs = compute_transition_matrix(results, num_codes)

    logging.info("Discovering communities via spectral clustering...")
    labels, embedding, centroids = discover_communities(trans_probs, n_communities)

    logging.info("Computing soft membership probabilities...")
    membership_probs = compute_soft_membership(embedding, centroids)

    logging.info("Building community structure...")
    structure = build_community_structure(
        trans_probs=trans_probs,
        trans_counts=trans_counts,
        labels=labels,
        membership_probs=membership_probs,
        overlap_threshold=overlap_threshold,
    )

    logging.info(f"Found {structure.n_communities} communities")
    logging.info(f"  Modularity: {structure.modularity:.3f}")
    logging.info(f"  Overlapping codes: {len(structure.overlapping_codes)}")

    # Log community summaries
    for comm in structure.communities:
        logging.info(
            f"  Community {comm.id}: {len(comm.code_indices)} codes "
            f"({len(comm.core_codes)} core, {len(comm.boundary_codes)} boundary), "
            f"internal_rate={comm.internal_transition_rate:.2f}"
        )

    # Save results
    paths = save_community_analysis(output_dir, structure, trans_probs, membership_probs)

    return structure, paths


def save_community_analysis(
    output_dir: Path,
    structure: CommunityStructure,
    trans_probs: np.ndarray,
    membership_probs: np.ndarray,
) -> dict[str, str]:
    """Save community analysis results to files.

    Args:
        output_dir: Directory to save outputs.
        structure: CommunityStructure results.
        trans_probs: Transition probability matrix.
        membership_probs: Soft membership probabilities.

    Returns:
        Dictionary mapping output names to file paths.
    """
    paths = {}

    # Save community structure as JSON
    structure_data = {
        "n_communities": structure.n_communities,
        "modularity": structure.modularity,
        "overlapping_codes": structure.overlapping_codes,
        "overlap_stats": structure.overlap_stats,
        "code_to_community": structure.code_to_community,
        "communities": [asdict(c) for c in structure.communities],
    }
    json_path = output_dir / "community_structure.json"
    with open(json_path, "w") as f:
        json.dump(structure_data, f, indent=2)
    paths["structure"] = str(json_path)

    # Save membership probabilities as NPZ
    npz_path = output_dir / "community_membership.npz"
    np.savez(
        npz_path,
        membership_probs=membership_probs,
        coarsened_transitions=structure.coarsened_transitions,
        trans_probs=trans_probs,
    )
    paths["membership"] = str(npz_path)

    logging.info(f"Saved community analysis to {output_dir}")
    return paths
