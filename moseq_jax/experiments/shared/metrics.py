"""Analysis metrics: transition matrices, divergence, reward decomposition."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .plotting import set_nature_style

# ---------------------------------------------------------------------------
# Transition matrix
# ---------------------------------------------------------------------------


def compute_transition_matrix(
    code_arrays: list[np.ndarray],
    num_codes: int,
) -> np.ndarray:
    """Count-based transition matrix ``[num_codes, num_codes]``.

    Args:
        code_arrays: List of 1-D code index sequences.
        num_codes: Codebook size.

    Returns:
        Integer count matrix.
    """
    T = np.zeros((num_codes, num_codes), dtype=np.int64)
    for seq in code_arrays:
        for t in range(len(seq) - 1):
            T[seq[t], seq[t + 1]] += 1
    return T


def plot_transition_matrix(
    T: np.ndarray,
    title: str = "Transition Matrix",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Log-scale heatmap of a transition matrix (Nature style)."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    T_plot = T.astype(float)
    T_plot[T_plot == 0] = np.nan
    im = ax.imshow(
        np.log10(T_plot + 1),
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    ax.set_xlabel("To code")
    ax.set_ylabel("From code")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log10(count + 1)")
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Pairwise joint divergence
# ---------------------------------------------------------------------------


def compute_pairwise_joint_divergence(
    trajectories_qpos: list[np.ndarray],
    root_dof: int = 7,
) -> np.ndarray:
    """Mean pairwise joint-L2 distance over time across K trajectories.

    Args:
        trajectories_qpos: ``K`` arrays each ``[T_i, nq]``.
        root_dof: Number of root DOFs to skip (default 7 for free-body).

    Returns:
        Array ``[min_T]`` of mean pairwise L2 distances.
    """
    K = len(trajectories_qpos)
    min_T = min(len(q) for q in trajectories_qpos)
    pair_dists: list[np.ndarray] = []
    for i in range(K):
        for j in range(i + 1, K):
            qi = trajectories_qpos[i][:min_T, root_dof:]
            qj = trajectories_qpos[j][:min_T, root_dof:]
            pair_dists.append(np.linalg.norm(qi - qj, axis=1))
    if not pair_dists:
        return np.zeros(min_T)
    return np.mean(pair_dists, axis=0)


# ---------------------------------------------------------------------------
# Reward decomposition
# ---------------------------------------------------------------------------

COARSE_TERMS = ("root_pos", "root_quat", "torso_z_range")
FINE_TERMS = ("joints", "end_eff")
PENALTY_TERMS = ("control_cost", "control_diff_cost", "energy_cost")


def decompose_rewards(
    per_step_metrics: list[dict[str, float]],
) -> dict[str, np.ndarray]:
    """Extract coarse / fine / penalty reward curves from per-step metrics.

    Args:
        per_step_metrics: List of dicts, one per timestep, containing
            keys like ``"rewards/root_pos"``, ``"rewards/joints"``, etc.

    Returns:
        Dict with keys ``"total"``, ``"coarse"``, ``"fine"``, ``"penalty"``
        each mapping to a ``[T]`` float array.
    """
    T = len(per_step_metrics)
    total = np.zeros(T)
    coarse = np.zeros(T)
    fine = np.zeros(T)
    penalty = np.zeros(T)

    for t, m in enumerate(per_step_metrics):
        for term in COARSE_TERMS:
            key = f"rewards/{term}"
            if key in m:
                coarse[t] += float(m[key])
        for term in FINE_TERMS:
            key = f"rewards/{term}"
            if key in m:
                fine[t] += float(m[key])
        for term in PENALTY_TERMS:
            key = f"rewards/{term}"
            if key in m:
                penalty[t] += float(m[key])
        total[t] = coarse[t] + fine[t] + penalty[t]

    return {"total": total, "coarse": coarse, "fine": fine, "penalty": penalty}


# ---------------------------------------------------------------------------
# Transition-boundary analysis
# ---------------------------------------------------------------------------


def compute_transition_window_rewards(
    code_indices: np.ndarray,
    rewards: np.ndarray,
    window: int = 25,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Mean reward trajectory in a window around code transitions.

    Args:
        code_indices: ``[T]`` code index array.
        rewards: ``[T]`` reward array.
        window: Half-window size (frames before and after transition).

    Returns:
        ``(mean_curve, std_curve, n_transitions)`` where curves have
        shape ``[2*window+1]``.
    """
    T = len(code_indices)
    transitions = np.where(code_indices[1:] != code_indices[:-1])[0] + 1
    # Filter transitions that have full window
    valid = transitions[(transitions >= window) & (transitions < T - window)]

    if len(valid) == 0:
        return np.zeros(2 * window + 1), np.zeros(2 * window + 1), 0

    windows = np.array([rewards[t - window : t + window + 1] for t in valid])
    return windows.mean(axis=0), windows.std(axis=0), len(valid)
