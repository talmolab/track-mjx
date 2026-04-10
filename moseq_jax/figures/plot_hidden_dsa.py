"""DSA (Dynamical Similarity Analysis) of RNN hidden states per behavior.

Loads hidden_dynamics.npz (from run_hidden_dynamics), treats each clip as
a separate "system" (30 total: 10 walk + 10 groom + 10 rear), fits a linear
dynamical system via delay-embedded DMD, and compares all pairs.

Produces:
  - 30x30 DSA distance heatmap showing block-diagonal structure
  - Within-behavior vs between-behavior bar chart

If the ``dsa-metric`` package is installed (``pip install dsa-metric``),
uses the full Procrustes-aligned DSA metric.  Otherwise falls back to a
lightweight eigenvalue-spectrum comparison (Wasserstein on eigenvalues
in the complex plane).

Adapted from mech-complexity/dsa.py (Charles Xu).

Usage:
    cd moseq_jax/figures
    python plot_hidden_dsa.py            # compute + plot
    python plot_hidden_dsa.py --replot   # replot from cached distances
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import linear_sum_assignment

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "hidden_dynamics"

# ── Behavior config ─────────────────────────────────────────────────────────
BEHAVIORS = ["walk", "groom", "rear"]
BEHAVIOR_LABELS = {
    "walk": "Walking Code",
    "groom": "Grooming Code",
    "rear": "Rearing Code",
}
BEHAVIOR_COLORS = {
    "walk": "#D55E00",
    "groom": "#0072B2",
    "rear": "#009E73",
}

# ── DSA defaults ─────────────────────────────────────────────────────────────
DEFAULT_N_DELAYS = 5
DEFAULT_RANK = 10
DEFAULT_SCORE_METHOD = "angular"


# ── Style ────────────────────────────────────────────────────────────────────


def _setup_nature_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _add_rounded_border(fig: plt.Figure) -> mpatches.FancyBboxPatch:
    fig.patch.set_facecolor("white")
    rect = mpatches.FancyBboxPatch(
        (0.005, 0.005),
        0.99,
        0.99,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        transform=fig.transFigure,
        facecolor="white",
        edgecolor="#cccccc",
        linewidth=0.6,
        zorder=-1,
    )
    fig.patches.append(rect)
    return rect


# ── Data loading ─────────────────────────────────────────────────────────────


def load_hidden_trajectories() -> tuple[list[np.ndarray], list[str], list[str]]:
    """Load hidden states and return as list of individual trajectories.

    Returns:
        trajectories: list of 30 arrays, each ``[T, hidden_dim]``
        names: list of 30 labels (e.g. ``"walk_0"``, ``"groom_3"``)
        behaviors: list of 30 behavior categories
    """
    raw = np.load(DATA_DIR / "hidden_dynamics.npz", allow_pickle=True)
    trajectories = []
    names = []
    behaviors = []

    for beh in BEHAVIORS:
        key = f"hidden_{beh}"
        if key not in raw:
            continue
        arr = np.array(raw[key])  # [K, T, hidden_dim]
        for ki in range(arr.shape[0]):
            trajectories.append(arr[ki])  # [T, hidden_dim]
            names.append(f"{beh}_{ki}")
            behaviors.append(beh)

    return trajectories, names, behaviors


def group_by_behavior(
    behaviors: list[str],
) -> tuple[list[str], dict[str, list[int]]]:
    """Group trajectory indices by behavior, preserving BEHAVIORS order."""
    beh_indices: dict[str, list[int]] = {}
    for i, beh in enumerate(behaviors):
        if beh not in beh_indices:
            beh_indices[beh] = []
        beh_indices[beh].append(i)

    ordered = [b for b in BEHAVIORS if b in beh_indices]
    return ordered, beh_indices


# ═════════════════════════════════════════════════════════════════════════════
# DSA computation
# ═════════════════════════════════════════════════════════════════════════════


def _try_dsa_package(
    trajectories: list[np.ndarray],
    n_delays: int,
    rank: int,
    score_method: str,
) -> np.ndarray | None:
    """Try to compute DSA using the dsa-metric package. Returns None if unavailable."""
    try:
        from DSA import DSA as PlainDSA

        print("  Using DSA package (dsa-metric)")
        dsa = PlainDSA(
            X=trajectories,
            n_delays=n_delays,
            rank=rank,
            score_method=score_method,
            device="cpu",
            verbose=True,
        )
        return np.array(dsa.fit_score())
    except ImportError:
        return None


# ── Fallback: manual delay-embed DMD + eigenvalue comparison ─────────────


def _delay_embed(X: np.ndarray, n_delays: int) -> np.ndarray:
    """Construct delay-embedded (Hankel) matrix.

    Args:
        X: ``[T, N]`` time series.
        n_delays: number of delays.

    Returns:
        ``[T - n_delays + 1, n_delays * N]`` Hankel matrix.
    """
    T, N = X.shape
    rows = T - n_delays + 1
    H = np.zeros((rows, n_delays * N), dtype=X.dtype)
    for d in range(n_delays):
        H[:, d * N : (d + 1) * N] = X[d : d + rows]
    return H


def _fit_dmd(X: np.ndarray, n_delays: int, rank: int) -> np.ndarray:
    """Fit a linear dynamical system via delay-embedded DMD.

    Returns:
        A: ``[rank, rank]`` dynamics matrix.
    """
    H = _delay_embed(X, n_delays)
    # Center
    H = H - H.mean(axis=0, keepdims=True)
    # SVD truncation
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vt_r = Vt[:rank]
    # Project into reduced space
    X_r = U_r * S_r  # [T', rank]
    # Fit A: X_r[1:] ≈ X_r[:-1] @ A.T  →  A = X_r[1:].T @ X_r[:-1] @ inv(...)
    X1 = X_r[:-1]
    X2 = X_r[1:]
    A = np.linalg.lstsq(X1, X2, rcond=None)[0].T  # [rank, rank]
    return A


def _eigenvalue_wasserstein(eig1: np.ndarray, eig2: np.ndarray) -> float:
    """Wasserstein distance between eigenvalue sets in the complex plane."""
    n = len(eig1)
    m = len(eig2)
    cost = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            cost[i, j] = abs(eig1[i] - eig2[j])
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(cost[row_ind, col_ind].sum() / max(n, m))


def _compute_dsa_fallback(
    trajectories: list[np.ndarray],
    n_delays: int,
    rank: int,
) -> np.ndarray:
    """Compute DSA distance matrix via manual DMD + eigenvalue comparison."""
    print("  DSA package not available, using eigenvalue-spectrum fallback")
    n = len(trajectories)

    # Fit DMD for each trajectory
    print(f"  Fitting {n} DMD models (n_delays={n_delays}, rank={rank})...")
    A_matrices = []
    eigenvalues = []
    for i, traj in enumerate(trajectories):
        A = _fit_dmd(traj.astype(np.float64), n_delays, rank)
        A_matrices.append(A)
        eigenvalues.append(np.linalg.eigvals(A))
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{n} done")

    # Pairwise eigenvalue Wasserstein distance
    print(f"  Computing {n * (n - 1) // 2} pairwise distances...")
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _eigenvalue_wasserstein(eigenvalues[i], eigenvalues[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return dist_matrix


def compute_dsa_matrix(
    trajectories: list[np.ndarray],
    n_delays: int = DEFAULT_N_DELAYS,
    rank: int = DEFAULT_RANK,
    score_method: str = DEFAULT_SCORE_METHOD,
) -> np.ndarray:
    """Compute pairwise DSA distances. Tries dsa-metric, falls back to manual."""
    result = _try_dsa_package(trajectories, n_delays, rank, score_method)
    if result is not None:
        # DSA package returns [n, n] or [n, n, k]; handle both
        if result.ndim == 3:
            return result[:, :, 0]  # take joint distance
        return result
    return _compute_dsa_fallback(trajectories, n_delays, rank)


# ═════════════════════════════════════════════════════════════════════════════
# Plotting (adapted from mech-complexity/dsa.py)
# ═════════════════════════════════════════════════════════════════════════════


def plot_heatmap(
    dist_matrix: np.ndarray,
    names: list[str],
    behaviors: list[str],
) -> tuple[plt.Figure, mpatches.FancyBboxPatch]:
    """30x30 DSA distance heatmap with behavior block annotations."""
    beh_order, beh_indices = group_by_behavior(behaviors)
    n = len(names)

    # Reorder by behavior
    order = []
    for beh in beh_order:
        order.extend(beh_indices[beh])
    reordered = dist_matrix[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    im = ax.imshow(reordered, cmap="viridis", aspect="equal", interpolation="none")

    # Behavior block boundaries + labels
    boundaries = []
    tick_positions = []
    offset = 0
    for beh in beh_order:
        k = len(beh_indices[beh])
        tick_positions.append(offset + k / 2 - 0.5)
        offset += k
        if beh != beh_order[-1]:
            boundaries.append(offset - 0.5)

    for b in boundaries:
        ax.axhline(y=b, color="white", linewidth=1.2)
        ax.axvline(x=b, color="white", linewidth=1.2)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [BEHAVIOR_LABELS[b] for b in beh_order], fontsize=6,
    )
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(
        [BEHAVIOR_LABELS[b] for b in beh_order], fontsize=6,
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("DSA Distance", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    ax.set_title(
        "DSA Distance Matrix",
        fontsize=8, fontweight="bold", pad=6,
    )

    fig.tight_layout()
    rect = _add_rounded_border(fig)
    return fig, rect


def plot_within_between(
    dist_matrix: np.ndarray,
    behaviors: list[str],
) -> tuple[plt.Figure, mpatches.FancyBboxPatch]:
    """Bar chart: within-behavior vs between-behavior DSA distance."""
    beh_order, beh_indices = group_by_behavior(behaviors)

    within_dists = []
    between_dists = []

    for beh in beh_order:
        indices = beh_indices[beh]
        for ii in range(len(indices)):
            for jj in range(ii + 1, len(indices)):
                within_dists.append(dist_matrix[indices[ii], indices[jj]])

    for bi in range(len(beh_order)):
        for bj in range(bi + 1, len(beh_order)):
            for idx_i in beh_indices[beh_order[bi]]:
                for idx_j in beh_indices[beh_order[bj]]:
                    between_dists.append(dist_matrix[idx_i, idx_j])

    within_arr = np.array(within_dists)
    between_arr = np.array(between_dists)

    fig, ax = plt.subplots(figsize=(2.8, 3.0))

    means = [within_arr.mean(), between_arr.mean()]
    sems = [
        within_arr.std(ddof=1) / np.sqrt(len(within_arr)),
        between_arr.std(ddof=1) / np.sqrt(len(between_arr)),
    ]

    bar_colors = ["#3498db", "#e74c3c"]
    x = np.array([0, 1])
    bars = ax.bar(
        x, means, yerr=sems, color=bar_colors, alpha=0.8,
        width=0.55, capsize=4, edgecolor="none",
        error_kw={"linewidth": 0.8},
    )

    # Strip plot (individual pairwise distances)
    rng = np.random.default_rng(42)
    jitter_w = rng.uniform(-0.12, 0.12, size=len(within_arr))
    jitter_b = rng.uniform(-0.12, 0.12, size=len(between_arr))
    ax.scatter(
        0 + jitter_w, within_arr,
        c=bar_colors[0], s=4, alpha=0.3, edgecolors="none", zorder=3,
    )
    ax.scatter(
        1 + jitter_b, between_arr,
        c=bar_colors[1], s=4, alpha=0.3, edgecolors="none", zorder=3,
    )

    # Annotations
    for bar, m, s in zip(bars, means, sems):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + s + 0.01 * ax.get_ylim()[1],
            f"{m:.3f}",
            ha="center", va="bottom", fontsize=6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["Within\nbehavior", "Between\nbehavior"], fontsize=7)
    ax.set_ylabel("DSA Distance", fontsize=7)
    ax.set_ylim(bottom=0)

    for ytick in ax.get_yticks():
        if ytick > 0:
            ax.axhline(ytick, color="#e0e0e0", linewidth=0.3, zorder=0)

    ax.set_title(
        "Dynamical Similarity",
        fontsize=8, fontweight="bold", pad=6,
    )

    fig.tight_layout()
    rect = _add_rounded_border(fig)
    return fig, rect


# ── Save helper ──────────────────────────────────────────────────────────────


def _save_figure(
    fig: plt.Figure,
    rect: mpatches.FancyBboxPatch,
    stem: str,
) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf")
    fig.savefig(OUTPUT_DIR / f"{stem}.png")
    rect.set_facecolor("none")
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", transparent=True)
    rect.set_facecolor("white")
    plt.close(fig)
    print(f"  Saved: {stem}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="DSA analysis of RNN hidden states")
    parser.add_argument("--replot", action="store_true", help="Replot from cached distances")
    parser.add_argument("--n_delays", type=int, default=DEFAULT_N_DELAYS)
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument("--score_method", type=str, default=DEFAULT_SCORE_METHOD)
    args = parser.parse_args()

    _setup_nature_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cache_path = DATA_DIR / "dsa_distances.npz"

    if args.replot and cache_path.exists():
        print("Loading cached DSA distances...")
        cached = np.load(cache_path, allow_pickle=True)
        dist_matrix = cached["dist_matrix"]
        names = list(cached["names"])
        behaviors = list(cached["behaviors"])
    else:
        # Load data
        trajectories, names, behaviors = load_hidden_trajectories()
        print(
            f"Loaded {len(trajectories)} trajectories, "
            f"shape: {trajectories[0].shape}"
        )
        for beh in BEHAVIORS:
            n = sum(1 for b in behaviors if b == beh)
            print(f"  {beh}: {n} clips")

        # Compute DSA
        print("\nComputing DSA distances...")
        dist_matrix = compute_dsa_matrix(
            trajectories,
            n_delays=args.n_delays,
            rank=args.rank,
            score_method=args.score_method,
        )

        # Cache
        np.savez_compressed(
            cache_path,
            dist_matrix=dist_matrix,
            names=np.array(names),
            behaviors=np.array(behaviors),
            n_delays=args.n_delays,
            rank=args.rank,
        )
        print(f"Cached to: {cache_path}")

    # Print summary
    beh_order, beh_indices = group_by_behavior(behaviors)
    print("\nDSA Distance Summary:")
    for bi, beh_i in enumerate(beh_order):
        within = []
        for ii in range(len(beh_indices[beh_i])):
            for jj in range(ii + 1, len(beh_indices[beh_i])):
                within.append(
                    dist_matrix[beh_indices[beh_i][ii], beh_indices[beh_i][jj]]
                )
        print(f"  {beh_i} within: {np.mean(within):.4f} +/- {np.std(within):.4f}")

    for bi in range(len(beh_order)):
        for bj in range(bi + 1, len(beh_order)):
            between = []
            for idx_i in beh_indices[beh_order[bi]]:
                for idx_j in beh_indices[beh_order[bj]]:
                    between.append(dist_matrix[idx_i, idx_j])
            print(
                f"  {beh_order[bi]} vs {beh_order[bj]}: "
                f"{np.mean(between):.4f} +/- {np.std(between):.4f}"
            )

    # Plot
    print("\nGenerating figures...")
    fig, rect = plot_heatmap(dist_matrix, names, behaviors)
    _save_figure(fig, rect, "dsa_heatmap")

    fig, rect = plot_within_between(dist_matrix, behaviors)
    _save_figure(fig, rect, "dsa_within_between")

    print("\nDone.")


if __name__ == "__main__":
    main()
