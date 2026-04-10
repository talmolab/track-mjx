"""DSA (Dynamical Similarity Analysis) of RNN hidden states per behavior.

Loads hidden_dynamics.npz (from run_hidden_dynamics), splits each behavior's
K=10 clips into two random groups of 5, fits one DSA model per group
(6 systems total: 2 per behavior), and compares all pairs.

This split-half design lets DSA compare *systems* with multiple trials:
  - Within-behavior: walk_A vs walk_B, groom_A vs groom_B, rear_A vs rear_B
  - Between-behavior: all cross-behavior pairs

Produces:
  - 6x6 DSA distance heatmap
  - Within-behavior vs between-behavior bar chart

Requires: ``pip install dsa-metric``

Adapted from mech-complexity/dsa.py (Charles Xu).

Usage:
    cd moseq_jax/figures
    python plot_hidden_dsa.py                 # compute + plot
    python plot_hidden_dsa.py --replot        # replot from cached distances
    python plot_hidden_dsa.py --n_delays 10 --rank 15
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "hidden_dynamics"

# ── Behavior config ─────────────────────────────────────────────────────────
BEHAVIORS = ["walk", "groom", "rear"]
BEHAVIOR_LABELS = {
    "walk": "Walking",
    "groom": "Grooming",
    "rear": "Rearing",
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


# ── Data loading + split-half ────────────────────────────────────────────────


def load_and_split(seed: int = 42) -> tuple[list[np.ndarray], list[str], list[str]]:
    """Load hidden states and split each behavior into two random groups.

    Returns:
        systems: list of 6 arrays, each ``[K//2, T, hidden_dim]``
        names: list of 6 labels (e.g. ``"walk_A"``, ``"walk_B"``)
        behaviors: list of 6 behavior categories
    """
    raw = np.load(DATA_DIR / "hidden_dynamics.npz", allow_pickle=True)
    rng = np.random.default_rng(seed)

    systems = []
    names = []
    behaviors = []

    for beh in BEHAVIORS:
        key = f"hidden_{beh}"
        if key not in raw:
            continue
        arr = np.array(raw[key])  # [K, T, hidden_dim]
        K = arr.shape[0]
        indices = rng.permutation(K)
        half = K // 2

        group_a = arr[indices[:half]]  # [half, T, hidden_dim]
        group_b = arr[indices[half : 2 * half]]

        systems.append(group_a)
        names.append(f"{beh}_A")
        behaviors.append(beh)

        systems.append(group_b)
        names.append(f"{beh}_B")
        behaviors.append(beh)

    return systems, names, behaviors


# ── DSA computation ──────────────────────────────────────────────────────────


def compute_dsa_matrix(
    systems: list[np.ndarray],
    n_delays: int = DEFAULT_N_DELAYS,
    rank: int = DEFAULT_RANK,
    score_method: str = DEFAULT_SCORE_METHOD,
) -> np.ndarray:
    """Compute pairwise DSA distances between split-half systems."""
    from DSA import DSA as PlainDSA

    print(f"  DSA config: n_delays={n_delays}, rank={rank}, method={score_method}")
    for i, (s, name) in enumerate(zip(systems, ["walk_A", "walk_B", "groom_A", "groom_B", "rear_A", "rear_B"])):
        print(f"    {name}: {s.shape}")

    dsa = PlainDSA(
        X=systems,
        n_delays=n_delays,
        rank=rank,
        score_method=score_method,
        device="cpu",
        verbose=True,
    )
    result = np.array(dsa.fit_score())

    # Handle multi-component output
    if result.ndim == 3:
        return result[:, :, 0]
    return result


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_heatmap(
    dist_matrix: np.ndarray,
    names: list[str],
    behaviors: list[str],
) -> tuple[plt.Figure, mpatches.FancyBboxPatch]:
    """6x6 DSA distance heatmap with behavior block annotations."""
    n = len(names)
    fig, ax = plt.subplots(figsize=(3.8, 3.4))

    im = ax.imshow(dist_matrix, cmap="viridis", aspect="equal", interpolation="none")

    # Block boundaries between behaviors (every 2 systems)
    for b in [2, 4]:
        ax.axhline(y=b - 0.5, color="white", linewidth=1.5)
        ax.axvline(x=b - 0.5, color="white", linewidth=1.5)

    # Tick labels
    tick_labels = []
    for name in names:
        beh = name.split("_")[0]
        group = name.split("_")[1]
        tick_labels.append(f"{BEHAVIOR_LABELS[beh]} {group}")

    ax.set_xticks(range(n))
    ax.set_xticklabels(tick_labels, fontsize=5.5, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(tick_labels, fontsize=5.5)

    # Annotate cells with values
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(
                    j, i, f"{dist_matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=4.5,
                    color="white" if dist_matrix[i, j] < dist_matrix.max() * 0.6 else "black",
                )

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("DSA Distance", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    ax.set_title(
        "DSA Distance Matrix (Split-Half)",
        fontsize=8, fontweight="bold", pad=6,
    )

    fig.tight_layout()
    rect = _add_rounded_border(fig)
    return fig, rect


def plot_within_between(
    dist_matrix: np.ndarray,
    names: list[str],
    behaviors: list[str],
) -> tuple[plt.Figure, mpatches.FancyBboxPatch]:
    """Bar chart: within-behavior vs between-behavior DSA distance."""
    n = len(names)

    within_dists = []
    between_dists = []

    for i in range(n):
        for j in range(i + 1, n):
            d = dist_matrix[i, j]
            if behaviors[i] == behaviors[j]:
                within_dists.append(d)
            else:
                between_dists.append(d)

    within_arr = np.array(within_dists)
    between_arr = np.array(between_dists)

    fig, ax = plt.subplots(figsize=(2.8, 3.0))

    means = [within_arr.mean(), between_arr.mean()]
    sems = [
        within_arr.std(ddof=1) / np.sqrt(len(within_arr)) if len(within_arr) > 1 else 0,
        between_arr.std(ddof=1) / np.sqrt(len(between_arr)) if len(between_arr) > 1 else 0,
    ]

    bar_colors = ["#3498db", "#e74c3c"]
    x = np.array([0, 1])
    bars = ax.bar(
        x, means, yerr=sems, color=bar_colors, alpha=0.8,
        width=0.55, capsize=4, edgecolor="none",
        error_kw={"linewidth": 0.8},
    )

    # Strip plot
    rng = np.random.default_rng(42)
    jitter_w = rng.uniform(-0.12, 0.12, size=len(within_arr))
    jitter_b = rng.uniform(-0.12, 0.12, size=len(between_arr))
    ax.scatter(
        0 + jitter_w, within_arr,
        c=bar_colors[0], s=15, alpha=0.6, edgecolors="none", zorder=3,
    )
    ax.scatter(
        1 + jitter_b, between_arr,
        c=bar_colors[1], s=15, alpha=0.6, edgecolors="none", zorder=3,
    )

    # Annotations
    for bar, m, s in zip(bars, means, sems):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + s + 0.02 * max(means),
            f"{m:.2f}",
            ha="center", va="bottom", fontsize=6.5,
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split-half")
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
        # Load and split
        systems, names, behaviors = load_and_split(seed=args.seed)
        print(f"Split-half: {len(systems)} systems")
        for name, s in zip(names, systems):
            print(f"  {name}: {s.shape}")

        # Compute DSA
        print("\nComputing DSA distances...")
        dist_matrix = compute_dsa_matrix(
            systems,
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
    n = len(names)
    within_dists = []
    between_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = dist_matrix[i, j]
            if behaviors[i] == behaviors[j]:
                within_dists.append(d)
                print(f"  WITHIN  {names[i]} vs {names[j]}: {d:.4f}")
            else:
                between_dists.append(d)

    print(f"\n  Within-behavior mean:  {np.mean(within_dists):.4f} (n={len(within_dists)})")
    print(f"  Between-behavior mean: {np.mean(between_dists):.4f} (n={len(between_dists)})")

    # Plot
    print("\nGenerating figures...")
    fig, rect = plot_heatmap(dist_matrix, names, behaviors)
    _save_figure(fig, rect, "dsa_heatmap")

    fig, rect = plot_within_between(dist_matrix, names, behaviors)
    _save_figure(fig, rect, "dsa_within_between")

    print("\nDone.")


if __name__ == "__main__":
    main()
