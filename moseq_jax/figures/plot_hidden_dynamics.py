"""RNN hidden state dynamics visualization (3D PCA).

Creates 2 Nature-style figures, each with 3 panels (Walk | Groom | Rear):
  - Scatter: point clouds, one behavior per panel, shared PCA axes
  - Trajectory: lines colored by time, one behavior per panel, shared PCA axes

PCA is fitted on ALL behaviors jointly so the coordinate system is shared.

Usage:
    cd moseq_jax/figures
    python plot_hidden_dynamics.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.lines import Line2D
import numpy as np
from sklearn.decomposition import PCA

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
    "walk": "#D55E00",  # orange (Wong)
    "groom": "#0072B2",  # blue (Wong)
    "rear": "#009E73",  # green (Wong)
}

# View angle
VIEW_ELEV = 20
VIEW_AZIM = 140


# ── Style ────────────────────────────────────────────────────────────────────


def _setup_nature_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 6,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
            "legend.fontsize": 6,
            "axes.linewidth": 0.5,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _style_3d_ax(
    ax: plt.Axes,
    var_explained: np.ndarray,
    xlim: tuple,
    ylim: tuple,
    zlim: tuple,
    show_ylabel: bool = True,
) -> None:
    """Apply clean Nature-style formatting to a 3D axes."""
    ax.set_xlabel(f"PC 1 ({var_explained[0]:.1%})", labelpad=0, fontsize=6)
    if show_ylabel:
        ax.set_ylabel(f"PC 2 ({var_explained[1]:.1%})", labelpad=0, fontsize=6)
    else:
        ax.set_ylabel("")
    ax.set_zlabel(f"PC 3 ({var_explained[2]:.1%})", labelpad=0, fontsize=6)

    ax.tick_params(axis="both", which="major", pad=0, labelsize=4.5)
    ax.tick_params(axis="z", pad=0)

    # Shared limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    # Transparent panes with subtle edges
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#d5d5d5")
        pane.set_alpha(0.6)

    # Subtle grid
    ax.grid(True, linewidth=0.2, alpha=0.4, color="#cccccc")

    # Thin axis lines
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_linewidth(0.4)
        axis.line.set_color("#666666")

    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)


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


def _compute_shared_limits(
    emb_3d: np.ndarray, pad_frac: float = 0.05,
) -> tuple[tuple, tuple, tuple]:
    """Compute shared axis limits across all data."""
    lims = []
    for i in range(3):
        lo, hi = emb_3d[:, i].min(), emb_3d[:, i].max()
        d = (hi - lo) * pad_frac
        lims.append((lo - d, hi + d))
    return tuple(lims)


# ── Data loading ─────────────────────────────────────────────────────────────


def load_hidden_data() -> dict[str, np.ndarray]:
    """Load hidden dynamics NPZ.

    Returns:
        ``{behavior: ndarray [K, T, hidden_dim]}``.
    """
    raw = np.load(DATA_DIR / "hidden_dynamics.npz", allow_pickle=True)
    data = {}
    for beh in BEHAVIORS:
        key = f"hidden_{beh}"
        if key in raw:
            data[beh] = np.array(raw[key])
    return data


# ── Embedding ────────────────────────────────────────────────────────────────


def fit_pca_3d(
    all_points: np.ndarray, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit 3-component PCA. Returns ``([N, 3], variance_explained[3])``."""
    reducer = PCA(n_components=3, random_state=seed)
    result = reducer.fit_transform(all_points)
    var = reducer.explained_variance_ratio_
    print(
        f"  PCA variance explained: "
        f"PC1={var[0]:.1%}, PC2={var[1]:.1%}, PC3={var[2]:.1%} "
        f"(total={sum(var):.1%})"
    )
    return result, var


# ── Behavior-specific colormaps ──────────────────────────────────────────────


def _behavior_cmap(hex_color: str) -> mcolors.LinearSegmentedColormap:
    """Create a light-tint -> full-color sequential colormap."""
    rgb = mcolors.to_rgb(hex_color)
    light = tuple(0.80 + 0.20 * c for c in rgb)
    return mcolors.LinearSegmentedColormap.from_list("beh", [light, rgb])


BEHAVIOR_CMAPS = {beh: _behavior_cmap(c) for beh, c in BEHAVIOR_COLORS.items()}


# ── 3-panel scatter ──────────────────────────────────────────────────────────


def plot_scatter_3d(
    emb_3d: np.ndarray,
    beh_slices: dict[str, tuple[int, int]],
    var_explained: np.ndarray,
) -> tuple[plt.Figure, mpatches.FancyBboxPatch]:
    """3-panel scatter: Walk | Groom | Rear, shared PCA axes."""
    fig = plt.figure(figsize=(9.0, 3.2))
    xlim, ylim, zlim = _compute_shared_limits(emb_3d)

    for pi, beh in enumerate(BEHAVIORS):
        if beh not in beh_slices:
            continue
        ax = fig.add_subplot(1, 3, pi + 1, projection="3d")
        s, e = beh_slices[beh]
        pts = emb_3d[s:e]

        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            c=BEHAVIOR_COLORS[beh],
            s=6,
            alpha=0.20,
            edgecolors="none",
            rasterized=True,
            depthshade=True,
        )

        ax.set_title(
            BEHAVIOR_LABELS[beh],
            fontsize=8, fontweight="bold", pad=-2,
            color=BEHAVIOR_COLORS[beh],
        )
        _style_3d_ax(ax, var_explained, xlim, ylim, zlim, show_ylabel=(pi == 0))

    fig.suptitle(
        "PCA of RNN Hidden State",
        fontsize=9, fontweight="bold", y=0.98,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.88, wspace=0.05)
    rect = _add_rounded_border(fig)
    return fig, rect


# ── 3-panel trajectory ───────────────────────────────────────────────────────


def plot_trajectories_3d(
    emb_3d: np.ndarray,
    beh_slices: dict[str, tuple[int, int]],
    beh_shapes: dict[str, tuple[int, int]],
    var_explained: np.ndarray,
) -> tuple[plt.Figure, mpatches.FancyBboxPatch]:
    """3-panel trajectories: Walk | Groom | Rear, shared PCA axes.

    Each panel shows one behavior's trajectories colored by time
    (light -> dark). Open circle = start, filled = end.
    """
    fig = plt.figure(figsize=(9.0, 3.2))
    xlim, ylim, zlim = _compute_shared_limits(emb_3d)

    for pi, beh in enumerate(BEHAVIORS):
        if beh not in beh_slices:
            continue
        ax = fig.add_subplot(1, 3, pi + 1, projection="3d")
        K_beh, T_beh = beh_shapes[beh]
        s, _ = beh_slices[beh]
        cmap = BEHAVIOR_CMAPS[beh]
        t_norm = np.linspace(0, 1, T_beh)

        for ki in range(K_beh):
            start = s + ki * T_beh
            pts = emb_3d[start : start + T_beh]

            if len(pts) < 2:
                continue

            segments = np.array(
                [[pts[i], pts[i + 1]] for i in range(len(pts) - 1)],
            )
            colors = cmap(t_norm[:-1])
            lc = Line3DCollection(
                segments,
                colors=colors,
                linewidths=0.8,
                alpha=0.65,
            )
            ax.add_collection3d(lc)

            # Start (open) and end (filled)
            ax.scatter(
                [pts[0, 0]], [pts[0, 1]], [pts[0, 2]],
                c="white",
                s=20,
                edgecolors=BEHAVIOR_COLORS[beh],
                linewidths=0.6,
                zorder=5,
                depthshade=False,
            )
            ax.scatter(
                [pts[-1, 0]], [pts[-1, 1]], [pts[-1, 2]],
                c=BEHAVIOR_COLORS[beh],
                s=20,
                edgecolors=BEHAVIOR_COLORS[beh],
                linewidths=0.6,
                zorder=5,
                depthshade=False,
            )

        ax.set_title(
            BEHAVIOR_LABELS[beh],
            fontsize=8, fontweight="bold", pad=-2,
            color=BEHAVIOR_COLORS[beh],
        )
        _style_3d_ax(ax, var_explained, xlim, ylim, zlim, show_ylabel=(pi == 0))

    # Shared start/end legend on last panel
    legend_handles = [
        Line2D(
            [0], [0],
            marker="o", color="w",
            markeredgecolor="#555555", markeredgewidth=0.5,
            markersize=4, label="Start", linestyle="None",
        ),
        Line2D(
            [0], [0],
            marker="o", color="#555555",
            markeredgecolor="#555555", markeredgewidth=0.5,
            markersize=4, label="End", linestyle="None",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True, framealpha=0.92, edgecolor="none",
        borderpad=0.4, handletextpad=0.3, fancybox=True, fontsize=5.5,
    ).get_frame().set_linewidth(0)

    fig.suptitle(
        "PCA of RNN Hidden State",
        fontsize=9, fontweight="bold", y=0.98,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.88, wspace=0.05)
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
    _setup_nature_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_hidden_data()

    # Build flat array with behavior boundaries
    all_points: list[np.ndarray] = []
    beh_slices: dict[str, tuple[int, int]] = {}
    beh_shapes: dict[str, tuple[int, int]] = {}
    offset = 0

    for beh in BEHAVIORS:
        if beh not in data:
            continue
        arr = data[beh]  # [K, T, hidden_dim]
        K_beh, T_beh = arr.shape[0], arr.shape[1]
        beh_shapes[beh] = (K_beh, T_beh)
        flat = arr.reshape(-1, arr.shape[-1])
        all_points.append(flat)
        beh_slices[beh] = (offset, offset + flat.shape[0])
        offset += flat.shape[0]

    all_points_arr = np.concatenate(all_points, axis=0)
    print(
        f"Total points: {all_points_arr.shape[0]}, "
        f"dim: {all_points_arr.shape[1]}"
    )
    for beh in BEHAVIORS:
        if beh in beh_shapes:
            K_b, T_b = beh_shapes[beh]
            print(f"  {beh}: K={K_b}, T={T_b}")

    # Fit 3D PCA on ALL behaviors (shared space)
    print("\nFitting 3D PCA...")
    emb, var = fit_pca_3d(all_points_arr)

    fig, rect = plot_scatter_3d(emb, beh_slices, var)
    _save_figure(fig, rect, "hidden_scatter_pca3d")

    fig, rect = plot_trajectories_3d(emb, beh_slices, beh_shapes, var)
    _save_figure(fig, rect, "hidden_trajectories_pca3d")

    print("\nDone.")


if __name__ == "__main__":
    main()
