"""Render all KPMS syllable trajectories in a 5-column grid figure.

Each cell shows XY (top-down) and XZ (side) projections side-by-side
for one syllable.  5 columns, rows grow with number of syllables.
Transparent background.  Outputs SVG, PDF, and PNG.

Usage:
    cd moseq_jax
    python figures/render_syllable_trajectory_grid.py
    python figures/render_syllable_trajectory_grid.py --cols 4
"""

import argparse
import math
import os
import sys
from pathlib import Path

os.environ["JAX_ENABLE_X64"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import matplotlib

matplotlib.use("Agg")
import matplotlib.backends.backend_agg
import matplotlib.pyplot as plt
import numpy as np

# kpms import workaround (bokeh/numpy incompatibility)
sys.modules["keypoint_moseq.analysis"] = type(sys)("mock_analysis")
from keypoint_moseq import io as kpms_io
from keypoint_moseq import viz as kpms_viz

SCRIPT_DIR = Path(__file__).parent
MOSEQ_DIR = SCRIPT_DIR.parent
REPO_ROOT = MOSEQ_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"

for _p in (str(MOSEQ_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.shared.keypoint_fk import setup_stac_model, qpos_to_keypoints_fk
from experiments.shared.clip_selection import load_balanced_splits
from kpms.keypoint_loader import prepare_keypoints_for_kpms

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_H5 = str(REPO_ROOT / "data/rodent/rodent_reference_clips.h5")
DEFAULT_SPLITS = str(REPO_ROOT / "data/rodent/rodent_balanced_splits.json")
DEFAULT_KPMS_DIR = str(
    MOSEQ_DIR / "outputs/kpms_sweep/s50_k1e+04_l10_arhmm/seed1"
)
DEFAULT_MODEL = "keypoint_arhmm_states50_seed1"

RODENT_SKELETON = [
    ("Snout", "SpineF"), ("SpineF", "SpineM"), ("SpineM", "SpineL"),
    ("SpineL", "TailBase"),
    ("Snout", "EarL"), ("Snout", "EarR"),
    ("SpineF", "ShoulderL"), ("ShoulderL", "ElbowL"),
    ("ElbowL", "WristL"), ("WristL", "HandL"),
    ("SpineF", "ShoulderR"), ("ShoulderR", "ElbowR"),
    ("ElbowR", "WristR"), ("WristR", "HandR"),
    ("SpineL", "HipL"), ("HipL", "KneeL"),
    ("KneeL", "AnkleL"), ("AnkleL", "FootL"),
    ("SpineL", "HipR"), ("HipR", "KneeR"),
    ("KneeR", "AnkleR"), ("AnkleR", "FootR"),
]

_NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
}


# ---------------------------------------------------------------------------
# Data loading (mirrors run_syllable_viz.py Steps 1-3)
# ---------------------------------------------------------------------------


def load_data(
    h5_path: str,
    splits_path: str,
    kpms_dir: str,
    model_name: str,
) -> tuple[dict, dict, list[str], list[str]]:
    """Load keypoints, KPMS results, and config.

    Returns:
        ``(coordinates, results, bodyparts, use_bodyparts)``
    """
    import h5py

    print("Loading qpos ...")
    with h5py.File(h5_path, "r") as f:
        qpos_all = f["qpos"][:]
    n_clips = qpos_all.shape[0] // 250

    print("Running FK ...")
    mj_model, mj_data, site_ids, kp_names = setup_stac_model(h5_path)
    kps_flat = qpos_to_keypoints_fk(
        qpos_all, mj_model, mj_data, site_ids, batch_size=1000,
    )
    kps_all = kps_flat.reshape(n_clips, 250, -1, 3)

    splits = load_balanced_splits(splits_path)
    all_idx = sorted(
        set(splits["balanced"]["train_indices"])
        | set(splits["balanced"]["test_indices"])
    )
    kps_balanced = kps_all[all_idx] * 1000.0  # meters → mm

    coordinates, _ = prepare_keypoints_for_kpms(kps_balanced)

    print("Loading KPMS results ...")
    results = kpms_io.load_results(
        project_dir=kpms_dir, model_name=model_name,
    )
    for k in results:
        if "centroid" in results[k]:
            results[k]["centroid"] = results[k]["centroid"] * 1000.0

    cfg = kpms_io.load_config(kpms_dir)
    bodyparts = cfg["bodyparts"]
    use_bodyparts = cfg.get("use_bodyparts", bodyparts)

    print(f"  {len(coordinates)} recordings, {len(bodyparts)} bodyparts")
    return coordinates, results, bodyparts, use_bodyparts


# ---------------------------------------------------------------------------
# Grid rendering
# ---------------------------------------------------------------------------


def _draw_trajectory(
    ax: plt.Axes,
    X: np.ndarray,
    edges: list[tuple[int, int]],
    colors: list,
    num_timesteps: int = 10,
) -> None:
    """Draw a single trajectory with temporal fading on *ax*."""
    alphas = np.linspace(0.15, 1.0, num_timesteps) ** 2
    for t in range(num_timesteps):
        alpha = float(alphas[t])
        for e0, e1 in edges:
            ax.plot(
                [X[t, e0, 0], X[t, e1, 0]],
                [X[t, e0, 1], X[t, e1, 1]],
                color="k", alpha=alpha * 0.6,
                linewidth=0.8, zorder=t * 2,
            )
        for ki in range(X.shape[1]):
            ax.plot(
                X[t, ki, 0], X[t, ki, 1],
                "o", color=colors[ki],
                markersize=2.5, alpha=alpha,
                markeredgecolor="k", markeredgewidth=0.2,
                zorder=t * 2 + 1,
            )
    ax.set_aspect("equal")
    ax.axis("off")


def render_syllable_grid(
    coordinates: dict,
    results: dict,
    bodyparts: list[str],
    use_bodyparts: list[str],
    n_cols: int = 2,
    top_n: int | None = 6,
    cell_width: float = 2.0,
    fps: int = 30,
    pre: float = 0.167,
    post: float = 0.5,
    min_frequency: float = 0.005,
    min_duration: int = 3,
    keypoint_colormap: str = "autumn",
) -> plt.Figure:
    """Render syllable trajectories in a grid figure.

    Each cell shows XY (top-down) on the left and XZ (side view) on
    the right, with a centred syllable title above.

    Args:
        n_cols: Number of syllable columns in the grid.
        top_n: Keep only the *top_n* most popular syllables.
            ``None`` keeps all that pass the frequency filter.
        cell_width: Width of each cell pair in inches.

    Returns:
        Matplotlib figure with transparent background.
    """
    from keypoint_moseq.util import get_edges

    edges = get_edges(use_bodyparts, RODENT_SKELETON)

    pre_frames = round(pre * fps)
    post_frames = round(post * fps)

    typical = kpms_viz.get_typical_trajectories(
        coordinates, results, pre_frames, post_frames,
        min_frequency, min_duration,
        bodyparts, use_bodyparts,
        density_sample=True,
        sampling_options={"n_neighbors": 50},
    )

    # Rank syllables by frequency and keep top_n
    all_sylls = np.concatenate(
        [results[k]["syllable"] for k in sorted(results.keys())]
    )
    unique, counts = np.unique(all_sylls, return_counts=True)
    freq_order = unique[np.argsort(-counts)]
    available = sorted(typical.keys())
    ranked = [s for s in freq_order if s in available]
    if top_n is not None:
        ranked = ranked[:top_n]

    syllable_ixs = ranked
    n_syll = len(syllable_ixs)
    n_rows = math.ceil(n_syll / n_cols)
    print(f"Rendering {n_syll} syllables in {n_rows}x{n_cols} grid (xy + xz)")

    Xs_3d = np.stack([typical[s] for s in syllable_ixs])  # [S, T, K, 3]

    # Colormap
    n_kp = Xs_3d.shape[2]
    cmap = plt.colormaps[keypoint_colormap]
    colors = [cmap(i / max(n_kp - 1, 1)) for i in range(n_kp)]

    # Interpolate to 10 timesteps
    num_ts = 10
    t_new = np.linspace(0, Xs_3d.shape[1] - 1, num_ts)
    t_old = np.arange(Xs_3d.shape[1])

    Xs_interp = np.zeros((n_syll, num_ts, n_kp, 3))
    for si in range(n_syll):
        for ki in range(n_kp):
            for di in range(3):
                Xs_interp[si, :, ki, di] = np.interp(
                    t_new, t_old, Xs_3d[si, :, ki, di],
                )

    # Each syllable gets 2 sub-columns (xy, xz), so total columns = 2*n_cols
    sub_cols = 2 * n_cols
    half_w = cell_width / 2.0
    cell_height = half_w  # square sub-cells

    with plt.rc_context(_NATURE_RC):
        fig_w = cell_width * n_cols
        fig_h = cell_height * n_rows
        fig, axes = plt.subplots(
            n_rows, sub_cols, figsize=(fig_w, fig_h),
            squeeze=False,
        )
        fig.patch.set_alpha(0.0)

        for idx, syll_ix in enumerate(syllable_ixs):
            row = idx // n_cols
            col = idx % n_cols
            ax_xy = axes[row][col * 2]
            ax_xz = axes[row][col * 2 + 1]

            X = Xs_interp[idx]  # [T, K, 3]
            X_xy = X[..., [0, 1]]
            X_xz = X[..., [0, 2]]

            _draw_trajectory(ax_xy, X_xy, edges, colors, num_ts)
            _draw_trajectory(ax_xz, X_xz, edges, colors, num_ts)

            # Centred title spanning both sub-axes:
            # place at x=1.0 of the xy axis (= the boundary between xy/xz)
            ax_xy.set_title(
                f"Syllable {syll_ix}", fontsize=7, fontweight="bold",
                pad=3, loc="center",
                x=1.0,  # right edge of xy = midpoint of the pair
            )

            # Small plane labels
            ax_xy.text(
                0.02, 0.02, "xy", transform=ax_xy.transAxes,
                fontsize=4, color="grey", va="bottom",
            )
            ax_xz.text(
                0.02, 0.02, "xz", transform=ax_xz.transAxes,
                fontsize=4, color="grey", va="bottom",
            )

        # Hide unused cells
        for idx in range(n_syll, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row][col * 2].set_visible(False)
            axes[row][col * 2 + 1].set_visible(False)

        fig.subplots_adjust(
            left=0.01, right=0.99, top=0.94, bottom=0.01,
            wspace=0.02, hspace=0.18,
        )

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Render KPMS syllable trajectory grid (xy + xz)",
    )
    parser.add_argument("--h5", default=DEFAULT_H5)
    parser.add_argument("--splits", default=DEFAULT_SPLITS)
    parser.add_argument("--kpms-dir", default=DEFAULT_KPMS_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument(
        "--top-n", type=int, default=6,
        help="Show only the top N most popular syllables (0 = all)",
    )
    parser.add_argument("--cell-width", type=float, default=2.0)
    parser.add_argument(
        "--output", default=str(OUTPUT_DIR / "syllable_trajectory_grid"),
        help="Output path prefix (without extension)",
    )
    args = parser.parse_args()

    coordinates, results, bodyparts, use_bodyparts = load_data(
        args.h5, args.splits, args.kpms_dir, args.model_name,
    )

    fig = render_syllable_grid(
        coordinates, results, bodyparts, use_bodyparts,
        n_cols=args.cols,
        top_n=args.top_n or None,
        cell_width=args.cell_width,
    )

    out_prefix = args.output
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)

    for ext in ("svg", "pdf", "png"):
        path = f"{out_prefix}.{ext}"
        fig.savefig(
            path, dpi=300, bbox_inches="tight",
            transparent=True, pad_inches=0.05,
        )
        print(f"Saved: {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
