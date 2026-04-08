"""Nature-style combined figure: (A) training curves + (B) reward decomposition.

Side-by-side panel with large legends and panel labels.

Usage:
    cd moseq_jax/figures
    python plot_training_and_decomposition.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

# ═══════════════════════════════════════════════════════════════════════════════
# Panel (A) — Training curves
# ═══════════════════════════════════════════════════════════════════════════════

TRAIN_CSV = DATA_DIR / "train_data.csv"

TRAIN_COLORS = {
    "RNN Distillation": "#0072B2",       # blue (Wong)
    "RNN Concat Discrete": "#009E73",    # green (Wong)
    "RNN Concat Full": "#D55E00",        # orange (Wong)
}

TRAIN_SERIES = {
    "RNN Distillation": {
        "mean": "C2A (Distill 1.5KL) - moseq/episode_reward_mean",
        "min": "C2A (Distill 1.5KL) - moseq/episode_reward_mean__MIN",
        "max": "C2A (Distill 1.5KL) - moseq/episode_reward_mean__MAX",
    },
    "RNN Concat Discrete": {
        "mean": "C2A (readout continuous) - decoder_only/episode_reward_mean",
        "min": "C2A (readout continuous) - decoder_only/episode_reward_mean__MIN",
        "max": "C2A (readout continuous) - decoder_only/episode_reward_mean__MAX",
    },
    "RNN Concat Full": {
        "mean": "C2A (readout continuous) - moseq/episode_reward_mean",
        "min": "C2A (readout continuous) - moseq/episode_reward_mean__MIN",
        "max": "C2A (readout continuous) - moseq/episode_reward_mean__MAX",
    },
}


def _smooth(y: np.ndarray, window: int = 11, **_kwargs) -> np.ndarray:
    if len(y) < window:
        window = max(3, len(y))
    if window > len(y):
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def _load_train_data() -> dict[str, pd.DataFrame]:
    raw = pd.read_csv(TRAIN_CSV)
    result = {}
    for label, cols in TRAIN_SERIES.items():
        mask = raw[cols["mean"]].notna() & (raw[cols["mean"]] != "")
        sub = raw.loc[mask, ["num_steps_thousands", cols["mean"], cols["min"], cols["max"]]].copy()
        sub.columns = ["x", "mean", "min", "max"]
        sub = sub.apply(pd.to_numeric, errors="coerce").dropna().sort_values("x")
        sub["x"] = sub["x"] / 1_000
        result[label] = sub.reset_index(drop=True)
    return result


def _plot_panel_a(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    for label, df in data.items():
        x = df["x"].values
        y_mean = df["mean"].values
        y_min = df["min"].values
        y_max = df["max"].values
        color = TRAIN_COLORS[label]

        y_smooth = _smooth(y_mean)

        has_band = not np.allclose(y_min, y_max)
        if has_band:
            ax.fill_between(x, _smooth(y_min), _smooth(y_max), color=color, alpha=0.12, linewidth=0)

        ax.scatter(x, y_mean, color=color, alpha=0.20, s=4, linewidths=0, zorder=1)
        ax.plot(x, y_smooth, color=color, label=label, linewidth=1.8, zorder=2)

    ax.set_xlabel("Training Steps (millions)")
    ax.set_ylabel("Episode Reward")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=600)

    for ytick in ax.get_yticks():
        if ytick > 0:
            ax.axhline(ytick, color="#e0e0e0", linewidth=0.3, zorder=0)

    leg = ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
        borderpad=0.4,
        handlelength=1.5,
        handletextpad=0.4,
        fancybox=True,
        fontsize=7,
    )
    leg.get_frame().set_linewidth(0)
    leg.get_frame().set_boxstyle("round,pad=0.3,rounding_size=0.2")


# ═══════════════════════════════════════════════════════════════════════════════
# Panels (B) & (C) — Reward decomposition (inference + generalization)
# ═══════════════════════════════════════════════════════════════════════════════

DECOMP_DATASETS = {
    "inference": {
        "Code2Act": DATA_DIR / "test_code2act.npz",
        "Mimic-MJX": DATA_DIR / "test_mimic_mjx.npz",
    },
    "generalization": {
        "Code2Act": DATA_DIR / "generalization_code2act.npz",
        "Mimic-MJX": DATA_DIR / "generalization_mimic_mjx.npz",
    },
}

DECOMP_COLORS = {
    "Code2Act": "#0072B2",   # blue (Wong)
    "Mimic-MJX": "#D55E00",  # orange (Wong)
}

COMP_LINESTYLE = {"coarse": "-", "fine": "--"}
COMP_LABELS = {"coarse": "Coarse (root)", "fine": "Fine (joints + end-eff)"}


def _load_decomp_data(dataset: str) -> dict[str, dict[str, np.ndarray]]:
    data = {}
    for mode, path in DECOMP_DATASETS[dataset].items():
        d = np.load(path, allow_pickle=True)
        data[mode] = {"coarse": d["decomp_coarse"], "fine": d["decomp_fine"]}
    return data


def _plot_decomp_panel(
    ax: plt.Axes,
    data: dict[str, dict[str, np.ndarray]],
    title: str,
) -> None:
    max_t = min(d[comp].shape[1] for d in data.values() for comp in ("coarse", "fine"))

    norm_factors = {}
    for comp in ("coarse", "fine"):
        global_max = max(d[comp][:, :max_t].mean(axis=0).max() for d in data.values())
        norm_factors[comp] = max(global_max, 1e-8)

    for mode, curves in data.items():
        color = DECOMP_COLORS[mode]
        for comp in ("coarse", "fine"):
            arr = curves[comp][:, :max_t]
            mean = arr.mean(axis=0)
            sem = arr.std(axis=0) / np.sqrt(arr.shape[0])
            nf = norm_factors[comp]
            mean_n = mean / nf
            sem_n = sem / nf
            t = np.arange(len(mean_n))

            label = f"{mode} — {COMP_LABELS[comp]}"
            ax.plot(t, mean_n, color=color, linestyle=COMP_LINESTYLE[comp], label=label, zorder=2)
            ax.fill_between(t, mean_n - sem_n, mean_n + sem_n, color=color, alpha=0.12, linewidth=0, zorder=1)

    ax.set_xlabel("Episode Timestep")
    ax.set_ylabel("Normalized Reward")
    ax.set_title(title, fontsize=7)
    ax.set_ylim(bottom=0, top=1.08)
    ax.set_xlim(left=0)

    for y in (0.25, 0.5, 0.75, 1.0):
        ax.axhline(y, color="#e0e0e0", linewidth=0.3, zorder=0)

    leg = ax.legend(
        loc="lower left",
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
        borderpad=0.4,
        handlelength=2.0,
        handletextpad=0.4,
        fancybox=True,
        fontsize=5.5,
    )
    leg.get_frame().set_linewidth(0)
    leg.get_frame().set_boxstyle("round,pad=0.3,rounding_size=0.2")


# ═══════════════════════════════════════════════════════════════════════════════
# Combined figure
# ═══════════════════════════════════════════════════════════════════════════════


def _setup_nature_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 7,
        "lines.linewidth": 1.5,
        "lines.markersize": 2,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": False,
    })


def main() -> None:
    _setup_nature_style()

    train_data = _load_train_data()
    inf_data = _load_decomp_data("inference")
    gen_data = _load_decomp_data("generalization")

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(10.5, 2.8))

    _plot_panel_a(ax_a, train_data)
    _plot_decomp_panel(ax_b, inf_data, "500-Frame Test Set")
    _plot_decomp_panel(ax_c, gen_data, "1000-Frame Test Set")

    # Panel labels
    for ax, label in zip([ax_a, ax_b, ax_c], ["(A)", "(B)", "(C)"]):
        ax.text(
            -0.12, 1.08, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left",
        )

    fig.tight_layout(w_pad=2.0)

    # Rounded figure border
    fig.patch.set_facecolor("white")
    for ax in (ax_a, ax_b, ax_c):
        for spine in ax.spines.values():
            spine.set_visible(False)

    rect = mpatches.FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        transform=fig.transFigure,
        facecolor="white",
        edgecolor="#cccccc",
        linewidth=0.8,
        zorder=-1,
    )
    fig.patches.append(rect)

    for ax in (ax_a, ax_b, ax_c):
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)

    out_pdf = OUTPUT_DIR / "training_and_decomposition.pdf"
    out_png = OUTPUT_DIR / "training_and_decomposition.png"
    out_svg = OUTPUT_DIR / "training_and_decomposition.svg"
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    rect.set_facecolor("none")
    fig.savefig(out_svg, transparent=True)
    rect.set_facecolor("white")
    plt.close(fig)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_svg}")


if __name__ == "__main__":
    main()
