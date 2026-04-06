"""Nature-style matplotlib configuration and WandB plotting helpers."""

import io

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Colorblind-safe palette (Wong, Nature Methods 2011)
# ---------------------------------------------------------------------------
NATURE_COLORS = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "vermillion": "#E69F00",
    "sky_blue": "#56B4E9",
    "purple": "#CC79A7",
    "black": "#000000",
    "gray": "#999999",
}

BEHAVIOR_COLORS = {
    "groom": "#0072B2",
    "walk": "#D55E00",
    "rear": "#009E73",
}

CONDITION_COLORS = {
    "correct": "#009E73",
    "shuffled_step": "#D55E00",
    "shuffled_trajectory": "#0072B2",
}

MODE_COLORS = {
    "code2act": "#0072B2",
    "mimic_mjx": "#D55E00",
}

MODE_LABELS = {
    "code2act": "Code2Act",
    "mimic_mjx": "Mimic-MJX (oracle)",
}


def set_nature_style() -> None:
    """Apply Nature-style matplotlib rcParams."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "axes.titleweight": "bold",
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
        }
    )


def fig_to_image(fig: plt.Figure):
    """Convert matplotlib figure to wandb.Image (import wandb lazily)."""
    import wandb

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    buf.seek(0)
    from PIL import Image

    img = Image.open(buf)
    return wandb.Image(img)


def get_trajectory_colors(k: int) -> list[list[float]]:
    """Return *k* qualitative RGBA colours suitable for ghost-body overlay."""
    cmap = plt.cm.get_cmap("tab10")
    return [list(cmap(i % 10)) for i in range(k)]


def get_code_colormap(num_codes: int) -> np.ndarray:
    """Return ``[num_codes, 3]`` uint8 RGB array for code timeline bars."""
    if num_codes <= 10:
        cmap = plt.cm.get_cmap("tab10")
    elif num_codes <= 20:
        cmap = plt.cm.get_cmap("tab20")
    else:
        cmap = plt.cm.get_cmap("hsv")
    colors = (cmap(np.linspace(0, 1, num_codes))[:, :3] * 255).astype(np.uint8)
    return colors
