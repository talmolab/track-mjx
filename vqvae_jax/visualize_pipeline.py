"""Visualize the two-pathway pipeline for discrete-code → continuous-control.

Usage:
    cd vqvae_jax
    python visualize_pipeline.py                     # saves pipeline_figure.pdf + .png
    python visualize_pipeline.py --output fig.pdf    # custom output path
"""

import argparse

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
import numpy as np


# =============================================================================
# Color palette
# =============================================================================
C_BLUE = "#7BA3CC"
C_BLUE_DARK = "#5B83AC"
C_DECODER = "#C0A0D0"
C_DECODER_DARK = "#9070A0"
C_CODE = "#7BC47F"
C_CODE_DARK = "#4A9E4F"
C_PHYSICS = "#888888"
C_PRIOR = "#CC6677"
C_PRIOR_DARK = "#A04050"
C_ARROW = "#444444"
C_TEXT = "#222222"
C_PANEL_BG = "#F6F6FA"
C_FEEDBACK = "#666666"


# =============================================================================
# Drawing primitives
# =============================================================================

def _block(ax, x, y, w, h, color, label, fontsize=9, fontweight="bold",
           border_color=None, sublabel=None, sublabel_size=7,
           text_color="white", zorder=3):
    bc = border_color or color
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.04",
        facecolor=color, edgecolor=bc, linewidth=1.5, zorder=zorder,
    ))
    ty = y + h * 0.58 if sublabel else y + h / 2
    ax.text(x + w / 2, ty, label, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=text_color,
            zorder=zorder + 1)
    if sublabel:
        ax.text(x + w / 2, y + h * 0.25, sublabel, ha="center", va="center",
                fontsize=sublabel_size, fontstyle="italic", color=text_color,
                alpha=0.85, zorder=zorder + 1)


def _trapezoid(ax, x, y, w, h, color, label, direction="right",
               fontsize=9, fontweight="bold", border_color=None,
               sublabel=None, sublabel_size=7, text_color="white"):
    bc = border_color or color
    taper = h * 0.22
    if direction == "right":
        verts = [(x, y - taper), (x + w, y), (x + w, y + h),
                 (x, y + h + taper)]
    else:
        verts = [(x, y), (x + w, y - taper), (x + w, y + h + taper),
                 (x, y + h)]
    ax.add_patch(Polygon(verts, closed=True, facecolor=color,
                         edgecolor=bc, linewidth=1.5, zorder=3))
    ty = y + h * 0.58 if sublabel else y + h / 2
    ax.text(x + w / 2, ty, label, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=text_color, zorder=4)
    if sublabel:
        ax.text(x + w / 2, y + h * 0.25, sublabel, ha="center", va="center",
                fontsize=sublabel_size, fontstyle="italic", color=text_color,
                alpha=0.85, zorder=4)


def _arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.5, style="-|>", zorder=2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, color=color, lw=lw,
        zorder=zorder, mutation_scale=13,
    ))


def _code_strip(ax, x, y, w, h, n=10):
    gap = w * 0.04
    cell_w = (w - gap * (n + 1)) / n
    rng = np.random.RandomState(42)
    for i in range(n):
        cx = x + gap + i * (cell_w + gap)
        shade = rng.uniform(0.65, 1.0)
        c = tuple(np.array(matplotlib.colors.to_rgb(C_CODE)) * shade)
        ax.add_patch(FancyBboxPatch(
            (cx, y), cell_w, h, boxstyle="round,pad=0.008",
            facecolor=c, edgecolor=C_CODE_DARK, linewidth=0.6, zorder=4,
        ))


# =============================================================================
# Panel A: MoSeq
# =============================================================================

def draw_pathway_a(ax):
    #
    # MAIN ROW:  keypoints → [MoSeq] → c_t → [Decoder] → a_t → [Physics]
    #
    # All blocks share the same vertical band: bottom=1.6, height=0.5
    #
    by = 1.6   # block bottom y
    bh = 0.5   # block height
    cy = by + bh / 2   # center y of main row

    # 1. "keypoints" text
    ax.text(0.3, cy, r"$x_{t:T}^g$", ha="center", va="center",
            fontsize=13, color=C_TEXT)

    # 2. Arrow → MoSeq
    _arrow(ax, 0.55, cy, 0.85, cy)

    # 3. MoSeq block
    _block(ax, 0.9, by, 0.9, bh, C_BLUE, "Keypoint-\nMoSeq",
           fontsize=9.5, border_color=C_BLUE_DARK,
           sublabel=r"$c_t = \mathrm{MoSeq}(x_t^g)$", sublabel_size=7)

    # 4. Arrow → c_t
    _arrow(ax, 1.85, cy, 2.15, cy)

    # 5. c_t text
    ct_x = 2.3
    ax.text(ct_x, cy, r"$c_t$", ha="center", va="center",
            fontsize=14, color=C_CODE_DARK, fontweight="bold")

    # 6. Arrow → Decoder
    _arrow(ax, 2.45, cy, 2.85, cy)

    # 7. Decoder
    _trapezoid(ax, 2.9, by, 0.8, bh, C_DECODER, "Decoder",
               direction="left", border_color=C_DECODER_DARK,
               sublabel=r"$p_\theta(a_t \mid c_t, s_t^p)$", sublabel_size=7)

    # 8. Arrow → a_t
    _arrow(ax, 3.75, cy, 4.05, cy)

    # 9. a_t text
    ax.text(4.15, cy, r"$a_t$", ha="center", va="center",
            fontsize=13, color=C_TEXT)

    # 10. Arrow → Physics
    _arrow(ax, 4.3, cy, 4.55, cy)

    # 11. Physics block
    _block(ax, 4.6, by + 0.03, 0.6, bh - 0.06, C_PHYSICS,
           "Physics\nSim", fontsize=9, border_color="#666666",
           text_color="white", sublabel=r"$s_{t+1}^p$", sublabel_size=7.5)

    #
    # s_t^p: sits below decoder, with feedback from physics
    #
    sp_x = 3.3   # center x (under decoder)
    sp_y = 0.90  # center y

    ax.text(sp_x, sp_y, r"$s_t^p$", ha="center", va="center",
            fontsize=13, color=C_TEXT, fontweight="bold")

    # Arrow: s_t^p up into decoder
    _arrow(ax, sp_x, sp_y + 0.15, sp_x, by - 0.08,
           color=C_FEEDBACK, lw=1.2)

    # Feedback: physics bottom → down → left → up → s_t^p
    phys_bot_x = 4.9   # center of physics
    fb_y_low = 0.50
    ax.plot([phys_bot_x, phys_bot_x, sp_x, sp_x],
            [by + 0.03, fb_y_low, fb_y_low, sp_y - 0.18],
            color=C_FEEDBACK, lw=1.0, ls="--", zorder=2)
    ax.annotate("", xy=(sp_x, sp_y - 0.15),
                xytext=(sp_x, sp_y - 0.18),
                arrowprops=dict(arrowstyle="-|>", color=C_FEEDBACK, lw=1.0),
                zorder=2)

    #
    # CODE STRIP: below c_t, with downward arrow from c_t
    #
    strip_y = 0.25
    strip_h = 0.14

    # Arrow from c_t down to strip (shorter)
    _arrow(ax, ct_x, cy - 0.18, ct_x, strip_y + strip_h + 0.20,
           color=C_CODE_DARK, lw=1.5)

    # Label above strip
    ax.text(ct_x, strip_y + strip_h + 0.13,
            r"$[c_1, c_2, \ldots, c_T]$",
            ha="center", va="center", fontsize=9, color=C_TEXT)

    # Strip
    strip_w = 1.0
    _code_strip(ax, ct_x - strip_w / 2, strip_y, strip_w, strip_h, n=10)

    #
    # GENERATIVE PRIOR
    #
    prior_w = 1.3
    prior_h = 0.40
    prior_y = -0.70

    _block(ax, ct_x - prior_w / 2, prior_y, prior_w, prior_h,
           C_PRIOR, "Generative Prior", fontsize=10.5,
           border_color=C_PRIOR_DARK,
           sublabel="HMM / Transformer / ...", sublabel_size=8,
           text_color="white")

    # Train arrow (down, right of center)
    tx = ct_x + 0.18
    _arrow(ax, tx, strip_y - 0.10, tx, prior_y + prior_h + 0.10,
           color=C_PRIOR_DARK, lw=1.8)
    ax.text(tx + 0.14, (strip_y + prior_y + prior_h) / 2 - 0.05,
            "train", ha="left", va="center", fontsize=9,
            color=C_PRIOR_DARK, fontstyle="italic")

    # Generate arrow (up dashed, left of center)
    gx = ct_x - 0.18
    ax.annotate(
        "", xy=(gx, strip_y - 0.10),
        xytext=(gx, prior_y + prior_h + 0.10),
        arrowprops=dict(arrowstyle="-|>", color=C_PRIOR_DARK,
                        lw=1.5, ls="--"),
        zorder=2,
    )
    ax.text(gx - 0.14, (strip_y + prior_y + prior_h) / 2 - 0.05,
            "generate", ha="right", va="center", fontsize=9,
            color=C_PRIOR_DARK, fontstyle="italic")


# =============================================================================
# Panel B: VQ-VAE
# =============================================================================

def draw_pathway_b(ax):
    #
    # MAIN ROW:  s_{t:T}^g → [Encoder] → [VQ] → c_t → [Decoder] → a_t → [Physics]
    #
    by = 1.6
    bh = 0.5
    cy = by + bh / 2

    # 1. Input text
    ax.text(0.15, cy, r"$s_{t:T}^g$", ha="center", va="center",
            fontsize=13, color=C_TEXT)

    # 2. Arrow → Encoder
    _arrow(ax, 0.38, cy, 0.63, cy)

    # 3. Encoder
    _trapezoid(ax, 0.68, by, 0.72, bh, C_BLUE, "Encoder",
               direction="right", border_color=C_BLUE_DARK,
               sublabel=r"$q_\phi(z_t \mid s_t^g)$", sublabel_size=7)

    # 4. Arrow → VQ
    _arrow(ax, 1.45, cy, 1.70, cy)

    # 5. VQ block
    _block(ax, 1.75, by + 0.03, 0.42, bh - 0.06, C_BLUE, "VQ",
           fontsize=11, fontweight="bold", border_color=C_BLUE_DARK,
           sublabel=r"$z_t \!\to\! c_t$", sublabel_size=7.5)

    # 6. Arrow → c_t
    _arrow(ax, 2.22, cy, 2.50, cy)

    # 7. c_t text
    ct_x = 2.62
    ax.text(ct_x, cy, r"$c_t$", ha="center", va="center",
            fontsize=14, color=C_CODE_DARK, fontweight="bold")

    # 8. Arrow → Decoder
    _arrow(ax, 2.78, cy, 3.08, cy)

    # 9. Decoder
    _trapezoid(ax, 3.13, by, 0.8, bh, C_DECODER, "Decoder",
               direction="left", border_color=C_DECODER_DARK,
               sublabel=r"$p_\theta(a_t \mid c_t, s_t^p)$", sublabel_size=7)

    # 10. Arrow → a_t
    _arrow(ax, 3.98, cy, 4.25, cy)

    # 11. a_t text
    ax.text(4.35, cy, r"$a_t$", ha="center", va="center",
            fontsize=13, color=C_TEXT)

    # 12. Arrow → Physics
    _arrow(ax, 4.50, cy, 4.75, cy)

    # 13. Physics block
    _block(ax, 4.80, by + 0.03, 0.6, bh - 0.06, C_PHYSICS,
           "Physics\nSim", fontsize=9, border_color="#666666",
           text_color="white", sublabel=r"$s_{t+1}^p$", sublabel_size=7.5)

    #
    # s_t^p
    #
    sp_x = 3.53
    sp_y = 0.90

    ax.text(sp_x, sp_y, r"$s_t^p$", ha="center", va="center",
            fontsize=13, color=C_TEXT, fontweight="bold")

    _arrow(ax, sp_x, sp_y + 0.15, sp_x, by - 0.08,
           color=C_FEEDBACK, lw=1.2)

    phys_bot_x = 5.1
    fb_y_low = 0.50
    ax.plot([phys_bot_x, phys_bot_x, sp_x, sp_x],
            [by + 0.03, fb_y_low, fb_y_low, sp_y - 0.18],
            color=C_FEEDBACK, lw=1.0, ls="--", zorder=2)
    ax.annotate("", xy=(sp_x, sp_y - 0.15),
                xytext=(sp_x, sp_y - 0.18),
                arrowprops=dict(arrowstyle="-|>", color=C_FEEDBACK, lw=1.0),
                zorder=2)

    #
    # CODE STRIP
    #
    strip_y = 0.25
    strip_h = 0.14

    _arrow(ax, ct_x, cy - 0.18, ct_x, strip_y + strip_h + 0.20,
           color=C_CODE_DARK, lw=1.5)

    ax.text(ct_x, strip_y + strip_h + 0.13,
            r"$[c_1, c_2, \ldots, c_T]$",
            ha="center", va="center", fontsize=9, color=C_TEXT)

    strip_w = 1.1
    _code_strip(ax, ct_x - strip_w / 2, strip_y, strip_w, strip_h, n=12)

    #
    # GENERATIVE PRIOR
    #
    prior_w = 1.3
    prior_h = 0.40
    prior_y = -0.70

    _block(ax, ct_x - prior_w / 2, prior_y, prior_w, prior_h,
           C_PRIOR, "Generative Prior", fontsize=10.5,
           border_color=C_PRIOR_DARK,
           sublabel="HMM / Transformer / ...", sublabel_size=8,
           text_color="white")

    tx = ct_x + 0.18
    _arrow(ax, tx, strip_y - 0.10, tx, prior_y + prior_h + 0.10,
           color=C_PRIOR_DARK, lw=1.8)
    ax.text(tx + 0.14, (strip_y + prior_y + prior_h) / 2 - 0.05,
            "train", ha="left", va="center", fontsize=9,
            color=C_PRIOR_DARK, fontstyle="italic")

    gx = ct_x - 0.18
    ax.annotate(
        "", xy=(gx, strip_y - 0.10),
        xytext=(gx, prior_y + prior_h + 0.10),
        arrowprops=dict(arrowstyle="-|>", color=C_PRIOR_DARK,
                        lw=1.5, ls="--"),
        zorder=2,
    )
    ax.text(gx - 0.14, (strip_y + prior_y + prior_h) / 2 - 0.05,
            "generate", ha="right", va="center", fontsize=9,
            color=C_PRIOR_DARK, fontstyle="italic")


# =============================================================================
# Main figure
# =============================================================================

def make_figure(output_path: str = "pipeline_figure"):
    fig, axes = plt.subplots(2, 1, figsize=(11, 11),
                             gridspec_kw={"hspace": 0.12})

    for ax_obj in axes:
        ax_obj.set_xlim(-0.2, 5.7)
        ax_obj.set_ylim(-1.05, 2.50)
        ax_obj.set_aspect("equal")
        ax_obj.axis("off")

    # Panel A
    ax_a = axes[0]
    ax_a.add_patch(FancyBboxPatch(
        (-0.10, -0.85), 5.70, 3.10,
        boxstyle="round,pad=0.05", facecolor=C_PANEL_BG,
        edgecolor="#CCCCCC", linewidth=1.0, zorder=0))
    ax_a.text(0.10, 2.35, "(A)  MoSeq Tokenizer", fontsize=12,
              fontweight="bold", color=C_TEXT, zorder=5)
    draw_pathway_a(ax_a)

    # Panel B
    ax_b = axes[1]
    ax_b.add_patch(FancyBboxPatch(
        (-0.10, -0.85), 5.70, 3.10,
        boxstyle="round,pad=0.05", facecolor=C_PANEL_BG,
        edgecolor="#CCCCCC", linewidth=1.0, zorder=0))
    ax_b.text(0.10, 2.35, "(B)  VQ-VAE Tokenizer", fontsize=12,
              fontweight="bold", color=C_TEXT, zorder=5)
    draw_pathway_b(ax_b)

    base = str(output_path).replace(".pdf", "").replace(".png", "")
    for ext in [".pdf", ".png"]:
        p = base + ext
        fig.savefig(p, dpi=400, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"Saved: {p}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize the two-pathway pipeline")
    parser.add_argument("--output", default="pipeline_figure",
                        help="Output path (saves .pdf and .png)")
    args = parser.parse_args()
    make_figure(args.output)
