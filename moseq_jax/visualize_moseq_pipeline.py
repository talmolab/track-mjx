"""Visualize the KPMS decoder pipeline with RNN and continuous encoder.

Layout (matching organizations.png):
  Top row:  s^g_{t:T} -> [Encoder] -> [mu,sigma] ~eps-> z_0  (blue)
                ^                                        |
  IK over       |                                        | (into readout)
  T frames      |                                        v
  Main row: x^g_t -> [KPMS] -> c_t -> (RNN) -> [Readout] -> a_t -> [Physics]
                                         ^                            |
                                       s^p_t <--- feedback -----------+
                         |
                    [c_1,...,c_T]
                    [Gen. Prior]

Usage:
    cd moseq_jax
    python visualize_moseq_pipeline.py
    python visualize_moseq_pipeline.py --output outputs/moseq_pipeline
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import (
    Circle as MplCircle,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
)
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
C_FEEDBACK = "#999999"
C_KL = "#CC6677"
C_RNN_FWD = "#9070A0"
C_RNN_REC = "#5BA8D0"
C_STAC = "#777777"
C_Z0 = "#5B83AC"       # blue for z_0
C_Z0_DARK = "#3A6390"


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
        verts = [(x, y - taper), (x + w, y), (x + w, y + h), (x, y + h + taper)]
    else:
        verts = [(x, y), (x + w, y - taper), (x + w, y + h + taper), (x, y + h)]
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
# RNN circle diagram (organic style, matching rnn.png)
# =============================================================================

def _rnn_diagram(ax, cx, cy, radius):
    """Draw a stylized RNN as a double-ring circle with organic nodes."""
    # Outer circle (thick black border like rnn.png)
    ax.add_patch(MplCircle(
        (cx, cy), radius, facecolor=C_DECODER, alpha=0.08,
        edgecolor="#222222", linewidth=3.0, zorder=3,
    ))
    # Inner ring
    ax.add_patch(MplCircle(
        (cx, cy), radius * 0.84, facecolor="none",
        edgecolor=C_DECODER_DARK, linewidth=0.7, zorder=3, alpha=0.30,
    ))

    r = radius * 0.60
    nr = radius * 0.072

    # Organic node layout (scattered like rnn.png)
    nodes = [
        (cx - r * 0.50, cy + r * 0.65),
        (cx + r * 0.15, cy + r * 0.80),
        (cx + r * 0.65, cy + r * 0.45),
        (cx - r * 0.75, cy + r * 0.00),
        (cx - r * 0.15, cy + r * 0.15),
        (cx + r * 0.35, cy + r * 0.05),
        (cx + r * 0.78, cy + r * 0.15),
        (cx - r * 0.45, cy - r * 0.55),
        (cx + r * 0.15, cy - r * 0.55),
        (cx + r * 0.60, cy - r * 0.45),
    ]

    def _thin(p1, p2, color, lw=0.80, alpha=0.65):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        d = np.hypot(dx, dy)
        if d < 1e-6:
            return
        s = nr * 1.5 / d
        ax.add_patch(FancyArrowPatch(
            (p1[0] + dx * s, p1[1] + dy * s),
            (p2[0] - dx * s, p2[1] - dy * s),
            arrowstyle="-|>", color=color, lw=lw,
            zorder=4, mutation_scale=7, alpha=alpha,
        ))

    # Forward arrows (magenta)
    for i, j in [(0, 1), (0, 4), (1, 2), (1, 5), (2, 6),
                 (3, 4), (3, 7), (4, 5), (4, 8), (5, 6), (5, 9),
                 (7, 8), (8, 9), (0, 3), (2, 5), (6, 9)]:
        _thin(nodes[i], nodes[j], C_RNN_FWD, lw=0.80)

    # Recurrent arrows (light blue, curved)
    for i, j, rad in [(6, 0, -0.45), (9, 3, 0.50), (5, 0, -0.35),
                       (6, 4, -0.30), (9, 7, 0.40), (8, 3, 0.35)]:
        ax.annotate(
            "", xy=(nodes[j][0] + nr * 0.5, nodes[j][1]),
            xytext=(nodes[i][0] - nr * 0.5, nodes[i][1]),
            arrowprops=dict(arrowstyle="-|>", color=C_RNN_REC, lw=1.0,
                            connectionstyle=f"arc3,rad={rad}", alpha=0.55),
            zorder=4,
        )

    # Nodes
    for nx, ny in nodes:
        ax.add_patch(MplCircle(
            (nx, ny), nr, facecolor="white",
            edgecolor="#444444", linewidth=0.9, zorder=5,
        ))

    # Blue accent dots
    for dx, dy in [(cx - r * 0.30, cy + r * 0.35),
                   (cx + r * 0.05, cy - r * 0.18),
                   (cx + r * 0.48, cy + r * 0.28)]:
        ax.add_patch(MplCircle(
            (dx, dy), nr * 0.45, facecolor=C_BLUE,
            edgecolor="none", zorder=5, alpha=0.70,
        ))


# =============================================================================
# Encoder bracket + squiggly arrow
# =============================================================================

def _bracket_mu_sigma(ax, cx, cy, w=0.40, h=0.38):
    left, right = cx - w / 2, cx + w / 2
    top, bot = cy + h / 2, cy - h / 2
    hook = 0.04
    ax.plot([left + hook, left, left, left + hook], [top, top, bot, bot],
            color=C_TEXT, lw=1.6, zorder=4, solid_capstyle="round")
    ax.plot([right - hook, right, right, right - hook], [top, top, bot, bot],
            color=C_TEXT, lw=1.6, zorder=4, solid_capstyle="round")
    ax.text(cx, cy + h * 0.25, r"$\mu_t$", ha="center", va="center",
            fontsize=10, color=C_TEXT, zorder=5)
    ax.text(cx, cy - h * 0.25, r"$\sigma_t$", ha="center", va="center",
            fontsize=10, color=C_TEXT, zorder=5)


def _squiggly_arrow(ax, x1, y, x2):
    xs = np.linspace(x1 + 0.02, x2 - 0.06, 80)
    amp = 0.025
    n_waves = max(2, int((x2 - x1) / 0.12))
    ys = y + amp * np.sin(2 * np.pi * n_waves * (xs - xs[0]) / (xs[-1] - xs[0]))
    ax.plot(xs, ys, color=C_TEXT, lw=1.0, zorder=4)
    ax.annotate("", xy=(x2, y), xytext=(xs[-1], ys[-1]),
                arrowprops=dict(arrowstyle="-|>", color=C_TEXT, lw=1.0), zorder=4)
    ax.text((x1 + x2) / 2, y + 0.07, r"$\epsilon$", ha="center", va="bottom",
            fontsize=10, color=C_TEXT, zorder=5)


def _bracket_labeled(ax, cx, cy, top_text, bot_text, w=0.40, h=0.38):
    """Draw a bracket pair with custom top/bottom labels."""
    left, right = cx - w / 2, cx + w / 2
    top, bot = cy + h / 2, cy - h / 2
    hook = 0.04
    ax.plot([left + hook, left, left, left + hook], [top, top, bot, bot],
            color=C_TEXT, lw=1.6, zorder=4, solid_capstyle="round")
    ax.plot([right - hook, right, right, right - hook], [top, top, bot, bot],
            color=C_TEXT, lw=1.6, zorder=4, solid_capstyle="round")
    ax.text(cx, cy + h * 0.25, top_text, ha="center", va="center",
            fontsize=10, color=C_TEXT, zorder=5)
    ax.text(cx, cy - h * 0.25, bot_text, ha="center", va="center",
            fontsize=10, color=C_TEXT, zorder=5)


# =============================================================================
# Main pipeline drawing
# =============================================================================

def draw_moseq_pipeline(ax):

    # =====================================================================
    # MAIN ROW: x^g_t -> [KPMS] -> c_t -> (RNN) -> [Readout] -> a_t -> [Phy]
    # =====================================================================
    main_cy = 1.20

    # x^g_t
    xg_x = 0.40
    ax.text(xg_x, main_cy, r"$x_t^g$", ha="center", va="center",
            fontsize=13, color=C_TEXT)

    # -> KPMS
    _arrow(ax, xg_x + 0.18, main_cy, xg_x + 0.48, main_cy)
    kpms_x = xg_x + 0.53
    kpms_w = 0.95
    kpms_h = 0.50
    kpms_by = main_cy - kpms_h / 2
    _block(ax, kpms_x, kpms_by, kpms_w, kpms_h, C_BLUE, "Keypoint-\nMoSeq",
           fontsize=9, border_color=C_BLUE_DARK,
           sublabel=r"$c_t = \mathrm{MoSeq}(x_t^g)$", sublabel_size=6.5)

    # -> c_t
    ct_x = kpms_x + kpms_w + 0.30
    _arrow(ax, kpms_x + kpms_w + 0.05, main_cy, ct_x - 0.12, main_cy)
    ax.text(ct_x, main_cy, r"$c_t$", ha="center", va="center",
            fontsize=14, color=C_CODE_DARK, fontweight="bold")

    # -> RNN
    rnn_cx = ct_x + 0.95
    rnn_cy = main_cy
    rnn_r = 0.48
    _arrow(ax, ct_x + 0.14, main_cy, rnn_cx - rnn_r - 0.06, main_cy)

    _rnn_diagram(ax, rnn_cx, rnn_cy, rnn_r)
    ax.text(rnn_cx, rnn_cy + rnn_r + 0.10, "RNN / SSM",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold",
            color=C_DECODER_DARK, zorder=6)

    # -> h_t -> Readout
    rnn_right = rnn_cx + rnn_r + 0.06
    ah_x = rnn_right + 0.48
    _arrow(ax, rnn_right, main_cy + 0.03, ah_x - 0.05, main_cy + 0.03)
    ax.text(rnn_right + 0.20, main_cy + 0.18, r"$h_t$", ha="center", va="bottom",
            fontsize=9.5, color=C_DECODER_DARK, fontstyle="italic")

    ah_w = 0.62
    ah_h = 0.42
    ah_by = main_cy + 0.03 - ah_h / 2
    _block(ax, ah_x, ah_by, ah_w, ah_h, C_DECODER, "Readout",
           fontsize=9, fontweight="bold", border_color=C_DECODER_DARK,
           sublabel=r"$[h_t, z_0] \to a_t$", sublabel_size=6,
           text_color="white")
    ah_cx = ah_x + ah_w / 2

    # -> a_t
    at_x = ah_x + ah_w + 0.35
    _arrow(ax, ah_x + ah_w + 0.05, main_cy + 0.03, at_x - 0.12, main_cy + 0.03)
    ax.text(at_x, main_cy + 0.03, r"$a_t$", ha="center", va="center",
            fontsize=13, color=C_TEXT)

    # -> Physics
    phys_x = at_x + 0.32
    phys_w = 0.58
    phys_h = 0.46
    phys_by = main_cy + 0.03 - phys_h / 2
    _arrow(ax, at_x + 0.13, main_cy + 0.03, phys_x - 0.05, main_cy + 0.03)
    _block(ax, phys_x, phys_by, phys_w, phys_h, C_PHYSICS,
           "Physics\nSim", fontsize=8.5, border_color="#666666",
           text_color="white", sublabel=r"$s_{t+1}^p$", sublabel_size=7)
    phys_cx = phys_x + phys_w / 2

    # =====================================================================
    # s^p_t + feedback
    # =====================================================================
    sp_x = rnn_cx
    sp_y = main_cy - rnn_r - 0.55
    ax.text(sp_x, sp_y, r"$s_t^p$", ha="center", va="center",
            fontsize=12, color=C_TEXT, fontweight="bold")

    # s^p_t -> RNN (straight vertical arrow up)
    _arrow(ax, sp_x, sp_y + 0.15, sp_x, rnn_cy - rnn_r - 0.06,
           color=C_FEEDBACK, lw=1.2)

    # Feedback: Physics -> dashed -> s^p_t
    fb_y = sp_y - 0.35
    ax.plot([phys_cx, phys_cx, sp_x, sp_x],
            [phys_by, fb_y, fb_y, sp_y - 0.18],
            color=C_FEEDBACK, lw=1.0, ls="--", zorder=1)
    ax.annotate("", xy=(sp_x, sp_y - 0.15), xytext=(sp_x, sp_y - 0.18),
                arrowprops=dict(arrowstyle="-|>", color=C_FEEDBACK, lw=1.0),
                zorder=2)

    # =====================================================================
    # TOP ROW: Encoder  s^g_{t:T} -> [Enc] -> [mu,sigma] ~eps-> z_0
    # =====================================================================
    enc_cy = 2.80

    # s^g_{t:T}
    sg_x = xg_x
    ax.text(sg_x, enc_cy, r"$s_{t:T}^g$", ha="center", va="center",
            fontsize=12, color=C_TEXT)

    # -> Encoder
    _arrow(ax, sg_x + 0.22, enc_cy, sg_x + 0.50, enc_cy)
    enc_x = sg_x + 0.55
    enc_w = 0.70
    enc_h = 0.50
    enc_by = enc_cy - enc_h / 2
    _trapezoid(ax, enc_x, enc_by, enc_w, enc_h, C_BLUE, "Encoder",
               direction="right", fontsize=9.5, border_color=C_BLUE_DARK,
               sublabel=r"$q_\phi(z \mid s_{t:T}^g)$", sublabel_size=6.5)

    # -> bracket
    _arrow(ax, enc_x + enc_w + 0.06, enc_cy, enc_x + enc_w + 0.30, enc_cy)
    bkt_cx = enc_x + enc_w + 0.52
    _bracket_mu_sigma(ax, bkt_cx, enc_cy, w=0.40, h=0.38)

    # -> squiggly -> z_0
    sq_x1 = bkt_cx + 0.24
    sq_x2 = sq_x1 + 0.42
    _squiggly_arrow(ax, sq_x1, enc_cy, sq_x2)

    z0_x = sq_x2 + 0.18
    z0_bw, z0_bh = 0.30, 0.26
    ax.add_patch(FancyBboxPatch(
        (z0_x - z0_bw / 2, enc_cy - z0_bh / 2), z0_bw, z0_bh,
        boxstyle="round,pad=0.03", facecolor=C_Z0, alpha=0.25,
        edgecolor=C_Z0_DARK, linewidth=1.2, zorder=3,
    ))
    ax.text(z0_x, enc_cy, r"$z_0$", ha="center", va="center",
            fontsize=13, color=C_Z0_DARK, fontweight="bold", zorder=4)

    # =====================================================================
    # IK arrow: x^g_t UP to s^g_{t:T}   labeled "IK over T frames"
    # =====================================================================
    _arrow(ax, xg_x, main_cy + 0.20, xg_x, enc_cy - 0.20,
           color=C_STAC, lw=1.5)
    ax.text(xg_x + 0.15, (main_cy + enc_cy) / 2, "IK over\nT frames",
            ha="left", va="center", fontsize=8.5, color=C_STAC,
            fontstyle="italic")

    # =====================================================================
    # z_0 -> Readout  (L-path: right then down)
    # =====================================================================
    # Horizontal segment from z_0 right to above Readout
    z0_right = z0_x + z0_bw / 2 + 0.03
    ax.plot([z0_right, ah_cx], [enc_cy, enc_cy],
            color=C_Z0_DARK, lw=1.8, ls="-", zorder=2)
    # Vertical arrow down into Readout top
    _arrow(ax, ah_cx, enc_cy, ah_cx, ah_by + ah_h + 0.05,
           color=C_Z0_DARK, lw=1.8)

    # =====================================================================
    # KL annotation (below encoder bracket)
    # =====================================================================
    kl_y = enc_cy - 0.50
    ax.text(bkt_cx, kl_y, r"$\mathrm{KL}(q_\phi \| \mathcal{N}(0, I))$",
            ha="center", va="center", fontsize=8.5, color=C_KL,
            fontweight="bold")
    _arrow(ax, bkt_cx, kl_y + 0.10, bkt_cx, enc_by - 0.05,
           color=C_KL, lw=1.2, style="-|>")
    ax.text(bkt_cx, kl_y - 0.13, "(heavily regularized)",
            ha="center", va="top", fontsize=7, color=C_KL, fontstyle="italic")

    # =====================================================================
    # CODE STRIP below c_t
    # =====================================================================
    strip_y = -0.55
    strip_h = 0.12

    _arrow(ax, ct_x, main_cy - 0.22, ct_x, strip_y + strip_h + 0.22,
           color=C_CODE_DARK, lw=1.3)
    ax.text(ct_x, strip_y + strip_h + 0.12,
            r"$[c_1, c_2, \ldots, c_T]$",
            ha="center", va="bottom", fontsize=8.5, color=C_TEXT)

    strip_w = 0.95
    _code_strip(ax, ct_x - strip_w / 2, strip_y, strip_w, strip_h, n=10)

    # =====================================================================
    # GENERATIVE PRIOR
    # =====================================================================
    prior_w = 1.20
    prior_h = 0.35
    prior_y = -1.22
    prior_cx = ct_x

    _block(ax, prior_cx - prior_w / 2, prior_y, prior_w, prior_h,
           C_PRIOR, "Generative Prior", fontsize=9.5,
           border_color=C_PRIOR_DARK,
           sublabel="HMM / Transformer / ...", sublabel_size=7,
           text_color="white")

    # train arrow (down, right of center) — tall shaft like want.png
    tx = prior_cx + 0.18
    _arrow(ax, tx, strip_y - 0.06, tx, prior_y + prior_h + 0.06,
           color=C_PRIOR_DARK, lw=2.0, zorder=5)
    ax.text(tx + 0.12, (strip_y + prior_y + prior_h) / 2,
            "train", ha="left", va="center", fontsize=8,
            color=C_PRIOR_DARK, fontstyle="italic", zorder=6)

    # generate arrow (up, dashed, left of center)
    gx = prior_cx - 0.18
    ax.annotate(
        "", xy=(gx, strip_y - 0.06),
        xytext=(gx, prior_y + prior_h + 0.06),
        arrowprops=dict(arrowstyle="-|>", color=C_PRIOR_DARK, lw=1.8, ls="--"),
        zorder=5,
    )
    ax.text(gx - 0.12, (strip_y + prior_y + prior_h) / 2,
            "generate", ha="right", va="center", fontsize=8,
            color=C_PRIOR_DARK, fontstyle="italic", zorder=6)


# =============================================================================
# Figure assembly
# =============================================================================

def make_figure(output_path: str = "outputs/moseq_pipeline"):
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))

    ax.set_xlim(-0.30, 6.50)
    ax.set_ylim(-1.55, 3.60)
    ax.set_aspect("equal")
    ax.axis("off")

    # Panel background
    ax.add_patch(FancyBboxPatch(
        (-0.15, -1.40), 6.50, 4.80,
        boxstyle="round,pad=0.05", facecolor=C_PANEL_BG,
        edgecolor="#CCCCCC", linewidth=1.0, zorder=0,
    ))

    ax.text(0.05, 3.30, "KPMS Decoder Pipeline",
            fontsize=13, fontweight="bold", color=C_TEXT, zorder=5)

    draw_moseq_pipeline(ax)

    base = str(output_path).replace(".pdf", "").replace(".png", "")
    Path(base).parent.mkdir(parents=True, exist_ok=True)
    for ext in [".pdf", ".png"]:
        p = base + ext
        fig.savefig(p, dpi=400, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"Saved: {p}")
    plt.close(fig)


# =============================================================================
# Prior distillation variant
# =============================================================================

# Warm orange for the distillation prior head
C_DISTILL = "#E89850"
C_DISTILL_DARK = "#C07030"


def draw_moseq_prior_distill_pipeline(ax):
    """Draw pipeline with distillation head replacing direct z concat.

    Compared with ``draw_moseq_pipeline``:

    * A **Distill Head** (short-side-up trapezoid) reads h_t from the
      RNN and outputs its own (μ^d, σ^d).
    * The encoder (train-only) outputs (μ^e, σ^e).
    * A KL node sits between the two distributions — encoder feeds from
      the left, distill head feeds from below.
    * The Readout receives only h_t (no z concatenation).
    """

    # =====================================================================
    # MAIN ROW
    # =====================================================================
    main_cy = 1.20

    xg_x = 0.40
    ax.text(xg_x, main_cy, r"$x_t^g$", ha="center", va="center",
            fontsize=13, color=C_TEXT)

    _arrow(ax, xg_x + 0.18, main_cy, xg_x + 0.48, main_cy)
    kpms_x = xg_x + 0.53
    kpms_w = 0.95
    kpms_h = 0.50
    kpms_by = main_cy - kpms_h / 2
    _block(ax, kpms_x, kpms_by, kpms_w, kpms_h, C_BLUE, "Keypoint-\nMoSeq",
           fontsize=9, border_color=C_BLUE_DARK,
           sublabel=r"$c_t = \mathrm{MoSeq}(x_t^g)$", sublabel_size=6.5)

    ct_x = kpms_x + kpms_w + 0.30
    _arrow(ax, kpms_x + kpms_w + 0.05, main_cy, ct_x - 0.12, main_cy)
    ax.text(ct_x, main_cy, r"$c_t$", ha="center", va="center",
            fontsize=14, color=C_CODE_DARK, fontweight="bold")

    rnn_cx = ct_x + 0.95
    rnn_cy = main_cy
    rnn_r = 0.48
    _arrow(ax, ct_x + 0.14, main_cy, rnn_cx - rnn_r - 0.06, main_cy)
    _rnn_diagram(ax, rnn_cx, rnn_cy, rnn_r)
    # Label moved LEFT so RNN top is clear for the h_t upward arrow
    ax.text(rnn_cx - rnn_r - 0.12, rnn_cy + 0.15, "RNN /\nSSM",
            ha="right", va="center", fontsize=9, fontweight="bold",
            color=C_DECODER_DARK, zorder=6)

    rnn_right = rnn_cx + rnn_r + 0.06
    ah_x = rnn_right + 0.48
    _arrow(ax, rnn_right, main_cy + 0.03, ah_x - 0.05, main_cy + 0.03)
    ax.text(rnn_right + 0.20, main_cy + 0.18, r"$h_t$", ha="center",
            va="bottom", fontsize=9.5, color=C_DECODER_DARK,
            fontstyle="italic")

    ah_w = 0.62
    ah_h = 0.42
    ah_by = main_cy + 0.03 - ah_h / 2
    ah_cx = ah_x + ah_w / 2
    # Readout only receives h_t — no z concatenation
    _block(ax, ah_x, ah_by, ah_w, ah_h, C_DECODER, "Readout",
           fontsize=9, fontweight="bold", border_color=C_DECODER_DARK,
           sublabel=r"$h_t \to a_t$", sublabel_size=6,
           text_color="white")

    at_x = ah_x + ah_w + 0.35
    _arrow(ax, ah_x + ah_w + 0.05, main_cy + 0.03, at_x - 0.12,
           main_cy + 0.03)
    ax.text(at_x, main_cy + 0.03, r"$a_t$", ha="center", va="center",
            fontsize=13, color=C_TEXT)

    phys_x = at_x + 0.32
    phys_w = 0.58
    phys_h = 0.46
    phys_by = main_cy + 0.03 - phys_h / 2
    _arrow(ax, at_x + 0.13, main_cy + 0.03, phys_x - 0.05, main_cy + 0.03)
    _block(ax, phys_x, phys_by, phys_w, phys_h, C_PHYSICS,
           "Physics\nSim", fontsize=8.5, border_color="#666666",
           text_color="white", sublabel=r"$s_{t+1}^p$", sublabel_size=7)
    phys_cx = phys_x + phys_w / 2

    # =====================================================================
    # s^p_t + feedback
    # =====================================================================
    sp_x = rnn_cx
    sp_y = main_cy - rnn_r - 0.55
    ax.text(sp_x, sp_y, r"$s_t^p$", ha="center", va="center",
            fontsize=12, color=C_TEXT, fontweight="bold")
    _arrow(ax, sp_x, sp_y + 0.15, sp_x, rnn_cy - rnn_r - 0.06,
           color=C_FEEDBACK, lw=1.2)
    fb_y = sp_y - 0.35
    ax.plot([phys_cx, phys_cx, sp_x, sp_x],
            [phys_by, fb_y, fb_y, sp_y - 0.18],
            color=C_FEEDBACK, lw=1.0, ls="--", zorder=1)
    ax.annotate("", xy=(sp_x, sp_y - 0.15), xytext=(sp_x, sp_y - 0.18),
                arrowprops=dict(arrowstyle="-|>", color=C_FEEDBACK, lw=1.0),
                zorder=2)

    # =====================================================================
    # DISTILL HEAD — short-side-up trapezoid above RNN
    # =====================================================================
    dh_cx = rnn_cx            # centered above RNN
    dh_w = 0.55               # top (short) width
    dh_h = 0.40               # height
    dh_by = 1.88              # bottom y
    dh_ty = dh_by + dh_h      # top y
    taper = dh_w * 0.22

    # Wide at bottom, narrow at top
    dh_verts = [
        (dh_cx - dh_w / 2 - taper, dh_by),   # bottom-left (wide)
        (dh_cx + dh_w / 2 + taper, dh_by),   # bottom-right (wide)
        (dh_cx + dh_w / 2, dh_ty),            # top-right (narrow)
        (dh_cx - dh_w / 2, dh_ty),            # top-left (narrow)
    ]
    ax.add_patch(Polygon(dh_verts, closed=True, facecolor=C_DISTILL,
                         edgecolor=C_DISTILL_DARK, linewidth=1.5, zorder=3))
    ax.text(dh_cx, dh_by + dh_h * 0.58, "Distill\nHead",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="white", zorder=4)
    ax.text(dh_cx, dh_by + dh_h * 0.20,
            r"$p_\psi(z \mid h_t)$", ha="center", va="center",
            fontsize=6.5, fontstyle="italic", color="white",
            alpha=0.85, zorder=4)

    # h_t upward from RNN top to Distill Head bottom
    _arrow(ax, rnn_cx, rnn_cy + rnn_r + 0.05,
           dh_cx, dh_by - 0.06,
           color=C_DISTILL_DARK, lw=1.5)
    ax.text(rnn_cx + 0.15,
            (rnn_cy + rnn_r + 0.05 + dh_by - 0.06) / 2,
            r"$h_t$", ha="left", va="center", fontsize=8.5,
            color=C_DISTILL_DARK, fontstyle="italic", zorder=5)

    # Distill bracket [μ^d, σ^d] above Distill Head
    dist_bkt_cx = dh_cx
    dist_bkt_cy = dh_ty + 0.23
    _bracket_labeled(ax, dist_bkt_cx, dist_bkt_cy,
                     r"$\mu_t^d$", r"$\sigma_t^d$", w=0.35, h=0.30)

    # =====================================================================
    # ENCODER ROW — pushed higher, no sampling / z_e
    # =====================================================================
    enc_cy = 3.30

    sg_x = xg_x
    ax.text(sg_x, enc_cy, r"$s_{t:T}^g$", ha="center", va="center",
            fontsize=12, color=C_TEXT)

    _arrow(ax, sg_x + 0.22, enc_cy, sg_x + 0.50, enc_cy)
    enc_x = sg_x + 0.55
    enc_w = 0.70
    enc_h = 0.50
    enc_by = enc_cy - enc_h / 2
    _trapezoid(ax, enc_x, enc_by, enc_w, enc_h, C_BLUE, "Encoder",
               direction="right", fontsize=9.5, border_color=C_BLUE_DARK,
               sublabel=r"$q_\phi(z \mid s_{t:T}^g)$", sublabel_size=6.5)

    # Encoder -> bracket [μ^e, σ^e]  (no squiggly / z_e)
    _arrow(ax, enc_x + enc_w + 0.06, enc_cy, enc_x + enc_w + 0.30, enc_cy)
    enc_bkt_cx = enc_x + enc_w + 0.52
    _bracket_labeled(ax, enc_bkt_cx, enc_cy,
                     r"$\mu_t^e$", r"$\sigma_t^e$")

    # "train only" annotation
    ax.text(enc_x + enc_w / 2, enc_by - 0.15, "(train only)",
            ha="center", va="top", fontsize=7.5, color=C_BLUE_DARK,
            fontstyle="italic", alpha=0.7)

    # IK arrow
    _arrow(ax, xg_x, main_cy + 0.20, xg_x, enc_cy - 0.20,
           color=C_STAC, lw=1.5)
    ax.text(xg_x + 0.15, (main_cy + enc_cy) / 2, "IK over\nT frames",
            ha="left", va="center", fontsize=8.5, color=C_STAC,
            fontstyle="italic")

    # =====================================================================
    # KL NODE — junction of encoder (from left) and distill (from below)
    # =====================================================================
    kl_cx = dh_cx             # same x as distill head
    kl_cy = enc_cy            # same y as encoder

    # Dashed line: encoder bracket → KL (rightward)
    ax.plot([enc_bkt_cx + 0.22, kl_cx - 0.35], [kl_cy, kl_cy],
            color=C_KL, lw=1.5, ls="--", zorder=2)

    # Dashed line: distill bracket → KL (upward)
    dist_bkt_top = dist_bkt_cy + 0.15 + 0.05
    ax.plot([kl_cx, kl_cx], [dist_bkt_top, kl_cy - 0.15],
            color=C_KL, lw=1.5, ls="--", zorder=2)

    # KL text (plain, no box)
    ax.text(kl_cx, kl_cy, r"$\mathrm{KL}(p_\psi \| q_\phi)$",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=C_KL, zorder=5)

    # =====================================================================
    # CODE STRIP
    # =====================================================================
    strip_y = -0.55
    strip_h = 0.12

    _arrow(ax, ct_x, main_cy - 0.22, ct_x, strip_y + strip_h + 0.22,
           color=C_CODE_DARK, lw=1.3)
    ax.text(ct_x, strip_y + strip_h + 0.12,
            r"$[c_1, c_2, \ldots, c_T]$",
            ha="center", va="bottom", fontsize=8.5, color=C_TEXT)

    strip_w = 0.95
    _code_strip(ax, ct_x - strip_w / 2, strip_y, strip_w, strip_h, n=10)

    # =====================================================================
    # GENERATIVE PRIOR
    # =====================================================================
    gen_prior_w = 1.20
    gen_prior_h = 0.35
    gen_prior_y = -1.22
    gen_prior_cx = ct_x

    _block(ax, gen_prior_cx - gen_prior_w / 2, gen_prior_y,
           gen_prior_w, gen_prior_h, C_PRIOR, "Generative Prior",
           fontsize=9.5, border_color=C_PRIOR_DARK,
           sublabel="HMM / Transformer / ...", sublabel_size=7,
           text_color="white")

    tx = gen_prior_cx + 0.18
    _arrow(ax, tx, strip_y - 0.06, tx, gen_prior_y + gen_prior_h + 0.06,
           color=C_PRIOR_DARK, lw=2.0, zorder=5)
    ax.text(tx + 0.12, (strip_y + gen_prior_y + gen_prior_h) / 2,
            "train", ha="left", va="center", fontsize=8,
            color=C_PRIOR_DARK, fontstyle="italic", zorder=6)

    gx = gen_prior_cx - 0.18
    ax.annotate(
        "", xy=(gx, strip_y - 0.06),
        xytext=(gx, gen_prior_y + gen_prior_h + 0.06),
        arrowprops=dict(arrowstyle="-|>", color=C_PRIOR_DARK, lw=1.8,
                        ls="--"),
        zorder=5,
    )
    ax.text(gx - 0.12, (strip_y + gen_prior_y + gen_prior_h) / 2,
            "generate", ha="right", va="center", fontsize=8,
            color=C_PRIOR_DARK, fontstyle="italic", zorder=6)


def make_prior_distill_figure(
    output_path: str = "outputs/moseq_prior_distill_pipeline",
):
    """Generate the prior-distillation variant of the pipeline figure."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    ax.set_xlim(-0.30, 6.50)
    ax.set_ylim(-1.55, 4.20)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(FancyBboxPatch(
        (-0.15, -1.40), 6.50, 5.40,
        boxstyle="round,pad=0.05", facecolor=C_PANEL_BG,
        edgecolor="#CCCCCC", linewidth=1.0, zorder=0,
    ))

    ax.text(0.05, 3.85, "KPMS Decoder Pipeline (Prior Distillation)",
            fontsize=13, fontweight="bold", color=C_TEXT, zorder=5)

    draw_moseq_prior_distill_pipeline(ax)

    base = str(output_path).replace(".pdf", "").replace(".png", "")
    Path(base).parent.mkdir(parents=True, exist_ok=True)
    for ext in [".pdf", ".png"]:
        p = base + ext
        fig.savefig(p, dpi=400, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"Saved: {p}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize the KPMS decoder pipeline",
    )
    parser.add_argument(
        "--output", default="outputs/moseq_pipeline",
        help="Output path for the base pipeline (saves .pdf and .png)",
    )
    parser.add_argument(
        "--distill-output",
        default="outputs/moseq_prior_distill_pipeline",
        help="Output path for the prior distillation variant",
    )
    args = parser.parse_args()
    make_figure(args.output)
    make_prior_distill_figure(args.distill_output)
