"""Visualize the Mimic-MJX and Code2Act pipeline architectures.

Outputs two publication-quality schematic figures:
  1. mimic_training — Encoder-decoder training architecture (Mimic-MJX)
  2. code2act_pipeline — Code2Act inference with KPMS decoder + distillation

Usage:
    cd moseq_jax/figures
    python visualize_moseq_pipeline.py
    python visualize_moseq_pipeline.py --output-dir outputs
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
import matplotlib.image as mpimg
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


# =============================================================================
# Color palette (matching mimic.png)
# =============================================================================
C_BLUE = "#7BA3CC"
C_BLUE_DARK = "#5B83AC"
C_DECODER = "#C0A0D0"
C_DECODER_DARK = "#9070A0"
C_CODE = "#7BC47F"
C_CODE_DARK = "#4A9E4F"
C_PHYSICS = "#888888"
C_ARROW = "#444444"
C_TEXT = "#222222"
C_PANEL_BG = "#F8F8FC"
C_FEEDBACK = "#999999"
C_KL = "#CC6677"
C_RNN_FWD = "#9070A0"
C_RNN_REC = "#5BA8D0"
C_DISTILL = "#E89850"
C_DISTILL_DARK = "#C07030"
C_Z0 = "#5B83AC"
C_Z0_DARK = "#3A6390"
C_STAC = "#777777"
C_REWARD = "#E06040"
C_REWARD_DARK = "#B84030"

STRIPE_COLORS = ["#7BA3CC", "#B090C0", "#7BC47F", "#CC9966", "#6699BB", "#C08080"]


# =============================================================================
# Drawing primitives
# =============================================================================


def _block(
    ax,
    x,
    y,
    w,
    h,
    color,
    label,
    fontsize=9,
    fontweight="bold",
    border_color=None,
    sublabel=None,
    sublabel_size=7,
    text_color="white",
    zorder=3,
):
    bc = border_color or color
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04",
            facecolor=color,
            edgecolor=bc,
            linewidth=1.5,
            zorder=zorder,
        )
    )
    ty = y + h * 0.58 if sublabel else y + h / 2
    ax.text(
        x + w / 2,
        ty,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color=text_color,
        zorder=zorder + 1,
    )
    if sublabel:
        ax.text(
            x + w / 2,
            y + h * 0.25,
            sublabel,
            ha="center",
            va="center",
            fontsize=sublabel_size,
            fontstyle="italic",
            color=text_color,
            alpha=0.85,
            zorder=zorder + 1,
        )


def _trapezoid(
    ax,
    x,
    y,
    w,
    h,
    color,
    label,
    direction="right",
    fontsize=9,
    fontweight="bold",
    border_color=None,
    sublabel=None,
    sublabel_size=7,
    text_color="white",
):
    bc = border_color or color
    taper = h * 0.22
    if direction == "right":
        verts = [(x, y - taper), (x + w, y), (x + w, y + h), (x, y + h + taper)]
    else:
        verts = [(x, y), (x + w, y - taper), (x + w, y + h + taper), (x, y + h)]
    ax.add_patch(
        Polygon(
            verts,
            closed=True,
            facecolor=color,
            edgecolor=bc,
            linewidth=1.5,
            zorder=3,
        )
    )
    ty = y + h * 0.58 if sublabel else y + h / 2
    ax.text(
        x + w / 2,
        ty,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color=text_color,
        zorder=4,
    )
    if sublabel:
        ax.text(
            x + w / 2,
            y + h * 0.25,
            sublabel,
            ha="center",
            va="center",
            fontsize=sublabel_size,
            fontstyle="italic",
            color=text_color,
            alpha=0.85,
            zorder=4,
        )


def _arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.5, style="-|>", zorder=2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle=style,
            color=color,
            lw=lw,
            zorder=zorder,
            mutation_scale=13,
        )
    )


def _bracket_labeled(ax, cx, cy, top_text, bot_text, w=0.40, h=0.38):
    """Draw a bracket pair with custom top/bottom labels."""
    left, right = cx - w / 2, cx + w / 2
    top, bot = cy + h / 2, cy - h / 2
    hook = 0.04
    ax.plot(
        [left + hook, left, left, left + hook],
        [top, top, bot, bot],
        color=C_TEXT,
        lw=1.6,
        zorder=4,
        solid_capstyle="round",
    )
    ax.plot(
        [right - hook, right, right, right - hook],
        [top, top, bot, bot],
        color=C_TEXT,
        lw=1.6,
        zorder=4,
        solid_capstyle="round",
    )
    ax.text(
        cx,
        cy + h * 0.25,
        top_text,
        ha="center",
        va="center",
        fontsize=10,
        color=C_TEXT,
        zorder=5,
    )
    ax.text(
        cx,
        cy - h * 0.25,
        bot_text,
        ha="center",
        va="center",
        fontsize=10,
        color=C_TEXT,
        zorder=5,
    )


def _squiggly_arrow(ax, x1, y, x2, label=r"$\epsilon$"):
    xs = np.linspace(x1 + 0.02, x2 - 0.06, 80)
    amp = 0.025
    n_waves = max(2, int((x2 - x1) / 0.12))
    ys = y + amp * np.sin(2 * np.pi * n_waves * (xs - xs[0]) / (xs[-1] - xs[0]))
    ax.plot(xs, ys, color=C_TEXT, lw=1.0, zorder=4)
    ax.annotate(
        "",
        xy=(x2, y),
        xytext=(xs[-1], ys[-1]),
        arrowprops=dict(arrowstyle="-|>", color=C_TEXT, lw=1.0),
        zorder=4,
    )
    if label:
        ax.text(
            (x1 + x2) / 2,
            y + 0.07,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            color=C_TEXT,
            zorder=5,
        )


def _latent_stripes(ax, x, y, w, h, n=5, alpha=0.6):
    """Draw horizontal colored stripes representing a latent vector."""
    gap = h * 0.08
    bar_h = (h - gap * (n + 1)) / n
    for i in range(n):
        by = y + gap + i * (bar_h + gap)
        c = STRIPE_COLORS[i % len(STRIPE_COLORS)]
        ax.add_patch(
            FancyBboxPatch(
                (x, by),
                w,
                bar_h,
                boxstyle="round,pad=0.005",
                facecolor=c,
                alpha=alpha,
                edgecolor="none",
                zorder=3,
            )
        )


def _distribution_curve(ax, cx, cy, w=0.25, h=0.20):
    """Draw a small Gaussian distribution bell curve."""
    xs = np.linspace(-2.5, 2.5, 60)
    ys = np.exp(-(xs**2) / 2)
    x_plot = cx + xs * w / 5
    y_plot = cy - h * 0.3 + ys * h * 0.6
    ax.plot(x_plot, y_plot, color=C_TEXT, lw=1.0, zorder=4)


def _rnn_diagram(ax, cx, cy, radius):
    """Draw a stylized RNN as a double-ring circle with organic nodes."""
    ax.add_patch(
        MplCircle(
            (cx, cy),
            radius,
            facecolor=C_DECODER,
            alpha=0.08,
            edgecolor="#222222",
            linewidth=3.0,
            zorder=3,
        )
    )
    ax.add_patch(
        MplCircle(
            (cx, cy),
            radius * 0.84,
            facecolor="none",
            edgecolor=C_DECODER_DARK,
            linewidth=0.7,
            zorder=3,
            alpha=0.30,
        )
    )

    r = radius * 0.60
    nr = radius * 0.072

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
        ax.add_patch(
            FancyArrowPatch(
                (p1[0] + dx * s, p1[1] + dy * s),
                (p2[0] - dx * s, p2[1] - dy * s),
                arrowstyle="-|>",
                color=color,
                lw=lw,
                zorder=4,
                mutation_scale=7,
                alpha=alpha,
            )
        )

    for i, j in [
        (0, 1),
        (0, 4),
        (1, 2),
        (1, 5),
        (2, 6),
        (3, 4),
        (3, 7),
        (4, 5),
        (4, 8),
        (5, 6),
        (5, 9),
        (7, 8),
        (8, 9),
        (0, 3),
        (2, 5),
        (6, 9),
    ]:
        _thin(nodes[i], nodes[j], C_RNN_FWD, lw=0.80)

    for i, j, rad in [
        (6, 0, -0.45),
        (9, 3, 0.50),
        (5, 0, -0.35),
        (6, 4, -0.30),
        (9, 7, 0.40),
        (8, 3, 0.35),
    ]:
        ax.annotate(
            "",
            xy=(nodes[j][0] + nr * 0.5, nodes[j][1]),
            xytext=(nodes[i][0] - nr * 0.5, nodes[i][1]),
            arrowprops=dict(
                arrowstyle="-|>",
                color=C_RNN_REC,
                lw=1.0,
                connectionstyle=f"arc3,rad={rad}",
                alpha=0.55,
            ),
            zorder=4,
        )

    for nx, ny in nodes:
        ax.add_patch(
            MplCircle(
                (nx, ny),
                nr,
                facecolor="white",
                edgecolor="#444444",
                linewidth=0.9,
                zorder=5,
            )
        )

    for dx, dy in [
        (cx - r * 0.30, cy + r * 0.35),
        (cx + r * 0.05, cy - r * 0.18),
        (cx + r * 0.48, cy + r * 0.28),
    ]:
        ax.add_patch(
            MplCircle(
                (dx, dy),
                nr * 0.45,
                facecolor=C_BLUE,
                edgecolor="none",
                zorder=5,
                alpha=0.70,
            )
        )


# =============================================================================
# Figure 1: Mimic-MJX Encoder-Decoder Training
# =============================================================================


def draw_mimic_training(ax):
    """Draw Mimic-MJX encoder-decoder training (matching mimic.png)."""
    main_y = 1.20

    # --- s^g_{t:T} ---
    sg_x = 0.40
    ax.text(sg_x, main_y, r"$s_{t:T}^g$", ha="center", va="center",
            fontsize=14, color=C_TEXT, zorder=5)

    # --- Arrow to Encoder ---
    _arrow(ax, sg_x + 0.25, main_y, 0.95, main_y)

    # --- Encoder ---
    enc_x = 1.00
    enc_w = 1.10
    enc_h = 0.82
    enc_by = main_y - enc_h / 2
    _trapezoid(ax, enc_x, enc_by, enc_w, enc_h, C_BLUE, "Encoder",
               direction="right", fontsize=13, border_color=C_BLUE_DARK,
               sublabel=r"$q_\phi(z_t \mid s_t^g)$", sublabel_size=9)

    # --- Bracket [mu_t, sigma_t] — no arrow, height matches encoder ---
    bkt_cx = enc_x + enc_w + 0.22
    _bracket_labeled(ax, bkt_cx, main_y, r"$\mu_t$", r"$\sigma_t$",
                     w=0.38, h=enc_h)

    # --- Squiggly epsilon ---
    sq_x1 = bkt_cx + 0.22
    sq_x2 = sq_x1 + 0.48
    _squiggly_arrow(ax, sq_x1, main_y, sq_x2)

    # --- z_t BLUE block ---
    zt_x = sq_x2 + 0.10
    zt_w = 0.30
    zt_h = 0.55
    zt_by = main_y - zt_h / 2
    ax.add_patch(FancyBboxPatch(
        (zt_x, zt_by), zt_w, zt_h, boxstyle="round,pad=0.03",
        facecolor=C_Z0, alpha=0.30, edgecolor=C_Z0_DARK,
        linewidth=1.2, zorder=3))
    ax.text(zt_x + zt_w / 2, main_y, r"$z_t$", ha="center", va="center",
            fontsize=13, color=C_Z0_DARK, fontweight="bold", zorder=6)

    # --- Arrow z_t -> decoder ---
    dec_x = zt_x + zt_w + 0.40
    _arrow(ax, zt_x + zt_w + 0.05, main_y, dec_x - 0.05, main_y)

    # --- Decoder ---
    dec_w = 1.10
    dec_h = 0.82
    dec_by = main_y - dec_h / 2
    _trapezoid(ax, dec_x, dec_by, dec_w, dec_h, C_DECODER, "Decoder",
               direction="left", fontsize=13, border_color=C_DECODER_DARK,
               sublabel=r"$p_\theta(a_t \mid z_t, s_t^p)$", sublabel_size=9)

    # --- p(a_t) text (no arrow before it) ---
    pa_x = dec_x + dec_w + 0.20
    ax.text(pa_x, main_y, r"$p(a_t)$", ha="center", va="center",
            fontsize=11, color=C_TEXT, zorder=5)

    # --- Squiggly epsilon to a_t ---
    sq2_x1 = pa_x + 0.22
    sq2_x2 = sq2_x1 + 0.48
    _squiggly_arrow(ax, sq2_x1, main_y, sq2_x2)

    # --- a_t ---
    at_x = sq2_x2 + 0.15
    ax.text(at_x, main_y, r"$a_t$", ha="center", va="center",
            fontsize=14, color=C_TEXT, zorder=5)

    # --- Arrow to Physics Sim ---
    _arrow(ax, at_x + 0.15, main_y, at_x + 0.40, main_y)

    # --- Physics Sim ---
    phys_x = at_x + 0.45
    phys_w = 0.65
    phys_h = 0.55
    phys_by = main_y - phys_h / 2
    _block(ax, phys_x, phys_by, phys_w, phys_h, C_PHYSICS,
           "Physics\nSim", fontsize=10, border_color="#666666", text_color="white")

    # --- Arrow to s^p_{t+1} ---
    spt1_x = phys_x + phys_w + 0.42
    _arrow(ax, phys_x + phys_w + 0.05, main_y, spt1_x - 0.12, main_y)
    ax.text(spt1_x, main_y, r"$s_{t+1}^p$", ha="center", va="center",
            fontsize=12, color=C_TEXT, zorder=5)

    # --- Dashed feedback with s^p_t in the middle of horizontal line ---
    fb_y = main_y - 0.72
    dec_fb_x = dec_x + 0.15
    ax.plot([spt1_x, spt1_x], [main_y - 0.18, fb_y],
            color=C_FEEDBACK, lw=1.2, ls="--", zorder=1)
    ax.plot([spt1_x, dec_fb_x], [fb_y, fb_y],
            color=C_FEEDBACK, lw=1.2, ls="--", zorder=1)
    ax.plot([dec_fb_x, dec_fb_x], [fb_y, dec_by - 0.02],
            color=C_FEEDBACK, lw=1.2, ls="--", zorder=1)
    _arrow(ax, dec_fb_x, fb_y + 0.08, dec_fb_x, dec_by - 0.02,
           color=C_FEEDBACK, lw=1.2)
    # s^p_t label on the horizontal dashed line
    sp_mid_x = (spt1_x + dec_fb_x) / 2
    ax.text(sp_mid_x, fb_y, r"$s_t^p$", ha="center", va="center",
            fontsize=11, color=C_TEXT, fontweight="bold", zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))


def make_mimic_figure(output_dir: Path):
    """Generate Figure 1: Mimic-MJX encoder-decoder training."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 3.0))

    ax.set_xlim(0.10, 7.80)
    ax.set_ylim(0.20, 2.10)
    ax.set_aspect("equal")
    ax.axis("off")


    draw_mimic_training(ax)

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in [".pdf", ".png", ".svg"]:
        p = output_dir / f"mimic_training{ext}"
        fc = "none" if ext == ".svg" else "white"
        fig.savefig(p, dpi=400, bbox_inches="tight",
                    facecolor=fc, edgecolor="none", transparent=(ext == ".svg"))
        print(f"Saved: {p}")
    plt.close(fig)


# =============================================================================
# Figure 2: Code2Act Pipeline
# =============================================================================


def draw_code2act_pipeline(ax):
    """Draw the Code2Act inference pipeline with KPMS decoder."""
    main_y = 1.10
    enc_cy = 2.50

    # --- Rodent image above KPMS block ---
    kpms_x = 1.67  # defined here for positioning; block drawn below
    kpms_w = 0.82
    rod_cx = kpms_x + kpms_w / 2
    rod_cy = enc_cy + 0.15
    img_path = SCRIPT_DIR / "data" / "rodent_stac.png"
    if img_path.exists():
        img = mpimg.imread(str(img_path))
        h_img, w_img = img.shape[:2]
        img_rodent = img[:, int(w_img * 0.75) :, :]
        rh, rw = img_rodent.shape[:2]
        aspect_r = rw / rh
        display_h = 0.85
        display_w = display_h * aspect_r
        extent = [
            rod_cx - display_w / 2, rod_cx + display_w / 2,
            rod_cy - display_h / 2, rod_cy + display_h / 2,
        ]
        ax.imshow(img_rodent, extent=extent, aspect="auto", zorder=3,
                  clip_on=True)
    else:
        ax.text(rod_cx, rod_cy, "[Rodent]", ha="center", va="center",
                fontsize=9, color=C_TEXT)

    # --- Straight-down arrow: Rodent -> x^g_t -> KPMS ---
    xg_label_y = (rod_cy - 0.42 + main_y + 0.30) / 2
    _arrow(ax, rod_cx, rod_cy - 0.48, rod_cx, main_y + 0.30,
           color=C_STAC, lw=1.3)
    ax.text(rod_cx + 0.18, xg_label_y, r"$x_t^g$", ha="left", va="center",
            fontsize=11, color=C_TEXT, zorder=5)
    ax.text(rod_cx + 0.18, xg_label_y - 0.16, "keypoints", ha="left", va="top",
            fontsize=7.5, color=C_STAC, fontstyle="italic")

    # --- Straight-right arrow: Rodent -> s^g_t -> Encoder ---
    # Encoder x position (defined later, but we know it from the distill head)
    # dh_x is ~rnn_cx + rnn_r + 0.46; enc_x = dh_x. For now use enc_cy row.
    sg_arrow_start_x = rod_cx + 0.30
    sg_arrow_end_x = kpms_x + kpms_w + 1.10 + 0.48 + 0.46 + 0.40 - 0.05  # enc_x approx
    _arrow(ax, sg_arrow_start_x, enc_cy, sg_arrow_end_x, enc_cy,
           color=C_STAC, lw=1.3)
    sg_label_x = (sg_arrow_start_x + sg_arrow_end_x) / 2
    ax.text(sg_label_x, enc_cy + 0.14, r"$s_t^g$", ha="center", va="bottom",
            fontsize=11, color=C_TEXT, zorder=5)
    ax.text(sg_label_x, enc_cy + 0.30, "joint angles", ha="center", va="bottom",
            fontsize=7.5, color=C_STAC, fontstyle="italic")

    # --- KPMS block (kpms_x, kpms_w defined above for rodent positioning) ---
    kpms_h = 0.46
    kpms_by = main_y - kpms_h / 2
    _block(ax, kpms_x, kpms_by, kpms_w, kpms_h, C_BLUE,
           "Keypoint-\nMoSeq", fontsize=8.5, border_color=C_BLUE_DARK,
           sublabel=r"$c_t = \mathrm{MoSeq}(x_t^g)$", sublabel_size=7.5)

    # --- Single arrow KPMS -> RNN with c_t label on top ---
    rnn_cx = kpms_x + kpms_w + 1.10
    rnn_cy = main_y
    rnn_r = 0.48
    _arrow(ax, kpms_x + kpms_w + 0.05, main_y, rnn_cx - rnn_r - 0.06, main_y)
    ct_label_x = (kpms_x + kpms_w + 0.05 + rnn_cx - rnn_r - 0.06) / 2
    ax.text(ct_label_x, main_y + 0.10, r"$c_t$", ha="center", va="bottom",
            fontsize=12, color=C_CODE_DARK, fontweight="bold", zorder=5)
    ax.text(ct_label_x, main_y + 0.25, "syllable", ha="center", va="bottom",
            fontsize=7, color=C_CODE_DARK, fontstyle="italic", zorder=5)

    # --- RNN (correct aspect ratio) ---
    _rnn_diagram(ax, rnn_cx, rnn_cy, rnn_r)
    ax.text(rnn_cx, rnn_cy + rnn_r + 0.08, "RNN / SSM",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold",
            color=C_DECODER_DARK, zorder=6)

    # --- h_t ---
    rnn_right = rnn_cx + rnn_r + 0.06
    _arrow(ax, rnn_right, main_y, rnn_right + 0.35, main_y)
    ax.text(rnn_right + 0.16, main_y + 0.14, r"$h_t$",
            ha="center", va="bottom", fontsize=9.5, color=C_DECODER_DARK,
            fontstyle="italic", zorder=5)

    # --- Distill Head (BLUE trapezoid, like encoder) ---
    dh_x = rnn_right + 0.40
    dh_w = 0.70
    dh_h = 0.46
    dh_by = main_y - dh_h / 2
    _trapezoid(ax, dh_x, dh_by, dh_w, dh_h, C_BLUE, "Distill\nHead",
               direction="right", fontsize=9, border_color=C_BLUE_DARK,
               sublabel=r"$p_\psi(z \mid h_t)$", sublabel_size=7.5)
    dh_cx = dh_x + dh_w / 2

    # --- [mu^d, sigma^d] bracket (no arrow, height matches head) ---
    bkt_d_cx = dh_x + dh_w + 0.22
    _bracket_labeled(ax, bkt_d_cx, main_y, r"$\mu_t^d$", r"$\sigma_t^d$",
                     w=0.35, h=dh_h)

    # --- Squiggly eps to z ---
    sq_x1 = bkt_d_cx + 0.20
    sq_x2 = sq_x1 + 0.38
    _squiggly_arrow(ax, sq_x1, main_y, sq_x2)

    # --- z block (BLUE) ---
    z0_x = sq_x2 + 0.08
    z0_w = 0.25
    z0_h = 0.28
    ax.add_patch(FancyBboxPatch(
        (z0_x, main_y - z0_h / 2), z0_w, z0_h, boxstyle="round,pad=0.03",
        facecolor=C_Z0, alpha=0.30, edgecolor=C_Z0_DARK,
        linewidth=1.2, zorder=3))
    ax.text(z0_x + z0_w / 2, main_y, r"$z$", ha="center", va="center",
            fontsize=12, color=C_Z0_DARK, fontweight="bold", zorder=4)
    ax.text(z0_x + z0_w / 2, main_y + z0_h / 2 + 0.06, "continuous\nlatent",
            ha="center", va="bottom", fontsize=6.5, color=C_Z0_DARK,
            fontstyle="italic", zorder=5, linespacing=0.9)

    # --- Clear arrow z -> Decoder ---
    dec_x = z0_x + z0_w + 0.28
    _arrow(ax, z0_x + z0_w + 0.05, main_y, dec_x - 0.05, main_y, lw=1.8)

    # --- Decoder (Inverse Dynamics Model) ---
    dec_w = 0.90
    dec_h = 0.50
    dec_by = main_y - dec_h / 2
    _trapezoid(ax, dec_x, dec_by, dec_w, dec_h, C_DECODER, "Inverse\nDynamics",
               direction="left", fontsize=9, border_color=C_DECODER_DARK,
               sublabel=r"$[z, s_t^p] \to a_t$", sublabel_size=7.5)
    # Frozen symbol on decoder
    ax.text(dec_x + 0.12, dec_by + dec_h * 0.22, "\u2744",
            ha="center", va="center", fontsize=11, color="white",
            alpha=0.9, zorder=5)
    dec_cx = dec_x + dec_w / 2

    # --- a_t ---
    at_x = dec_x + dec_w + 0.38
    _arrow(ax, dec_x + dec_w + 0.05, main_y, at_x - 0.12, main_y,
           color="#222222", lw=2.0)
    ax.text(at_x, main_y, r"$a_t$", ha="center", va="center",
            fontsize=13, fontweight="bold", color=C_TEXT, zorder=5)
    ax.text(at_x, main_y + 0.18, "action", ha="center", va="bottom",
            fontsize=6.5, color=C_TEXT, fontstyle="italic", zorder=5)

    # --- Physics Sim ---
    phys_x = at_x + 0.35
    phys_w = 0.58
    phys_h = 0.44
    phys_by = main_y - phys_h / 2
    _arrow(ax, at_x + 0.12, main_y, phys_x - 0.05, main_y,
           color="#222222", lw=2.0)
    _block(ax, phys_x, phys_by, phys_w, phys_h, C_PHYSICS,
           "Physics\nSim", fontsize=8, border_color="#666666",
           text_color="white", sublabel=r"$s_{t+1}^p$", sublabel_size=7.5)
    phys_cx = phys_x + phys_w / 2

    # =====================================================================
    # Feedback: dashed lines from Physics to RNN and Decoder
    # =====================================================================
    fb_y = main_y - rnn_r - 0.25

    # Physics -> down -> left (all dashed)
    ax.plot([phys_cx, phys_cx, rnn_cx],
            [phys_by, fb_y, fb_y],
            color=C_FEEDBACK, lw=1.0, ls="--", zorder=1)

    # s^p_t label on horizontal line
    sp_label_x = (rnn_cx + dec_cx) / 2
    ax.text(sp_label_x, fb_y, r"$s_t^p$", ha="center", va="center",
            fontsize=10, color=C_TEXT, fontweight="bold", zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.text(sp_label_x, fb_y - 0.14, "proprioceptive state",
            ha="center", va="top", fontsize=6.5, color=C_TEXT,
            fontstyle="italic", zorder=5)

    # Dashed up to RNN
    ax.plot([rnn_cx, rnn_cx], [fb_y, rnn_cy - rnn_r - 0.06],
            color=C_FEEDBACK, lw=1.0, ls="--", zorder=1)
    _arrow(ax, rnn_cx, fb_y + 0.05, rnn_cx, rnn_cy - rnn_r - 0.06,
           color=C_FEEDBACK, lw=1.2)

    # Dashed up to Decoder
    ax.plot([dec_cx, dec_cx], [fb_y, dec_by - 0.06],
            color=C_FEEDBACK, lw=1.0, ls="--", zorder=1)
    _arrow(ax, dec_cx, fb_y + 0.05, dec_cx, dec_by - 0.06,
           color=C_FEEDBACK, lw=1.2)

    # =====================================================================
    # Reward arrow: upward from Physics Sim with rodent reward image
    # =====================================================================
    # --- Rodent reward image directly above Physics Sim ---
    reward_img_path = SCRIPT_DIR / "data" / "rodent_reward.png"
    if reward_img_path.exists():
        reward_img = mpimg.imread(str(reward_img_path))
        rh, rw = reward_img.shape[:2]
        aspect = rw / rh
        display_h = 2.00
        display_w = display_h * aspect
        img_bottom = main_y + phys_h / 2 - 0.20
        extent = [
            phys_cx - display_w / 2, phys_cx + display_w / 2,
            img_bottom, img_bottom + display_h,
        ]
        ax.imshow(reward_img, extent=extent, aspect="auto", zorder=3,
                  clip_on=True)

    # =====================================================================
    # Top row: Encoder (frozen) directly above Distill Head
    # =====================================================================
    enc_x = dh_x
    enc_w = dh_w
    enc_h = dh_h
    enc_by = enc_cy - enc_h / 2
    _trapezoid(ax, enc_x, enc_by, enc_w, enc_h, C_BLUE, "Encoder",
               direction="right", fontsize=9, border_color=C_BLUE_DARK,
               sublabel=r"$q_\phi(z \mid s_t^g)$", sublabel_size=7.5)
    ax.text(enc_x + 0.12, enc_by + enc_h * 0.22, "\u2744",
            ha="center", va="center", fontsize=11, color="white",
            alpha=0.9, zorder=5)

    # Arrow from s^g_t to Encoder (drawn above with rodent arrows)

    # --- [mu^e, sigma^e] bracket (no arrow, height matches head) ---
    enc_bkt_cx = enc_x + enc_w + 0.22
    _bracket_labeled(ax, enc_bkt_cx, enc_cy, r"$\mu_t^e$", r"$\sigma_t^e$",
                     w=0.35, h=enc_h)

    # =====================================================================
    # KL: arrows FROM KL TO both brackets
    # =====================================================================
    kl_y = (enc_cy - enc_h / 2 + main_y + dh_h / 2) / 2
    # KL label
    ax.text(enc_bkt_cx, kl_y, r"$\mathrm{KL}$",
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=C_KL, zorder=5)
    # Arrow FROM KL UP to encoder bracket
    _arrow(ax, enc_bkt_cx, kl_y + 0.14, enc_bkt_cx, enc_cy - enc_h / 2 - 0.02,
           color=C_KL, lw=1.3)
    # Arrow FROM KL DOWN to distill bracket
    _arrow(ax, enc_bkt_cx, kl_y - 0.14, enc_bkt_cx, main_y + dh_h / 2 + 0.02,
           color=C_KL, lw=1.3)

    # =====================================================================
    # Dashed box tight around Encoder + Distill Head + KL + brackets
    # =====================================================================
    box_pad = 0.08
    box_left = dh_x - 0.04  # tight to trapezoid base
    box_bottom = main_y - dh_h / 2 - box_pad
    box_right = enc_bkt_cx + 0.22  # right of bracket
    box_top = enc_cy + enc_h / 2 + box_pad
    ax.add_patch(
        FancyBboxPatch(
            (box_left, box_bottom),
            box_right - box_left,
            box_top - box_bottom,
            boxstyle="round,pad=0.05",
            facecolor="none",
            edgecolor="#999999",
            linewidth=1.2,
            linestyle="--",
            zorder=1,
        )
    )
    ax.text(
        (box_left + box_right) / 2, box_bottom - 0.08,
        "optional",
        ha="center", va="top",
        fontsize=8, fontstyle="italic", color="#888888", zorder=5,
    )

    # =====================================================================
    # "MIMIC-MJX Pre-trained" above Inverse Dynamics, arrows to both
    # =====================================================================
    pretrained_x = dec_cx
    pretrained_y = enc_cy + 0.10
    ax.text(pretrained_x, pretrained_y, "MIMIC-MJX\nPre-trained",
            ha="center", va="center", fontsize=7.5, fontweight="bold",
            color=C_STAC, fontstyle="italic", zorder=5)

    # Arrow from MIMIC-MJX: up above dashed box, then left, then down toward Encoder
    enc_mid_x = enc_x + enc_w / 2
    turn_y = box_top + 0.18  # above the dashed box
    ax.plot([pretrained_x, pretrained_x], [pretrained_y + 0.12, turn_y],
            color=C_STAC, lw=1.0, zorder=4)
    ax.plot([pretrained_x, enc_mid_x], [turn_y, turn_y],
            color=C_STAC, lw=1.0, zorder=4)
    ax.plot([enc_mid_x, enc_mid_x], [turn_y, box_top + 0.10],
            color=C_STAC, lw=1.0, zorder=4)
    _arrow(ax, enc_mid_x, box_top + 0.10,
           enc_mid_x, box_top + 0.02,
           color=C_STAC, lw=1.0)

    # Straight-down arrow from MIMIC-MJX to Inverse Dynamics (with gap)
    _arrow(ax, pretrained_x, pretrained_y - 0.22, dec_cx, main_y + dec_h / 2 + 0.12,
           color=C_STAC, lw=1.0)


def make_code2act_figure(output_dir: Path):
    """Generate Figure 2: Code2Act inference pipeline."""
    fig, ax = plt.subplots(1, 1, figsize=(15, 7))

    ax.set_xlim(-0.25, 9.40)
    ax.set_ylim(0.05, 4.80)
    ax.set_aspect("equal")
    ax.axis("off")


    draw_code2act_pipeline(ax)

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in [".pdf", ".png", ".svg"]:
        p = output_dir / f"code2act_pipeline{ext}"
        fc = "none" if ext == ".svg" else "white"
        fig.savefig(
            p, dpi=400, bbox_inches="tight", facecolor=fc, edgecolor="none",
            transparent=(ext == ".svg"),
        )
        print(f"Saved: {p}")
    plt.close(fig)


# =============================================================================
# Entry point
def make_combined_figure(output_dir: Path):
    """Generate combined (A) Mimic-MJX + (B) Code2Act stacked figure."""
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(14, 9),
                                      gridspec_kw={"height_ratios": [1, 1.5]})

    # --- Panel (A): Mimic-MJX ---
    ax_a.set_xlim(-0.05, 8.30)
    ax_a.set_ylim(0.00, 2.15)
    ax_a.set_aspect("equal")
    ax_a.axis("off")
    ax_a.add_patch(FancyBboxPatch(
        (0.05, 0.08), 8.10, 1.95, boxstyle="round,pad=0.08",
        facecolor=C_PANEL_BG, edgecolor="#AAAAAA", linewidth=1.5,
        linestyle="--", zorder=0))
    draw_mimic_training(ax_a)
    ax_a.text(-0.02, 2.05, "(A) Mimic-MJX", fontsize=12, fontweight="bold",
              color=C_TEXT, va="top", ha="left", zorder=10)

    # --- Panel (B): Code2Act ---
    ax_b.set_xlim(-0.25, 9.00)
    ax_b.set_ylim(0.05, 3.10)
    ax_b.set_aspect("equal")
    ax_b.axis("off")
    ax_b.add_patch(FancyBboxPatch(
        (-0.15, 0.12), 8.95, 2.88, boxstyle="round,pad=0.08",
        facecolor=C_PANEL_BG, edgecolor="#AAAAAA", linewidth=1.5,
        linestyle="--", zorder=0))
    draw_code2act_pipeline(ax_b)
    ax_b.text(-0.22, 3.00, "(B) Code2Act", fontsize=12, fontweight="bold",
              color=C_TEXT, va="top", ha="left", zorder=10)

    fig.tight_layout(h_pad=0.5)

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in [".pdf", ".png"]:
        p = output_dir / f"combined_pipeline{ext}"
        fig.savefig(p, dpi=400, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"Saved: {p}")
    plt.close(fig)


# =============================================================================
# Figure 3: High-level Code2Act overview
# =============================================================================


def _draw_highlevel_pipeline(ax, y: float = 1.0, bh: float = 0.70,
                             draw_legend: bool = True):
    """Draw the high-level Code2Act pipeline on *ax*.

    Args:
        ax: Matplotlib axes.
        y: Vertical centre of the main row.
        bh: Block height.
        draw_legend: If True, draw the legend above the diagram.
    """
    # =====================================================================
    # Block 1: Code timeline strip (colored rectangles)
    # =====================================================================
    CODE_COLORS = {
        "c₆": "#E07070",  # red
        "c₄": "#7BC47F",  # green
        "c₁": "#7BA3CC",  # blue
        "c₃": "#E8A050",  # orange
    }
    codes = [
        ("c₆", "c₆", True),   # (label, key, highlighted)
        ("c₄", "c₄", False),
        ("c₁", "c₁", False),
        ("c₃", "c₃", False),
        ("c₁", "c₁", False),
    ]

    strip_x = 0.0
    seg_w = 0.50
    seg_h = 0.50
    seg_y = y - seg_h / 2
    total_strip_w = len(codes) * seg_w

    for i, (label, key, highlighted) in enumerate(codes):
        sx = strip_x + i * seg_w
        color = CODE_COLORS[key]
        alpha = 1.0 if highlighted else 0.45
        lw = 2.5 if highlighted else 0.8
        ec = "#333333" if highlighted else color

        ax.add_patch(FancyBboxPatch(
            (sx + 0.02, seg_y), seg_w - 0.04, seg_h,
            boxstyle="round,pad=0.02",
            facecolor=color, alpha=alpha, edgecolor=ec,
            linewidth=lw, zorder=3,
        ))
        ax.text(sx + seg_w / 2, y, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=4)

    # Label above code strip
    ax.text(strip_x + total_strip_w / 2, y + seg_h / 2 + 0.08,
            "Keypoint-MoSeq code", ha="center", va="bottom",
            fontsize=7.5, color="#555555", fontstyle="italic", zorder=5)

    # Legend above the code strip (centered over full diagram)
    if draw_legend:
        legend_y = y + seg_h / 2 + 0.55
        legend_items = [
            (r"$c_t$ = code", "#C04040"),
            (r"$a_t$ = action", C_TEXT),
            (r"$s_t^p$ = state", C_TEXT),
        ]
        total_legend_w = 5.0
        spacing = total_legend_w / len(legend_items)
        center_x = 4.5  # center of canvas
        start_x = center_x - total_legend_w / 2 + spacing / 2
        for i, (label, color) in enumerate(legend_items):
            ax.text(start_x + i * spacing, legend_y, label,
                    ha="center", va="center", fontsize=8, color=color, zorder=5)

    # Highlight arrow pointing to current code
    ax.annotate("", xy=(strip_x + seg_w / 2, seg_y - 0.02),
                xytext=(strip_x + seg_w / 2, seg_y - 0.22),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.5),
                zorder=5)
    ax.text(strip_x + seg_w / 2, seg_y - 0.28, "current",
            ha="center", va="top", fontsize=6.5, color="#555555",
            fontstyle="italic")

    # --- Arrow strip -> decoder ---
    gap = 0.55
    arr_start = strip_x + total_strip_w + 0.08
    arr_end = strip_x + total_strip_w + gap - 0.08
    _arrow(ax, arr_start, y, arr_end, y, lw=2.0)
    ax.text((arr_start + arr_end) / 2, y + 0.16, r"$c_t$",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
            color="#C04040", zorder=5)

    # =====================================================================
    # Block 2: Recurrent Code Decoder (with small RNN icon)
    # =====================================================================
    b2_w = 2.0
    b2_x = strip_x + total_strip_w + gap
    _block(ax, b2_x, y - bh / 2, b2_w, bh, "#E8E8E8",
           "Recurrent Code\nDecoder", fontsize=10,
           border_color="#AAAAAA", text_color="#333333")

    # --- Arrow decoder -> physics ---
    _arrow(ax, b2_x + b2_w + 0.08, y, b2_x + b2_w + gap - 0.08, y, lw=2.0)
    ax.text(b2_x + b2_w + gap / 2, y + 0.16, r"$a_t$",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
            color=C_TEXT, zorder=5)

    # =====================================================================
    # Block 3: Physics Simulation
    # =====================================================================
    b3_w = 1.8
    b3_x = b2_x + b2_w + gap
    _block(ax, b3_x, y - bh / 2, b3_w, bh, C_PHYSICS,
           "Physics\nSimulation", fontsize=10, border_color="#666666",
           text_color="white")
    b3_cx = b3_x + b3_w / 2

    # =====================================================================
    # Feedback: Physics -> Decoder
    # =====================================================================
    fb_y = y - bh / 2 - 0.28
    b2_cx = b2_x + b2_w / 2
    ax.plot([b3_cx, b3_cx], [y - bh / 2, fb_y],
            color=C_FEEDBACK, lw=1.2, ls="--", zorder=1)
    ax.plot([b3_cx, b2_cx], [fb_y, fb_y],
            color=C_FEEDBACK, lw=1.2, ls="--", zorder=1)
    _arrow(ax, b2_cx, fb_y, b2_cx, y - bh / 2 - 0.02,
           color=C_FEEDBACK, lw=1.2)
    ax.text((b2_cx + b3_cx) / 2, fb_y, r"$s_t^p$", ha="center", va="center",
            fontsize=10, color=C_TEXT, fontweight="bold", zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))


def _draw_highlevel_legend(ax):
    """Draw only the legend for the high-level Code2Act pipeline on *ax*."""
    legend_items = [
        (r"$c_t$ = code", "#C04040"),
        (r"$a_t$ = action", C_TEXT),
        (r"$s_t^p$ = state", C_TEXT),
    ]
    total_w = 5.0
    spacing = total_w / len(legend_items)
    center_x = total_w / 2
    start_x = center_x - total_w / 2 + spacing / 2
    for i, (label, color) in enumerate(legend_items):
        ax.text(start_x + i * spacing, 0.5, label,
                ha="center", va="center", fontsize=8, color=color, zorder=5)


def make_highlevel_figure(output_dir: Path):
    """Generate high-level Code2Act overview with code timeline strip.

    Outputs:
      - code2act_highlevel.{pdf,png,svg} — full figure with legend
      - code2act_highlevel_nolegend.svg   — pipeline only (no legend)
      - code2act_highlevel_legend.svg     — legend only

    Layout:
      [c₆ c₄ c₁ c₃ c₁ colored bar]  →  [Recurrent Action Decoder]  →  [Physics Sim]
       current code highlighted           ← ─ ─  s_t^p  ─ ─ ─ ─ ─ ┘
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Full figure (pipeline + legend) ---
    fig, ax = plt.subplots(1, 1, figsize=(11, 3.2))
    ax.set_xlim(-0.2, 9.2)
    ax.set_ylim(-0.3, 2.5)
    ax.set_aspect("equal")
    ax.axis("off")
    _draw_highlevel_pipeline(ax, draw_legend=True)

    for ext in [".pdf", ".png", ".svg"]:
        p = output_dir / f"code2act_highlevel{ext}"
        fc = "none" if ext == ".svg" else "white"
        fig.savefig(p, dpi=400, bbox_inches="tight",
                    facecolor=fc, edgecolor="none", transparent=(ext == ".svg"))
        print(f"Saved: {p}")
    plt.close(fig)

    # --- Pipeline only (no legend) ---
    fig_nl, ax_nl = plt.subplots(1, 1, figsize=(11, 2.8))
    ax_nl.set_xlim(-0.2, 9.2)
    ax_nl.set_ylim(-0.3, 2.0)
    ax_nl.set_aspect("equal")
    ax_nl.axis("off")
    _draw_highlevel_pipeline(ax_nl, draw_legend=False)

    p_nl = output_dir / "code2act_highlevel_nolegend.svg"
    fig_nl.savefig(p_nl, dpi=400, bbox_inches="tight",
                   facecolor="none", edgecolor="none", transparent=True)
    print(f"Saved: {p_nl}")
    plt.close(fig_nl)

    # --- Legend only ---
    fig_lg, ax_lg = plt.subplots(1, 1, figsize=(6, 0.6))
    ax_lg.set_xlim(0, 5.0)
    ax_lg.set_ylim(0, 1.0)
    ax_lg.set_aspect("auto")
    ax_lg.axis("off")
    _draw_highlevel_legend(ax_lg)

    p_lg = output_dir / "code2act_highlevel_legend.svg"
    fig_lg.savefig(p_lg, dpi=400, bbox_inches="tight",
                   facecolor="none", edgecolor="none", transparent=True)
    print(f"Saved: {p_lg}")
    plt.close(fig_lg)


# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize Mimic-MJX and Code2Act pipeline architectures",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "outputs"),
        help="Output directory for figures (default: outputs/)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    make_mimic_figure(output_dir)
    make_code2act_figure(output_dir)
    make_highlevel_figure(output_dir)
