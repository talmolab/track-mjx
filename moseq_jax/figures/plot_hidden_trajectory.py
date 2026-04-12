"""Visualize natural hidden state trajectories colored by KPMS code.

Loads hidden_trajectory.npz (from run_hidden_trajectory) and produces:
  - 3D PCA scatter colored by code identity (top N codes colored, rest grey)
  - Synchronized video: rotating PCA trajectory + MuJoCo rollout

Usage:
    cd moseq_jax/figures
    python plot_hidden_trajectory.py                  # static plots only
    python plot_hidden_trajectory.py --video          # also render video
    python plot_hidden_trajectory.py --top-codes 8    # show top 8 codes
    python plot_hidden_trajectory.py --traj-idx 0     # which trajectory for video
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from sklearn.decomposition import PCA

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "hidden_trajectory"

# ── Code color palette ───────────────────────────────────────────────────────
# Top N codes get distinct tab10 colors; rest are grey
GREY = "#bbbbbb"


def build_code_colormap(
    codes_flat: np.ndarray, top_n: int = 5,
) -> tuple[dict[int, str], list[int]]:
    """Assign colors to the top N most frequent codes.

    Returns:
        code_colors: ``{code_id: hex_color}`` (unlisted codes → grey)
        top_codes: sorted list of top N code IDs
    """
    unique, counts = np.unique(codes_flat, return_counts=True)
    order = np.argsort(-counts)
    top_codes = [int(unique[o]) for o in order[:top_n]]

    cmap = plt.colormaps["tab10"]
    code_colors = {}
    for i, c in enumerate(top_codes):
        code_colors[c] = mcolors.to_hex(cmap(i % 10))

    return code_colors, top_codes


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


def _style_3d_ax(ax, var_explained, xlim, ylim, zlim):
    ax.set_xlabel(f"PC 1 ({var_explained[0]:.1%})", labelpad=0, fontsize=6)
    ax.set_ylabel(f"PC 2 ({var_explained[1]:.1%})", labelpad=0, fontsize=6)
    ax.set_zlabel(f"PC 3 ({var_explained[2]:.1%})", labelpad=0, fontsize=6)
    ax.tick_params(axis="both", which="major", pad=0, labelsize=4.5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#d5d5d5")
    ax.grid(True, linewidth=0.2, alpha=0.4)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_linewidth(0.4)
        axis.line.set_color("#666666")


def _add_rounded_border(fig):
    fig.patch.set_facecolor("white")
    rect = mpatches.FancyBboxPatch(
        (0.005, 0.005), 0.99, 0.99,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        transform=fig.transFigure, facecolor="white",
        edgecolor="#cccccc", linewidth=0.6, zorder=-1,
    )
    fig.patches.append(rect)
    return rect


def _compute_limits(emb, pad=0.08):
    lims = []
    for i in range(3):
        lo, hi = emb[:, i].min(), emb[:, i].max()
        d = (hi - lo) * pad
        lims.append((lo - d, hi + d))
    return tuple(lims)


# ── Data loading ─────────────────────────────────────────────────────────────


def load_data():
    raw = np.load(DATA_DIR / "hidden_trajectory.npz", allow_pickle=True)
    return {
        "hidden": raw["hidden"],     # [K, T, 256]
        "codes": raw["codes"],       # [K, T]
        "qpos": raw["qpos"],         # [K, T, 74]
        "survivals": raw["survivals"],
    }


# ── Static PCA plot ──────────────────────────────────────────────────────────


def plot_pca_by_code(
    data: dict,
    top_n: int = 5,
    traj_indices: list[int] | None = None,
) -> tuple[plt.Figure, mpatches.FancyBboxPatch]:
    """3D PCA scatter of hidden states, colored by active code.

    Args:
        data: loaded data dict
        top_n: number of codes to highlight
        traj_indices: which trajectories to plot (None = all)
    """
    hidden = data["hidden"]  # [K, T, 256]
    codes = data["codes"]    # [K, T]
    K, T = hidden.shape[0], hidden.shape[1]

    if traj_indices is None:
        traj_indices = list(range(K))

    # Fit PCA on all data
    all_h = hidden.reshape(-1, hidden.shape[-1])
    reducer = PCA(n_components=3, random_state=42)
    emb_all = reducer.fit_transform(all_h)
    var = reducer.explained_variance_ratio_
    print(f"  PCA: {var[0]:.1%}, {var[1]:.1%}, {var[2]:.1%} (total={sum(var):.1%})")

    emb = emb_all.reshape(K, T, 3)
    xlim, ylim, zlim = _compute_limits(emb_all)

    # Code colors
    codes_flat = codes[traj_indices].ravel()
    code_colors, top_codes = build_code_colormap(codes_flat, top_n=top_n)

    fig = plt.figure(figsize=(5.0, 4.5))
    ax = fig.add_subplot(111, projection="3d")

    # Plot grey (non-top) points first
    for ki in traj_indices:
        for t in range(T):
            c = int(codes[ki, t])
            if c not in code_colors:
                ax.scatter(
                    [emb[ki, t, 0]], [emb[ki, t, 1]], [emb[ki, t, 2]],
                    c=GREY, s=2, alpha=0.08, edgecolors="none",
                    rasterized=True, depthshade=True,
                )

    # Plot top codes on top (with larger points)
    for code_id in top_codes:
        color = code_colors[code_id]
        for ki in traj_indices:
            mask = codes[ki] == code_id
            if not mask.any():
                continue
            pts = emb[ki][mask]
            ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                c=color, s=4, alpha=0.25, edgecolors="none",
                rasterized=True, depthshade=True,
            )

    _style_3d_ax(ax, var, xlim, ylim, zlim)

    # Legend
    legend_handles = [
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=code_colors[c], markersize=5,
            markeredgewidth=0, label=f"Code {c}",
            linestyle="None",
        )
        for c in top_codes
    ]
    legend_handles.append(
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=GREY, markersize=4,
            markeredgewidth=0, label="Other codes",
            linestyle="None",
        )
    )
    leg = ax.legend(
        handles=legend_handles, loc="upper left",
        frameon=True, framealpha=0.92, edgecolor="none",
        borderpad=0.4, handletextpad=0.2, fancybox=True,
        fontsize=5.5, ncol=1,
    )
    leg.get_frame().set_linewidth(0)
    leg.get_frame().set_boxstyle("round,pad=0.3,rounding_size=0.2")

    ax.set_title(
        "Hidden State Trajectory (Natural Behavior)",
        fontsize=8, fontweight="bold", pad=-2,
    )
    ax.view_init(elev=20, azim=140)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95)
    rect = _add_rounded_border(fig)

    return fig, rect, reducer, var


# ── Synchronized video ───────────────────────────────────────────────────────


def render_trajectory_video(
    data: dict,
    reducer: PCA,
    var: np.ndarray,
    traj_idx: int = 0,
    top_n: int = 5,
    fps: int = 50,
) -> str:
    """Render 2-panel video: [Rotating PCA] | [MuJoCo rollout].

    PCA panel shows the trajectory growing frame by frame, colored by code.
    """
    import os
    import sys
    os.environ["MUJOCO_GL"] = "egl"
    import mujoco

    MOSEQ_DIR_local = SCRIPT_DIR.parent
    REPO_ROOT_local = MOSEQ_DIR_local.parent
    for _p in (str(MOSEQ_DIR_local), str(REPO_ROOT_local)):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    from vqvae_jax.analysis.rendering import add_multi_line_overlay
    from track_mjx.agent import checkpointing
    from omegaconf import OmegaConf
    from track_mjx.config import utils as cfg_utils
    from vnl_playground.tasks.rodent.imitation import ReferenceClips
    from moseq_env_wrapper import MoSeqImitation

    hidden = data["hidden"]  # [K, T, 256]
    codes = data["codes"]    # [K, T]
    qpos = data["qpos"]      # [K, T, 74]
    K, T = hidden.shape[0], hidden.shape[1]

    # PCA transform
    emb = reducer.transform(hidden.reshape(-1, 256)).reshape(K, T, 3)
    emb_all = emb.reshape(-1, 3)
    xlim, ylim, zlim = _compute_limits(emb_all)

    # Code colors (from all trajectories)
    code_colors, top_codes = build_code_colormap(codes.ravel(), top_n=top_n)

    from experiments.shared.ghost_rendering import build_ghost_model

    # Set up MuJoCo ghost renderer (all K bodies)
    ckpt_path = str(MOSEQ_DIR_local / "model_checkpoints" / "260407_031233_484020")
    cfg = checkpointing.load_config_from_checkpoint(
        ckpt_path, step_prefix="MoSeqPPONetwork",
    )
    cfg = OmegaConf.create(cfg)

    _, _, env_cfg = cfg_utils.prepare_config(cfg)
    env_cfg.start_frame_range = [0, 0]
    env_cfg.domain_randomization.use_domain_randomization = False

    ref_clips = ReferenceClips(
        data_path=str(REPO_ROOT_local / "data" / "rodent" / "rodent_reference_clips.h5"),
        n_frames_per_clip=int(cfg.env_config.clip_length),
    )
    dummy_codes = np.zeros((ref_clips.qpos.shape[0], int(cfg.env_config.clip_length)), dtype=np.int32)
    env = MoSeqImitation(config=env_cfg, clips=ref_clips, kpms_codes=dummy_codes, code_stack_size=1)

    # Ghost model with K bodies, zoomed in
    panel_w, panel_h = 640, 336
    traj_colors = [list(plt.colormaps["tab10"](i % 10)) for i in range(K)]
    ghost_model, base_nq = build_ghost_model(
        env,
        num_ghosts=K - 1,
        ghost_colors=traj_colors[1:],
        camera_distance=0.7,
        camera_elevation=-25.0,
        camera_azimuth=135.0,
        camera_fovy=50.0,
    )
    ghost_model.vis.global_.offwidth = panel_w
    ghost_model.vis.global_.offheight = panel_h
    mj_data = mujoco.MjData(ghost_model)
    renderer = mujoco.Renderer(ghost_model, height=panel_h, width=panel_w)

    # Center qpos: shift all trajectories so centroid is at origin each frame
    all_qpos = [qpos[ki] for ki in range(K)]
    min_len = min(len(q) for q in all_qpos)
    stacked_xy = np.stack([q[:min_len, :2] for q in all_qpos], axis=0)
    mean_xy = stacked_xy.mean(axis=0)
    centered_qpos = []
    for q in all_qpos:
        qc = q[:min_len].copy()
        qc[:, 0] -= mean_xy[:, 0]
        qc[:, 1] -= mean_xy[:, 1]
        centered_qpos.append(qc)

    # Matplotlib figure for PCA
    fig = plt.figure(figsize=(panel_w / 100, panel_h / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.90)

    # Video writer
    label_h = 28
    total_w = panel_w * 2
    total_h = panel_h + label_h
    total_w = ((total_w + 15) // 16) * 16
    total_h = ((total_h + 15) // 16) * 16

    tmp_path = OUTPUT_DIR / "_tmp_trajectory.mp4"
    output_path = OUTPUT_DIR / "hidden_trajectory_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (total_w, total_h))

    azim_start, azim_end = 140, 260
    n_render = min(T, min_len)

    print(f"  Rendering {n_render} frames ({total_w}x{total_h}), {K} bodies...")

    for t in range(n_render):
        combined = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255

        # ── PCA panel (all K trajectories) ───────────────────────────
        ax.cla()
        azim = azim_start + (azim_end - azim_start) * t / max(n_render - 1, 1)

        # Draw all K trajectories up to t, segment-colored
        for ki in range(K):
            ki_emb = emb[ki]
            ki_codes = codes[ki]
            if t > 0:
                for s in range(min(t, T - 1)):
                    c = int(ki_codes[s])
                    color = code_colors.get(c, GREY)
                    ax.plot(
                        [ki_emb[s, 0], ki_emb[s + 1, 0]],
                        [ki_emb[s, 1], ki_emb[s + 1, 1]],
                        [ki_emb[s, 2], ki_emb[s + 1, 2]],
                        color=color, linewidth=0.4, alpha=0.3,
                    )

            # Current position marker
            c = int(ki_codes[min(t, T - 1)])
            cur_color = code_colors.get(c, GREY)
            ax.scatter(
                [ki_emb[t, 0]], [ki_emb[t, 1]], [ki_emb[t, 2]],
                c=cur_color, s=20, edgecolors="black",
                linewidths=0.4, zorder=10, depthshade=False,
            )

        _style_3d_ax(ax, var, xlim, ylim, zlim)
        ax.view_init(elev=20, azim=azim)

        # Legend (compact)
        legend_handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=code_colors[c_id], markersize=3.5,
                   markeredgewidth=0, label=f"Code {c_id}", linestyle="None")
            for c_id in top_codes
        ]
        ax.legend(
            handles=legend_handles, loc="upper left",
            frameon=True, framealpha=0.9, edgecolor="none",
            fontsize=4.5, handletextpad=0.1, borderpad=0.2,
        )

        fig.canvas.draw()
        pca_img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        pca_img = cv2.resize(pca_img, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        combined[label_h : label_h + panel_h, :panel_w] = pca_img

        # ── MuJoCo panel (all K ghost bodies) ────────────────────────
        t_clip = min(t, min_len - 1)
        mj_data.qpos[:base_nq] = centered_qpos[0][t_clip]
        for gi in range(1, K):
            qs = base_nq + (gi - 1) * base_nq
            qe = qs + base_nq
            mj_data.qpos[qs:qe] = centered_qpos[gi][t_clip]

        mujoco.mj_forward(ghost_model, mj_data)
        renderer.update_scene(mj_data, camera="divergent_cam")
        frame = renderer.render().copy()
        combined[label_h : label_h + panel_h, panel_w : panel_w * 2] = frame

        # Labels
        combined = add_multi_line_overlay(
            combined, ["Hidden State Trajectory"],
            start_position=(panel_w // 2 - 60, 4), font_size=12,
        )
        combined = add_multi_line_overlay(
            combined, [f"K={K} bodies | t={t}"],
            start_position=(panel_w + panel_w // 2 - 35, 4), font_size=12,
        )

        writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

        if (t + 1) % 200 == 0:
            print(f"    {t + 1}/{T}")

    writer.release()
    renderer.close()
    plt.close(fig)

    # Re-encode to H.264
    import imageio_ffmpeg
    import subprocess
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([
        ffmpeg, "-y", "-i", str(tmp_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
        "-movflags", "+faststart", str(output_path),
    ], check=True, capture_output=True)
    tmp_path.unlink()

    print(f"  Saved: {output_path}")
    return str(output_path)


# ── Save helper ──────────────────────────────────────────────────────────────


def _save_figure(fig, rect, stem):
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf")
    fig.savefig(OUTPUT_DIR / f"{stem}.png")
    rect.set_facecolor("none")
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", transparent=True)
    rect.set_facecolor("white")
    plt.close(fig)
    print(f"  Saved: {stem}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", action="store_true", help="Also render video")
    parser.add_argument("--top-codes", type=int, default=5, help="Top N codes to color")
    parser.add_argument("--traj-idx", type=int, default=0, help="Trajectory index for video")
    args = parser.parse_args()

    _setup_nature_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()
    K, T = data["hidden"].shape[0], data["hidden"].shape[1]
    print(f"Loaded: K={K}, T={T}")

    # Static PCA (all trajectories)
    fig, rect, reducer, var = plot_pca_by_code(data, top_n=args.top_codes)
    _save_figure(fig, rect, "hidden_trajectory_pca3d")

    # Static PCA (single trajectory)
    fig, rect, _, _ = plot_pca_by_code(
        data, top_n=args.top_codes, traj_indices=[args.traj_idx],
    )
    _save_figure(fig, rect, f"hidden_trajectory_pca3d_traj{args.traj_idx}")

    # Video
    if args.video:
        render_trajectory_video(
            data, reducer, var,
            traj_idx=args.traj_idx, top_n=args.top_codes,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
