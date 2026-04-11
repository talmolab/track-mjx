"""Render hidden dynamics triptych with synchronized rotating PCA.

4-panel video: [Rotating 3D PCA] | [Walk] | [Immobility] | [Rear]

The PCA panel shows hidden state trajectories progressing in sync with
the MuJoCo rollouts. All panels advance frame-by-frame together.

Uses qpos and hidden states from hidden_dynamics.npz (produced by
run_hidden_dynamics).

Usage:
    cd moseq_jax/figures
    python render_hidden_dynamics_triptych.py
"""

import logging
import os
import sys
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import mujoco
import numpy as np
from sklearn.decomposition import PCA

SCRIPT_DIR = Path(__file__).parent
MOSEQ_DIR = SCRIPT_DIR.parent
REPO_ROOT = MOSEQ_DIR.parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "hidden_dynamics"

for _p in (str(MOSEQ_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.shared.ghost_rendering import build_ghost_model
from vqvae_jax.analysis.rendering import add_multi_line_overlay

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
BEHAVIORS = ["walk", "immobility", "rear"]
BEHAVIOR_LABELS = {"walk": "Walk", "immobility": "Immobility", "rear": "Rear"}
BEHAVIOR_COLORS = {
    "walk": "#D55E00",
    "immobility": "#0072B2",
    "rear": "#009E73",
}


def _behavior_cmap(hex_color):
    rgb = mcolors.to_rgb(hex_color)
    light = tuple(0.80 + 0.20 * c for c in rgb)
    return mcolors.LinearSegmentedColormap.from_list("beh", [light, rgb])


BEHAVIOR_CMAPS = {b: _behavior_cmap(c) for b, c in BEHAVIOR_COLORS.items()}


def get_ghost_colors(k):
    cmap = plt.colormaps["tab10"]
    return [list(cmap(i % 10)) for i in range(k)]


def _center_qpos(trajectories_qpos):
    min_len = min(len(q) for q in trajectories_qpos)
    stacked_xy = np.stack([q[:min_len, :2] for q in trajectories_qpos], axis=0)
    mean_xy = stacked_xy.mean(axis=0)
    centered = []
    for q in trajectories_qpos:
        qc = q[:min_len].copy()
        qc[:, 0] -= mean_xy[:, 0]
        qc[:, 1] -= mean_xy[:, 1]
        centered.append(qc)
    return centered


# ── PCA frame rendering ─────────────────────────────────────────────────────


def render_pca_frame(
    ax,
    emb_3d: np.ndarray,
    beh_slices: dict,
    beh_shapes: dict,
    var_explained: np.ndarray,
    t: int,
    azim: float,
    xlim: tuple,
    ylim: tuple,
    zlim: tuple,
) -> None:
    """Draw PCA trajectories up to frame t on the given axes."""
    ax.cla()

    for beh in BEHAVIORS:
        if beh not in beh_slices:
            continue
        K_beh, T_beh = beh_shapes[beh]
        s, _ = beh_slices[beh]
        cmap = BEHAVIOR_CMAPS[beh]
        color = BEHAVIOR_COLORS[beh]

        t_clip = min(t + 1, T_beh)
        if t_clip < 2:
            continue

        t_norm = np.linspace(0, 1, t_clip)

        for ki in range(K_beh):
            start = s + ki * T_beh
            pts = emb_3d[start : start + t_clip]

            # Line segments
            segments = [[pts[i], pts[i + 1]] for i in range(len(pts) - 1)]
            if segments:
                colors = cmap(t_norm[:-1])
                lc = Line3DCollection(
                    segments, colors=colors, linewidths=0.8, alpha=0.6,
                )
                ax.add_collection3d(lc)

            # Current position marker
            ax.scatter(
                [pts[-1, 0]], [pts[-1, 1]], [pts[-1, 2]],
                c=color, s=25, edgecolors="white",
                linewidths=0.5, zorder=10, depthshade=False,
            )

    ax.set_xlabel(f"PC1 ({var_explained[0]:.0%})", fontsize=6, labelpad=0)
    ax.set_ylabel(f"PC2 ({var_explained[1]:.0%})", fontsize=6, labelpad=0)
    ax.set_zlabel(f"PC3 ({var_explained[2]:.0%})", fontsize=6, labelpad=0)
    ax.tick_params(axis="both", which="major", labelsize=4, pad=0)

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

    # Legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=BEHAVIOR_COLORS[b], markersize=4,
               markeredgewidth=0, label=BEHAVIOR_LABELS[b], linestyle="None")
        for b in BEHAVIORS if b in beh_slices
    ]
    ax.legend(
        handles=legend_handles, loc="upper left",
        frameon=True, framealpha=0.9, edgecolor="none",
        fontsize=5.5, handletextpad=0.2, borderpad=0.3,
    )

    ax.view_init(elev=20, azim=azim)


def pca_frame_to_image(
    fig, ax, emb_3d, beh_slices, beh_shapes, var_explained,
    t, azim, xlim, ylim, zlim, width, height,
) -> np.ndarray:
    """Render one PCA frame to an RGB numpy array."""
    render_pca_frame(
        ax, emb_3d, beh_slices, beh_shapes, var_explained,
        t, azim, xlim, ylim, zlim,
    )
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()
    # Resize to target dimensions
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return img


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    data = np.load(DATA_DIR / "hidden_dynamics.npz", allow_pickle=True)

    # Load env for MuJoCo rendering
    import json
    from omegaconf import OmegaConf
    from track_mjx.config import utils
    from track_mjx.agent import checkpointing
    from vnl_playground.tasks.rodent.imitation import ReferenceClips
    from moseq_env_wrapper import MoSeqImitation

    ckpt_path = str(MOSEQ_DIR / "model_checkpoints" / "260407_031233_484020")
    cfg = checkpointing.load_config_from_checkpoint(
        ckpt_path, step_prefix="MoSeqPPONetwork",
    )
    cfg = OmegaConf.create(cfg)
    _, _, env_cfg = utils.prepare_config(cfg)
    env_cfg.start_frame_range = [0, 0]
    env_cfg.domain_randomization.use_domain_randomization = False

    splits_path = REPO_ROOT / "data" / "rodent" / "rodent_balanced_splits.json"
    with open(splits_path) as f:
        splits = json.load(f)
    test_indices = splits["balanced"]["test_indices"]
    codes_data = np.load(
        str(MOSEQ_DIR / "outputs" / "kpms_sweep" / "best_codes.npz"),
    )
    test_codes = codes_data["test_codes"]

    test_clips = ReferenceClips(
        data_path=str(REPO_ROOT / "data" / "rodent" / "rodent_reference_clips.h5"),
        n_frames_per_clip=int(cfg.env_config.clip_length),
        keep_clips_idx=np.array(test_indices),
    )
    code_stack_size = int(cfg.network_config.get("code_stack_size", 1))
    env = MoSeqImitation(
        config=env_cfg, clips=test_clips, kpms_codes=test_codes,
        code_stack_size=code_stack_size,
    )

    # ── Prepare PCA ──────────────────────────────────────────────────────
    all_points = []
    beh_slices = {}
    beh_shapes = {}
    beh_qpos = {}
    offset = 0

    for beh in BEHAVIORS:
        h = np.array(data[f"hidden_{beh}"])  # [K, T, 256]
        q = np.array(data[f"qpos_{beh}"])    # [K, T, 74]
        K, T = h.shape[0], h.shape[1]
        beh_shapes[beh] = (K, T)
        beh_qpos[beh] = [q[ki] for ki in range(K)]
        flat = h.reshape(-1, h.shape[-1])
        all_points.append(flat)
        beh_slices[beh] = (offset, offset + flat.shape[0])
        offset += flat.shape[0]

    all_points = np.concatenate(all_points, axis=0)
    reducer = PCA(n_components=3, random_state=42)
    emb_3d = reducer.fit_transform(all_points)
    var = reducer.explained_variance_ratio_
    log.info(f"PCA: {var[0]:.1%}, {var[1]:.1%}, {var[2]:.1%}")

    # Shared limits
    pad = 0.08
    xlim = (emb_3d[:, 0].min() * (1 + pad), emb_3d[:, 0].max() * (1 + pad))
    ylim = (emb_3d[:, 1].min() * (1 + pad), emb_3d[:, 1].max() * (1 + pad))
    zlim = (emb_3d[:, 2].min() * (1 + pad), emb_3d[:, 2].max() * (1 + pad))

    # ── Build ghost models per behavior ──────────────────────────────────
    panel_w, panel_h = 400, 400
    pca_w, pca_h = 480, 400
    n_frames = min(beh_shapes[b][1] for b in BEHAVIORS)
    K = beh_shapes[BEHAVIORS[0]][0]
    traj_colors = get_ghost_colors(K)

    ghost_models = {}
    for beh in BEHAVIORS:
        gm, base_nq = build_ghost_model(
            env,
            num_ghosts=K - 1,
            ghost_colors=traj_colors[1:],
            camera_distance=1.0,
            camera_elevation=-25.0,
            camera_azimuth=135.0,
            camera_fovy=50.0,
        )
        gm.vis.global_.offwidth = panel_w
        gm.vis.global_.offheight = panel_h
        ghost_models[beh] = (gm, base_nq, mujoco.MjData(gm), mujoco.Renderer(gm, height=panel_h, width=panel_w))

    # Center qpos per behavior
    centered_qpos = {}
    for beh in BEHAVIORS:
        centered_qpos[beh] = _center_qpos(beh_qpos[beh])

    # ── Matplotlib figure for PCA (reused each frame) ────────────────────
    fig = plt.figure(figsize=(pca_w / 100, pca_h / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.92)

    # ── Render combined video ────────────────────────────────────────────
    label_h = 32
    total_w = pca_w + panel_w * 3
    total_h = panel_h + label_h
    total_w = ((total_w + 15) // 16) * 16
    total_h = ((total_h + 15) // 16) * 16

    output_path = OUTPUT_DIR / "hidden_dynamics_triptych.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, 50, (total_w, total_h))

    azim_start, azim_end = 140, 260

    log.info(f"Rendering {n_frames} frames ({total_w}x{total_h})...")

    for t in range(n_frames):
        combined = np.ones((total_h, total_w, 3), dtype=np.uint8) * 40

        # PCA panel (rotating)
        azim = azim_start + (azim_end - azim_start) * t / max(n_frames - 1, 1)
        pca_img = pca_frame_to_image(
            fig, ax, emb_3d, beh_slices, beh_shapes, var,
            t, azim, xlim, ylim, zlim, pca_w, panel_h,
        )
        combined[label_h : label_h + panel_h, :pca_w] = pca_img

        # MuJoCo panels
        for bi, beh in enumerate(BEHAVIORS):
            gm, base_nq, md, renderer = ghost_models[beh]
            centered = centered_qpos[beh]
            min_len = min(len(q) for q in centered)
            t_clip = min(t, min_len - 1)

            md.qpos[:base_nq] = centered[0][t_clip]
            for gi in range(1, K):
                qs = base_nq + (gi - 1) * base_nq
                qe = qs + base_nq
                md.qpos[qs:qe] = centered[gi][t_clip]

            mujoco.mj_forward(gm, md)
            renderer.update_scene(md, camera="divergent_cam")
            frame = renderer.render().copy()

            x_off = pca_w + bi * panel_w
            combined[label_h : label_h + panel_h, x_off : x_off + panel_w] = frame

        # Labels
        combined = add_multi_line_overlay(
            combined, ["Hidden State"],
            start_position=(pca_w // 2 - 40, 6), font_size=14,
        )
        for bi, beh in enumerate(BEHAVIORS):
            x_center = pca_w + bi * panel_w + panel_w // 2 - len(BEHAVIOR_LABELS[beh]) * 4
            combined = add_multi_line_overlay(
                combined, [BEHAVIOR_LABELS[beh]],
                start_position=(max(x_center, pca_w + bi * panel_w + 5), 6),
                font_size=14,
            )

        # Timestep
        combined = add_multi_line_overlay(
            combined, [f"t={t}"],
            start_position=(total_w - 55, total_h - 18), font_size=11,
        )

        writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

        if (t + 1) % 50 == 0:
            log.info(f"  {t + 1}/{n_frames}")

    writer.release()
    plt.close(fig)

    # Close renderers
    for beh in BEHAVIORS:
        ghost_models[beh][3].close()

    log.info(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
