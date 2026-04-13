"""Render triptych ghost comparison: Real | MIMIC | C2A.

Three-panel video with N ghost bodies per panel, all 2000 frames.
Left: reference (real imitation targets), Middle: MIMIC rollouts, Right: C2A rollouts.

Usage:
    cd moseq_jax
    python -m figures.render_generalization_comparison [--n_render 10]
"""

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import h5py
import imageio
import mujoco
import numpy as np

MOSEQ_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = MOSEQ_DIR.parent
for _p in (str(MOSEQ_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vqvae_jax.analysis.rendering import add_multi_line_overlay
from vqvae_jax.ablation.run_divergent_futures import (
    build_ghost_model as _build_ghost_model,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_ROLLOUT_PATH = MOSEQ_DIR / "outputs" / "moseq_generalization_kid" / "data" / "rollouts.npz"
DEFAULT_SEGMENT_PATH = MOSEQ_DIR / "outputs" / "moseq_generalization_kid" / "data" / "segments.h5"
DEFAULT_RESULTS_PATH = MOSEQ_DIR / "outputs" / "moseq_generalization_kid" / "results.json"
DEFAULT_OUTPUT_DIR = MOSEQ_DIR / "figures" / "outputs" / "generalization_comparison"

# Semi-transparent ghost colors (RGBA) — 10 distinct hues
GHOST_COLORS_10 = [
    [0.12, 0.47, 0.71, 0.5],  # blue
    [1.00, 0.50, 0.05, 0.5],  # orange
    [0.17, 0.63, 0.17, 0.5],  # green
    [0.84, 0.15, 0.16, 0.5],  # red
    [0.58, 0.40, 0.74, 0.5],  # purple
    [0.55, 0.34, 0.29, 0.5],  # brown
    [0.89, 0.47, 0.76, 0.5],  # pink
    [0.50, 0.50, 0.50, 0.5],  # gray
    [0.74, 0.74, 0.13, 0.5],  # olive
    [0.09, 0.75, 0.81, 0.5],  # cyan
]


def build_ghost_model_via_env(
    ref_h5_path: str,
    num_bodies: int,
    colors: list[list[float]],
) -> tuple:
    """Build ghost model using the existing build_ghost_model infrastructure.

    Creates a minimal env from reference clips, then delegates to the
    existing ghost model builder from run_divergent_futures.

    Returns (compiled_model, base_nq).
    """
    from track_mjx.config import utils as cfg_utils
    from experiments.shared.checkpoint_utils import load_moseq_checkpoint

    # Load a checkpoint just for the config (to build env)
    ckpt_path = str(MOSEQ_DIR / "model_checkpoints" / "260407_031233_484020")
    ckpt_cfg, _, _, _ = load_moseq_checkpoint(ckpt_path)

    from vnl_playground.tasks.rodent.imitation import ReferenceClips, Imitation

    clips = ReferenceClips(data_path=ref_h5_path, n_frames_per_clip=250)
    _, _, env_cfg = cfg_utils.prepare_config(ckpt_cfg)
    env_cfg.clip_length = 250
    env_cfg.nconmax = 256
    env_cfg.njmax = 128
    env = Imitation(config=env_cfg, clips=clips)

    # num_ghosts is ADDITIONAL bodies beyond the primary (body 0).
    # So for N trajectories, we need N-1 ghosts.
    ghost_model, base_nq = _build_ghost_model(
        env,
        num_ghosts=num_bodies - 1,
        ghost_colors=colors[1:],
        camera_distance=0.7,
        camera_elevation=-25.0,
        camera_azimuth=135.0,
        camera_fovy=60.0,
    )
    return ghost_model, base_nq


def render_ghost_panel(
    ghost_model,
    base_nq: int,
    trajectories: list[np.ndarray],
    camera: str,
    width: int,
    height: int,
) -> list[np.ndarray]:
    """Render K overlaid ghost bodies, return list of frames."""
    K = len(trajectories)
    T = min(len(q) for q in trajectories)

    ghost_model.vis.global_.offwidth = width
    ghost_model.vis.global_.offheight = height
    data = mujoco.MjData(ghost_model)
    renderer = mujoco.Renderer(ghost_model, height=height, width=width)

    frames = []
    for t in range(T):
        # Primary body
        data.qpos[:base_nq] = trajectories[0][t]
        # Ghost bodies
        for gi in range(1, K):
            q_start = base_nq + (gi - 1) * base_nq
            data.qpos[q_start : q_start + base_nq] = trajectories[gi][t]

        mujoco.mj_forward(ghost_model, data)
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render().copy())

    renderer.close()
    return frames


def plot_kid_comparison(results_path: str, output_dir: str) -> None:
    """Create a publication-quality KID barplot matching kid_barplot.py style."""
    import matplotlib.patches as mpatches

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.labelsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
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

    with open(results_path) as f:
        data = json.load(f)

    agg = data["aggregated"]
    per_seed = data["per_seed_results"]

    methods = ["mimic_mjx", "code2act"]
    labels = ["Mimic-MJX", "Code2Act"]
    colors = ["#56B4E9", "#56B4E9"]
    x = np.arange(len(methods))

    means = [agg[m]["kid_mean"] for m in methods]
    stds = [agg[m]["kid_std"] for m in methods]

    fig, ax = plt.subplots(figsize=(4.0, 3.3))

    bars = ax.bar(
        x, means, yerr=stds,
        color=colors, alpha=0.85, capsize=4, width=0.6,
        error_kw={"linewidth": 0.8, "capthick": 0.8},
        edgecolor="white", linewidth=0.5,
    )

    # Value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 0.01,
            f"{mean:.2f}",
            ha="center", va="bottom", fontsize=6.5, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("KID (Kernel Inception Distance)")
    ax.set_title("Generalization: Quality on Unseen Data")
    ax.set_ylim(bottom=0, top=max(means) + max(stds) + 0.08)
    ax.axhline(0, color="#e0e0e0", linewidth=0.4, zorder=0)

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])

    # Rounded border (matching kid_barplot.py)
    fig.patch.set_facecolor("white")
    rect = mpatches.FancyBboxPatch(
        (0.005, 0.005), 0.99, 0.99,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        transform=fig.transFigure,
        facecolor="white", edgecolor="#cccccc", linewidth=0.8, zorder=-1,
    )
    fig.patches.append(rect)

    out_dir = Path(output_dir)
    for ext in ("png", "pdf"):
        fig.savefig(str(out_dir / f"kid_comparison.{ext}"))
    rect.set_facecolor("none")
    fig.savefig(str(out_dir / "kid_comparison.svg"), transparent=True)
    plt.close(fig)

    log.info(f"KID plots saved to: {out_dir}/kid_comparison.{{png,pdf,svg}}")


def main():
    parser = argparse.ArgumentParser(
        description="Render triptych: Real | MIMIC | C2A with ghost bodies"
    )
    parser.add_argument("--rollouts", type=str, default=str(DEFAULT_ROLLOUT_PATH))
    parser.add_argument("--segments", type=str, default=str(DEFAULT_SEGMENT_PATH))
    parser.add_argument("--results", type=str, default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n_render", type=int, default=10)
    parser.add_argument("--panel_width", type=int, default=480)
    parser.add_argument("--panel_height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=50)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load rollout data
    log.info(f"Loading rollouts from: {args.rollouts}")
    data = np.load(args.rollouts)
    mimic_qpos = data["mimic_qpos"]  # [K, T, 74]
    c2a_qpos = data["c2a_qpos"]      # [K, T, 74]
    K_total = mimic_qpos.shape[0]
    T = mimic_qpos.shape[1]

    # Load reference qpos from segments
    log.info(f"Loading reference from: {args.segments}")
    with h5py.File(args.segments, "r") as f:
        ref_qpos_flat = f["qpos"][:]  # [K*frames_per_segment, 74]
    frames_per_seg = T  # should match
    ref_qpos = ref_qpos_flat.reshape(K_total, frames_per_seg, -1)

    n_render = min(args.n_render, K_total)

    # --- KID barplot ---
    if Path(args.results).exists():
        log.info("Generating KID comparison plot...")
        plot_kid_comparison(args.results, args.output_dir)
    else:
        log.warning(f"Results not found: {args.results}, skipping KID plot")

    log.info(f"Rendering {n_render} ghost bodies, {T} frames, 3 panels")

    # Select which bodies to render
    ref_trajs = [ref_qpos[i] for i in range(n_render)]
    mimic_trajs = [mimic_qpos[i] for i in range(n_render)]
    c2a_trajs = [c2a_qpos[i] for i in range(n_render)]

    colors = GHOST_COLORS_10[:n_render]

    # Build ghost model using existing env infrastructure
    log.info(f"Building ghost model with {n_render} bodies...")
    # Use standard reference clips H5 (just need any valid clips for env init)
    ref_h5 = str(REPO_ROOT / "data" / "rodent" / "rodent_reference_clips.h5")
    ghost_model, base_nq = build_ghost_model_via_env(
        ref_h5_path=ref_h5,
        num_bodies=n_render,
        colors=colors,
    )

    pw, ph = args.panel_width, args.panel_height
    camera = "divergent_cam"

    # Render each panel
    log.info("Rendering Real panel...")
    real_frames = render_ghost_panel(ghost_model, base_nq, ref_trajs, camera, pw, ph)
    log.info(f"  {len(real_frames)} frames")

    log.info("Rendering MIMIC panel...")
    mimic_frames = render_ghost_panel(ghost_model, base_nq, mimic_trajs, camera, pw, ph)
    log.info(f"  {len(mimic_frames)} frames")

    log.info("Rendering C2A panel...")
    c2a_frames = render_ghost_panel(ghost_model, base_nq, c2a_trajs, camera, pw, ph)
    log.info(f"  {len(c2a_frames)} frames")

    # Combine into triptych
    T_out = min(len(real_frames), len(mimic_frames), len(c2a_frames))
    divider = np.full((ph, 3, 3), 40, dtype=np.uint8)

    out_path = output_dir / "generalization_triptych.mp4"
    writer = imageio.get_writer(
        str(out_path), fps=args.fps,
        codec="libx264",
        output_params=["-pix_fmt", "yuv420p"],
        macro_block_size=1,
    )

    log.info(f"Writing triptych video ({T_out} frames)...")
    for t in range(T_out):
        left = add_multi_line_overlay(
            real_frames[t], ["Real"], start_position=(10, 10), font_size=14,
        )
        mid = add_multi_line_overlay(
            mimic_frames[t], ["MIMIC"], start_position=(10, 10), font_size=14,
        )
        right = add_multi_line_overlay(
            c2a_frames[t], ["C2A"], start_position=(10, 10), font_size=14,
        )
        combined = np.concatenate([left, divider, mid, divider, right], axis=1)
        # Add frame counter at bottom center
        combined = add_multi_line_overlay(
            combined, [f"t={t}"],
            start_position=(combined.shape[1] // 2 - 20, ph - 20),
            font_size=12,
        )
        writer.append_data(combined)

        if (t + 1) % 500 == 0:
            log.info(f"  {t+1}/{T_out} frames written")

    writer.close()
    log.info(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
