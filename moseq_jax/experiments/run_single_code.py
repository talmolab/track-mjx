"""Experiment 6: Single-code sustain grid.

For each KPMS code (ordered most→least popular), holds the code constant
for K frames with two body instantiations (low-z / high-z starting pose).
Outputs: code frequency histogram, bout stats, and a grid video.

Usage:
    cd moseq_jax
    python -m experiments.run_single_code

    # Override sustain duration:
    python -m experiments.run_single_code sustain_frames=200
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import hydra
import imageio
import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from omegaconf import DictConfig

MOSEQ_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = MOSEQ_DIR.parent
for _p in (str(MOSEQ_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from track_mjx.config import utils
from vnl_playground.tasks.rodent.imitation import ReferenceClips
from moseq_env_wrapper import MoSeqImitation

from experiments.shared.checkpoint_utils import (
    load_moseq_checkpoint,
    make_inference_fn,
    run_rollout,
)
from experiments.shared.clip_selection import load_balanced_splits
from experiments.shared.ghost_rendering import build_ghost_model
from experiments.shared.plotting import set_nature_style, get_code_colormap

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Code analysis
# ---------------------------------------------------------------------------


def compute_code_stats(
    all_codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[int, list[int]]]:
    """Compute code frequencies and bout durations.

    Args:
        all_codes: ``[n_clips, clip_length]`` int array of KPMS codes.

    Returns:
        ``(code_order, code_counts, bout_durations)`` where code_order
        is sorted by frequency (most popular first), code_counts are the
        corresponding frame counts, and bout_durations maps code_id to
        list of bout lengths.
    """
    unique, counts = np.unique(all_codes, return_counts=True)
    order = np.argsort(-counts)
    code_order = unique[order]
    code_counts = counts[order]

    bout_durations: dict[int, list[int]] = defaultdict(list)
    for clip in all_codes:
        current = int(clip[0])
        length = 1
        for t in range(1, len(clip)):
            if clip[t] == current:
                length += 1
            else:
                bout_durations[current].append(length)
                current = int(clip[t])
                length = 1
        bout_durations[current].append(length)

    return code_order, code_counts, dict(bout_durations)


def plot_code_frequency(
    code_order: np.ndarray,
    code_counts: np.ndarray,
    bout_durations: dict[int, list[int]],
    output_path: Path,
) -> None:
    """Save Nature-style code frequency bar chart."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.0))

    x = np.arange(len(code_order))
    total = code_counts.sum()
    pct = code_counts / total * 100

    ax.bar(x, pct, color="#0072B2", edgecolor="none", width=0.8)
    ax.set_xlabel("Code (ordered by frequency)")
    ax.set_ylabel("Frequency (%)")
    ax.set_title("KPMS Code Distribution")
    ax.set_xticks(x[::5])
    ax.set_xticklabels([str(code_order[i]) for i in range(0, len(code_order), 5)],
                       fontsize=5)
    ax.set_xlim(-0.5, len(code_order) - 0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved code frequency plot: {output_path}")


# ---------------------------------------------------------------------------
# Per-code panel rendering
# ---------------------------------------------------------------------------


def render_code_panel_frames(
    env,
    qpos_low: np.ndarray,
    qpos_high: np.ndarray,
    code_id: int,
    median_bout: float,
    code_colors: np.ndarray,
    panel_width: int = 240,
    panel_height: int = 240,
    camera_distance: float = 0.55,
    camera_elevation: float = -25.0,
    camera_azimuth: float = 135.0,
    camera_fovy: float = 50.0,
) -> list[np.ndarray]:
    """Render 2-body panel (low + high) for one code."""
    from vqvae_jax.analysis.rendering import add_multi_line_overlay

    min_len = min(len(qpos_low), len(qpos_high))

    # Build ghost model with 1 ghost (2 bodies total)
    ghost_colors = [[0.85, 0.45, 0.15, 0.7]]  # orange-ish for high body
    ghost_model, base_nq = build_ghost_model(
        env,
        num_ghosts=1,
        ghost_colors=ghost_colors,
        camera_distance=camera_distance,
        camera_elevation=camera_elevation,
        camera_azimuth=camera_azimuth,
        camera_fovy=camera_fovy,
    )
    ghost_model.vis.global_.offwidth = panel_width
    ghost_model.vis.global_.offheight = panel_height
    data = mujoco.MjData(ghost_model)
    renderer = mujoco.Renderer(ghost_model, height=panel_height, width=panel_width)

    label = f"Code {code_id} (med: {median_bout:.0f}f)"

    frames = []
    for t in range(min_len):
        data.qpos[:base_nq] = qpos_low[t]
        data.qpos[base_nq:base_nq * 2] = qpos_high[t]
        mujoco.mj_forward(ghost_model, data)
        renderer.update_scene(data, camera="divergent_cam")
        frame = renderer.render().copy()

        # Label overlay
        frame = add_multi_line_overlay(
            frame, [label],
            start_position=(4, 4),
            font_size=11,
        )
        frames.append(frame)

    renderer.close()
    return frames


# ---------------------------------------------------------------------------
# Grid video composition
# ---------------------------------------------------------------------------


def compose_grid_video(
    all_panel_frames: list[list[np.ndarray]],
    grid_columns: int,
    output_path: Path,
    fps: int = 50,
    bg_color: int = 40,
) -> None:
    """Stitch per-code panel frame lists into a single grid video."""
    n_codes = len(all_panel_frames)
    grid_rows = (n_codes + grid_columns - 1) // grid_columns

    # Get panel dimensions from first frame
    ph, pw = all_panel_frames[0][0].shape[:2]
    min_frames = min(len(f) for f in all_panel_frames)

    total_w = pw * grid_columns
    total_h = ph * grid_rows
    # Round to multiple of 16 for codec compatibility
    total_w = ((total_w + 15) // 16) * 16
    total_h = ((total_h + 15) // 16) * 16

    writer = imageio.get_writer(str(output_path), fps=fps)

    for t in range(min_frames):
        canvas = np.full((total_h, total_w, 3), bg_color, dtype=np.uint8)

        for idx in range(n_codes):
            row = idx // grid_columns
            col = idx % grid_columns
            x_off = col * pw
            y_off = row * ph
            canvas[y_off:y_off + ph, x_off:x_off + pw] = all_panel_frames[idx][t]

        writer.append_data(canvas)

    writer.close()
    log.info(f"  Saved grid video: {output_path} ({grid_rows}×{grid_columns}, {min_frames} frames)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="configs", config_name="single_code_exp")
def main(cfg: DictConfig) -> None:
    log.info("=== Single-Code Sustain Grid Experiment ===")

    output_dir = Path(cfg.output.base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    ckpt_cfg, norm_state, policy_params, ppo_networks = load_moseq_checkpoint(
        cfg.checkpoint.path,
    )
    use_rnn = bool(ckpt_cfg.network_config.get("use_rnn_decoder", False))
    num_codes = int(ckpt_cfg.network_config.num_codes)
    code2act_params = (norm_state, policy_params)

    # Load codes
    codes_data = np.load(cfg.data.codes_path)
    all_codes = codes_data["all_codes"]
    test_codes = codes_data["test_codes"]
    splits = load_balanced_splits(cfg.data.balanced_split_path)
    test_indices = splits["balanced"]["test_indices"]

    # Create env
    test_clips = ReferenceClips(
        data_path=cfg.data.reference_data_path,
        n_frames_per_clip=int(ckpt_cfg.env_config.clip_length),
        keep_clips_idx=np.array(test_indices),
    )
    _, _, env_cfg = utils.prepare_config(ckpt_cfg)
    env_cfg.start_frame_range = [0, 0]
    env_cfg.domain_randomization.use_domain_randomization = False
    code_stack_size = int(ckpt_cfg.network_config.get("code_stack_size", 1))
    env = MoSeqImitation(
        config=env_cfg, clips=test_clips, kpms_codes=test_codes,
        code_stack_size=code_stack_size,
    )

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    inf_fn = make_inference_fn(ppo_networks, use_rnn=use_rnn, deterministic=True)

    # ===================================================================
    # Step 1: Code analysis
    # ===================================================================
    log.info("\n--- Code Analysis ---")
    code_order, code_counts, bout_durations = compute_code_stats(all_codes)
    log.info(f"  Active codes: {len(code_order)}/{num_codes}")
    for i in range(min(10, len(code_order))):
        c = int(code_order[i])
        bouts = bout_durations.get(c, [])
        med = float(np.median(bouts)) if bouts else 0
        log.info(f"    Code {c}: {code_counts[i]} frames ({code_counts[i]/all_codes.size*100:.1f}%), median bout={med:.0f}f")

    # Save frequency plot
    plot_code_frequency(code_order, code_counts, bout_durations,
                        output_dir / "code_frequency.png")

    # Save bout stats JSON
    bout_stats = {}
    for c in code_order:
        c = int(c)
        bouts = bout_durations.get(c, [])
        bout_stats[str(c)] = {
            "total_frames": int(sum(bouts)),
            "n_bouts": len(bouts),
            "median_duration": float(np.median(bouts)) if bouts else 0,
            "mean_duration": float(np.mean(bouts)) if bouts else 0,
        }
    with open(output_dir / "bout_stats.json", "w") as f:
        json.dump(bout_stats, f, indent=2)

    # ===================================================================
    # Step 2: Select anchor clips (low / high root-z)
    # ===================================================================
    log.info("\n--- Anchor Clip Selection ---")
    all_z0 = test_clips.qpos[:, 0, 2]
    low_clip_idx = int(np.argmin(np.abs(all_z0 - np.median(all_z0))))
    high_z_target = float(np.percentile(all_z0, 90))
    high_clip_idx = int(np.argmin(np.abs(all_z0 - high_z_target)))
    anchor_clips = {"low": low_clip_idx, "high": high_clip_idx}
    log.info(f"  Low anchor: clip {low_clip_idx} (z={all_z0[low_clip_idx]:.4f})")
    log.info(f"  High anchor: clip {high_clip_idx} (z={all_z0[high_clip_idx]:.4f})")

    # ===================================================================
    # Step 3: Per-code rollouts
    # ===================================================================
    sustain_frames = int(cfg.sustain_frames)
    seed = int(cfg.seed)
    log.info(f"\n--- Running {len(code_order)} codes × 2 heights = {len(code_order)*2} rollouts ---")

    code_qpos: dict[int, dict[str, np.ndarray]] = {}
    code_rewards: dict[int, dict[str, np.ndarray]] = {}
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for ci, code_id in enumerate(code_order):
        code_id = int(code_id)
        code_seq = np.full(sustain_frames, code_id, dtype=np.int32)
        code_qpos[code_id] = {}
        code_rewards[code_id] = {}

        for height, clip_idx in anchor_clips.items():
            key = jax.random.PRNGKey(seed + code_id * 100 + (0 if height == "low" else 1))
            result = run_rollout(
                env, inf_fn, code2act_params, ppo_networks, use_rnn, key,
                max_steps=sustain_frames,
                code_override=code_seq,
                reset_clip_idx=clip_idx,
                jit_reset=jit_reset, jit_step=jit_step,
                ignore_done=True,
            )
            code_qpos[code_id][height] = result["qpos"][:-1]
            code_rewards[code_id][height] = result["rewards"]

        # Save per-code rollout data
        np.savez_compressed(
            data_dir / f"code_{code_id}.npz",
            qpos_low=code_qpos[code_id]["low"],
            qpos_high=code_qpos[code_id]["high"],
            rewards_low=code_rewards[code_id]["low"],
            rewards_high=code_rewards[code_id]["high"],
            code_id=code_id,
            sustain_frames=sustain_frames,
        )

        if (ci + 1) % 10 == 0 or ci == len(code_order) - 1:
            log.info(f"    {ci+1}/{len(code_order)} codes done")

    # ===================================================================
    # Step 4: Render per-code panels
    # ===================================================================
    log.info("\n--- Rendering panels ---")
    code_colors = get_code_colormap(num_codes)
    all_panel_frames: list[list[np.ndarray]] = []

    for ci, code_id in enumerate(code_order):
        code_id = int(code_id)
        bouts = bout_durations.get(code_id, [])
        median_bout = float(np.median(bouts)) if bouts else 0

        frames = render_code_panel_frames(
            env,
            code_qpos[code_id]["low"],
            code_qpos[code_id]["high"],
            code_id=code_id,
            median_bout=median_bout,
            code_colors=code_colors,
            panel_width=int(cfg.rendering.panel_width),
            panel_height=int(cfg.rendering.panel_height),
            camera_distance=float(cfg.rendering.camera_distance),
            camera_elevation=float(cfg.rendering.camera_elevation),
            camera_azimuth=float(cfg.rendering.camera_azimuth),
            camera_fovy=float(cfg.rendering.camera_fovy),
        )
        all_panel_frames.append(frames)

        if (ci + 1) % 10 == 0 or ci == len(code_order) - 1:
            log.info(f"    Rendered {ci+1}/{len(code_order)} panels")

    # ===================================================================
    # Step 5: Compose grid videos (5 rows per video)
    # ===================================================================
    log.info("\n--- Composing grid videos ---")
    grid_cols = int(cfg.grid_columns)
    grid_rows_per_video = int(cfg.grid_rows_per_video)
    codes_per_video = grid_cols * grid_rows_per_video

    for vi in range(0, len(all_panel_frames), codes_per_video):
        chunk = all_panel_frames[vi:vi + codes_per_video]
        video_idx = vi // codes_per_video
        compose_grid_video(
            chunk,
            grid_columns=grid_cols,
            output_path=output_dir / f"single_code_grid_{video_idx}.mp4",
            fps=int(cfg.rendering.fps),
        )

    log.info("=== Single-Code Sustain Grid Experiment Complete ===")


if __name__ == "__main__":
    main()
