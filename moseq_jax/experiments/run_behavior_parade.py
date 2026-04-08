"""Experiment 7: Behavior transition parade.

K bodies spaced on the x-axis, viewed from a top-down camera. All bodies
receive the same code sequence: walk → groom → rear.  Behavior-representative
codes are selected from real data via kinematic criteria.

Usage:
    cd moseq_jax
    python -m experiments.run_behavior_parade

    # Override body count:
    python -m experiments.run_behavior_parade num_bodies=5
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

import h5py
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
from experiments.shared.plotting import set_nature_style, BEHAVIOR_COLORS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kinematic code analysis
# ---------------------------------------------------------------------------


def analyze_code_kinematics(
    all_codes: np.ndarray,
    reference_qpos: np.ndarray,
    clip_length: int,
    min_bout_frames: int = 10,
) -> dict[int, dict[str, float]]:
    """Compute per-code kinematic features from reference trajectories.

    For each bout of each code, extracts root XY displacement rate,
    root Z change (rise), and total XYZ stillness from the reference qpos.

    Args:
        all_codes: ``[n_clips, clip_length]`` KPMS code array.
        reference_qpos: ``[n_clips * clip_length, 74]`` flat reference qpos.
        clip_length: Frames per clip.
        min_bout_frames: Minimum bout length to consider.

    Returns:
        ``{code_id: {"xy_rate": median, "z_rise": median, "xyz_still": median}}``
    """
    n_clips = all_codes.shape[0]
    qpos_clips = reference_qpos[:n_clips * clip_length].reshape(n_clips, clip_length, -1)

    # Collect per-bout kinematics
    bout_stats: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"xy_rate": [], "z_rise": [], "xyz_still": []}
    )

    for ci in range(n_clips):
        codes = all_codes[ci]
        qpos = qpos_clips[ci]

        current = int(codes[0])
        start = 0
        for t in range(1, clip_length):
            if codes[t] != current or t == clip_length - 1:
                end = t if codes[t] != current else t + 1
                length = end - start
                if length >= min_bout_frames:
                    seg = qpos[start:end]
                    # XY displacement rate
                    xy_diff = np.diff(seg[:, :2], axis=0)
                    xy_path = np.sum(np.sqrt((xy_diff ** 2).sum(axis=1)))
                    xy_rate = xy_path / length

                    # Z rise: max Z relative to starting Z
                    z_rise = float(np.max(seg[:, 2]) - seg[0, 2])

                    # XYZ stillness: total displacement from start to end
                    xyz_total = float(np.sqrt(((seg[-1, :3] - seg[0, :3]) ** 2).sum()))

                    bout_stats[current]["xy_rate"].append(xy_rate)
                    bout_stats[current]["z_rise"].append(z_rise)
                    bout_stats[current]["xyz_still"].append(xyz_total)

                if codes[t] != current:
                    current = int(codes[t])
                    start = t

    # Compute medians
    result = {}
    for code_id, stats in bout_stats.items():
        result[code_id] = {
            "xy_rate": float(np.median(stats["xy_rate"])) if stats["xy_rate"] else 0.0,
            "z_rise": float(np.median(stats["z_rise"])) if stats["z_rise"] else 0.0,
            "xyz_still": float(np.median(stats["xyz_still"])) if stats["xyz_still"] else 0.0,
            "n_bouts": len(stats["xy_rate"]),
        }
    return result


def select_behavior_codes(
    kinematics: dict[int, dict[str, float]],
) -> dict[str, int]:
    """Select best code for walk, rear, groom based on kinematics.

    Returns:
        ``{"walk": code_id, "rear": code_id, "groom": code_id}``
    """
    codes = list(kinematics.keys())

    # Walk: highest XY displacement rate
    walk_code = max(codes, key=lambda c: kinematics[c]["xy_rate"])

    # Rear: highest Z rise
    rear_code = max(codes, key=lambda c: kinematics[c]["z_rise"])

    # Groom: lowest XYZ total displacement (stillest)
    groom_code = min(codes, key=lambda c: kinematics[c]["xyz_still"])

    # Resolve collisions: if any codes overlap, pick runner-up
    selected = {"walk": walk_code}
    remaining = [c for c in codes if c != walk_code]
    rear_code = max(remaining, key=lambda c: kinematics[c]["z_rise"])
    selected["rear"] = rear_code
    remaining = [c for c in remaining if c != rear_code]
    groom_code = min(remaining, key=lambda c: kinematics[c]["xyz_still"])
    selected["groom"] = groom_code

    return selected


def find_longest_bout(
    all_codes: np.ndarray,
    target_code: int,
) -> int:
    """Find the longest consecutive run of target_code across all clips."""
    best = 0
    for clip in all_codes:
        current_len = 0
        for t in range(len(clip)):
            if clip[t] == target_code:
                current_len += 1
                best = max(best, current_len)
            else:
                current_len = 0
    return best


def plot_code_selection(
    kinematics: dict[int, dict[str, float]],
    selected: dict[str, int],
    output_path: Path,
) -> None:
    """Plot kinematic features with selected codes highlighted."""
    set_nature_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))

    codes = sorted(kinematics.keys())
    metrics = [
        ("xy_rate", "XY Displacement Rate", "walk"),
        ("z_rise", "Z Rise (max - start)", "rear"),
        ("xyz_still", "XYZ Total Displacement", "groom"),
    ]

    for ax, (metric, title, beh) in zip(axes, metrics):
        vals = [kinematics[c][metric] for c in codes]
        colors = ["#cccccc"] * len(codes)
        sel_code = selected[beh]
        if sel_code in codes:
            colors[codes.index(sel_code)] = BEHAVIOR_COLORS[beh]

        ax.bar(range(len(codes)), vals, color=colors, edgecolor="none", width=0.8)
        ax.set_xlabel("Code")
        ax.set_title(title, fontsize=7)
        ax.set_xticks(range(0, len(codes), 5))
        ax.set_xticklabels([str(codes[i]) for i in range(0, len(codes), 5)], fontsize=5)

        # Annotate selected
        idx = codes.index(sel_code)
        ax.annotate(
            f"{beh}\n(code {sel_code})",
            xy=(idx, vals[idx]), xytext=(idx + 3, vals[idx] * 1.1),
            fontsize=6, color=BEHAVIOR_COLORS[beh], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=BEHAVIOR_COLORS[beh], lw=0.8),
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved code selection plot: {output_path}")


# ---------------------------------------------------------------------------
# Top-down parade rendering
# ---------------------------------------------------------------------------


def render_parade_video(
    env,
    all_qpos: list[np.ndarray],
    behavior_boundaries: list[tuple[str, int, int]],
    output_path: Path,
    width: int = 1280,
    height: int = 720,
    fps: int = 50,
    camera_distance: float = 1.5,
    camera_elevation: float = -89.5,
    camera_azimuth: float = 0.0,
    camera_fovy: float = 90.0,
) -> None:
    """Render top-down parade video with behavior phase labels."""
    from vqvae_jax.analysis.rendering import add_multi_line_overlay

    K = len(all_qpos)
    min_len = min(len(q) for q in all_qpos)

    # Ghost colors: spread across tab10
    cmap = plt.colormaps["tab10"]
    ghost_colors = [list(cmap(i % 10)) for i in range(1, K)]

    ghost_model, base_nq = build_ghost_model(
        env,
        num_ghosts=K - 1,
        ghost_colors=ghost_colors,
        camera_distance=camera_distance,
        camera_elevation=camera_elevation,
        camera_azimuth=camera_azimuth,
        camera_fovy=camera_fovy,
    )
    ghost_model.vis.global_.offwidth = width
    ghost_model.vis.global_.offheight = height
    data = mujoco.MjData(ghost_model)
    renderer = mujoco.Renderer(ghost_model, height=height, width=width)

    writer = imageio.get_writer(str(output_path), fps=fps)

    for t in range(min_len):
        # Set qpos for all bodies
        data.qpos[:base_nq] = all_qpos[0][t]
        for gi in range(1, K):
            q_start = base_nq + (gi - 1) * base_nq
            data.qpos[q_start:q_start + base_nq] = all_qpos[gi][t]

        mujoco.mj_forward(ghost_model, data)
        renderer.update_scene(data, camera="divergent_cam")
        frame = renderer.render().copy()

        # Behavior phase label
        current_beh = "—"
        for beh_name, beh_start, beh_end in behavior_boundaries:
            if beh_start <= t < beh_end:
                current_beh = beh_name.capitalize()
                break

        frame = add_multi_line_overlay(
            frame,
            [f"Behavior: {current_beh}  |  t={t}"],
            start_position=(10, 10),
            font_size=18,
        )

        writer.append_data(frame)

    renderer.close()
    writer.close()
    log.info(f"  Saved parade video: {output_path} ({K} bodies, {min_len} frames)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="configs", config_name="behavior_parade_exp")
def main(cfg: DictConfig) -> None:
    log.info("=== Behavior Transition Parade Experiment ===")

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

    # Load reference qpos for kinematic analysis
    with h5py.File(cfg.data.reference_data_path, "r") as f:
        reference_qpos = f["qpos"][:]

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
    # Step 1: Kinematic code analysis
    # ===================================================================
    log.info("\n--- Kinematic Code Analysis ---")
    clip_length = int(ckpt_cfg.env_config.clip_length)
    kinematics = analyze_code_kinematics(all_codes, reference_qpos, clip_length)

    for c in sorted(kinematics.keys()):
        k = kinematics[c]
        log.info(f"  Code {c}: xy_rate={k['xy_rate']:.4f}, z_rise={k['z_rise']:.4f}, "
                 f"xyz_still={k['xyz_still']:.4f} ({k['n_bouts']} bouts)")

    # ===================================================================
    # Step 2: Select behavior codes
    # ===================================================================
    log.info("\n--- Behavior Code Selection ---")
    selected = select_behavior_codes(kinematics)
    for beh, code_id in selected.items():
        k = kinematics[code_id]
        longest = find_longest_bout(all_codes, code_id)
        log.info(f"  {beh}: code {code_id} (xy={k['xy_rate']:.4f}, z_rise={k['z_rise']:.4f}, "
                 f"xyz={k['xyz_still']:.4f}, longest_bout={longest}f)")

    # Save selection
    selection_info = {
        beh: {
            "code_id": int(code_id),
            "kinematics": kinematics[code_id],
            "longest_bout": find_longest_bout(all_codes, code_id),
        }
        for beh, code_id in selected.items()
    }
    with open(output_dir / "code_selection.json", "w") as f:
        json.dump(selection_info, f, indent=2)

    plot_code_selection(kinematics, selected, output_dir / "code_selection.png")

    # ===================================================================
    # Step 3: Build code sequence
    # ===================================================================
    log.info("\n--- Building Code Sequence ---")
    frames_per_beh = int(cfg.frames_per_behavior)
    behavior_seq = list(cfg.behavior_sequence)

    code_sequence_parts = []
    behavior_boundaries = []
    offset = 0

    for beh in behavior_seq:
        code_id = selected[beh]
        part = np.full(frames_per_beh, code_id, dtype=np.int32)
        code_sequence_parts.append(part)
        behavior_boundaries.append((beh, offset, offset + frames_per_beh))
        offset += frames_per_beh
        log.info(f"  {beh}: code {code_id} × {frames_per_beh} frames")

    code_sequence = np.concatenate(code_sequence_parts)
    total_frames = len(code_sequence)
    log.info(f"  Total sequence: {total_frames} frames")

    # ===================================================================
    # Step 4: Select anchor clip + compute x-offsets
    # ===================================================================
    log.info("\n--- Body Setup ---")
    num_bodies = int(cfg.num_bodies)
    x_spacing = float(cfg.x_spacing)
    seed = int(cfg.seed)

    all_z0 = test_clips.qpos[:, 0, 2]
    anchor_clip = int(np.argmin(np.abs(all_z0 - np.median(all_z0))))
    log.info(f"  Anchor clip: {anchor_clip} (z={all_z0[anchor_clip]:.4f})")

    x_offsets = [
        i * x_spacing - (num_bodies - 1) * x_spacing / 2
        for i in range(num_bodies)
    ]
    log.info(f"  {num_bodies} bodies, x_offsets: [{x_offsets[0]:.3f} ... {x_offsets[-1]:.3f}]")

    # ===================================================================
    # Step 5: Run rollouts
    # ===================================================================
    log.info(f"\n--- Running {num_bodies} rollouts ---")
    all_qpos: list[np.ndarray] = []
    all_rewards: list[np.ndarray] = []

    for bi in range(num_bodies):
        key = jax.random.PRNGKey(seed + bi)
        result = run_rollout(
            env, inf_fn, code2act_params, ppo_networks, use_rnn, key,
            max_steps=total_frames,
            code_override=code_sequence,
            reset_clip_idx=anchor_clip,
            jit_reset=jit_reset, jit_step=jit_step,
            ignore_done=True,
        )
        qpos = result["qpos"][:-1].copy()
        all_rewards.append(result["rewards"])
        # Apply x-offset
        qpos[:, 0] += x_offsets[bi]
        all_qpos.append(qpos)
        log.info(f"  Body {bi}: {len(qpos)} frames, survival={result['survival']}")

    # ===================================================================
    # Step 6: Save rollout data
    # ===================================================================
    log.info("\n--- Saving rollout data ---")
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        data_dir / "parade_rollouts.npz",
        qpos=np.array(all_qpos, dtype=object),
        rewards=np.array(all_rewards, dtype=object),
        code_sequence=code_sequence,
        x_offsets=np.array(x_offsets),
        anchor_clip=anchor_clip,
        num_bodies=num_bodies,
    )
    log.info(f"  Saved rollout data to {data_dir}")

    # ===================================================================
    # Step 7: Render parade video
    # ===================================================================
    log.info("\n--- Rendering parade video ---")
    render_parade_video(
        env, all_qpos, behavior_boundaries,
        output_path=output_dir / "behavior_parade.mp4",
        width=int(cfg.rendering.width),
        height=int(cfg.rendering.height),
        fps=int(cfg.rendering.fps),
        camera_distance=float(cfg.rendering.camera_distance),
        camera_elevation=float(cfg.rendering.camera_elevation),
        camera_azimuth=float(cfg.rendering.camera_azimuth),
        camera_fovy=float(cfg.rendering.camera_fovy),
    )

    log.info("=== Behavior Transition Parade Experiment Complete ===")


if __name__ == "__main__":
    main()
