"""Experiment 7: Behavior transition parade.

K bodies spaced on the x-axis, viewed from a top-down camera. All bodies
receive the same code sequence: walk → groom → rear.  For each behavior,
a real representative clip is selected from the balanced splits via kinematic
criteria, and its full KPMS code *sequence* is used (not a single held code).

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
# Representative clip selection
# ---------------------------------------------------------------------------


def select_representative_clips(
    test_codes: np.ndarray,
    test_categories: list[str],
    test_indices: list[int],
    reference_qpos: np.ndarray,
    clip_length: int,
) -> dict[str, dict]:
    """Select the best representative test clip for each behavior.

    Instead of picking a single code, selects a real clip whose KPMS code
    *sequence* captures the temporal dynamics of the behavior (e.g. rear
    involves a crouch-to-rise transition, not a single static code).

    Selection criteria:
        walk: highest XY path length (sustained horizontal movement).
        rear: highest Z rise from start (rises up and stays high).
        groom: lowest total XYZ displacement (stays still).

    Args:
        test_codes: ``[n_test, clip_length]`` KPMS code array for test clips.
        test_categories: Category label per test clip.
        test_indices: Original clip index per test clip.
        reference_qpos: ``[n_total_frames, nq]`` flat reference qpos.
        clip_length: Frames per clip.

    Returns:
        ``{behavior: {"test_pos": int, "clip_idx": int,
        "code_sequence": ndarray, "kinematics": dict}}``
    """
    score_fns = {
        "walk": lambda xy, zr, zrise, xyz: xy,
        "rear": lambda xy, zr, zrise, xyz: zrise,
        "groom": lambda xy, zr, zrise, xyz: -xyz,
    }

    result = {}
    for cat, score_fn in score_fns.items():
        cat_positions = [i for i, c in enumerate(test_categories) if c == cat]

        best_pos = None
        best_score = None
        best_kin = None

        for pos in cat_positions:
            orig_idx = test_indices[pos]
            start = orig_idx * clip_length
            end = start + clip_length
            if end > len(reference_qpos):
                continue
            seg = reference_qpos[start:end]

            xy_diff = np.diff(seg[:, :2], axis=0)
            xy_path = float(np.sum(np.sqrt((xy_diff ** 2).sum(axis=1))))
            z_range = float(np.max(seg[:, 2]) - np.min(seg[:, 2]))
            z_rise = float(np.max(seg[:, 2]) - seg[0, 2])
            xyz_disp = float(np.sqrt(((seg[-1, :3] - seg[0, :3]) ** 2).sum()))

            score = score_fn(xy_path, z_range, z_rise, xyz_disp)
            if best_score is None or score > best_score:
                best_score = score
                best_pos = pos
                best_kin = {
                    "xy_path": xy_path,
                    "z_range": z_range,
                    "z_rise": z_rise,
                    "xyz_disp": xyz_disp,
                }

        result[cat] = {
            "test_pos": best_pos,
            "clip_idx": test_indices[best_pos],
            "code_sequence": test_codes[best_pos].copy(),
            "kinematics": best_kin,
        }

    return result


def _run_length_encode(codes: np.ndarray) -> list[tuple[int, int]]:
    """Run-length encode a 1-D code sequence."""
    runs = []
    cur, cnt = int(codes[0]), 1
    for t in range(1, len(codes)):
        if codes[t] == cur:
            cnt += 1
        else:
            runs.append((cur, cnt))
            cur, cnt = int(codes[t]), 1
    runs.append((cur, cnt))
    return runs


def plot_code_sequences(
    selected: dict[str, dict],
    output_path: Path,
) -> None:
    """Plot code-sequence timelines for each selected behavior clip."""
    set_nature_style()
    behaviors = [b for b in ("walk", "groom", "rear") if b in selected]
    fig, axes = plt.subplots(len(behaviors), 1, figsize=(7.2, 1.2 * len(behaviors)),
                             sharex=True)
    if len(behaviors) == 1:
        axes = [axes]

    for ax, beh in zip(axes, behaviors):
        codes = selected[beh]["code_sequence"]
        clip_idx = selected[beh]["clip_idx"]
        runs = _run_length_encode(codes)

        offset = 0
        for code_id, dur in runs:
            ax.barh(0, dur, left=offset, height=0.6,
                    color=BEHAVIOR_COLORS.get(beh, "#999999"), alpha=0.8,
                    edgecolor="white", linewidth=0.5)
            if dur > 10:
                ax.text(offset + dur / 2, 0, str(code_id),
                        ha="center", va="center", fontsize=6, fontweight="bold")
            offset += dur

        ax.set_ylabel(f"{beh}\n(clip {clip_idx})", fontsize=7, rotation=0,
                       labelpad=55, va="center")
        ax.set_yticks([])
        ax.set_xlim(0, len(codes))

    axes[-1].set_xlabel("Frame")
    fig.suptitle("Selected Behavior Code Sequences", fontsize=9, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved code sequence plot: {output_path}")


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
    # Step 1: Select representative clips per behavior
    # ===================================================================
    log.info("\n--- Representative Clip Selection ---")
    clip_length = int(ckpt_cfg.env_config.clip_length)
    test_categories = splits["balanced"]["test_categories"]

    selected = select_representative_clips(
        test_codes, test_categories, test_indices, reference_qpos, clip_length,
    )
    for beh, info in selected.items():
        k = info["kinematics"]
        codes = info["code_sequence"]
        unique = np.unique(codes)
        log.info(
            f"  {beh}: clip {info['clip_idx']} ({len(unique)} unique codes, "
            f"xy={k['xy_path']:.4f}, z_rise={k['z_rise']:.4f}, "
            f"xyz={k['xyz_disp']:.4f})"
        )

    # Save selection
    selection_info = {
        beh: {
            "clip_idx": info["clip_idx"],
            "test_pos": info["test_pos"],
            "code_sequence": info["code_sequence"].tolist(),
            "kinematics": info["kinematics"],
        }
        for beh, info in selected.items()
    }
    with open(output_dir / "code_selection.json", "w") as f:
        json.dump(selection_info, f, indent=2)

    plot_code_sequences(selected, output_dir / "code_selection.png")

    # ===================================================================
    # Step 2: Build code sequence from representative clips
    # ===================================================================
    log.info("\n--- Building Code Sequence ---")
    frames_per_beh = int(cfg.frames_per_behavior)
    behavior_seq = list(cfg.behavior_sequence)

    code_sequence_parts = []
    behavior_boundaries = []
    offset = 0

    for beh in behavior_seq:
        clip_codes = selected[beh]["code_sequence"][:frames_per_beh]
        code_sequence_parts.append(clip_codes)
        behavior_boundaries.append((beh, offset, offset + len(clip_codes)))
        offset += len(clip_codes)
        runs = _run_length_encode(clip_codes)
        log.info(f"  {beh}: {' -> '.join(f'{c}x{n}' for c, n in runs)}")

    code_sequence = np.concatenate(code_sequence_parts)
    total_frames = len(code_sequence)
    log.info(f"  Total sequence: {total_frames} frames")

    # ===================================================================
    # Step 3: Select anchor clip + compute x-offsets
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
    # Step 4: Run rollouts
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
    # Step 5: Save rollout data
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
        behavior_names=np.array([b for b, _, _ in behavior_boundaries]),
        behavior_starts=np.array([s for _, s, _ in behavior_boundaries]),
        behavior_ends=np.array([e for _, _, e in behavior_boundaries]),
    )
    log.info(f"  Saved rollout data to {data_dir}")

    # ===================================================================
    # Step 6: Render parade video
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
