"""Render code generation methods as a triptych: [ARHMM | HMM | TM].

Each panel shows multiple sample trajectories from one generation method
as shaded ghost bodies (same style as render_behavior_triptych).  No code
timeline bar — clean rendering for publication.

Usage:
    cd moseq_jax
    python figures/render_code_generation_ghost.py
"""

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import cv2
import mujoco
import numpy as np

SCRIPT_DIR = Path(__file__).parent
MOSEQ_DIR = SCRIPT_DIR.parent
REPO_ROOT = MOSEQ_DIR.parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

for _p in (str(MOSEQ_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.shared.ghost_rendering import build_ghost_model
from vqvae_jax.analysis.rendering import add_multi_line_overlay

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

CODE_GEN_DIR = MOSEQ_DIR / "outputs" / "moseq_code_generation"

# ── Triptych configurations ─────────────────────────────────────────────────

TRIPTYCH_CONFIGS = {
    "methods": {
        "methods": ["arhmm_level2", "hmm_dynamax", "tm_T1.0"],
        "labels": {
            "arhmm_level2": "ARHMM Inference",
            "hmm_dynamax": "HMM Fit + Inference",
            "tm_T1.0": "TM Sample",
        },
        "output_name": "code_generation_triptych",
    },
    "baselines": {
        "methods": ["uniform_random", "single_code", "reversed"],
        "labels": {
            "uniform_random": "Uniform Random",
            "single_code": "Single Code",
            "reversed": "Reversed",
        },
        "output_name": "code_generation_baselines_triptych",
    },
}

# Method name → rollout .npz path
METHOD_ROLLOUT_PATHS = {
    "arhmm_level2": CODE_GEN_DIR / "arhmm" / "rollouts_arhmm_level2.npz",
    "hmm_dynamax": CODE_GEN_DIR / "hmm" / "rollouts_hmm_dynamax.npz",
    "tm_T1.0": CODE_GEN_DIR / "tm" / "rollouts_tm_T1.0.npz",
    "uniform_random": CODE_GEN_DIR / "baselines" / "rollouts_uniform_random.npz",
    "single_code": CODE_GEN_DIR / "baselines" / "rollouts_single_code.npz",
    "reversed": CODE_GEN_DIR / "baselines" / "rollouts_reversed.npz",
}


def get_distinct_ghost_colors(k: int) -> list[list[float]]:
    """Return k visually distinct RGBA colours for ghost bodies (tab10)."""
    import matplotlib.pyplot as plt
    cmap = plt.colormaps["tab10"]
    return [list(cmap(i % 10)) for i in range(k)]


def _center_qpos(trajectories_qpos: list[np.ndarray]) -> list[np.ndarray]:
    """Shift all trajectories so XY centroid is at origin each frame."""
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


def render_panel_frames(
    env,
    trajectories_qpos: list[np.ndarray],
    traj_colors: list[list[float]],
    panel_width: int = 480,
    panel_height: int = 480,
    max_frames: int = 400,
    camera_distance: float = 0.55,
    camera_elevation: float = -25.0,
    camera_azimuth: float = 135.0,
    camera_fovy: float = 50.0,
) -> list[np.ndarray]:
    """Render centered ghost frames for one method panel (no code bar)."""
    K = len(trajectories_qpos)
    centered = _center_qpos(trajectories_qpos)
    min_len = min(len(q) for q in centered)
    n_frames = min(min_len, max_frames)

    ghost_model, base_nq = build_ghost_model(
        env,
        num_ghosts=K - 1,
        ghost_colors=traj_colors[1:],
        camera_distance=camera_distance,
        camera_elevation=camera_elevation,
        camera_azimuth=camera_azimuth,
        camera_fovy=camera_fovy,
    )
    ghost_model.vis.global_.offwidth = panel_width
    ghost_model.vis.global_.offheight = panel_height
    data = mujoco.MjData(ghost_model)
    renderer = mujoco.Renderer(ghost_model, height=panel_height, width=panel_width)

    frames = []
    for t in range(n_frames):
        data.qpos[:base_nq] = centered[0][t]
        for gi in range(1, K):
            q_start = base_nq + (gi - 1) * base_nq
            q_end = q_start + base_nq
            data.qpos[q_start:q_end] = centered[gi][t]

        mujoco.mj_forward(ghost_model, data)
        renderer.update_scene(data, camera="divergent_cam")
        frame = renderer.render().copy()
        frames.append(frame)

    renderer.close()
    return frames


def load_method_trajectories(
    method: str,
    max_samples: int = 6,
) -> list[np.ndarray]:
    """Load full-length trajectories for a method, sorted by length."""
    rollout_path = METHOD_ROLLOUT_PATHS[method]
    roll = np.load(rollout_path, allow_pickle=True)
    survivals = roll["survivals"]

    # Sort by survival descending, pick longest
    sorted_idx = np.argsort(survivals)[::-1]
    selected = sorted_idx[:max_samples].tolist()

    trajs = []
    for idx in selected:
        qpos = np.asarray(roll["qpos"][idx], dtype=np.float64)
        trajs.append(qpos)

    # Truncate to same length
    min_len = min(len(q) for q in trajs)
    trajs = [q[:min_len] for q in trajs]
    return trajs


def render_triptych_for_config(
    env,
    config_name: str,
    args: argparse.Namespace,
) -> None:
    """Render a single triptych video for a given config (methods or baselines)."""
    tcfg = TRIPTYCH_CONFIGS[config_name]
    methods = tcfg["methods"]
    labels = tcfg["labels"]
    output_name = tcfg["output_name"]

    all_panel_frames: dict[str, list[np.ndarray]] = {}

    for method in methods:
        rollout_path = METHOD_ROLLOUT_PATHS[method]
        if not rollout_path.exists():
            log.warning(f"  Skipping {method}: {rollout_path} not found")
            continue

        label = labels[method]
        log.info(f"\n=== {label} ({method}) ===")

        trajs = load_method_trajectories(method, max_samples=args.max_traj)
        K = len(trajs)
        log.info(f"  {K} trajectories, {len(trajs[0])} frames")

        traj_colors = get_distinct_ghost_colors(K)

        frames = render_panel_frames(
            env, trajs, traj_colors,
            panel_width=args.panel_size, panel_height=args.panel_size,
            max_frames=args.max_frames,
            camera_distance=args.cam_dist, camera_elevation=args.cam_elev,
            camera_azimuth=args.cam_azim, camera_fovy=args.cam_fovy,
        )
        all_panel_frames[method] = frames
        log.info(f"  Rendered {len(frames)} frames")

    if not all_panel_frames:
        log.error(f"No panels rendered for '{config_name}'")
        return

    # ── Stitch into triptych ─────────────────────────────────────────
    rendered_methods = [m for m in methods if m in all_panel_frames]
    log.info(f"\n=== Compositing {config_name} triptych ({len(rendered_methods)} panels) ===")
    panel_w = args.panel_size
    panel_h = args.panel_size
    min_frames = min(len(f) for f in all_panel_frames.values())

    label_height = 36
    total_width = panel_w * len(rendered_methods)
    total_height = panel_h + label_height
    total_height = ((total_height + 15) // 16) * 16
    total_width = ((total_width + 15) // 16) * 16

    output_path = OUTPUT_DIR / f"{output_name}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (total_width, total_height))

    for t in range(min_frames):
        combined = np.ones((total_height, total_width, 3), dtype=np.uint8) * 40

        for mi, method in enumerate(rendered_methods):
            panel = all_panel_frames[method][t]
            x_off = mi * panel_w
            combined[label_height:label_height + panel_h, x_off:x_off + panel_w] = panel

        # Labels
        for mi, method in enumerate(rendered_methods):
            label = labels[method]
            x_center = mi * panel_w + panel_w // 2 - len(label) * 3
            combined = add_multi_line_overlay(
                combined, [label],
                start_position=(max(x_center, 5), 6),
                font_size=16,
            )

        # Timestep
        combined = add_multi_line_overlay(
            combined, [f"t={t}"],
            start_position=(total_width - 60, total_height - 20),
            font_size=12,
        )

        writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    writer.release()
    log.info(f"  Wrote triptych: {output_path}")
    log.info(f"  Size: {total_width} x {total_height}, {min_frames} frames")


def main():
    parser = argparse.ArgumentParser(description="Render code generation triptych")
    parser.add_argument("--mode", choices=["methods", "baselines", "both"], default="both",
                        help="Which triptych to render (default: both)")
    parser.add_argument("--max-traj", type=int, default=5,
                        help="Max trajectories per panel (default: 5)")
    parser.add_argument("--max-frames", type=int, default=400,
                        help="Max frames to render (default: 400)")
    parser.add_argument("--panel-size", type=int, default=480,
                        help="Panel width and height in pixels (default: 480)")
    parser.add_argument("--fps", type=int, default=50, help="Video FPS (default: 50)")
    parser.add_argument("--cam-dist", type=float, default=0.55)
    parser.add_argument("--cam-elev", type=float, default=-25.0)
    parser.add_argument("--cam-azim", type=float, default=135.0)
    parser.add_argument("--cam-fovy", type=float, default=50.0)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load env ─────────────────────────────────────────────────────
    import json
    from omegaconf import OmegaConf
    from track_mjx.config import utils
    from track_mjx.agent import checkpointing
    from vnl_playground.tasks.rodent.imitation import ReferenceClips
    from moseq_env_wrapper import MoSeqImitation

    ckpt_path = str(MOSEQ_DIR / "model_checkpoints" / "260406_155304_389732")
    cfg = checkpointing.load_config_from_checkpoint(ckpt_path, step_prefix="MoSeqPPONetwork")
    cfg = OmegaConf.create(cfg)
    _, _, env_cfg = utils.prepare_config(cfg)
    env_cfg.start_frame_range = [0, 0]
    env_cfg.domain_randomization.use_domain_randomization = False

    splits_path = REPO_ROOT / "data" / "rodent" / "rodent_balanced_splits.json"
    with open(splits_path) as f:
        splits = json.load(f)
    test_indices = splits["balanced"]["test_indices"]

    codes_data = np.load(MOSEQ_DIR / "outputs" / "kpms_sweep" / "best_codes.npz")
    test_codes = codes_data["test_codes"]

    test_clips = ReferenceClips(
        data_path=str(REPO_ROOT / "data" / "rodent" / "rodent_reference_clips.h5"),
        n_frames_per_clip=int(cfg.env_config.clip_length),
        keep_clips_idx=np.array(test_indices),
    )
    env = MoSeqImitation(config=env_cfg, clips=test_clips, kpms_codes=test_codes)
    log.info("Environment loaded")

    modes = ["methods", "baselines"] if args.mode == "both" else [args.mode]
    for mode in modes:
        log.info(f"\n{'='*60}")
        log.info(f"Rendering triptych: {mode}")
        log.info(f"{'='*60}")
        render_triptych_for_config(env, mode, args)


if __name__ == "__main__":
    main()
