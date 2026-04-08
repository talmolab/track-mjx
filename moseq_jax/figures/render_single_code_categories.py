"""Render single-code rollouts grouped by behavior category.

Reads per-code rollout data from ``outputs/moseq_single_code/data/``,
classifies each code into groom / walk / rear based on kinematics, and
renders a 3-panel triptych where each panel shows all codes of that
category as ghost bodies in a uniform colour.

Classification (from qpos_low — low starting height):
  - Rear:  z rises significantly from start and stays elevated
  - Walk:  high XY displacement rate, low z rise
  - Groom: low XY displacement rate, low z rise (still)

Uses the same camera angles as render_behavior_triptych.py.

Usage:
    cd moseq_jax
    python figures/render_single_code_categories.py
"""

import argparse
import json
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
OUTPUT_DIR = SCRIPT_DIR / "outputs"

for _p in (str(MOSEQ_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.shared.ghost_rendering import build_ghost_model
from vqvae_jax.analysis.rendering import add_multi_line_overlay

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

SINGLE_CODE_DIR = MOSEQ_DIR / "outputs" / "moseq_single_code" / "data"

BEHAVIOR_LABELS = {
    "groom": "Groom",
    "walk": "Walk",
    "rear": "Rear",
}
BEHAVIOR_ORDER = ["groom", "walk", "rear"]

# Wong Nature Methods 2011 — one colour per behavior
BEHAVIOR_RGBA = {
    "groom": [0.84, 0.37, 0.0, 1.0],    # #D55E00 orange
    "walk": [0.0, 0.45, 0.70, 1.0],      # #0072B2 blue
    "rear": [0.0, 0.62, 0.45, 1.0],      # #009E73 green
}


# ── Classification ───────────────────────────────────────────────────────────


def classify_codes(
    z_rise_thresh: float = 0.005,
    xy_rate_thresh: float = 0.0008,
) -> dict[str, list[int]]:
    """Classify each code into groom / walk / rear from qpos_low kinematics.

    Rear: z rises from start (max_z - start_z > z_rise_thresh) AND the rise
          is sustained (mean z in second half > start_z + z_rise_thresh/2).
    Walk: high xy displacement rate (> xy_rate_thresh), not rear.
    Groom: everything else (low movement).
    """
    categories: dict[str, list[int]] = {"groom": [], "walk": [], "rear": []}

    for code_id in range(50):
        fp = SINGLE_CODE_DIR / f"code_{code_id}.npz"
        if not fp.exists():
            continue
        d = np.load(fp)
        qpos = d["qpos_low"]
        T = len(qpos)

        # XY displacement rate
        xy_diff = np.diff(qpos[:, :2], axis=0)
        xy_path = np.sum(np.sqrt((xy_diff**2).sum(axis=1))) / T

        # Z rise: max z relative to starting z
        z_rise = float(np.max(qpos[:, 2]) - qpos[0, 2])

        # Z sustained: mean z in second half relative to start
        z_sustained = float(np.mean(qpos[T // 2 :, 2]) - qpos[0, 2])

        if z_rise > z_rise_thresh and z_sustained > z_rise_thresh / 2:
            categories["rear"].append(code_id)
        elif xy_path > xy_rate_thresh:
            categories["walk"].append(code_id)
        else:
            categories["groom"].append(code_id)

    return categories


# ── Rendering ────────────────────────────────────────────────────────────────


def _center_qpos(trajectories: list[np.ndarray]) -> list[np.ndarray]:
    """Shift all trajectories so XY centroid is at origin each frame."""
    min_len = min(len(q) for q in trajectories)
    stacked_xy = np.stack([q[:min_len, :2] for q in trajectories], axis=0)
    mean_xy = stacked_xy.mean(axis=0)
    centered = []
    for q in trajectories:
        qc = q[:min_len].copy()
        qc[:, 0] -= mean_xy[:, 0]
        qc[:, 1] -= mean_xy[:, 1]
        centered.append(qc)
    return centered


def render_panel_frames(
    env,
    trajectories_qpos: list[np.ndarray],
    body_color: list[float],
    panel_width: int = 480,
    panel_height: int = 480,
    max_frames: int = 100,
    camera_distance: float = 0.55,
    camera_elevation: float = -25.0,
    camera_azimuth: float = 135.0,
    camera_fovy: float = 50.0,
) -> list[np.ndarray]:
    """Render centered ghost frames for one category panel.

    All ghost bodies use the same colour.
    """
    K = len(trajectories_qpos)
    centered = _center_qpos(trajectories_qpos)
    min_len = min(len(q) for q in centered)
    n_frames = min(min_len, max_frames)

    # All ghosts same colour
    ghost_colors = [body_color] * (K - 1)

    ghost_model, base_nq = build_ghost_model(
        env,
        num_ghosts=K - 1,
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


def main():
    parser = argparse.ArgumentParser(
        description="Render single-code categories triptych"
    )
    parser.add_argument(
        "--variant",
        choices=["low", "high"],
        default="low",
        help="Which starting height to render (default: low)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Max frames to render per code (default: 100)",
    )
    parser.add_argument(
        "--panel-size",
        type=int,
        default=480,
        help="Panel width and height in pixels (default: 480)",
    )
    parser.add_argument("--fps", type=int, default=50, help="Video FPS (default: 50)")
    parser.add_argument("--cam-dist", type=float, default=0.55)
    parser.add_argument("--cam-elev", type=float, default=-25.0)
    parser.add_argument("--cam-azim", type=float, default=135.0)
    parser.add_argument("--cam-fovy", type=float, default=50.0)
    parser.add_argument(
        "--z-rise-thresh",
        type=float,
        default=0.005,
        help="Z-rise threshold for rear classification (default: 0.005)",
    )
    parser.add_argument(
        "--xy-rate-thresh",
        type=float,
        default=0.0008,
        help="XY displacement rate threshold for walk (default: 0.0008)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Classify codes ───────────────────────────────────────────────
    categories = classify_codes(
        z_rise_thresh=args.z_rise_thresh,
        xy_rate_thresh=args.xy_rate_thresh,
    )
    for beh in BEHAVIOR_ORDER:
        log.info(f"  {beh}: {len(categories[beh])} codes — {categories[beh]}")

    # Save classification
    classification_path = OUTPUT_DIR / "single_code_categories.json"
    with open(classification_path, "w") as f:
        json.dump(categories, f, indent=2)
    log.info(f"  Saved classification: {classification_path}")

    # ── Load env ─────────────────────────────────────────────────────
    from omegaconf import OmegaConf
    from track_mjx.config import utils
    from track_mjx.agent import checkpointing
    from vnl_playground.tasks.rodent.imitation import ReferenceClips
    from moseq_env_wrapper import MoSeqImitation

    ckpt_path = str(MOSEQ_DIR / "model_checkpoints" / "260407_031233_484020")
    cfg = checkpointing.load_config_from_checkpoint(
        ckpt_path, step_prefix="MoSeqPPONetwork"
    )
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

    # ── Load trajectories and render each panel ──────────────────────
    qpos_key = f"qpos_{args.variant}"
    all_panel_frames: dict[str, list[np.ndarray]] = {}

    for beh in BEHAVIOR_ORDER:
        code_ids = categories[beh]
        if not code_ids:
            log.warning(f"  No codes for {beh}, skipping")
            continue

        # Load one trajectory per code
        trajs = []
        for code_id in code_ids:
            fp = SINGLE_CODE_DIR / f"code_{code_id}.npz"
            d = np.load(fp)
            qpos = np.asarray(d[qpos_key], dtype=np.float64)
            trajs.append(qpos)

        log.info(
            f"\n=== {BEHAVIOR_LABELS[beh]} ({len(trajs)} codes: {code_ids}) ==="
        )

        body_color = BEHAVIOR_RGBA[beh]
        frames = render_panel_frames(
            env,
            trajs,
            body_color,
            panel_width=args.panel_size,
            panel_height=args.panel_size,
            max_frames=args.max_frames,
            camera_distance=args.cam_dist,
            camera_elevation=args.cam_elev,
            camera_azimuth=args.cam_azim,
            camera_fovy=args.cam_fovy,
        )
        all_panel_frames[beh] = frames
        log.info(f"  Rendered {len(frames)} frames")

    if not all_panel_frames:
        log.error("No panels rendered")
        return

    # ── Stitch into triptych ─────────────────────────────────────────
    rendered_behs = [b for b in BEHAVIOR_ORDER if b in all_panel_frames]
    min_frames = min(len(f) for f in all_panel_frames.values())
    log.info(
        f"\n=== Compositing triptych ({len(rendered_behs)} panels, "
        f"{min_frames} frames) ==="
    )

    panel_w = args.panel_size
    panel_h = args.panel_size
    label_height = 36
    total_width = panel_w * len(rendered_behs)
    total_height = panel_h + label_height
    total_height = ((total_height + 15) // 16) * 16
    total_width = ((total_width + 15) // 16) * 16

    output_path = OUTPUT_DIR / f"single_code_categories_{args.variant}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path), fourcc, args.fps, (total_width, total_height)
    )

    for t in range(min_frames):
        combined = np.ones((total_height, total_width, 3), dtype=np.uint8) * 40

        for bi, beh in enumerate(rendered_behs):
            panel = all_panel_frames[beh][t]
            x_off = bi * panel_w
            combined[
                label_height : label_height + panel_h, x_off : x_off + panel_w
            ] = panel

        # Labels with code count
        for bi, beh in enumerate(rendered_behs):
            n_codes = len(categories[beh])
            label = f"{BEHAVIOR_LABELS[beh]} ({n_codes} codes)"
            x_center = bi * panel_w + panel_w // 2 - len(label) * 3
            combined = add_multi_line_overlay(
                combined,
                [label],
                start_position=(max(x_center, 5), 6),
                font_size=16,
            )

        # Timestep
        combined = add_multi_line_overlay(
            combined,
            [f"t={t}"],
            start_position=(total_width - 60, total_height - 20),
            font_size=12,
        )

        writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    writer.release()
    log.info(f"  Wrote triptych: {output_path}")
    log.info(f"  Size: {total_width} x {total_height}, {min_frames} frames")


if __name__ == "__main__":
    main()
