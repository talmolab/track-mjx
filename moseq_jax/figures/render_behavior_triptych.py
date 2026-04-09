"""Render walk / rear / groom ghost videos side by side as a triptych.

For each behavior, renders K overlaid trajectories (from killer demo data)
with walkers centered in each panel. Uses the same ghost-body rendering
style as the experiment pipeline (per-trajectory tab10 colours, standard
scene lighting). No code bar overlay.

Outputs a single combined video: [Groom | Walk | Rear].

Usage:
    cd moseq_jax/figures
    python render_behavior_triptych.py

    # Use high-start data instead of low:
    python render_behavior_triptych.py --height high
"""

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import cv2
import matplotlib.pyplot as plt
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

from experiments.shared.ghost_rendering import build_ghost_model, render_ghost_video
from vqvae_jax.analysis.rendering import add_multi_line_overlay

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

BEHAVIORS = ["groom", "walk", "rear"]
BEHAVIOR_LABELS = {
    "groom": "Groom Pose Instantiation",
    "walk": "Walk Pose Instantiation",
    "rear": "Rear Pose Instantiation",
}

def get_distinct_ghost_colors(k: int) -> list[list[float]]:
    """Return k visually distinct RGBA colours for ghost bodies (tab10)."""
    import matplotlib.pyplot as plt
    cmap = plt.colormaps["tab10"]
    return [list(cmap(i % 10)) for i in range(k)]


def _center_qpos(trajectories_qpos: list[np.ndarray]) -> list[np.ndarray]:
    """Shift all trajectories so the XY centroid is at the origin each frame."""
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
    """Render centered ghost frames for one behavior panel."""
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


def render_triptych(
    env,
    height: str = "low",
    panel_width: int = 480,
    panel_height: int = 480,
    max_trajectories: int = 6,
    max_frames: int = 400,
    fps: int = 50,
    output_name: str | None = None,
    camera_distance: float = 0.55,
    camera_elevation: float = -25.0,
    camera_azimuth: float = 0.0,
    camera_fovy: float = 50.0,
):
    """Render 3-panel triptych: [Groom | Walk | Rear] with gait diagrams."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_panel_frames = {}

    for beh in BEHAVIORS:
        data_path = DATA_DIR / f"killer_{beh}_{height}.npz"
        if not data_path.exists():
            log.warning(f"Missing data: {data_path}")
            continue

        d = np.load(data_path, allow_pickle=True)
        all_qpos = [np.asarray(d["qpos"][i], dtype=np.float64) for i in range(len(d["qpos"]))]
        # Sort by length descending and pick the longest trajectories
        all_qpos.sort(key=lambda q: len(q), reverse=True)
        qpos_list = all_qpos[:max_trajectories]
        K = len(qpos_list)
        lens = [len(q) for q in qpos_list]
        log.info(f"  {beh}: {K} trajectories, lengths={lens}, min={min(lens)}")

        traj_colors = get_distinct_ghost_colors(K)

        frames = render_panel_frames(
            env, qpos_list, traj_colors,
            panel_width=panel_width, panel_height=panel_height,
            max_frames=max_frames,
            camera_distance=camera_distance, camera_elevation=camera_elevation,
            camera_azimuth=camera_azimuth, camera_fovy=camera_fovy,
        )
        all_panel_frames[beh] = frames
        log.info(f"    Rendered {len(frames)} frames")

    if not all_panel_frames:
        log.error("No panels rendered")
        return

    # Stitch panels side by side
    min_frames = min(len(f) for f in all_panel_frames.values())
    log.info(f"  Stitching {min_frames} frames across {len(all_panel_frames)} panels")

    # Layout
    label_height = 36
    total_width = panel_width * len(BEHAVIORS)
    total_height = panel_height + label_height
    # Round up to multiple of 16 for codec compatibility
    total_height = ((total_height + 15) // 16) * 16
    total_width = ((total_width + 15) // 16) * 16

    if output_name is None:
        output_name = f"behavior_triptych_{height}"
    output_path = OUTPUT_DIR / f"{output_name}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (total_width, total_height))

    for t in range(min_frames):
        combined = np.ones((total_height, total_width, 3), dtype=np.uint8) * 40  # dark grey bg

        for bi, beh in enumerate(BEHAVIORS):
            if beh not in all_panel_frames:
                continue
            panel = all_panel_frames[beh][t]
            x_off = bi * panel_width
            combined[label_height:label_height + panel_height, x_off:x_off + panel_width] = panel

        # Add behavior labels centred above each panel
        for bi, beh in enumerate(BEHAVIORS):
            if beh not in all_panel_frames:
                continue
            label = BEHAVIOR_LABELS[beh]
            # Approximate centering: ~6px per char at font_size=16
            x_center = bi * panel_width + panel_width // 2 - len(label) * 3
            combined = add_multi_line_overlay(
                combined, [label],
                start_position=(max(x_center, 5), 6),
                font_size=16,
            )

        # Add timestep in bottom-right corner
        combined = add_multi_line_overlay(
            combined, [f"t={t}"],
            start_position=(total_width - 60, total_height - 20),
            font_size=12,
        )

        writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    writer.release()
    log.info(f"  Wrote triptych: {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Render behavior triptych ghost video")
    parser.add_argument("--height", choices=["low", "high", "both"], default="both",
                        help="Starting height condition (default: both)")
    parser.add_argument("--max-traj", type=int, default=6,
                        help="Max trajectories per panel (default: 6)")
    parser.add_argument("--max-frames", type=int, default=400,
                        help="Max frames to render (default: 400)")
    parser.add_argument("--panel-size", type=int, default=480,
                        help="Panel width and height in pixels (default: 480)")
    parser.add_argument("--fps", type=int, default=50, help="Video FPS (default: 50)")
    parser.add_argument("--cam-dist", type=float, default=0.55,
                        help="Camera distance (default: 0.55)")
    parser.add_argument("--cam-elev", type=float, default=-25.0,
                        help="Camera elevation degrees (default: -25)")
    parser.add_argument("--cam-azim", type=float, default=0.0,
                        help="Camera azimuth degrees (default: 0, front-facing)")
    parser.add_argument("--cam-fovy", type=float, default=50.0,
                        help="Camera vertical FOV degrees (default: 50)")
    args = parser.parse_args()

    # Load env from checkpoint config
    import json
    from omegaconf import OmegaConf
    from track_mjx.config import utils
    from track_mjx.agent import checkpointing
    from vnl_playground.tasks.rodent.imitation import ReferenceClips
    from moseq_env_wrapper import MoSeqImitation

    ckpt_path = str(MOSEQ_DIR / "model_checkpoints" / "260407_031233_484020")
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

    heights = ["low", "high"] if args.height == "both" else [args.height]
    for h in heights:
        log.info(f"\n=== Rendering triptych: height={h} ===")
        render_triptych(
            env, height=h,
            panel_width=args.panel_size, panel_height=args.panel_size,
            max_trajectories=args.max_traj, max_frames=args.max_frames,
            fps=args.fps,
            camera_distance=args.cam_dist, camera_elevation=args.cam_elev,
            camera_azimuth=args.cam_azim, camera_fovy=args.cam_fovy,
        )

    log.info("Done")


if __name__ == "__main__":
    main()
