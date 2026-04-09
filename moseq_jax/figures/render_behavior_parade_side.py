"""Re-render behavior parade from a side-elevated camera angle.

Reads pre-computed parade rollout data from
``outputs/moseq_behavior_parade/data/parade_rollouts.npz`` and renders the
same K-body parade from a side + slightly elevated viewpoint so that
rearing behavior is clearly visible.

Outputs:
    figures/outputs/behavior_parade_side.mp4

Usage:
    cd moseq_jax
    python figures/render_behavior_parade_side.py

    # Tweak camera:
    python figures/render_behavior_parade_side.py --cam-elev -20 --cam-azim 90
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

PARADE_DATA = MOSEQ_DIR / "outputs" / "moseq_behavior_parade" / "data" / "parade_rollouts.npz"
PARADE_META = MOSEQ_DIR / "outputs" / "moseq_behavior_parade" / "code_selection.json"

BEHAVIOR_COLORS_RGB = {
    "walk": (0, 114, 178),     # #0072B2
    "rear": (0, 158, 115),     # #009E73
    "groom": (213, 94, 0),     # #D55E00
}


def load_parade_data() -> dict:
    """Load parade rollout data and code selection metadata."""
    d = np.load(PARADE_DATA, allow_pickle=True)
    qpos_arr = d["qpos"]
    all_qpos = [np.asarray(qpos_arr[i], dtype=np.float64) for i in range(len(qpos_arr))]
    code_sequence = d["code_sequence"]

    with open(PARADE_META) as f:
        selection = json.load(f)

    # Reconstruct behavior boundaries from per-behavior code_sequence lengths.
    # The parade concatenates behaviors in config order: walk → groom → rear.
    behavior_order = ["rear", "walk", "groom"]
    boundaries = []
    offset = 0
    for beh in behavior_order:
        if beh in selection:
            beh_len = len(selection[beh]["code_sequence"])
            boundaries.append((beh, offset, offset + beh_len))
            offset += beh_len

    return {
        "all_qpos": all_qpos,
        "code_sequence": code_sequence,
        "boundaries": boundaries,
    }


def _get_behavior_rgba(beh: str) -> list[float]:
    """Return RGBA (0-1) for a behavior name."""
    mapping = {
        "walk": [0.0, 0.45, 0.70, 1.0],     # #0072B2 blue
        "groom": [0.84, 0.37, 0.0, 1.0],     # #D55E00 orange
        "rear": [0.0, 0.62, 0.45, 1.0],      # #009E73 green
    }
    return mapping.get(beh, [0.5, 0.5, 0.5, 1.0])


def _get_behavior_rgb_uint8(beh: str) -> tuple[int, int, int]:
    """Return RGB (0-255) for a behavior name."""
    mapping = {
        "walk": (0, 114, 178),
        "groom": (213, 94, 0),
        "rear": (0, 158, 115),
    }
    return mapping.get(beh, (128, 128, 128))


def render_parade_side(
    env,
    all_qpos: list[np.ndarray],
    behavior_boundaries: list[tuple[str, int, int]],
    output_path: Path,
    width: int = 720,
    height: int = 720,
    fps: int = 50,
    camera_distance: float = 1.2,
    camera_elevation: float = -20.0,
    camera_azimuth: float = 0.0,
    camera_fovy: float = 60.0,
) -> None:
    """Render parade with bodies coloured by behavior phase + gait diagram."""
    K = len(all_qpos)
    min_len = min(len(q) for q in all_qpos)

    # Build a separate ghost model for each behavior phase (different colours)
    # Since we can't change geom colours per-frame easily, we build per-phase
    # and re-render. But that's expensive. Instead, modify model.geom_rgba
    # at runtime between frames.

    # Start with a neutral colour; we'll recolour per-frame
    neutral_color = [0.5, 0.5, 0.5, 1.0]
    ghost_colors = [neutral_color] * (K - 1)

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

    # Collect all geom indices (to recolour all bodies each frame)
    n_geom = ghost_model.ngeom
    # Floor geom is typically index 0; everything else is body geoms
    floor_id = mujoco.mj_name2id(
        ghost_model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
    )

    # Round to codec-compatible dimensions
    out_w = ((width + 15) // 16) * 16
    out_h = ((height + 15) // 16) * 16

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    for t in range(min_len):
        # Determine current behavior
        current_beh = "walk"
        for beh_name, beh_start, beh_end in behavior_boundaries:
            if beh_start <= t < beh_end:
                current_beh = beh_name
                break

        # Recolour all body geoms to the current behavior colour
        rgba = _get_behavior_rgba(current_beh)
        for gi in range(n_geom):
            if gi == floor_id:
                continue
            ghost_model.geom_rgba[gi] = rgba

        # Set qpos
        data.qpos[:base_nq] = all_qpos[0][t]
        for gi in range(1, K):
            q_start = base_nq + (gi - 1) * base_nq
            data.qpos[q_start:q_start + base_nq] = all_qpos[gi][t]

        mujoco.mj_forward(ghost_model, data)

        # Track camera to mean root position of all bodies
        mean_pos = np.mean([all_qpos[i][t, :3] for i in range(K)], axis=0)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [mean_pos[0], mean_pos[1], max(mean_pos[2], 0.04)]
        cam.distance = camera_distance
        cam.elevation = camera_elevation
        cam.azimuth = camera_azimuth
        renderer.update_scene(data, camera=cam)
        frame = renderer.render().copy()

        # Pad frame to codec-compatible size if needed
        if frame.shape[0] != out_h or frame.shape[1] != out_w:
            padded = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            padded[:frame.shape[0], :frame.shape[1]] = frame
            frame = padded

        # Behavior phase label
        frame = add_multi_line_overlay(
            frame,
            [f"Behavior: {current_beh.capitalize()}  |  t={t}"],
            start_position=(10, 10),
            font_size=18,
        )

        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    renderer.close()
    writer.release()
    log.info(f"  Saved side-view parade: {output_path} ({K} bodies, {min_len} frames)")


def main():
    parser = argparse.ArgumentParser(description="Re-render behavior parade from side view")
    parser.add_argument("--width", type=int, default=1280, help="Frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Frame height (default: 720)")
    parser.add_argument("--fps", type=int, default=50, help="Video FPS (default: 50)")
    parser.add_argument("--cam-dist", type=float, default=1.4,
                        help="Camera distance (default: 1.4)")
    parser.add_argument("--cam-elev", type=float, default=-35.0,
                        help="Camera elevation degrees (default: -35, tilted up looking down)")
    parser.add_argument("--cam-azim", type=float, default=0.0,
                        help="Camera azimuth degrees (default: 0, front-facing)")
    parser.add_argument("--cam-fovy", type=float, default=50.0,
                        help="Camera vertical FOV degrees (default: 50)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load env ─────────────────────────────────────────────────────
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

    # ── Load parade data ─────────────────────────────────────────────
    log.info("Loading parade rollout data...")
    parade = load_parade_data()
    log.info(f"  {len(parade['all_qpos'])} bodies, "
             f"{min(len(q) for q in parade['all_qpos'])} frames, "
             f"{len(parade['boundaries'])} behavior phases")
    for beh, start, end in parade["boundaries"]:
        log.info(f"    {beh}: frames {start}–{end}")

    # ── Render ───────────────────────────────────────────────────────
    output_path = OUTPUT_DIR / "behavior_parade_side.mp4"
    render_parade_side(
        env,
        parade["all_qpos"],
        parade["boundaries"],
        output_path=output_path,
        width=args.width,
        height=args.height,
        fps=args.fps,
        camera_distance=args.cam_dist,
        camera_elevation=args.cam_elev,
        camera_azimuth=args.cam_azim,
        camera_fovy=args.cam_fovy,
    )

    log.info("Done")


if __name__ == "__main__":
    main()
