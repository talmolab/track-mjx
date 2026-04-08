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
    behavior_order = ["walk", "groom", "rear"]
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


def render_parade_side(
    env,
    all_qpos: list[np.ndarray],
    behavior_boundaries: list[tuple[str, int, int]],
    output_path: Path,
    width: int = 1280,
    height: int = 720,
    fps: int = 50,
    camera_distance: float = 1.2,
    camera_elevation: float = -20.0,
    camera_azimuth: float = 90.0,
    camera_fovy: float = 60.0,
) -> None:
    """Render parade from a side-elevated camera that shows rearing clearly."""
    import matplotlib.pyplot as plt

    K = len(all_qpos)
    min_len = min(len(q) for q in all_qpos)

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

    # Round to codec-compatible dimensions
    out_w = ((width + 15) // 16) * 16
    out_h = ((height + 15) // 16) * 16

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    for t in range(min_len):
        data.qpos[:base_nq] = all_qpos[0][t]
        for gi in range(1, K):
            q_start = base_nq + (gi - 1) * base_nq
            data.qpos[q_start:q_start + base_nq] = all_qpos[gi][t]

        mujoco.mj_forward(ghost_model, data)
        renderer.update_scene(data, camera="divergent_cam")
        frame = renderer.render().copy()

        # Pad frame to codec-compatible size if needed
        if frame.shape[0] != out_h or frame.shape[1] != out_w:
            padded = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            padded[:frame.shape[0], :frame.shape[1]] = frame
            frame = padded

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

        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    renderer.close()
    writer.release()
    log.info(f"  Saved side-view parade: {output_path} ({K} bodies, {min_len} frames)")


def main():
    parser = argparse.ArgumentParser(description="Re-render behavior parade from side view")
    parser.add_argument("--width", type=int, default=720, help="Frame width (default: 720)")
    parser.add_argument("--height", type=int, default=720, help="Frame height (default: 720)")
    parser.add_argument("--fps", type=int, default=50, help="Video FPS (default: 50)")
    parser.add_argument("--cam-dist", type=float, default=1.2,
                        help="Camera distance (default: 1.2)")
    parser.add_argument("--cam-elev", type=float, default=-20.0,
                        help="Camera elevation degrees (default: -20, slightly above)")
    parser.add_argument("--cam-azim", type=float, default=0.0,
                        help="Camera azimuth degrees (default: 0, front-facing)")
    parser.add_argument("--cam-fovy", type=float, default=60.0,
                        help="Camera vertical FOV degrees (default: 60)")
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
