"""Behavior transition parade as a motion trail: walk -> groom -> rear.

Renders one body performing the consecutive walk -> groom -> rear transition
with frozen copies colored by behavior phase, trajectory trace line, and
a gait cycle timeline bar.

Data: outputs/moseq_behavior_parade/data/parade_rollouts.npz

Usage:
    cd moseq_jax/figures
    python render_behavior_parade_trail.py
    python render_behavior_parade_trail.py --n-copies-per-beh 4
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import mujoco
import numpy as np

SCRIPT_DIR = Path(__file__).parent
MOSEQ_DIR = SCRIPT_DIR.parent
REPO_ROOT = MOSEQ_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "behavior_parade_trail"

for _p in (str(MOSEQ_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vnl_playground.tasks import utils as vnl_utils
from vqvae_jax.ablation.run_divergent_futures import _disable_lights_recursive

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

PARADE_DATA = MOSEQ_DIR / "outputs" / "moseq_behavior_parade" / "data" / "parade_rollouts.npz"
PARADE_META = MOSEQ_DIR / "outputs" / "moseq_behavior_parade" / "code_selection.json"

BEHAVIOR_ORDER = ["rear", "walk", "groom"]
BEHAVIOR_LABELS = {"walk": "Walk", "groom": "Immobility", "rear": "Rear"}

# Behavior colors (matching experiments/shared/plotting.py)
BEHAVIOR_RGB = {
    "walk": (0.836, 0.369, 0.0),      # #D55E00 orange
    "groom": (0.0, 0.447, 0.698),     # #0072B2 blue
    "rear": (0.0, 0.620, 0.451),      # #009E73 green
}
BEHAVIOR_RGBA = {
    beh: [*rgb, 1.0] for beh, rgb in BEHAVIOR_RGB.items()
}
FPS = 50

LIMBS = ["FL", "FR", "HL", "HR"]
LIMB_FOOT_BODY = {"HL": "toe_L", "HR": "toe_R", "FL": "finger_L", "FR": "finger_R"}
STANCE_THRESHOLD = 0.005


# ── Helpers ─────────────────────────────────────────────────────────────────


def compute_foot_contacts(
    env,
    qpos_traj: np.ndarray,
    threshold: float = STANCE_THRESHOLD,
) -> dict[str, np.ndarray]:
    """Compute per-limb stance (True) / swing (False) from FK."""
    model = env.mj_model
    data = mujoco.MjData(model)
    suffix = getattr(env, "_suffix", "")

    foot_ids = {}
    for limb, body_name in LIMB_FOOT_BODY.items():
        bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, body_name + suffix
        )
        if bid < 0:
            bid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
        foot_ids[limb] = bid

    T = len(qpos_traj)
    contacts = {limb: np.zeros(T, dtype=bool) for limb in LIMBS}
    for t in range(T):
        data.qpos[:] = qpos_traj[t]
        mujoco.mj_forward(model, data)
        for limb, bid in foot_ids.items():
            contacts[limb][t] = data.xpos[bid, 2] < threshold

    return contacts


def recolor_copies(
    model: mujoco.MjModel,
    copy_rgbas: list[list[float]],
) -> None:
    """Recolor each copy's body geoms. copy_rgbas[i] applies to suffix -t{i}."""
    for gi in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi)
        if not name:
            continue
        for ci, rgba in enumerate(copy_rgbas):
            if f"-t{ci}" in name:
                model.geom_rgba[gi] = rgba
                break


def timestep_to_behavior(
    t: int,
    boundaries: list[tuple[str, int, int]],
) -> str:
    """Return behavior name for a given timestep."""
    for beh, start, end in boundaries:
        if start <= t < end:
            return beh
    return boundaries[-1][0]


# ── Trail model construction ────────────────────────────────────────────────


def build_parade_trail_model(
    env,
    num_copies: int,
    trace_segments: list[tuple[np.ndarray, list[float]]],
    trace_radius: float = 0.003,
    camera_lookat: np.ndarray | None = None,
    camera_distance: float = 1.0,
    camera_elevation: float = -25.0,
    camera_azimuth: float = -135.0,
    camera_fovy: float = 45.0,
) -> tuple[mujoco.MjModel, int]:
    """Build trail model with multi-colored trace line for behavior phases."""
    from vnl_playground.tasks.rodent import consts as rodent_consts

    arena_xml = str(rodent_consts.ARENA_XML_PATH)
    walker_path = str(env._walker_xml_path)
    rescale = env.reference_clips._config["model"]["SCALE_FACTOR"]

    spec = mujoco.MjSpec.from_file(arena_xml)

    for i in range(num_copies):
        walker = mujoco.MjSpec.from_file(walker_path)
        if rescale != 1.0:
            walker = vnl_utils.dm_scale_spec(walker, rescale)
        if i > 0:
            _disable_lights_recursive(walker.worldbody)
        frame = spec.worldbody.add_frame(pos=(0, 0, 0), quat=(1, 0, 0, 0))
        gb = frame.attach_body(walker.body("walker"), "", suffix=f"-t{i}")
        gb.add_freejoint(name=f"root-t{i}")

    seg_idx = 0
    for positions, rgba in trace_segments:
        for j in range(len(positions) - 1):
            p1 = positions[j].tolist()
            p2 = positions[j + 1].tolist()
            if np.linalg.norm(positions[j + 1] - positions[j]) < 1e-6:
                continue
            spec.worldbody.add_geom(
                name=f"trace_{seg_idx}",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=[*p1, *p2],
                size=[trace_radius, 0, 0],
                rgba=rgba,
                contype=0, conaffinity=0,
            )
            seg_idx += 1

    all_positions = np.concatenate([p for p, _ in trace_segments], axis=0)
    if camera_lookat is None:
        camera_lookat = all_positions.mean(axis=0)
        camera_lookat[2] = max(camera_lookat[2], 0.06)

    el_rad = np.radians(camera_elevation)
    az_rad = np.radians(camera_azimuth)
    cam_pos = camera_lookat + np.array([
        camera_distance * np.cos(el_rad) * np.cos(az_rad),
        camera_distance * np.cos(el_rad) * np.sin(az_rad),
        -camera_distance * np.sin(el_rad),
    ])

    forward = camera_lookat - cam_pos
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)
    R = np.stack([right, up, -forward], axis=1)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    s = 0.5 / np.sqrt(tr + 1.0)
    cam_quat = [
        0.25 / s, (R[2, 1] - R[1, 2]) * s,
        (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s,
    ]

    spec.worldbody.add_camera(
        name="trail_cam", pos=cam_pos.tolist(),
        quat=cam_quat, fovy=camera_fovy,
    )

    overhead = spec.worldbody.add_light(name="trail_overhead")
    overhead.pos = [camera_lookat[0], camera_lookat[1], 3.0]
    overhead.dir = [0, 0, -1]
    overhead.diffuse = [0.8, 0.8, 0.8]
    overhead.specular = [0.2, 0.2, 0.2]

    model = spec.compile()
    base_nq = model.nq // num_copies

    for i in range(model.nlight):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_LIGHT, i)
        if name and "tracking_light" in name:
            model.light_diffuse[i] = [0, 0, 0]
            model.light_specular[i] = [0, 0, 0]

    return model, base_nq


def render_trail_frame(
    trail_model: mujoco.MjModel,
    base_nq: int,
    copy_qpos_list: list[np.ndarray],
    width: int = 2048,
    height: int = 768,
) -> np.ndarray:
    """Set qpos for each copy and render a single frame."""
    trail_model.vis.global_.offwidth = width
    trail_model.vis.global_.offheight = height
    data = mujoco.MjData(trail_model)
    renderer = mujoco.Renderer(trail_model, height=height, width=width)

    for i, qp in enumerate(copy_qpos_list):
        start = i * base_nq
        data.qpos[start:start + base_nq] = qp

    mujoco.mj_forward(trail_model, data)
    renderer.update_scene(data, camera="trail_cam")
    frame = renderer.render().copy()
    renderer.close()
    return frame


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Render behavior transition parade as motion trail"
    )
    parser.add_argument("--body-idx", type=int, default=0,
                        help="Which body from the parade to use (default: 0)")
    parser.add_argument("--n-copies-per-beh", type=int, default=4,
                        help="Frozen copies per behavior phase")
    parser.add_argument("--trace-step", type=int, default=3,
                        help="Trace line resolution: connect every N frames")
    parser.add_argument("--width", type=int, default=2048,
                        help="Render width in pixels")
    parser.add_argument("--height", type=int, default=768,
                        help="Render height in pixels")
    parser.add_argument("--cam-dist", type=float, default=0.0,
                        help="Camera distance (0 = auto)")
    parser.add_argument("--cam-elev", type=float, default=-25.0)
    parser.add_argument("--cam-azim", type=float, default=-135.0)
    parser.add_argument("--cam-fovy", type=float, default=45.0)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    # ── Load parade data ─────────────────────────────────────────────
    d = np.load(PARADE_DATA, allow_pickle=True)
    qpos = np.asarray(d["qpos"][args.body_idx], dtype=np.float64)
    log.info(f"Parade body {args.body_idx}: {qpos.shape}")

    with open(PARADE_META) as f:
        selection = json.load(f)

    boundaries = []
    offset = 0
    for beh in BEHAVIOR_ORDER:
        if beh in selection:
            beh_len = len(selection[beh]["code_sequence"])
            boundaries.append((beh, offset, offset + beh_len))
            offset += beh_len

    for beh, start, end in boundaries:
        log.info(f"  {beh}: frames {start}-{end}")

    # ── Compute copy timesteps and trace segments ────────────────────
    all_trail_indices = []
    trace_segments = []

    for beh, beh_start, beh_end in boundaries:
        phase_indices = np.linspace(
            beh_start, beh_end - 1, args.n_copies_per_beh
        ).round().astype(int)
        all_trail_indices.extend(phase_indices.tolist())

        trace_timesteps = np.arange(beh_start, beh_end, args.trace_step)
        trace_pos = qpos[trace_timesteps, :3].copy()
        trace_pos[:, 2] += 0.005
        trace_rgb = BEHAVIOR_RGB[beh]
        trace_segments.append((trace_pos, [*trace_rgb, 0.85]))

    total_copies = len(all_trail_indices)
    log.info(f"Total copies: {total_copies} at timesteps: {all_trail_indices}")

    # Auto-compute camera distance
    positions = qpos[all_trail_indices, :3]
    extent = positions.max(axis=0) - positions.min(axis=0)
    max_span = max(extent[0], extent[1])
    fovy_rad = math.radians(args.cam_fovy)
    cam_dist = max(
        (max_span + 0.2) / (2 * math.tan(fovy_rad / 2)),
        0.35,
    )
    if args.cam_dist > 0:
        cam_dist = args.cam_dist
    log.info(f"  extent={max_span:.3f}m, cam_dist={cam_dist:.3f}m")

    # ── Build trail model and render ─────────────────────────────────
    trail_model, base_nq = build_parade_trail_model(
        env,
        num_copies=total_copies,
        trace_segments=trace_segments,
        camera_distance=cam_dist,
        camera_elevation=args.cam_elev,
        camera_azimuth=args.cam_azim,
        camera_fovy=args.cam_fovy,
    )

    # Recolor each copy by its behavior phase
    copy_colors = []
    for t in all_trail_indices:
        beh = timestep_to_behavior(t, boundaries)
        copy_colors.append(BEHAVIOR_RGBA[beh])
    recolor_copies(trail_model, copy_colors)

    copy_qpos = [qpos[t] for t in all_trail_indices]
    frame = render_trail_frame(
        trail_model, base_nq, copy_qpos,
        width=args.width, height=args.height,
    )
    log.info(f"Rendered trail: {frame.shape}")

    # ── Compute foot contacts ────────────────────────────────────────
    log.info("Computing foot contacts...")
    contacts = compute_foot_contacts(env, qpos)

    # ── Assemble figure: behavior bar + trail + gait diagram ─────────
    total_frames = sum(end - start for _, start, end in boundaries)
    total_sec = total_frames / FPS
    n_limbs = len(LIMBS)

    fig = plt.figure(figsize=(16, 8.5))
    gs = gridspec.GridSpec(
        3, 1, height_ratios=[1, 14, 4],
        hspace=0.05,
    )

    # Top: behavior phase bar
    ax_bar_top = fig.add_subplot(gs[0])
    for beh, beh_start, beh_end in boundaries:
        color = BEHAVIOR_RGB[beh]
        ax_bar_top.barh(
            0, (beh_end - beh_start) / FPS, left=beh_start / FPS,
            height=1, color=color, edgecolor="none",
        )
        ax_bar_top.text(
            (beh_start + beh_end) / 2 / FPS, 0,
            BEHAVIOR_LABELS[beh], ha="center", va="center",
            fontsize=11, fontweight="bold", color="white",
        )
    ax_bar_top.set_xlim(0, total_sec)
    ax_bar_top.set_ylim(-0.5, 0.5)
    ax_bar_top.set_yticks([])
    ax_bar_top.set_xticks([])
    for sp in ax_bar_top.spines.values():
        sp.set_visible(False)

    # Middle: trail render
    ax_img = fig.add_subplot(gs[1])
    ax_img.imshow(frame)
    ax_img.set_axis_off()

    # Bottom: real gait diagram (stance/swing per limb)
    ax_gait = fig.add_subplot(gs[2])

    for li, limb in enumerate(LIMBS):
        stance = contacts[limb]
        in_stance = False
        start_t = 0
        for t in range(len(stance)):
            # Get behavior color at this timestep
            if stance[t] and not in_stance:
                start_t = t
                in_stance = True
            elif not stance[t] and in_stance:
                beh = timestep_to_behavior(start_t, boundaries)
                color = BEHAVIOR_RGB[beh]
                ax_gait.fill_between(
                    [start_t / FPS, t / FPS],
                    li - 0.4, li + 0.4,
                    color=color, alpha=0.7, edgecolor="none",
                )
                in_stance = False
        if in_stance:
            beh = timestep_to_behavior(start_t, boundaries)
            color = BEHAVIOR_RGB[beh]
            ax_gait.fill_between(
                [start_t / FPS, total_frames / FPS],
                li - 0.4, li + 0.4,
                color=color, alpha=0.7, edgecolor="none",
            )

    # Behavior phase boundaries
    for _, beh_start, beh_end in boundaries:
        if beh_start > 0:
            ax_gait.axvline(
                beh_start / FPS, color="black", linewidth=0.8,
                linestyle="--", alpha=0.5,
            )

    # Copy position markers
    for t in all_trail_indices:
        ax_gait.axvline(
            t / FPS, color="black", linewidth=0.5, alpha=0.3,
            linestyle=":",
        )

    ax_gait.set_xlim(0, total_sec)
    ax_gait.set_ylim(-0.6, n_limbs - 0.4)
    ax_gait.set_yticks(range(n_limbs))
    ax_gait.set_yticklabels(LIMBS, fontsize=7)
    ax_gait.set_xlabel("Time (s)", fontsize=9)
    ax_gait.tick_params(axis="x", labelsize=7)
    ax_gait.tick_params(axis="y", length=0)
    ax_gait.spines["top"].set_visible(False)
    ax_gait.spines["right"].set_visible(False)
    ax_gait.spines["left"].set_visible(False)
    ax_gait.invert_yaxis()

    out_base = OUTPUT_DIR / "behavior_parade_trail"
    for ext in (".pdf", ".png"):
        fig.savefig(
            str(out_base) + ext, dpi=300, bbox_inches="tight",
            facecolor="white", pad_inches=0.02,
        )
    plt.close(fig)
    log.info(f"Saved: {out_base}.pdf / .png")


if __name__ == "__main__":
    main()
