"""Three-panel behavior motion trail: Walk | Groom | Rear.

Each panel renders multiple frozen copies of the rodent at evenly-spaced
timesteps with a colored trajectory trace line and a gait cycle timeline
bar (Aidan's trail style).

Data: figures/data/killer_{walk,groom,rear}_{low,high}.npz

Usage:
    cd moseq_jax/figures
    python render_behavior_triptych_trail.py
    python render_behavior_triptych_trail.py --height high --n-copies 7
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

from vnl_playground.tasks import utils as vnl_utils
from vqvae_jax.ablation.run_divergent_futures import _disable_lights_recursive

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

BEHAVIORS = ["walk", "groom", "rear"]
BEHAVIOR_LABELS = {"walk": "Walk", "groom": "Groom", "rear": "Rear"}

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

# Limb labels and colors for gait diagram
LIMBS = ["FL", "FR", "HL", "HR"]
LIMB_LABELS = {"FL": "FL", "FR": "FR", "HL": "HL", "HR": "HR"}
LIMB_FOOT_BODY = {"HL": "toe_L", "HR": "toe_R", "FL": "finger_L", "FR": "finger_R"}
STANCE_THRESHOLD = 0.005  # Z below this = foot on ground


# ── Helpers ─────────────────────────────────────────────────────────────────


def compute_foot_contacts(
    env,
    qpos_traj: np.ndarray,
    threshold: float = STANCE_THRESHOLD,
) -> dict[str, np.ndarray]:
    """Compute per-limb stance (True) / swing (False) from FK.

    Returns:
        {limb_name: bool array [T]} where True = stance (foot on ground).
    """
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


# ── Trail model construction ────────────────────────────────────────────────


def build_trail_model(
    env,
    num_copies: int,
    trace_positions: np.ndarray,
    trace_rgba: list[float],
    trace_radius: float = 0.003,
    camera_lookat: np.ndarray | None = None,
    camera_distance: float = 0.6,
    camera_elevation: float = -20.0,
    camera_azimuth: float = 0.0,
    camera_fovy: float = 45.0,
) -> tuple[mujoco.MjModel, int]:
    """Build multi-rodent trail model with trace line and fixed camera."""
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

    for j in range(len(trace_positions) - 1):
        p1 = trace_positions[j].tolist()
        p2 = trace_positions[j + 1].tolist()
        if np.linalg.norm(trace_positions[j + 1] - trace_positions[j]) < 1e-6:
            continue
        spec.worldbody.add_geom(
            name=f"trace_{j}",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[*p1, *p2],
            size=[trace_radius, 0, 0],
            rgba=trace_rgba,
            contype=0, conaffinity=0,
        )

    if camera_lookat is None:
        camera_lookat = trace_positions.mean(axis=0)
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
    cam_quat = [0.25 / s, (R[2, 1] - R[1, 2]) * s,
                (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s]

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
    width: int = 1024,
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
        description="Render three-panel behavior motion trail"
    )
    parser.add_argument("--height", choices=["low", "high"], default="low",
                        help="Starting height condition (default: low)")
    parser.add_argument("--n-copies", type=int, default=7,
                        help="Number of frozen rodent copies per panel")
    parser.add_argument("--trace-step", type=int, default=3,
                        help="Trace line resolution: connect every N frames")
    parser.add_argument("--max-frames", type=int, default=400,
                        help="Max frames to use per trajectory")
    parser.add_argument("--panel-w", type=int, default=1024,
                        help="Panel render width in pixels")
    parser.add_argument("--panel-h", type=int, default=768,
                        help="Panel render height in pixels")
    parser.add_argument("--cam-dist", type=float, default=0.0,
                        help="Camera distance (0 = auto)")
    parser.add_argument("--cam-elev", type=float, default=-25.0)
    parser.add_argument("--cam-azim", type=float, default=0.0,
                        help="Camera azimuth (default: 0, front-facing)")
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

    # ── Load trajectories and render ─────────────────────────────────
    panel_data = {}  # beh -> {"frame": ndarray, "trail_indices": ndarray, "n_frames": int}

    for beh in BEHAVIORS:
        data_path = DATA_DIR / f"killer_{beh}_{args.height}.npz"
        if not data_path.exists():
            log.warning(f"Missing: {data_path}")
            continue
        d = np.load(data_path, allow_pickle=True)
        all_qpos = [
            np.asarray(d["qpos"][i], dtype=np.float64)
            for i in range(len(d["qpos"]))
        ]
        # Select best trajectory per behavior:
        #   walk: highest XY path length (clearest locomotion)
        #   rear: highest Z rise (clearest rearing)
        #   groom: lowest XY path length (most stationary)
        def _xy_path(q):
            return float(np.sum(np.linalg.norm(np.diff(q[:, :2], axis=0), axis=1)))
        def _z_rise(q):
            return float(q[:, 2].max() - q[0, 2])
        if beh == "walk":
            all_qpos.sort(key=_xy_path, reverse=True)
        elif beh == "rear":
            all_qpos.sort(key=_z_rise, reverse=True)
        else:  # groom — stationary but upright (highest z while low xy)
            all_qpos.sort(key=lambda q: float(q[:, 2].mean()), reverse=True)
        traj = all_qpos[0][:args.max_frames]
        log.info(f"  {beh}: selected best trajectory (xy_path={_xy_path(traj):.3f}m, z_rise={_z_rise(traj):.3f}m)")
        n_frames = len(traj)
        log.info(f"  {beh}: {n_frames} frames")

        trail_indices = np.linspace(
            0, n_frames - 1, args.n_copies
        ).round().astype(int)

        trace_timesteps = np.arange(
            trail_indices[0], trail_indices[-1] + 1, args.trace_step
        )
        trace_positions = traj[trace_timesteps, :3].copy()
        trace_positions[:, 2] += 0.005

        trace_rgb = BEHAVIOR_RGB[beh]
        trace_rgba = [*trace_rgb, 0.85]

        # Auto-compute camera distance
        positions = traj[trail_indices, :3]
        extent = positions.max(axis=0) - positions.min(axis=0)
        max_span = max(extent[0], extent[1], extent[2])
        fovy_rad = math.radians(args.cam_fovy)
        cam_dist = max(
            (max_span + 0.25) / (2 * math.tan(fovy_rad / 2)),
            0.35,
        )
        if args.cam_dist > 0:
            cam_dist = args.cam_dist
        log.info(f"    extent={max_span:.3f}m, cam_dist={cam_dist:.3f}m")

        trail_model, base_nq = build_trail_model(
            env,
            num_copies=args.n_copies,
            trace_positions=trace_positions,
            trace_rgba=trace_rgba,
            camera_distance=cam_dist,
            camera_elevation=args.cam_elev,
            camera_azimuth=args.cam_azim,
            camera_fovy=args.cam_fovy,
        )

        # Recolor all copies to behavior color
        recolor_copies(trail_model, [BEHAVIOR_RGBA[beh]] * args.n_copies)

        copy_qpos = [traj[t] for t in trail_indices]
        frame = render_trail_frame(
            trail_model, base_nq, copy_qpos,
            width=args.panel_w, height=args.panel_h,
        )
        # Compute foot contacts
        log.info(f"    Computing foot contacts...")
        contacts = compute_foot_contacts(env, traj)

        panel_data[beh] = {
            "frame": frame,
            "trail_indices": trail_indices,
            "n_frames": n_frames,
            "contacts": contacts,
        }
        log.info(f"    Rendered trail: {frame.shape}")

    if not panel_data:
        log.error("No panels rendered")
        return

    # ── Assemble figure: trail + gait diagram per panel ──────────────
    n_cols = len(panel_data)
    n_limbs = len(LIMBS)
    fig = plt.figure(figsize=(4.8 * n_cols, 6.5))
    gs = gridspec.GridSpec(
        2, n_cols, height_ratios=[10, 3.5],
        hspace=0.02, wspace=0.08,
    )

    for ci, beh in enumerate(BEHAVIORS):
        if beh not in panel_data:
            continue
        pd = panel_data[beh]
        color = BEHAVIOR_RGB[beh]
        total_sec = pd["n_frames"] / FPS

        # Trail image
        ax_img = fig.add_subplot(gs[0, ci])
        ax_img.imshow(pd["frame"])
        ax_img.set_title(
            BEHAVIOR_LABELS[beh], fontsize=14, fontweight="bold",
            color=color, pad=6,
        )
        ax_img.set_axis_off()

        # Gait diagram: one row per limb, filled when stance
        ax_gait = fig.add_subplot(gs[1, ci])
        contacts = pd["contacts"]
        time_axis = np.arange(pd["n_frames"]) / FPS

        for li, limb in enumerate(LIMBS):
            stance = contacts[limb]
            # Draw stance blocks as filled spans
            in_stance = False
            start_t = 0
            for t in range(len(stance)):
                if stance[t] and not in_stance:
                    start_t = t
                    in_stance = True
                elif not stance[t] and in_stance:
                    ax_gait.fill_between(
                        [start_t / FPS, t / FPS],
                        li - 0.4, li + 0.4,
                        color=color, alpha=0.7, edgecolor="none",
                    )
                    in_stance = False
            if in_stance:
                ax_gait.fill_between(
                    [start_t / FPS, pd["n_frames"] / FPS],
                    li - 0.4, li + 0.4,
                    color=color, alpha=0.7, edgecolor="none",
                )

        # Copy position markers
        for t in pd["trail_indices"]:
            ax_gait.axvline(
                t / FPS, color="black", linewidth=0.8, alpha=0.4,
                linestyle="--", zorder=1,
            )

        ax_gait.set_xlim(0, total_sec)
        ax_gait.set_ylim(-0.6, n_limbs - 0.4)
        ax_gait.set_yticks(range(n_limbs))
        ax_gait.set_yticklabels(LIMBS, fontsize=7.5, fontweight="bold")
        ax_gait.set_xlabel("Time (s)", fontsize=7)
        ax_gait.tick_params(axis="x", labelsize=6)
        ax_gait.tick_params(axis="y", length=0)
        ax_gait.spines["top"].set_visible(False)
        ax_gait.spines["right"].set_visible(False)
        ax_gait.spines["left"].set_visible(False)
        ax_gait.invert_yaxis()

    out_base = OUTPUT_DIR / f"behavior_triptych_trail_{args.height}"
    for ext in (".pdf", ".png"):
        fig.savefig(
            str(out_base) + ext, dpi=300, bbox_inches="tight",
            facecolor="white", pad_inches=0.02,
        )
    plt.close(fig)
    log.info(f"Saved: {out_base}.pdf / .png")


if __name__ == "__main__":
    main()
