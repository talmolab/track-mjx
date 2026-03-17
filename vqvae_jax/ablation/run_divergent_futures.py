"""Divergent Futures Experiment for VQ-VAE D0 Code Ablation.

Demonstrates that D0 codes encode categorically different motor plans
by finding clips with similar initial poses but divergent futures, then
testing 3 decoder-only conditions:
  A. Correct codes (from H5 inference)
  B. Random step-wise excluded codes (no overlap with correct at each step)
  C. Random trajectory-wise excluded codes (globally disjoint code set)

Renders ghost-body overlays showing K trajectories simultaneously.

Usage:
    cd vqvae_jax
    WANDB_MODE=offline python -m ablation.run_divergent_futures
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import base64
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent
VQVAE_DIR = SCRIPT_DIR.parent
REPO_ROOT = VQVAE_DIR.parent
sys.path.insert(0, str(VQVAE_DIR))
sys.path.insert(0, str(REPO_ROOT))

import hydra
import imageio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from absl import logging
from omegaconf import DictConfig
from vnl_playground.tasks import utils as vnl_utils
from vnl_playground.tasks.rodent import imitation
from vnl_playground.tasks.rodent.reference_clips import ReferenceClips

from track_mjx.config import utils as config_utils

from analysis.checkpoint_utils import (
    get_all_codebooks,
    load_vq_checkpoint,
)
from analysis.code_analysis import load_rollouts_from_h5
from analysis.inference_cache import InferenceResult
from analysis.rendering import (
    add_multi_line_overlay,
    get_nature_colormap,
)

from ablation.run_ablation import (
    make_decoder_only_step_fn,
    subset_clips,
    zero_continuous_encoder_params,
)

# =============================================================================
# GHOST MODEL CONSTRUCTION
# =============================================================================


def _disable_lights_recursive(body: Any) -> None:
    """Disable all lights in a body tree to prevent wash-out with many ghosts."""
    for light in body.lights:
        light.active = 0
    for child in body.bodies:
        _disable_lights_recursive(child)


def build_ghost_model(
    env: imitation.Imitation,
    num_ghosts: int,
    ghost_colors: list[list[float]],
    camera_distance: float = 0.8,
    camera_elevation: float = -30.0,
    camera_azimuth: float = 135.0,
    camera_fovy: float = 60.0,
) -> tuple[Any, int]:
    """Build a MuJoCo model with ghost bodies for overlaid rendering.

    Adds a fixed "divergent_cam" to the worldbody for a 3/4 angled view
    that captures diverging trajectories better than body-tracking cameras.

    Args:
        env: Imitation environment with _spec and _walker_xml_path.
        num_ghosts: Number of ghost bodies to add.
        ghost_colors: RGBA color per ghost, shape [num_ghosts, 4].
        camera_distance: Distance from the origin.
        camera_elevation: Elevation angle in degrees (negative = looking down).
        camera_azimuth: Azimuth angle in degrees.
        camera_fovy: Vertical field of view in degrees.

    Returns:
        Tuple of (compiled mj_model, base_nq) where base_nq is the
        qpos dimensionality of the original body.
    """
    spec = env._spec.copy()
    walker_path = str(env._walker_xml_path)
    rescale = env.reference_clips._config["model"]["SCALE_FACTOR"]

    for gi in range(num_ghosts):
        ghost = mujoco.MjSpec.from_file(walker_path)
        if rescale != 1.0:
            ghost = vnl_utils.dm_scale_spec(ghost, rescale)
        for body in ghost.worldbody.bodies:
            vnl_utils._recolour_tree(body, rgba=ghost_colors[gi])
        # Disable lights on ghost bodies to prevent scene wash-out with many ghosts.
        _disable_lights_recursive(ghost.worldbody)
        frame = spec.worldbody.add_frame(pos=(0, 0, 0.05), quat=(1, 0, 0, 0))
        gb = frame.attach_body(ghost.worldbody, "", suffix=f"-ghost{gi}")
        gb.add_freejoint()

    # Add fixed world camera for diverging trajectory overview.
    # Compute position from spherical coords and orientation via look-at.
    el_rad = np.radians(camera_elevation)
    az_rad = np.radians(camera_azimuth)
    cam_pos = np.array(
        [
            camera_distance * np.cos(el_rad) * np.cos(az_rad),
            camera_distance * np.cos(el_rad) * np.sin(az_rad),
            -camera_distance * np.sin(el_rad) + 0.1,
        ]
    )
    look_target = np.array([0.0, 0.0, 0.06])

    # MuJoCo camera looks along -Z in its local frame.
    # Build rotation matrix [right, up, -forward] then convert to quat.
    forward = look_target - cam_pos
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)
    # Rotation matrix columns: right, up, -forward (MuJoCo convention)
    R = np.stack([right, up, -forward], axis=1)
    # Convert rotation matrix to quaternion (w, x, y, z)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    cam_quat = [float(w), float(x), float(y), float(z)]

    spec.worldbody.add_camera(
        name="divergent_cam",
        pos=cam_pos.tolist(),
        quat=cam_quat,
        fovy=camera_fovy,
    )

    mj_model = spec.compile()
    base_nq = env.mj_model.nq
    return mj_model, base_nq


# Default camera name for ghost rendering (fixed world camera).
GHOST_CAMERA = "divergent_cam"


# =============================================================================
# CLIP SELECTION — FIND DIVERGENT FUTURES
# =============================================================================


def _score_clip_for_posture(
    qpos: np.ndarray,
    posture: str,
    search_window: int,
) -> tuple[int, float]:
    """Return (characteristic_frame, score) for posture-based clip selection.

    Args:
        qpos: Clip qpos array, shape [T, nq].
        posture: One of "rearing", "walking", "grooming", "sustained_rearing".
        search_window: Number of frames to search within.

    Returns:
        Tuple of (frame_index, score). Higher score = better match.
    """
    window = min(search_window, len(qpos))
    if posture == "rearing":
        # Peak root Z height
        root_z = qpos[:window, 2]
        frame = int(np.argmax(root_z))
        score = float(root_z[frame])
    elif posture == "walking":
        # Find peak velocity frame in search window, then score by
        # total XY displacement from that frame onwards (matching
        # the code extraction range). This ensures selected clips
        # walk throughout, not just briefly.
        diffs_window = np.diff(qpos[:window, :2], axis=0)
        xy_vel_window = np.linalg.norm(diffs_window, axis=1)
        frame = int(np.argmax(xy_vel_window)) + 1
        frame = min(frame, window - 1)
        diffs_full = np.diff(qpos[frame:, :2], axis=0)
        score = float(np.sum(np.linalg.norm(diffs_full, axis=1)))
    elif posture == "grooming":
        # Frame 0; score = negative total motion over FULL clip
        # (want clips that stay still throughout, not just initially).
        frame = 0
        diffs = np.diff(qpos[:, :2], axis=0)
        xy_disp = float(np.sum(np.linalg.norm(diffs, axis=1)))
        z_range = float(np.max(qpos[:, 2]) - np.min(qpos[:, 2]))
        score = -(xy_disp + z_range)
    elif posture == "sustained_rearing":
        # Sustained high root Z: start from peak frame, score by mean Z
        # from that frame onwards (matching the code extraction range).
        # Also penalize Z variance to prefer clips that hold position
        # steadily rather than bobbing up and down.
        root_z = qpos[:, 2]
        frame = int(np.argmax(root_z[:window]))
        z_from_peak = root_z[frame:]
        mean_z = float(np.mean(z_from_peak))
        std_z = float(np.std(z_from_peak))
        score = mean_z - std_z  # high mean, low variance = sustained
    else:
        raise ValueError(f"Unknown posture: {posture}")
    return frame, score


def find_divergent_clips_by_posture(
    rollouts: list[InferenceResult],
    K: int,
    cfg: DictConfig,
    posture: str = "rearing",
) -> dict[str, Any]:
    """Find K clips with similar initial poses but divergent D0 code futures.

    Uses posture-specific scoring to find characteristic frames:
    - rearing: peak root-Z height
    - walking: peak XY velocity / total displacement
    - grooming: minimal motion (low XY displacement + Z range)

    Args:
        rollouts: Loaded inference results with qpos and rvq_indices.
        K: Number of clips to select.
        cfg: Experiment config with pose_selection parameters.
        posture: One of "rearing", "walking", "grooming".

    Returns:
        Dict with keys: anchor_idx, anchor_frame, clip_indices (list of K),
        start_frames (list of K), posture_scores, pairwise_divergence.
    """
    sel = cfg.experiment.pose_selection
    search_window = int(sel.search_window_frames)
    joint_threshold = float(sel.joint_distance_threshold)
    min_code_div = float(sel.min_code_divergence)
    min_length = int(sel.get("min_rollout_length", 450))
    score_percentile = float(sel.get("score_percentile", 75))

    # Step 1: Score each clip for the target posture
    peak_info = []  # (clip_idx, char_frame, score, joint_config, codes)
    for i, r in enumerate(rollouts):
        if len(r.qpos) < min_length:
            continue
        frame, score = _score_clip_for_posture(r.qpos, posture, search_window)

        # Joint config: strip root XY (indices 0,1), keep Z + quat + joints
        joint_config = r.qpos[frame, 2:]

        # D0 codes from characteristic frame onwards
        if r.rvq_indices is not None and len(r.rvq_indices) > 0:
            codes = np.array(r.rvq_indices[0][frame:])
        else:
            codes = np.array(r.code_indices[frame:])

        peak_info.append((i, frame, score, joint_config, codes))

    if not peak_info:
        raise ValueError(f"No valid clips found for posture '{posture}'")

    logging.info(
        f"  [{posture}] {len(peak_info)}/{len(rollouts)} clips pass "
        f"min_rollout_length={min_length}"
    )

    # Step 2: Filter by posture-specific score percentile
    all_scores = np.array([p[2] for p in peak_info])
    if posture == "rearing":
        # Use z_percentile for backward compat
        pct = float(sel.get("z_percentile", score_percentile))
        threshold = float(np.percentile(all_scores, pct))
        candidates = [p for p in peak_info if p[2] >= threshold]
        logging.info(
            f"  [{posture}] Score threshold (p{pct:.0f}): {threshold:.4f}, "
            f"max: {all_scores.max():.4f}"
        )
    elif posture == "walking":
        # High displacement → good walking clip
        threshold = float(np.percentile(all_scores, score_percentile))
        # Also reject clips with high Z range (likely rearing, not walking)
        candidates = []
        for p in peak_info:
            if p[2] < threshold:
                continue
            qpos = rollouts[p[0]].qpos
            window = min(search_window, len(qpos))
            z_range = float(np.max(qpos[:window, 2]) - np.min(qpos[:window, 2]))
            if z_range < 0.04:  # Reject high-rearing clips
                candidates.append(p)
        if not candidates:
            # Relax Z-range filter
            candidates = [p for p in peak_info if p[2] >= threshold]
        logging.info(
            f"  [{posture}] Score threshold (p{score_percentile:.0f}): "
            f"{threshold:.4f}, {len(candidates)} candidates"
        )
    elif posture == "grooming":
        # Negative score: want highest (least negative = least motion)
        threshold = float(np.percentile(all_scores, score_percentile))
        candidates = [p for p in peak_info if p[2] >= threshold]
        logging.info(
            f"  [{posture}] Score threshold (p{score_percentile:.0f}): "
            f"{threshold:.4f}, {len(candidates)} candidates"
        )
    else:
        threshold = float(np.percentile(all_scores, score_percentile))
        candidates = [p for p in peak_info if p[2] >= threshold]

    logging.info(f"  [{posture}] {len(candidates)} clips above threshold")

    if len(candidates) < K:
        logging.warning(
            f"  [{posture}] Only {len(candidates)} candidates, "
            f"relaxing to top-{K} by score"
        )
        top_indices = np.argsort(all_scores)[::-1][:K]
        candidates = [peak_info[j] for j in top_indices]

    # Step 3: Find group of K clips maximizing code divergence
    best_group = None
    best_score = -1.0

    for anchor_pos, anchor in enumerate(candidates):
        a_idx, a_frame, a_score, a_joints, a_codes = anchor
        companions = []

        for comp in candidates:
            c_idx, c_frame, c_score, c_joints, c_codes = comp
            if c_idx == a_idx:
                continue

            # Joint distance
            min_len = min(len(a_joints), len(c_joints))
            joint_dist = float(np.linalg.norm(a_joints[:min_len] - c_joints[:min_len]))
            if joint_dist > joint_threshold:
                continue

            # Code divergence: fraction of non-overlapping codes
            compare_len = min(100, len(a_codes), len(c_codes))
            if compare_len == 0:
                continue
            overlap = np.mean(a_codes[:compare_len] == c_codes[:compare_len])
            code_div = 1.0 - overlap
            if code_div < min_code_div:
                continue

            companions.append((c_idx, c_frame, joint_dist, code_div))

        if len(companions) < K - 1:
            continue

        # Sort by code divergence (descending), take top K-1
        companions.sort(key=lambda x: -x[3])
        selected = companions[: K - 1]

        # Score = mean pairwise code divergence
        all_group_codes = [a_codes]
        for s in selected:
            s_codes = None
            for p in peak_info:
                if p[0] == s[0]:
                    s_codes = p[4]
                    break
            if s_codes is not None:
                all_group_codes.append(s_codes)

        pair_divs = []
        for gi in range(len(all_group_codes)):
            for gj in range(gi + 1, len(all_group_codes)):
                clen = min(100, len(all_group_codes[gi]), len(all_group_codes[gj]))
                if clen > 0:
                    ov = np.mean(
                        all_group_codes[gi][:clen] == all_group_codes[gj][:clen]
                    )
                    pair_divs.append(1.0 - ov)
        group_score = float(np.mean(pair_divs)) if pair_divs else 0.0

        if group_score > best_score:
            best_score = group_score
            best_group = {
                "anchor_idx": a_idx,
                "anchor_frame": a_frame,
                "clip_indices": [a_idx] + [s[0] for s in selected],
                "start_frames": [a_frame] + [s[1] for s in selected],
                "posture_scores": [a_score]
                + [
                    peak_info[next(j for j, p in enumerate(peak_info) if p[0] == s[0])][
                        2
                    ]
                    for s in selected
                ],
                "pairwise_divergence": best_score,
                "posture": posture,
            }

    if best_group is None:
        # Fallback: just take the top-K by posture score
        logging.warning(
            f"  [{posture}] No group meeting divergence criteria, "
            f"using top-{K} by score"
        )
        top_k = sorted(candidates, key=lambda x: -x[2])[:K]
        best_group = {
            "anchor_idx": top_k[0][0],
            "anchor_frame": top_k[0][1],
            "clip_indices": [t[0] for t in top_k],
            "start_frames": [t[1] for t in top_k],
            "posture_scores": [t[2] for t in top_k],
            "pairwise_divergence": 0.0,
            "posture": posture,
        }

    logging.info(
        f"  [{posture}] Selected group: clips={best_group['clip_indices']}, "
        f"frames={best_group['start_frames']}, "
        f"divergence={best_group['pairwise_divergence']:.3f}"
    )
    return best_group


def find_divergent_clips(
    rollouts: list[InferenceResult],
    K: int,
    cfg: DictConfig,
) -> dict[str, Any]:
    """Find K clips with similar initial rearing poses but divergent futures.

    Thin wrapper around find_divergent_clips_by_posture for backward compat.
    """
    return find_divergent_clips_by_posture(rollouts, K, cfg, posture="rearing")


# =============================================================================
# DECODER-ONLY ROLLOUT
# =============================================================================


def run_decoder_condition(
    env: imitation.Imitation,
    jit_decode: Any,
    jit_reset: Any,
    jit_step: Any,
    code_sequences: list[np.ndarray],
    max_steps: int,
    seed: int,
    initial_qpos: np.ndarray | None = None,
) -> list[dict]:
    """Run K decoder-only rollouts from the env's starting pose.

    Each rollout uses a different code sequence but starts from the same
    state (the env resets to its single reference clip).

    Args:
        env: Single-clip environment.
        jit_decode: JIT-compiled (code_idx, obs) -> action function.
        jit_reset: JIT-compiled env.reset.
        jit_step: JIT-compiled env.step.
        code_sequences: List of K code arrays, each shape [T].
        max_steps: Maximum rollout steps.
        seed: Random seed.
        initial_qpos: If provided, override the reset qpos with this
            array (e.g. H5 inference qpos at the rearing frame).
            Shape [nq].

    Returns:
        List of K dicts with keys: qpos [T, nq], rewards [T],
        code_indices [T], survival (int).
    """
    from mujoco import mjx as mjx_lib

    rng = jax.random.PRNGKey(seed)
    results = []

    for i, codes in enumerate(code_sequences):
        rng, reset_rng = jax.random.split(rng)
        state = jit_reset(reset_rng)

        # Override qpos with H5 inference pose (e.g. rearing pose)
        if initial_qpos is not None:
            new_data = state.data.replace(qpos=jnp.array(initial_qpos))
            new_data = mjx_lib.forward(env.mjx_model, new_data)
            obs = env._get_obs(new_data, state.info)
            state = state.replace(data=new_data, obs=obs)

        qpos_list = []
        rewards = []
        code_list = []
        steps_survived = 0

        for t in range(min(max_steps, len(codes))):
            code_idx = int(codes[t])

            if hasattr(state, "data"):
                qpos_list.append(np.array(state.data.qpos))
            elif hasattr(state, "pipeline_state"):
                qpos_list.append(np.array(state.pipeline_state.q))

            action = jit_decode(code_idx, state.obs)
            next_state = jit_step(state, action)
            rewards.append(float(next_state.reward))
            code_list.append(code_idx)
            steps_survived += 1

            # Bypass termination — only stop on NaN (sim blowup).
            # Decoder-only mode hits tight termination criteria too easily.
            if jnp.any(jnp.isnan(next_state.reward)):
                logging.info(f"  NaN at step {t} for rollout {i}, stopping.")
                break
            state = next_state

        results.append(
            {
                "qpos": np.stack(qpos_list) if qpos_list else np.zeros((0, 74)),
                "rewards": np.array(rewards),
                "code_indices": np.array(code_list),
                "survival": steps_survived,
            }
        )

    return results


# =============================================================================
# CODE SEQUENCE GENERATION FOR CONDITIONS
# =============================================================================


def make_correct_code_sequences(
    rollouts: list[InferenceResult],
    clip_indices: list[int],
    start_frames: list[int],
    max_steps: int,
) -> list[np.ndarray]:
    """Extract correct D0 code sequences from H5 rollouts (Condition A).

    Codes start from each clip's start_frame to match the env starting pose
    (all rollouts begin at the anchor's peak-rearing frame).

    Args:
        rollouts: Full H5 inference results.
        clip_indices: Selected clip indices.
        start_frames: Starting frame per clip (peak-rearing frame).
        max_steps: Max sequence length.

    Returns:
        List of K code arrays.
    """
    sequences = []
    for clip_idx, start_frame in zip(clip_indices, start_frames):
        r = rollouts[clip_idx]
        if r.rvq_indices is not None and len(r.rvq_indices) > 0:
            codes = np.array(r.rvq_indices[0][start_frame:])
        else:
            codes = np.array(r.code_indices[start_frame:])

        # Pad if shorter than max_steps (repeat last code)
        if len(codes) < max_steps:
            pad = np.full(max_steps - len(codes), codes[-1] if len(codes) > 0 else 0)
            codes = np.concatenate([codes, pad])
        else:
            codes = codes[:max_steps]
        sequences.append(codes)
    return sequences


def make_step_excluded_sequences(
    correct_sequences: list[np.ndarray],
    num_codes: int,
    max_steps: int,
    seed: int,
) -> list[np.ndarray]:
    """Generate random codes excluding correct codes at each step (Condition B).

    At each timestep t, the set of codes used by any of the K correct
    sequences is excluded, and a random code is drawn from the remainder.

    Args:
        correct_sequences: K correct code sequences from Condition A.
        num_codes: Total number of D0 codes.
        max_steps: Sequence length.
        seed: Random seed.

    Returns:
        List of K random code sequences.
    """
    rng = np.random.default_rng(seed)
    K = len(correct_sequences)
    sequences = []

    for i in range(K):
        codes = np.zeros(max_steps, dtype=int)
        for t in range(max_steps):
            excluded = set()
            for seq in correct_sequences:
                if t < len(seq):
                    excluded.add(int(seq[t]))
            available = [c for c in range(num_codes) if c not in excluded]
            if not available:
                available = list(range(num_codes))
            codes[t] = rng.choice(available)
        sequences.append(codes)

    return sequences


def make_trajectory_excluded_sequences(
    correct_sequences: list[np.ndarray],
    num_codes: int,
    max_steps: int,
    seed: int,
) -> list[np.ndarray]:
    """Generate random codes from globally disjoint code set (Condition C).

    The union of all codes appearing in any correct sequence is excluded.
    Random codes are drawn uniformly from the remaining set.

    Args:
        correct_sequences: K correct code sequences from Condition A.
        num_codes: Total number of D0 codes.
        max_steps: Sequence length.
        seed: Random seed.

    Returns:
        List of K random code sequences.
    """
    rng = np.random.default_rng(seed)
    excluded = set()
    for seq in correct_sequences:
        excluded.update(int(c) for c in seq)

    available = [c for c in range(num_codes) if c not in excluded]
    if not available:
        logging.error(
            f"  All {num_codes} codes used by correct sequences, "
            f"falling back to code 0"
        )
        available = [0]

    logging.info(
        f"  Trajectory-excluded: {len(excluded)} codes excluded, "
        f"{len(available)} available"
    )

    sequences = []
    for i in range(len(correct_sequences)):
        codes = rng.choice(available, size=max_steps)
        sequences.append(codes.astype(int))

    return sequences


# =============================================================================
# DIVERGENCE METRICS
# =============================================================================


def compute_divergence_metrics(
    condition_results: list[dict],
) -> dict[str, Any]:
    """Compute quantitative trajectory divergence metrics.

    Args:
        condition_results: List of K result dicts from run_decoder_condition.

    Returns:
        Dict with: pairwise_joint_l2 (array over time), mean_reward,
        mean_survival, root_displacement (per trajectory),
        root_z_range (per trajectory).
    """
    K = len(condition_results)
    min_len = min(r["survival"] for r in condition_results)
    if min_len == 0:
        return {
            "pairwise_joint_l2": np.zeros(0),
            "mean_reward": 0.0,
            "mean_survival": 0,
            "root_displacements": [],
            "root_z_ranges": [],
        }

    # Pairwise joint L2 over time
    pair_curves = []
    for i in range(K):
        for j in range(i + 1, K):
            q_i = condition_results[i]["qpos"][:min_len, 7:]
            q_j = condition_results[j]["qpos"][:min_len, 7:]
            pair_curves.append(np.linalg.norm(q_i - q_j, axis=1))
    mean_pairwise = np.mean(pair_curves, axis=0) if pair_curves else np.zeros(min_len)

    # Per-trajectory metrics
    root_disps = []
    z_ranges = []
    for r in condition_results:
        qpos = r["qpos"]
        if len(qpos) >= 2:
            diffs = np.diff(qpos[:, :2], axis=0)
            root_disps.append(float(np.sum(np.linalg.norm(diffs, axis=1))))
            z_ranges.append(float(np.max(qpos[:, 2]) - np.min(qpos[:, 2])))
        else:
            root_disps.append(0.0)
            z_ranges.append(0.0)

    mean_reward = float(np.mean([np.mean(r["rewards"]) for r in condition_results]))
    mean_survival = float(np.mean([r["survival"] for r in condition_results]))

    return {
        "pairwise_joint_l2": mean_pairwise,
        "mean_reward": mean_reward,
        "mean_survival": mean_survival,
        "root_displacements": root_disps,
        "root_z_ranges": z_ranges,
    }


# =============================================================================
# GHOST VIDEO RENDERING
# =============================================================================


def _make_code_bar(
    width: int,
    code_sequences: list[np.ndarray],
    frame_idx: int,
    colors: list[tuple[int, int, int]],
    code_colors: np.ndarray,
    bar_height: int = 20,
    gap: int = 2,
) -> np.ndarray:
    """Build stacked code timeline bars for K trajectories.

    Args:
        width: Image width.
        code_sequences: K code arrays.
        frame_idx: Current playhead position.
        colors: Trajectory colors (for border).
        code_colors: Code colormap [num_codes, 3].
        bar_height: Height of each bar.
        gap: Gap between bars.

    Returns:
        Bar image, shape [total_h, width, 3].
    """
    K = len(code_sequences)
    total_h = K * bar_height + (K - 1) * gap
    img = np.ones((total_h, width, 3), dtype=np.uint8) * 40

    for ki, codes in enumerate(code_sequences):
        y0 = ki * (bar_height + gap)
        num_frames = len(codes)
        for t, code_idx in enumerate(codes):
            x0 = int(t * width / num_frames)
            x1 = int((t + 1) * width / num_frames)
            img[y0 : y0 + bar_height, x0:x1] = code_colors[
                int(code_idx) % len(code_colors)
            ]

        # Playhead
        if frame_idx < num_frames:
            px = int(frame_idx * width / num_frames)
            px = min(px, width - 2)
            img[y0 : y0 + bar_height, px : px + 2] = [255, 255, 255]

        # Left-side trajectory color marker
        img[y0 : y0 + bar_height, :3] = colors[ki]

    return img


def render_ghost_video(
    ghost_model: Any,
    base_nq: int,
    trajectories_qpos: list[np.ndarray],
    code_sequences: list[np.ndarray],
    trajectory_colors: list[list[float]],
    output_path: Path,
    title: str,
    camera: str = GHOST_CAMERA,
    width: int = 800,
    height: int = 600,
    fps: int = 50,
    code_colors: np.ndarray | None = None,
) -> str:
    """Render K overlaid trajectories as a ghost-body video.

    Args:
        ghost_model: Compiled MuJoCo model with ghost bodies.
        base_nq: qpos dimensionality of single body.
        trajectories_qpos: List of K qpos arrays, each [T_i, nq].
        code_sequences: List of K code index arrays.
        trajectory_colors: RGBA per trajectory for display.
        output_path: Output video file path.
        title: Title overlay text.
        camera: Camera name.
        width: Frame width.
        height: Frame height.
        fps: Frames per second.
        code_colors: Code colormap for timeline bars.

    Returns:
        Path to written video file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    K = len(trajectories_qpos)
    min_len = min(len(q) for q in trajectories_qpos)

    ghost_model.vis.global_.offwidth = width
    ghost_model.vis.global_.offheight = height
    data = mujoco.MjData(ghost_model)
    renderer = mujoco.Renderer(ghost_model, height=height, width=width)

    # RGB colors for bars
    bar_colors = [
        (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)) for c in trajectory_colors
    ]

    frames = []
    for t in range(min_len):
        # Set primary body qpos
        data.qpos[:base_nq] = trajectories_qpos[0][t]

        # Set ghost body qpos
        for gi in range(1, K):
            q_start = base_nq + (gi - 1) * base_nq
            q_end = q_start + base_nq
            data.qpos[q_start:q_end] = trajectories_qpos[gi][t]

        mujoco.mj_forward(ghost_model, data)
        renderer.update_scene(data, camera=camera)
        frame = renderer.render().copy()

        # Add title overlay
        frame = add_multi_line_overlay(
            frame,
            [title, f"t={t}"],
            start_position=(10, 10),
            font_size=16,
        )

        # Add code timeline bars at bottom
        if code_colors is not None:
            bar_img = _make_code_bar(
                width=width,
                code_sequences=code_sequences,
                frame_idx=t,
                colors=bar_colors,
                code_colors=code_colors,
            )
            bar_h = bar_img.shape[0]
            # Overlay bar at bottom of frame
            frame[-bar_h:, :] = bar_img

        frames.append(frame)

    renderer.close()

    writer = imageio.get_writer(str(output_path), fps=fps)
    for f in frames:
        writer.append_data(f)
    writer.close()

    logging.info(f"  Wrote ghost video ({min_len} frames): {output_path}")
    return str(output_path)


# =============================================================================
# REFERENCE TRAJECTORY VIDEO
# =============================================================================


def render_reference_ghost_video(
    env: imitation.Imitation,
    rollouts: list[InferenceResult],
    clip_indices: list[int],
    start_frames: list[int],
    trajectory_colors: list[list[float]],
    output_path: Path,
    cfg: DictConfig,
    code_colors: np.ndarray,
) -> str:
    """Render reference H5 trajectories as overlaid ghosts.

    Args:
        env: Base environment (for model spec).
        rollouts: H5 inference results.
        clip_indices: Selected clip indices.
        start_frames: Start frame per clip (peak-rearing frame).
        trajectory_colors: RGBA per trajectory.
        output_path: Output path.
        cfg: Render config.
        code_colors: Code colormap.

    Returns:
        Video file path.
    """
    K = len(clip_indices)
    # Need K-1 ghosts (first body is the main one)
    ghost_colors = [trajectory_colors[gi] for gi in range(1, K)]
    render_cfg = cfg.render
    ghost_model, base_nq = build_ghost_model(
        env,
        K - 1,
        ghost_colors,
        camera_distance=float(render_cfg.get("camera_distance", 0.8)),
        camera_elevation=float(render_cfg.get("camera_elevation", -30.0)),
        camera_azimuth=float(render_cfg.get("camera_azimuth", 135.0)),
        camera_fovy=float(render_cfg.get("camera_fovy", 60.0)),
    )

    max_steps = int(cfg.experiment.max_steps)
    trajectories_qpos = []
    code_sequences = []
    for ci, sf in zip(clip_indices, start_frames):
        r = rollouts[ci]
        qpos = r.qpos[sf : sf + max_steps]
        trajectories_qpos.append(qpos)
        if r.rvq_indices is not None and len(r.rvq_indices) > 0:
            codes = np.array(r.rvq_indices[0][sf : sf + max_steps])
        else:
            codes = np.array(r.code_indices[sf : sf + max_steps])
        code_sequences.append(codes)

    return render_ghost_video(
        ghost_model=ghost_model,
        base_nq=base_nq,
        trajectories_qpos=trajectories_qpos,
        code_sequences=code_sequences,
        trajectory_colors=trajectory_colors,
        output_path=output_path,
        title="Reference Trajectories",
        camera=GHOST_CAMERA,
        width=int(cfg.render.width),
        height=int(cfg.render.height),
        fps=int(cfg.render.fps),
        code_colors=code_colors,
    )


# =============================================================================
# PLOTTING
# =============================================================================


def plot_divergence_curves(
    metrics_by_condition: dict[str, dict],
    output_path: Path,
) -> str:
    """Plot mean pairwise joint L2 divergence over time for all conditions.

    Args:
        metrics_by_condition: Mapping condition name -> metrics dict.
        output_path: Output PNG path.

    Returns:
        Path to saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    condition_colors = {
        "A: Correct": "#2196F3",
        "B: Step-excluded": "#FF9800",
        "C: Traj-excluded": "#4CAF50",
    }

    for name, metrics in metrics_by_condition.items():
        curve = metrics["pairwise_joint_l2"]
        if len(curve) > 0:
            color = condition_colors.get(name, None)
            ax.plot(curve, label=name, linewidth=1.5, color=color)

    ax.set_xlabel("Timestep", fontsize=11)
    ax.set_ylabel("Mean Pairwise Joint L2", fontsize=11)
    ax.set_title("Trajectory Divergence by Condition", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


# =============================================================================
# HTML SUMMARY
# =============================================================================


def _encode_file_b64(path: str, mime: str) -> str:
    """Read file and return base64 data URI."""
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"


def build_summary_html(
    video_paths: dict[str, str],
    plot_paths: dict[str, str],
    metrics_table: dict[str, dict[str, float]],
    group_info: dict,
    title: str,
) -> str:
    """Build single-page HTML summary with embedded videos and plots.

    Args:
        video_paths: Mapping name -> video file path.
        plot_paths: Mapping name -> plot PNG path.
        metrics_table: Condition name -> scalar metrics.
        group_info: Clip selection info.
        title: Page title.

    Returns:
        HTML string.
    """
    video_embeds = []
    for name, path in video_paths.items():
        data_uri = _encode_file_b64(path, "video/mp4")
        video_embeds.append(
            f'<div class="section">'
            f"<h3>{name}</h3>"
            f'<video src="{data_uri}" width="640" autoplay loop muted controls></video>'
            f"</div>"
        )

    plot_embeds = []
    for name, path in plot_paths.items():
        data_uri = _encode_file_b64(path, "image/png")
        plot_embeds.append(
            f'<div class="section">'
            f"<h3>{name}</h3>"
            f'<img src="{data_uri}" style="max-width:100%;" />'
            f"</div>"
        )

    # Metrics table
    rows = []
    for cond, metrics in metrics_table.items():
        rows.append(
            f"<tr><td><b>{cond}</b></td>"
            f"<td>{metrics.get('mean_reward', 0):.1f}</td>"
            f"<td>{metrics.get('mean_survival', 0):.0f}</td>"
            f"<td>{', '.join(f'{d:.3f}' for d in metrics.get('root_displacements', []))}</td>"
            f"<td>{', '.join(f'{z:.3f}' for z in metrics.get('root_z_ranges', []))}</td>"
            f"</tr>"
        )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: sans-serif; margin: 20px; background: #fafafa; }}
h2 {{ color: #333; }}
.section {{ margin: 16px 0; padding: 12px; background: #fff;
           border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #f5f5f5; }}
.info {{ font-size: 13px; color: #666; margin: 4px 0; }}
</style>
</head><body>
<h2>{title}</h2>
<div class="section">
<h3>Clip Selection</h3>
<p class="info">Clips: {group_info.get('clip_indices', [])}</p>
<p class="info">Start frames: {group_info.get('start_frames', [])}</p>
<p class="info">Posture scores: {[f'{s:.4f}' for s in group_info.get('posture_scores', group_info.get('peak_z_values', []))]}</p>
<p class="info">Pairwise code divergence: {group_info.get('pairwise_divergence', 0):.3f}</p>
</div>
{''.join(video_embeds)}
{''.join(plot_embeds)}
<div class="section">
<h3>Metrics Summary</h3>
<table>
<tr><th>Condition</th><th>Mean Reward</th><th>Mean Survival</th>
<th>Root Displacements</th><th>Root Z Ranges</th></tr>
{''.join(rows)}
</table>
</div>
</body></html>"""
    return html


# =============================================================================
# WANDB HELPERS
# =============================================================================


def init_wandb(cfg: DictConfig) -> bool:
    """Initialize WandB for divergent futures experiment."""
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return False
    try:
        import wandb

        run_name = f"divergent_futures_{datetime.now().strftime('%y%m%d_%H%M%S')}"
        wandb.init(
            project=wandb_cfg.get("project", "vqvae-eval"),
            entity=wandb_cfg.get("entity"),
            name=run_name,
            config={
                "checkpoint_path": str(cfg.checkpoint.path),
                "K": cfg.experiment.K,
                "max_steps": cfg.experiment.max_steps,
            },
        )
        return True
    except Exception as e:
        logging.warning(f"Failed to init WandB: {e}")
        return False


# =============================================================================
# POSTURE EXPERIMENT (reusable per-posture pipeline)
# =============================================================================


def run_posture_experiment(
    posture: str,
    rollouts: list[InferenceResult],
    K: int,
    cfg: DictConfig,
    jit_decode: Any,
    vq_cfg: Any,
    policy_params: Any,
    num_codes: int,
    code_colors: np.ndarray,
    trajectory_colors: list[list[float]],
    output_dir: Path,
    seed: int,
    max_steps: int,
    test_clips: Any,
) -> dict[str, Any]:
    """Run one posture experiment: clip selection, 3 conditions, render, HTML.

    Args:
        posture: One of "rearing", "walking", "grooming".
        rollouts: H5 inference results.
        K: Number of clips per group.
        cfg: Full Hydra config.
        jit_decode: JIT-compiled decoder step.
        vq_cfg: VQ-VAE training config.
        policy_params: Policy parameters.
        num_codes: Number of D0 codes.
        code_colors: Code colormap [num_codes, 3].
        trajectory_colors: RGBA per trajectory.
        output_dir: Per-posture output directory.
        seed: Random seed.
        max_steps: Max rollout steps.
        test_clips: Test reference clips (for env creation).

    Returns:
        Dict with group_info, condition_results, metrics_by_condition,
        video_paths, plot_paths, html_path, html_str, anchor_h5_qpos.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    render_enabled = cfg.render.get("enabled", True)

    # --- Find divergent clips ---
    logging.info(f"\n[{posture}] Finding divergent clips...")
    group_info = find_divergent_clips_by_posture(rollouts, K, cfg, posture)

    clip_indices = group_info["clip_indices"]
    start_frames = group_info["start_frames"]

    # --- Create single-clip env for the anchor ---
    anchor_clip_idx = clip_indices[0]
    anchor_start_frame = start_frames[0]
    logging.info(
        f"  [{posture}] Creating env (clip {anchor_clip_idx}, "
        f"start_frame={anchor_start_frame})"
    )
    _, cfg_dict, env_cfg_ml = config_utils.prepare_config(cfg)
    env_cfg_ml.start_frame_range = [anchor_start_frame, anchor_start_frame + 1]

    single_clip = subset_clips(test_clips, anchor_clip_idx)
    env = imitation.Imitation(config=env_cfg_ml, clips=single_clip)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # H5 inference qpos at anchor's characteristic frame
    anchor_h5_qpos = rollouts[anchor_clip_idx].qpos[anchor_start_frame]
    logging.info(f"  [{posture}] H5 anchor qpos root_z={anchor_h5_qpos[2]:.4f}")

    # --- Generate code sequences for 3 conditions ---
    logging.info(f"  [{posture}] Generating code sequences...")
    correct_sequences = make_correct_code_sequences(
        rollouts, clip_indices, start_frames, max_steps
    )
    step_excluded_sequences = make_step_excluded_sequences(
        correct_sequences, num_codes, max_steps, seed + 100
    )
    traj_excluded_sequences = make_trajectory_excluded_sequences(
        correct_sequences, num_codes, max_steps, seed + 200
    )

    # --- Run 3 conditions ---
    conditions = {
        "A: Correct": correct_sequences,
        "B: Step-excluded": step_excluded_sequences,
        "C: Traj-excluded": traj_excluded_sequences,
    }

    condition_results: dict[str, list[dict]] = {}
    metrics_by_condition: dict[str, dict] = {}

    for cond_name, code_seqs in conditions.items():
        logging.info(f"  [{posture}] Running condition: {cond_name}")
        results = run_decoder_condition(
            env=env,
            jit_decode=jit_decode,
            jit_reset=jit_reset,
            jit_step=jit_step,
            code_sequences=code_seqs,
            max_steps=max_steps,
            seed=seed,
            initial_qpos=anchor_h5_qpos,
        )
        condition_results[cond_name] = results
        metrics = compute_divergence_metrics(results)
        metrics_by_condition[cond_name] = metrics

        logging.info(
            f"  [{posture}] {cond_name}: reward={metrics['mean_reward']:.1f}, "
            f"survival={metrics['mean_survival']:.0f}"
        )

    # --- Render videos ---
    video_paths: dict[str, str] = {}

    if render_enabled:
        logging.info(f"  [{posture}] Rendering videos...")

        # Reference trajectory video
        ref_path = render_reference_ghost_video(
            env=env,
            rollouts=rollouts,
            clip_indices=clip_indices,
            start_frames=start_frames,
            trajectory_colors=trajectory_colors,
            output_path=output_dir / "reference_trajectories.mp4",
            cfg=cfg,
            code_colors=code_colors,
        )
        video_paths["Reference Trajectories"] = ref_path

        # Condition videos
        ghost_colors = [trajectory_colors[gi] for gi in range(1, K)]
        render_cfg = cfg.render
        ghost_model, base_nq = build_ghost_model(
            env,
            K - 1,
            ghost_colors,
            camera_distance=float(render_cfg.get("camera_distance", 0.8)),
            camera_elevation=float(render_cfg.get("camera_elevation", -30.0)),
            camera_azimuth=float(render_cfg.get("camera_azimuth", 135.0)),
            camera_fovy=float(render_cfg.get("camera_fovy", 60.0)),
        )

        for cond_name, results in condition_results.items():
            safe_name = cond_name.replace(": ", "_").replace(" ", "_").lower()
            trajs = [r["qpos"] for r in results]
            code_seqs_render = [r["code_indices"] for r in results]

            vid_path = render_ghost_video(
                ghost_model=ghost_model,
                base_nq=base_nq,
                trajectories_qpos=trajs,
                code_sequences=code_seqs_render,
                trajectory_colors=trajectory_colors,
                output_path=output_dir / f"condition_{safe_name}.mp4",
                title=f"{posture.capitalize()} — {cond_name}",
                camera=GHOST_CAMERA,
                width=int(cfg.render.width),
                height=int(cfg.render.height),
                fps=int(cfg.render.fps),
                code_colors=code_colors,
            )
            video_paths[cond_name] = vid_path

    # --- Divergence plots ---
    divergence_plot_path = plot_divergence_curves(
        metrics_by_condition, output_dir / "divergence_curves.png"
    )
    plot_paths = {"Divergence Curves": divergence_plot_path}

    # --- Build HTML summary ---
    metrics_table = {}
    for cond_name, m in metrics_by_condition.items():
        metrics_table[cond_name] = {
            "mean_reward": m["mean_reward"],
            "mean_survival": m["mean_survival"],
            "root_displacements": m["root_displacements"],
            "root_z_ranges": m["root_z_ranges"],
        }

    html = build_summary_html(
        video_paths=video_paths,
        plot_paths=plot_paths,
        metrics_table=metrics_table,
        group_info=group_info,
        title=f"Divergent Futures — {posture.capitalize()}",
    )
    html_path = output_dir / f"divergent_futures_{posture}.html"
    with open(html_path, "w") as f:
        f.write(html)
    logging.info(f"  [{posture}] HTML summary: {html_path}")

    return {
        "group_info": group_info,
        "condition_results": condition_results,
        "metrics_by_condition": metrics_by_condition,
        "video_paths": video_paths,
        "plot_paths": plot_paths,
        "html_path": str(html_path),
        "html_str": html,
        "anchor_h5_qpos": anchor_h5_qpos,
        "correct_sequences": correct_sequences,
        "env": env,
    }


# =============================================================================
# KILLER DEMO — SIDE-BY-SIDE CODE SWAP VIDEO
# =============================================================================


def render_multi_panel_video(
    env: imitation.Imitation,
    panels: list[dict],
    trajectory_colors: list[list[float]],
    output_path: Path,
    cfg: DictConfig,
    code_colors: np.ndarray,
) -> str:
    """Render N panels of K trajectories side by side.

    Each panel gets its own ghost model and renderer. Frames are concatenated
    horizontally with label overlays and code bars.

    Args:
        env: Base environment (for model spec).
        panels: List of dicts, each with keys:
            - "trajectories": list of K qpos arrays [T, nq]
            - "codes": list of K code index arrays [T]
            - "label": str label for the panel
        trajectory_colors: RGBA per trajectory.
        output_path: Output video path.
        cfg: Render config.
        code_colors: Code colormap [num_codes, 3].

    Returns:
        Path to written video file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_panels = len(panels)
    K = len(panels[0]["trajectories"])
    render_cfg = cfg.render
    panel_w = int(render_cfg.width)
    panel_h = int(render_cfg.height)
    fps = int(render_cfg.fps)
    cam_dist = float(render_cfg.get("killer_demo_camera_distance", 1.2))

    # Build ghost models for each panel
    ghost_colors = [trajectory_colors[gi] for gi in range(1, K)]

    def _build_panel_model():
        return build_ghost_model(
            env,
            K - 1,
            ghost_colors,
            camera_distance=cam_dist,
            camera_elevation=float(render_cfg.get("camera_elevation", -30.0)),
            camera_azimuth=float(render_cfg.get("camera_azimuth", 135.0)),
            camera_fovy=float(render_cfg.get("camera_fovy", 60.0)),
        )

    models = []
    datas = []
    renderers = []
    base_nq = None
    for _ in range(n_panels):
        model, bnq = _build_panel_model()
        if base_nq is None:
            base_nq = bnq
        model.vis.global_.offwidth = panel_w
        model.vis.global_.offheight = panel_h
        models.append(model)
        datas.append(mujoco.MjData(model))
        renderers.append(mujoco.Renderer(model, height=panel_h, width=panel_w))

    bar_colors = [
        (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)) for c in trajectory_colors
    ]

    # Find shortest trajectory across all panels
    min_len = min(min(len(q) for q in panel["trajectories"]) for panel in panels)

    frames = []
    for t in range(min_len):
        panel_frames = []
        panel_bars = []
        for pi, panel in enumerate(panels):
            data = datas[pi]
            model = models[pi]
            renderer = renderers[pi]
            trajs = panel["trajectories"]

            # Set primary + ghost body qpos
            data.qpos[:base_nq] = trajs[0][t]
            for gi in range(1, K):
                q_start = base_nq + (gi - 1) * base_nq
                data.qpos[q_start : q_start + base_nq] = trajs[gi][t]

            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=GHOST_CAMERA)
            frame = renderer.render().copy()

            # Add label
            frame = add_multi_line_overlay(
                frame,
                [panel["label"], f"t={t}"],
                start_position=(10, 10),
                font_size=16,
            )

            panel_frames.append(frame)

            # Build code bar (appended below, not overlaid)
            if code_colors is not None:
                bar = _make_code_bar(
                    panel_w, panel["codes"], t, bar_colors, code_colors
                )
                panel_bars.append(bar)

        # Concatenate panels horizontally, then append code bars below
        render_row = np.concatenate(panel_frames, axis=1)
        if panel_bars:
            bar_row = np.concatenate(panel_bars, axis=1)
            combined = np.concatenate([render_row, bar_row], axis=0)
        else:
            combined = render_row
        frames.append(combined)

    for r in renderers:
        r.close()

    writer = imageio.get_writer(str(output_path), fps=fps)
    for f in frames:
        writer.append_data(f)
    writer.close()

    logging.info(f"  Wrote {n_panels}-panel video ({min_len} frames): {output_path}")
    return str(output_path)


def _mean_curves_from_results(
    results: list[dict],
    extract_fn,
) -> np.ndarray:
    """Extract per-trajectory curves via extract_fn, return mean curve."""
    curves = []
    for r in results:
        curve = extract_fn(r)
        if curve is not None:
            curves.append(curve)
    if not curves:
        return np.zeros(1)
    min_len = min(len(c) for c in curves)
    stacked = np.stack([c[:min_len] for c in curves])
    return np.mean(stacked, axis=0)


def _cumulative_xy(r: dict) -> np.ndarray | None:
    qpos = r["qpos"]
    if len(qpos) < 2:
        return None
    diffs = np.diff(qpos[:, :2], axis=0)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs, axis=1))])


def _cumulative_z(r: dict) -> np.ndarray | None:
    qpos = r["qpos"]
    if len(qpos) < 2:
        return None
    diffs = np.abs(np.diff(qpos[:, 2]))
    return np.concatenate([[0.0], np.cumsum(diffs)])


def _mean_xy(r: dict) -> np.ndarray | None:
    """Running mean of per-step XY displacement."""
    qpos = r["qpos"]
    if len(qpos) < 2:
        return None
    diffs = np.linalg.norm(np.diff(qpos[:, :2], axis=0), axis=1)
    # Cumulative mean: cumsum / arange
    cum = np.cumsum(diffs)
    return cum / np.arange(1, len(cum) + 1)


def _mean_z(r: dict) -> np.ndarray | None:
    """Running mean of root Z height."""
    qpos = r["qpos"]
    if len(qpos) < 1:
        return None
    z = qpos[:, 2]
    return np.cumsum(z) / np.arange(1, len(z) + 1)


def plot_killer_demo_displacement(
    conditions: list[tuple[str, list[dict]]],
    output_path: Path,
    title: str = "Root Motion: Code Swap from Rearing Pose",
) -> str:
    """Plot 4 root motion subplots for N conditions.

    Subplots: cumulative XY, cumulative Z, mean XY per step, mean Z height.

    Args:
        conditions: List of (label, results) tuples.
        output_path: Output PNG path.
        title: Suptitle for the figure.

    Returns:
        Path to saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    condition_colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

    subplot_specs = [
        ("Cumulative XY Displacement", "Displacement", _cumulative_xy),
        ("Cumulative Z Displacement", "Displacement", _cumulative_z),
        ("Running Mean XY Speed", "Mean XY/step", _mean_xy),
        ("Running Mean Root Z", "Mean Z height", _mean_z),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for ax_idx, (title, ylabel, extract_fn) in enumerate(subplot_specs):
        ax = axes[ax_idx]
        for i, (label, results) in enumerate(conditions):
            curve = _mean_curves_from_results(results, extract_fn)
            color = condition_colors[i % len(condition_colors)]
            ax.plot(curve, label=label, linewidth=2, color=color)
        ax.set_xlabel("Timestep", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Wrote displacement plot: {output_path}")
    return str(output_path)


def _select_top_clips_for_posture(
    rollouts: list[InferenceResult],
    posture: str,
    K: int,
    cfg: DictConfig,
) -> tuple[list[int], list[int], list[np.ndarray]]:
    """Select top-K clips by posture score and return their code sequences.

    Unlike find_divergent_clips_by_posture (which optimizes for code
    divergence within a group), this simply picks the K best-scoring
    clips for the target posture — suitable for the killer demo where
    we want many diverse trajectories rather than a matched group.

    Args:
        rollouts: H5 inference results.
        posture: One of "rearing", "walking", "grooming".
        K: Number of clips to select.
        cfg: Config with pose_selection params.

    Returns:
        Tuple of (clip_indices, start_frames, code_sequences).
    """
    sel = cfg.experiment.pose_selection
    search_window = int(sel.search_window_frames)
    min_length = int(sel.get("min_rollout_length", 450))
    max_steps = int(cfg.experiment.max_steps)

    scored = []  # (clip_idx, frame, score)
    for i, r in enumerate(rollouts):
        if len(r.qpos) < min_length:
            continue
        frame, score = _score_clip_for_posture(r.qpos, posture, search_window)
        scored.append((i, frame, score))

    # Sort by score descending, take top K
    scored.sort(key=lambda x: -x[2])
    selected = scored[:K]

    clip_indices = [s[0] for s in selected]
    start_frames = [s[1] for s in selected]
    code_sequences = make_correct_code_sequences(
        rollouts, clip_indices, start_frames, max_steps
    )

    logging.info(
        f"  [killer_demo/{posture}] Selected {len(selected)} clips: " f"{clip_indices}"
    )
    return clip_indices, start_frames, code_sequences


def run_killer_demo(
    all_posture_results: dict[str, dict],
    rollouts: list[InferenceResult],
    jit_decode: Any,
    cfg: DictConfig,
    vq_cfg: Any,
    code_colors: np.ndarray,
    trajectory_colors: list[list[float]],
    output_dir: Path,
    seed: int,
    max_steps: int,
    test_clips: Any,
    anchor_z_frac_override: float | None = None,
    demo_label: str = "Killer Demo",
) -> dict[str, Any] | None:
    """Run killer demo: same starting pose, 3 code conditions side by side.

    Panels: walking codes | grooming codes | sustained rearing codes.
    Uses its own K (from cfg.experiment.killer_demo.K, default 10).

    Requires "rearing" posture experiment results for the anchor pose.

    Args:
        all_posture_results: Results from run_posture_experiment per posture.
        rollouts: H5 inference results.
        jit_decode: JIT-compiled decoder step.
        cfg: Full Hydra config.
        vq_cfg: VQ-VAE training config.
        code_colors: Code colormap.
        trajectory_colors: RGBA per trajectory.
        output_dir: Output directory for killer demo.
        seed: Random seed.
        max_steps: Max rollout steps.
        test_clips: Test reference clips.

    Returns:
        Dict with video_path, plot_path, html_str, or None if prerequisites
        not met.
    """
    if "rearing" not in all_posture_results:
        logging.warning("  Killer demo requires 'rearing' posture results")
        return None

    rearing_result = all_posture_results["rearing"]
    rearing_env = rearing_result["env"]

    killer_demo_cfg = cfg.experiment.get("killer_demo", {})
    killer_K = int(killer_demo_cfg.get("K", 10))

    # Pick a moderate rearing anchor pose. Compute target Z as a fraction
    # between global standing height and peak rearing height, then find the
    # closest qpos across all clips. This avoids clips that start fully reared.
    anchor_z_frac = (
        anchor_z_frac_override
        if anchor_z_frac_override is not None
        else float(killer_demo_cfg.get("anchor_z_fraction", 0.5))
    )
    # Global standing Z = median root Z at frame 0 across clips
    standing_z = float(np.median([r.qpos[0, 2] for r in rollouts]))
    peak_z = float(
        np.max([np.max(r.qpos[:, 2]) for r in rollouts if len(r.qpos) > 0])
    )
    target_z = standing_z + anchor_z_frac * (peak_z - standing_z)
    # Search all clips for the frame closest to target_z
    best_clip, best_frame, best_diff = 0, 0, float("inf")
    for i, r in enumerate(rollouts):
        diffs = np.abs(r.qpos[:, 2] - target_z)
        t = int(np.argmin(diffs))
        if diffs[t] < best_diff:
            best_clip, best_frame, best_diff = i, t, diffs[t]
    anchor_clip_idx = best_clip
    anchor_frame = best_frame
    anchor_h5_qpos = rollouts[anchor_clip_idx].qpos[anchor_frame]
    logging.info(
        f"  [killer_demo] Anchor: clip {anchor_clip_idx}, frame {anchor_frame} "
        f"(z={anchor_h5_qpos[2]:.4f}, target={target_z:.4f}, "
        f"frac={anchor_z_frac}, standing={standing_z:.4f}, peak={peak_z:.4f})"
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define the conditions for the killer demo panels
    # Each: (posture_for_code_selection, label)
    panel_defs = []
    if "walking" in all_posture_results:
        panel_defs.append(("walking", "Walking Codes"))
    if "grooming" in all_posture_results:
        panel_defs.append(("grooming", "Grooming Codes"))
    # Always add sustained rearing (uses _select_top_clips_for_posture
    # with "sustained_rearing" scorer — selects clips with highest mean Z)
    panel_defs.append(("sustained_rearing", "Sustained Rearing Codes"))

    if not panel_defs:
        logging.warning("  Killer demo: no conditions available")
        return None

    logging.info(
        f"\nRunning killer demo (K={killer_K}): "
        f"{', '.join(d[1] for d in panel_defs)} from rearing pose"
    )

    # Use the rearing env + anchor qpos for all panels
    jit_reset = jax.jit(rearing_env.reset)
    jit_step = jax.jit(rearing_env.step)

    # Run each condition
    panels_for_video = []  # For render_multi_panel_video (code swap)
    panels_for_ref = []  # For render_multi_panel_video (H5 reference)
    conditions_for_plot = []  # For plot_killer_demo_displacement
    all_panel_metrics = []  # For HTML

    killer_colors = _TRAJECTORY_COLORS[:killer_K]

    for posture_key, label in panel_defs:
        logging.info(f"  Selecting {killer_K} {posture_key} clips...")
        clip_idxs, start_frs, code_seqs = _select_top_clips_for_posture(
            rollouts, posture_key, killer_K, cfg
        )

        logging.info(f"  Running {killer_K} {posture_key} codes from rearing pose...")
        results = run_decoder_condition(
            env=rearing_env,
            jit_decode=jit_decode,
            jit_reset=jit_reset,
            jit_step=jit_step,
            code_sequences=code_seqs,
            max_steps=max_steps,
            seed=seed,
            initial_qpos=anchor_h5_qpos,
        )

        trajs = [r["qpos"] for r in results]
        code_arrs = [r["code_indices"] for r in results]
        metrics = compute_divergence_metrics(results)

        panels_for_video.append(
            {"trajectories": trajs, "codes": code_arrs, "label": label}
        )
        conditions_for_plot.append((label, results))
        all_panel_metrics.append((label, metrics))

        # Collect original H5 inference trajectories for verification
        ref_trajs = []
        ref_codes = []
        for ci, sf in zip(clip_idxs, start_frs):
            r = rollouts[ci]
            ref_trajs.append(r.qpos[sf : sf + max_steps])
            if r.rvq_indices is not None and len(r.rvq_indices) > 0:
                ref_codes.append(np.array(r.rvq_indices[0][sf : sf + max_steps]))
            else:
                ref_codes.append(np.array(r.code_indices[sf : sf + max_steps]))
        ref_label = label.replace("Codes", "Reference")
        panels_for_ref.append(
            {"trajectories": ref_trajs, "codes": ref_codes, "label": ref_label}
        )

    # Render multi-panel code swap video
    video_path = render_multi_panel_video(
        env=rearing_env,
        panels=panels_for_video,
        trajectory_colors=killer_colors,
        output_path=output_dir / "code_swap_side_by_side.mp4",
        cfg=cfg,
        code_colors=code_colors,
    )

    # Render multi-panel reference verification video
    # Shows original H5 inference trajectories for each posture's K clips
    ref_video_path = render_multi_panel_video(
        env=rearing_env,
        panels=panels_for_ref,
        trajectory_colors=killer_colors,
        output_path=output_dir / "reference_verification.mp4",
        cfg=cfg,
        code_colors=code_colors,
    )

    # Plot mean cumulative root displacement comparison
    plot_path = plot_killer_demo_displacement(
        conditions=conditions_for_plot,
        output_path=output_dir / "root_displacement.png",
    )

    # Build summary HTML
    plot_data_uri = _encode_file_b64(plot_path, "image/png")
    video_data_uri = _encode_file_b64(video_path, "video/mp4")
    ref_video_data_uri = _encode_file_b64(ref_video_path, "video/mp4")

    metrics_lines = []
    for label, metrics in all_panel_metrics:
        metrics_lines.append(
            f'<p class="info">{label}: '
            f"reward={metrics['mean_reward']:.1f}, "
            f"survival={metrics['mean_survival']:.0f}, "
            f"mean displacement="
            f"{np.mean(metrics['root_displacements']):.3f}</p>"
        )

    panel_labels_str = " | ".join(d[1] for d in panel_defs)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{demo_label} — Code Swap</title>
<style>
body {{ font-family: sans-serif; margin: 20px; background: #fafafa; }}
h2 {{ color: #333; }}
.section {{ margin: 16px 0; padding: 12px; background: #fff;
           border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.info {{ font-size: 13px; color: #666; margin: 4px 0; }}
</style>
</head><body>
<h2>{demo_label}: Code Swap (K={killer_K})</h2>
<div class="section">
<h3>Setup</h3>
<p class="info">All panels start from the same pose
(clip {anchor_clip_idx}, frame {anchor_frame},
root_z={anchor_h5_qpos[2]:.4f}, {anchor_z_frac:.0%} of peak)</p>
<p class="info">Panels: {panel_labels_str} ({killer_K} trajectories each)</p>
</div>
<div class="section">
<h3>Side-by-Side Video (Code Swap)</h3>
<video src="{video_data_uri}" width="100%"
       autoplay loop muted controls></video>
</div>
<div class="section">
<h3>Reference Verification (Original H5 Inference)</h3>
<p class="info">Shows the original behavior of the K clips whose codes
are used in each panel above. Confirms walking clips walk, grooming clips
groom, and sustained rearing clips rear.</p>
<video src="{ref_video_data_uri}" width="100%"
       autoplay loop muted controls></video>
</div>
<div class="section">
<h3>Root Displacement (Mean over {killer_K} trajectories)</h3>
<img src="{plot_data_uri}" style="max-width:100%;" />
</div>
<div class="section">
<h3>Metrics</h3>
{''.join(metrics_lines)}
</div>
</body></html>"""

    html_path = output_dir / "killer_demo.html"
    with open(html_path, "w") as f:
        f.write(html)
    logging.info(f"  Killer demo HTML: {html_path}")

    return {
        "video_path": video_path,
        "ref_video_path": ref_video_path,
        "plot_path": plot_path,
        "html_str": html,
        "html_path": str(html_path),
        "panel_labels": [d[1] for d in panel_defs],
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

# Trajectory colors: opaque primary + semi-transparent ghosts
_TRAJECTORY_COLORS = [
    [0.0, 0.5, 0.5, 1.0],  # Teal (primary body)
    [0.9, 0.3, 0.1, 0.35],  # Red-orange ghost
    [0.1, 0.3, 0.9, 0.35],  # Blue ghost
    [0.1, 0.8, 0.2, 0.35],  # Green ghost
    [0.8, 0.1, 0.8, 0.35],  # Purple ghost
    [0.9, 0.8, 0.1, 0.35],  # Yellow ghost
    [0.1, 0.7, 0.7, 0.35],  # Cyan ghost
    [0.7, 0.2, 0.5, 0.35],  # Magenta ghost
    [0.5, 0.5, 0.1, 0.35],  # Olive ghost
    [0.3, 0.1, 0.6, 0.35],  # Indigo ghost
    [0.9, 0.5, 0.3, 0.35],  # Peach ghost
]


@hydra.main(
    version_base=None, config_path="../configs", config_name="divergent_futures"
)
def main(cfg: DictConfig):
    """Run divergent futures experiment for multiple postures."""
    logging.set_verbosity(logging.INFO)

    print("=" * 60)
    print("Divergent Futures Experiment (Multi-Posture)")
    print("=" * 60)

    output_dir = Path(cfg.output.base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    K = int(cfg.experiment.K)
    max_steps = int(cfg.experiment.max_steps)
    seed = int(cfg.experiment.seed)

    # --- Load checkpoint ---
    logging.info("\nLoading checkpoint...")
    ckpt = load_vq_checkpoint(cfg.checkpoint.path, step=cfg.checkpoint.step)
    vq_cfg = ckpt["cfg"]
    policy_params = ckpt["policy"]

    codebooks = get_all_codebooks(policy_params)
    num_codes = codebooks[0].shape[0]
    logging.info(f"  {num_codes} codes, {len(codebooks)} depth(s)")

    code_colors = get_nature_colormap(num_codes)
    trajectory_colors = _TRAJECTORY_COLORS[:K]

    # Zero continuous encoder if present
    use_continuous_latent = bool(
        vq_cfg.network_config.get("use_continuous_latent", False)
    )
    if use_continuous_latent:
        logging.info("  Zeroing continuous encoder head")
        policy_params = zero_continuous_encoder_params(policy_params)

    # --- Load H5 data ---
    logging.info("\nLoading H5 inference data...")
    h5_path = cfg.data.h5_path_test
    rollouts, h5_metadata = load_rollouts_from_h5(h5_path)
    logging.info(f"  {len(rollouts)} rollouts loaded")

    # --- Build decoder-only step fn ---
    logging.info("\nBuilding decoder-only step function...")
    decode_step, action_size = make_decoder_only_step_fn(vq_cfg, policy_params)
    jit_decode = jax.jit(decode_step)

    # --- Prepare test clips (shared across postures) ---
    reference_clips = ReferenceClips(
        data_path=vq_cfg.env_config.reference_data_path,
        n_frames_per_clip=vq_cfg.env_config.clip_length,
        keep_clips_idx=vq_cfg.env_config.get("keep_clips_idx", None),
    )
    train_ratio = float(vq_cfg.train_setup.get("train_subset_ratio", 1.0))
    train_seed = int(vq_cfg.train_setup.train_config.get("seed", 0))
    key_split, _ = jax.random.split(jax.random.PRNGKey(train_seed))
    _, test_clips = reference_clips.split(train_ratio=train_ratio, seed=key_split)

    # --- Init WandB ---
    wandb_enabled = init_wandb(cfg)

    # --- Run posture experiments ---
    postures = list(cfg.experiment.get("postures", ["rearing"]))
    all_posture_results: dict[str, dict] = {}

    for posture in postures:
        posture_dir = output_dir / posture
        result = run_posture_experiment(
            posture=posture,
            rollouts=rollouts,
            K=K,
            cfg=cfg,
            jit_decode=jit_decode,
            vq_cfg=vq_cfg,
            policy_params=policy_params,
            num_codes=num_codes,
            code_colors=code_colors,
            trajectory_colors=trajectory_colors,
            output_dir=posture_dir,
            seed=seed,
            max_steps=max_steps,
            test_clips=test_clips,
        )
        all_posture_results[posture] = result

    # --- Killer demos (two starting heights) ---
    killer_results = {}  # key → result dict
    killer_demo_cfg = cfg.experiment.get("killer_demo", {})
    if killer_demo_cfg.get("enabled", True):
        anchor_frac = float(killer_demo_cfg.get("anchor_z_fraction", 0.5))
        low_frac = float(killer_demo_cfg.get("low_anchor_z_fraction", 0.1))

        killer_demo_args = dict(
            all_posture_results=all_posture_results,
            rollouts=rollouts,
            jit_decode=jit_decode,
            cfg=cfg,
            vq_cfg=vq_cfg,
            code_colors=code_colors,
            trajectory_colors=trajectory_colors,
            seed=seed,
            max_steps=max_steps,
            test_clips=test_clips,
        )

        # Moderate rearing start
        killer_results["rear"] = run_killer_demo(
            **killer_demo_args,
            output_dir=output_dir / "killer_demo_rear",
            anchor_z_frac_override=anchor_frac,
            demo_label="Killer Demo (Rearing Start)",
        )

        # Low / standing start
        killer_results["low"] = run_killer_demo(
            **killer_demo_args,
            output_dir=output_dir / "killer_demo_low",
            anchor_z_frac_override=low_frac,
            demo_label="Killer Demo (Low Start)",
        )

    # --- Save JSON summary ---
    summary = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint_path": str(cfg.checkpoint.path),
        "postures": postures,
    }
    for posture, result in all_posture_results.items():
        summary[posture] = {
            "group_info": {
                k: v if not isinstance(v, np.ndarray) else v.tolist()
                for k, v in result["group_info"].items()
            },
            "metrics": {
                cond: {
                    "mean_reward": m["mean_reward"],
                    "mean_survival": m["mean_survival"],
                    "root_displacements": m["root_displacements"],
                    "root_z_ranges": m["root_z_ranges"],
                }
                for cond, m in result["metrics_by_condition"].items()
            },
        }
    json_path = output_dir / "divergent_futures_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # --- WandB logging ---
    if wandb_enabled:
        import wandb

        wandb_items: dict[str, Any] = {}

        for posture, result in all_posture_results.items():
            prefix = f"divergent_{posture}"

            for name, path in result["video_paths"].items():
                safe = name.replace(": ", "_").replace(" ", "_").lower()
                wandb_items[f"{prefix}/{safe}"] = wandb.Video(path, format="mp4")

            for name, path in result["plot_paths"].items():
                safe = name.replace(" ", "_").lower()
                wandb_items[f"{prefix}/{safe}"] = wandb.Image(path)

            wandb_items[f"{prefix}/summary"] = wandb.Html(result["html_str"])

        for demo_key, kr in killer_results.items():
            if kr is None:
                continue
            prefix = f"divergent_killer_{demo_key}"
            wandb_items[f"{prefix}/code_swap_side_by_side"] = wandb.Video(
                kr["video_path"], format="mp4"
            )
            wandb_items[f"{prefix}/reference_verification"] = wandb.Video(
                kr["ref_video_path"], format="mp4"
            )
            wandb_items[f"{prefix}/root_displacement"] = wandb.Image(
                kr["plot_path"]
            )
            wandb_items[f"{prefix}/summary"] = wandb.Html(kr["html_str"])

        if wandb_items and wandb.run is not None:
            wandb.log(wandb_items)
        wandb.finish()

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("Divergent Futures experiment complete!")
    print(f"Results: {output_dir}")
    for posture, result in all_posture_results.items():
        print(f"\n  [{posture}]")
        for cond, m in result["metrics_by_condition"].items():
            print(
                f"    {cond}: reward={m['mean_reward']:.1f}, "
                f"survival={m['mean_survival']:.0f}"
            )
    for demo_key, kr in killer_results.items():
        if kr is not None:
            print(f"\n  Killer demo ({demo_key}): {kr['video_path']}")
    print(f"\nJSON: {json_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
