"""VQ-VAE Code Ablation Experiments.

Empirically test what each D0 code does via codebook mutation:
1. Null ablation — force z_q=0 everywhere to confirm model stops moving
2. Code injection — force one D0 code at every timestep
3. Burst truncation — cap D0 burst duration to test impulse timing

Each experiment runs from two starting poses (lowest/highest torso z-height).
Results logged to WandB.

Usage:
    cd vqvae_jax
    WANDB_MODE=offline python run_ablation.py checkpoint.path=<path>
"""

import os

# Must set rendering backend before importing MuJoCo
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from omegaconf import DictConfig
from vnl_playground.tasks.rodent import imitation
from vnl_playground.tasks.rodent.reference_clips import ReferenceClips

from track_mjx.agent.observation_utils import flatten_obs_dict
from track_mjx.config import utils as config_utils

from analysis.checkpoint_utils import (
    get_all_codebooks,
    load_vq_checkpoint,
    load_vq_inference_fn_with_stickiness,
)
from analysis.correction_analysis import identify_null_code
from analysis.inference_cache import InferenceResult
from analysis.rendering import render_rollout_to_video


# =============================================================================
# D0 CODEBOOK MUTATION
# =============================================================================
#
# At inference, _quantize_single_level() computes z_q = codebook[argmin(dist)].
# Setting all D0 codebook entries to the same vector forces that z_q regardless
# of z_e.  L1 continues to operate normally on the residual.
# =============================================================================


def make_null_d0_params(
    policy_params: tuple[Any, Any],
) -> tuple[Any, Any]:
    """Set all D0 codebook entries to zeros.

    Args:
        policy_params: Tuple of (normalizer_state, policy_params_dict).

    Returns:
        New policy_params with zeroed D0 codebook. L1 codebook untouched.
    """
    normalizer, params = policy_params
    quantizer = params["params"]["quantizer"]
    new_cb0 = dict(quantizer["codebooks_0"])
    new_cb0["embeddings"] = jnp.zeros_like(new_cb0["embeddings"])
    new_quantizer = dict(quantizer)
    new_quantizer["codebooks_0"] = new_cb0
    new_params = dict(params)
    new_params["params"] = dict(params["params"])
    new_params["params"]["quantizer"] = new_quantizer
    return (normalizer, new_params)


def make_injection_d0_params(
    policy_params: tuple[Any, Any],
    target_embedding: jnp.ndarray,
) -> tuple[Any, Any]:
    """Set all D0 codebook entries to a single target embedding.

    Args:
        policy_params: Tuple of (normalizer_state, policy_params_dict).
        target_embedding: Embedding vector to tile across all D0 entries.

    Returns:
        New policy_params with all D0 entries = target_embedding.
    """
    normalizer, params = policy_params
    quantizer = params["params"]["quantizer"]
    old_embeddings = quantizer["codebooks_0"]["embeddings"]
    num_codes = old_embeddings.shape[0]
    new_embeddings = jnp.tile(target_embedding[None, :], (num_codes, 1))
    new_cb0 = dict(quantizer["codebooks_0"])
    new_cb0["embeddings"] = new_embeddings
    new_quantizer = dict(quantizer)
    new_quantizer["codebooks_0"] = new_cb0
    new_params = dict(params)
    new_params["params"] = dict(params["params"])
    new_params["params"]["quantizer"] = new_quantizer
    return (normalizer, new_params)


# =============================================================================
# STARTING POSE SELECTION
# =============================================================================


def select_starting_clips(clips: ReferenceClips) -> dict[str, int]:
    """Find clips with extreme initial torso z-heights.

    Examines qpos[:, 0, 2] (first frame, z component of root position)
    across all clips.

    Args:
        clips: Reference clips with qpos shape [num_clips, T, nq].

    Returns:
        Dict mapping pose name to clip index within ``clips``.
    """
    initial_z = np.array(clips.qpos[:, 0, 2])
    low_idx = int(np.argmin(initial_z))
    high_idx = int(np.argmax(initial_z))
    logging.info(f"  Low-height clip: {low_idx} (z={initial_z[low_idx]:.4f})")
    logging.info(f"  High-height clip: {high_idx} (z={initial_z[high_idx]:.4f})")
    return {"low_height": low_idx, "high_height": high_idx}


def subset_clips(clips: ReferenceClips, idx: int) -> ReferenceClips:
    """Create a single-clip ReferenceClips subset.

    Args:
        clips: Full reference clips.
        idx: Index of the clip to extract.

    Returns:
        New ReferenceClips containing only clip ``idx``.
    """
    sub = copy.copy(clips)
    sub._data_arrays = {
        k: clips._data_arrays[k][idx : idx + 1] for k in clips._DATA_ARRAYS
    }
    return sub


# =============================================================================
# ROLLOUT FUNCTIONS
# =============================================================================


def run_ablation_rollout(
    env: Any,
    inference_fn: Any,
    num_repeats: int,
    max_steps: int,
    seed: int,
    rvq_depth: int,
    num_render: int = 0,
) -> list[InferenceResult]:
    """Run ablation rollouts on a single-clip environment.

    Args:
        env: Imitation environment (single clip).
        inference_fn: VQ-VAE inference function with stickiness support.
        num_repeats: Number of rollout repeats (different random seeds).
        max_steps: Maximum steps per rollout.
        seed: Base random seed.
        rvq_depth: Number of RVQ depth levels.
        num_render: Number of rollouts for which to store env states
            (for video rendering). States are only stored for the first
            ``num_render`` rollouts to save memory.

    Returns:
        List of InferenceResult objects.
    """
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    results = []
    rng = jax.random.PRNGKey(seed)

    for i in range(num_repeats):
        rng, reset_rng = jax.random.split(rng)
        state = jit_reset(reset_rng)

        code_indices: list[int] = []
        rvq_per_depth: list[list[int]] = [[] for _ in range(rvq_depth)]
        qpos_list: list[np.ndarray] = []
        qvel_list: list[np.ndarray] = []
        rewards: list[float] = []
        store_states = i < num_render
        states: list[Any] | None = [] if store_states else None
        prev_indices = None

        for step in range(max_steps):
            obs = flatten_obs_dict(state.obs)
            rng, action_rng = jax.random.split(rng)
            action, extras = inference_fn(obs, action_rng, prev_indices)

            code_idx = int(extras["indices"])
            code_indices.append(code_idx)

            all_idx = extras.get("all_indices")
            if all_idx is not None:
                prev_indices = all_idx
                for d in range(rvq_depth):
                    if isinstance(all_idx, tuple) and d < len(all_idx):
                        rvq_per_depth[d].append(int(all_idx[d]))
                    elif d == 0:
                        rvq_per_depth[d].append(code_idx)
            else:
                prev_indices = jnp.array(code_idx)
                rvq_per_depth[0].append(code_idx)

            if hasattr(state, "data"):
                qpos_list.append(np.array(state.data.qpos))
                qvel_list.append(np.array(state.data.qvel))
            elif hasattr(state, "pipeline_state"):
                qpos_list.append(np.array(state.pipeline_state.q))
                qvel_list.append(np.array(state.pipeline_state.qd))

            if store_states:
                states.append(state)

            next_state = jit_step(state, action)
            rewards.append(float(next_state.reward))

            if next_state.done:
                break
            state = next_state

        rvq_indices = None
        if rvq_depth > 1 and rvq_per_depth[0]:
            rvq_indices = tuple(np.array(rvq_per_depth[d]) for d in range(rvq_depth))

        results.append(
            InferenceResult(
                clip_idx=i,
                code_indices=np.array(code_indices),
                qpos=np.stack(qpos_list) if qpos_list else np.zeros((0, 0)),
                qvel=np.stack(qvel_list) if qvel_list else np.zeros((0, 0)),
                rewards=np.array(rewards),
                states=states,
                rvq_indices=rvq_indices,
            )
        )

    return results


def run_burst_truncation_rollout(
    env: Any,
    normal_fn: Any,
    null_d0_fn: Any,
    num_repeats: int,
    max_steps: int,
    seed: int,
    rvq_depth: int,
    max_burst_length: int,
    null_code: int,
    num_render: int = 0,
) -> list[InferenceResult]:
    """Run burst truncation rollout with dual inference functions.

    Tracks D0 burst length (consecutive non-null frames). When the burst
    reaches ``max_burst_length``, switches to the null D0 inference function
    for that timestep.

    Args:
        env: Imitation environment (single clip).
        normal_fn: Normal VQ-VAE inference function.
        null_d0_fn: Inference function with zeroed D0 codebook.
        num_repeats: Number of rollout repeats.
        max_steps: Maximum steps per rollout.
        seed: Base random seed.
        rvq_depth: Number of RVQ depth levels.
        max_burst_length: Maximum consecutive non-null D0 frames.
        null_code: Index of the null (most frequent) D0 code.
        num_render: Number of rollouts to store states for.

    Returns:
        List of InferenceResult objects.
    """
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    results = []
    rng = jax.random.PRNGKey(seed)

    for i in range(num_repeats):
        rng, reset_rng = jax.random.split(rng)
        state = jit_reset(reset_rng)

        code_indices: list[int] = []
        rvq_per_depth: list[list[int]] = [[] for _ in range(rvq_depth)]
        qpos_list: list[np.ndarray] = []
        qvel_list: list[np.ndarray] = []
        rewards: list[float] = []
        store_states = i < num_render
        states: list[Any] | None = [] if store_states else None
        prev_indices = None
        burst_counter = 0

        for step in range(max_steps):
            obs = flatten_obs_dict(state.obs)
            rng, action_rng = jax.random.split(rng)

            # Switch to null D0 when burst exceeds limit
            if burst_counter >= max_burst_length:
                action, extras = null_d0_fn(obs, action_rng, prev_indices)
                forced_null = True
            else:
                action, extras = normal_fn(obs, action_rng, prev_indices)
                forced_null = False

            code_idx = int(extras["indices"])

            # Record actual D0 code and update burst counter
            if forced_null:
                actual_d0 = null_code
                burst_counter = 0
            else:
                actual_d0 = code_idx
                if actual_d0 != null_code:
                    burst_counter += 1
                else:
                    burst_counter = 0

            code_indices.append(actual_d0)

            # Track per-depth indices
            all_idx = extras.get("all_indices")
            if all_idx is not None:
                prev_indices = all_idx
                for d in range(rvq_depth):
                    if d == 0:
                        # Override D0 with actual_d0 for burst tracking accuracy
                        rvq_per_depth[d].append(actual_d0)
                    elif isinstance(all_idx, tuple) and d < len(all_idx):
                        rvq_per_depth[d].append(int(all_idx[d]))
            else:
                prev_indices = jnp.array(actual_d0)
                rvq_per_depth[0].append(actual_d0)

            if hasattr(state, "data"):
                qpos_list.append(np.array(state.data.qpos))
                qvel_list.append(np.array(state.data.qvel))
            elif hasattr(state, "pipeline_state"):
                qpos_list.append(np.array(state.pipeline_state.q))
                qvel_list.append(np.array(state.pipeline_state.qd))

            if store_states:
                states.append(state)

            next_state = jit_step(state, action)
            rewards.append(float(next_state.reward))

            if next_state.done:
                break
            state = next_state

        rvq_indices = None
        if rvq_depth > 1 and rvq_per_depth[0]:
            rvq_indices = tuple(np.array(rvq_per_depth[d]) for d in range(rvq_depth))

        results.append(
            InferenceResult(
                clip_idx=i,
                code_indices=np.array(code_indices),
                qpos=np.stack(qpos_list) if qpos_list else np.zeros((0, 0)),
                qvel=np.stack(qvel_list) if qvel_list else np.zeros((0, 0)),
                rewards=np.array(rewards),
                states=states,
                rvq_indices=rvq_indices,
            )
        )

    return results


# =============================================================================
# METRICS
# =============================================================================


def compute_condition_metrics(
    results: list[InferenceResult],
    null_code: int | None = None,
) -> dict[str, float]:
    """Compute summary metrics for a set of rollout results.

    Args:
        results: Rollout results for one condition.
        null_code: If provided, compute null D0 frame fraction.

    Returns:
        Dict of metric name to value.
    """
    rewards = [float(np.sum(r.rewards)) for r in results]
    lengths = [len(r.rewards) for r in results]

    displacements = []
    for r in results:
        if len(r.qpos) >= 2:
            delta = r.qpos[-1, :2] - r.qpos[0, :2]
            displacements.append(float(np.linalg.norm(delta)))
        else:
            displacements.append(0.0)

    metrics = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_episode_length": float(np.mean(lengths)),
        "std_episode_length": float(np.std(lengths)),
        "mean_root_displacement": float(np.mean(displacements)),
        "std_root_displacement": float(np.std(displacements)),
    }

    if null_code is not None:
        null_fracs = []
        for r in results:
            n_null = int(np.sum(r.code_indices == null_code))
            null_fracs.append(n_null / max(len(r.code_indices), 1))
        metrics["null_d0_fraction"] = float(np.mean(null_fracs))

    return metrics


# =============================================================================
# PLOTTING
# =============================================================================


def plot_comparison(
    all_metrics: dict[str, dict[str, float]],
    output_dir: Path,
) -> dict[str, str]:
    """Create grouped bar charts comparing conditions across poses.

    Args:
        all_metrics: Mapping from ``"condition/pose"`` to metrics dict.
        output_dir: Directory to save plots.

    Returns:
        Mapping from plot name to file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    # Group by condition name
    conditions: dict[str, dict[str, dict]] = {}
    for key, metrics in all_metrics.items():
        parts = key.rsplit("/", 1)
        condition = parts[0]
        pose = parts[1] if len(parts) > 1 else "all"
        if condition not in conditions:
            conditions[condition] = {}
        conditions[condition][pose] = metrics

    poses = ["low_height", "high_height"]
    metric_defs = [
        ("mean_reward", "std_reward", "Mean Reward"),
        ("mean_episode_length", "std_episode_length", "Mean Episode Length"),
        ("mean_root_displacement", "std_root_displacement", "Mean Root Displacement"),
    ]

    for metric_key, std_key, metric_label in metric_defs:
        fig, ax = plt.subplots(figsize=(max(10, len(conditions) * 1.5), 6))

        cond_names = list(conditions.keys())
        x = np.arange(len(cond_names))
        bar_width = 0.35

        for i, pose in enumerate(poses):
            values = []
            errors = []
            for cond in cond_names:
                m = conditions[cond].get(pose, {})
                values.append(m.get(metric_key, 0))
                errors.append(m.get(std_key, 0))

            offset = (i - 0.5) * bar_width
            ax.bar(
                x + offset,
                values,
                bar_width,
                yerr=errors,
                label=pose,
                capsize=3,
            )

        ax.set_xlabel("Condition")
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} by Condition and Starting Pose")
        ax.set_xticks(x)
        ax.set_xticklabels(cond_names, rotation=45, ha="right")
        ax.legend()
        plt.tight_layout()

        fig_path = output_dir / f"{metric_key}_bars.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths[f"{metric_key}_bars"] = str(fig_path)

    return paths


# =============================================================================
# WANDB HELPERS
# =============================================================================


def init_wandb(cfg: DictConfig) -> bool:
    """Initialize WandB for ablation experiments."""
    log_cfg = cfg.logging
    if not log_cfg.get("wandb_enabled", False):
        return False
    try:
        import wandb

        run_name = log_cfg.get("run_name") or (
            f"ablation_{datetime.now().strftime('%y%m%d_%H%M%S')}"
        )
        wandb.init(
            project=log_cfg.get("project_name", "vqvae_ablation"),
            group=log_cfg.get("group_name", "ablation_experiments"),
            name=run_name,
            config={
                "checkpoint_path": str(cfg.checkpoint.path),
                "checkpoint_step": cfg.checkpoint.step,
                "experiments": list(cfg.ablation.experiments),
                "num_clips": cfg.ablation.num_clips,
                "max_steps": cfg.ablation.max_steps,
                "top_k": cfg.ablation.top_k,
            },
        )
        return True
    except Exception as e:
        logging.warning(f"Failed to init WandB: {e}")
        return False


def log_wandb(key: str, value: Any, wandb_enabled: bool) -> None:
    """Log a single item to WandB if enabled."""
    if not wandb_enabled:
        return
    try:
        import wandb

        if wandb.run is not None:
            wandb.log({key: value})
    except Exception as e:
        logging.warning(f"Failed to log {key}: {e}")


# =============================================================================
# VIDEO RENDERING HELPER
# =============================================================================


def render_condition_videos(
    results: list[InferenceResult],
    env: Any,
    condition_key: str,
    output_dir: Path,
    camera: str,
    num_codes: int,
    cfg: DictConfig,
    wandb_enabled: bool,
    num_videos: int = 3,
) -> None:
    """Render and log videos for a set of rollout results.

    Args:
        results: Rollout results (must have states stored).
        env: Environment for rendering.
        condition_key: WandB/file prefix, e.g. "null_ablation/low_height".
        output_dir: Directory for video files.
        camera: Camera name.
        num_codes: Number of codes for colormap.
        cfg: Config with render section.
        wandb_enabled: Whether WandB is active.
        num_videos: Max videos to render.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_key = condition_key.replace("/", "_")

    for vi in range(min(num_videos, len(results))):
        r = results[vi]
        if not r.states:
            continue

        video_path = output_dir / f"{safe_key}_{vi}.mp4"
        indices_per_depth = None
        if r.rvq_indices is not None and len(r.rvq_indices) > 1:
            indices_per_depth = [np.array(a) for a in r.rvq_indices]

        render_rollout_to_video(
            env=env,
            rollout_states=r.states,
            output_path=video_path,
            camera=camera,
            width=cfg.render.width,
            height=cfg.render.height,
            fps=cfg.render.fps,
            indices=r.code_indices,
            num_codes=num_codes,
            rewards=r.rewards,
            clip_idx=r.clip_idx,
            indices_per_depth=indices_per_depth,
        )

        if wandb_enabled:
            import wandb

            log_wandb(
                f"ablation/{condition_key}/video_{vi}",
                wandb.Video(str(video_path), format="mp4"),
                wandb_enabled,
            )


# =============================================================================
# MAIN PIPELINE
# =============================================================================


@hydra.main(version_base=None, config_path="configs", config_name="ablation")
def main(cfg: DictConfig):
    """Run VQ-VAE code ablation experiments."""
    logging.set_verbosity(logging.INFO)

    print("=" * 60)
    print("VQ-VAE Code Ablation Experiments")
    print("=" * 60)

    output_dir = Path(cfg.output.base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load checkpoint ---
    logging.info("\nLoading checkpoint...")
    ckpt = load_vq_checkpoint(cfg.checkpoint.path, step=cfg.checkpoint.step)
    vq_cfg = ckpt["cfg"]
    policy_params = ckpt["policy"]

    codebooks = get_all_codebooks(policy_params)
    d0_codebook = np.array(codebooks[0])
    num_codes = d0_codebook.shape[0]
    rvq_depth = len(codebooks)
    logging.info(
        f"  {num_codes} codes, {d0_codebook.shape[1]} dims, {rvq_depth} depth(s)"
    )

    # --- Create normal inference fn ---
    inference_fn, _ = load_vq_inference_fn_with_stickiness(
        vq_cfg, policy_params, deterministic=True
    )

    # --- Load reference clips and select starting poses ---
    logging.info("\nLoading reference clips and selecting starting poses...")
    (_, cfg_dict, env_cfg_ml) = config_utils.prepare_config(cfg)

    data_split = cfg.ablation.get("data_split", "test")
    if data_split in ("train", "test"):
        reference_clips = ReferenceClips(
            data_path=vq_cfg.env_config.reference_data_path,
            n_frames_per_clip=vq_cfg.env_config.clip_length,
            keep_clips_idx=vq_cfg.env_config.get("keep_clips_idx", None),
        )
        train_ratio = float(vq_cfg.train_setup.get("train_subset_ratio", 1.0))
        train_seed = int(vq_cfg.train_setup.train_config.get("seed", 0))
        key_split, _ = jax.random.split(jax.random.PRNGKey(train_seed))
        train_clips, test_clips = reference_clips.split(
            train_ratio=train_ratio, seed=key_split
        )
        clips = train_clips if data_split == "train" else test_clips
    else:
        clips = ReferenceClips(
            data_path=vq_cfg.env_config.reference_data_path,
            n_frames_per_clip=vq_cfg.env_config.get("clip_length", 250),
            keep_clips_idx=None,
        )

    starting_clips = select_starting_clips(clips)

    # Create per-pose single-clip environments
    pose_envs: dict[str, Any] = {}
    for pose_name, clip_idx in starting_clips.items():
        pose_cfg = cfg.ablation.starting_poses.get(pose_name, {})
        if not pose_cfg.get("enabled", True):
            continue
        single = subset_clips(clips, clip_idx)
        pose_envs[pose_name] = imitation.Imitation(config=env_cfg_ml, clips=single)
        logging.info(f"  Created env for {pose_name} (clip {clip_idx})")

    # --- Init WandB ---
    wandb_enabled = init_wandb(cfg)

    # --- Config shortcuts ---
    env_suffix = "-rodent"
    camera_name = f"{cfg.render.camera}{env_suffix}"
    num_clips = cfg.ablation.num_clips
    max_steps = cfg.ablation.max_steps
    seed = cfg.ablation.seed
    top_k = cfg.ablation.top_k
    render_enabled = cfg.render.get("enabled", True)
    num_render = cfg.render.get("num_videos", 3)
    experiments = list(cfg.ablation.experiments)

    all_metrics: dict[str, dict[str, float]] = {}

    # ================================================================
    # STEP 1: Baseline rollout
    # ================================================================
    logging.info("\n" + "=" * 40)
    logging.info("Running baseline rollouts...")

    baseline_results: dict[str, list[InferenceResult]] = {}
    for pose_name, env in pose_envs.items():
        logging.info(f"  Baseline on {pose_name}...")
        results = run_ablation_rollout(
            env=env,
            inference_fn=inference_fn,
            num_repeats=num_clips,
            max_steps=max_steps,
            seed=seed,
            rvq_depth=rvq_depth,
            num_render=num_render if render_enabled else 0,
        )
        baseline_results[pose_name] = results

    # Identify null code and top-K from pooled baseline
    all_baseline = [r for rs in baseline_results.values() for r in rs]
    null_code = identify_null_code(all_baseline)
    logging.info(f"  Null code (most frequent): {null_code}")

    all_codes = np.concatenate([r.code_indices for r in all_baseline])
    code_counts = np.bincount(all_codes, minlength=num_codes)
    code_counts[null_code] = 0  # exclude null
    top_k_indices = np.argsort(code_counts)[::-1][:top_k]
    top_k_codes = [
        (int(idx), int(code_counts[idx]))
        for idx in top_k_indices
        if code_counts[idx] > 0
    ]
    logging.info(f"  Top-{top_k} non-null D0 codes: {[c[0] for c in top_k_codes]}")

    # Log baseline metrics + videos
    for pose_name, results in baseline_results.items():
        key = f"baseline/{pose_name}"
        metrics = compute_condition_metrics(results, null_code)
        all_metrics[key] = metrics
        logging.info(
            f"  {key}: reward={metrics['mean_reward']:.1f}, "
            f"length={metrics['mean_episode_length']:.0f}, "
            f"displacement={metrics['mean_root_displacement']:.3f}"
        )
        if wandb_enabled:
            for mk, mv in metrics.items():
                log_wandb(f"ablation/{key}/{mk}", mv, wandb_enabled)

        if render_enabled:
            render_condition_videos(
                results,
                pose_envs[pose_name],
                key,
                output_dir,
                camera_name,
                num_codes,
                cfg,
                wandb_enabled,
                num_render,
            )

    # ================================================================
    # STEP 2: Null ablation
    # ================================================================
    if "null_ablation" in experiments:
        logging.info("\n" + "=" * 40)
        logging.info("Running null ablation (all D0 codes -> zero)...")

        null_params = make_null_d0_params(policy_params)
        null_fn, _ = load_vq_inference_fn_with_stickiness(
            vq_cfg, null_params, deterministic=True
        )

        for pose_name, env in pose_envs.items():
            logging.info(f"  Null ablation on {pose_name}...")
            results = run_ablation_rollout(
                env=env,
                inference_fn=null_fn,
                num_repeats=num_clips,
                max_steps=max_steps,
                seed=seed,
                rvq_depth=rvq_depth,
                num_render=num_render if render_enabled else 0,
            )

            key = f"null_ablation/{pose_name}"
            metrics = compute_condition_metrics(results, null_code)
            all_metrics[key] = metrics
            logging.info(
                f"  {key}: reward={metrics['mean_reward']:.1f}, "
                f"length={metrics['mean_episode_length']:.0f}, "
                f"displacement={metrics['mean_root_displacement']:.3f}"
            )
            if wandb_enabled:
                for mk, mv in metrics.items():
                    log_wandb(f"ablation/{key}/{mk}", mv, wandb_enabled)

            if render_enabled:
                render_condition_videos(
                    results,
                    env,
                    key,
                    output_dir,
                    camera_name,
                    num_codes,
                    cfg,
                    wandb_enabled,
                    num_render,
                )

    # ================================================================
    # STEP 3: Code injection (top-K non-null D0 codes)
    # ================================================================
    if "code_injection" in experiments:
        logging.info("\n" + "=" * 40)
        logging.info("Running code injection experiments...")

        for code_idx, code_count in top_k_codes:
            logging.info(
                f"  Injecting code {code_idx} ({code_count} baseline frames)..."
            )
            target_embedding = jnp.array(d0_codebook[code_idx])
            inj_params = make_injection_d0_params(policy_params, target_embedding)
            inj_fn, _ = load_vq_inference_fn_with_stickiness(
                vq_cfg, inj_params, deterministic=True
            )

            for pose_name, env in pose_envs.items():
                logging.info(f"    Code {code_idx} on {pose_name}...")
                results = run_ablation_rollout(
                    env=env,
                    inference_fn=inj_fn,
                    num_repeats=num_clips,
                    max_steps=max_steps,
                    seed=seed,
                    rvq_depth=rvq_depth,
                    num_render=num_render if render_enabled else 0,
                )

                key = f"inject_code_{code_idx}/{pose_name}"
                metrics = compute_condition_metrics(results, null_code)
                all_metrics[key] = metrics
                logging.info(
                    f"    {key}: reward={metrics['mean_reward']:.1f}, "
                    f"displacement={metrics['mean_root_displacement']:.3f}"
                )
                if wandb_enabled:
                    for mk, mv in metrics.items():
                        log_wandb(f"ablation/{key}/{mk}", mv, wandb_enabled)

                if render_enabled:
                    render_condition_videos(
                        results,
                        env,
                        key,
                        output_dir,
                        camera_name,
                        num_codes,
                        cfg,
                        wandb_enabled,
                        num_render,
                    )

    # ================================================================
    # STEP 4: Burst truncation
    # ================================================================
    if "burst_truncation" in experiments:
        logging.info("\n" + "=" * 40)
        logging.info("Running burst truncation experiments...")

        null_params = make_null_d0_params(policy_params)
        null_fn, _ = load_vq_inference_fn_with_stickiness(
            vq_cfg, null_params, deterministic=True
        )

        burst_lengths = list(cfg.ablation.burst_truncation_lengths)
        for max_burst in burst_lengths:
            logging.info(f"  Burst truncation L={max_burst}...")

            for pose_name, env in pose_envs.items():
                logging.info(f"    L={max_burst} on {pose_name}...")
                results = run_burst_truncation_rollout(
                    env=env,
                    normal_fn=inference_fn,
                    null_d0_fn=null_fn,
                    num_repeats=num_clips,
                    max_steps=max_steps,
                    seed=seed,
                    rvq_depth=rvq_depth,
                    max_burst_length=max_burst,
                    null_code=null_code,
                    num_render=num_render if render_enabled else 0,
                )

                key = f"burst_trunc_L{max_burst}/{pose_name}"
                metrics = compute_condition_metrics(results, null_code)
                all_metrics[key] = metrics
                logging.info(
                    f"    {key}: reward={metrics['mean_reward']:.1f}, "
                    f"null_frac={metrics.get('null_d0_fraction', 0):.2f}"
                )
                if wandb_enabled:
                    for mk, mv in metrics.items():
                        log_wandb(f"ablation/{key}/{mk}", mv, wandb_enabled)

                if render_enabled:
                    render_condition_videos(
                        results,
                        env,
                        key,
                        output_dir,
                        camera_name,
                        num_codes,
                        cfg,
                        wandb_enabled,
                        num_render,
                    )

    # ================================================================
    # STEP 5: Comparison plots and summary
    # ================================================================
    logging.info("\n" + "=" * 40)
    logging.info("Generating comparison plots...")

    plot_paths = plot_comparison(all_metrics, output_dir / "comparison")
    if wandb_enabled:
        import wandb

        for plot_key, plot_path in plot_paths.items():
            log_wandb(
                f"ablation/comparison/{plot_key}",
                wandb.Image(plot_path),
                wandb_enabled,
            )

    # Save summary JSON
    summary = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint_path": str(cfg.checkpoint.path),
        "null_code": null_code,
        "top_k_codes": [{"code": c, "count": n} for c, n in top_k_codes],
        "starting_clips": starting_clips,
        "metrics": all_metrics,
    }
    json_path = output_dir / "ablation_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    if wandb_enabled:
        import wandb

        wandb.finish()

    print("\n" + "=" * 60)
    print(f"Ablation experiments complete! Results saved to {output_dir}")
    print(f"Null code: {null_code}")
    print(f"Top-K codes: {[c[0] for c in top_k_codes]}")
    print(f"Starting clips: {starting_clips}")
    print(f"Summary: {json_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
