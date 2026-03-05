"""VQ-VAE Code Ablation Experiments.

Empirically test what each D0 code does via codebook mutation:
1. Code injection — force one D0 code at every timestep, zero D1
2. D0-only — natural D0 selection via encoder, zero D1

Each experiment runs from two starting poses (lowest/highest torso z-height).
Results logged to WandB with HTML slider viewer per experiment.

Usage:
    cd vqvae_jax
    WANDB_MODE=offline python -m ablation.run_ablation checkpoint.path=<path>
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
VQVAE_DIR = SCRIPT_DIR.parent
REPO_ROOT = VQVAE_DIR.parent
sys.path.insert(0, str(VQVAE_DIR))
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
    load_vq_chunked_inference_fn,
    load_vq_inference_fn_with_stickiness,
)
from analysis.utils import build_slider_html, identify_null_code
from analysis.inference_cache import InferenceResult
from analysis.rendering import render_rollout_to_video


# =============================================================================
# D0 CODEBOOK MUTATION
# =============================================================================


def zero_continuous_encoder_params(
    policy_params: tuple[Any, Any],
) -> tuple[Any, Any]:
    """Zero the continuous encoder head so z_e_sampled = 0.

    When a checkpoint is trained with use_continuous_latent=True, the decoder
    expects [z_hat_st, z_e_sampled, proprioception]. For ablation we want
    z_e_sampled = 0 so the only varying signal comes from D0 codes.

    Args:
        policy_params: Tuple of (normalizer_state, policy_params_dict).

    Returns:
        New policy_params with zeroed continuous mean projection.
    """
    normalizer, params = policy_params
    encoder = params["params"]["encoder"]
    if "continuous_mean" not in encoder:
        return policy_params  # No-op if not a continuous latent model

    new_encoder = dict(encoder)
    new_encoder["continuous_mean"] = {
        "kernel": jnp.zeros_like(encoder["continuous_mean"]["kernel"]),
        "bias": jnp.zeros_like(encoder["continuous_mean"]["bias"]),
    }
    new_params = dict(params)
    new_params["params"] = dict(params["params"])
    new_params["params"]["encoder"] = new_encoder
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


def make_zero_d1_params(
    policy_params: tuple[Any, Any],
) -> tuple[Any, Any]:
    """Zero all D1 codebook entries so only D0 contributes.

    Args:
        policy_params: Tuple of (normalizer_state, policy_params_dict).

    Returns:
        New policy_params with zeroed D1 codebook. D0 untouched.
    """
    normalizer, params = policy_params
    quantizer = params["params"]["quantizer"]
    if "codebooks_1" not in quantizer:
        return policy_params  # No D1 codebook
    new_cb1 = dict(quantizer["codebooks_1"])
    new_cb1["embeddings"] = jnp.zeros_like(new_cb1["embeddings"])
    new_quantizer = dict(quantizer)
    new_quantizer["codebooks_1"] = new_cb1
    new_params = dict(params)
    new_params["params"] = dict(params["params"])
    new_params["params"]["quantizer"] = new_quantizer
    return (normalizer, new_params)


def make_injection_d0_zero_d1_params(
    policy_params: tuple[Any, Any],
    target_embedding: jnp.ndarray,
) -> tuple[Any, Any]:
    """Set all D0 entries to target and zero D1.

    Args:
        policy_params: Tuple of (normalizer_state, policy_params_dict).
        target_embedding: Embedding vector for D0.

    Returns:
        New policy_params with injected D0 and zeroed D1.
    """
    params = make_injection_d0_params(policy_params, target_embedding)
    return make_zero_d1_params(params)


# =============================================================================
# STARTING POSE SELECTION
# =============================================================================


def select_starting_clips(clips: ReferenceClips) -> dict[str, int]:
    """Find clips with extreme initial torso z-heights.

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
    override_d0_index: int | None = None,
    reward_terms: list[str] | None = None,
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
            (for video rendering).
        override_d0_index: If set, record this as the D0 code index
            instead of the argmin result.
        reward_terms: If provided, collect per-term rewards from
            ``next_state.metrics["rewards/<term>"]`` for these terms.

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
        comp_lists: dict[str, list[float]] = (
            {t: [] for t in reward_terms} if reward_terms else {}
        )
        store_states = i < num_render
        states: list[Any] | None = [] if store_states else None
        prev_indices = None

        for step in range(max_steps):
            obs = flatten_obs_dict(state.obs)
            rng, action_rng = jax.random.split(rng)
            action, extras = inference_fn(obs, action_rng, prev_indices)

            raw_d0 = int(extras["indices"])
            code_idx = (
                override_d0_index if override_d0_index is not None else raw_d0
            )
            code_indices.append(code_idx)

            all_idx = extras.get("all_indices")
            if all_idx is not None:
                prev_indices = all_idx
                for d in range(rvq_depth):
                    if isinstance(all_idx, tuple) and d < len(all_idx):
                        idx_d = (
                            code_idx
                            if d == 0 and override_d0_index is not None
                            else int(all_idx[d])
                        )
                        rvq_per_depth[d].append(idx_d)
                    elif d == 0:
                        rvq_per_depth[d].append(code_idx)
            else:
                prev_indices = jnp.array(raw_d0)
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

            for term in comp_lists:
                key = f"rewards/{term}"
                val = next_state.metrics.get(key, 0.0)
                comp_lists[term].append(float(val))

            if next_state.done:
                break
            state = next_state

        rvq_indices = None
        if rvq_depth > 1 and rvq_per_depth[0]:
            rvq_indices = tuple(
                np.array(rvq_per_depth[d]) for d in range(rvq_depth)
            )

        reward_components = (
            {t: np.array(v) for t, v in comp_lists.items()}
            if comp_lists
            else None
        )

        results.append(
            InferenceResult(
                clip_idx=i,
                code_indices=np.array(code_indices),
                qpos=np.stack(qpos_list) if qpos_list else np.zeros((0, 0)),
                qvel=np.stack(qvel_list) if qvel_list else np.zeros((0, 0)),
                rewards=np.array(rewards),
                states=states,
                rvq_indices=rvq_indices,
                reward_components=reward_components,
            )
        )

    return results


def run_ablation_rollout_chunked(
    env: Any,
    inference_fn: Any,
    initial_chunk_state_fn: Any,
    num_repeats: int,
    max_steps: int,
    seed: int,
    rvq_depth: int,
    num_render: int = 0,
    override_d0_index: int | None = None,
    reward_terms: list[str] | None = None,
) -> list[InferenceResult]:
    """Run ablation rollouts with code-chunked temporal commitment.

    Args:
        env: Imitation environment (single clip).
        inference_fn: Chunked inference function.
        initial_chunk_state_fn: Callable returning initial chunk_state tuple.
        num_repeats: Number of rollout repeats.
        max_steps: Maximum steps per rollout.
        seed: Base random seed.
        rvq_depth: Number of RVQ depth levels.
        num_render: Number of rollouts to store states for.
        override_d0_index: If set, record this as the D0 code index
            instead of the argmin result (for injection experiments where
            all codebook entries are identical).
        reward_terms: If provided, collect per-term rewards from
            ``next_state.metrics["rewards/<term>"]`` for these terms.

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
        comp_lists: dict[str, list[float]] = (
            {t: [] for t in reward_terms} if reward_terms else {}
        )
        store_states = i < num_render
        states: list[Any] | None = [] if store_states else None
        chunk_state = initial_chunk_state_fn()

        for step in range(max_steps):
            obs = flatten_obs_dict(state.obs)
            rng, action_rng = jax.random.split(rng)
            action, extras, chunk_state = inference_fn(
                obs, chunk_state, action_rng
            )

            raw_d0 = int(extras["indices"])
            code_idx = (
                override_d0_index if override_d0_index is not None else raw_d0
            )
            code_indices.append(code_idx)

            all_idx = extras.get("all_indices")
            if all_idx is not None:
                for d in range(rvq_depth):
                    if isinstance(all_idx, tuple) and d < len(all_idx):
                        idx_d = (
                            code_idx
                            if d == 0 and override_d0_index is not None
                            else int(all_idx[d])
                        )
                        rvq_per_depth[d].append(idx_d)
                    elif d == 0:
                        rvq_per_depth[d].append(code_idx)
            else:
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

            for term in comp_lists:
                key = f"rewards/{term}"
                val = next_state.metrics.get(key, 0.0)
                comp_lists[term].append(float(val))

            if next_state.done:
                break
            state = next_state

        rvq_indices = None
        if rvq_depth > 1 and rvq_per_depth[0]:
            rvq_indices = tuple(
                np.array(rvq_per_depth[d]) for d in range(rvq_depth)
            )

        reward_components = (
            {t: np.array(v) for t, v in comp_lists.items()}
            if comp_lists
            else None
        )

        results.append(
            InferenceResult(
                clip_idx=i,
                code_indices=np.array(code_indices),
                qpos=np.stack(qpos_list) if qpos_list else np.zeros((0, 0)),
                qvel=np.stack(qvel_list) if qvel_list else np.zeros((0, 0)),
                rewards=np.array(rewards),
                states=states,
                rvq_indices=rvq_indices,
                reward_components=reward_components,
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


# Reward terms worth tracking during ablation rollouts.
ABLATION_REWARD_TERMS = [
    "root_pos",
    "joints",
    "end_eff",
    "torso_z_range",
    "root_quat",
]

# Display names for each term (used in plot titles).
_REWARD_TERM_LABELS = {
    "root_pos": "Root Position",
    "joints": "Joint Angles",
    "end_eff": "End Effectors",
    "torso_z_range": "Torso Z Range",
    "root_quat": "Root Orientation",
}


def plot_reward_curves(
    results: list[InferenceResult],
    output_path: Path,
    title: str = "Reward Components",
) -> str:
    """Plot per-component reward curves averaged across rollouts.

    Creates one subplot per reward term showing mean +/- std over time.

    Args:
        results: Rollout results with ``reward_components`` populated.
        output_path: Path to save the PNG figure.
        title: Figure title.

    Returns:
        Path to the saved figure, or empty string on failure.
    """
    # Collect results that have reward_components
    valid = [r for r in results if r.reward_components]
    if not valid:
        return ""

    terms = list(valid[0].reward_components.keys())
    if not terms:
        return ""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_terms = len(terms)
    fig, axes = plt.subplots(
        n_terms, 1, figsize=(10, 2.5 * n_terms), sharex=True, squeeze=False
    )

    # Truncate all rollouts to the shortest length for alignment
    min_len = min(len(r.reward_components[terms[0]]) for r in valid)

    for idx, term in enumerate(terms):
        ax = axes[idx, 0]
        arr = np.array([r.reward_components[term][:min_len] for r in valid])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        t_axis = np.arange(min_len)

        ax.plot(t_axis, mean, linewidth=1.2)
        ax.fill_between(t_axis, mean - std, mean + std, alpha=0.25)
        label = _REWARD_TERM_LABELS.get(term, term)
        ax.set_ylabel(label, fontsize=9)
        ax.tick_params(labelsize=8)

    axes[-1, 0].set_xlabel("Step", fontsize=10)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


# =============================================================================
# WANDB HELPERS
# =============================================================================


def init_wandb(cfg: DictConfig) -> bool:
    """Initialize WandB for ablation experiments."""
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return False
    try:
        import wandb

        run_name = f"ablation_{datetime.now().strftime('%y%m%d_%H%M%S')}"
        wandb.init(
            project=wandb_cfg.get("project", "vqvae-eval"),
            entity=wandb_cfg.get("entity"),
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


# =============================================================================
# VIDEO RENDERING HELPER
# =============================================================================


def render_condition_videos(
    results: list[InferenceResult],
    env: Any,
    condition_key: str,
    pose_name: str,
    output_dir: Path,
    camera: str,
    num_codes: int,
    cfg: DictConfig,
    wandb_enabled: bool,
    num_videos: int = 3,
    d0_label: str | None = None,
) -> list[str]:
    """Render videos for a set of rollout results.

    Args:
        results: Rollout results (must have states stored).
        env: Environment for rendering.
        condition_key: File prefix.
        pose_name: Starting pose.
        output_dir: Directory for video files.
        camera: Camera name.
        num_codes: Number of codes for colormap.
        cfg: Config with render section.
        wandb_enabled: Whether WandB is active.
        num_videos: Max videos to render.
        d0_label: If set, override D0 label in video overlay.

    Returns:
        List of video file paths rendered.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_key = condition_key.replace("/", "_")
    paths: list[str] = []

    for vi in range(min(num_videos, len(results))):
        r = results[vi]
        if not r.states:
            continue

        video_path = output_dir / f"{safe_key}_{pose_name}_{vi}.mp4"
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
            clip_idx=r.clip_idx,
            indices_per_depth=indices_per_depth,
            d0_label=d0_label,
        )
        paths.append(str(video_path))

    return paths


# =============================================================================
# MAIN PIPELINE
# =============================================================================


@hydra.main(version_base=None, config_path="../configs", config_name="code_ablation")
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

    # --- Zero continuous encoder if present ---
    use_continuous_latent = bool(
        vq_cfg.network_config.get("use_continuous_latent", False)
    )
    if use_continuous_latent:
        logging.info("  Zeroing continuous encoder head for ablation")
        policy_params = zero_continuous_encoder_params(policy_params)

    # --- Detect chunked mode ---
    use_code_chunking = bool(
        vq_cfg.network_config.get("use_code_chunking", False)
    )
    commitment_horizon = int(
        vq_cfg.network_config.get("code_commitment_horizon", 10)
    )
    if use_code_chunking:
        logging.info(
            f"  Code chunking ENABLED (H={commitment_horizon}), "
            f"baseline will use chunked inference"
        )

    # --- Create normal inference fn (used for mutation experiments) ---
    inference_fn, _ = load_vq_inference_fn_with_stickiness(
        vq_cfg, policy_params, deterministic=True
    )

    # --- Create chunked inference fn (used for baseline when chunking) ---
    chunked_inference_fn = None
    initial_chunk_state_fn = None
    if use_code_chunking:
        chunked_inference_fn, initial_chunk_state_fn = (
            load_vq_chunked_inference_fn(
                vq_cfg,
                policy_params,
                commitment_horizon=commitment_horizon,
                deterministic=True,
            )
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
    wandb_items: dict[str, Any] = {}

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
        if use_code_chunking:
            results = run_ablation_rollout_chunked(
                env=env,
                inference_fn=chunked_inference_fn,
                initial_chunk_state_fn=initial_chunk_state_fn,
                num_repeats=num_clips,
                max_steps=max_steps,
                seed=seed,
                rvq_depth=rvq_depth,
                num_render=num_render if render_enabled else 0,
            )
        else:
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
    baseline_videos: dict[str, list[str]] = {}
    for pose_name, results in baseline_results.items():
        key = f"baseline/{pose_name}"
        metrics = compute_condition_metrics(results, null_code)
        all_metrics[key] = metrics
        logging.info(
            f"  {key}: reward={metrics['mean_reward']:.1f}, "
            f"length={metrics['mean_episode_length']:.0f}, "
            f"displacement={metrics['mean_root_displacement']:.3f}"
        )
        if render_enabled:
            baseline_videos[pose_name] = render_condition_videos(
                results,
                pose_envs[pose_name],
                "baseline",
                pose_name,
                output_dir,
                camera_name,
                num_codes,
                cfg,
                wandb_enabled,
                num_render,
            )

    # Add baseline videos to wandb_items
    if wandb_enabled and baseline_videos:
        import wandb

        for pose_name, paths in baseline_videos.items():
            for vi, path in enumerate(paths):
                wandb_items[f"baseline/{pose_name}_{vi}"] = wandb.Video(
                    path, format="mp4"
                )

    # ================================================================
    # STEP 2: Code injection (top-K non-null D0 codes, D1 zeroed)
    # ================================================================
    if "code_injection" in experiments:
        logging.info("\n" + "=" * 40)
        logging.info("Running code injection experiments (D1 zeroed)...")

        all_inj_videos: dict[int, dict[str, list[str]]] = {}

        for code_idx, code_count in top_k_codes:
            logging.info(
                f"  Injecting code {code_idx} ({code_count} baseline frames)..."
            )
            target_embedding = jnp.array(d0_codebook[code_idx])
            inj_params = make_injection_d0_zero_d1_params(
                policy_params, target_embedding
            )

            # Use chunked inference when checkpoint was trained with chunking
            if use_code_chunking:
                inj_chunked_fn, inj_chunk_state_fn = (
                    load_vq_chunked_inference_fn(
                        vq_cfg,
                        inj_params,
                        commitment_horizon=commitment_horizon,
                        deterministic=True,
                    )
                )
            else:
                inj_fn, _ = load_vq_inference_fn_with_stickiness(
                    vq_cfg, inj_params, deterministic=True
                )

            inj_videos: dict[str, list[str]] = {}
            for pose_name, env in pose_envs.items():
                logging.info(f"    Code {code_idx} on {pose_name}...")
                if use_code_chunking:
                    results = run_ablation_rollout_chunked(
                        env=env,
                        inference_fn=inj_chunked_fn,
                        initial_chunk_state_fn=inj_chunk_state_fn,
                        num_repeats=num_clips,
                        max_steps=max_steps,
                        seed=seed,
                        rvq_depth=rvq_depth,
                        num_render=num_render if render_enabled else 0,
                        override_d0_index=code_idx,
                    )
                else:
                    results = run_ablation_rollout(
                        env=env,
                        inference_fn=inj_fn,
                        num_repeats=num_clips,
                        max_steps=max_steps,
                        seed=seed,
                        rvq_depth=rvq_depth,
                        num_render=num_render if render_enabled else 0,
                        override_d0_index=code_idx,
                    )

                key = f"inject_code_{code_idx}/{pose_name}"
                metrics = compute_condition_metrics(results, null_code)
                all_metrics[key] = metrics
                logging.info(
                    f"    {key}: reward={metrics['mean_reward']:.1f}, "
                    f"displacement={metrics['mean_root_displacement']:.3f}"
                )
                if render_enabled:
                    inj_videos[pose_name] = render_condition_videos(
                        results,
                        env,
                        f"inject_code_{code_idx}",
                        pose_name,
                        output_dir,
                        camera_name,
                        num_codes,
                        cfg,
                        wandb_enabled,
                        num_render,
                    )
            all_inj_videos[code_idx] = inj_videos

        # Build HTML slider viewer for code injection
        if render_enabled and all_inj_videos:
            for pose_name in pose_envs:
                vid_paths = []
                vid_labels = []
                for code_idx, _ in top_k_codes:
                    vids = all_inj_videos.get(code_idx, {}).get(pose_name, [])
                    if vids:
                        vid_paths.append(vids[0])  # First video per code
                        vid_labels.append(f"Code {code_idx}")

                if vid_paths:
                    html = build_slider_html(
                        vid_paths,
                        vid_labels,
                        f"Code Injection - {pose_name}",
                    )
                    html_path = output_dir / f"code_injection_{pose_name}.html"
                    with open(html_path, "w") as f:
                        f.write(html)

                    if wandb_enabled:
                        import wandb

                        wandb_items[
                            f"ablation/code_injection/{pose_name}_viewer"
                        ] = wandb.Html(html)

    # ================================================================
    # STEP 3: D0-only (natural D0, zeroed D1)
    # ================================================================
    if "d0_only" in experiments:
        logging.info("\n" + "=" * 40)
        logging.info("Running D0-only experiment (natural D0, D1 zeroed)...")

        d0_params = make_zero_d1_params(policy_params)

        # Use chunked inference when checkpoint was trained with chunking
        if use_code_chunking:
            d0_chunked_fn, d0_chunk_state_fn = load_vq_chunked_inference_fn(
                vq_cfg,
                d0_params,
                commitment_horizon=commitment_horizon,
                deterministic=True,
            )
        else:
            d0_fn, _ = load_vq_inference_fn_with_stickiness(
                vq_cfg, d0_params, deterministic=True
            )

        d0_videos: dict[str, list[str]] = {}
        d0_results_by_pose: dict[str, list[InferenceResult]] = {}
        for pose_name, env in pose_envs.items():
            logging.info(f"  D0-only on {pose_name}...")
            if use_code_chunking:
                results = run_ablation_rollout_chunked(
                    env=env,
                    inference_fn=d0_chunked_fn,
                    initial_chunk_state_fn=d0_chunk_state_fn,
                    num_repeats=num_clips,
                    max_steps=max_steps,
                    seed=seed,
                    rvq_depth=rvq_depth,
                    num_render=num_render if render_enabled else 0,
                    reward_terms=ABLATION_REWARD_TERMS,
                )
            else:
                results = run_ablation_rollout(
                    env=env,
                    inference_fn=d0_fn,
                    num_repeats=num_clips,
                    max_steps=max_steps,
                    seed=seed,
                    rvq_depth=rvq_depth,
                    num_render=num_render if render_enabled else 0,
                    reward_terms=ABLATION_REWARD_TERMS,
                )
            d0_results_by_pose[pose_name] = results

            key = f"d0_only/{pose_name}"
            metrics = compute_condition_metrics(results, null_code)
            all_metrics[key] = metrics
            logging.info(
                f"  {key}: reward={metrics['mean_reward']:.1f}, "
                f"length={metrics['mean_episode_length']:.0f}, "
                f"displacement={metrics['mean_root_displacement']:.3f}"
            )
            if render_enabled:
                d0_videos[pose_name] = render_condition_videos(
                    results,
                    env,
                    "d0_only",
                    pose_name,
                    output_dir,
                    camera_name,
                    num_codes,
                    cfg,
                    wandb_enabled,
                    num_render,
                )

        # Build HTML slider viewer for d0_only videos
        if render_enabled and d0_videos:
            all_d0_paths = []
            all_d0_labels = []
            for pose_name, vids in d0_videos.items():
                for vi, path in enumerate(vids):
                    all_d0_paths.append(path)
                    all_d0_labels.append(f"{pose_name} #{vi}")

            if all_d0_paths:
                html = build_slider_html(
                    all_d0_paths,
                    all_d0_labels,
                    "D0-Only (Natural D0, D1 Zeroed)",
                )
                html_path = output_dir / "d0_only_viewer.html"
                with open(html_path, "w") as f:
                    f.write(html)

                if wandb_enabled:
                    import wandb

                    wandb_items["ablation/d0_only/viewer"] = wandb.Html(html)

        # Plot reward component curves for d0_only
        reward_curve_paths: list[str] = []
        reward_curve_labels: list[str] = []
        for pose_name, results in d0_results_by_pose.items():
            fig_path = output_dir / f"d0_only_{pose_name}_reward_curves.png"
            path = plot_reward_curves(
                results,
                fig_path,
                title=f"D0-Only Reward Components - {pose_name}",
            )
            if path:
                reward_curve_paths.append(path)
                reward_curve_labels.append(pose_name)

        if reward_curve_paths:
            html = build_slider_html(
                reward_curve_paths,
                reward_curve_labels,
                "D0-Only Reward Curves",
                media_type="image",
            )
            html_path = output_dir / "d0_only_reward_curves.html"
            with open(html_path, "w") as f:
                f.write(html)

            if wandb_enabled:
                import wandb

                wandb_items["ablation/d0_only/reward_curves"] = wandb.Html(
                    html
                )

    # ================================================================
    # STEP 4: Comparison plots and summary
    # ================================================================
    logging.info("\n" + "=" * 40)
    logging.info("Generating comparison plots...")

    plot_paths = plot_comparison(all_metrics, output_dir / "comparison")
    if wandb_enabled:
        import wandb

        for plot_key, plot_path in plot_paths.items():
            wandb_items[f"ablation/comparison/{plot_key}"] = wandb.Image(plot_path)

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

    # Single WandB log call with all accumulated items, then finish
    if wandb_enabled:
        import wandb

        if wandb_items and wandb.run is not None:
            wandb.log(wandb_items)
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
