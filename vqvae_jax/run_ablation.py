"""VQ-VAE Code Ablation Experiments.

Empirically test what each D0 code does via codebook mutation:
1. Null ablation — force z_q=0 everywhere (disable correction channel)
2. Code injection — force one D0 code at every timestep

Each experiment runs from two starting poses (lowest/highest torso z-height).
Results logged to WandB with one panel per condition (code × pose grouped).

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

from ml_collections import config_dict as mlc_config
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
# WRAPPING ENVIRONMENT FOR LONG ROLLOUTS
# =============================================================================


class WrappingImitation(imitation.Imitation):
    """Imitation env with wrapping reference and no root_too_far termination.

    The standard Imitation env hard-truncates at ~244 steps because the frame
    counter (derived from sim time) exceeds the usable clip length. This
    subclass wraps the frame index modulo the usable length so episodes can
    run indefinitely. ``root_too_far`` termination is also removed so the
    agent can drift from the reference trajectory.

    Other termination criteria (``pose_error``, ``root_too_rotated``,
    ``nan_termination``) remain active for safety.
    """

    def __init__(self, config, clips=None, **kwargs):
        cfg_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        tc = cfg_dict.get("termination_criteria", {})
        tc.pop("root_too_far", None)
        cfg_dict["termination_criteria"] = tc
        new_config = mlc_config.ConfigDict(cfg_dict)
        super().__init__(config=new_config, clips=clips, **kwargs)

    def _get_cur_frame(self, data, info):
        """Wrap frame index so the reference trajectory cycles indefinitely."""
        time_in_frames = data.time * self._config.mocap_hz
        frame = jnp.floor(time_in_frames + info["start_frame"]).astype(int)
        usable_length = self._clip_length() - self._config.reference_length
        return frame % usable_length


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


def select_diverse_starting_clips(
    clips: ReferenceClips,
    num_positions: int,
) -> dict[str, int]:
    """Pick clips at evenly-spaced torso z-height percentiles.

    Args:
        clips: Reference clips with qpos shape [num_clips, T, nq].
        num_positions: Number of starting positions to select.

    Returns:
        Dict mapping ``"pos_0"`` … ``"pos_{N-1}"`` to clip indices,
        ordered from lowest to highest initial torso z.
    """
    initial_z = np.array(clips.qpos[:, 0, 2])
    sorted_indices = np.argsort(initial_z)
    # Pick N clips at evenly-spaced percentiles
    percentiles = np.linspace(0, 1, num_positions)
    picks = np.round(percentiles * (len(sorted_indices) - 1)).astype(int)
    result: dict[str, int] = {}
    for i, p in enumerate(picks):
        clip_idx = int(sorted_indices[p])
        result[f"pos_{i}"] = clip_idx
        logging.info(
            f"  pos_{i}: clip {clip_idx} (z={initial_z[clip_idx]:.4f}, "
            f"percentile={percentiles[i]:.0%})"
        )
    return result


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
        override_d0_index: If set, record this as the D0 code index
            instead of the argmin result. Use when all D0 entries are
            identical (e.g. code injection) so rendering shows the
            correct injected code.

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

            raw_d0 = int(extras["indices"])
            code_idx = override_d0_index if override_d0_index is not None else raw_d0
            code_indices.append(code_idx)

            all_idx = extras.get("all_indices")
            if all_idx is not None:
                prev_indices = all_idx
                for d in range(rvq_depth):
                    if isinstance(all_idx, tuple) and d < len(all_idx):
                        idx_d = code_idx if d == 0 and override_d0_index is not None else int(all_idx[d])
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


def build_inference_fn_map(
    vq_cfg: Any,
    policy_params: tuple[Any, Any],
    d0_codebook: np.ndarray,
    top_k_code_indices: list[int],
) -> dict[int | None, Any]:
    """Precompute inference functions for null D0 and each top-K code.

    Returns a dict mapping ``None`` (null D0) and each code index to a
    ready-to-call inference function. Each function has the same signature
    as :func:`load_vq_inference_fn_with_stickiness` output:
    ``(obs, rng, prev_indices) -> (action, extras)``.

    Args:
        vq_cfg: VQ-VAE config from checkpoint.
        policy_params: Original policy parameters.
        d0_codebook: D0 codebook embeddings [num_codes, dim].
        top_k_code_indices: Code indices to build injection fns for.

    Returns:
        Dict mapping code index (or None for null) to inference function.
    """
    fn_map: dict[int | None, Any] = {}

    # Null D0
    null_params = make_null_d0_params(policy_params)
    null_fn, _ = load_vq_inference_fn_with_stickiness(
        vq_cfg, null_params, deterministic=True
    )
    fn_map[None] = null_fn

    # Per-code injection
    for code_idx in top_k_code_indices:
        target = jnp.array(d0_codebook[code_idx])
        inj_params = make_injection_d0_params(policy_params, target)
        inj_fn, _ = load_vq_inference_fn_with_stickiness(
            vq_cfg, inj_params, deterministic=True
        )
        fn_map[code_idx] = inj_fn

    logging.info(
        f"  Built {len(fn_map)} inference fns "
        f"(null + {len(top_k_code_indices)} codes)"
    )
    return fn_map


def run_code_sequence_rollout(
    env: Any,
    schedule: list[int | None],
    inference_fn_map: dict[int | None, Any],
    max_steps: int,
    seed: int,
    rvq_depth: int,
    store_states: bool = False,
) -> InferenceResult:
    """Run a single rollout with time-varying D0 code forcing.

    At each step *t*, the inference function is selected from
    ``inference_fn_map[schedule[t]]`` — either null (``None``) or a
    specific code index. The schedule is clamped to ``max_steps``.

    Args:
        env: Environment (should be a WrappingImitation for long runs).
        schedule: Per-step code index (or None for null D0).
        inference_fn_map: Precomputed inference functions from
            :func:`build_inference_fn_map`.
        max_steps: Maximum number of steps.
        seed: Random seed.
        rvq_depth: Number of RVQ depth levels.
        store_states: Whether to store env states for video rendering.

    Returns:
        Single InferenceResult for the rollout.
    """
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    state = jit_reset(reset_rng)

    code_indices: list[int] = []
    rvq_per_depth: list[list[int]] = [[] for _ in range(rvq_depth)]
    qpos_list: list[np.ndarray] = []
    qvel_list: list[np.ndarray] = []
    rewards: list[float] = []
    states: list[Any] | None = [] if store_states else None
    prev_indices = None

    for step in range(min(max_steps, len(schedule))):
        scheduled_code = schedule[step]
        inference_fn = inference_fn_map[scheduled_code]

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

    return InferenceResult(
        clip_idx=0,
        code_indices=np.array(code_indices),
        qpos=np.stack(qpos_list) if qpos_list else np.zeros((0, 0)),
        qvel=np.stack(qvel_list) if qvel_list else np.zeros((0, 0)),
        rewards=np.array(rewards),
        states=states,
        rvq_indices=rvq_indices,
    )


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
# TRANSITION MATRIX & SCHEDULE GENERATION
# =============================================================================


def build_top_k_transition_matrix(
    results: list[InferenceResult],
    top_k_indices: list[int],
    num_codes: int,
) -> np.ndarray:
    """Build a K x K row-normalized transition probability matrix.

    Uses the full-size transition counts from baseline rollouts, then
    extracts the submatrix for the top-K codes and row-normalizes.

    Args:
        results: Baseline rollout results.
        top_k_indices: Top-K code indices (sorted by popularity).
        num_codes: Total number of D0 codes.

    Returns:
        Row-normalized transition probabilities [K, K]. Rows that sum
        to zero are set to uniform 1/K.
    """
    from analysis.transition_context_analysis import compute_global_transition_matrix

    full_counts, _ = compute_global_transition_matrix(results, num_codes)
    plt.close("all")  # close the figure created by compute_global_transition_matrix

    k = len(top_k_indices)
    sub = np.zeros((k, k), dtype=np.float64)
    for i, fi in enumerate(top_k_indices):
        for j, fj in enumerate(top_k_indices):
            sub[i, j] = full_counts[fi, fj]

    # Row-normalize
    row_sums = sub.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # avoid division by zero
    probs = sub / row_sums
    # Rows with no transitions → uniform
    zero_rows = probs.sum(axis=1) == 0
    if zero_rows.any():
        probs[zero_rows] = 1.0 / k
    return probs


def plot_transition_matrix(
    probs: np.ndarray,
    indices: list[int],
    output_dir: Path,
) -> str:
    """Create a heatmap of the K x K transition probability matrix.

    Args:
        probs: Row-normalized transition probabilities [K, K].
        indices: Code indices corresponding to rows/columns.
        output_dir: Directory to save the plot.

    Returns:
        File path of the saved plot.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    k = len(indices)
    fig, ax = plt.subplots(figsize=(max(6, k * 1.2), max(5, k)))
    im = ax.imshow(probs, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels([str(i) for i in indices])
    ax.set_yticklabels([str(i) for i in indices])
    ax.set_xlabel("To Code")
    ax.set_ylabel("From Code")
    ax.set_title("Top-K Transition Probabilities")

    # Annotate cells
    for i in range(k):
        for j in range(k):
            ax.text(
                j,
                i,
                f"{probs[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if probs[i, j] > 0.5 else "black",
            )

    plt.colorbar(im, ax=ax, label="P(to | from)", shrink=0.8)
    plt.tight_layout()
    fig_path = output_dir / "transition_matrix_topk.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(fig_path)


def generate_uniform_schedule(
    top_k_indices: list[int],
    total_steps: int,
    window_size: int,
    null_prefix_steps: int = 0,
) -> list[int | None]:
    """Generate a schedule that cycles top-K codes in popularity order.

    Args:
        top_k_indices: Code indices sorted by descending popularity.
        total_steps: Total number of steps in the schedule.
        window_size: Number of steps per code window.
        null_prefix_steps: Steps of null D0 at the start.

    Returns:
        List of length ``total_steps`` with code index or None per step.
    """
    schedule: list[int | None] = [None] * null_prefix_steps
    k = len(top_k_indices)
    while len(schedule) < total_steps:
        for code_idx in top_k_indices:
            schedule.extend([code_idx] * window_size)
            if len(schedule) >= total_steps:
                break
    return schedule[:total_steps]


def generate_transition_schedule(
    top_k_indices: list[int],
    probs: np.ndarray,
    total_steps: int,
    window_size: int,
    null_prefix_steps: int = 0,
    seed: int = 0,
) -> list[int | None]:
    """Generate a schedule by sampling from the transition matrix.

    Starts with the most popular code (index 0 in top_k_indices) after
    the optional null prefix, then samples the next code from the
    transition matrix row of the current code.

    Args:
        top_k_indices: Code indices sorted by descending popularity.
        probs: Row-normalized transition probabilities [K, K].
        total_steps: Total number of steps in the schedule.
        window_size: Number of steps per code window.
        null_prefix_steps: Steps of null D0 at the start.
        seed: Random seed for sampling.

    Returns:
        List of length ``total_steps`` with code index or None per step.
    """
    rng = np.random.default_rng(seed)
    schedule: list[int | None] = [None] * null_prefix_steps
    k = len(top_k_indices)
    current_k_idx = 0  # start with most popular
    while len(schedule) < total_steps:
        code_idx = top_k_indices[current_k_idx]
        schedule.extend([code_idx] * window_size)
        # Sample next code from transition row
        current_k_idx = int(rng.choice(k, p=probs[current_k_idx]))
    return schedule[:total_steps]


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
    pose_name: str,
    output_dir: Path,
    camera: str,
    num_codes: int,
    cfg: DictConfig,
    wandb_enabled: bool,
    num_videos: int = 3,
) -> list[str]:
    """Render videos for a set of rollout results.

    Videos are saved locally. WandB logging is NOT done here — the caller
    should batch all poses for a condition into a single ``wandb.log()``
    call so they appear in the same panel.

    Args:
        results: Rollout results (must have states stored).
        env: Environment for rendering.
        condition_key: File prefix, e.g. "null_ablation".
        pose_name: Starting pose, e.g. "low_height".
        output_dir: Directory for video files.
        camera: Camera name.
        num_codes: Number of codes for colormap.
        cfg: Config with render section.
        wandb_enabled: Whether WandB is active.
        num_videos: Max videos to render.

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
        )
        paths.append(str(video_path))

    return paths


def log_condition_panel(
    condition_key: str,
    video_paths_by_pose: dict[str, list[str]],
    wandb_enabled: bool,
) -> None:
    """Log all videos for one condition as a single WandB panel.

    Groups both poses (low/high) under one key prefix so they appear
    in the same WandB panel.

    Args:
        condition_key: Panel name, e.g. "null_ablation" or "inject_code_3".
        video_paths_by_pose: Mapping from pose name to list of video paths.
        wandb_enabled: Whether WandB is active.
    """
    if not wandb_enabled:
        return
    try:
        import wandb

        if wandb.run is None:
            return

        log_dict = {}
        for pose_name, paths in video_paths_by_pose.items():
            for vi, path in enumerate(paths):
                log_dict[f"{condition_key}/{pose_name}_{vi}"] = wandb.Video(
                    path, format="mp4"
                )

        if log_dict:
            wandb.log(log_dict)
    except Exception as e:
        logging.warning(f"Failed to log {condition_key} panel: {e}")


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
    log_condition_panel("baseline", baseline_videos, wandb_enabled)

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

        null_videos: dict[str, list[str]] = {}
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
            if render_enabled:
                null_videos[pose_name] = render_condition_videos(
                    results,
                    env,
                    "null_ablation",
                    pose_name,
                    output_dir,
                    camera_name,
                    num_codes,
                    cfg,
                    wandb_enabled,
                    num_render,
                )
        log_condition_panel("null_ablation", null_videos, wandb_enabled)

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

            inj_videos: dict[str, list[str]] = {}
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
            log_condition_panel(f"inject_code_{code_idx}", inj_videos, wandb_enabled)

    # ================================================================
    # STEP 4: Null long runs (extended rollouts with null D0)
    # ================================================================
    if "null_long_run" in experiments:
        logging.info("\n" + "=" * 40)
        logging.info("Running null long-run experiments...")

        nlr_cfg = cfg.ablation.null_long_run
        nlr_max_steps = nlr_cfg.max_steps
        nlr_num_positions = nlr_cfg.num_starting_positions
        nlr_num_render = nlr_cfg.get("num_render", 1)

        # Select diverse starting positions
        diverse_clips = select_diverse_starting_clips(clips, nlr_num_positions)

        # Build null D0 inference fn
        null_params = make_null_d0_params(policy_params)
        null_fn, _ = load_vq_inference_fn_with_stickiness(
            vq_cfg, null_params, deterministic=True
        )

        for pos_name, clip_idx in diverse_clips.items():
            logging.info(
                f"  Null long run on {pos_name} (clip {clip_idx}), "
                f"{nlr_max_steps} steps..."
            )
            single = subset_clips(clips, clip_idx)
            wrap_env = WrappingImitation(config=env_cfg_ml, clips=single)

            results = run_ablation_rollout(
                env=wrap_env,
                inference_fn=null_fn,
                num_repeats=num_clips,
                max_steps=nlr_max_steps,
                seed=seed,
                rvq_depth=rvq_depth,
                num_render=nlr_num_render if render_enabled else 0,
            )

            key = f"null_long_run/{pos_name}_clip{clip_idx}"
            metrics = compute_condition_metrics(results, null_code)
            all_metrics[key] = metrics
            logging.info(
                f"  {key}: reward={metrics['mean_reward']:.1f}, "
                f"length={metrics['mean_episode_length']:.0f}, "
                f"displacement={metrics['mean_root_displacement']:.3f}"
            )
            if render_enabled:
                vid_paths = render_condition_videos(
                    results,
                    wrap_env,
                    f"null_long_run_{pos_name}_clip{clip_idx}",
                    pos_name,
                    output_dir,
                    camera_name,
                    num_codes,
                    cfg,
                    wandb_enabled,
                    nlr_num_render,
                )
                log_condition_panel(
                    f"null_long_run/{pos_name}_clip{clip_idx}",
                    {pos_name: vid_paths},
                    wandb_enabled,
                )

    # ================================================================
    # STEP 5: Code sequence injection (time-varying D0 forcing)
    # ================================================================
    if "code_sequence_injection" in experiments:
        logging.info("\n" + "=" * 40)
        logging.info("Running code sequence injection experiments...")

        csi_cfg = cfg.ablation.code_sequence_injection
        csi_max_steps = csi_cfg.max_steps
        csi_num_positions = csi_cfg.num_starting_positions
        csi_window_size = csi_cfg.window_size
        csi_null_prefix = csi_cfg.get("null_prefix_steps", 0)
        csi_modes = list(csi_cfg.modes)
        csi_num_sequences = csi_cfg.num_sequences
        csi_num_render = csi_cfg.get("num_render", 1)

        # Select diverse starting positions
        diverse_clips = select_diverse_starting_clips(clips, csi_num_positions)

        # Build inference fn map (null + top-K codes)
        top_k_idx_list = [int(idx) for idx in top_k_indices]
        fn_map = build_inference_fn_map(
            vq_cfg, policy_params, d0_codebook, top_k_idx_list
        )

        # Build transition matrix from pooled baseline if needed
        trans_probs = None
        if "transition_matrix" in csi_modes:
            logging.info("  Building transition matrix from baseline data...")
            trans_probs = build_top_k_transition_matrix(
                all_baseline, top_k_idx_list, num_codes
            )
            tm_path = plot_transition_matrix(
                trans_probs, top_k_idx_list, output_dir / "transition"
            )
            logging.info(f"  Transition matrix plot: {tm_path}")
            if wandb_enabled:
                import wandb

                log_wandb(
                    "ablation/code_sequence/transition_matrix",
                    wandb.Image(tm_path),
                    wandb_enabled,
                )

        for mode in csi_modes:
            logging.info(f"\n  Mode: {mode}")

            for pos_name, clip_idx in diverse_clips.items():
                single = subset_clips(clips, clip_idx)
                wrap_env = WrappingImitation(config=env_cfg_ml, clips=single)

                for seq_i in range(csi_num_sequences):
                    # Generate schedule
                    if mode == "uniform":
                        schedule = generate_uniform_schedule(
                            top_k_idx_list,
                            csi_max_steps,
                            csi_window_size,
                            csi_null_prefix,
                        )
                    elif mode == "transition_matrix":
                        schedule = generate_transition_schedule(
                            top_k_idx_list,
                            trans_probs,
                            csi_max_steps,
                            csi_window_size,
                            csi_null_prefix,
                            seed=seed + seq_i,
                        )
                    else:
                        logging.warning(f"  Unknown mode: {mode}, skipping")
                        continue

                    store = render_enabled and seq_i < csi_num_render
                    logging.info(
                        f"    {mode}/{pos_name}_clip{clip_idx}/seq_{seq_i} "
                        f"({csi_max_steps} steps, window={csi_window_size})"
                    )

                    result = run_code_sequence_rollout(
                        env=wrap_env,
                        schedule=schedule,
                        inference_fn_map=fn_map,
                        max_steps=csi_max_steps,
                        seed=seed + seq_i,
                        rvq_depth=rvq_depth,
                        store_states=store,
                    )

                    key = f"{mode}_injection/" f"{pos_name}_clip{clip_idx}/seq_{seq_i}"
                    metrics = compute_condition_metrics([result], null_code)
                    all_metrics[key] = metrics
                    logging.info(
                        f"    {key}: reward={metrics['mean_reward']:.1f}, "
                        f"length={metrics['mean_episode_length']:.0f}, "
                        f"displacement="
                        f"{metrics['mean_root_displacement']:.3f}"
                    )
                    if store:
                        vid_paths = render_condition_videos(
                            [result],
                            wrap_env,
                            f"{mode}_inj_{pos_name}_clip{clip_idx}_seq{seq_i}",
                            pos_name,
                            output_dir,
                            camera_name,
                            num_codes,
                            cfg,
                            wandb_enabled,
                            1,
                        )
                        log_condition_panel(
                            f"{mode}_injection/{pos_name}_clip{clip_idx}",
                            {f"seq_{seq_i}": vid_paths},
                            wandb_enabled,
                        )

    # ================================================================
    # STEP 6: Comparison plots and summary
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
