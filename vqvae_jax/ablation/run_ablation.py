"""VQ-VAE Code Ablation Experiments.

Empirically test what each D0 code does:
1. Code injection — decoder-only: hold one D0 code constant, no encoder
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

import base64
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

from brax.training import distribution
from brax.training.acme import running_statistics

from analysis.checkpoint_utils import (
    create_standalone_decoder,
    get_all_codebooks,
    get_codebook,
    get_decoder_params,
    load_vq_checkpoint,
    load_vq_chunked_inference_fn,
    load_vq_inference_fn_with_stickiness,
)
from analysis.utils import build_slider_html, identify_null_code
from analysis.inference_cache import InferenceResult
from analysis.rendering import render_rollout_to_video

# =============================================================================
# D1 CODEBOOK MUTATION (for D0-only experiment)
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


# =============================================================================
# DECODER-ONLY STEP FUNCTION (for code injection)
# =============================================================================


def make_decoder_only_step_fn(
    cfg: DictConfig,
    policy_params: tuple[Any, Any],
) -> tuple[Any, int]:
    """Build a decoder-only function: (d0_code_index, obs) -> action.

    The encoder is NOT used. D1 residual is zero by construction.
    Continuous latent (if present) is zeroed.

    Args:
        cfg: Checkpoint config with network_config section.
        policy_params: Tuple of (normalizer_state, policy_params).

    Returns:
        Tuple of (decode_step_fn, action_size).
    """
    normalizer_state, _ = policy_params
    codebook_0 = get_codebook(policy_params, depth=0)
    decoder = create_standalone_decoder(cfg)
    decoder_params = get_decoder_params(policy_params)

    use_continuous = bool(cfg.network_config.get("use_continuous_latent", False))
    continuous_dim = int(cfg.network_config.get("continuous_latent_dim", 4))
    action_size = cfg.network_config.action_size
    action_dist = distribution.NormalTanhDistribution(event_size=action_size)

    latent_dim = codebook_0.shape[1]
    logging.info(
        f"  Decoder-only mode: latent_dim={latent_dim}, "
        f"use_continuous={use_continuous}, "
        f"continuous_dim={continuous_dim}"
    )

    def decode_step(d0_code_index: int, obs: dict) -> jnp.ndarray:
        """Decode a single D0 code to an action."""
        z_q = codebook_0[d0_code_index]

        flat_obs = flatten_obs_dict(obs)
        proprio_norm = running_statistics.normalize(
            flat_obs["proprioception"],
            normalizer_state.proprioception,
        )

        if use_continuous:
            z_e_zeros = jnp.zeros(continuous_dim)
            x = jnp.concatenate([z_q, z_e_zeros, proprio_norm], axis=-1)
        else:
            x = jnp.concatenate([z_q, proprio_norm], axis=-1)

        action_params, _ = decoder.apply({"params": decoder_params}, x)
        return jnp.array(action_dist.mode(action_params))

    return decode_step, action_size


def make_decoder_d0d1_step_fn(
    cfg: DictConfig,
    policy_params: tuple[Any, Any],
) -> tuple[Any, int]:
    """Build a decoder function using both D0 and D1 codes: (d0, d1, obs) -> action.

    The encoder is NOT used. The decoder input is the residual sum of D0 and D1
    codebook embeddings, matching what the decoder sees during training.
    Continuous latent (if present) is zeroed.

    Args:
        cfg: Checkpoint config with network_config section.
        policy_params: Tuple of (normalizer_state, policy_params).

    Returns:
        Tuple of (decode_step_fn, action_size).
    """
    normalizer_state, _ = policy_params
    codebook_0 = get_codebook(policy_params, depth=0)
    codebook_1 = get_codebook(policy_params, depth=1)
    decoder = create_standalone_decoder(cfg)
    decoder_params = get_decoder_params(policy_params)

    use_continuous = bool(cfg.network_config.get("use_continuous_latent", False))
    continuous_dim = int(cfg.network_config.get("continuous_latent_dim", 4))
    action_size = cfg.network_config.action_size
    action_dist = distribution.NormalTanhDistribution(event_size=action_size)

    latent_dim = codebook_0.shape[1]
    logging.info(
        f"  Decoder D0+D1 mode: latent_dim={latent_dim}, "
        f"use_continuous={use_continuous}, "
        f"continuous_dim={continuous_dim}"
    )

    def decode_step(d0_code_index: int, d1_code_index: int, obs: dict) -> jnp.ndarray:
        """Decode D0+D1 codes to an action (residual sum)."""
        z_q = codebook_0[d0_code_index] + codebook_1[d1_code_index]

        flat_obs = flatten_obs_dict(obs)
        proprio_norm = running_statistics.normalize(
            flat_obs["proprioception"],
            normalizer_state.proprioception,
        )

        if use_continuous:
            z_e_zeros = jnp.zeros(continuous_dim)
            x = jnp.concatenate([z_q, z_e_zeros, proprio_norm], axis=-1)
        else:
            x = jnp.concatenate([z_q, proprio_norm], axis=-1)

        action_params, _ = decoder.apply({"params": decoder_params}, x)
        return jnp.array(action_dist.mode(action_params))

    return decode_step, action_size


def run_decoder_only_rollout(
    env: Any,
    jit_decode: Any,
    code_idx: int,
    num_repeats: int,
    max_steps: int,
    seed: int,
    num_render: int = 0,
) -> list[InferenceResult]:
    """Run decoder-only rollouts holding a single D0 code constant.

    No encoder is involved — the code embedding is looked up directly
    and fed to the decoder with proprioception.

    Args:
        env: Imitation environment (single clip).
        jit_decode: JIT-compiled decoder step fn: (code_idx, obs) -> action.
        code_idx: D0 code index to hold constant.
        num_repeats: Number of rollout repeats.
        max_steps: Maximum steps per rollout.
        seed: Base random seed.
        num_render: Number of rollouts to store env states for rendering.

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

        qpos_list: list[np.ndarray] = []
        qvel_list: list[np.ndarray] = []
        rewards: list[float] = []
        store_states = i < num_render
        states: list[Any] | None = [] if store_states else None

        for step in range(max_steps):
            action = jit_decode(code_idx, state.obs)

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

        results.append(
            InferenceResult(
                clip_idx=i,
                code_indices=np.full(len(rewards), code_idx, dtype=int),
                qpos=(np.stack(qpos_list) if qpos_list else np.zeros((0, 0))),
                qvel=(np.stack(qvel_list) if qvel_list else np.zeros((0, 0))),
                rewards=np.array(rewards),
                states=states,
            )
        )

    return results


# =============================================================================
# MOVEMENT CLASSIFICATION & HTML HELPERS
# =============================================================================


def classify_codes_by_movement(
    per_code_qpos: dict[int, np.ndarray],
) -> dict[str, list[int]]:
    """Classify codes into movement categories using data-driven thresholds.

    For each code's H-step rollout qpos, computes:
    - XY path length: cumulative euclidean distance in the XY plane
    - Z range: max - min of root z-height

    Splits at median of each metric across all codes.

    Args:
        per_code_qpos: Mapping from code index to qpos array [T, nq].

    Returns:
        Dict with keys "high_xy", "high_xyz", "high_z", "stationary",
        each containing a list of code indices.
    """
    xy_paths = {}
    z_ranges = {}
    for code_idx, qpos in per_code_qpos.items():
        if len(qpos) < 2:
            xy_paths[code_idx] = 0.0
            z_ranges[code_idx] = 0.0
            continue
        diffs = np.diff(qpos[:, :2], axis=0)
        xy_paths[code_idx] = float(np.sum(np.linalg.norm(diffs, axis=1)))
        z_ranges[code_idx] = float(np.max(qpos[:, 2]) - np.min(qpos[:, 2]))

    all_xy = np.array(list(xy_paths.values()))
    all_z = np.array(list(z_ranges.values()))
    xy_median = float(np.median(all_xy))
    z_median = float(np.median(all_z))

    categories: dict[str, list[int]] = {
        "high_xy": [],
        "high_xyz": [],
        "high_z": [],
        "stationary": [],
    }
    for code_idx in per_code_qpos:
        xy = xy_paths[code_idx]
        z = z_ranges[code_idx]
        if xy >= xy_median and z < z_median:
            categories["high_xy"].append(code_idx)
        elif xy >= xy_median and z >= z_median:
            categories["high_xyz"].append(code_idx)
        elif xy < xy_median and z >= z_median:
            categories["high_z"].append(code_idx)
        else:
            categories["stationary"].append(code_idx)

    return categories


def plot_code_histogram(
    code_counts: np.ndarray,
    highlighted_codes: list[int],
    title: str,
    num_codes: int,
    output_path: Path,
) -> str:
    """Bar chart of code usage with highlighted codes in color.

    Args:
        code_counts: Array of shape [num_codes] with frame counts.
        highlighted_codes: Codes to highlight in color (rest greyed out).
        title: Plot title.
        num_codes: Total number of codes.
        output_path: Path to save PNG.

    Returns:
        Path to saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(6, num_codes * 0.25), 2.5))

    colors = ["#cccccc"] * num_codes
    highlight_cmap = plt.cm.tab10
    highlighted_set = set(highlighted_codes)
    rank = 0
    for c in range(num_codes):
        if c in highlighted_set:
            colors[c] = highlight_cmap(rank % 10)
            rank += 1

    ax.bar(range(num_codes), code_counts[:num_codes], color=colors, edgecolor="none")
    ax.set_xlabel("Code Index", fontsize=9)
    ax.set_ylabel("Frame Count", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _encode_file_b64(path: str, mime: str) -> str:
    """Read a file and return a base64 data URI."""
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"


# Category display metadata.
_CATEGORY_LABELS = {
    "high_xy": "Walking / Locomotion",
    "high_xyz": "Combined Movement",
    "high_z": "Rearing / Vertical",
    "stationary": "Stationary / Low Movement",
}

_CATEGORY_COLORS = {
    "high_xy": "#4CAF50",
    "high_xyz": "#FF9800",
    "high_z": "#2196F3",
    "stationary": "#9E9E9E",
}


def build_code_injection_html(
    categories: dict[str, list[int]],
    per_code_videos: dict[int, str],
    per_code_labels: dict[int, str],
    histogram_paths: dict[str, str],
    title: str,
) -> str:
    """Build static HTML with tabbed categories of code injection videos.

    Each tab shows a video grid (small thumbnails) and a histogram below.
    All media is base64-encoded.  Tab buttons toggle visibility via JS.

    Args:
        categories: Mapping from category name to list of code indices.
        per_code_videos: Mapping from code index to video file path.
        per_code_labels: Mapping from code index to label string.
        histogram_paths: Mapping from category name to histogram PNG path.
        title: Page title.

    Returns:
        HTML string.
    """
    # Pre-encode all videos and histograms
    video_data: dict[int, str] = {}
    for code_idx, path in per_code_videos.items():
        video_data[code_idx] = _encode_file_b64(path, "video/mp4")
    hist_data: dict[str, str] = {}
    for cat, path in histogram_paths.items():
        hist_data[cat] = _encode_file_b64(path, "image/png")

    tab_buttons = []
    tab_contents = []
    for i, (cat, codes) in enumerate(categories.items()):
        if not codes:
            continue
        label = _CATEGORY_LABELS.get(cat, cat)
        color = _CATEGORY_COLORS.get(cat, "#666")
        active = " active" if i == 0 else ""
        display = "flex" if i == 0 else "none"

        tab_buttons.append(
            f'<button class="tab-btn{active}" '
            f"onclick=\"showTab('{cat}')\" "
            f'style="border-bottom: 3px solid {color}">'
            f"{label} ({len(codes)})</button>"
        )

        # Video grid
        grid_items = []
        for code_idx in sorted(codes):
            vid_src = video_data.get(code_idx, "")
            lbl = per_code_labels.get(code_idx, f"Code {code_idx}")
            grid_items.append(
                f'<div class="vid-cell">'
                f'<video src="{vid_src}" width="200" autoplay loop muted></video>'
                f'<div class="vid-label">{lbl}</div>'
                f"</div>"
            )

        hist_img = ""
        if cat in hist_data:
            hist_img = (
                f'<img src="{hist_data[cat]}" '
                f'style="max-width:100%; margin-top:12px;" />'
            )

        tab_contents.append(
            f'<div class="tab-content" id="tab-{cat}" '
            f'style="display:{display}; flex-wrap:wrap; gap:8px;">'
            f'{"".join(grid_items)}'
            f'<div style="width:100%">{hist_img}</div>'
            f"</div>"
        )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: sans-serif; margin: 16px; background: #fafafa; }}
h2 {{ margin-bottom: 8px; }}
.tab-bar {{ display: flex; gap: 4px; margin-bottom: 12px; }}
.tab-btn {{ padding: 8px 16px; cursor: pointer; background: #eee;
            border: none; border-radius: 4px 4px 0 0; font-size: 13px; }}
.tab-btn.active {{ background: #fff; font-weight: bold; }}
.vid-cell {{ text-align: center; }}
.vid-label {{ font-size: 11px; margin-top: 2px; }}
</style>
<script>
function showTab(cat) {{
  document.querySelectorAll('.tab-content').forEach(
    el => el.style.display = 'none');
  document.querySelectorAll('.tab-btn').forEach(
    el => el.classList.remove('active'));
  document.getElementById('tab-' + cat).style.display = 'flex';
  event.target.classList.add('active');
}}
</script>
</head><body>
<h2>{title}</h2>
<div class="tab-bar">{"".join(tab_buttons)}</div>
{"".join(tab_contents)}
</body></html>"""
    return html


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
            code_idx = override_d0_index if override_d0_index is not None else raw_d0
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
            rvq_indices = tuple(np.array(rvq_per_depth[d]) for d in range(rvq_depth))

        reward_components = (
            {t: np.array(v) for t, v in comp_lists.items()} if comp_lists else None
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
            action, extras, chunk_state = inference_fn(obs, chunk_state, action_rng)

            raw_d0 = int(extras["indices"])
            code_idx = override_d0_index if override_d0_index is not None else raw_d0
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
            rvq_indices = tuple(np.array(rvq_per_depth[d]) for d in range(rvq_depth))

        reward_components = (
            {t: np.array(v) for t, v in comp_lists.items()} if comp_lists else None
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
    use_code_chunking = bool(vq_cfg.network_config.get("use_code_chunking", False))
    commitment_horizon = int(vq_cfg.network_config.get("code_commitment_horizon", 10))
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
        chunked_inference_fn, initial_chunk_state_fn = load_vq_chunked_inference_fn(
            vq_cfg,
            policy_params,
            commitment_horizon=commitment_horizon,
            deterministic=True,
        )

    # --- Load reference clips and select starting poses ---
    logging.info("\nLoading reference clips and selecting starting poses...")
    _, cfg_dict, env_cfg_ml = config_utils.prepare_config(cfg)

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
    # STEP 2: Code injection — ALL codes, decoder-only, per-pose
    # ================================================================
    if "code_injection" in experiments:
        logging.info("\n" + "=" * 40)
        logging.info("Running code injection for ALL codes (decoder-only)...")

        # Build decoder-only step function — encoder never runs
        decode_step, _ = make_decoder_only_step_fn(vq_cfg, policy_params)
        jit_decode = jax.jit(decode_step)

        # Steps per code injection rollout — match HMM gallery (H * 20)
        H = commitment_horizon * 20 if use_code_chunking else max_steps

        for pose_name, env in pose_envs.items():
            logging.info(f"\n  === Code injection on {pose_name} ===")

            per_code_qpos: dict[int, np.ndarray] = {}
            per_code_videos: dict[int, str] = {}
            per_code_labels: dict[int, str] = {}

            for code_idx in range(num_codes):
                logging.info(f"    Injecting code {code_idx}/{num_codes}...")
                results = run_decoder_only_rollout(
                    env=env,
                    jit_decode=jit_decode,
                    code_idx=code_idx,
                    num_repeats=1,
                    max_steps=H,
                    seed=seed,
                    num_render=1 if render_enabled else 0,
                )

                r = results[0]
                per_code_qpos[code_idx] = r.qpos

                # Compute metrics
                key = f"inject_code_{code_idx}/{pose_name}"
                metrics = compute_condition_metrics(results, null_code)
                all_metrics[key] = metrics

                # Render single video for this code
                if render_enabled and r.states:
                    vid_path = output_dir / f"inject_{pose_name}_c{code_idx}.mp4"
                    render_rollout_to_video(
                        env=env,
                        rollout_states=r.states,
                        output_path=vid_path,
                        camera=camera_name,
                        width=cfg.render.width,
                        height=cfg.render.height,
                        fps=cfg.render.fps,
                        indices=r.code_indices,
                        num_codes=num_codes,
                        d0_label=f"D0:{code_idx}",
                    )
                    per_code_videos[code_idx] = str(vid_path)
                    per_code_labels[code_idx] = f"Code {code_idx}"

            # Classify codes by root movement
            categories = classify_codes_by_movement(per_code_qpos)
            for cat, codes in categories.items():
                logging.info(f"    {cat}: {len(codes)} codes — {codes}")

            # Build per-category histograms (baseline code counts)
            histogram_paths: dict[str, str] = {}
            for cat, codes in categories.items():
                if not codes:
                    continue
                hist_path = output_dir / f"hist_{pose_name}_{cat}.png"
                plot_code_histogram(
                    code_counts=code_counts,
                    highlighted_codes=codes,
                    title=f"{_CATEGORY_LABELS.get(cat, cat)} — {pose_name}",
                    num_codes=num_codes,
                    output_path=hist_path,
                )
                histogram_paths[cat] = str(hist_path)

            # Build tabbed HTML
            if render_enabled and per_code_videos:
                html = build_code_injection_html(
                    categories=categories,
                    per_code_videos=per_code_videos,
                    per_code_labels=per_code_labels,
                    histogram_paths=histogram_paths,
                    title=f"Code Injection — {pose_name}",
                )
                html_path = output_dir / f"code_injection_{pose_name}.html"
                with open(html_path, "w") as f:
                    f.write(html)
                logging.info(f"    Saved HTML: {html_path}")

                if wandb_enabled:
                    import wandb

                    wandb_items[f"ablation/code_injection/{pose_name}/viewer"] = (
                        wandb.Html(html)
                    )

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

                wandb_items["ablation/d0_only/reward_curves"] = wandb.Html(html)

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
