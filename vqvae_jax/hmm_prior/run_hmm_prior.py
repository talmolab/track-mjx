"""HMM Prior: Fit a discrete HMM on D0 code sequences and generate free-loop behavior.

Validates whether the learned VQ-VAE code space has meaningful temporal structure
that can be captured by a simple probabilistic model. The encoder is NOT used at
all — only the decoder converts HMM-sampled codes into actions.

Pipeline:
1. Load D0 code sequences from H5 rollout data
2. Subsample at commitment horizon H
3. Sweep HMM num_states via EM, select best K by held-out log-likelihood
4. Generate free-loop rollouts using HMM-sampled codes + decoder only
5. Render videos with code overlays, log to WandB

Usage:
    cd vqvae_jax
    WANDB_MODE=offline python -m hmm_prior.run_hmm_prior
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
from scipy.stats import wasserstein_distance
from brax.training import distribution
from brax.training.acme import running_statistics
from omegaconf import DictConfig
from vnl_playground.tasks.rodent import imitation
from vnl_playground.tasks.rodent.imitation import ReferenceClips

from track_mjx.agent.observation_utils import flatten_obs_dict
from track_mjx.config import utils as config_utils

from analysis.checkpoint_utils import (
    create_standalone_decoder,
    get_codebook,
    get_decoder_params,
    load_vq_checkpoint,
)
from analysis.code_analysis import load_rollouts_from_h5
from analysis.rendering import (
    get_nature_colormap,
    add_multi_line_overlay,
    render_rollout_to_video,
)
from analysis.utils import build_slider_html

# =============================================================================
# LIGHTWEIGHT DISCRETE HMM (JAX)
# =============================================================================


def _normalize_rows(log_mat: np.ndarray) -> np.ndarray:
    """Normalize rows of a log-probability matrix to sum to 1 in probability space."""
    mat = np.exp(log_mat - log_mat.max(axis=-1, keepdims=True))
    return mat / mat.sum(axis=-1, keepdims=True)


def hmm_forward(
    log_pi: np.ndarray,
    log_A: np.ndarray,
    log_B: np.ndarray,
    observations: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Forward algorithm for discrete HMM.

    Args:
        log_pi: Log initial state probabilities, shape [K].
        log_A: Log transition matrix, shape [K, K].
        log_B: Log emission matrix, shape [K, C].
        observations: Integer observation sequence, shape [T].

    Returns:
        Tuple of (log_alpha, log_likelihood) where log_alpha has shape [T, K].
    """
    K = log_pi.shape[0]
    T = len(observations)
    log_alpha = np.zeros((T, K))

    # t=0
    log_alpha[0] = log_pi + log_B[:, observations[0]]

    for t in range(1, T):
        for j in range(K):
            log_alpha[t, j] = (
                np.logaddexp.reduce(log_alpha[t - 1] + log_A[:, j])
                + log_B[j, observations[t]]
            )

    log_likelihood = float(np.logaddexp.reduce(log_alpha[-1]))
    return log_alpha, log_likelihood


def hmm_backward(
    log_A: np.ndarray,
    log_B: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    """Backward algorithm for discrete HMM.

    Args:
        log_A: Log transition matrix, shape [K, K].
        log_B: Log emission matrix, shape [K, C].
        observations: Integer observation sequence, shape [T].

    Returns:
        log_beta, shape [T, K].
    """
    K = log_A.shape[0]
    T = len(observations)
    log_beta = np.zeros((T, K))
    # log_beta[T-1] = 0 (log(1) = 0)

    for t in range(T - 2, -1, -1):
        for i in range(K):
            log_beta[t, i] = np.logaddexp.reduce(
                log_A[i, :] + log_B[:, observations[t + 1]] + log_beta[t + 1]
            )

    return log_beta


def hmm_e_step(
    log_pi: np.ndarray,
    log_A: np.ndarray,
    log_B: np.ndarray,
    observations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """E-step: compute posterior state probabilities and pairwise posteriors.

    Args:
        log_pi: Log initial state probabilities, shape [K].
        log_A: Log transition matrix, shape [K, K].
        log_B: Log emission matrix, shape [K, C].
        observations: Integer observation sequence, shape [T].

    Returns:
        Tuple of (gamma, xi, log_likelihood) where:
        - gamma: shape [T, K], posterior state probabilities
        - xi: shape [T-1, K, K], pairwise posteriors
        - log_likelihood: marginal log-likelihood
    """
    log_alpha, log_likelihood = hmm_forward(log_pi, log_A, log_B, observations)
    log_beta = hmm_backward(log_A, log_B, observations)

    # gamma[t, i] = P(z_t=i | obs)
    log_gamma = log_alpha + log_beta
    log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma)

    # xi[t, i, j] = P(z_t=i, z_{t+1}=j | obs)
    T, K = log_alpha.shape
    xi = np.zeros((T - 1, K, K))
    for t in range(T - 1):
        log_xi_t = (
            log_alpha[t, :, None]
            + log_A
            + log_B[None, :, observations[t + 1]]
            + log_beta[t + 1, None, :]
        )
        log_xi_t -= np.logaddexp.reduce(log_xi_t.ravel())
        xi[t] = np.exp(log_xi_t)

    return gamma, xi, log_likelihood


def hmm_m_step(
    gamma_list: list[np.ndarray],
    xi_list: list[np.ndarray],
    observations_list: list[np.ndarray],
    num_classes: int,
    smoothing: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """M-step: update HMM parameters from sufficient statistics.

    Args:
        gamma_list: List of gamma arrays from E-step, each [T_i, K].
        xi_list: List of xi arrays from E-step, each [T_i-1, K, K].
        observations_list: List of observation sequences.
        num_classes: Number of emission classes (C).
        smoothing: Laplace smoothing constant.

    Returns:
        Tuple of (log_pi, log_A, log_B).
    """
    K = gamma_list[0].shape[1]

    # Initial state distribution
    pi = np.zeros(K) + smoothing
    for gamma in gamma_list:
        pi += gamma[0]
    pi /= pi.sum()

    # Transition matrix
    A = np.zeros((K, K)) + smoothing
    for xi in xi_list:
        A += xi.sum(axis=0)
    A /= A.sum(axis=1, keepdims=True)

    # Emission matrix
    B = np.zeros((K, num_classes)) + smoothing
    for gamma, obs in zip(gamma_list, observations_list):
        for t in range(len(obs)):
            B[:, obs[t]] += gamma[t]
    B /= B.sum(axis=1, keepdims=True)

    return np.log(pi), np.log(A), np.log(B)


def fit_hmm_em(
    observations_list: list[np.ndarray],
    num_states: int,
    num_classes: int,
    num_iters: int = 200,
    seed: int = 0,
    test_observations: list[np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], list[float], list[float]]:
    """Fit a discrete HMM using Expectation-Maximization.

    Args:
        observations_list: List of integer observation sequences, each shape [T_i].
        num_states: Number of hidden states K.
        num_classes: Number of emission classes C.
        num_iters: Number of EM iterations.
        seed: Random seed for initialization.
        test_observations: Optional held-out sequences for tracking test LL.

    Returns:
        Tuple of (params, train_log_likelihoods, test_log_likelihoods) where
        params has keys "log_pi", "log_A", "log_B".
    """
    rng = np.random.RandomState(seed)
    K, C = num_states, num_classes

    # Random initialization (Dirichlet-like)
    pi = rng.dirichlet(np.ones(K))
    A = rng.dirichlet(np.ones(K), size=K)
    B = rng.dirichlet(np.ones(C), size=K)

    log_pi = np.log(pi)
    log_A = np.log(A)
    log_B = np.log(B)

    train_lls: list[float] = []
    test_lls: list[float] = []

    for iteration in range(num_iters):
        # E-step
        gamma_list = []
        xi_list = []
        total_ll = 0.0

        for obs in observations_list:
            gamma, xi, ll = hmm_e_step(log_pi, log_A, log_B, obs)
            gamma_list.append(gamma)
            xi_list.append(xi)
            total_ll += ll

        avg_ll = total_ll / len(observations_list)
        train_lls.append(avg_ll)

        # Evaluate on test set
        if test_observations:
            test_ll = np.mean(
                [
                    hmm_marginal_log_prob(
                        {"log_pi": log_pi, "log_A": log_A, "log_B": log_B}, seq
                    )
                    for seq in test_observations
                ]
            )
            test_lls.append(float(test_ll))

        if iteration % 50 == 0:
            msg = f"    EM iter {iteration}: train LL = {avg_ll:.2f}"
            if test_lls:
                msg += f", test LL = {test_lls[-1]:.2f}"
            logging.info(msg)

        # M-step
        log_pi, log_A, log_B = hmm_m_step(gamma_list, xi_list, observations_list, C)

    return {"log_pi": log_pi, "log_A": log_A, "log_B": log_B}, train_lls, test_lls


def hmm_marginal_log_prob(
    params: dict[str, np.ndarray],
    observations: np.ndarray,
) -> float:
    """Compute marginal log-probability of an observation sequence.

    Args:
        params: HMM parameters with keys "log_pi", "log_A", "log_B".
        observations: Integer observation sequence, shape [T].

    Returns:
        Marginal log-probability.
    """
    _, ll = hmm_forward(
        params["log_pi"], params["log_A"], params["log_B"], observations
    )
    return ll


def hmm_sample(
    params: dict[str, np.ndarray],
    num_timesteps: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a sequence from the fitted HMM.

    Args:
        params: HMM parameters with keys "log_pi", "log_A", "log_B".
        num_timesteps: Length of sequence to generate.
        seed: Random seed.

    Returns:
        Tuple of (hidden_states, emissions), each shape [num_timesteps].
    """
    rng = np.random.RandomState(seed)
    pi = np.exp(params["log_pi"])
    A = np.exp(params["log_A"])
    B = np.exp(params["log_B"])

    states = np.zeros(num_timesteps, dtype=int)
    emissions = np.zeros(num_timesteps, dtype=int)

    states[0] = rng.choice(len(pi), p=pi)
    emissions[0] = rng.choice(B.shape[1], p=B[states[0]])

    for t in range(1, num_timesteps):
        states[t] = rng.choice(len(pi), p=A[states[t - 1]])
        emissions[t] = rng.choice(B.shape[1], p=B[states[t]])

    return states, emissions


def hmm_sample_with_temperature(
    params: dict[str, np.ndarray],
    num_timesteps: int,
    temperature: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample from the HMM with temperature-scaled emission distribution.

    Temperature controls diversity: T<1 = greedy (peaky), T=1 = original,
    T>1 = more uniform (diverse).

    Args:
        params: HMM parameters with keys "log_pi", "log_A", "log_B".
        num_timesteps: Length of sequence to generate.
        temperature: Emission temperature. Applied as softmax(log_B / T).
        seed: Random seed.

    Returns:
        Tuple of (hidden_states, emissions), each shape [num_timesteps].
    """
    rng = np.random.RandomState(seed)
    pi = np.exp(params["log_pi"])
    A = np.exp(params["log_A"])
    log_B = params["log_B"]

    # Temperature-scale the emission matrix
    scaled_log_B = log_B / max(temperature, 1e-8)
    # Softmax per row
    B_temp = np.exp(scaled_log_B - scaled_log_B.max(axis=-1, keepdims=True))
    B_temp = B_temp / B_temp.sum(axis=-1, keepdims=True)

    states = np.zeros(num_timesteps, dtype=int)
    emissions = np.zeros(num_timesteps, dtype=int)

    states[0] = rng.choice(len(pi), p=pi)
    emissions[0] = rng.choice(B_temp.shape[1], p=B_temp[states[0]])

    for t in range(1, num_timesteps):
        states[t] = rng.choice(len(pi), p=A[states[t - 1]])
        emissions[t] = rng.choice(B_temp.shape[1], p=B_temp[states[t]])

    return states, emissions


# =============================================================================
# STARTING POSE SELECTION (reused from ablation)
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
# HMM VIDEO RENDERING
# =============================================================================


def _build_dual_bar(
    width: int,
    current_frame_idx: int,
    hmm_state_indices: np.ndarray,
    d0_indices: np.ndarray,
    hmm_colors: np.ndarray,
    code_colors: np.ndarray,
    bar_height: int = 30,
    separator_height: int = 2,
    playhead_width: int = 3,
) -> np.ndarray:
    """Build a stacked bar with HMM state on top, VQ code on bottom.

    Args:
        width: Width of the bar in pixels.
        current_frame_idx: Current frame index for playhead.
        hmm_state_indices: HMM hidden state per frame, shape [T].
        d0_indices: VQ D0 code per frame, shape [T].
        hmm_colors: Colors for HMM states, shape [K, 3].
        code_colors: Colors for VQ codes, shape [C, 3].
        bar_height: Height of each bar in pixels.
        separator_height: Height of separator between bars.
        playhead_width: Width of playhead marker.

    Returns:
        Stacked bar image, shape [total_height, width, 3].
    """
    total_height = 2 * bar_height + separator_height
    bar_img = np.ones((total_height, width, 3), dtype=np.uint8) * 255
    num_frames = len(d0_indices)

    rows = [
        (0, hmm_state_indices, hmm_colors),
        (bar_height + separator_height, d0_indices, code_colors),
    ]

    playhead_x = int(current_frame_idx * width / num_frames)
    playhead_x = min(playhead_x, width - playhead_width)

    for y_start, indices, colors in rows:
        for j, idx in enumerate(indices):
            x_start = int(j * width / num_frames)
            x_end = int((j + 1) * width / num_frames)
            bar_img[y_start : y_start + bar_height, x_start:x_end] = colors[
                int(idx) % len(colors)
            ]

        # Playhead
        if playhead_x > 0:
            bar_img[y_start : y_start + bar_height, playhead_x - 1 : playhead_x] = [
                50,
                50,
                50,
            ]
        bar_img[
            y_start : y_start + bar_height,
            playhead_x : playhead_x + playhead_width,
        ] = [255, 255, 255]
        if playhead_x + playhead_width < width:
            bar_img[
                y_start : y_start + bar_height,
                playhead_x + playhead_width : playhead_x + playhead_width + 1,
            ] = [50, 50, 50]

    # Separator
    bar_img[bar_height : bar_height + separator_height, :] = [50, 50, 50]

    return bar_img


def render_hmm_video(
    env: Any,
    rollout_states: list[Any],
    output_path: str | Path,
    camera: str | None,
    width: int,
    height: int,
    fps: int,
    d0_indices: np.ndarray,
    hmm_state_indices: np.ndarray,
    num_codes: int,
    num_hmm_states: int,
    rewards: np.ndarray | None = None,
    bar_height: int = 30,
) -> str:
    """Render free-loop rollout with HMM state bar on top, VQ code bar below.

    Args:
        env: Environment with render method.
        rollout_states: Sequence of environment states.
        output_path: Path to save video.
        camera: Camera name.
        width: Video width.
        height: Video height.
        fps: Frames per second.
        d0_indices: VQ D0 code per frame, shape [T].
        hmm_state_indices: HMM hidden state per frame, shape [T].
        num_codes: Number of VQ codes.
        num_hmm_states: Number of HMM hidden states (K).
        rewards: Optional rewards per frame.
        bar_height: Height of each bar in pixels.

    Returns:
        Path to saved video.
    """
    import imageio

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"  Rendering {len(rollout_states)} frames...")
    frames = env.render(rollout_states, camera=camera, height=height, width=width)

    code_colors = get_nature_colormap(num_codes)
    # Use a distinct colormap for HMM states so they're visually separate
    hmm_colors = get_nature_colormap(num_hmm_states)

    logging.info("  Adding HMM + VQ overlays...")
    processed = []
    for i, frame in enumerate(frames):
        # Build dual bar (HMM state top, VQ code bottom)
        bar = _build_dual_bar(
            width=frame.shape[1],
            current_frame_idx=i,
            hmm_state_indices=hmm_state_indices,
            d0_indices=d0_indices,
            hmm_colors=hmm_colors,
            code_colors=code_colors,
            bar_height=bar_height,
        )
        frame = np.vstack([frame, bar])

        # Text badge: HMM state + VQ code
        lines = []
        if i < len(hmm_state_indices):
            lines.append(f"HMM:{int(hmm_state_indices[i])}")
        if i < len(d0_indices):
            lines.append(f"D0:{int(d0_indices[i])}")
        if rewards is not None and i < len(rewards):
            lines.append(f"R:{float(rewards[i]):.2f}")

        if lines:
            frame = add_multi_line_overlay(
                frame, lines, start_position=(10, 10), font_size=16
            )

        processed.append(frame)

    logging.info(f"  Writing video ({len(processed)} frames at {fps} fps)...")
    with imageio.get_writer(str(output_path), fps=fps) as writer:
        for frame in processed:
            writer.append_data(frame)

    logging.info(f"  Saved to {output_path}")
    return str(output_path)


# =============================================================================
# DECODER-ONLY STEP FUNCTION
# =============================================================================


def make_decoder_only_step_fn(
    cfg: DictConfig,
    policy_params: tuple[Any, Any],
) -> tuple[Any, int]:
    """Build a decoder-only function that maps (d0_code_index, obs) -> action.

    The encoder is NOT used at all. D1 residual is zero by construction.

    Args:
        cfg: Checkpoint config with network_config section.
        policy_params: Tuple of (normalizer_state, policy_params).

    Returns:
        Tuple of (decode_step_fn, action_size) where decode_step_fn is
        (d0_code_index: int, obs: dict) -> action: jnp.ndarray.
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
        f"use_continuous={use_continuous}, continuous_dim={continuous_dim}"
    )

    # Assert D1 is not used
    assert codebook_0.shape == (
        cfg.network_config.num_codes,
        latent_dim,
    ), f"Codebook shape mismatch: {codebook_0.shape}"

    def decode_step(d0_code_index: int, obs: dict) -> jnp.ndarray:
        """Decode a single D0 code to an action.

        Args:
            d0_code_index: D0 code index.
            obs: Environment observation dict.

        Returns:
            Action array, shape [action_size].
        """
        z_q = codebook_0[d0_code_index]

        flat_obs = flatten_obs_dict(obs)
        proprio_norm = running_statistics.normalize(
            flat_obs["proprioception"], normalizer_state.proprioception
        )

        if use_continuous:
            # Insert zeros for continuous latent between z_q and proprio
            z_e_zeros = jnp.zeros(continuous_dim)
            x = jnp.concatenate([z_q, z_e_zeros, proprio_norm], axis=-1)
        else:
            x = jnp.concatenate([z_q, proprio_norm], axis=-1)

        action_params, _ = decoder.apply({"params": decoder_params}, x)
        action = jnp.array(action_dist.mode(action_params))
        return action

    return decode_step, action_size


# =============================================================================
# HMM FITTING DIAGNOSTICS
# =============================================================================


def plot_em_curves(
    all_curves: dict[int, list[float]],
    output_path: Path,
) -> str:
    """Plot EM log-likelihood curves for each K.

    Args:
        all_curves: Mapping from K to list of per-iteration log-likelihoods.
        output_path: Path to save PNG.

    Returns:
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for K, curve in sorted(all_curves.items()):
        ax.plot(curve, label=f"K={K}")
    ax.set_xlabel("EM Iteration")
    ax.set_ylabel("Avg Log-Likelihood")
    ax.set_title("HMM EM Convergence")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_train_test_ll(
    train_lls: list[float],
    test_lls: list[float],
    best_K: int,
    output_path: Path,
) -> str:
    """Plot train and test log-likelihood curves over EM iterations.

    Args:
        train_lls: Per-iteration train log-likelihood.
        test_lls: Per-iteration test log-likelihood.
        best_K: The number of HMM states (for title).
        output_path: Path to save PNG.

    Returns:
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_lls, label="Train", color="steelblue")
    if test_lls:
        ax.plot(test_lls, label="Test", color="darkorange")
    ax.set_xlabel("EM Iteration")
    ax.set_ylabel("Avg Log-Likelihood")
    ax.set_title(f"HMM Train vs Test LL (K={best_K})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_model_selection(
    test_lls: dict[int, float],
    output_path: Path,
) -> str:
    """Bar chart of test log-likelihood vs K.

    Args:
        test_lls: Mapping from K to test log-likelihood.
        output_path: Path to save PNG.

    Returns:
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ks = sorted(test_lls.keys())
    vals = [test_lls[k] for k in ks]
    bars = ax.bar([str(k) for k in ks], vals, color="steelblue")

    # Highlight best
    best_idx = int(np.argmax(vals))
    bars[best_idx].set_color("darkorange")

    ax.set_xlabel("Number of Hidden States (K)")
    ax.set_ylabel("Test Log-Likelihood")
    ax.set_title("HMM Model Selection")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_transition_matrix(
    params: dict[str, np.ndarray],
    output_path: Path,
) -> str:
    """Heatmap of HMM transition matrix.

    Args:
        params: HMM parameters with "log_A".
        output_path: Path to save PNG.

    Returns:
        Path to saved figure.
    """
    A = np.exp(params["log_A"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(A, cmap="Blues", vmin=0, vmax=A.max())
    ax.set_xlabel("To State")
    ax.set_ylabel("From State")
    ax.set_title("HMM Transition Matrix")
    fig.colorbar(im, ax=ax, label="Probability")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_emission_matrix(
    params: dict[str, np.ndarray],
    num_codes: int,
    output_path: Path,
) -> str:
    """Heatmap of HMM emission matrix (states x codes).

    Args:
        params: HMM parameters with "log_B".
        num_codes: Number of VQ codes.
        output_path: Path to save PNG.

    Returns:
        Path to saved figure.
    """
    B = np.exp(params["log_B"])
    fig, ax = plt.subplots(figsize=(max(8, num_codes * 0.3), 5))
    im = ax.imshow(B, cmap="Oranges", aspect="auto", vmin=0, vmax=B.max())
    ax.set_xlabel("VQ Code")
    ax.set_ylabel("HMM State")
    ax.set_title("HMM Emission Matrix (P(code | state))")
    fig.colorbar(im, ax=ax, label="Probability")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_stationary_distribution(
    params: dict[str, np.ndarray],
    output_path: Path,
) -> str:
    """Bar chart of HMM stationary distribution.

    Args:
        params: HMM parameters with "log_A".
        output_path: Path to save PNG.

    Returns:
        Path to saved figure.
    """
    A = np.exp(params["log_A"])
    # Compute stationary distribution via eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(A.T)
    # Find eigenvector for eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    stationary = np.real(eigenvectors[:, idx])
    stationary = np.abs(stationary) / np.abs(stationary).sum()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(stationary)), stationary, color="steelblue")
    ax.set_xlabel("HMM State")
    ax.set_ylabel("Stationary Probability")
    ax.set_title("HMM Stationary Distribution")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


# =============================================================================
# EMISSION GALLERY HELPERS
# =============================================================================


def plot_emission_bar(
    emission_probs: np.ndarray,
    state_idx: int,
    top_codes: np.ndarray,
    top_probs: np.ndarray,
    num_codes: int,
    output_path: Path,
) -> str:
    """Bar chart of a single HMM state's emission distribution.

    Highlights the top-N codes with distinct colors, dims the rest.

    Args:
        emission_probs: Full emission row for this state, shape [C].
        state_idx: HMM hidden state index (for title).
        top_codes: Indices of top-N codes.
        top_probs: Probabilities of top-N codes.
        num_codes: Total number of VQ codes.
        output_path: Path to save PNG.

    Returns:
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(max(6, num_codes * 0.25), 2.5))

    colors = ["#cccccc"] * num_codes
    highlight_cmap = plt.cm.tab10
    for rank, c in enumerate(top_codes):
        colors[int(c)] = highlight_cmap(rank / max(len(top_codes) - 1, 1))

    ax.bar(range(num_codes), emission_probs, color=colors, edgecolor="none")
    ax.set_xlabel("VQ Code")
    ax.set_ylabel("P(code | state)")
    ax.set_title(f"State {state_idx} Emission Distribution")
    ax.set_xlim(-0.5, num_codes - 0.5)
    plt.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def build_emission_gallery_html(
    per_state_video_paths: list[list[str]],
    per_state_code_labels: list[list[str]],
    bar_chart_paths: list[str],
    state_labels: list[str],
    title: str,
) -> str:
    """Build HTML with one slider position per hidden state.

    Each position shows top-N code videos side-by-side on top,
    with an emission probability bar chart below.

    Args:
        per_state_video_paths: List of K lists, each with N video paths.
        per_state_code_labels: List of K lists, each with N code labels.
        bar_chart_paths: List of K bar chart PNG paths.
        state_labels: Label for each state (shown with slider).
        title: Page title.

    Returns:
        HTML string.
    """
    import base64

    num_states = len(per_state_video_paths)

    # Pre-encode all videos and bar charts
    encoded_videos: list[list[str]] = []
    for state_vids in per_state_video_paths:
        state_encoded = []
        for vpath in state_vids:
            try:
                with open(vpath, "rb") as f:
                    state_encoded.append(base64.b64encode(f.read()).decode("ascii"))
            except Exception:
                state_encoded.append("")
        encoded_videos.append(state_encoded)

    encoded_bars: list[str] = []
    for bpath in bar_chart_paths:
        try:
            with open(bpath, "rb") as f:
                encoded_bars.append(base64.b64encode(f.read()).decode("ascii"))
        except Exception:
            encoded_bars.append("")

    # Build per-state div blocks
    state_divs = []
    for k in range(num_states):
        videos_html_parts = []
        for j, (b64, label) in enumerate(
            zip(encoded_videos[k], per_state_code_labels[k])
        ):
            videos_html_parts.append(
                f'<div class="vcell">'
                f'<video src="data:video/mp4;base64,{b64}" '
                f"autoplay loop muted playsinline></video>"
                f'<div class="clbl">{label}</div></div>'
            )
        videos_row = "\n".join(videos_html_parts)

        bar_img = (
            f'<img src="data:image/png;base64,{encoded_bars[k]}" ' f'class="barchart">'
        )

        div = (
            f'<div class="state-slide" id="state-{k}" '
            f'style="display:none;">\n'
            f'<div class="vrow">{videos_row}</div>\n'
            f"{bar_img}\n</div>"
        )
        state_divs.append(div)

    all_divs = "\n".join(state_divs)
    labels_json = json.dumps(state_labels)

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>{title}</title>
<style>
  body {{ font-family: sans-serif; text-align: center;
         background: #fff; color: #222; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
  input[type=range] {{ width: 80%; margin: 15px 0; }}
  .label {{ font-size: 18px; font-weight: bold; margin: 10px 0; }}
  .vrow {{ display: flex; justify-content: center;
           gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }}
  .vcell {{ display: flex; flex-direction: column;
            align-items: center; max-width: 220px; }}
  .vcell video {{ width: 200px; height: 150px;
                  object-fit: cover; border: 1px solid #888;
                  border-radius: 4px; }}
  .clbl {{ font-size: 12px; margin-top: 4px; color: #555; }}
  .barchart {{ max-width: 100%; border: 1px solid #ccc;
               border-radius: 4px; margin-top: 6px; }}
</style>
</head>
<body>
<div class="container">
  <h2>{title}</h2>
  {all_divs}
  <input type="range" id="slider" min="0"
         max="{num_states - 1}" value="0">
  <div class="label" id="lbl"></div>
</div>
<script>
var labels = {labels_json};
var slider = document.getElementById('slider');
var lbl = document.getElementById('lbl');
function update() {{
  var i = parseInt(slider.value);
  for (var s = 0; s < {num_states}; s++) {{
    document.getElementById('state-' + s).style.display =
      (s === i) ? 'block' : 'none';
  }}
  lbl.textContent = labels[i];
}}
slider.addEventListener('input', update);
update();
</script>
</body>
</html>"""
    return html


# =============================================================================
# CONTROL EXPERIMENT HELPERS
# =============================================================================


def compute_empirical_transition_matrix(
    code_sequences: list[np.ndarray],
    num_codes: int,
    exclude_self: bool = False,
) -> np.ndarray:
    """Compute row-normalized empirical transition matrix from code sequences.

    Args:
        code_sequences: List of integer code sequences, each shape [T_i].
        num_codes: Number of VQ codes (C).
        exclude_self: If True, zero diagonal before normalizing.

    Returns:
        Row-normalized transition matrix, shape [C, C].
    """
    counts = np.zeros((num_codes, num_codes), dtype=float)
    for seq in code_sequences:
        for t in range(len(seq) - 1):
            counts[int(seq[t]), int(seq[t + 1])] += 1

    if exclude_self:
        np.fill_diagonal(counts, 0.0)

    # Row-normalize; zero-count rows → uniform 1/C
    row_sums = counts.sum(axis=1, keepdims=True)
    zero_rows = row_sums.squeeze() == 0
    counts[zero_rows] = 1.0  # uniform before normalization
    row_sums[zero_rows] = num_codes
    trans = counts / row_sums
    return trans


def compute_transition_metrics(
    trans_matrix: np.ndarray,
    sparsity_threshold: float = 0.01,
) -> dict[str, Any]:
    """Compute entropy and sparsity metrics for a transition matrix.

    Args:
        trans_matrix: Row-normalized transition matrix, shape [C, C].
        sparsity_threshold: Entries below this are considered near-zero.

    Returns:
        Dict with per_row_entropy, mean_entropy, uniform_entropy,
        entropy_reduction, sparsity_ratio, num_near_zero.
    """
    C = trans_matrix.shape[0]
    uniform_entropy = np.log2(C)

    # Per-row Shannon entropy (bits)
    per_row_entropy = np.zeros(C)
    for i in range(C):
        row = trans_matrix[i]
        nonzero = row[row > 0]
        per_row_entropy[i] = -np.sum(nonzero * np.log2(nonzero))

    mean_entropy = float(np.mean(per_row_entropy))
    entropy_reduction = float(uniform_entropy - mean_entropy)

    # Sparsity
    num_near_zero = int(np.sum(trans_matrix < sparsity_threshold))
    total_entries = C * C
    sparsity_ratio = num_near_zero / total_entries

    return {
        "per_row_entropy": per_row_entropy,
        "mean_entropy": mean_entropy,
        "uniform_entropy": float(uniform_entropy),
        "entropy_reduction": entropy_reduction,
        "sparsity_ratio": sparsity_ratio,
        "num_near_zero": num_near_zero,
        "total_entries": total_entries,
    }


def plot_transition_analysis(
    trans_with_self: np.ndarray,
    trans_no_self: np.ndarray,
    metrics_with: dict[str, Any],
    metrics_no: dict[str, Any],
    num_codes: int,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot 2x2 transition analysis figure.

    Top row: heatmaps (with/without self-transitions, log scale).
    Bottom row: entropy bar chart + summary text.

    Args:
        trans_with_self: Transition matrix with self-transitions, shape [C, C].
        trans_no_self: Transition matrix without self-transitions, shape [C, C].
        metrics_with: Metrics dict for with-self matrix.
        metrics_no: Metrics dict for no-self matrix.
        num_codes: Number of VQ codes.
        output_dir: Directory to save plots.

    Returns:
        Dict mapping plot name to file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = {}

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Top-left: heatmap with self-transitions (log scale)
    ax = axes[0, 0]
    # Add small epsilon to avoid log(0)
    im = ax.imshow(np.log10(trans_with_self + 1e-10), cmap="Blues", aspect="auto")
    ax.set_title("Transition Matrix (with self)")
    ax.set_xlabel("To Code")
    ax.set_ylabel("From Code")
    fig.colorbar(im, ax=ax, label="log10(P)")

    # Top-right: heatmap without self-transitions
    ax = axes[0, 1]
    im = ax.imshow(np.log10(trans_no_self + 1e-10), cmap="Oranges", aspect="auto")
    ax.set_title("Transition Matrix (no self)")
    ax.set_xlabel("To Code")
    ax.set_ylabel("From Code")
    fig.colorbar(im, ax=ax, label="log10(P)")

    # Bottom-left: entropy bar chart
    ax = axes[1, 0]
    x = np.arange(num_codes)
    width = 0.35
    ax.bar(
        x - width / 2,
        metrics_with["per_row_entropy"],
        width,
        label="With self",
        color="steelblue",
        alpha=0.7,
    )
    ax.bar(
        x + width / 2,
        metrics_no["per_row_entropy"],
        width,
        label="No self",
        color="darkorange",
        alpha=0.7,
    )
    ax.axhline(
        metrics_with["uniform_entropy"],
        color="red",
        linestyle="--",
        label=f"Uniform ({metrics_with['uniform_entropy']:.2f} bits)",
    )
    ax.set_xlabel("From Code")
    ax.set_ylabel("Shannon Entropy (bits)")
    ax.set_title("Per-Row Transition Entropy")
    ax.legend(fontsize=8)

    # Bottom-right: summary text
    ax = axes[1, 1]
    ax.axis("off")
    summary = (
        f"WITH self-transitions:\n"
        f"  Mean entropy: {metrics_with['mean_entropy']:.3f} bits\n"
        f"  Uniform entropy: {metrics_with['uniform_entropy']:.3f} bits\n"
        f"  Entropy reduction: {metrics_with['entropy_reduction']:.3f} bits\n"
        f"  Sparsity: {metrics_with['sparsity_ratio']:.1%} "
        f"({metrics_with['num_near_zero']}/{metrics_with['total_entries']})\n"
        f"\nWITHOUT self-transitions:\n"
        f"  Mean entropy: {metrics_no['mean_entropy']:.3f} bits\n"
        f"  Entropy reduction: {metrics_no['entropy_reduction']:.3f} bits\n"
        f"  Sparsity: {metrics_no['sparsity_ratio']:.1%} "
        f"({metrics_no['num_near_zero']}/{metrics_no['total_entries']})\n"
        f"\nNum codes: {num_codes}"
    )
    ax.text(
        0.05,
        0.95,
        summary,
        transform=ax.transAxes,
        verticalalignment="top",
        fontfamily="monospace",
        fontsize=10,
    )

    plt.tight_layout()
    path = output_dir / "transition_analysis.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots["transition_analysis"] = path

    return plots


def generate_marginal_random_codes(
    code_sequences: list[np.ndarray],
    num_code_steps: int,
    num_codes: int,
    seed: int,
) -> np.ndarray:
    """Sample codes from empirical marginal p(c), destroying temporal order.

    Args:
        code_sequences: List of integer code sequences.
        num_code_steps: Number of codes to sample.
        num_codes: Number of VQ codes.
        seed: Random seed.

    Returns:
        Sampled code indices, shape [num_code_steps].
    """
    all_codes = np.concatenate(code_sequences)
    counts = np.bincount(all_codes.astype(int), minlength=num_codes).astype(float)
    p = counts / counts.sum()

    rng = np.random.RandomState(seed)
    return rng.choice(num_codes, size=num_code_steps, p=p)


def generate_uniform_random_codes(
    num_code_steps: int,
    num_codes: int,
    seed: int,
) -> np.ndarray:
    """Sample codes uniformly at random.

    Args:
        num_code_steps: Number of codes to sample.
        num_codes: Number of VQ codes.
        seed: Random seed.

    Returns:
        Sampled code indices, shape [num_code_steps].
    """
    return np.random.RandomState(seed).randint(0, num_codes, size=num_code_steps)


def run_code_sequence_rollout(
    env: Any,
    jit_decode: Any,
    jit_reset: Any,
    jit_step: Any,
    code_sequence: np.ndarray,
    H: int,
    max_steps: int,
    seed: int,
    collect_states: bool = False,
    collect_qpos: bool = False,
) -> dict[str, Any]:
    """Run a free-loop rollout with a given code sequence.

    Args:
        env: Environment instance.
        jit_decode: JIT-compiled decode_step function.
        jit_reset: JIT-compiled env.reset.
        jit_step: JIT-compiled env.step.
        code_sequence: Code indices, shape [num_code_steps].
        H: Commitment horizon (sim steps per code).
        max_steps: Maximum simulation steps.
        seed: Random seed for reset.
        collect_states: If True, store states for rendering.
        collect_qpos: If True, collect qpos/qvel at each step.

    Returns:
        Dict with survival_steps, rewards, mean_reward, code_indices,
        and optionally states_for_render, qpos_arr, qvel_arr.
    """
    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    state = jit_reset(reset_rng)

    code_indices: list[int] = []
    rewards: list[float] = []
    states_for_render: list[Any] = [] if collect_states else None
    qpos_list: list[np.ndarray] = [] if collect_qpos else None
    qvel_list: list[np.ndarray] = [] if collect_qpos else None

    for t in range(max_steps):
        code_t = int(code_sequence[min(t // H, len(code_sequence) - 1)])
        code_indices.append(code_t)

        action = jit_decode(code_t, state.obs)
        if collect_states:
            states_for_render.append(state)
        if collect_qpos:
            qpos_list.append(np.array(state.data.qpos))
            qvel_list.append(np.array(state.data.qvel))

        next_state = jit_step(state, action)
        rewards.append(float(next_state.reward))

        if jnp.any(jnp.isnan(next_state.reward)):
            break
        state = next_state

    return {
        "survival_steps": len(code_indices),
        "rewards": np.array(rewards),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "code_indices": np.array(code_indices),
        "states_for_render": states_for_render,
        "qpos_arr": np.stack(qpos_list) if collect_qpos else None,
        "qvel_arr": np.stack(qvel_list) if collect_qpos else None,
    }


def compute_wasserstein_distances(
    real_qpos_joints: np.ndarray,
    real_qvel_joints: np.ndarray,
    gen_qpos_joints: np.ndarray,
    gen_qvel_joints: np.ndarray,
) -> dict[str, Any]:
    """Compute per-joint 1D Wasserstein distances between real and generated data.

    Args:
        real_qpos_joints: Real joint positions, shape [N_real, num_joints].
        real_qvel_joints: Real joint velocities, shape [N_real, num_joints].
        gen_qpos_joints: Generated joint positions, shape [N_gen, num_joints].
        gen_qvel_joints: Generated joint velocities, shape [N_gen, num_joints].

    Returns:
        Dict with qpos_w1 [num_joints], qvel_w1 [num_joints],
        mean_qpos_w1 (float), mean_qvel_w1 (float).
    """
    num_joints = real_qpos_joints.shape[1]
    qpos_w1 = np.array(
        [
            wasserstein_distance(real_qpos_joints[:, j], gen_qpos_joints[:, j])
            for j in range(num_joints)
        ]
    )
    qvel_w1 = np.array(
        [
            wasserstein_distance(real_qvel_joints[:, j], gen_qvel_joints[:, j])
            for j in range(num_joints)
        ]
    )
    return {
        "qpos_w1": qpos_w1,
        "qvel_w1": qvel_w1,
        "mean_qpos_w1": float(np.mean(qpos_w1)),
        "mean_qvel_w1": float(np.mean(qvel_w1)),
    }


def plot_random_null_comparison(
    hmm_w1: dict[str, Any],
    marginal_w1: dict[str, Any],
    uniform_w1: dict[str, Any],
    hmm_results: list[dict],
    marginal_results: list[dict],
    uniform_results: list[dict],
    joint_names: list[str],
    output_dir: Path,
) -> dict[str, Path]:
    """Plot per-joint Wasserstein distances and survival for HMM vs random controls.

    Args:
        hmm_w1: W1 distances dict for HMM condition.
        marginal_w1: W1 distances dict for marginal random condition.
        uniform_w1: W1 distances dict for uniform random condition.
        hmm_results: List of rollout result dicts for HMM condition.
        marginal_results: List of rollout result dicts for marginal random.
        uniform_results: List of rollout result dicts for uniform random.
        joint_names: List of joint name strings.
        output_dir: Directory to save plots.

    Returns:
        Dict mapping plot name to file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = {}

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    num_joints = len(hmm_w1["qpos_w1"])
    joint_names = joint_names[:num_joints]
    x = np.arange(num_joints)
    bar_width = 0.25
    colors = ["steelblue", "darkorange", "firebrick"]
    cond_labels = ["HMM", "Marginal", "Uniform"]
    w1_dicts = [hmm_w1, marginal_w1, uniform_w1]

    # Top-left: per-joint qpos W1
    ax = axes[0, 0]
    for i, (label, w1, color) in enumerate(zip(cond_labels, w1_dicts, colors)):
        ax.bar(x + i * bar_width, w1["qpos_w1"], bar_width, label=label, color=color)
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(joint_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("W1 Distance")
    ax.set_title("Per-Joint qpos Wasserstein Distance")
    ax.legend(fontsize=8)

    # Top-right: per-joint qvel W1
    ax = axes[0, 1]
    for i, (label, w1, color) in enumerate(zip(cond_labels, w1_dicts, colors)):
        ax.bar(x + i * bar_width, w1["qvel_w1"], bar_width, label=label, color=color)
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(joint_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("W1 Distance")
    ax.set_title("Per-Joint qvel Wasserstein Distance")
    ax.legend(fontsize=8)

    # Bottom-left: mean W1 summary
    ax = axes[1, 0]
    summary_x = np.arange(2)
    for i, (label, w1, color) in enumerate(zip(cond_labels, w1_dicts, colors)):
        vals = [w1["mean_qpos_w1"], w1["mean_qvel_w1"]]
        ax.bar(summary_x + i * bar_width, vals, bar_width, label=label, color=color)
    ax.set_xticks(summary_x + bar_width)
    ax.set_xticklabels(["qpos", "qvel"])
    ax.set_ylabel("Mean W1 Distance")
    ax.set_title("Mean Wasserstein Distance Summary")
    ax.legend()

    # Bottom-right: survival bar chart (sanity check)
    ax = axes[1, 1]
    conditions = [
        ("HMM", hmm_results, "steelblue"),
        ("Marginal", marginal_results, "darkorange"),
        ("Uniform", uniform_results, "firebrick"),
    ]
    bar_data = []
    bar_labels_surv = []
    bar_errors = []
    for label, results, color in conditions:
        survivals = [r["survival_steps"] for r in results]
        bar_data.append(np.mean(survivals) if survivals else 0)
        bar_errors.append(np.std(survivals) if len(survivals) > 1 else 0)
        bar_labels_surv.append(label)
    ax.bar(
        bar_labels_surv,
        bar_data,
        yerr=bar_errors,
        color=colors,
        capsize=5,
    )
    ax.set_ylabel("Survival Steps")
    ax.set_title("Survival (mean +/- std)")

    plt.tight_layout()
    path = output_dir / "random_null_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots["random_null_comparison"] = path

    return plots


# =============================================================================
# MAIN PIPELINE
# =============================================================================


@hydra.main(version_base=None, config_path="../configs", config_name="hmm_prior")
def main(cfg: DictConfig):
    """Run HMM Prior pipeline: fit HMM on D0 codes, then generate free-loop behavior."""
    logging.set_verbosity(logging.INFO)

    print("=" * 60)
    print("HMM Prior: Fit + Free-Loop Generative Pipeline")
    print("=" * 60)

    output_dir = Path(cfg.output.base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load checkpoint and rollout data
    # ------------------------------------------------------------------
    logging.info("\n[Step 1] Loading checkpoint and rollout data...")

    ckpt = load_vq_checkpoint(cfg.checkpoint.path, step=cfg.checkpoint.step)
    vq_cfg = ckpt["cfg"]
    policy_params = ckpt["policy"]
    codebook_0 = get_codebook(policy_params, depth=0)
    num_codes = codebook_0.shape[0]
    logging.info(f"  Codebook: {num_codes} codes, dim={codebook_0.shape[1]}")

    # Select H5 path
    analysis_split = cfg.data.get("analysis_split", "test")
    if analysis_split == "test" and cfg.data.get("h5_path_test"):
        h5_path = cfg.data.h5_path_test
    else:
        h5_path = cfg.data.get("h5_path", cfg.data.h5_path_test)

    if not Path(h5_path).exists():
        raise FileNotFoundError(
            f"H5 file not found: {h5_path}\n"
            "Generate rollout data first:\n"
            "  python -m inference.run_inference checkpoint.path=/path/to/checkpoint"
        )

    results, h5_metadata = load_rollouts_from_h5(h5_path)
    logging.info(f"  Loaded {len(results)} rollouts from {h5_path}")

    # ------------------------------------------------------------------
    # Step 2: Extract and subsample D0 codes
    # ------------------------------------------------------------------
    logging.info("\n[Step 2] Extracting D0 code sequences...")

    H = cfg.free_loop.commitment_horizon
    logging.info(f"  Commitment horizon H={H}")

    # Subsample each clip's D0 codes at rate H
    all_codes_subsampled: list[np.ndarray] = []
    for r in results:
        codes = r.code_indices[::H]
        all_codes_subsampled.append(codes.astype(int))

    # Truncate to minimum length
    min_len = min(len(c) for c in all_codes_subsampled)
    all_codes_subsampled = [c[:min_len] for c in all_codes_subsampled]
    logging.info(
        f"  {len(all_codes_subsampled)} sequences, "
        f"subsampled length={min_len} (from T={len(results[0].code_indices)})"
    )

    # ------------------------------------------------------------------
    # Step 3: Train/test split
    # ------------------------------------------------------------------
    logging.info("\n[Step 3] Splitting train/test...")

    rng_split = np.random.RandomState(cfg.hmm.seed)
    n_total = len(all_codes_subsampled)
    n_train = max(1, int(n_total * cfg.hmm.train_ratio))
    perm = rng_split.permutation(n_total)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    train_emissions = [all_codes_subsampled[i] for i in train_idx]
    test_emissions = [all_codes_subsampled[i] for i in test_idx]
    logging.info(f"  Train: {len(train_emissions)}, Test: {len(test_emissions)}")

    if len(test_emissions) == 0:
        logging.warning("  No test sequences — using train for model selection")
        test_emissions = train_emissions

    # ------------------------------------------------------------------
    # Step 4: Sweep num_states
    # ------------------------------------------------------------------
    logging.info("\n[Step 4] Sweeping HMM num_states...")

    hmm_dir = output_dir / "hmm_fitting"
    hmm_dir.mkdir(parents=True, exist_ok=True)

    all_em_curves: dict[int, list[float]] = {}
    test_lls: dict[int, float] = {}
    all_params: dict[int, dict[str, np.ndarray]] = {}

    all_train_curves: dict[int, list[float]] = {}
    all_test_curves: dict[int, list[float]] = {}

    for K in cfg.hmm.num_states_sweep:
        logging.info(f"\n  Fitting K={K}...")
        params, train_curve, test_curve = fit_hmm_em(
            train_emissions,
            num_states=K,
            num_classes=num_codes,
            num_iters=cfg.hmm.num_em_iters,
            seed=cfg.hmm.seed + K,
            test_observations=test_emissions,
        )
        all_em_curves[K] = train_curve
        all_train_curves[K] = train_curve
        all_test_curves[K] = test_curve
        all_params[K] = params

        # Final test LL
        test_ll = test_curve[-1] if test_curve else float(train_curve[-1])
        test_lls[K] = float(test_ll)
        logging.info(f"  K={K}: test log-likelihood = {test_ll:.2f}")

    # Select best K
    best_K = max(test_lls, key=test_lls.get)
    best_params = all_params[best_K]
    logging.info(f"\n  Best K={best_K} (test LL={test_lls[best_K]:.2f})")

    # ------------------------------------------------------------------
    # Step 5: Save HMM diagnostics
    # ------------------------------------------------------------------
    logging.info("\n[Step 5] Saving HMM diagnostics...")

    em_curve_path = plot_em_curves(all_em_curves, hmm_dir / "em_curves.png")
    selection_path = plot_model_selection(test_lls, hmm_dir / "model_selection.png")
    trans_path = plot_transition_matrix(best_params, hmm_dir / "transition_matrix.png")
    emit_path = plot_emission_matrix(
        best_params, num_codes, hmm_dir / "emission_matrix.png"
    )
    stat_path = plot_stationary_distribution(
        best_params, hmm_dir / "stationary_distribution.png"
    )
    train_test_path = plot_train_test_ll(
        all_train_curves[best_K],
        all_test_curves[best_K],
        best_K,
        hmm_dir / "train_test_ll.png",
    )

    # Save params as npz
    np.savez(
        hmm_dir / "best_hmm_params.npz",
        log_pi=best_params["log_pi"],
        log_A=best_params["log_A"],
        log_B=best_params["log_B"],
        best_K=best_K,
    )

    # Save model selection results
    with open(hmm_dir / "model_selection.json", "w") as f:
        json.dump(
            {
                "test_log_likelihoods": {str(k): v for k, v in test_lls.items()},
                "best_K": best_K,
            },
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Step 6: Initialize WandB
    # ------------------------------------------------------------------
    wandb_enabled = False
    if cfg.wandb.get("enabled", False):
        try:
            import wandb

            wandb.init(
                project=cfg.wandb.get("project", "vqvae-eval"),
                entity=cfg.wandb.get("entity"),
                name=f"hmm_prior_{datetime.now().strftime('%y%m%d_%H%M%S')}",
                config={
                    "checkpoint_path": cfg.checkpoint.path,
                    "best_K": best_K,
                    "commitment_horizon": H,
                    "num_codes": num_codes,
                    "num_train_seqs": len(train_emissions),
                    "num_test_seqs": len(test_emissions),
                },
            )
            wandb_enabled = True
        except Exception as e:
            logging.warning(f"Failed to init WandB: {e}")

    wandb_items: dict[str, Any] = {}

    if wandb_enabled:
        import wandb

        # EM curves
        wandb_items["hmm_fitting/em_curves"] = wandb.Image(em_curve_path)
        wandb_items["hmm_fitting/model_selection"] = wandb.Image(selection_path)
        wandb_items["hmm_fitting/transition_matrix"] = wandb.Image(trans_path)
        wandb_items["hmm_fitting/emission_matrix"] = wandb.Image(emit_path)
        wandb_items["hmm_fitting/stationary_distribution"] = wandb.Image(stat_path)
        wandb_items["hmm_fitting/train_test_ll"] = wandb.Image(train_test_path)

    # ------------------------------------------------------------------
    # Step 7: Free-loop generative rollout
    # ------------------------------------------------------------------
    if not cfg.render.get("enabled", True):
        logging.info("\nRendering disabled, skipping free-loop rollout.")
    else:
        logging.info("\n[Step 7] Free-loop generative rollout...")

        # Build environment — keep original clip_length and termination config.
        # The free-loop ignores state.done (see rollout loop below) so neither
        # termination criteria nor clip truncation will cut the episode short.
        _, cfg_dict, env_cfg_ml = config_utils.prepare_config(cfg)
        reference_clips = ReferenceClips(
            data_path=vq_cfg.env_config.reference_data_path,
            n_frames_per_clip=cfg.data.get("clip_length", 500),
            keep_clips_idx=None,
        )

        # Select starting poses
        pose_clips = select_starting_clips(reference_clips)

        # Build decoder-only step function
        decode_step, action_size = make_decoder_only_step_fn(vq_cfg, policy_params)

        # Get camera name
        env_suffix = "-rodent"  # Default for rodent walker
        camera_name = f"{cfg.render.camera}{env_suffix}"

        max_steps = cfg.free_loop.max_steps
        free_seed = cfg.free_loop.seed

        video_dir = output_dir / "free_loop"
        video_dir.mkdir(parents=True, exist_ok=True)

        all_video_paths: list[str] = []
        all_video_labels: list[str] = []

        jit_decode = jax.jit(decode_step)

        for pose_name, clip_idx in pose_clips.items():
            logging.info(f"\n  Pose: {pose_name} (clip {clip_idx})")

            sub_clips = subset_clips(reference_clips, clip_idx)
            env = imitation.Imitation(config=env_cfg_ml, clips=sub_clips)
            jit_reset = jax.jit(env.reset)
            jit_step = jax.jit(env.step)

            # Sample HMM code sequence (keep hidden states for rendering)
            num_code_steps = max_steps // H + 1
            sampled_hmm_states, sampled_codes = hmm_sample(
                best_params, num_code_steps, seed=free_seed
            )
            logging.info(
                f"  Sampled {num_code_steps} codes from HMM "
                f"(unique codes: {len(np.unique(sampled_codes))}, "
                f"unique states: {len(np.unique(sampled_hmm_states))})"
            )

            # Run free-loop rollout
            rng = jax.random.PRNGKey(free_seed)
            rng, reset_rng = jax.random.split(rng)
            state = jit_reset(reset_rng)

            code_indices: list[int] = []
            hmm_state_indices: list[int] = []
            states_for_render: list[Any] = []
            rewards: list[float] = []

            for t in range(max_steps):
                code_t = int(sampled_codes[t // H])
                hmm_state_t = int(sampled_hmm_states[t // H])
                code_indices.append(code_t)
                hmm_state_indices.append(hmm_state_t)

                action = jit_decode(code_t, state.obs)
                states_for_render.append(state)

                next_state = jit_step(state, action)
                rewards.append(float(next_state.reward))

                # Ignore state.done (termination + clip truncation are irrelevant
                # for free-loop). Only break on NaN which indicates sim blowup.
                if jnp.any(jnp.isnan(next_state.reward)):
                    logging.info(f"  NaN detected at step {t}, stopping.")
                    break
                state = next_state

            actual_steps = len(code_indices)
            mean_reward = float(np.mean(rewards)) if rewards else 0.0
            logging.info(
                f"  Ran {actual_steps} steps, " f"mean reward={mean_reward:.2f}"
            )

            # Render video with HMM state bar on top of VQ code bar
            d0_indices = np.array(code_indices)
            hmm_states_arr = np.array(hmm_state_indices)

            # Build extra_info to show HMM state in text overlay
            extra_info = [
                {"HMM state": int(hmm_states_arr[t])} for t in range(len(code_indices))
            ]

            video_path = video_dir / f"free_loop_{pose_name}.mp4"
            render_hmm_video(
                env=env,
                rollout_states=states_for_render,
                output_path=video_path,
                camera=camera_name,
                width=cfg.render.width,
                height=cfg.render.height,
                fps=cfg.render.fps,
                d0_indices=d0_indices,
                hmm_state_indices=hmm_states_arr,
                num_codes=num_codes,
                num_hmm_states=best_K,
                rewards=np.array(rewards),
            )

            all_video_paths.append(str(video_path))
            all_video_labels.append(
                f"{pose_name} | {actual_steps} steps | " f"mean_r={mean_reward:.1f}"
            )

            # Increment seed for next pose
            free_seed += 1

        # Build slider HTML
        if all_video_paths:
            html = build_slider_html(
                all_video_paths,
                all_video_labels,
                "HMM Prior Free-Loop",
                media_type="video",
            )
            html_path = video_dir / "viewer.html"
            with open(html_path, "w") as f:
                f.write(html)
            logging.info(f"  Slider HTML saved to {html_path}")

            if wandb_enabled:
                wandb_items["free_loop/viewer"] = wandb.Html(html)

    # ------------------------------------------------------------------
    # Step 8: Temperature sweep (low_height pose only)
    # ------------------------------------------------------------------
    temperatures = list(cfg.free_loop.get("temperatures", [0.1, 0.5, 1.0, 2.0]))
    if cfg.render.get("enabled", True) and temperatures:
        logging.info("\n[Step 8] Temperature sweep (low_height)...")

        # Build env + decoder if Step 7 was skipped
        try:
            env_cfg_ml  # noqa: F841 — check if already defined
        except NameError:
            _, cfg_dict, env_cfg_ml = config_utils.prepare_config(cfg)
            reference_clips = ReferenceClips(
                data_path=vq_cfg.env_config.reference_data_path,
                n_frames_per_clip=cfg.data.get("clip_length", 500),
                keep_clips_idx=None,
            )
            decode_step, action_size = make_decoder_only_step_fn(vq_cfg, policy_params)
            jit_decode = jax.jit(decode_step)
            env_suffix = "-rodent"
            camera_name = f"{cfg.render.camera}{env_suffix}"

        # Use low_height clip
        pose_clips = select_starting_clips(reference_clips)
        low_clip_idx = pose_clips["low_height"]
        sub_clips = subset_clips(reference_clips, low_clip_idx)
        env = imitation.Imitation(config=env_cfg_ml, clips=sub_clips)
        jit_reset = jax.jit(env.reset)
        jit_step = jax.jit(env.step)

        temp_dir = output_dir / "temperature_sweep"
        temp_dir.mkdir(parents=True, exist_ok=True)

        temp_video_paths: list[str] = []
        temp_video_labels: list[str] = []
        max_steps = cfg.free_loop.max_steps

        for temp in temperatures:
            logging.info(f"\n  Temperature T={temp}...")

            num_code_steps = max_steps // H + 1
            temp_seed = cfg.free_loop.seed + int(temp * 1000)
            sampled_hmm_states, sampled_codes = hmm_sample_with_temperature(
                best_params, num_code_steps, temperature=temp, seed=temp_seed
            )
            logging.info(
                f"    Unique codes: {len(np.unique(sampled_codes))}, "
                f"unique states: {len(np.unique(sampled_hmm_states))}"
            )

            # Run rollout
            rng = jax.random.PRNGKey(temp_seed)
            rng, reset_rng = jax.random.split(rng)
            state = jit_reset(reset_rng)

            code_indices: list[int] = []
            hmm_state_indices: list[int] = []
            states_for_render: list[Any] = []
            rewards: list[float] = []

            for t in range(max_steps):
                code_t = int(sampled_codes[t // H])
                hmm_state_t = int(sampled_hmm_states[t // H])
                code_indices.append(code_t)
                hmm_state_indices.append(hmm_state_t)

                action = jit_decode(code_t, state.obs)
                states_for_render.append(state)

                next_state = jit_step(state, action)
                rewards.append(float(next_state.reward))

                if jnp.any(jnp.isnan(next_state.reward)):
                    logging.info(f"    NaN at step {t}, stopping.")
                    break
                state = next_state

            actual_steps = len(code_indices)
            mean_reward = float(np.mean(rewards)) if rewards else 0.0
            logging.info(
                f"    T={temp}: {actual_steps} steps, mean_r={mean_reward:.2f}"
            )

            # Render
            video_path = temp_dir / f"temp_{temp:.1f}.mp4"
            render_hmm_video(
                env=env,
                rollout_states=states_for_render,
                output_path=video_path,
                camera=camera_name,
                width=cfg.render.width,
                height=cfg.render.height,
                fps=cfg.render.fps,
                d0_indices=np.array(code_indices),
                hmm_state_indices=np.array(hmm_state_indices),
                num_codes=num_codes,
                num_hmm_states=best_K,
                rewards=np.array(rewards),
            )

            temp_video_paths.append(str(video_path))
            temp_video_labels.append(
                f"T={temp} | {actual_steps} steps | "
                f"mean_r={mean_reward:.1f} | "
                f"unique_codes={len(np.unique(code_indices))}"
            )

        # Build slider HTML for temperature sweep
        if temp_video_paths:
            html = build_slider_html(
                temp_video_paths,
                temp_video_labels,
                "HMM Temperature Sweep",
                media_type="video",
            )
            html_path = temp_dir / "viewer.html"
            with open(html_path, "w") as f:
                f.write(html)
            logging.info(f"  Temperature sweep HTML saved to {html_path}")

            if wandb_enabled:
                wandb_items["temperature_sweep/viewer"] = wandb.Html(html)

    # ------------------------------------------------------------------
    # Step 9: Emission gallery — per-state code visualization
    #         (one gallery per starting pose: low_height, high_height)
    # ------------------------------------------------------------------
    top_n = cfg.free_loop.get("emission_gallery_top_n", 5)
    if cfg.render.get("enabled", True) and top_n > 0:
        logging.info(f"\n[Step 9] Emission gallery (top-{top_n} codes per state)...")

        # Build env + decoder if previous render steps were skipped
        try:
            env_cfg_ml  # noqa: F841
        except NameError:
            _, cfg_dict, env_cfg_ml = config_utils.prepare_config(cfg)
            reference_clips = ReferenceClips(
                data_path=vq_cfg.env_config.reference_data_path,
                n_frames_per_clip=cfg.data.get("clip_length", 500),
                keep_clips_idx=None,
            )
            decode_step, action_size = make_decoder_only_step_fn(vq_cfg, policy_params)
            jit_decode = jax.jit(decode_step)
            env_suffix = "-rodent"
            camera_name = f"{cfg.render.camera}{env_suffix}"

        pose_clips = select_starting_clips(reference_clips)

        # Extract emission probabilities from best HMM
        log_B = best_params["log_B"]  # [K, C]
        emission_probs = np.exp(log_B - log_B.max(axis=1, keepdims=True))
        emission_probs /= emission_probs.sum(axis=1, keepdims=True)

        gallery_steps = cfg.free_loop.get("emission_gallery_steps", H * 20)

        for pose_name, clip_idx in pose_clips.items():
            logging.info(f"\n  Emission gallery — pose: {pose_name}")

            sub_clips = subset_clips(reference_clips, clip_idx)
            env = imitation.Imitation(config=env_cfg_ml, clips=sub_clips)
            jit_reset = jax.jit(env.reset)
            jit_step = jax.jit(env.step)

            gallery_dir = output_dir / "emission_gallery" / pose_name
            gallery_dir.mkdir(parents=True, exist_ok=True)

            per_state_video_paths: list[list[str]] = []
            per_state_code_labels: list[list[str]] = []
            bar_chart_paths: list[str] = []
            state_labels: list[str] = []

            for k in range(best_K):
                top_codes = np.argsort(emission_probs[k])[::-1][:top_n]
                top_probs = emission_probs[k][top_codes]

                state_vids: list[str] = []
                state_code_lbls: list[str] = []

                for rank, (code_idx, prob) in enumerate(zip(top_codes, top_probs)):
                    code_idx = int(code_idx)
                    logging.info(
                        f"    State {k}, rank {rank}: "
                        f"code {code_idx} (p={prob:.3f})"
                    )

                    # Run rollout holding this code constant
                    rng = jax.random.PRNGKey(cfg.free_loop.seed + k * 100 + rank)
                    rng, reset_rng = jax.random.split(rng)
                    state = jit_reset(reset_rng)

                    states_for_render: list[Any] = []
                    for t in range(gallery_steps):
                        action = jit_decode(code_idx, state.obs)
                        states_for_render.append(state)
                        next_state = jit_step(state, action)
                        if jnp.any(jnp.isnan(next_state.reward)):
                            break
                        state = next_state

                    # Render video
                    video_path = gallery_dir / f"state_{k}_code_{code_idx}.mp4"
                    render_rollout_to_video(
                        env=env,
                        rollout_states=states_for_render,
                        output_path=video_path,
                        camera=camera_name,
                        width=cfg.render.width,
                        height=cfg.render.height,
                        fps=cfg.render.fps,
                        indices=np.full(len(states_for_render), code_idx, dtype=int),
                        num_codes=num_codes,
                    )

                    state_vids.append(str(video_path))
                    state_code_lbls.append(f"Code {code_idx} (p={prob:.2f})")

                per_state_video_paths.append(state_vids)
                per_state_code_labels.append(state_code_lbls)

                # Render emission bar chart for this state
                bar_path = gallery_dir / f"state_{k}_emission_bar.png"
                plot_emission_bar(
                    emission_probs[k],
                    state_idx=k,
                    top_codes=top_codes,
                    top_probs=top_probs,
                    num_codes=num_codes,
                    output_path=bar_path,
                )
                bar_chart_paths.append(str(bar_path))
                state_labels.append(f"Hidden State {k}")

            # Build gallery HTML — one slider position per state
            if per_state_video_paths:
                title = f"HMM Emission Gallery ({pose_name})"
                html = build_emission_gallery_html(
                    per_state_video_paths,
                    per_state_code_labels,
                    bar_chart_paths,
                    state_labels,
                    title,
                )
                html_path = gallery_dir / "viewer.html"
                with open(html_path, "w") as f:
                    f.write(html)
                logging.info(f"    {pose_name} gallery HTML saved to {html_path}")

                if wandb_enabled:
                    wb_key = f"emission_gallery/{pose_name}/viewer"
                    wandb_items[wb_key] = wandb.Html(html)

    # ------------------------------------------------------------------
    # Step 10: Transition Matrix Analysis
    # ------------------------------------------------------------------
    empirical_trans_matrix = None
    if cfg.get("controls", {}).get("transition_analysis", {}).get("enabled", False):
        logging.info("\n[Step 10] Transition matrix analysis...")

        sparsity_thresh = cfg.controls.transition_analysis.get(
            "sparsity_threshold", 0.01
        )

        trans_with_self = compute_empirical_transition_matrix(
            all_codes_subsampled, num_codes, exclude_self=False
        )
        trans_no_self = compute_empirical_transition_matrix(
            all_codes_subsampled, num_codes, exclude_self=True
        )
        empirical_trans_matrix = trans_with_self

        metrics_with = compute_transition_metrics(trans_with_self, sparsity_thresh)
        metrics_no = compute_transition_metrics(trans_no_self, sparsity_thresh)

        logging.info(
            f"  With self-transitions: mean entropy={metrics_with['mean_entropy']:.3f} "
            f"bits, reduction={metrics_with['entropy_reduction']:.3f} bits, "
            f"sparsity={metrics_with['sparsity_ratio']:.1%}"
        )
        logging.info(
            f"  Without self-transitions: mean entropy={metrics_no['mean_entropy']:.3f} "
            f"bits, reduction={metrics_no['entropy_reduction']:.3f} bits, "
            f"sparsity={metrics_no['sparsity_ratio']:.1%}"
        )

        trans_dir = output_dir / "transition_analysis"
        trans_plots = plot_transition_analysis(
            trans_with_self,
            trans_no_self,
            metrics_with,
            metrics_no,
            num_codes,
            trans_dir,
        )

        # Save metrics as JSON
        metrics_json = {
            "with_self": {
                k: v for k, v in metrics_with.items() if k != "per_row_entropy"
            },
            "no_self": {k: v for k, v in metrics_no.items() if k != "per_row_entropy"},
        }
        with open(trans_dir / "metrics.json", "w") as f:
            json.dump(metrics_json, f, indent=2)

        # Save transition matrices
        np.savez(
            trans_dir / "transition_matrices.npz",
            with_self=trans_with_self,
            no_self=trans_no_self,
        )

        if wandb_enabled:
            import wandb

            for name, path in trans_plots.items():
                wandb_items[f"transition_analysis/{name}"] = wandb.Image(str(path))
            # Scalar metrics → summary (not time-series)
            for key, val in metrics_json["with_self"].items():
                wandb.run.summary[f"transition_analysis/with_self/{key}"] = val
            for key, val in metrics_json["no_self"].items():
                wandb.run.summary[f"transition_analysis/no_self/{key}"] = val

    # ------------------------------------------------------------------
    # Step 11: Random Code Null Control
    # ------------------------------------------------------------------
    if cfg.get("controls", {}).get("random_null", {}).get("enabled", False):
        logging.info("\n[Step 11] Random code null control...")

        rn_cfg = cfg.controls.random_null
        rn_num_seeds = rn_cfg.get("num_seeds", 5)
        rn_max_steps = rn_cfg.get("max_steps", cfg.free_loop.max_steps)

        # Build env + decoder if not already done
        try:
            env_cfg_ml  # noqa: F841
        except NameError:
            _, cfg_dict, env_cfg_ml = config_utils.prepare_config(cfg)
            reference_clips = ReferenceClips(
                data_path=vq_cfg.env_config.reference_data_path,
                n_frames_per_clip=cfg.data.get("clip_length", 500),
                keep_clips_idx=None,
            )
            decode_step, action_size = make_decoder_only_step_fn(vq_cfg, policy_params)
            jit_decode = jax.jit(decode_step)
            env_suffix = "-rodent"
            camera_name = f"{cfg.render.camera}{env_suffix}"

        pose_clips = select_starting_clips(reference_clips)

        null_dir = output_dir / "random_null"
        null_dir.mkdir(parents=True, exist_ok=True)

        num_code_steps = rn_max_steps // H + 1
        null_base_seed = cfg.free_loop.seed + 10000

        # Extract real joint data from loaded rollouts
        joint_names = list(cfg.walker_config.joint_names)
        real_qpos_joints = np.concatenate([r.qpos[:, 7:39] for r in results])  # [N, 32]
        real_qvel_joints = np.concatenate([r.qvel[:, 6:38] for r in results])  # [N, 32]
        logging.info(
            f"  Real data: {real_qpos_joints.shape[0]} frames, "
            f"{len(joint_names)} joints"
        )

        for pose_name, clip_idx in pose_clips.items():
            logging.info(f"\n  Pose: {pose_name} (clip {clip_idx})")

            sub_clips = subset_clips(reference_clips, clip_idx)
            env = imitation.Imitation(config=env_cfg_ml, clips=sub_clips)
            jit_reset = jax.jit(env.reset)
            jit_step = jax.jit(env.step)

            hmm_results_list: list[dict] = []
            marginal_results_list: list[dict] = []
            uniform_results_list: list[dict] = []

            all_null_videos: list[str] = []
            all_null_labels: list[str] = []

            for seed_i in range(rn_num_seeds):
                seed_offset = null_base_seed + seed_i

                # HMM condition
                sampled_hmm_states, sampled_codes = hmm_sample(
                    best_params, num_code_steps, seed=seed_offset
                )
                hmm_result = run_code_sequence_rollout(
                    env,
                    jit_decode,
                    jit_reset,
                    jit_step,
                    sampled_codes,
                    H,
                    rn_max_steps,
                    seed=seed_offset,
                    collect_states=(seed_i == 0),
                    collect_qpos=True,
                )
                hmm_results_list.append(hmm_result)

                # Marginal random condition
                marginal_codes = generate_marginal_random_codes(
                    all_codes_subsampled, num_code_steps, num_codes, seed_offset + 1000
                )
                marginal_result = run_code_sequence_rollout(
                    env,
                    jit_decode,
                    jit_reset,
                    jit_step,
                    marginal_codes,
                    H,
                    rn_max_steps,
                    seed=seed_offset,
                    collect_states=(seed_i == 0),
                    collect_qpos=True,
                )
                marginal_results_list.append(marginal_result)

                # Uniform random condition
                uniform_codes = generate_uniform_random_codes(
                    num_code_steps, num_codes, seed_offset + 2000
                )
                uniform_result = run_code_sequence_rollout(
                    env,
                    jit_decode,
                    jit_reset,
                    jit_step,
                    uniform_codes,
                    H,
                    rn_max_steps,
                    seed=seed_offset,
                    collect_states=(seed_i == 0),
                    collect_qpos=True,
                )
                uniform_results_list.append(uniform_result)

                logging.info(
                    f"    Seed {seed_i}: HMM={hmm_result['survival_steps']} steps "
                    f"(R={hmm_result['mean_reward']:.1f}), "
                    f"Marginal={marginal_result['survival_steps']} "
                    f"(R={marginal_result['mean_reward']:.1f}), "
                    f"Uniform={uniform_result['survival_steps']} "
                    f"(R={uniform_result['mean_reward']:.1f})"
                )

                # Render first seed only
                if seed_i == 0 and cfg.render.get("enabled", True):
                    for cond_name, result, cond_codes in [
                        ("hmm", hmm_result, sampled_codes),
                        ("marginal", marginal_result, marginal_codes),
                        ("uniform", uniform_result, uniform_codes),
                    ]:
                        if result["states_for_render"]:
                            video_path = (
                                null_dir / f"{pose_name}_{cond_name}_seed{seed_i}.mp4"
                            )
                            # Use zero-valued hmm_state_indices for non-HMM
                            d0_arr = result["code_indices"]
                            hmm_arr = np.zeros_like(d0_arr)
                            render_hmm_video(
                                env=env,
                                rollout_states=result["states_for_render"],
                                output_path=video_path,
                                camera=camera_name,
                                width=cfg.render.width,
                                height=cfg.render.height,
                                fps=cfg.render.fps,
                                d0_indices=d0_arr,
                                hmm_state_indices=hmm_arr,
                                num_codes=num_codes,
                                num_hmm_states=1,
                                rewards=result["rewards"],
                            )
                            all_null_videos.append(str(video_path))
                            all_null_labels.append(
                                f"{pose_name} | {cond_name} | "
                                f"{result['survival_steps']} steps | "
                                f"R={result['mean_reward']:.1f}"
                            )

            # Compute per-condition Wasserstein distances
            w1_results = {}
            for cond_name, results_list in [
                ("hmm", hmm_results_list),
                ("marginal", marginal_results_list),
                ("uniform", uniform_results_list),
            ]:
                gen_qpos = np.concatenate(
                    [r["qpos_arr"][:, 7:39] for r in results_list]
                )
                gen_qvel = np.concatenate(
                    [r["qvel_arr"][:, 6:38] for r in results_list]
                )
                w1 = compute_wasserstein_distances(
                    real_qpos_joints, real_qvel_joints, gen_qpos, gen_qvel
                )
                w1_results[cond_name] = w1
                logging.info(
                    f"    {cond_name} W1: qpos={w1['mean_qpos_w1']:.4f}, "
                    f"qvel={w1['mean_qvel_w1']:.4f}"
                )

            # Plot comparison
            pose_null_dir = null_dir / pose_name
            null_plots = plot_random_null_comparison(
                w1_results["hmm"],
                w1_results["marginal"],
                w1_results["uniform"],
                hmm_results_list,
                marginal_results_list,
                uniform_results_list,
                joint_names,
                pose_null_dir,
            )

            if wandb_enabled:
                import wandb

                for name, path in null_plots.items():
                    wb_key = f"random_null/{pose_name}/{name}"
                    wandb_items[wb_key] = wandb.Image(str(path))

                # Log W1 scalars to wandb summary
                for cond_name, w1 in w1_results.items():
                    wandb.run.summary[
                        f"random_null/{pose_name}/{cond_name}/mean_qpos_w1"
                    ] = w1["mean_qpos_w1"]
                    wandb.run.summary[
                        f"random_null/{pose_name}/{cond_name}/mean_qvel_w1"
                    ] = w1["mean_qvel_w1"]

            # Build slider HTML for videos
            if all_null_videos:
                html = build_slider_html(
                    all_null_videos,
                    all_null_labels,
                    f"Random Null Control ({pose_name})",
                    media_type="video",
                )
                html_path = null_dir / f"viewer_{pose_name}.html"
                with open(html_path, "w") as f:
                    f.write(html)

                if wandb_enabled:
                    wb_key = f"random_null/{pose_name}/viewer"
                    wandb_items[wb_key] = wandb.Html(html)

        # Summary stats
        logging.info("\n  Random null summary:")
        for cond_name, results_list in [
            ("HMM", hmm_results_list),
            ("Marginal", marginal_results_list),
            ("Uniform", uniform_results_list),
        ]:
            survivals = [r["survival_steps"] for r in results_list]
            mean_rewards = [r["mean_reward"] for r in results_list]
            logging.info(
                f"    {cond_name}: survival={np.mean(survivals):.0f}+/-"
                f"{np.std(survivals):.0f}, "
                f"mean_reward={np.mean(mean_rewards):.1f}+/-"
                f"{np.std(mean_rewards):.1f}"
            )

    # ------------------------------------------------------------------
    # Step 12: Final WandB logging
    # ------------------------------------------------------------------
    if wandb_enabled:
        import wandb

        if wandb_items and wandb.run is not None:
            wandb.log(wandb_items)
        wandb.finish()
        logging.info("WandB logging complete.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("HMM Prior pipeline complete!")
    print(f"  Best K: {best_K} (test LL: {test_lls[best_K]:.2f})")
    print(f"  Outputs: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
