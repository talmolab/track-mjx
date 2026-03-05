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
from brax.training import distribution
from brax.training.acme import running_statistics
from omegaconf import DictConfig
from vnl_playground.tasks.rodent import imitation
from vnl_playground.tasks.rodent.reference_clips import ReferenceClips

from track_mjx.agent.observation_utils import flatten_obs_dict
from track_mjx.config import utils as config_utils

from analysis.checkpoint_utils import (
    create_standalone_decoder,
    get_codebook,
    get_decoder_params,
    load_vq_checkpoint,
)
from analysis.code_analysis import load_rollouts_from_h5
from analysis.rendering import render_rollout_to_video
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
) -> tuple[dict[str, np.ndarray], list[float]]:
    """Fit a discrete HMM using Expectation-Maximization.

    Args:
        observations_list: List of integer observation sequences, each shape [T_i].
        num_states: Number of hidden states K.
        num_classes: Number of emission classes C.
        num_iters: Number of EM iterations.
        seed: Random seed for initialization.

    Returns:
        Tuple of (params, log_likelihoods) where params has keys
        "log_pi", "log_A", "log_B" and log_likelihoods is the EM curve.
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

    log_likelihoods: list[float] = []

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
        log_likelihoods.append(avg_ll)

        if iteration % 50 == 0:
            logging.info(f"    EM iter {iteration}: avg log-likelihood = {avg_ll:.2f}")

        # M-step
        log_pi, log_A, log_B = hmm_m_step(
            gamma_list, xi_list, observations_list, C
        )

    return {"log_pi": log_pi, "log_A": log_A, "log_B": log_B}, log_likelihoods


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
    _, ll = hmm_forward(params["log_pi"], params["log_A"], params["log_B"], observations)
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
    logging.info(f"  Decoder-only mode: latent_dim={latent_dim}, "
                 f"use_continuous={use_continuous}, continuous_dim={continuous_dim}")

    # Assert D1 is not used
    assert codebook_0.shape == (cfg.network_config.num_codes, latent_dim), (
        f"Codebook shape mismatch: {codebook_0.shape}"
    )

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

    for K in cfg.hmm.num_states_sweep:
        logging.info(f"\n  Fitting K={K}...")
        params, em_curve = fit_hmm_em(
            train_emissions,
            num_states=K,
            num_classes=num_codes,
            num_iters=cfg.hmm.num_em_iters,
            seed=cfg.hmm.seed + K,
        )
        all_em_curves[K] = em_curve
        all_params[K] = params

        # Evaluate on test set
        test_ll = np.mean([
            hmm_marginal_log_prob(params, seq) for seq in test_emissions
        ])
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
            {"test_log_likelihoods": {str(k): v for k, v in test_lls.items()},
             "best_K": best_K},
            f, indent=2,
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
        wandb_items["hmm_prior/em_curves"] = wandb.Image(em_curve_path)
        wandb_items["hmm_prior/model_selection"] = wandb.Image(selection_path)
        wandb_items["hmm_prior/transition_matrix"] = wandb.Image(trans_path)
        wandb_items["hmm_prior/emission_matrix"] = wandb.Image(emit_path)
        wandb_items["hmm_prior/stationary_distribution"] = wandb.Image(stat_path)

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
        (_, cfg_dict, env_cfg_ml) = config_utils.prepare_config(cfg)
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

            # Sample HMM code sequence
            num_code_steps = max_steps // H + 1
            _, sampled_codes = hmm_sample(
                best_params, num_code_steps, seed=free_seed
            )
            logging.info(
                f"  Sampled {num_code_steps} codes from HMM "
                f"(unique: {len(np.unique(sampled_codes))})"
            )

            # Run free-loop rollout
            rng = jax.random.PRNGKey(free_seed)
            rng, reset_rng = jax.random.split(rng)
            state = jit_reset(reset_rng)

            code_indices: list[int] = []
            states_for_render: list[Any] = []
            rewards: list[float] = []

            for t in range(max_steps):
                code_t = int(sampled_codes[t // H])
                code_indices.append(code_t)

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
                f"  Ran {actual_steps} steps, "
                f"mean reward={mean_reward:.2f}"
            )

            # Render video
            d0_indices = np.array(code_indices)
            d1_indices = np.zeros(len(code_indices), dtype=int)

            video_path = video_dir / f"free_loop_{pose_name}.mp4"
            render_rollout_to_video(
                env=env,
                rollout_states=states_for_render,
                output_path=video_path,
                camera=camera_name,
                width=cfg.render.width,
                height=cfg.render.height,
                fps=cfg.render.fps,
                indices=d0_indices,
                num_codes=num_codes,
                rewards=np.array(rewards),
                indices_per_depth=[d0_indices, d1_indices],
                d0_label="HMM",
            )

            all_video_paths.append(str(video_path))
            all_video_labels.append(
                f"{pose_name} | {actual_steps} steps | "
                f"mean_r={mean_reward:.1f}"
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
                wandb_items["hmm_prior/free_loop/viewer"] = wandb.Html(html)

    # ------------------------------------------------------------------
    # Step 8: Final WandB logging
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
