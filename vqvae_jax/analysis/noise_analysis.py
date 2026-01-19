"""VQ-VAE Prior Noise Analysis.

Analyzes the effect of adding Gaussian noise to the continuous latent (z_p)
before quantization. Tests how noise causes code switching and behavioral
diversity in the trained prior model.

Usage:
    cd vqvae_jax
    python -m analysis.noise_analysis

    # Override config values:
    python -m analysis.noise_analysis \
        noise_config.noise_levels=[0.0,0.5,1.0,2.0] \
        rollout_config.num_rollouts_per_noise=5

    # Specify output directory:
    python -m analysis.noise_analysis \
        output_config.output_dir=./my_analysis_results
"""

import os

# Must set rendering backend before importing MuJoCo
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

# Add paths for package imports
ANALYSIS_DIR = Path(__file__).parent
VQVAE_DIR = ANALYSIS_DIR.parent
REPO_ROOT = VQVAE_DIR.parent
sys.path.insert(0, str(VQVAE_DIR))
sys.path.insert(0, str(REPO_ROOT))

import hydra
import imageio
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import orbax.checkpoint as ocp
from absl import logging
from brax.training import distribution
from brax.training.acme import running_statistics
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw, ImageFont
from vnl_playground.tasks.rodent import imitation
from vnl_playground.tasks.rodent.reference_clips import ReferenceClips

# Local imports
from track_mjx.agent.observation_utils import flatten_obs_dict
from track_mjx.config import utils as config_utils

from distillation.vq_prior_networks import VQPrior
from vq_intention_network import Decoder
from analysis.checkpoint_utils import load_vq_checkpoint, get_codebook, get_decoder_params
from analysis.rendering import get_nature_colormap, add_code_transition_bar, add_text_overlay


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class NoisyRolloutResult:
    """Result from a single noisy rollout."""

    # Basic info
    noise_std: float
    seed: int
    survival_steps: int
    terminated: bool

    # Trajectory data
    states: list
    actions: np.ndarray
    rewards: np.ndarray

    # Latent data
    z_p: np.ndarray           # Prior outputs [T, latent_dim]
    z_p_noisy: np.ndarray     # After noise [T, latent_dim]
    z_q: np.ndarray           # Quantized [T, latent_dim]
    indices: np.ndarray       # Code indices [T]

    # Displacement metrics
    root_trajectory: np.ndarray     # [T, 3] xyz positions
    total_displacement: float
    total_distance: float
    max_displacement: float

    # Transition metrics
    num_transitions: int
    unique_codes_used: int


# =============================================================================
# CHECKPOINT LOADING
# =============================================================================


def load_prior_checkpoint(
    checkpoint_path: str,
    step: int | None = None,
) -> dict[str, Any]:
    """Load prior checkpoint.

    The prior checkpoint contains both the trained prior network AND the frozen
    VQ-VAE components (codebook, decoder, encoder).

    Args:
        checkpoint_path: Path to prior checkpoint directory.
        step: Specific step to load. If None, loads latest.

    Returns:
        Dictionary with:
        - cfg: Config
        - prior_params: (normalizer_params, prior_weights)
        - frozen_vqvae: dict with codebook, decoder_params, encoder_params
        - step: Loaded step
    """
    mgr_options = ocp.CheckpointManagerOptions(create=False, step_prefix="VQPriorDistill")
    with ocp.CheckpointManager(checkpoint_path, options=mgr_options) as ckpt_mgr:
        if step is None:
            step = ckpt_mgr.latest_step()

        logging.info(f"Loading prior from {checkpoint_path} at step {step}")

        # Load config
        config_result = ckpt_mgr.restore(
            step,
            args=ocp.args.Composite(config=ocp.args.JsonRestore()),
        )
        cfg = OmegaConf.create(config_result["config"])

        # Load policy
        # Structure: {'frozen_vqvae': {...}, 'prior': [normalizer_dict, prior_params]}
        policy_result = ckpt_mgr.restore(
            step,
            args=ocp.args.Composite(policy=ocp.args.StandardRestore(None)),
        )
        policy = policy_result["policy"]

        # Extract frozen VQ-VAE components
        frozen_vqvae = policy["frozen_vqvae"]
        codebook = frozen_vqvae["codebook"]
        decoder_params = frozen_vqvae["decoder_params"]

        logging.info(f"  Codebook shape: {codebook.shape}")

        # Extract prior params: [normalizer_dict, prior_params]
        prior_list = policy["prior"]
        normalizer_dict = prior_list[0]
        prior_params = prior_list[1]

        # Convert normalizer dict to proper structure
        from track_mjx.agent.observation_utils import DictRunningStatisticsState

        def _dict_to_running_statistics_state(d: dict) -> running_statistics.RunningStatisticsState:
            state = running_statistics.RunningStatisticsState(
                count=d["count"],
                mean=d["mean"],
                summed_variance=d["summed_variance"],
                std=d["std"],
            )
            if "std_eps" in d:
                state = state.replace(std_eps=d["std_eps"])
            if "mode" in d:
                state = state.replace(mode=d["mode"])
            return state

        if "proprioception" in normalizer_dict:
            normalizer_params = DictRunningStatisticsState(
                imitation_target=_dict_to_running_statistics_state(
                    normalizer_dict["imitation_target"]
                ),
                proprioception=_dict_to_running_statistics_state(
                    normalizer_dict["proprioception"]
                ),
            )
        else:
            normalizer_params = _dict_to_running_statistics_state(normalizer_dict)

        return {
            "cfg": cfg,
            "prior_params": (normalizer_params, prior_params),
            "frozen_vqvae": {
                "codebook": codebook,
                "decoder_params": decoder_params,
            },
            "step": step,
        }


def load_frozen_vqvae(
    checkpoint_path: str,
    step: int | None = None,
) -> dict[str, Any]:
    """Load frozen VQ-VAE components.

    Args:
        checkpoint_path: Path to VQ-VAE checkpoint directory.
        step: Specific step to load. If None, loads latest.

    Returns:
        Dictionary with decoder params, codebook, and config.
    """
    checkpoint = load_vq_checkpoint(checkpoint_path, step=step)
    cfg = checkpoint["cfg"]
    policy_params = checkpoint["policy"]

    decoder_params = get_decoder_params(policy_params)
    codebook = get_codebook(policy_params)

    logging.info(f"Loaded VQ-VAE from {checkpoint_path} at step {checkpoint['step']}")
    logging.info(f"  Codebook shape: {codebook.shape}")
    logging.info(f"  Num codes: {codebook.shape[0]}, Latent dim: {codebook.shape[1]}")

    return {
        "decoder_params": decoder_params,
        "codebook": codebook,
        "cfg": cfg,
        "step": checkpoint["step"],
    }


# =============================================================================
# CORE ROLLOUT FUNCTION
# =============================================================================


def extract_root_xyz(state: Any) -> np.ndarray:
    """Extract root (torso) xyz position from environment state.

    For rodent model, qpos layout is typically:
    [root_x, root_y, root_z, quat_w, quat_x, quat_y, quat_z, ...joints]
    """
    if hasattr(state, "data") and hasattr(state.data, "qpos"):
        qpos = state.data.qpos
    elif hasattr(state, "pipeline_state") and hasattr(state.pipeline_state, "q"):
        qpos = state.pipeline_state.q
    elif hasattr(state, "qpos"):
        qpos = state.qpos
    else:
        raise ValueError(f"Cannot find qpos in state: {type(state)}")

    return np.array(qpos[:3])


def compute_displacement_metrics(trajectory: np.ndarray) -> dict:
    """Compute displacement metrics from xyz trajectory.

    Args:
        trajectory: [T, 3] array of xyz positions.

    Returns:
        Dictionary with displacement metrics.
    """
    if len(trajectory) < 2:
        return {
            "total_displacement": 0.0,
            "total_distance": 0.0,
            "max_displacement": 0.0,
        }

    # Total displacement (euclidean start to end)
    total_displacement = float(np.linalg.norm(trajectory[-1] - trajectory[0]))

    # Total distance traveled (sum of step distances)
    diffs = np.diff(trajectory, axis=0)
    total_distance = float(np.sum(np.linalg.norm(diffs, axis=1)))

    # Maximum displacement from start
    displacements = np.linalg.norm(trajectory - trajectory[0], axis=1)
    max_displacement = float(np.max(displacements))

    return {
        "total_displacement": total_displacement,
        "total_distance": total_distance,
        "max_displacement": max_displacement,
    }


def run_noisy_rollout(
    env: Any,
    prior_params: tuple[Any, Any],
    decoder_params: dict[str, Any],
    codebook: jnp.ndarray,
    prior_module: VQPrior,
    decoder_module: Decoder,
    parametric_action_distribution: Any,
    max_steps: int,
    seed: int,
    noise_std: float,
    quantize_prior: bool = True,
    deterministic: bool = True,
) -> NoisyRolloutResult:
    """Run a single rollout with Gaussian noise injected into z_p.

    Noise injection happens AFTER prior prediction but BEFORE quantization:
        z_p = prior(proprio)           # Prior predicts continuous latent
        z_p_noisy = z_p + noise        # ADD NOISE HERE
        z_q = quantize(z_p_noisy)      # Quantize noisy latent
        action = decode(z_q, proprio)  # Decode to action

    Args:
        env: MuJoCo environment.
        prior_params: (normalizer_params, prior_weights).
        decoder_params: Frozen decoder weights.
        codebook: Frozen VQ codebook [num_codes, latent_dim].
        prior_module: VQPrior Flax module.
        decoder_module: Decoder Flax module.
        parametric_action_distribution: Action distribution.
        max_steps: Maximum rollout steps.
        seed: Random seed.
        noise_std: Standard deviation of Gaussian noise to add.
        quantize_prior: Whether to quantize (False = use z_p_noisy directly).
        deterministic: Whether to use deterministic actions.

    Returns:
        NoisyRolloutResult with full trajectory and metrics.
    """
    normalizer_params, prior_weights = prior_params

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(seed)
    reset_rng, rng = jax.random.split(rng)

    state = jit_reset(reset_rng)

    # Storage
    states = [state]
    actions = []
    rewards = []
    z_ps = []
    z_p_noisys = []
    z_qs = []
    indices_list = []
    root_positions = []

    # Extract initial root position
    root_positions.append(extract_root_xyz(state))

    step_count = 0
    for step_count in range(max_steps):
        # Extract proprioceptive observation
        obs = state.obs
        flat_obs = flatten_obs_dict(obs)
        proprio = flat_obs["proprioception"]

        # Normalize proprio
        proprio_normalizer = (
            normalizer_params.proprioception
            if hasattr(normalizer_params, "proprioception")
            else normalizer_params
        )
        proprio_normalized = running_statistics.normalize(proprio, proprio_normalizer)

        # Prior predicts z_p
        z_p = prior_module.apply(prior_weights, proprio_normalized)
        z_ps.append(np.array(z_p))

        # ========== NOISE INJECTION ==========
        rng, noise_key = jax.random.split(rng)
        noise = noise_std * jax.random.normal(noise_key, z_p.shape)
        z_p_noisy = z_p + noise
        z_p_noisys.append(np.array(z_p_noisy))

        # Quantize (using noisy z_p)
        if quantize_prior:
            distances = jnp.sum((z_p_noisy[None, :] - codebook) ** 2, axis=-1)
            idx = jnp.argmin(distances)
            z_q = codebook[idx]
            indices_list.append(int(idx))
        else:
            z_q = z_p_noisy
            indices_list.append(-1)
        z_qs.append(np.array(z_q))

        # Decode to action
        decoder_input = jnp.concatenate([z_q, proprio_normalized], axis=-1)
        action_logits, _ = decoder_module.apply(
            {"params": decoder_params}, decoder_input
        )

        # Sample action
        if deterministic:
            action = parametric_action_distribution.mode(action_logits)
        else:
            rng, sample_key = jax.random.split(rng)
            raw_action = parametric_action_distribution.sample_no_postprocessing(
                action_logits, sample_key
            )
            action = parametric_action_distribution.postprocess(raw_action)

        # Step environment
        next_state = jit_step(state, action)

        states.append(next_state)
        actions.append(np.array(action))
        rewards.append(float(next_state.reward))
        root_positions.append(extract_root_xyz(next_state))

        if next_state.done:
            break

        state = next_state

    # Compute displacement metrics
    root_trajectory = np.array(root_positions)
    displacement_metrics = compute_displacement_metrics(root_trajectory)

    # Compute transition metrics
    indices_arr = np.array(indices_list)
    num_transitions = int(np.sum(np.diff(indices_arr) != 0)) if len(indices_arr) > 1 else 0
    unique_codes = len(np.unique(indices_arr[indices_arr >= 0]))

    return NoisyRolloutResult(
        noise_std=noise_std,
        seed=seed,
        survival_steps=len(states) - 1,
        terminated=step_count < max_steps - 1,
        states=states,
        actions=np.stack(actions) if actions else np.zeros((0, env.action_size)),
        rewards=np.array(rewards) if rewards else np.zeros((0,)),
        z_p=np.stack(z_ps) if z_ps else np.zeros((0, codebook.shape[-1])),
        z_p_noisy=np.stack(z_p_noisys) if z_p_noisys else np.zeros((0, codebook.shape[-1])),
        z_q=np.stack(z_qs) if z_qs else np.zeros((0, codebook.shape[-1])),
        indices=indices_arr,
        root_trajectory=root_trajectory,
        total_displacement=displacement_metrics["total_displacement"],
        total_distance=displacement_metrics["total_distance"],
        max_displacement=displacement_metrics["max_displacement"],
        num_transitions=num_transitions,
        unique_codes_used=unique_codes,
    )


# =============================================================================
# TRANSITION MATRIX ANALYSIS
# =============================================================================


def compute_transition_matrices(
    results: dict[float, list[NoisyRolloutResult]],
    num_codes: int,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Compute transition matrices for each noise level.

    Args:
        results: Noise level -> list of rollout results.
        num_codes: Total number of codes in codebook.

    Returns:
        Noise level -> (transition_counts, transition_probs).
    """
    matrices = {}

    for noise_std, rollouts in results.items():
        # Aggregate all indices from all rollouts at this noise level
        all_indices = []
        for rollout in rollouts:
            if len(rollout.indices) > 0:
                all_indices.extend(list(rollout.indices))

        # Compute transition counts
        trans_counts = np.zeros((num_codes, num_codes), dtype=np.int32)
        for i in range(len(all_indices) - 1):
            from_code = all_indices[i]
            to_code = all_indices[i + 1]
            if from_code >= 0 and to_code >= 0:
                trans_counts[from_code, to_code] += 1

        # Normalize to probabilities
        row_sums = trans_counts.sum(axis=1, keepdims=True)
        trans_probs = np.where(row_sums > 0, trans_counts / row_sums, 0.0)

        matrices[noise_std] = (trans_counts, trans_probs)

    return matrices


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================


def plot_trajectory_comparison(
    results: dict[float, list[NoisyRolloutResult]],
    output_dir: str,
    num_trajectories_per_noise: int = 5,
    figsize: tuple[int, int] = (10, 10),
):
    """Plot 2D bird's-eye view of trajectories, colored by noise level.

    All trajectories start from origin (0, 0) for easy comparison.
    Each noise level gets a distinct color, with multiple rollouts
    shown as semi-transparent lines.
    """
    noise_levels = sorted(results.keys())
    num_levels = len(noise_levels)

    # Create colormap: blue (low noise) -> red (high noise)
    cmap = cm.get_cmap("coolwarm", max(num_levels, 2))
    noise_colors = {
        noise: cmap(i / max(num_levels - 1, 1)) for i, noise in enumerate(noise_levels)
    }

    fig, ax = plt.subplots(figsize=figsize)

    # Track bounds for axis limits
    all_x, all_y = [], []

    for noise_std in noise_levels:
        rollouts = results[noise_std][:num_trajectories_per_noise]
        color = noise_colors[noise_std]

        for i, rollout in enumerate(rollouts):
            traj = rollout.root_trajectory  # [T, 3] -> use x, y

            # Normalize to start at origin
            x = traj[:, 0] - traj[0, 0]
            y = traj[:, 1] - traj[0, 1]

            all_x.extend(x)
            all_y.extend(y)

            # Plot trajectory (first one gets label for legend)
            label = f"noise={noise_std}" if i == 0 else None
            ax.plot(
                x,
                y,
                color=color,
                alpha=0.6,
                linewidth=1.5,
                label=label,
            )

            # Mark end point
            ax.scatter(x[-1], y[-1], color=color, s=30, marker="x", zorder=5)

    # Add origin marker
    ax.scatter(0, 0, color="black", s=100, marker="*", zorder=10, label="Start")

    # Set equal aspect ratio and limits
    ax.set_aspect("equal")
    if all_x and all_y:
        margin = 0.1
        x_range = max(all_x) - min(all_x) if max(all_x) != min(all_x) else 1
        y_range = max(all_y) - min(all_y) if max(all_y) != min(all_y) else 1
        max_range = max(x_range, y_range) * (1 + margin)
        center_x = (max(all_x) + min(all_x)) / 2
        center_y = (max(all_y) + min(all_y)) / 2
        ax.set_xlim(center_x - max_range / 2, center_x + max_range / 2)
        ax.set_ylim(center_y - max_range / 2, center_y + max_range / 2)

    ax.set_xlabel("X Position (m)", fontsize=12)
    ax.set_ylabel("Y Position (m)", fontsize=12)
    ax.set_title(
        "Trajectory Comparison: Bird's-Eye View\n(colored by noise level)", fontsize=14
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/trajectory_comparison.png", dpi=200)
    plt.savefig(f"{output_dir}/trajectory_comparison.pdf")  # Vector format
    plt.close()

    logging.info(f"Saved trajectory plot to {output_dir}/trajectory_comparison.png")


def plot_trajectory_grid(
    results: dict[float, list[NoisyRolloutResult]],
    output_dir: str,
    num_trajectories_per_noise: int = 10,
):
    """Plot trajectory grids: one subplot per noise level."""
    noise_levels = sorted(results.keys())
    n_cols = min(4, len(noise_levels))
    n_rows = (len(noise_levels) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    cmap = cm.get_cmap("coolwarm", max(len(noise_levels), 2))

    # Find global bounds for consistent axis limits
    global_max = 0.01  # Minimum range
    for rollouts in results.values():
        for r in rollouts[:num_trajectories_per_noise]:
            traj = r.root_trajectory
            x = traj[:, 0] - traj[0, 0]
            y = traj[:, 1] - traj[0, 1]
            global_max = max(global_max, np.abs(x).max(), np.abs(y).max())
    global_max *= 1.1  # Add margin

    for idx, noise_std in enumerate(noise_levels):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        rollouts = results[noise_std][:num_trajectories_per_noise]
        color = cmap(idx / max(len(noise_levels) - 1, 1))

        for rollout in rollouts:
            traj = rollout.root_trajectory
            x = traj[:, 0] - traj[0, 0]
            y = traj[:, 1] - traj[0, 1]

            ax.plot(x, y, color=color, alpha=0.5, linewidth=1.0)

        ax.scatter(0, 0, color="black", s=80, marker="*", zorder=10)
        ax.set_xlim(-global_max, global_max)
        ax.set_ylim(-global_max, global_max)
        ax.set_aspect("equal")
        ax.set_title(f"noise_std = {noise_std}", fontsize=11)
        ax.grid(True, alpha=0.3)

        if row == n_rows - 1:
            ax.set_xlabel("X (m)")
        if col == 0:
            ax.set_ylabel("Y (m)")

    # Hide unused subplots
    for idx in range(len(noise_levels), n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    plt.suptitle("Trajectory Spread by Noise Level", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/trajectory_grid.png", dpi=150)
    plt.close()

    logging.info(f"Saved trajectory grid to {output_dir}/trajectory_grid.png")


def plot_displacement_vs_noise(
    results: dict[float, list[NoisyRolloutResult]],
    output_dir: str,
):
    """Plot displacement metrics vs noise level."""
    noise_levels = sorted(results.keys())

    # Extract metrics
    metrics = {
        "total_displacement": [],
        "total_distance": [],
        "max_displacement": [],
    }

    for noise_std in noise_levels:
        for metric_name in metrics:
            values = [getattr(r, metric_name) for r in results[noise_std]]
            metrics[metric_name].append(
                {
                    "mean": np.mean(values),
                    "std": np.std(values),
                }
            )

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (metric_name, data) in zip(axes, metrics.items()):
        means = [d["mean"] for d in data]
        stds = [d["std"] for d in data]

        ax.errorbar(noise_levels, means, yerr=stds, marker="o", capsize=5)
        ax.set_xlabel("Noise std")
        ax.set_ylabel(metric_name.replace("_", " ").title())
        ax.set_title(f"{metric_name.replace('_', ' ').title()} vs Noise")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/displacement_vs_noise.png", dpi=150)
    plt.close()

    logging.info(f"Saved displacement plot to {output_dir}/displacement_vs_noise.png")


def plot_code_switching_vs_noise(
    results: dict[float, list[NoisyRolloutResult]],
    output_dir: str,
):
    """Plot code switching metrics vs noise level."""
    noise_levels = sorted(results.keys())

    transitions = []
    unique_codes = []
    survival = []

    for noise_std in noise_levels:
        trans_vals = [r.num_transitions for r in results[noise_std]]
        code_vals = [r.unique_codes_used for r in results[noise_std]]
        surv_vals = [r.survival_steps for r in results[noise_std]]
        transitions.append((np.mean(trans_vals), np.std(trans_vals)))
        unique_codes.append((np.mean(code_vals), np.std(code_vals)))
        survival.append((np.mean(surv_vals), np.std(surv_vals)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Transitions
    axes[0].errorbar(
        noise_levels,
        [t[0] for t in transitions],
        yerr=[t[1] for t in transitions],
        marker="o",
        capsize=5,
        color="tab:blue",
    )
    axes[0].set_xlabel("Noise std")
    axes[0].set_ylabel("Number of Transitions")
    axes[0].set_title("Code Transitions vs Noise")
    axes[0].grid(True, alpha=0.3)

    # Unique codes
    axes[1].errorbar(
        noise_levels,
        [c[0] for c in unique_codes],
        yerr=[c[1] for c in unique_codes],
        marker="s",
        capsize=5,
        color="tab:orange",
    )
    axes[1].set_xlabel("Noise std")
    axes[1].set_ylabel("Unique Codes Used")
    axes[1].set_title("Code Diversity vs Noise")
    axes[1].grid(True, alpha=0.3)

    # Survival
    axes[2].errorbar(
        noise_levels,
        [s[0] for s in survival],
        yerr=[s[1] for s in survival],
        marker="^",
        capsize=5,
        color="tab:green",
    )
    axes[2].set_xlabel("Noise std")
    axes[2].set_ylabel("Survival Steps")
    axes[2].set_title("Survival vs Noise")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/code_switching_vs_noise.png", dpi=150)
    plt.close()

    logging.info(f"Saved code switching plot to {output_dir}/code_switching_vs_noise.png")


def plot_transition_matrices(
    matrices: dict[float, tuple[np.ndarray, np.ndarray]],
    output_dir: str,
    num_codes: int,
):
    """Plot transition matrices for different noise levels."""
    noise_levels = sorted(matrices.keys())
    n_plots = len(noise_levels)

    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    for ax, noise_std in zip(axes, noise_levels):
        counts, probs = matrices[noise_std]

        im = ax.imshow(probs, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"noise_std = {noise_std}")
        ax.set_xlabel("To Code")
        ax.set_ylabel("From Code")

        # Set tick labels
        ax.set_xticks(range(num_codes))
        ax.set_yticks(range(num_codes))
        ax.set_xticklabels(range(num_codes), fontsize=8)
        ax.set_yticklabels(range(num_codes), fontsize=8)

        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/transition_matrices.png", dpi=150)
    plt.close()

    logging.info(f"Saved transition matrices to {output_dir}/transition_matrices.png")


# =============================================================================
# VIDEO RENDERING
# =============================================================================


def render_with_code_grid(
    frame: np.ndarray,
    current_code: int,
    num_codes: int = 12,
    code_colors: np.ndarray | None = None,
    grid_width: int = 100,
    cell_size: int = 28,
    padding: int = 4,
) -> np.ndarray:
    """Add a code grid overlay on the left side of the frame.

    Shows all codes in a 3x4 grid, with the active code highlighted.
    """
    h, w, _ = frame.shape

    # Create grid panel (dark grey background)
    grid_panel = np.ones((h, grid_width, 3), dtype=np.uint8) * 40

    # Calculate grid layout (3 columns for 12 codes = 4 rows)
    cells_per_row = 3
    num_rows = (num_codes + cells_per_row - 1) // cells_per_row

    # Starting y position to center grid vertically
    total_height = num_rows * (cell_size + padding) - padding
    start_y = (h - total_height) // 2
    if start_y < padding:
        start_y = padding

    start_x = (grid_width - cells_per_row * (cell_size + padding) + padding) // 2

    # Draw each code cell
    for i in range(num_codes):
        row = i // cells_per_row
        col = i % cells_per_row

        x = start_x + col * (cell_size + padding)
        y = start_y + row * (cell_size + padding)

        # Bounds check
        if y + cell_size > h or x + cell_size > grid_width:
            continue

        # Determine cell color
        if i == current_code:
            # Active code: use its color, full brightness
            if code_colors is not None and i < len(code_colors):
                color = code_colors[i].tolist()
            else:
                color = [100, 200, 100]  # Green default
            border_color = [255, 255, 255]  # White border
            border_width = 3
        else:
            # Inactive code: grey
            color = [70, 70, 70]
            border_color = [50, 50, 50]
            border_width = 1

        # Draw cell background
        grid_panel[y : y + cell_size, x : x + cell_size] = color

        # Draw border
        grid_panel[y : y + border_width, x : x + cell_size] = border_color
        grid_panel[y + cell_size - border_width : y + cell_size, x : x + cell_size] = (
            border_color
        )
        grid_panel[y : y + cell_size, x : x + border_width] = border_color
        grid_panel[y : y + cell_size, x + cell_size - border_width : x + cell_size] = (
            border_color
        )

    # Add code numbers using PIL
    try:
        pil_panel = Image.fromarray(grid_panel)
        draw = ImageDraw.Draw(pil_panel)

        # Try to load a font
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10
            )
        except (IOError, OSError):
            font = ImageFont.load_default()

        for i in range(num_codes):
            row = i // cells_per_row
            col = i % cells_per_row
            x = start_x + col * (cell_size + padding) + cell_size // 2
            y = start_y + row * (cell_size + padding) + cell_size // 2

            # Choose text color for contrast
            if i == current_code and code_colors is not None:
                color = code_colors[i]
                brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
                text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
            else:
                text_color = (200, 200, 200)

            draw.text((x - 5, y - 5), str(i), font=font, fill=text_color)

        grid_panel = np.array(pil_panel)
    except Exception:
        pass  # Skip text if PIL fails

    # Combine grid panel with frame
    combined = np.concatenate([grid_panel, frame], axis=1)

    return combined


def render_noisy_rollout_video(
    env: Any,
    result: NoisyRolloutResult,
    output_path: str,
    num_codes: int = 12,
    fps: int = 50,
    camera: str = "close_profile",
):
    """Render a noisy rollout to video with code grid and timeline."""
    # Get code colors
    code_colors = get_nature_colormap(num_codes)

    # Render environment frames
    logging.info(f"  Rendering {len(result.states)} frames...")
    frames = env.render(result.states, camera=camera)

    processed_frames = []
    for i, frame in enumerate(frames):
        current_code = int(result.indices[i]) if i < len(result.indices) else -1

        # Add code grid on left
        frame = render_with_code_grid(
            frame,
            current_code=current_code,
            num_codes=num_codes,
            code_colors=code_colors,
        )

        # Add code timeline bar at bottom
        if len(result.indices) > 0:
            frame = add_code_transition_bar(
                frame,
                current_frame_idx=i,
                all_indices=result.indices,
                code_colors=code_colors,
            )

        # Add noise info overlay
        frame = add_text_overlay(
            frame,
            f"noise={result.noise_std:.2f}",
            position=(frame.shape[1] - 130, 10),
        )

        processed_frames.append(frame)

    # Write video
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in processed_frames:
            writer.append_data(frame)

    logging.info(f"  Saved video to {output_path}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


@hydra.main(version_base=None, config_path="../configs", config_name="noise_analysis")
def main(cfg: DictConfig):
    """Run VQ-VAE prior noise analysis."""
    logging.set_verbosity(logging.INFO)

    print("=" * 60)
    print("VQ-VAE Prior Noise Analysis")
    print("=" * 60)

    # Create output directory
    output_dir = Path(cfg.output_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")

    # Load prior checkpoint (contains both prior and frozen VQ-VAE)
    logging.info("\nLoading prior checkpoint...")
    prior_ckpt = load_prior_checkpoint(
        cfg.prior_config.checkpoint_path, cfg.prior_config.checkpoint_step
    )
    prior_cfg = prior_ckpt["cfg"]
    prior_params = prior_ckpt["prior_params"]

    # Extract frozen VQ-VAE from prior checkpoint
    frozen_vqvae = prior_ckpt["frozen_vqvae"]
    decoder_params = frozen_vqvae["decoder_params"]
    codebook = frozen_vqvae["codebook"]
    num_codes = codebook.shape[0]
    latent_dim = codebook.shape[1]

    logging.info(f"  Codebook: {num_codes} codes, {latent_dim} dims")

    # Create environment
    logging.info("\nCreating environment...")
    (_, cfg_dict, env_cfg_ml) = config_utils.prepare_config(cfg)

    reference_clips = ReferenceClips(
        data_path=prior_cfg.env_config.reference_data_path,
        n_frames_per_clip=cfg.env_config.clip_length,
        keep_clips_idx=cfg.env_config.keep_clips_idx,
    )
    env = imitation.Imitation(config=env_cfg_ml, clips=reference_clips)

    # Create network modules
    prior_module = VQPrior(
        layer_sizes=list(prior_cfg.network_config.prior_layer_sizes),
        latent_dim=latent_dim,
    )

    # Get decoder config from the VQ-VAE checkpoint that was used during prior training
    vqvae_ckpt_path = prior_cfg.network_config.vqvae_checkpoint_path
    vqvae_step = prior_cfg.network_config.get("vqvae_checkpoint_step", None)
    vqvae_ckpt = load_vq_checkpoint(vqvae_ckpt_path, step=vqvae_step)
    vqvae_cfg = vqvae_ckpt["cfg"]

    action_size = vqvae_cfg.network_config.action_size
    decoder_layer_sizes = list(vqvae_cfg.network_config.decoder_layer_sizes) + [
        action_size * 2
    ]
    decoder_module = Decoder(layer_sizes=decoder_layer_sizes)

    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )

    # Get camera name with suffix
    env_suffix = getattr(env, "_suffix", "-rodent")
    camera_name = f"{cfg.render_config.camera_name}{env_suffix}"

    # Run analysis
    noise_levels = list(cfg.noise_config.noise_levels)
    num_rollouts = cfg.rollout_config.num_rollouts_per_noise
    max_steps = cfg.rollout_config.max_steps

    logging.info(f"\nNoise levels: {noise_levels}")
    logging.info(f"Rollouts per noise: {num_rollouts}")
    logging.info(f"Max steps: {max_steps}")

    results: dict[float, list[NoisyRolloutResult]] = {}

    for noise_std in noise_levels:
        print(f"\n{'='*50}")
        print(f"Running noise_std = {noise_std}")
        print(f"{'='*50}")

        noise_results = []

        for rollout_idx in range(num_rollouts):
            # Use same seed for same rollout_idx across noise levels for fair comparison
            seed = hash(rollout_idx) % (2**31)

            result = run_noisy_rollout(
                env=env,
                prior_params=prior_params,
                decoder_params=decoder_params,
                codebook=codebook,
                prior_module=prior_module,
                decoder_module=decoder_module,
                parametric_action_distribution=parametric_action_distribution,
                max_steps=max_steps,
                seed=seed,
                noise_std=noise_std,
                quantize_prior=cfg.rollout_config.quantize_prior,
                deterministic=cfg.rollout_config.deterministic,
            )

            noise_results.append(result)

            print(
                f"  Rollout {rollout_idx}: survival={result.survival_steps}, "
                f"transitions={result.num_transitions}, "
                f"codes={result.unique_codes_used}, "
                f"displacement={result.total_displacement:.3f}"
            )

        results[noise_std] = noise_results

    # Compute transition matrices
    logging.info("\nComputing transition matrices...")
    matrices = compute_transition_matrices(results, num_codes)

    # Generate plots
    logging.info("\nGenerating plots...")
    plot_trajectory_comparison(results, str(output_dir))
    plot_trajectory_grid(results, str(output_dir))
    plot_displacement_vs_noise(results, str(output_dir))
    plot_code_switching_vs_noise(results, str(output_dir))
    plot_transition_matrices(matrices, str(output_dir), num_codes)

    # Render videos if enabled
    if cfg.render_config.enabled and cfg.output_config.save_videos:
        logging.info("\nRendering videos...")
        for noise_std, rollouts in results.items():
            # Render first rollout per noise level
            result = rollouts[0]
            video_path = output_dir / f"rollout_noise_{noise_std:.2f}.mp4"
            render_noisy_rollout_video(
                env=env,
                result=result,
                output_path=str(video_path),
                num_codes=num_codes,
                fps=cfg.render_config.fps,
                camera=camera_name,
            )

    # Save summary
    logging.info("\nSaving summary...")
    summary = {
        "noise_levels": noise_levels,
        "rollouts_per_noise": num_rollouts,
        "max_steps": max_steps,
        "num_codes": num_codes,
        "latent_dim": latent_dim,
        "prior_checkpoint": cfg.prior_config.checkpoint_path,
        "vqvae_checkpoint": cfg.vqvae_config.checkpoint_path,
    }

    for noise_std, rollouts in results.items():
        summary[f"noise_{noise_std}"] = {
            "avg_survival": float(np.mean([r.survival_steps for r in rollouts])),
            "std_survival": float(np.std([r.survival_steps for r in rollouts])),
            "avg_displacement": float(np.mean([r.total_displacement for r in rollouts])),
            "std_displacement": float(np.std([r.total_displacement for r in rollouts])),
            "avg_transitions": float(np.mean([r.num_transitions for r in rollouts])),
            "std_transitions": float(np.std([r.num_transitions for r in rollouts])),
            "avg_unique_codes": float(np.mean([r.unique_codes_used for r in rollouts])),
            "std_unique_codes": float(np.std([r.unique_codes_used for r in rollouts])),
        }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Analysis complete! Results saved to {output_dir}")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  - trajectory_comparison.png/pdf")
    print(f"  - trajectory_grid.png")
    print(f"  - displacement_vs_noise.png")
    print(f"  - code_switching_vs_noise.png")
    print(f"  - transition_matrices.png")
    if cfg.render_config.enabled:
        print(f"  - rollout_noise_*.mp4 (one per noise level)")
    print(f"  - summary.json")


if __name__ == "__main__":
    main()
