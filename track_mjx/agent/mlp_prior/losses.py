"""
Loss functions for prior network training.

The prior training loss consists of KL divergence between the encoder
(frozen, from pretrained mlp_ppo) and the prior (trainable) distributions.

Observations are expected as dictionaries with keys:
- "imitation_target": Reference trajectory observations (flat array)
- "proprioception": Proprioceptive state observations (flat array)
"""

from typing import Any, Callable, Dict, Tuple

from brax.training import types
from brax.training.acme import running_statistics

import jax
import jax.numpy as jnp
import optax

from track_mjx.agent.observation_utils import (
    DictRunningStatisticsState,
    normalize_dict_obs,
)


def compute_encoder_prior_kl_loss(
    encoder_mean: jnp.ndarray,
    encoder_logvar: jnp.ndarray,
    prior_mean: jnp.ndarray,
    prior_logvar: jnp.ndarray,
) -> jnp.ndarray:
    """Compute KL divergence between encoder and prior distributions.

    KL(q(z|x) || p(z|x_proprio)) where:
    - q(z|x) is the encoder distribution (Gaussian with encoder_mean, encoder_logvar)
    - p(z|x_proprio) is the prior distribution (Gaussian with prior_mean, prior_logvar)

    Uses sum over latent dimensions, then mean over samples.
    This is the mathematically correct KL for multivariate Gaussians with diagonal covariance.

    Args:
        encoder_mean: Mean of encoder distribution [T, B, latent_dim]
        encoder_logvar: Log-variance of encoder distribution [T, B, latent_dim]
        prior_mean: Mean of prior distribution [T, B, latent_dim]
        prior_logvar: Log-variance of prior distribution [T, B, latent_dim]

    Returns:
        Scalar KL divergence loss
    """
    # KL divergence between two Gaussians (element-wise per latent dimension):
    # KL_j = 0.5 * (log(σ_p^2/σ_q^2) + σ_q^2/σ_p^2 + (μ_q - μ_p)^2/σ_p^2 - 1)
    log_var_diff = prior_logvar - encoder_logvar  # log(σ_p^2) - log(σ_q^2)
    var_ratio = jnp.exp(encoder_logvar - prior_logvar)  # σ_q^2 / σ_p^2
    mean_diff_sq = jnp.square(encoder_mean - prior_mean) / jnp.exp(
        prior_logvar
    )  # (μ_q - μ_p)^2 / σ_p^2

    # Element-wise KL for each latent dimension
    element_wise_kl = 0.5 * (log_var_diff + var_ratio + mean_diff_sq - 1)  # [T, B, d]

    # Sum over latent dimensions (correct KL for multivariate Gaussian)
    # Then mean over samples (T × B)
    kl_per_sample = jnp.sum(element_wise_kl, axis=-1)  # [T, B]
    kl_loss = jnp.mean(kl_per_sample)  # scalar

    return kl_loss


def create_ramp_schedule(
    start_value: float = 0.0001,
    end_value: float = 0.1,
    total_steps: int = 100,
    start_frac: float = 0.0,
    end_frac: float = 1.0,
    schedule: str = "linear",
) -> Callable[[int], jnp.ndarray]:
    """
    Creates a schedule for loss weights.

    Args:
        start_value: The starting value of the schedule.
        end_value: The ending value of the schedule.
        total_steps: The total number of evaluation steps.
        start_frac: The fraction of total_steps at which ramping should begin.
        end_frac: The fraction of total_steps by which ramping should complete.
        schedule: Type of schedule - "linear" or "cosine".

    Returns:
        A function that takes a step and returns the scheduled value.
    """
    delay_steps = int(start_frac * total_steps)
    ramp_steps = int((end_frac - start_frac) * total_steps)
    ramp_steps = max(ramp_steps, 1)

    def schedule_fn(step: int) -> jnp.ndarray:
        step = jnp.asarray(step, dtype=jnp.float32)

        if schedule == "linear":
            effective_step = step - delay_steps
            progress = jnp.clip(effective_step / ramp_steps, 0.0, 1.0)
            is_delayed = step < delay_steps
            ramped_value = start_value + progress * (end_value - start_value)
            return jnp.where(is_delayed, start_value, ramped_value)
        elif schedule == "cosine":
            effective_step = jnp.maximum(step - delay_steps, 0.0)
            is_delayed = step < delay_steps
            progress = jnp.clip(effective_step / ramp_steps, 0.0, 1.0)
            cosine_value = start_value + 0.5 * (end_value - start_value) * (
                1 - jnp.cos(jnp.pi * progress)
            )
            return jnp.where(is_delayed, start_value, cosine_value)
        else:
            raise ValueError(f"schedule must be 'linear' or 'cosine', not {schedule}")

    return schedule_fn


def compute_prior_training_loss(
    prior_params: Dict,
    frozen_encoder_params: Dict,
    normalizer_params: DictRunningStatisticsState,
    data: types.Transition,
    rng: jnp.ndarray,
    step: int,
    encoder_apply_fn: Callable,
    prior_apply_fn: Callable,
    reference_obs_size: int,
    kl_weight: float = 1.0,
    kl_schedule: Callable | None = None,
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """Compute the prior training loss.

    The loss is KL divergence between encoder (frozen) and prior (trainable) distributions.

    Args:
        prior_params: Trainable prior network parameters
        frozen_encoder_params: Frozen encoder network parameters (from mlp_ppo)
        normalizer_params: Dict observation normalizer parameters
        data: Transition data with shape [B, T], observation is a dict
        rng: Random key
        step: Current training step (for scheduling)
        encoder_apply_fn: Function to apply encoder: (params, obs, key) -> (mean, logvar)
        prior_apply_fn: Function to apply prior: (params, obs) -> (mean, logvar)
        reference_obs_size: Size of reference/trajectory observations (unused with dict obs)
        kl_weight: Weight for KL divergence loss
        kl_schedule: Optional schedule function for KL weight

    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    _, encoder_key = jax.random.split(rng, 2)

    # Put the time dimension first: [B, T] -> [T, B]
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    # Merge time and batch dimensions: [T, B, ...] -> [T*B, ...]
    # This ensures observations have shape [T*B, features] for normalization
    data = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), data)

    # Normalize dict observations (flatten_obs_dict is called internally)
    normalized_obs = normalize_dict_obs(data.observation, normalizer_params)

    # Access observations by key
    traj_obs = normalized_obs["imitation_target"]
    proprio_obs = normalized_obs["proprioception"]

    # Get encoder outputs (frozen - apply stop_gradient)
    encoder_mean, encoder_logvar = encoder_apply_fn(
        frozen_encoder_params, traj_obs, encoder_key
    )
    encoder_mean = jax.lax.stop_gradient(encoder_mean)
    encoder_logvar = jax.lax.stop_gradient(encoder_logvar)

    # Get prior outputs (trainable)
    # prior_params is a PriorTrainingParams dataclass, extract the actual params
    prior_mean, prior_logvar = prior_apply_fn(prior_params.prior, proprio_obs)

    # Compute KL loss
    kl_loss = compute_encoder_prior_kl_loss(
        encoder_mean, encoder_logvar, prior_mean, prior_logvar
    )

    # Apply schedule if provided
    current_kl_weight = kl_weight
    if kl_schedule is not None:
        current_kl_weight = kl_schedule(step)

    # Total loss
    total_loss = current_kl_weight * kl_loss

    metrics = {
        "kl_loss": kl_loss,
        "kl_weight": current_kl_weight,
        "mean_diff_l2": jnp.mean(jnp.linalg.norm(encoder_mean - prior_mean, axis=-1)),
        "logvar_diff_l2": jnp.mean(
            jnp.linalg.norm(encoder_logvar - prior_logvar, axis=-1)
        ),
    }

    return total_loss, metrics
