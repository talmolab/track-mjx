"""Loss functions for VQ-VAE prior distillation training.

This module provides loss functions for training a Prior network to predict
the encoder's continuous embeddings (z_e) from proprioceptive observations only.
Unlike the VAE mlp_distill which uses KL divergence, we use MSE/L2 losses since
the VQ-VAE encoder outputs single embeddings rather than distributions.

Key design principle:
- The VQ-VAE encoder/decoder/codebook are FROZEN
- Only the Prior network receives gradients
- Loss = MSE(z_p, stop_gradient(z_e))

Loss options:
- MSE: Mean squared error (default, most common)
- L2: L2 norm per sample then average
- Smooth L1: Huber loss (less sensitive to outliers)
- Cosine: Cosine similarity (focuses on direction)
- Combined: MSE + weighted Cosine

Reference: track_mjx/agent/mlp_distill/losses.py (VAE version uses KL divergence)
"""

from collections.abc import Callable, Mapping
from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax
from brax.training import types
from brax.training.types import Params

from track_mjx.agent.observation_utils import flatten_obs_dict


@flax.struct.dataclass
class VQPriorDistillNetworkParams:
    """Container for VQ-VAE prior distillation network parameters.

    Attributes:
        prior: Prior network parameters (TRAINABLE - receives gradients).
    """

    prior: Params


def compute_mse_alignment_loss(
    z_p: jnp.ndarray,
    z_e: jnp.ndarray,
) -> jnp.ndarray:
    """Compute MSE loss between prior prediction and encoder output.

    Mean squared error - the standard choice for regression tasks.
    Computes element-wise squared error then averages over all dimensions.

    Args:
        z_p: Prior network output [..., latent_dim].
        z_e: Encoder output (frozen) [..., latent_dim].

    Returns:
        Scalar MSE loss.
    """
    # CRITICAL: Stop gradient on z_e - encoder is frozen
    z_e_sg = jax.lax.stop_gradient(z_e)

    # Element-wise squared error
    squared_error = jnp.square(z_p - z_e_sg)

    # Mean over all dimensions
    return jnp.mean(squared_error)


def compute_l2_alignment_loss(
    z_p: jnp.ndarray,
    z_e: jnp.ndarray,
) -> jnp.ndarray:
    """Compute L2 norm loss between prior prediction and encoder output.

    Different from MSE: Computes L2 norm per sample, then averages.
    More sensitive to large deviations in any dimension.

    Args:
        z_p: Prior network output [..., latent_dim].
        z_e: Encoder output (frozen) [..., latent_dim].

    Returns:
        Scalar L2 norm loss.
    """
    z_e_sg = jax.lax.stop_gradient(z_e)

    # L2 norm per sample (over latent dimension)
    l2_norms = jnp.linalg.norm(z_p - z_e_sg, axis=-1)

    return jnp.mean(l2_norms)


def compute_smooth_l1_loss(
    z_p: jnp.ndarray,
    z_e: jnp.ndarray,
    delta: float = 1.0,
) -> jnp.ndarray:
    """Compute Smooth L1 (Huber) loss - less sensitive to outliers than MSE.

    L = 0.5 * x² if |x| < delta
        delta * (|x| - 0.5 * delta) otherwise

    Args:
        z_p: Prior network output [..., latent_dim].
        z_e: Encoder output (frozen) [..., latent_dim].
        delta: Threshold for switching between quadratic and linear.

    Returns:
        Scalar Smooth L1 loss.
    """
    z_e_sg = jax.lax.stop_gradient(z_e)
    diff = z_p - z_e_sg
    abs_diff = jnp.abs(diff)

    quadratic = jnp.minimum(abs_diff, delta)
    linear = abs_diff - quadratic

    loss = 0.5 * quadratic**2 + delta * linear
    return jnp.mean(loss)


def compute_cosine_loss(
    z_p: jnp.ndarray,
    z_e: jnp.ndarray,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Compute cosine similarity loss - focuses on direction, not magnitude.

    Useful if the magnitude of z_e varies but direction matters more.
    Loss = 1 - cos_sim(z_p, z_e)

    Args:
        z_p: Prior network output [..., latent_dim].
        z_e: Encoder output (frozen) [..., latent_dim].
        eps: Small epsilon for numerical stability.

    Returns:
        Scalar cosine loss (0 when perfectly aligned, 2 when opposite).
    """
    z_e_sg = jax.lax.stop_gradient(z_e)

    # Normalize
    z_p_norm = z_p / (jnp.linalg.norm(z_p, axis=-1, keepdims=True) + eps)
    z_e_norm = z_e_sg / (jnp.linalg.norm(z_e_sg, axis=-1, keepdims=True) + eps)

    # Cosine similarity
    cos_sim = jnp.sum(z_p_norm * z_e_norm, axis=-1)

    # Loss: 1 - similarity (so lower is better when similar)
    return jnp.mean(1.0 - cos_sim)


def compute_combined_alignment_loss(
    z_p: jnp.ndarray,
    z_e: jnp.ndarray,
    mse_weight: float = 1.0,
    cosine_weight: float = 0.1,
) -> jnp.ndarray:
    """Combined MSE + Cosine loss.

    MSE ensures magnitude matching, Cosine ensures direction matching.

    Args:
        z_p: Prior network output [..., latent_dim].
        z_e: Encoder output (frozen) [..., latent_dim].
        mse_weight: Weight for MSE component.
        cosine_weight: Weight for cosine component.

    Returns:
        Scalar combined loss.
    """
    mse = compute_mse_alignment_loss(z_p, z_e)
    cosine = compute_cosine_loss(z_p, z_e)

    return mse_weight * mse + cosine_weight * cosine


def compute_ar1_loss(
    z_p: jnp.ndarray,
    discount: jnp.ndarray,
    phi: float = 0.99,
) -> jnp.ndarray:
    """Compute AR(1) temporal smoothness loss for prior outputs.

    Encourages temporal smoothness in the latent space by modeling the latent
    as an AR(1) process: z_t ≈ φ * z_{t-1}. Uses mean L2 norm like PULSE.

    Episode boundaries are handled by masking out pairs where discount=0,
    which indicates an episode ended at that timestep.

    Args:
        z_p: Prior network outputs [T, B, latent_dim].
        discount: Discount factor from environment [T, B], 0.0 at episode end.
        phi: AR(1) coefficient (default 0.99, same as PULSE).

    Returns:
        Scalar AR(1) loss (mean L2 norm of prediction errors).
    """
    if z_p.shape[0] <= 1:
        return jnp.array(0.0)

    # Get consecutive prior outputs
    z_prev = z_p[:-1]  # z_0, ..., z_{T-2}  [T-1, B, d]
    z_curr = z_p[1:]  # z_1, ..., z_{T-1}  [T-1, B, d]

    # AR(1) prediction error: z_t - φ * z_{t-1}
    error = z_curr - phi * z_prev  # [T-1, B, d]

    # Mask for valid pairs: discount[t] > 0 means episode didn't end at t
    # So (z_t, z_{t+1}) is a valid same-episode pair
    valid_mask = discount[:-1] > 0  # [T-1, B]

    # Compute L2 norm of error for each (t, batch) pair
    l2_norms = jnp.linalg.norm(error, axis=-1)  # [T-1, B]

    # Apply mask and compute mean only over valid pairs
    masked_norms = l2_norms * valid_mask
    num_valid = jnp.sum(valid_mask)

    # Avoid division by zero
    ar_loss = jnp.sum(masked_norms) / jnp.maximum(num_valid, 1.0)

    return ar_loss


def compute_vq_prior_distill_loss(
    params: VQPriorDistillNetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    step: int,
    prior_network: Any,
    frozen_encoder: Any,
    frozen_encoder_params: Params,
    reference_obs_size: int,
    loss_type: str = "mse",
    ar_weight: float = 0.0,
    phi: float = 0.99,
    ar_schedule: Callable[[int], float] | None = None,
    smooth_l1_delta: float = 1.0,
    mse_weight: float = 1.0,
    cosine_weight: float = 0.1,
) -> tuple[jnp.ndarray, types.Metrics]:
    """Compute the total loss for VQ-VAE prior distillation.

    The total loss is:
    L = alignment_loss + ar_weight * L_ar

    Where:
    - alignment_loss: One of MSE, L2, Smooth L1, Cosine, or Combined
    - L_ar: Optional AR(1) temporal smoothness loss

    CRITICAL DESIGN:
    - frozen_encoder_params are NEVER updated (stop_gradient is used)
    - Only params.prior receives gradients

    Args:
        params: Prior network parameters (TRAINABLE).
        normalizer_params: Observation normalizer parameters.
        data: Transition data with shape [B, T].
        rng: Random key (unused in VQ-VAE, kept for API compatibility).
        step: Current training step (for optional scheduling).
        prior_network: Prior network apply function.
        frozen_encoder: Frozen VQ-VAE encoder apply function.
        frozen_encoder_params: Frozen encoder parameters (NO gradients).
        reference_obs_size: Size of reference trajectory in observations.
        loss_type: One of "mse", "l2", "smooth_l1", "cosine", "combined".
        ar_weight: Weight for AR(1) temporal smoothness loss.
        phi: AR(1) coefficient (default 0.99).
        ar_schedule: Optional schedule function for AR weight.
        smooth_l1_delta: Delta for Smooth L1 loss.
        mse_weight: MSE weight for combined loss.
        cosine_weight: Cosine weight for combined loss.

    Returns:
        Tuple of (total_loss, metrics_dict).
    """
    # Put the time dimension first: [B, T] -> [T, B]
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    # Extract trajectory and proprioceptive observations from dict
    # data.observation is a dict: {"imitation_target": ..., "proprioception": ...}
    # Each value has shape [T, B, obs_dim] after swapaxes
    # Flatten nested observations and access by key
    flat_obs = flatten_obs_dict(data.observation)
    traj = flat_obs["imitation_target"]
    proprio = flat_obs["proprioception"]

    # Get encoder outputs (FROZEN - no gradients flow here)
    # z_e shape: [T, B, latent_dim]
    z_e = frozen_encoder.apply(frozen_encoder_params, traj)
    # Double protection: explicit stop_gradient
    z_e = jax.lax.stop_gradient(z_e)

    # Get prior outputs (TRAINABLE - gradients flow here)
    # z_p shape: [T, B, latent_dim]
    # Prior only uses proprioception, so pass just the proprioception normalizer
    z_p = prior_network.apply(params.prior, normalizer_params.proprioception, proprio)

    # Compute alignment loss based on loss_type
    if loss_type == "mse":
        alignment_loss = compute_mse_alignment_loss(z_p, z_e)
    elif loss_type == "l2":
        alignment_loss = compute_l2_alignment_loss(z_p, z_e)
    elif loss_type == "smooth_l1":
        alignment_loss = compute_smooth_l1_loss(z_p, z_e, delta=smooth_l1_delta)
    elif loss_type == "cosine":
        alignment_loss = compute_cosine_loss(z_p, z_e)
    elif loss_type == "combined":
        alignment_loss = compute_combined_alignment_loss(
            z_p, z_e, mse_weight=mse_weight, cosine_weight=cosine_weight
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Optional AR loss
    current_ar_weight = ar_weight
    if ar_schedule is not None:
        current_ar_weight = ar_schedule(step)

    ar_loss = jnp.array(0.0)
    if current_ar_weight > 0:
        ar_loss = compute_ar1_loss(z_p, data.discount, phi)

    # Total loss
    total_loss = alignment_loss + current_ar_weight * ar_loss

    # Compute metrics
    z_diff = z_p - z_e
    metrics = {
        "total_loss": total_loss,
        "alignment_loss": alignment_loss,
        "ar_loss": ar_loss,
        "ar_weight": current_ar_weight,
        # Prior output statistics
        "z_p_mean": jnp.mean(z_p),
        "z_p_std": jnp.std(z_p),
        "z_p_min": jnp.min(z_p),
        "z_p_max": jnp.max(z_p),
        # Encoder output statistics (for comparison)
        "z_e_mean": jnp.mean(z_e),
        "z_e_std": jnp.std(z_e),
        # Difference statistics
        "z_diff_mean": jnp.mean(jnp.abs(z_diff)),
        "z_diff_max": jnp.max(jnp.abs(z_diff)),
        # L2 distance between z_p and z_e
        "z_l2_dist": jnp.mean(jnp.linalg.norm(z_diff, axis=-1)),
    }

    return total_loss, metrics


def create_ar_schedule(
    start_value: float = 0.0,
    end_value: float = 1e-3,
    total_steps: int = 100,
    start_frac: float = 0.3,
    end_frac: float = 0.6,
    schedule_type: str = "linear",
) -> Callable[[int], float]:
    """Create a warmup schedule for AR loss weight.

    Starts at start_value and ramps up to end_value between
    start_frac and end_frac of total_steps.

    Args:
        start_value: Initial AR loss weight.
        end_value: Final AR loss weight.
        total_steps: Total number of training steps.
        start_frac: Fraction of total_steps at which ramping begins.
        end_frac: Fraction of total_steps by which ramping completes.
        schedule_type: Type of schedule - "linear" or "cosine".

    Returns:
        A function mapping step -> ar_weight.
    """
    start_step = int(start_frac * total_steps)
    end_step = int(end_frac * total_steps)
    ramp_steps = max(end_step - start_step, 1)

    def schedule_fn(step: int) -> float:
        step = jnp.asarray(step, dtype=jnp.float32)

        if schedule_type == "linear":
            effective_step = step - start_step
            progress = jnp.clip(effective_step / ramp_steps, 0.0, 1.0)
            is_before_start = step < start_step
            ramped_value = start_value + progress * (end_value - start_value)
            return jnp.where(is_before_start, start_value, ramped_value)
        elif schedule_type == "cosine":
            effective_step = jnp.maximum(step - start_step, 0.0)
            is_before_start = step < start_step
            progress = jnp.clip(effective_step / ramp_steps, 0.0, 1.0)
            cosine_value = start_value + 0.5 * (end_value - start_value) * (
                1 - jnp.cos(jnp.pi * progress)
            )
            return jnp.where(is_before_start, start_value, cosine_value)
        else:
            raise ValueError(
                f"schedule_type must be 'linear' or 'cosine', not {schedule_type}"
            )

    return schedule_fn
