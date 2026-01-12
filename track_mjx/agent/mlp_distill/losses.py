"""
Loss functions for distillation training.

The distillation loss consists of:
1. MSE loss between student and teacher actions
2. AR(1) loss between consecutive latent means (z_t - φ*z_{t-1}), matching PULSE
3. KL divergence loss between encoder and prior distributions

Observations are expected as dictionaries with keys:
- "imitation_target": Reference trajectory observations (flat array)
- "proprioception": Proprioceptive state observations (flat array)
"""

from typing import Any, Callable, Tuple

from brax.training import types
from brax.training.types import Params

import flax
import jax
import jax.numpy as jnp
import optax


@flax.struct.dataclass
class DistillNetworkParams:
    """Contains training parameters for the student network."""

    policy: Params  # Student policy parameters (encoder, decoder, prior)


def compute_action_loss(
    student_actions: jnp.ndarray,
    teacher_actions: jnp.ndarray,
    use_l2_norm: bool = False,
) -> jnp.ndarray:
    """Compute action reconstruction loss between student and teacher actions.

    Args:
        student_actions: Actions predicted by the student network [T, B, action_dim]
        teacher_actions: Actions from the teacher network [T, B, action_dim]
        use_l2_norm: If True, use mean L2 norm (like PULSE). If False, use MSE.

    Returns:
        Scalar action loss
    """
    if use_l2_norm:
        # Mean L2 norm (Euclidean distance) - same as PULSE
        # Computes ||a - a_gt||_2 for each sample, then averages
        l2_norms = jnp.linalg.norm(student_actions - teacher_actions, axis=-1)
        return jnp.mean(l2_norms)
    else:
        # MSE loss - mean of squared errors across all dimensions
        return jnp.mean(jnp.square(student_actions - teacher_actions))


def compute_autoregressive_loss(
    latent_means: jnp.ndarray,
    discount: jnp.ndarray,
    phi: float = 0.99,
) -> jnp.ndarray:
    """Compute AR(1) autoregressive loss between consecutive latent means.

    Encourages temporal smoothness in the latent space by modeling the latent
    as an AR(1) process: z_t ≈ φ * z_{t-1}. Uses mean L2 norm like PULSE.

    Episode boundaries are handled by masking out pairs where discount=0,
    which indicates an episode ended at that timestep.

    Args:
        latent_means: Latent means from encoder [T, B, latent_dim]
        discount: Discount factor from environment [T, B], 0.0 at episode end
        phi: AR(1) coefficient (default 0.99, same as PULSE)

    Returns:
        Scalar AR(1) loss (mean L2 norm of prediction errors)
    """
    if latent_means.shape[0] <= 1:
        return jnp.array(0.0)

    # Get consecutive latent means
    z_prev = latent_means[:-1]  # z_0, ..., z_{T-2}  [T-1, B, d]
    z_curr = latent_means[1:]  # z_1, ..., z_{T-1}  [T-1, B, d]

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

    Stop gradients are applied to the encoder outputs so that this loss only
    trains the prior network to match the encoder distribution.

    Uses PULSE-style aggregation: sum over latent dimensions, then mean over samples.
    This is the mathematically correct KL for multivariate Gaussians with diagonal covariance.

    Args:
        encoder_mean: Mean of encoder distribution [T, B, latent_dim]
        encoder_logvar: Log-variance of encoder distribution [T, B, latent_dim]
        prior_mean: Mean of prior distribution [T, B, latent_dim]
        prior_logvar: Log-variance of prior distribution [T, B, latent_dim]

    Returns:
        Scalar KL divergence loss
    """
    # Stop gradients on encoder outputs - this loss only trains the prior
    encoder_mean_sg = jax.lax.stop_gradient(encoder_mean)
    encoder_logvar_sg = jax.lax.stop_gradient(encoder_logvar)

    # KL divergence between two Gaussians (element-wise per latent dimension):
    # KL_j = 0.5 * (log(σ_p^2/σ_q^2) + σ_q^2/σ_p^2 + (μ_q - μ_p)^2/σ_p^2 - 1)
    log_var_diff = prior_logvar - encoder_logvar_sg  # log(σ_p^2) - log(σ_q^2)
    var_ratio = jnp.exp(encoder_logvar_sg - prior_logvar)  # σ_q^2 / σ_p^2
    mean_diff_sq = jnp.square(encoder_mean_sg - prior_mean) / jnp.exp(
        prior_logvar
    )  # (μ_q - μ_p)^2 / σ_p^2

    # Element-wise KL for each latent dimension
    element_wise_kl = 0.5 * (log_var_diff + var_ratio + mean_diff_sq - 1)  # [T, B, d]

    # Sum over latent dimensions (correct KL for multivariate Gaussian)
    # Then mean over samples (T × B) - matches PULSE aggregation
    kl_per_sample = jnp.sum(element_wise_kl, axis=-1)  # [T, B]
    kl_loss = jnp.mean(kl_per_sample)  # scalar

    return kl_loss


def compute_encoder_kl_to_standard_normal(
    encoder_mean: jnp.ndarray,
    encoder_logvar: jnp.ndarray,
) -> jnp.ndarray:
    """Compute KL divergence between encoder distribution and standard normal.

    KL(q(z|x) || N(0, I)) where:
    - q(z|x) is the encoder distribution (Gaussian with encoder_mean, encoder_logvar)
    - N(0, I) is the standard normal with mean=0 and variance=1

    This regularizes the encoder to produce distributions close to the standard normal.

    Args:
        encoder_mean: Mean of encoder distribution [T, B, latent_dim]
        encoder_logvar: Log-variance of encoder distribution [T, B, latent_dim]

    Returns:
        Scalar KL divergence loss
    """
    # KL(N(μ, σ²) || N(0, 1)) = 0.5 * (σ² + μ² - 1 - log(σ²))
    # = 0.5 * (exp(log_var) + μ² - 1 - log_var)
    element_wise_kl = 0.5 * (
        jnp.exp(encoder_logvar) + jnp.square(encoder_mean) - 1 - encoder_logvar
    )  # [T, B, d]

    # Sum over latent dimensions, then mean over samples
    kl_per_sample = jnp.sum(element_wise_kl, axis=-1)  # [T, B]
    kl_loss = jnp.mean(kl_per_sample)  # scalar

    return kl_loss


def compute_distillation_loss(
    params: DistillNetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    step: int,
    student_network: Any,
    teacher_policy_fn: Callable,
    teacher_params: Any,
    action_loss_weight: float = 1.0,
    autoregressive_weight: float = 1e-3,
    kl_weight: float = 1e-3,
    encoder_kl_weight: float = 1e-3,
    kl_schedule: Callable | None = None,
    ar_schedule: Callable | None = None,
    encoder_kl_schedule: Callable | None = None,
    use_l2_action_loss: bool = False,
) -> Tuple[jnp.ndarray, types.Metrics]:
    """Compute the combined distillation loss.

    The total loss is:
    L = action_loss_weight * L_action + autoregressive_weight * L_ar + kl_weight * L_kl_prior + encoder_kl_weight * L_kl_encoder

    Where:
    - L_action: Action reconstruction loss (MSE or mean L2 norm)
    - L_ar: AR(1) loss (mean L2 norm of z_t - φ*z_{t-1}), with episode boundary masking
    - L_kl_prior: KL divergence between encoder (stop_grad) and prior (trains prior only)
    - L_kl_encoder: KL divergence between encoder and standard normal N(0, I) (regularizes encoder)

    Args:
        params: Student network parameters
        normalizer_params: Observation normalizer parameters
        data: Transition data with shape [B, T]
        rng: Random key
        step: Current training step (for scheduling)
        student_network: Student policy network (FeedForwardNetwork)
        teacher_policy_fn: Teacher policy function
        teacher_params: Teacher policy parameters (frozen)
        action_loss_weight: Weight for action reconstruction loss
        autoregressive_weight: Weight for autoregressive loss
        kl_weight: Weight for encoder-prior KL divergence loss (trains prior only)
        encoder_kl_weight: Weight for encoder-to-standard-normal KL loss (regularizes encoder)
        kl_schedule: Optional schedule function for encoder-prior KL weight
        ar_schedule: Optional schedule function for autoregressive weight
        encoder_kl_schedule: Optional schedule function for encoder KL weight
        use_l2_action_loss: If True, use mean L2 norm (like PULSE). If False, use MSE.

    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    _, student_key, teacher_key = jax.random.split(rng, 3)

    # Put the time dimension first: [B, T] -> [T, B]
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    # Get student outputs
    student_logits, encoder_mean, encoder_logvar, prior_mean, prior_logvar = (
        student_network.apply(
            normalizer_params, params.policy, data.observation, student_key
        )
    )

    # Get teacher actions (frozen - no gradients)
    # Note: deterministic behavior is already captured in the teacher_policy_fn closure
    teacher_actions, teacher_extras = teacher_policy_fn(
        teacher_params, data.observation, teacher_key
    )
    teacher_actions = jax.lax.stop_gradient(teacher_actions)

    # Convert student logits to actions (using mean for distillation)
    # The logits contain (mean, log_std) for each action dimension
    action_dim = student_logits.shape[-1] // 2
    student_action_mean = student_logits[..., :action_dim]
    # Apply tanh to get actions in [-1, 1]
    student_actions = jnp.tanh(student_action_mean)

    # Compute individual losses
    action_loss = compute_action_loss(
        student_actions, teacher_actions, use_l2_action_loss
    )
    ar_loss = compute_autoregressive_loss(encoder_mean, data.discount)

    # KL loss between encoder (stop_grad) and prior - trains prior to match encoder
    kl_prior_loss = compute_encoder_prior_kl_loss(
        encoder_mean, encoder_logvar, prior_mean, prior_logvar
    )

    # KL loss between encoder and standard normal N(0, I) - regularizes encoder
    kl_encoder_loss = compute_encoder_kl_to_standard_normal(
        encoder_mean, encoder_logvar
    )

    # Apply schedules if provided
    current_kl_weight = kl_weight
    current_ar_weight = autoregressive_weight
    current_encoder_kl_weight = encoder_kl_weight
    if kl_schedule is not None:
        current_kl_weight = kl_schedule(step)
    if ar_schedule is not None:
        current_ar_weight = ar_schedule(step)
    if encoder_kl_schedule is not None:
        current_encoder_kl_weight = encoder_kl_schedule(step)

    # Compute weighted total loss
    total_loss = (
        action_loss_weight * action_loss
        + current_ar_weight * ar_loss
        + current_kl_weight * kl_prior_loss
        + current_encoder_kl_weight * kl_encoder_loss
    )

    metrics = {
        "total_loss": total_loss,
        "action_loss": action_loss,
        "autoregressive_loss": ar_loss,
        "kl_prior_loss": kl_prior_loss,
        "kl_encoder_loss": kl_encoder_loss,
        "kl_weight": current_kl_weight,
        "encoder_kl_weight": current_encoder_kl_weight,
        "ar_weight": current_ar_weight,
    }

    return total_loss, metrics


def create_ramp_schedule(
    start_value: float = 0.0001,
    end_value: float = 0.1,
    total_steps: int = 100,
    start_frac: float = 0.0,
    end_frac: float = 1.0,
    schedule: str = "linear",
) -> optax.Schedule:
    """
    Creates a schedule for loss weights.

    Args:
        start_value: The starting value of the schedule.
        end_value: The ending value of the schedule.
        total_steps: The total number of evaluation steps.
        start_frac: The fraction of total_steps at which ramping should begin.
        end_frac: The fraction of total_steps by which ramping should complete.
        schedule: Type of schedule - "linear" or "cosine".
    """
    delay_steps = int(start_frac * total_steps)
    ramp_steps = int((end_frac - start_frac) * total_steps)
    ramp_steps = max(ramp_steps, 1)

    def schedule_fn(step):
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
