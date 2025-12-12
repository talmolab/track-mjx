# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Loss functions for distillation training.

The distillation loss consists of:
1. MSE loss between student and teacher actions
2. Autoregressive loss between consecutive latent means of the encoder
3. KL divergence loss between encoder and prior distributions
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


def compute_mse_action_loss(
    student_actions: jnp.ndarray,
    teacher_actions: jnp.ndarray,
) -> jnp.ndarray:
    """Compute MSE loss between student and teacher actions.
    
    Args:
        student_actions: Actions predicted by the student network [T, B, action_dim]
        teacher_actions: Actions from the teacher network [T, B, action_dim]
        
    Returns:
        Scalar MSE loss
    """
    mse = jnp.mean(jnp.square(student_actions - teacher_actions))
    return mse


def compute_autoregressive_loss(
    latent_means: jnp.ndarray,
) -> jnp.ndarray:
    """Compute autoregressive loss between consecutive latent means.
    
    Encourages temporal smoothness in the latent space by minimizing
    the MSE between consecutive latent representations.
    
    Args:
        latent_means: Latent means from encoder [T, B, latent_dim]
        
    Returns:
        Scalar autoregressive loss
    """
    if latent_means.shape[0] <= 1:
        return jnp.array(0.0)
    
    # Get consecutive latent means
    z_prev = latent_means[:-1]  # z_0, ..., z_{T-2}
    z_curr = latent_means[1:]   # z_1, ..., z_{T-1}
    
    # MSE between consecutive latent means
    ar_loss = jnp.mean(jnp.square(z_curr - z_prev))
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
    
    Args:
        encoder_mean: Mean of encoder distribution [T, B, latent_dim]
        encoder_logvar: Log-variance of encoder distribution [T, B, latent_dim]
        prior_mean: Mean of prior distribution [T, B, latent_dim]
        prior_logvar: Log-variance of prior distribution [T, B, latent_dim]
        
    Returns:
        Scalar KL divergence loss
    """
    # KL divergence between two Gaussians:
    # KL = 0.5 * (log(σ_p^2/σ_q^2) + (σ_q^2 + (μ_q - μ_p)^2) / σ_p^2 - 1)
    var_ratio = jnp.exp(encoder_logvar - prior_logvar)  # σ_q^2 / σ_p^2
    mean_diff_sq = jnp.square(encoder_mean - prior_mean) / jnp.exp(prior_logvar)  # (μ_q - μ_p)^2 / σ_p^2
    log_var_diff = prior_logvar - encoder_logvar  # log(σ_p^2) - log(σ_q^2)
    
    kl_loss = 0.5 * jnp.mean(var_ratio + mean_diff_sq - 1 + log_var_diff)
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
    action_mse_weight: float = 1.0,
    autoregressive_weight: float = 1e-3,
    kl_weight: float = 1e-3,
    kl_schedule: Callable | None = None,
    ar_schedule: Callable | None = None,
) -> Tuple[jnp.ndarray, types.Metrics]:
    """Compute the combined distillation loss.
    
    The total loss is:
    L = action_mse_weight * L_action + autoregressive_weight * L_ar + kl_weight * L_kl
    
    Where:
    - L_action: MSE between student and teacher actions
    - L_ar: Autoregressive loss between consecutive encoder latent means
    - L_kl: KL divergence between encoder and prior distributions
    
    Args:
        params: Student network parameters
        normalizer_params: Observation normalizer parameters
        data: Transition data with shape [B, T]
        rng: Random key
        step: Current training step (for scheduling)
        student_network: Student policy network (FeedForwardNetwork)
        teacher_policy_fn: Teacher policy function
        teacher_params: Teacher policy parameters (frozen)
        action_mse_weight: Weight for action MSE loss
        autoregressive_weight: Weight for autoregressive loss
        kl_weight: Weight for KL divergence loss
        kl_schedule: Optional schedule function for KL weight
        ar_schedule: Optional schedule function for autoregressive weight
        
    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    _, student_key, teacher_key = jax.random.split(rng, 3)
    
    # Put the time dimension first: [B, T] -> [T, B]
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
    
    # Get student outputs
    student_logits, encoder_mean, encoder_logvar, prior_mean, prior_logvar = student_network.apply(
        normalizer_params, params.policy, data.observation, student_key
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
    action_loss = compute_mse_action_loss(student_actions, teacher_actions)
    ar_loss = compute_autoregressive_loss(encoder_mean)
    kl_loss = compute_encoder_prior_kl_loss(
        encoder_mean, encoder_logvar, prior_mean, prior_logvar
    )
    
    # Apply schedules if provided
    current_kl_weight = kl_weight
    current_ar_weight = autoregressive_weight
    if kl_schedule is not None:
        current_kl_weight = kl_schedule(step)
    if ar_schedule is not None:
        current_ar_weight = ar_schedule(step)
    
    # Compute weighted total loss
    total_loss = (
        action_mse_weight * action_loss +
        current_ar_weight * ar_loss +
        current_kl_weight * kl_loss
    )
    
    metrics = {
        "total_loss": total_loss,
        "action_mse_loss": action_loss,
        "autoregressive_loss": ar_loss,
        "kl_loss": kl_loss,
        "kl_weight": current_kl_weight,
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
            cosine_value = start_value + 0.5 * (end_value - start_value) * (1 - jnp.cos(jnp.pi * progress))
            return jnp.where(is_delayed, start_value, cosine_value)
        else:
            raise ValueError(f"schedule must be 'linear' or 'cosine', not {schedule}")

    return schedule_fn
