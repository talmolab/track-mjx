"""PPO loss functions for recurrent intention-based imitation learning.

This module provides loss computation for recurrent PPO training with VAE-style
intention networks. It extends the feedforward PPO losses with sequence-aware
processing for the recurrent decoder.

Key components:
- Sequence scan with hidden state reset on episode boundaries
- Standard PPO losses (clipped surrogate, value, entropy)
- VAE losses (KL divergence to Gaussian prior, AR1 temporal smoothness)
"""

from collections.abc import Callable
from typing import Any

import flax
import jax
import jax.numpy as jnp
from brax.training import types
from brax.training.types import Params

from track_mjx.agent.ff_ppo.losses import (
    compute_ar1_temporal_loss,
    compute_gae,
    compute_kl_to_gaussian_prior,
    create_ramp_schedule,
)
from track_mjx.agent.recurrent_ppo.networks import (
    HiddenState,
    RecurrentPPONetworks,
)


@flax.struct.dataclass
class RecurrentPPONetworkParams:
    """Container for recurrent PPO network parameters.

    Attributes:
        policy: Policy network parameters (encoder + recurrent decoder).
        value: Value function network parameters.
    """

    policy: Params
    value: Params


def _extract_initial_hidden(
    stored_hidden: HiddenState | list[HiddenState],
) -> list[HiddenState]:
    """Extract initial hidden state from stored trajectory data.

    The stored hidden has shape [T, B, hidden_size] or list thereof.
    We want the first timestep [B, hidden_size].

    Args:
        stored_hidden: Hidden state(s) stored during rollout.

    Returns:
        Initial hidden state(s) at t=0.
    """
    hidden_list = stored_hidden if isinstance(stored_hidden, list) else [stored_hidden]
    return [h[0] if h.ndim > 2 else h for h in hidden_list]


def compute_recurrent_ppo_loss(
    params: RecurrentPPONetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    step: int,
    recurrent_ppo_network: RecurrentPPONetworks,
    entropy_cost: float = 1e-4,
    latent_kl_weight: float = 1e-3,
    latent_ar1_weight: float = 1e-3,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
    vf_coefficient: float = 0.5,
    latent_kl_schedule: Callable[[int], float] | None = None,
    latent_ar1_schedule: Callable[[int], float] | None = None,
) -> tuple[jnp.ndarray, types.Metrics]:
    """Compute recurrent PPO loss with VAE KL divergence for intention networks.

    Computes the standard PPO clipped surrogate loss plus:
    - Value function MSE loss
    - Entropy bonus for exploration
    - KL divergence loss for VAE latent space
    - AR1 temporal smoothness loss for latent intentions

    Args:
        params: Recurrent PPO network parameters (policy and value).
        normalizer_params: Running statistics for observation normalization.
        data: Transition batch with shape [B, T, ...] (batch-major format).
            Internally swapped to [T, B, ...] for sequence processing.
            Required extra fields:
            - data.extras["state_extras"]["truncation"]
            - data.extras["policy_extras"]["raw_action"]
            - data.extras["policy_extras"]["log_prob"]
            - data.extras["policy_extras"]["initial_policy_hidden"]
        rng: JAX random key.
        step: Current training step (for KL schedule).
        recurrent_ppo_network: Recurrent PPO network container.
        entropy_cost: Entropy bonus coefficient (higher = more exploration).
        latent_kl_weight: Weight for KL divergence to standard Gaussian prior.
        latent_ar1_weight: Weight for AR1 temporal smoothness loss.
        discounting: Discount factor (gamma) for GAE.
        reward_scaling: Multiplier applied to rewards.
        gae_lambda: GAE lambda parameter.
        clipping_epsilon: PPO clipping range for policy ratio.
        normalize_advantage: Whether to normalize advantages to zero mean/unit std.
        vf_coefficient: Coefficient for value function loss.
        latent_kl_schedule: Optional schedule function(step) -> latent_kl_weight.
        latent_ar1_schedule: Optional schedule function(step) -> latent_ar1_weight.

    Returns:
        Tuple of:
            - total_loss: Scalar loss value.
            - metrics: Dict with individual loss components for logging.
    """
    _, policy_key, entropy_key = jax.random.split(rng, 3)
    parametric_action_distribution = (
        recurrent_ppo_network.parametric_action_distribution
    )
    policy_network = recurrent_ppo_network.policy_network
    value_apply = recurrent_ppo_network.value_network.apply

    initial_policy_hidden = data.extras["policy_extras"]["initial_policy_hidden"]

    # Put time dimension first: [B, T, ...] -> [T, B, ...]
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    # initial_policy_hidden has no time axis; keep it batch-major.
    policy_extras = dict(data.extras["policy_extras"])
    policy_extras["initial_policy_hidden"] = initial_policy_hidden
    data = types.Transition(
        observation=data.observation,
        action=data.action,
        reward=data.reward,
        discount=data.discount,
        next_observation=data.next_observation,
        extras={
            "policy_extras": policy_extras,
            "state_extras": data.extras["state_extras"],
        },
    )

    # Extract initial hidden state (stored at first timestep of each trajectory)
    initial_hidden = _extract_initial_hidden(
        data.extras["policy_extras"]["initial_policy_hidden"]
    )

    # Compute done flags for hidden state reset
    # discount is 0 when episode ends
    done = data.discount < 0.5

    # Check for stored policy_rng for deterministic stochastic layer replay
    policy_rng = data.extras["policy_extras"].get("policy_rng")

    # Forward pass through policy network over sequence
    # This scans through time, resetting hidden on done
    policy_logits, latent_mean, latent_logvar, _ = policy_network.apply_sequence(
        normalizer_params,
        params.policy,
        {
            "imitation_target": data.observation["imitation_target"],
            "proprioception": data.observation["proprioception"],
        },
        initial_hidden,
        done,
        policy_key,
        deterministic=False,
        stored_keys=policy_rng,  # Use stored keys if available for deterministic replay
    )

    # Value network is feedforward - apply to all timesteps at once
    # Shape: [T, B, ...] -> [T, B]
    baseline = value_apply(normalizer_params, params.value, data.observation)

    # Get bootstrap value from last next observation
    last_next_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
    bootstrap_value = value_apply(normalizer_params, params.value, last_next_obs)

    rewards = data.reward * reward_scaling
    truncation = data.extras["state_extras"]["truncation"]
    termination = (1 - data.discount) * (1 - truncation)

    # Compute log probs under new policy
    target_action_log_probs = parametric_action_distribution.log_prob(
        policy_logits, data.extras["policy_extras"]["raw_action"]
    )
    behaviour_action_log_probs = data.extras["policy_extras"]["log_prob"]

    # Compute GAE
    vs, advantages = compute_gae(
        truncation=truncation,
        termination=termination,
        rewards=rewards,
        values=baseline,
        bootstrap_value=bootstrap_value,
        lambda_=gae_lambda,
        discount=discounting,
    )

    if normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Importance sampling ratio
    rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)

    # Clipped surrogate loss
    surrogate_loss1 = rho_s * advantages
    surrogate_loss2 = (
        jnp.clip(rho_s, 1 - clipping_epsilon, 1 + clipping_epsilon) * advantages
    )
    policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

    # Value function loss
    v_error = vs - baseline
    v_loss = jnp.mean(v_error * v_error) * 0.5 * vf_coefficient

    # Entropy bonus
    entropy = jnp.mean(
        parametric_action_distribution.entropy(policy_logits, entropy_key)
    )
    entropy_loss = entropy_cost * -entropy

    # KL divergence loss - apply schedules if provided
    current_kl_weight = latent_kl_weight
    current_ar1_weight = latent_ar1_weight
    if latent_kl_schedule is not None:
        current_kl_weight = latent_kl_schedule(step)
    if latent_ar1_schedule is not None:
        current_ar1_weight = latent_ar1_schedule(step)

    # KL divergence to standard Gaussian prior N(0, I)
    kl_gaussian = compute_kl_to_gaussian_prior(latent_mean, latent_logvar)

    # AR1 temporal smoothness loss on latent
    ar1_loss = compute_ar1_temporal_loss(latent_mean, data.discount, truncation)

    # Apply weights
    kl_gaussian_weighted = current_kl_weight * kl_gaussian
    ar1_loss_weighted = current_ar1_weight * ar1_loss
    latent_loss = kl_gaussian_weighted + ar1_loss_weighted

    total_loss = policy_loss + v_loss + entropy_loss + latent_loss

    return total_loss, {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "v_loss": v_loss,
        "total_latent_loss": latent_loss,
        "latent_ar1_loss": ar1_loss_weighted,
        "latent_kl_loss": kl_gaussian_weighted,
        "entropy_loss": entropy_loss,
        "latent_kl_weight": current_kl_weight,
        "latent_ar1_weight": current_ar1_weight,
    }


# Re-export schedule function for convenience
__all__ = [
    "RecurrentPPONetworkParams",
    "compute_recurrent_ppo_loss",
    "create_ramp_schedule",
]
