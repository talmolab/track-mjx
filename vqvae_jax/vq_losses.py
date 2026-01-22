"""PPO loss functions for VQ-VAE intention-based imitation learning.

This module provides loss computation for PPO training with VQ-VAE
intention networks, including:
- Generalized Advantage Estimation (GAE)
- Clipped surrogate policy loss
- Value function loss
- Entropy bonus
- VQ-VAE commitment loss and codebook loss (replaces KL divergence)
- Codebook health metrics (perplexity, utilization)

The key difference from standard VAE losses is replacing KL divergence
with the VQ-VAE auxiliary losses that encourage encoder commitment to
codebook entries and codebook entries to track encoder outputs.
"""

from collections.abc import Callable
from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax
from brax.training import types
from brax.training.types import Params


@flax.struct.dataclass
class PPONetworkParams:
    """Container for PPO network parameters.

    Attributes:
        policy: Policy network parameters (encoder + quantizer + decoder).
        value: Value function network parameters.
    """

    policy: Params
    value: Params


def compute_gae(
    truncation: jnp.ndarray,
    termination: jnp.ndarray,
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    bootstrap_value: jnp.ndarray,
    lambda_: float = 1.0,
    discount: float = 0.99,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Generalized Advantage Estimation (GAE).

    GAE provides a variance-bias tradeoff for advantage estimation controlled
    by lambda_. When lambda_=0, this reduces to 1-step TD. When lambda_=1,
    this uses the full Monte Carlo return.

    Args:
        truncation: Truncation signal, shape [T, B]. 1 if episode was truncated.
        termination: Termination signal, shape [T, B]. 1 if episode ended.
        rewards: Rewards, shape [T, B].
        values: Value function estimates V(s_t), shape [T, B].
        bootstrap_value: Value estimate at time T for bootstrapping, shape [B].
        lambda_: GAE lambda parameter (0=TD(0), 1=Monte Carlo).
        discount: Discount factor (gamma).

    Returns:
        Tuple of:
            - vs: Value targets for training, shape [T, B].
            - advantages: GAE advantages, shape [T, B].
    """
    truncation_mask = 1 - truncation

    values_t_plus_1 = jnp.concatenate(
        [values[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0
    )
    deltas = rewards + discount * (1 - termination) * values_t_plus_1 - values
    deltas *= truncation_mask

    acc = jnp.zeros_like(bootstrap_value)

    def compute_vs_minus_v_xs(carry, target_t):
        lambda_, acc = carry
        truncation_mask, delta, termination = target_t
        acc = delta + discount * (1 - termination) * truncation_mask * lambda_ * acc
        return (lambda_, acc), acc

    (_, _), vs_minus_v_xs = jax.lax.scan(
        compute_vs_minus_v_xs,
        (lambda_, acc),
        (truncation_mask, deltas, termination),
        length=int(truncation_mask.shape[0]),
        reverse=True,
    )

    vs = jnp.add(vs_minus_v_xs, values)

    vs_t_plus_1 = jnp.concatenate([vs[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
    advantages = (
        rewards + discount * (1 - termination) * vs_t_plus_1 - values
    ) * truncation_mask

    return jax.lax.stop_gradient(vs), jax.lax.stop_gradient(advantages)


def compute_vq_loss(
    z_e: jnp.ndarray,
    z_q: jnp.ndarray,
    commitment_cost: float = 0.25,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute VQ-VAE auxiliary losses with proper gradient routing.

    The VQ-VAE has two auxiliary losses:
    1. Commitment loss: Encourages encoder to commit to codebook entries
       - Gradients flow to encoder only (codebook stopped)
    2. Codebook loss: Moves codebook entries toward encoder outputs
       - Gradients flow to codebook only (encoder stopped)

    Args:
        z_e: Encoder output (continuous), shape [..., latent_dim].
        z_q: Quantized vectors (from codebook lookup), shape [..., latent_dim].
        commitment_cost: Weight for commitment loss (beta in paper).

    Returns:
        Tuple of:
            - vq_loss: Combined commitment + codebook loss.
            - commitment_loss: Encoder commitment term.
            - codebook_loss: Codebook update term.
    """
    # Commitment loss: encoder learns to commit to codebook entries
    # Gradient flows to z_e only (z_q stopped)
    commitment_loss = jnp.mean((z_e - jax.lax.stop_gradient(z_q)) ** 2)

    # Codebook loss: codebook moves toward encoder outputs
    # Gradient flows to z_q only (z_e stopped)
    codebook_loss = jnp.mean((jax.lax.stop_gradient(z_e) - z_q) ** 2)

    vq_loss = commitment_cost * commitment_loss + codebook_loss

    return vq_loss, commitment_loss, codebook_loss


def compute_codebook_metrics(
    indices: jnp.ndarray,
    num_codes: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute codebook health metrics.

    These metrics help diagnose codebook collapse (when only a few
    codes are used) and monitor training health.

    Args:
        indices: Codebook indices, shape [...].
        num_codes: Total number of codebook entries.

    Returns:
        Tuple of:
            - perplexity: Effective number of codes used (exp(entropy)).
                Higher is better; max is num_codes.
            - utilization: Fraction of codes used at least once.
            - codes_used: Count of unique codes in batch.
    """
    # Flatten indices
    flat_indices = indices.reshape(-1)

    # Compute code usage histogram (one-hot then average)
    code_one_hot = jax.nn.one_hot(flat_indices, num_codes)  # [N, K]
    code_probs = jnp.mean(code_one_hot, axis=0)  # [K] - usage frequency

    # Perplexity: exp(entropy of code usage distribution)
    # Higher perplexity = more uniform usage = better
    code_entropy = -jnp.sum(
        jnp.where(code_probs > 0, code_probs * jnp.log(code_probs + 1e-10), 0.0)
    )
    perplexity = jnp.exp(code_entropy)

    # Utilization: fraction of codes used
    codes_used = jnp.sum(code_probs > 0)
    utilization = codes_used / num_codes

    return perplexity, utilization, codes_used


def compute_ce_stickiness_cost(
    z_e: jnp.ndarray,
    indices: jnp.ndarray,
    codebook: jnp.ndarray,
    valid_mask: jnp.ndarray,
    temperature: float = 1.0,
) -> tuple[jnp.ndarray, dict]:
    """Cross-entropy loss encouraging code persistence.

    Encourages z_e[t+1] to remain closest to the same codebook entry
    that z_e[t] was assigned to. This operates directly in code space
    rather than continuous embedding space, respecting the Voronoi
    tessellation of the codebook.

    Args:
        z_e: Encoder outputs with shape [T, B, D].
        indices: Hard code assignments with shape [T, B].
        codebook: Codebook embeddings with shape [K, D].
        valid_mask: Binary mask for valid transitions [T-1, B].
        temperature: Softmax temperature (lower = sharper).

    Returns:
        Tuple of (ce_stickiness_loss, metrics_dict).
    """
    num_codes = codebook.shape[0]

    # Step 1: Get consecutive timesteps
    z_e_curr = z_e[1:]  # [T-1, B, D] - encoder outputs at t+1
    targets = indices[:-1]  # [T-1, B] - code indices at t (our target)

    # Step 2: Compute squared distances from z_e[t+1] to all codes
    # Stop gradient on codebook to prevent this loss from moving codes
    codebook_sg = jax.lax.stop_gradient(codebook)

    # [T-1, B, 1, D] - [1, 1, K, D] -> [T-1, B, K, D] -> sum -> [T-1, B, K]
    sq_distances = jnp.sum(
        jnp.square(z_e_curr[:, :, None, :] - codebook_sg[None, None, :, :]),
        axis=-1,
    )

    # Step 3: Convert to logits (negate distances, scale by temperature)
    logits = -sq_distances / temperature  # [T-1, B, K]

    # Step 4: Cross-entropy loss
    # Target is the code from the previous timestep
    ce_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits, targets
    )  # [T-1, B]

    # Step 5: Mask and average
    num_valid = jnp.sum(valid_mask) + 1e-8
    ce_stickiness_loss = jnp.sum(ce_loss * valid_mask) / num_valid

    # Metrics for monitoring
    probs = jax.nn.softmax(logits, axis=-1)  # [T-1, B, K]
    target_one_hot = jax.nn.one_hot(targets, num_codes)  # [T-1, B, K]
    prob_of_target = jnp.sum(probs * target_one_hot, axis=-1)  # [T-1, B]
    mean_prob_of_prev_code = jnp.sum(prob_of_target * valid_mask) / num_valid

    metrics = {
        "ce_stickiness_loss": ce_stickiness_loss,
        "prob_of_prev_code": mean_prob_of_prev_code,
    }

    return ce_stickiness_loss, metrics


def compute_vq_ppo_loss(
    params: PPONetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    step: int,
    ppo_network: Any,
    entropy_cost: float = 1e-4,
    commitment_cost: float = 0.25,
    codebook_loss_weight: float = 1.0,
    ce_stickiness_cost: float = 0.0,
    ce_stickiness_temperature: float = 1.0,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
    vq_loss_schedule: Callable[[int], float] | None = None,
) -> tuple[jnp.ndarray, types.Metrics]:
    """Compute PPO loss with VQ-VAE auxiliary losses.

    Computes the standard PPO clipped surrogate loss plus:
    - Value function MSE loss
    - Entropy bonus for exploration
    - VQ-VAE commitment loss (encoder commits to codebook)
    - VQ-VAE codebook loss (codebook tracks encoder)
    - Cross-entropy stickiness loss (directly encourages code persistence)

    Unlike the VAE version, there is no KL divergence loss. The
    commitment and codebook losses serve as regularization.

    Args:
        params: PPO network parameters (policy and value).
        normalizer_params: Running statistics for observation normalization.
        data: Transition batch with shape [B, T]. Required extra fields:
            - data.extras["state_extras"]["truncation"]
            - data.extras["policy_extras"]["raw_action"]
            - data.extras["policy_extras"]["log_prob"]
        rng: JAX random key.
        step: Current training step (for optional VQ loss schedule).
        ppo_network: PPO network container with policy, value, and distribution.
        entropy_cost: Entropy bonus coefficient.
        commitment_cost: Weight for commitment loss (beta).
        codebook_loss_weight: Weight for codebook loss.
        ce_stickiness_cost: Weight for cross-entropy stickiness loss (code space).
        ce_stickiness_temperature: Temperature for CE stickiness softmax.
        discounting: Discount factor (gamma) for GAE.
        reward_scaling: Multiplier applied to rewards.
        gae_lambda: GAE lambda parameter.
        clipping_epsilon: PPO clipping range for policy ratio.
        normalize_advantage: Whether to normalize advantages.
        vq_loss_schedule: Optional schedule function(step) -> vq_weight.

    Returns:
        Tuple of:
            - total_loss: Scalar loss value.
            - metrics: Dict with individual loss components for logging.
    """
    _, policy_key, entropy_key = jax.random.split(rng, 3)
    parametric_action_distribution = ppo_network.parametric_action_distribution
    policy_apply = ppo_network.policy_network.apply
    value_apply = ppo_network.value_network.apply

    # Put the time dimension first: [B, T, ...] -> [T, B, ...]
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    # Forward pass through VQ policy
    # Returns: (action_params, z_e, indices)
    policy_logits, z_e, indices = policy_apply(
        normalizer_params, params.policy, data.observation, policy_key
    )

    # Value function
    baseline = value_apply(normalizer_params, params.value, data.observation)
    # Get the last timestep from dict observation (tree_map handles dict structure)
    last_next_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
    bootstrap_value = value_apply(normalizer_params, params.value, last_next_obs)

    # Reconstruct z_q from current codebook for loss computation
    # This ensures gradients flow to the current codebook parameters
    codebook = params.policy["params"]["quantizer"]["embeddings"]
    z_q = codebook[indices]  # [..., latent_dim]

    # VQ-VAE auxiliary losses
    vq_loss, commitment_loss, codebook_loss = compute_vq_loss(z_e, z_q, commitment_cost)

    # Apply schedule if provided
    vq_weight = 1.0
    if vq_loss_schedule is not None:
        vq_weight = vq_loss_schedule(step)

    scaled_vq_loss = vq_weight * (
        commitment_cost * commitment_loss + codebook_loss_weight * codebook_loss
    )

    # Codebook health metrics
    num_codes = codebook.shape[0]
    perplexity, utilization, codes_used = compute_codebook_metrics(indices, num_codes)

    # Standard PPO loss computation
    rewards = data.reward * reward_scaling
    truncation = data.extras["state_extras"]["truncation"]
    termination = (1 - data.discount) * (1 - truncation)

    # Compute valid mask and transition rate for temporal metrics
    # Shapes after axis swap: z_e [T, B, D], indices [T, B], discount [T, B]
    if z_e.shape[0] > 1:
        # Mask: valid if episode continues (not done AND not truncated)
        valid_mask = data.discount[:-1] * (1 - truncation[:-1])  # [T-1, B]
        num_valid = jnp.sum(valid_mask) + 1e-8

        # Transition rate metric (monitoring only, no gradient)
        indices_prev = indices[:-1]  # [T-1, B]
        indices_curr = indices[1:]  # [T-1, B]
        code_changed = (indices_curr != indices_prev).astype(jnp.float32)
        transition_rate = jnp.sum(code_changed * valid_mask) / num_valid
    else:
        transition_rate = jnp.array(0.0)
        valid_mask = jnp.array(0.0)  # Placeholder for single-step case

    # Cross-entropy stickiness loss (operates in code space)
    # Directly encourages code persistence by penalizing code boundary crossings
    if z_e.shape[0] > 1 and ce_stickiness_cost > 0.0:
        ce_stickiness_loss, ce_stickiness_metrics = compute_ce_stickiness_cost(
            z_e=z_e,
            indices=indices,
            codebook=codebook,
            valid_mask=valid_mask,
            temperature=ce_stickiness_temperature,
        )
        prob_of_prev_code = ce_stickiness_metrics["prob_of_prev_code"]
    else:
        ce_stickiness_loss = jnp.array(0.0)
        prob_of_prev_code = jnp.array(0.0)

    scaled_ce_stickiness_loss = ce_stickiness_cost * ce_stickiness_loss

    target_action_log_probs = parametric_action_distribution.log_prob(
        policy_logits, data.extras["policy_extras"]["raw_action"]
    )
    behaviour_action_log_probs = data.extras["policy_extras"]["log_prob"]

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

    rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)

    surrogate_loss1 = rho_s * advantages
    surrogate_loss2 = (
        jnp.clip(rho_s, 1 - clipping_epsilon, 1 + clipping_epsilon) * advantages
    )
    policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

    # Value function loss
    v_error = vs - baseline
    v_loss = jnp.mean(v_error * v_error) * 0.5 * 0.5

    # Entropy bonus
    entropy = jnp.mean(
        parametric_action_distribution.entropy(policy_logits, entropy_key)
    )
    entropy_loss = entropy_cost * -entropy

    # Total loss
    total_loss = (
        policy_loss + v_loss + entropy_loss + scaled_vq_loss + scaled_ce_stickiness_loss
    )

    return total_loss, {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "v_loss": v_loss,
        "entropy_loss": entropy_loss,
        # VQ-VAE specific metrics
        "vq_loss": vq_loss,
        "commitment_loss": commitment_loss,
        "codebook_loss": codebook_loss,
        "scaled_vq_loss": scaled_vq_loss,
        "vq_weight": vq_weight,
        # Codebook health metrics
        "perplexity": perplexity,
        "codebook_utilization": utilization,
        "codes_used": codes_used,
        # Cross-entropy stickiness metrics
        "ce_stickiness_loss": ce_stickiness_loss,
        "scaled_ce_stickiness_loss": scaled_ce_stickiness_loss,
        "prob_of_prev_code": prob_of_prev_code,
        "transition_rate": transition_rate,
    }


def create_vq_schedule(
    max_value: float = 1.0,
    min_value: float = 0.1,
    warmup_steps: int = 100,
    ramp_steps: int = 500,
) -> Callable[[int], float]:
    """Create a warmup schedule for VQ loss weight.

    Starts at min_value and ramps up to max_value, which can help
    stabilize early training when the codebook is still being learned.

    Args:
        max_value: Final VQ loss weight.
        min_value: Initial VQ loss weight during warmup.
        warmup_steps: Steps to hold at min_value.
        ramp_steps: Steps to ramp from min to max.

    Returns:
        A function mapping step -> vq_weight.
    """

    def schedule_fn(step: int) -> float:
        step = jnp.asarray(step, dtype=jnp.float32)
        progress = jnp.clip((step - warmup_steps) / ramp_steps, 0.0, 1.0)
        is_warmup = step < warmup_steps
        return jnp.where(
            is_warmup, min_value, min_value + progress * (max_value - min_value)
        )

    return schedule_fn
