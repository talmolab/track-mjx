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
    z_q: jnp.ndarray | None = None,
    commitment_cost: float = 0.25,
    all_z_q: tuple[jnp.ndarray, ...] | None = None,
    all_residuals: tuple[jnp.ndarray, ...] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute VQ-VAE auxiliary losses with proper gradient routing.

    Supports both single-level (z_q) and multi-level RVQ (all_z_q + all_residuals).
    When multi-level inputs are provided, per-depth losses are averaged with 1/D scaling.

    Args:
        z_e: Encoder output (continuous), shape [..., latent_dim].
        z_q: Single-level quantized vectors, shape [..., latent_dim].
            Used when all_z_q is None (backward compat).
        commitment_cost: Weight for commitment loss (beta in paper).
        all_z_q: Tuple of D quantized vectors (no STE), each [..., latent_dim].
        all_residuals: Tuple of D+1 residuals. residuals[d] is the input
            to level d (residuals[0] = z_e).

    Returns:
        Tuple of (vq_loss, commitment_loss, codebook_loss).
    """
    if all_z_q is not None and all_residuals is not None:
        # Multi-depth RVQ loss with 1/D scaling
        D = len(all_z_q)
        scale = 1.0 / D
        total_commitment = jnp.array(0.0)
        total_codebook = jnp.array(0.0)

        for d in range(D):
            r_d = all_residuals[d]  # Input to level d
            z_q_d = all_z_q[d]  # Quantized output of level d
            commitment_d = jnp.mean((r_d - jax.lax.stop_gradient(z_q_d)) ** 2)
            codebook_d = jnp.mean((jax.lax.stop_gradient(r_d) - z_q_d) ** 2)
            total_commitment = total_commitment + scale * commitment_d
            total_codebook = total_codebook + scale * codebook_d

        vq_loss = commitment_cost * total_commitment + total_codebook
        return vq_loss, total_commitment, total_codebook
    else:
        # Single-level backward compat
        assert z_q is not None, "Must provide z_q or (all_z_q, all_residuals)"
        commitment_loss = jnp.mean((z_e - jax.lax.stop_gradient(z_q)) ** 2)
        codebook_loss = jnp.mean((jax.lax.stop_gradient(z_e) - z_q) ** 2)
        vq_loss = commitment_cost * commitment_loss + codebook_loss
        return vq_loss, commitment_loss, codebook_loss


def _compute_single_codebook_metrics(
    indices: jnp.ndarray,
    num_codes: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute codebook health metrics for a single level.

    Args:
        indices: Codebook indices, shape [...].
        num_codes: Total number of codebook entries.

    Returns:
        Tuple of (perplexity, utilization, codes_used).
    """
    flat_indices = indices.reshape(-1)
    code_one_hot = jax.nn.one_hot(flat_indices, num_codes)
    code_probs = jnp.mean(code_one_hot, axis=0)

    code_entropy = -jnp.sum(
        jnp.where(code_probs > 0, code_probs * jnp.log(code_probs + 1e-10), 0.0)
    )
    perplexity = jnp.exp(code_entropy)

    codes_used = jnp.sum(code_probs > 0)
    utilization = codes_used / num_codes

    return perplexity, utilization, codes_used


def compute_codebook_metrics(
    indices: jnp.ndarray | tuple[jnp.ndarray, ...],
    num_codes: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute codebook health metrics.

    Supports both single-level indices (array) and multi-level (tuple).
    For multi-level, returns metrics for level 0 (primary/coarse).

    Args:
        indices: Codebook indices. Single array [...] or tuple of D arrays.
        num_codes: Total number of codebook entries per level.

    Returns:
        Tuple of (perplexity, utilization, codes_used) for the primary level.
    """
    if isinstance(indices, tuple):
        return _compute_single_codebook_metrics(indices[0], num_codes)
    return _compute_single_codebook_metrics(indices, num_codes)


def compute_codebook_metrics_per_depth(
    all_indices: tuple[jnp.ndarray, ...],
    num_codes: int,
) -> dict[str, jnp.ndarray]:
    """Compute codebook health metrics for each RVQ depth level.

    Args:
        all_indices: Tuple of D index arrays, each shape [...].
        num_codes: Number of codebook entries per level.

    Returns:
        Dict with per-depth metrics: perplexity_d{i}, utilization_d{i},
        codes_used_d{i} for each depth i.
    """
    metrics = {}
    for d, indices_d in enumerate(all_indices):
        perp, util, used = _compute_single_codebook_metrics(indices_d, num_codes)
        metrics[f"perplexity_d{d}"] = perp
        metrics[f"utilization_d{d}"] = util
        metrics[f"codes_used_d{d}"] = used
    return metrics


def compute_codebook_entropy_loss(
    z_e: jnp.ndarray,
    codebooks: tuple[jnp.ndarray, ...],
    all_residuals: tuple[jnp.ndarray, ...],
    temperature: float = 1.0,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Compute differentiable codebook entropy regularization.

    Uses soft code assignments via softmax(-distances/temperature) to compute
    a differentiable Shannon entropy of the batch-averaged code distribution.
    Maximizing entropy encourages uniform codebook usage.

    Args:
        z_e: Encoder output, shape [..., latent_dim].
        codebooks: Tuple of D codebook arrays, each [K, latent_dim].
        all_residuals: Tuple of D+1 residuals. residuals[d] is the input
            to codebook level d.
        temperature: Softmax temperature. Lower = sharper assignments.

    Returns:
        Tuple of (negative_mean_entropy, metrics_dict).
        negative_mean_entropy is averaged across depths with 1/D scaling.
    """
    D = len(codebooks)
    scale = 1.0 / D
    total_neg_entropy = jnp.array(0.0)
    metrics = {}

    for d in range(D):
        r_d = all_residuals[d]  # Input to level d: [..., latent_dim]
        cb_d = codebooks[d]  # [K, latent_dim]

        # Squared distances: [..., K]
        r_sq = jnp.sum(r_d**2, axis=-1, keepdims=True)  # [..., 1]
        cb_sq = jnp.sum(cb_d**2, axis=-1)  # [K]
        cross = jnp.matmul(r_d, cb_d.T)  # [..., K]
        sq_distances = r_sq + cb_sq - 2 * cross  # [..., K]

        # Soft assignments via softmax
        soft_assignments = jax.nn.softmax(-sq_distances / temperature, axis=-1)

        # Batch-averaged soft distribution
        flat_soft = soft_assignments.reshape(-1, soft_assignments.shape[-1])
        avg_distribution = jnp.mean(flat_soft, axis=0)  # [K]

        # Shannon entropy: H = -sum(p * log(p))
        entropy_d = -jnp.sum(
            jnp.where(
                avg_distribution > 1e-10,
                avg_distribution * jnp.log(avg_distribution + 1e-10),
                0.0,
            )
        )

        total_neg_entropy = total_neg_entropy + scale * (-entropy_d)
        metrics[f"soft_code_entropy_d{d}"] = entropy_d

    return total_neg_entropy, metrics


def _compute_ce_stickiness_single(
    z_e: jnp.ndarray,
    indices: jnp.ndarray,
    codebook: jnp.ndarray,
    valid_mask: jnp.ndarray,
    temperature: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """CE stickiness for a single codebook level.

    Args:
        z_e: Input vectors with shape [T, B, D].
        indices: Code assignments with shape [T, B].
        codebook: Codebook [K, D].
        valid_mask: [T-1, B].
        temperature: Softmax temperature.

    Returns:
        (loss, prob_of_prev_code) scalars.
    """
    num_codes = codebook.shape[0]
    z_e_curr = z_e[1:]
    targets = indices[:-1]

    codebook_sg = jax.lax.stop_gradient(codebook)
    sq_distances = jnp.sum(
        jnp.square(z_e_curr[:, :, None, :] - codebook_sg[None, None, :, :]),
        axis=-1,
    )
    logits = -sq_distances / temperature

    ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    num_valid = jnp.sum(valid_mask) + 1e-8
    loss = jnp.sum(ce_loss * valid_mask) / num_valid

    probs = jax.nn.softmax(logits, axis=-1)
    target_one_hot = jax.nn.one_hot(targets, num_codes)
    prob_of_target = jnp.sum(probs * target_one_hot, axis=-1)
    mean_prob = jnp.sum(prob_of_target * valid_mask) / num_valid

    return loss, mean_prob


def compute_ce_stickiness_cost(
    z_e: jnp.ndarray,
    indices: jnp.ndarray | tuple[jnp.ndarray, ...],
    codebook: jnp.ndarray | tuple[jnp.ndarray, ...],
    valid_mask: jnp.ndarray,
    temperature: float = 1.0,
    all_residuals: tuple[jnp.ndarray, ...] | None = None,
) -> tuple[jnp.ndarray, dict]:
    """Cross-entropy loss encouraging code persistence.

    For multi-level RVQ, applies CE stickiness to each level using that
    level's residual input and codebook, then averages with 1/D scaling.

    Args:
        z_e: Encoder outputs [T, B, D].
        indices: Code assignments [T, B] or tuple of D arrays.
        codebook: Codebook [K, D] or tuple of D codebooks.
        valid_mask: [T-1, B].
        temperature: Softmax temperature.
        all_residuals: Tuple of D+1 residuals for multi-level.
            residuals[d] is input to level d (shape [T, B, D]).

    Returns:
        Tuple of (ce_stickiness_loss, metrics_dict).
    """
    if isinstance(indices, tuple) and isinstance(codebook, tuple):
        # Multi-level: apply per-level and average
        D = len(indices)
        scale = 1.0 / D
        total_loss = jnp.array(0.0)
        total_prob = jnp.array(0.0)
        metrics = {}

        for d in range(D):
            # Use residual input for each level
            if all_residuals is not None and d < len(all_residuals):
                input_d = all_residuals[d]  # [T, B, D]
            else:
                input_d = z_e  # Fallback for depth=0

            loss_d, prob_d = _compute_ce_stickiness_single(
                input_d, indices[d], codebook[d], valid_mask, temperature
            )
            total_loss = total_loss + scale * loss_d
            total_prob = total_prob + scale * prob_d
            metrics[f"ce_stickiness_loss_d{d}"] = loss_d
            metrics[f"prob_of_prev_code_d{d}"] = prob_d

        metrics["ce_stickiness_loss"] = total_loss
        metrics["prob_of_prev_code"] = total_prob
        return total_loss, metrics
    else:
        # Single-level backward compat
        loss, prob = _compute_ce_stickiness_single(
            z_e, indices, codebook, valid_mask, temperature
        )
        return loss, {
            "ce_stickiness_loss": loss,
            "prob_of_prev_code": prob,
        }


def _has_any_stickiness(stickiness_bias) -> bool:
    """Check if any level has nonzero stickiness bias."""
    if isinstance(stickiness_bias, (int, float)):
        return stickiness_bias > 0
    return any(b > 0 for b in stickiness_bias)


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
    stickiness_bias: float | tuple[float, ...] = 0.0,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
    vq_loss_schedule: Callable[[int], float] | None = None,
    rvq_depth: int = 1,
    codebook_entropy_weight: float = 0.0,
    codebook_entropy_temperature: float = 1.0,
) -> tuple[jnp.ndarray, types.Metrics]:
    """Compute PPO loss with VQ-VAE auxiliary losses.

    Supports multi-depth RVQ. Codebooks are accessed from the param tree at
    params.policy["params"]["quantizer"]["codebooks_{d}"]["embeddings"].

    Args:
        params: PPO network parameters (policy and value).
        normalizer_params: Running statistics for observation normalization.
        data: Transition batch with shape [B, T].
        rng: JAX random key.
        step: Current training step.
        ppo_network: PPO network container.
        entropy_cost: Entropy bonus coefficient.
        commitment_cost: Weight for commitment loss (beta).
        codebook_loss_weight: Weight for codebook loss.
        ce_stickiness_cost: Weight for CE stickiness loss.
        ce_stickiness_temperature: Temperature for CE stickiness softmax.
        stickiness_bias: Per-level stickiness bias. Float or tuple.
        discounting: Discount factor (gamma).
        reward_scaling: Reward multiplier.
        gae_lambda: GAE lambda.
        clipping_epsilon: PPO clipping range.
        normalize_advantage: Whether to normalize advantages.
        vq_loss_schedule: Optional schedule function(step) -> vq_weight.
        rvq_depth: Number of RVQ depth levels.

    Returns:
        Tuple of (total_loss, metrics_dict).
    """
    _, policy_key, entropy_key = jax.random.split(rng, 3)
    parametric_action_distribution = ppo_network.parametric_action_distribution
    policy_network = ppo_network.policy_network
    value_apply = ppo_network.value_network.apply

    # Put the time dimension first: [B, T, ...] -> [T, B, ...]
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    # Forward pass through VQ policy
    # all_indices is a tuple of D arrays, each [T, B]
    has_stickiness = _has_any_stickiness(stickiness_bias)
    if has_stickiness and hasattr(policy_network, "apply_temporal"):
        truncation = data.extras["state_extras"]["truncation"]
        discount = data.discount
        continues = discount * (1 - truncation)
        episode_mask = jnp.concatenate(
            [jnp.zeros((1, continues.shape[1])), continues[:-1]], axis=0
        )
        policy_logits, z_e, all_indices = policy_network.apply_temporal(
            normalizer_params,
            params.policy,
            data.observation,
            episode_mask,
            proprio_noise_key=policy_key,
        )
    else:
        policy_logits, z_e, all_indices = policy_network.apply(
            normalizer_params, params.policy, data.observation, policy_key
        )

    # Value function
    baseline = value_apply(normalizer_params, params.value, data.observation)
    last_next_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
    bootstrap_value = value_apply(normalizer_params, params.value, last_next_obs)

    # Extract codebooks from param tree and reconstruct z_q per level
    quantizer_params = params.policy["params"]["quantizer"]
    codebooks = tuple(
        quantizer_params[f"codebooks_{d}"]["embeddings"] for d in range(rvq_depth)
    )

    # Reconstruct all_z_q and all_residuals for loss computation
    all_z_q = tuple(codebooks[d][all_indices[d]] for d in range(rvq_depth))

    # Reconstruct residuals: residuals[0] = z_e, residuals[d] = residuals[d-1] - z_q[d-1]
    all_residuals = [z_e]
    residual = z_e
    for d in range(rvq_depth):
        residual = residual - jax.lax.stop_gradient(all_z_q[d])
        all_residuals.append(residual)
    all_residuals = tuple(all_residuals)

    # VQ-VAE auxiliary losses (multi-depth with 1/D scaling)
    vq_loss, commitment_loss, codebook_loss = compute_vq_loss(
        z_e=z_e,
        commitment_cost=commitment_cost,
        all_z_q=all_z_q,
        all_residuals=all_residuals,
    )

    # Apply schedule if provided
    vq_weight = 1.0
    if vq_loss_schedule is not None:
        vq_weight = vq_loss_schedule(step)

    scaled_vq_loss = vq_weight * (
        commitment_cost * commitment_loss + codebook_loss_weight * codebook_loss
    )

    # Codebook health metrics (primary level for backward compat)
    num_codes = codebooks[0].shape[0]
    perplexity, utilization, codes_used = compute_codebook_metrics(
        all_indices, num_codes
    )

    # Per-depth metrics
    depth_metrics = compute_codebook_metrics_per_depth(all_indices, num_codes)

    # Codebook entropy regularization (differentiable)
    if codebook_entropy_weight > 0.0:
        neg_entropy, entropy_metrics = compute_codebook_entropy_loss(
            z_e=z_e,
            codebooks=codebooks,
            all_residuals=all_residuals,
            temperature=codebook_entropy_temperature,
        )
        scaled_entropy_reg = codebook_entropy_weight * neg_entropy
    else:
        scaled_entropy_reg = jnp.array(0.0)
        entropy_metrics = {}

    # Standard PPO loss computation
    rewards = data.reward * reward_scaling
    truncation = data.extras["state_extras"]["truncation"]
    termination = (1 - data.discount) * (1 - truncation)

    # Transition rate using primary (L0) indices
    primary_indices = all_indices[0]
    if z_e.shape[0] > 1:
        valid_mask = data.discount[:-1] * (1 - truncation[:-1])
        num_valid = jnp.sum(valid_mask) + 1e-8

        indices_prev = primary_indices[:-1]
        indices_curr = primary_indices[1:]
        code_changed = (indices_curr != indices_prev).astype(jnp.float32)
        transition_rate = jnp.sum(code_changed * valid_mask) / num_valid
    else:
        transition_rate = jnp.array(0.0)
        valid_mask = jnp.array(0.0)

    # Cross-entropy stickiness loss
    if z_e.shape[0] > 1 and ce_stickiness_cost > 0.0:
        ce_stickiness_loss, ce_stickiness_metrics = compute_ce_stickiness_cost(
            z_e=z_e,
            indices=all_indices if rvq_depth > 1 else all_indices[0],
            codebook=codebooks if rvq_depth > 1 else codebooks[0],
            valid_mask=valid_mask,
            temperature=ce_stickiness_temperature,
            all_residuals=all_residuals if rvq_depth > 1 else None,
        )
        prob_of_prev_code = ce_stickiness_metrics["prob_of_prev_code"]
    else:
        ce_stickiness_loss = jnp.array(0.0)
        ce_stickiness_metrics = {}
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

    v_error = vs - baseline
    v_loss = jnp.mean(v_error * v_error) * 0.5 * 0.5

    entropy = jnp.mean(
        parametric_action_distribution.entropy(policy_logits, entropy_key)
    )
    entropy_loss = entropy_cost * -entropy

    total_loss = (
        policy_loss
        + v_loss
        + entropy_loss
        + scaled_vq_loss
        + scaled_ce_stickiness_loss
        + scaled_entropy_reg
    )

    metrics = {
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
        # Codebook health metrics (primary level)
        "perplexity": perplexity,
        "codebook_utilization": utilization,
        "codes_used": codes_used,
        # Stickiness metrics
        "ce_stickiness_loss": ce_stickiness_loss,
        "scaled_ce_stickiness_loss": scaled_ce_stickiness_loss,
        "prob_of_prev_code": prob_of_prev_code,
        "transition_rate": transition_rate,
        # Codebook entropy regularization
        "scaled_codebook_entropy_reg": scaled_entropy_reg,
    }

    # Add per-depth metrics
    metrics.update(depth_metrics)

    # Add per-depth CE stickiness metrics
    for key, val in ce_stickiness_metrics.items():
        if key not in metrics:
            metrics[key] = val

    # Add per-depth soft entropy metrics
    metrics.update(entropy_metrics)

    return total_loss, metrics


def reinit_dead_codes(
    policy_params: dict,
    z_e_samples: jnp.ndarray,
    all_indices: tuple[jnp.ndarray, ...],
    num_codes: int,
    rvq_depth: int,
    threshold: float = 0.01,
    noise_scale: float = 0.01,
    rng: jnp.ndarray | None = None,
) -> dict:
    """Reinitialize dead codebook entries from encoder samples.

    Dead codes (usage below threshold) are replaced with randomly sampled
    z_e vectors plus small noise. This prevents permanent codebook collapse.

    Args:
        policy_params: Policy parameter dict (FrozenDict-compatible).
        z_e_samples: Encoder outputs from recent rollout, shape [N, latent_dim].
        all_indices: Tuple of D index arrays from recent rollout.
        num_codes: Number of codes per level.
        rvq_depth: Number of RVQ depth levels.
        threshold: Fraction of uniform usage below which a code is "dead".
        noise_scale: Scale of noise added to reinitialized codes.
        rng: Optional JAX random key for sampling.

    Returns:
        New policy_params dict with dead codes replaced.
    """
    import numpy as np
    from flax.core import freeze, unfreeze
    from flax.core.frozen_dict import FrozenDict

    if rng is None:
        rng = jax.random.PRNGKey(0)

    was_frozen = isinstance(policy_params, FrozenDict)
    params = unfreeze(policy_params)
    flat_z_e = z_e_samples.reshape(-1, z_e_samples.shape[-1])
    n_samples = flat_z_e.shape[0]

    for d in range(rvq_depth):
        indices_d = all_indices[d]
        flat_idx = np.array(indices_d).reshape(-1)
        counts = np.bincount(flat_idx, minlength=num_codes)
        total = counts.sum()
        usage = counts / max(total, 1)

        # Dead = usage below threshold fraction of uniform (1/K)
        uniform_usage = 1.0 / num_codes
        dead_mask = usage < (threshold * uniform_usage)
        dead_indices = np.where(dead_mask)[0]

        if len(dead_indices) == 0:
            continue

        # Sample replacement vectors from z_e
        rng, sample_key = jax.random.split(rng)
        sample_idx = jax.random.randint(sample_key, (len(dead_indices),), 0, n_samples)
        replacements = flat_z_e[sample_idx]

        # Add noise
        rng, noise_key = jax.random.split(rng)
        noise = jax.random.normal(noise_key, replacements.shape) * noise_scale
        replacements = replacements + noise

        # Update codebook
        cb_key = f"codebooks_{d}"
        embeddings = params["params"]["quantizer"][cb_key]["embeddings"]
        embeddings = embeddings.at[dead_indices].set(replacements)
        params["params"]["quantizer"][cb_key]["embeddings"] = embeddings

    return freeze(params) if was_frozen else params


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
