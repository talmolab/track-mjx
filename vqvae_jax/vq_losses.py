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


def compute_kl_loss(
    mean: jnp.ndarray,
    logvar: jnp.ndarray,
) -> jnp.ndarray:
    """KL divergence KL(N(mean, exp(logvar)) || N(0,1)).

    Args:
        mean: Encoder mean, shape [..., latent_dim].
        logvar: Encoder log-variance, shape [..., latent_dim].

    Returns:
        Scalar KL divergence averaged over all elements.
    """
    return 0.5 * jnp.mean(jnp.exp(logvar) + mean**2 - 1.0 - logvar)


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
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
    vq_loss_schedule: Callable[[int], float] | None = None,
    rvq_depth: int = 1,
    codebook_entropy_weight: float = 0.0,
    codebook_entropy_temperature: float = 1.0,
    kl_weight: float = 0.0,
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
        discounting: Discount factor (gamma).
        reward_scaling: Reward multiplier.
        gae_lambda: GAE lambda.
        clipping_epsilon: PPO clipping range.
        normalize_advantage: Whether to normalize advantages.
        vq_loss_schedule: Optional schedule function(step) -> vq_weight.
        rvq_depth: Number of RVQ depth levels.
        codebook_entropy_weight: Weight for soft codebook entropy regularization.
        codebook_entropy_temperature: Temperature for soft code assignments.
        kl_weight: KL divergence weight for continuous latent.

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
    sb = ppo_network.stickiness_bias
    has_stickiness = sb > 0 if isinstance(sb, (int, float)) else any(b > 0 for b in sb)
    if has_stickiness and hasattr(policy_network, "apply_temporal"):
        truncation = data.extras["state_extras"]["truncation"]
        discount = data.discount
        continues = discount * (1 - truncation)
        episode_mask = jnp.concatenate(
            [jnp.zeros((1, continues.shape[1])), continues[:-1]], axis=0
        )
        policy_logits, z_e, all_indices, logvar = policy_network.apply_temporal(
            normalizer_params,
            params.policy,
            data.observation,
            episode_mask,
            proprio_noise_key=policy_key,
        )
    else:
        policy_logits, z_e, all_indices, logvar = policy_network.apply(
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

    # KL divergence loss (continuous latent)
    # logvar is (cont_mean, cont_logvar) tuple when continuous, else None
    if logvar is not None and kl_weight > 0:
        continuous_mean, continuous_logvar = logvar
        kl_loss = compute_kl_loss(continuous_mean, continuous_logvar)
        scaled_kl_loss = kl_weight * kl_loss
    else:
        kl_loss = jnp.array(0.0)
        scaled_kl_loss = jnp.array(0.0)

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
        + scaled_entropy_reg
        + scaled_kl_loss
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
        # Stickiness metrics (zero — CE stickiness removed)
        "ce_stickiness_loss": jnp.array(0.0),
        "scaled_ce_stickiness_loss": jnp.array(0.0),
        "prob_of_prev_code": jnp.array(0.0),
        "transition_rate": transition_rate,
        # Codebook entropy regularization
        "scaled_codebook_entropy_reg": scaled_entropy_reg,
        # KL divergence (continuous latent)
        "kl_loss": kl_loss,
        "scaled_kl_loss": scaled_kl_loss,
    }

    # Continuous latent metrics (only when use_continuous_latent=True)
    if logvar is not None:
        continuous_mean, continuous_logvar = logvar
        # Discrete z_e stats
        metrics["discrete_latent/z_e_l2_norm"] = jnp.mean(
            jnp.linalg.norm(z_e, axis=-1)
        )
        # Continuous head stats
        z_e_l2_norm = jnp.mean(jnp.linalg.norm(continuous_mean, axis=-1))
        z_e_mean_abs = jnp.mean(jnp.abs(continuous_mean))
        logvar_mean = jnp.mean(continuous_logvar)
        logvar_min = jnp.min(continuous_logvar)
        logvar_max = jnp.max(continuous_logvar)
        posterior_std_mean = jnp.mean(jnp.exp(0.5 * continuous_logvar))
        metrics["continuous_latent/z_e_l2_norm"] = z_e_l2_norm
        metrics["continuous_latent/z_e_mean_abs"] = z_e_mean_abs
        metrics["continuous_latent/logvar_mean"] = logvar_mean
        metrics["continuous_latent/logvar_min"] = logvar_min
        metrics["continuous_latent/logvar_max"] = logvar_max
        metrics["continuous_latent/posterior_std_mean"] = posterior_std_mean

    # Add per-depth metrics
    metrics.update(depth_metrics)

    # Add per-depth soft entropy metrics
    metrics.update(entropy_metrics)

    return total_loss, metrics


def compute_vq_chunked_ppo_loss(
    params: PPONetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    step: int,
    ppo_network: Any,
    entropy_cost: float = 1e-4,
    commitment_cost: float = 0.25,
    codebook_loss_weight: float = 1.0,
    commitment_horizon: int = 5,
    num_codes: int = 32,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
    codebook_entropy_weight: float = 0.0,
    codebook_entropy_temperature: float = 1.0,
    kl_weight: float = 0.0,
) -> tuple[jnp.ndarray, types.Metrics]:
    """Compute PPO loss with D0 temporal commitment (code chunking).

    Key differences from compute_vq_ppo_loss:
    1. Forward pass uses apply_temporal_chunked with D0 commitment
    2. Value function is augmented with D0 code identity and timer (tau)
    3. D0 commitment loss is masked to manager steps only (tau == 0)
    4. D1 commitment loss applies every step (D1 is always fresh)
    5. Bootstrap value uses d0_indices[-1] and (tau[-1] + 1) % H

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
        commitment_horizon: H, number of steps to hold D0 code.
        num_codes: Number of codebook entries per level.
        discounting: Discount factor (gamma).
        reward_scaling: Reward multiplier.
        gae_lambda: GAE lambda.
        clipping_epsilon: PPO clipping range.
        normalize_advantage: Whether to normalize advantages.
        codebook_entropy_weight: Weight for soft codebook entropy reg.
        codebook_entropy_temperature: Temperature for soft code assignments.
        kl_weight: KL divergence weight for continuous latent.

    Returns:
        Tuple of (total_loss, metrics_dict).
    """
    _, policy_key, entropy_key = jax.random.split(rng, 3)
    parametric_action_distribution = ppo_network.parametric_action_distribution
    policy_network = ppo_network.policy_network
    value_apply = ppo_network.value_network.apply

    # Extract initial carry state BEFORE tree_map (it has no time dimension)
    initial_held_d0_idx = None
    initial_tau = None
    initial_carry = data.extras["policy_extras"].get("initial_carry_state", None)
    if initial_carry is not None:
        initial_held_d0_idx, initial_tau = initial_carry
        # Remove from data so tree_map doesn't try to swapaxes on 1D arrays
        policy_extras_clean = {
            k: v
            for k, v in data.extras["policy_extras"].items()
            if k != "initial_carry_state"
        }
        data = data._replace(
            extras={
                **data.extras,
                "policy_extras": policy_extras_clean,
            }
        )

    # Put the time dimension first: [B, T, ...] -> [T, B, ...]
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    # Compute episode mask for chunking
    truncation = data.extras["state_extras"]["truncation"]
    discount = data.discount
    continues = discount * (1 - truncation)
    # When initial_carry is provided, step 0 is a continuation from the
    # previous unroll — NOT an episode start. The carry already encodes
    # episode boundary info (reset_chunk_state_on_done zeros it on done).
    # Without initial_carry, step 0 has no predecessor so treat as start.
    first_mask = (
        jnp.ones((1, continues.shape[1]))
        if initial_carry is not None
        else jnp.zeros((1, continues.shape[1]))
    )
    episode_mask = jnp.concatenate([first_mask, continues[:-1]], axis=0)

    # Forward pass through chunked VQ policy
    (policy_logits, z_e, all_indices, logvar, tau) = (
        policy_network.apply_temporal_chunked(
            normalizer_params,
            params.policy,
            data.observation,
            commitment_horizon=commitment_horizon,
            episode_mask=episode_mask,
            proprio_noise_key=policy_key,
            initial_held_d0_idx=initial_held_d0_idx,
            initial_tau=initial_tau,
        )
    )

    d0_indices, d1_indices = all_indices

    # Value function with augmented inputs (D0 code + tau)
    baseline = value_apply(
        normalizer_params, params.value, data.observation,
        d0_code_idx=d0_indices, tau=tau,
    )

    # Bootstrap value: use last step's D0 code, advance tau by 1
    last_next_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
    bootstrap_d0_idx = d0_indices[-1]
    bootstrap_tau = (tau[-1] + 1) % commitment_horizon
    bootstrap_value = value_apply(
        normalizer_params, params.value, last_next_obs,
        d0_code_idx=bootstrap_d0_idx, tau=bootstrap_tau,
    )

    # Extract codebooks from param tree
    quantizer_params = params.policy["params"]["quantizer"]
    codebook_0 = quantizer_params["codebooks_0"]["embeddings"]
    codebook_1 = quantizer_params["codebooks_1"]["embeddings"]

    # Reconstruct z_q for loss computation
    d0_z_q = codebook_0[d0_indices]  # [T, B, D]
    d1_z_q = codebook_1[d1_indices]  # [T, B, D]

    # D0 commitment loss: masked to manager steps only (tau == 0)
    is_manager_step = (tau == 0).astype(jnp.float32)  # [T, B]
    n_manager = jnp.sum(is_manager_step) + 1e-8

    d0_commitment = jnp.sum(
        jnp.mean((z_e - jax.lax.stop_gradient(d0_z_q)) ** 2, axis=-1)
        * is_manager_step
    ) / n_manager
    d0_codebook = jnp.sum(
        jnp.mean((jax.lax.stop_gradient(z_e) - d0_z_q) ** 2, axis=-1)
        * is_manager_step
    ) / n_manager

    # D1 commitment loss: every step (D1 is always fresh)
    d1_residual = z_e - jax.lax.stop_gradient(d0_z_q)
    d1_commitment = jnp.mean(
        (d1_residual - jax.lax.stop_gradient(d1_z_q)) ** 2
    )
    d1_codebook = jnp.mean(
        (jax.lax.stop_gradient(d1_residual) - d1_z_q) ** 2
    )

    # Combined VQ losses (1/2 scaling for 2 levels)
    commitment_loss = 0.5 * (d0_commitment + d1_commitment)
    codebook_loss = 0.5 * (d0_codebook + d1_codebook)
    scaled_vq_loss = (
        commitment_cost * commitment_loss
        + codebook_loss_weight * codebook_loss
    )

    # Codebook health metrics
    perplexity_d0, utilization_d0, codes_used_d0 = (
        _compute_single_codebook_metrics(d0_indices, num_codes)
    )
    perplexity_d1, utilization_d1, codes_used_d1 = (
        _compute_single_codebook_metrics(d1_indices, num_codes)
    )

    # Codebook entropy regularization
    if codebook_entropy_weight > 0.0:
        codebooks = (codebook_0, codebook_1)
        # Reconstruct residuals for entropy computation
        all_residuals = (z_e, d1_residual, d1_residual - jax.lax.stop_gradient(d1_z_q))
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

    # KL divergence loss (continuous latent)
    # logvar is (cont_mean, cont_logvar) tuple when continuous, else None
    if logvar is not None and kl_weight > 0:
        continuous_mean, continuous_logvar = logvar
        kl_loss = compute_kl_loss(continuous_mean, continuous_logvar)
        scaled_kl_loss = kl_weight * kl_loss
    else:
        kl_loss = jnp.array(0.0)
        scaled_kl_loss = jnp.array(0.0)

    # Standard PPO loss computation
    rewards = data.reward * reward_scaling
    termination = (1 - data.discount) * (1 - truncation)

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
        + scaled_entropy_reg
        + scaled_kl_loss
    )

    # Chunking-specific metrics
    manager_rate = jnp.mean(is_manager_step)
    avg_tau = jnp.mean(tau.astype(jnp.float32))

    # D0 hold fidelity: fraction of worker steps where D0 didn't change
    # (This checks if fresh D0 would have matched held D0)
    is_worker_step = 1.0 - is_manager_step
    n_worker = jnp.sum(is_worker_step) + 1e-8
    # Compute what fresh D0 would be
    fresh_distances = (
        jnp.sum(z_e**2, axis=-1, keepdims=True)
        + jnp.sum(codebook_0**2, axis=-1)
        - 2 * jnp.matmul(z_e, codebook_0.T)
    )
    fresh_d0_idx = jnp.argmin(fresh_distances, axis=-1)
    d0_held_matches = (fresh_d0_idx == d0_indices).astype(jnp.float32)
    d0_hold_fidelity = jnp.sum(d0_held_matches * is_worker_step) / n_worker

    # Transition rates
    T = z_e.shape[0]
    if T > 1:
        valid_mask = data.discount[:-1] * (1 - truncation[:-1])
        num_valid = jnp.sum(valid_mask) + 1e-8
        d0_changed = (d0_indices[1:] != d0_indices[:-1]).astype(jnp.float32)
        d1_changed = (d1_indices[1:] != d1_indices[:-1]).astype(jnp.float32)
        d0_transition_rate = jnp.sum(d0_changed * valid_mask) / num_valid
        d1_transition_rate = jnp.sum(d1_changed * valid_mask) / num_valid
        transition_rate = d0_transition_rate
    else:
        d0_transition_rate = jnp.array(0.0)
        d1_transition_rate = jnp.array(0.0)
        transition_rate = jnp.array(0.0)

    metrics = {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "v_loss": v_loss,
        "entropy_loss": entropy_loss,
        # VQ losses
        "vq_loss": commitment_cost * commitment_loss + codebook_loss,
        "commitment_loss": commitment_loss,
        "codebook_loss": codebook_loss,
        "scaled_vq_loss": scaled_vq_loss,
        "vq_weight": jnp.array(1.0),
        # Per-level VQ losses
        "d0_commitment_loss": d0_commitment,
        "d0_codebook_loss": d0_codebook,
        "d1_commitment_loss": d1_commitment,
        "d1_codebook_loss": d1_codebook,
        # Codebook health (primary = D0)
        "perplexity": perplexity_d0,
        "codebook_utilization": utilization_d0,
        "codes_used": codes_used_d0,
        "perplexity_d0": perplexity_d0,
        "utilization_d0": utilization_d0,
        "codes_used_d0": codes_used_d0,
        "perplexity_d1": perplexity_d1,
        "utilization_d1": utilization_d1,
        "codes_used_d1": codes_used_d1,
        # Transition rates
        "transition_rate": transition_rate,
        "d0_transition_rate": d0_transition_rate,
        "d1_transition_rate": d1_transition_rate,
        # Chunking metrics
        "manager_rate": manager_rate,
        "avg_tau": avg_tau,
        "d0_hold_fidelity": d0_hold_fidelity,
        # Stickiness (zero for chunking — not applicable)
        "ce_stickiness_loss": jnp.array(0.0),
        "scaled_ce_stickiness_loss": jnp.array(0.0),
        "prob_of_prev_code": jnp.array(0.0),
        # Entropy reg
        "scaled_codebook_entropy_reg": scaled_entropy_reg,
        # KL
        "kl_loss": kl_loss,
        "scaled_kl_loss": scaled_kl_loss,
    }

    # Continuous latent metrics (only when use_continuous_latent=True)
    if logvar is not None:
        continuous_mean, continuous_logvar = logvar
        # Discrete z_e stats
        metrics["discrete_latent/z_e_l2_norm"] = jnp.mean(
            jnp.linalg.norm(z_e, axis=-1)
        )
        # Continuous head stats
        z_e_l2_norm = jnp.mean(jnp.linalg.norm(continuous_mean, axis=-1))
        z_e_mean_abs = jnp.mean(jnp.abs(continuous_mean))
        logvar_mean = jnp.mean(continuous_logvar)
        logvar_min = jnp.min(continuous_logvar)
        logvar_max = jnp.max(continuous_logvar)
        posterior_std_mean = jnp.mean(jnp.exp(0.5 * continuous_logvar))
        metrics["continuous_latent/z_e_l2_norm"] = z_e_l2_norm
        metrics["continuous_latent/z_e_mean_abs"] = z_e_mean_abs
        metrics["continuous_latent/logvar_mean"] = logvar_mean
        metrics["continuous_latent/logvar_min"] = logvar_min
        metrics["continuous_latent/logvar_max"] = logvar_max
        metrics["continuous_latent/posterior_std_mean"] = posterior_std_mean

    # Add entropy metrics if present
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
