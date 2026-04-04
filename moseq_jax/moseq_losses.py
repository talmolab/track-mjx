"""PPO loss computation for MoSeq decoder training.

Provides two loss functions:

* ``compute_moseq_ppo_loss`` — feedforward decoder (original).
* ``compute_moseq_recurrent_ppo_loss`` — RNN decoder with scan-based
  forward pass, hidden-state carry, and deterministic z_e replay.

Standard PPO losses (clipped surrogate, value, entropy) without any VQ-VAE
auxiliary losses.  Code indices are extracted from ``data.observation`` for
metrics only — they do not contribute to the loss.
"""

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
        policy: Decoder policy parameters.
        value: Value function parameters.
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
    """Compute Generalized Advantage Estimation.

    Args:
        truncation: Truncation signal, shape ``[T, B]``.
        termination: Termination signal, shape ``[T, B]``.
        rewards: Rewards, shape ``[T, B]``.
        values: Value estimates V(s_t), shape ``[T, B]``.
        bootstrap_value: Value at time T, shape ``[B]``.
        lambda_: GAE lambda.
        discount: Discount factor gamma.

    Returns:
        ``(value_targets, advantages)`` each shape ``[T, B]``.
    """
    truncation_mask = 1 - truncation

    values_tp1 = jnp.concatenate(
        [values[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0
    )
    deltas = rewards + discount * (1 - termination) * values_tp1 - values
    deltas *= truncation_mask

    acc = jnp.zeros_like(bootstrap_value)

    def _scan_fn(carry, target_t):
        lam, acc = carry
        trunc_mask, delta, term = target_t
        acc = delta + discount * (1 - term) * trunc_mask * lam * acc
        return (lam, acc), acc

    (_, _), vs_minus_v = jax.lax.scan(
        _scan_fn,
        (lambda_, acc),
        (truncation_mask, deltas, termination),
        length=int(truncation_mask.shape[0]),
        reverse=True,
    )

    vs = jnp.add(vs_minus_v, values)
    vs_tp1 = jnp.concatenate([vs[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
    advantages = (
        rewards + discount * (1 - termination) * vs_tp1 - values
    ) * truncation_mask

    return jax.lax.stop_gradient(vs), jax.lax.stop_gradient(advantages)


def compute_moseq_ppo_loss(
    params: PPONetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    step: int,
    ppo_network: Any,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
    vf_coefficient: float = 0.5,
    kl_weight: float = 0.0,
) -> tuple[jnp.ndarray, types.Metrics]:
    """Compute standard PPO loss for the MoSeq decoder policy.

    Args:
        params: Policy and value network parameters.
        normalizer_params: Running statistics for observation normalization.
        data: Transition batch, shape ``[B, T, ...]``.
        rng: JAX PRNG key.
        step: Current training step (unused, kept for interface).
        ppo_network: ``MoSeqPPONetworks`` container.
        entropy_cost: Entropy bonus coefficient.
        discounting: Discount factor gamma.
        reward_scaling: Reward multiplier.
        gae_lambda: GAE lambda.
        clipping_epsilon: PPO clipping range.
        normalize_advantage: Whether to normalize advantages.
        vf_coefficient: Value loss coefficient.

    Returns:
        ``(total_loss, metrics_dict)``.
    """
    _, policy_key, entropy_key = jax.random.split(rng, 3)
    action_dist = ppo_network.parametric_action_distribution
    policy_network = ppo_network.policy_network
    value_apply = ppo_network.value_network.apply

    # [B, T, ...] -> [T, B, ...]
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    # Forward pass through decoder policy
    policy_logits, code_idx, cont_mean, cont_logvar = policy_network.apply(
        normalizer_params, params.policy, data.observation, policy_key
    )

    # Value function
    baseline = value_apply(normalizer_params, params.value, data.observation)
    last_next_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
    bootstrap_value = value_apply(normalizer_params, params.value, last_next_obs)

    # Standard PPO losses
    rewards = data.reward * reward_scaling
    truncation = data.extras["state_extras"]["truncation"]
    termination = (1 - data.discount) * (1 - truncation)

    target_log_probs = action_dist.log_prob(
        policy_logits, data.extras["policy_extras"]["raw_action"]
    )
    behaviour_log_probs = data.extras["policy_extras"]["log_prob"]

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

    rho = jnp.exp(target_log_probs - behaviour_log_probs)

    surrogate1 = rho * advantages
    surrogate2 = jnp.clip(rho, 1 - clipping_epsilon, 1 + clipping_epsilon) * advantages
    policy_loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2))

    v_error = vs - baseline
    v_loss = jnp.mean(v_error * v_error) * 0.5 * vf_coefficient

    entropy = jnp.mean(action_dist.entropy(policy_logits, entropy_key))
    entropy_loss = entropy_cost * -entropy

    # KL loss for continuous encoder
    if cont_mean is not None:
        kl_loss = 0.5 * jnp.mean(
            jnp.exp(cont_logvar) + cont_mean**2 - 1.0 - cont_logvar
        )
        scaled_kl_loss = kl_weight * kl_loss
    else:
        kl_loss = jnp.array(0.0)
        scaled_kl_loss = jnp.array(0.0)

    total_loss = policy_loss + v_loss + entropy_loss + scaled_kl_loss

    # Code transition rate (diagnostic only)
    if code_idx.shape[0] > 1:
        valid_mask = data.discount[:-1] * (1 - truncation[:-1])
        num_valid = jnp.sum(valid_mask) + 1e-8
        changed = (code_idx[1:] != code_idx[:-1]).astype(jnp.float32)
        transition_rate = jnp.sum(changed * valid_mask) / num_valid
    else:
        transition_rate = jnp.array(0.0)

    # Code utilization
    num_codes = ppo_network.num_codes
    code_flat = code_idx.reshape(-1)
    counts = jnp.bincount(code_flat, length=num_codes).astype(jnp.float32)
    probs = counts / (counts.sum() + 1e-8)
    perplexity = jnp.exp(-jnp.sum(probs * jnp.log(probs + 1e-8)))
    codes_used = jnp.sum(counts > 0).astype(jnp.float32)
    utilization = codes_used / num_codes

    # z_e diagnostics
    if cont_mean is not None:
        z_e_norm = jnp.mean(jnp.sqrt(jnp.sum(cont_mean**2, axis=-1)))
        z_e_std = jnp.mean(jnp.std(cont_mean, axis=0))
    else:
        z_e_norm = jnp.array(0.0)
        z_e_std = jnp.array(0.0)

    metrics = {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "v_loss": v_loss,
        "entropy_loss": entropy_loss,
        "transition_rate": transition_rate,
        "perplexity": perplexity,
        "codebook_utilization": utilization,
        "codes_used": codes_used,
        "z_e_norm": z_e_norm,
        "z_e_std": z_e_std,
        # Placeholders for VQ metrics (enables shared logging code)
        "vq_loss": jnp.array(0.0),
        "commitment_loss": jnp.array(0.0),
        "codebook_loss": jnp.array(0.0),
        "scaled_vq_loss": jnp.array(0.0),
        "vq_weight": jnp.array(1.0),
        "ce_stickiness_loss": jnp.array(0.0),
        "scaled_ce_stickiness_loss": jnp.array(0.0),
        "prob_of_prev_code": jnp.array(0.0),
        "scaled_codebook_entropy_reg": jnp.array(0.0),
        "kl_loss": kl_loss,
        "scaled_kl_loss": scaled_kl_loss,
    }

    return total_loss, metrics


def compute_moseq_recurrent_ppo_loss(
    params: PPONetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    step: int,
    ppo_network: Any,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
    vf_coefficient: float = 0.5,
    kl_weight: float = 0.0,
    z_e_scale: float = 1.0,
) -> tuple[jnp.ndarray, types.Metrics]:
    """Compute PPO loss for the recurrent MoSeq decoder.

    Uses ``apply_sequence`` (jax.lax.scan) to recompute the policy forward
    pass over time, threading hidden state through timesteps and replaying
    stored PRNG keys for deterministic z_e samples.

    Args:
        params: Policy and value network parameters.
        normalizer_params: Running statistics for observation normalization.
        data: Transition batch, shape ``[B, T, ...]``.  Must contain
            ``data.extras["policy_extras"]["initial_carry_state"]`` (RNN
            hidden state at the start of the unroll) and optionally
            ``data.extras["policy_extras"]["policy_rng"]`` (stored PRNG
            keys for deterministic z_e replay).
        rng: JAX PRNG key.
        step: Current training step (unused, kept for interface).
        ppo_network: ``MoSeqRecurrentPPONetworks`` container.
        entropy_cost: Entropy bonus coefficient.
        discounting: Discount factor gamma.
        reward_scaling: Reward multiplier.
        gae_lambda: GAE lambda.
        clipping_epsilon: PPO clipping range.
        normalize_advantage: Whether to normalize advantages.
        vf_coefficient: Value loss coefficient.
        kl_weight: KL regularization weight for continuous encoder.
        z_e_scale: Multiplier on z_e (1.0 = full, 0.0 = decoder-only).

    Returns:
        ``(total_loss, metrics_dict)``.
    """
    _, policy_key, entropy_key = jax.random.split(rng, 3)
    action_dist = ppo_network.parametric_action_distribution
    policy_network = ppo_network.policy_network
    value_apply = ppo_network.value_network.apply

    # --- Extract initial_carry_state BEFORE swapaxes (it has no T dim) ---
    initial_carry = data.extras["policy_extras"]["initial_carry_state"]

    # [B, T, ...] -> [T, B, ...]
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    # Restore initial_carry (swapaxes would have corrupted it)
    policy_extras = dict(data.extras["policy_extras"])
    policy_extras["initial_carry_state"] = initial_carry
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

    # Extract stored PRNG keys for deterministic z_e replay
    stored_keys = data.extras["policy_extras"].get("policy_rng")

    # Done flags for hidden state reset
    done = data.discount < 0.5

    # Forward pass via scan over time
    policy_logits, cont_mean, cont_logvar, final_hidden = policy_network.apply_sequence(
        normalizer_params,
        params.policy,
        data.observation,
        initial_carry,
        done,
        policy_key,
        deterministic=False,
        stored_keys=stored_keys,
        z_e_scale=z_e_scale,
    )

    # Value function (feedforward — no hidden state)
    baseline = value_apply(normalizer_params, params.value, data.observation)
    last_next_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
    bootstrap_value = value_apply(normalizer_params, params.value, last_next_obs)

    # Standard PPO losses
    rewards = data.reward * reward_scaling
    truncation = data.extras["state_extras"]["truncation"]
    termination = (1 - data.discount) * (1 - truncation)

    target_log_probs = action_dist.log_prob(
        policy_logits, data.extras["policy_extras"]["raw_action"]
    )
    behaviour_log_probs = data.extras["policy_extras"]["log_prob"]

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

    rho = jnp.exp(target_log_probs - behaviour_log_probs)

    surrogate1 = rho * advantages
    surrogate2 = jnp.clip(rho, 1 - clipping_epsilon, 1 + clipping_epsilon) * advantages
    policy_loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2))

    v_error = vs - baseline
    v_loss = jnp.mean(v_error * v_error) * 0.5 * vf_coefficient

    entropy = jnp.mean(action_dist.entropy(policy_logits, entropy_key))
    entropy_loss = entropy_cost * -entropy

    # KL loss for continuous encoder
    if cont_mean is not None:
        kl_loss = 0.5 * jnp.mean(
            jnp.exp(cont_logvar) + cont_mean**2 - 1.0 - cont_logvar
        )
        scaled_kl_loss = kl_weight * kl_loss
    else:
        kl_loss = jnp.array(0.0)
        scaled_kl_loss = jnp.array(0.0)

    total_loss = policy_loss + v_loss + entropy_loss + scaled_kl_loss

    # --- z_e and hidden state diagnostics ---
    if cont_mean is not None:
        z_e_norm = jnp.mean(jnp.sqrt(jnp.sum(cont_mean**2, axis=-1)))
        z_e_std = jnp.mean(jnp.std(cont_mean, axis=0))
    else:
        z_e_norm = jnp.array(0.0)
        z_e_std = jnp.array(0.0)

    # Hidden state norm (mean across batch and layers)
    hidden_norms = [jnp.mean(jnp.sqrt(jnp.sum(h**2, axis=-1))) for h in final_hidden]
    hidden_state_norm = jnp.mean(jnp.stack(hidden_norms))

    # Code metrics (from observation, not from scan — codes don't change)
    kpms_code = data.observation["state"].get("kpms_code")
    if kpms_code is not None:
        code_idx = jnp.round(kpms_code[..., 0]).astype(jnp.int32)
        num_codes = ppo_network.num_codes
        code_flat = code_idx.reshape(-1)
        counts = jnp.bincount(code_flat, length=num_codes).astype(jnp.float32)
        probs = counts / (counts.sum() + 1e-8)
        perplexity = jnp.exp(-jnp.sum(probs * jnp.log(probs + 1e-8)))
        codes_used = jnp.sum(counts > 0).astype(jnp.float32)
        utilization = codes_used / num_codes

        if code_idx.shape[0] > 1:
            valid_mask = data.discount[:-1] * (1 - truncation[:-1])
            num_valid = jnp.sum(valid_mask) + 1e-8
            changed = (code_idx[1:] != code_idx[:-1]).astype(jnp.float32)
            transition_rate = jnp.sum(changed * valid_mask) / num_valid
        else:
            transition_rate = jnp.array(0.0)
    else:
        perplexity = jnp.array(0.0)
        codes_used = jnp.array(0.0)
        utilization = jnp.array(0.0)
        transition_rate = jnp.array(0.0)
        num_codes = ppo_network.num_codes

    metrics = {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "v_loss": v_loss,
        "entropy_loss": entropy_loss,
        "transition_rate": transition_rate,
        "perplexity": perplexity,
        "codebook_utilization": utilization,
        "codes_used": codes_used,
        "kl_loss": kl_loss,
        "scaled_kl_loss": scaled_kl_loss,
        "z_e_norm": z_e_norm,
        "z_e_std": z_e_std,
        "z_e_scale": jnp.array(z_e_scale),
        "hidden_state_norm": hidden_state_norm,
        # Placeholders for VQ metrics (shared logging code)
        "vq_loss": jnp.array(0.0),
        "commitment_loss": jnp.array(0.0),
        "codebook_loss": jnp.array(0.0),
        "scaled_vq_loss": jnp.array(0.0),
        "vq_weight": jnp.array(1.0),
        "ce_stickiness_loss": jnp.array(0.0),
        "scaled_ce_stickiness_loss": jnp.array(0.0),
        "prob_of_prev_code": jnp.array(0.0),
        "scaled_codebook_entropy_reg": jnp.array(0.0),
    }

    return total_loss, metrics
