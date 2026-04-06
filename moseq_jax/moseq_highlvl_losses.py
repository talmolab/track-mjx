"""PPO loss for MoSeq high-level RNN intention policy.

Standard PPO losses (clipped surrogate, value, entropy) without any
encoder/VQ auxiliary losses. The policy produces latent intentions that
are routed through a frozen decoder externally (by the wrapper).
"""

from typing import Any

import jax
import jax.numpy as jnp
from brax.training import types

from moseq_losses import compute_gae


def compute_moseq_highlvl_ppo_loss(
    params,
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
    # Unused — kept for interface compatibility with ppo.train loss signature
    latent_kl_weight: float = 0.0,
    latent_ar1_weight: float = 0.0,
    latent_kl_schedule=None,
    latent_ar1_schedule=None,
) -> tuple[jnp.ndarray, dict]:
    """Compute PPO loss for the high-level intention RNN policy."""
    _, policy_key, entropy_key = jax.random.split(rng, 3)
    action_dist = ppo_network.parametric_action_distribution
    policy_network = ppo_network.policy_network
    value_apply = ppo_network.value_network.apply

    # Extract initial_carry BEFORE swapaxes (it has no T dim)
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

    done = data.discount < 0.5

    # Forward pass via scan
    policy_logits, final_hidden = policy_network.apply_sequence(
        normalizer_params,
        params.policy,
        data.observation,
        initial_carry,
        done,
        policy_key,
        deterministic=False,
    )

    # Value function (feedforward)
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
    surrogate2 = (
        jnp.clip(rho, 1 - clipping_epsilon, 1 + clipping_epsilon) * advantages
    )
    policy_loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2))

    v_error = vs - baseline
    v_loss = jnp.mean(v_error * v_error) * 0.5 * vf_coefficient

    entropy = jnp.mean(action_dist.entropy(policy_logits, entropy_key))
    entropy_loss = entropy_cost * -entropy

    total_loss = policy_loss + v_loss + entropy_loss

    # --- Diagnostics ---
    hidden_norms = [
        jnp.mean(jnp.sqrt(jnp.sum(h**2, axis=-1))) for h in final_hidden
    ]
    hidden_state_norm = jnp.mean(jnp.stack(hidden_norms))

    # Intention diagnostics
    raw_actions = data.extras["policy_extras"]["raw_action"]
    intention_norm = jnp.mean(jnp.sqrt(jnp.sum(raw_actions**2, axis=-1)))
    intention_std = jnp.mean(jnp.std(raw_actions, axis=0))

    # Code metrics
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

    metrics = {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "v_loss": v_loss,
        "entropy_loss": entropy_loss,
        "transition_rate": transition_rate,
        "perplexity": perplexity,
        "codebook_utilization": utilization,
        "codes_used": codes_used,
        "intention_norm": intention_norm,
        "intention_std": intention_std,
        "hidden_state_norm": hidden_state_norm,
        # Placeholders for shared logging code
        "vq_loss": jnp.array(0.0),
        "commitment_loss": jnp.array(0.0),
        "codebook_loss": jnp.array(0.0),
        "scaled_vq_loss": jnp.array(0.0),
        "vq_weight": jnp.array(1.0),
        "ce_stickiness_loss": jnp.array(0.0),
        "scaled_ce_stickiness_loss": jnp.array(0.0),
        "prob_of_prev_code": jnp.array(0.0),
        "scaled_codebook_entropy_reg": jnp.array(0.0),
        "kl_loss": jnp.array(0.0),
        "scaled_kl_loss": jnp.array(0.0),
        "z_e_norm": jnp.array(0.0),
        "z_e_std": jnp.array(0.0),
    }

    return total_loss, metrics
