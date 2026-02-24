"""Losses for temporal PPO with latent commitment and optional learned boundaries."""

from collections.abc import Callable
from typing import Any

import flax
import jax
import jax.numpy as jnp
from brax.training import types
from brax.training.types import Params

from track_mjx.agent.ff_ppo.losses import (
    compute_gae,
    create_ramp_schedule,
)
from track_mjx.agent.temporal_ppo import networks
from track_mjx.agent.temporal_ppo.types import TemporalPolicyCarry


@flax.struct.dataclass
class TemporalPPONetworkParams:
    """Container for temporal PPO network parameters."""

    policy: Params
    value: Params


def _build_initial_carry(data: types.Transition) -> TemporalPolicyCarry:
    """Reconstructs initial temporal carry stored in policy extras."""
    policy_extras = data.extras["policy_extras"]
    return TemporalPolicyCarry(
        decoder_hidden=policy_extras["initial_policy_hidden"],
        current_latent=policy_extras["initial_current_latent"],
        current_latent_mean=policy_extras["initial_latent_mean"],
        current_latent_logvar=policy_extras["initial_latent_logvar"],
        segment_step=policy_extras["initial_segment_step"],
    )


def _compute_kl_refresh_only(
    latent_mean: jnp.ndarray,
    latent_logvar: jnp.ndarray,
    refresh_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Computes KL only on refresh steps."""
    kl_per_step = -0.5 * jnp.mean(
        1.0 + latent_logvar - jnp.square(latent_mean) - jnp.exp(latent_logvar),
        axis=-1,
    )
    denom = jnp.maximum(jnp.sum(refresh_mask), 1.0)
    return jnp.sum(kl_per_step * refresh_mask) / denom


def _ar1_single_batch(
    latent_mean: jnp.ndarray,
    refresh_mask: jnp.ndarray,
    discount: jnp.ndarray,
    truncation: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Interval-weighted AR1 for a single batch element."""

    def scan_fn(carry, inputs):
        prev_z, has_prev, steps_since_prev = carry
        z_t, refresh_t, discount_t, trunc_t = inputs

        refresh_t = refresh_t > 0.5
        continuation = (discount_t > 0.5) & (trunc_t < 0.5)

        can_pair = refresh_t & has_prev
        interval = jnp.maximum(steps_since_prev, 1)
        diff = jnp.mean(jnp.square(z_t - prev_z))

        contrib = jnp.where(can_pair, diff / interval, 0.0)
        pair_count = jnp.where(can_pair, 1.0, 0.0)
        interval_sum = jnp.where(can_pair, interval.astype(jnp.float32), 0.0)

        next_prev_z = jnp.where(refresh_t[..., None], z_t, prev_z)
        next_has_prev = jnp.where(refresh_t, True, has_prev)
        next_steps = jnp.where(refresh_t, 1, steps_since_prev + has_prev.astype(jnp.int32))

        # Break latent continuity across episode boundaries.
        next_has_prev = jnp.where(continuation, next_has_prev, False)
        next_steps = jnp.where(continuation, next_steps, 0)

        return (next_prev_z, next_has_prev, next_steps), (
            contrib,
            pair_count,
            interval_sum,
        )

    init_prev_z = jnp.zeros_like(latent_mean[0])
    init_carry = (init_prev_z, False, jnp.asarray(0, dtype=jnp.int32))

    _, (contribs, pair_counts, interval_sums) = jax.lax.scan(
        scan_fn,
        init_carry,
        (latent_mean, refresh_mask, discount, truncation),
    )

    return (
        jnp.sum(contribs),
        jnp.sum(pair_counts),
        jnp.sum(interval_sums),
    )


def compute_interval_weighted_ar1(
    latent_mean: jnp.ndarray,
    refresh_mask: jnp.ndarray,
    discount: jnp.ndarray,
    truncation: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """AR1 over refresh events weighted by refresh interval length."""
    # Inputs are [T, B, ...]. Vmap over batch dimension.
    per_batch = jax.vmap(_ar1_single_batch, in_axes=(1, 1, 1, 1))(
        latent_mean,
        refresh_mask,
        discount,
        truncation,
    )
    total_contrib = jnp.sum(per_batch[0])
    total_pairs = jnp.sum(per_batch[1])
    total_interval = jnp.sum(per_batch[2])

    ar1 = total_contrib / jnp.maximum(total_pairs, 1.0)
    mean_interval = total_interval / jnp.maximum(total_pairs, 1.0)
    return ar1, mean_interval, total_pairs


def compute_temporal_ppo_loss(
    params: TemporalPPONetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    step: int,
    temporal_ppo_network: networks.TemporalPPONetworks,
    entropy_cost: float = 1e-4,
    gate_entropy_cost: float = 1e-4,
    latent_kl_weight: float = 1e-3,
    latent_ar1_weight: float = 1e-3,
    discounting: float = 0.9,
    discounting_gate: float | None = None,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
    vf_coefficient: float = 0.5,
    latent_kl_schedule: Callable[[int], float] | None = None,
    latent_ar1_schedule: Callable[[int], float] | None = None,
    target_refresh_rate: float | None = None,
    lambda_refresh_rate: float = 0.0,
) -> tuple[jnp.ndarray, types.Metrics]:
    """Computes temporal PPO loss."""
    _, policy_key, entropy_key = jax.random.split(rng, 3)

    policy_network = temporal_ppo_network.policy_network
    value_apply = temporal_ppo_network.value_network.apply
    parametric_action_distribution = temporal_ppo_network.parametric_action_distribution

    # Preserve initial carry (no time dimension) before swapaxes.
    initial_hidden = data.extras["policy_extras"]["initial_policy_hidden"]
    initial_segment_step = data.extras["policy_extras"]["initial_segment_step"]
    initial_current_latent = data.extras["policy_extras"]["initial_current_latent"]
    initial_latent_mean = data.extras["policy_extras"]["initial_latent_mean"]
    initial_latent_logvar = data.extras["policy_extras"]["initial_latent_logvar"]

    data = jax.tree_util.tree_map(
        lambda x: jnp.swapaxes(x, 0, 1) if getattr(x, "ndim", 0) >= 2 else x,
        data,
    )

    policy_extras = dict(data.extras["policy_extras"])
    policy_extras["initial_policy_hidden"] = initial_hidden
    policy_extras["initial_segment_step"] = initial_segment_step
    policy_extras["initial_current_latent"] = initial_current_latent
    policy_extras["initial_latent_mean"] = initial_latent_mean
    policy_extras["initial_latent_logvar"] = initial_latent_logvar

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

    initial_carry = _build_initial_carry(data)
    done = data.discount < 0.5

    stored_keys = data.extras["policy_extras"].get("policy_rng")
    stored_gate_samples = data.extras["policy_extras"].get("gate_sample")

    (
        motor_logits,
        latent_mean,
        latent_logvar,
        latent_z,
        gate_logits,
        _,
        gate_samples,
        gate_valid,
        refresh_mask,
        final_carry,
    ) = policy_network.apply_sequence(
        normalizer_params,
        params.policy,
        data.observation,
        initial_carry,
        done,
        policy_key,
        deterministic=False,
        train_step=step,
        stored_keys=stored_keys,
        stored_gate_samples=stored_gate_samples,
    )

    baseline = value_apply(normalizer_params, params.value, data.observation, latent_z)
    last_next_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
    bootstrap_value = value_apply(
        normalizer_params,
        params.value,
        last_next_obs,
        final_carry.current_latent,
    )

    rewards = data.reward * reward_scaling
    truncation = data.extras["state_extras"]["truncation"]
    termination = (1 - data.discount) * (1 - truncation)

    target_action_log_probs = parametric_action_distribution.log_prob(
        motor_logits,
        data.extras["policy_extras"]["raw_action"],
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

    motor_ratio = jnp.exp(target_action_log_probs - behaviour_action_log_probs)
    motor_surr1 = motor_ratio * advantages
    motor_surr2 = (
        jnp.clip(motor_ratio, 1 - clipping_epsilon, 1 + clipping_epsilon) * advantages
    )
    policy_loss = -jnp.mean(jnp.minimum(motor_surr1, motor_surr2))

    v_error = vs - baseline
    v_loss = jnp.mean(v_error * v_error) * 0.5 * vf_coefficient

    entropy = jnp.mean(parametric_action_distribution.entropy(motor_logits, entropy_key))
    entropy_loss = entropy_cost * -entropy

    # Gate losses (learned boundary mode only).
    gate_policy_loss = jnp.array(0.0, dtype=jnp.float32)
    gate_entropy_loss = jnp.array(0.0, dtype=jnp.float32)
    gate_refresh_reg_loss = jnp.array(0.0, dtype=jnp.float32)
    gate_entropy = jnp.array(0.0, dtype=jnp.float32)

    if temporal_ppo_network.boundary_mode == "learned":
        behaviour_gate_log_prob = data.extras["policy_extras"]["gate_log_prob"]
        target_gate_log_prob = networks.bernoulli_log_prob(gate_logits, gate_samples)

        gate_discount = discounting if discounting_gate is None else discounting_gate
        _, gate_advantages = compute_gae(
            truncation=truncation,
            termination=termination,
            rewards=rewards,
            values=baseline,
            bootstrap_value=bootstrap_value,
            lambda_=gae_lambda,
            discount=gate_discount,
        )
        if normalize_advantage:
            gate_advantages = (gate_advantages - gate_advantages.mean()) / (
                gate_advantages.std() + 1e-8
            )

        gate_ratio = jnp.exp(target_gate_log_prob - behaviour_gate_log_prob)
        gate_surr1 = gate_ratio * gate_advantages
        gate_surr2 = (
            jnp.clip(gate_ratio, 1 - clipping_epsilon, 1 + clipping_epsilon)
            * gate_advantages
        )

        valid_denom = jnp.maximum(jnp.sum(gate_valid), 1.0)
        gate_policy_loss = -jnp.sum(gate_valid * jnp.minimum(gate_surr1, gate_surr2)) / valid_denom

        gate_entropy_per_step = networks.bernoulli_entropy(gate_logits)
        gate_entropy = jnp.sum(gate_valid * gate_entropy_per_step) / valid_denom
        gate_entropy_loss = gate_entropy_cost * -gate_entropy

        if target_refresh_rate is not None and lambda_refresh_rate > 0.0:
            refresh_rate = jnp.mean(refresh_mask)
            gate_refresh_reg_loss = lambda_refresh_rate * jnp.square(
                refresh_rate - target_refresh_rate
            )

    current_kl_weight = latent_kl_weight
    current_ar1_weight = latent_ar1_weight
    if latent_kl_schedule is not None:
        current_kl_weight = latent_kl_schedule(step)
    if latent_ar1_schedule is not None:
        current_ar1_weight = latent_ar1_schedule(step)

    kl_gaussian = _compute_kl_refresh_only(latent_mean, latent_logvar, refresh_mask)

    ar1_loss_raw, mean_segment_length, refresh_pairs = compute_interval_weighted_ar1(
        latent_mean,
        refresh_mask,
        data.discount,
        truncation,
    )

    kl_weighted = current_kl_weight * kl_gaussian
    ar1_weighted = current_ar1_weight * ar1_loss_raw
    latent_loss = kl_weighted + ar1_weighted

    refresh_rate = jnp.mean(refresh_mask)

    total_loss = (
        policy_loss
        + v_loss
        + entropy_loss
        + latent_loss
        + gate_policy_loss
        + gate_entropy_loss
        + gate_refresh_reg_loss
    )

    return total_loss, {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "v_loss": v_loss,
        "entropy_loss": entropy_loss,
        "total_latent_loss": latent_loss,
        "latent_kl_loss": kl_weighted,
        "latent_ar1_loss": ar1_weighted,
        "latent_kl_weight": current_kl_weight,
        "latent_ar1_weight": current_ar1_weight,
        "gate_policy_loss": gate_policy_loss,
        "gate_entropy_loss": gate_entropy_loss,
        "gate_refresh_rate_loss": gate_refresh_reg_loss,
        "gate_entropy": gate_entropy,
        "refresh_rate": refresh_rate,
        "mean_segment_length": mean_segment_length,
        "refresh_pairs": refresh_pairs,
    }


__all__ = [
    "TemporalPPONetworkParams",
    "compute_temporal_ppo_loss",
    "create_ramp_schedule",
]
