"""PPO loss function for recurrent shared-vision networks.

This module provides the loss computation for recurrent PPO training with a
shared CNN+GRU backbone (``RecurrentSharedVisionModule``). It is the recurrent
analog of ``compute_shared_vision_ppo_loss`` in ``ff_ppo/losses.py``.

Key differences from the feedforward version:
- Temporal scanning: instead of processing each sample independently, we scan
  through time with a GRU hidden state via ``jax.lax.scan``.
- Hidden state reset: the GRU carry is reset to zeros at episode boundaries
  (where ``done`` is True).
- Bootstrap value: computed by running the shared module one more step on the
  final next observation using the final hidden state from the scan.
- No VAE/KL/AR1 losses: this is a non-variational architecture.

The data flow in the loss:
- ``data.extras["policy_extras"]["raw_action"]`` -- actions sampled during rollout
- ``data.extras["policy_extras"]["log_prob"]`` -- log probs from rollout policy
- ``data.extras["policy_extras"]["initial_policy_hidden"]`` -- hidden states at
  start of each trajectory (no time dimension)
- ``data.extras["state_extras"]["truncation"]`` -- truncation flags
- ``data.discount`` -- 0 when episode ends, 1 otherwise
"""

import jax
import jax.numpy as jnp
from brax.training import types

from track_mjx.agent.ff_ppo.losses import compute_gae
from track_mjx.agent.observation_utils import normalize_dict_obs
from track_mjx.agent.recurrent_ppo.losses import (
    RecurrentPPONetworkParams,
    _extract_initial_hidden,
)
from track_mjx.agent.recurrent_ppo.networks import (
    RecurrentPPONetworks,
    reset_hidden_on_done,
)
from track_mjx.agent.recurrent_ppo.recurrent_vision_networks import (
    RecurrentSharedVisionModule,
)


def compute_recurrent_shared_vision_ppo_loss(
    params: RecurrentPPONetworkParams,
    normalizer_params,
    data: types.Transition,
    rng: jnp.ndarray,
    step: int,
    recurrent_ppo_network: RecurrentPPONetworks,
    shared_module: RecurrentSharedVisionModule,
    entropy_cost: float = 1e-3,
    discounting: float = 0.97,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.2,
    normalize_advantage: bool = True,
    vf_coefficient: float = 0.5,
) -> tuple[jnp.ndarray, types.Metrics]:
    """Compute PPO loss for recurrent shared-vision networks.

    Processes data in [T, B, ...] format via ``jax.lax.scan`` over the time
    dimension, using the shared CNN+GRU module for both policy and value
    outputs in a single forward pass per timestep.

    All parameters live in ``params.policy``; ``params.value`` is empty (the
    value head is part of the shared module). Both policy and value loss
    gradients flow through the shared CNN+GRU backbone.

    Args:
        params: Recurrent PPO network parameters. ``params.policy`` contains
            the full shared module (CNN + GRU + policy head + value head).
        normalizer_params: Running statistics for observation normalization.
        data: Transition batch with shape [B, T, ...] (batch-major).
            Internally swapped to [T, B, ...] for temporal scanning.
            Required extra fields:
            - ``data.extras["state_extras"]["truncation"]``
            - ``data.extras["policy_extras"]["raw_action"]``
            - ``data.extras["policy_extras"]["log_prob"]``
            - ``data.extras["policy_extras"]["initial_policy_hidden"]``
        rng: JAX random key.
        step: Current training step (required by the recurrent PPO training
            loop interface).
        recurrent_ppo_network: Recurrent PPO network container (used for
            the parametric action distribution).
        shared_module: The ``RecurrentSharedVisionModule`` Flax module.
        entropy_cost: Entropy bonus coefficient (higher = more exploration).
        discounting: Discount factor (gamma) for GAE.
        reward_scaling: Multiplier applied to rewards.
        gae_lambda: GAE lambda parameter.
        clipping_epsilon: PPO clipping range for the policy ratio.
        normalize_advantage: Whether to normalize advantages to zero mean
            and unit standard deviation.
        vf_coefficient: Coefficient for the value function loss.

    Returns:
        Tuple of (total_loss, metrics_dict). The metrics dict has keys:
        ``total_loss``, ``policy_loss``, ``v_loss``, ``entropy_loss``.
    """
    _, _, entropy_key = jax.random.split(rng, 3)
    parametric_action_distribution = (
        recurrent_ppo_network.parametric_action_distribution
    )

    # ------------------------------------------------------------------ #
    # 1. Extract initial_policy_hidden BEFORE swapaxes (it has no time dim)
    # ------------------------------------------------------------------ #
    initial_policy_hidden = data.extras["policy_extras"]["initial_policy_hidden"]

    # ------------------------------------------------------------------ #
    # 2. Swap to time-major: [B, T, ...] -> [T, B, ...]
    # ------------------------------------------------------------------ #
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    # Restore initial_policy_hidden (it was corrupted by the swapaxes
    # because it has no time dimension).
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

    # ------------------------------------------------------------------ #
    # 3. Extract initial hidden state for the GRU
    # ------------------------------------------------------------------ #
    initial_hidden_list = _extract_initial_hidden(
        data.extras["policy_extras"]["initial_policy_hidden"]
    )
    # Single-layer GRU: extract the one carry array [B, gru_hidden_size]
    initial_carry = initial_hidden_list[0]

    # ------------------------------------------------------------------ #
    # 4. Normalize observations once
    # ------------------------------------------------------------------ #
    normalized_obs = normalize_dict_obs(data.observation, normalizer_params)

    # Done flags for hidden state reset: discount < 0.5 means episode ended
    done = data.discount < 0.5  # [T, B]

    # ------------------------------------------------------------------ #
    # 5. Scan through time using the shared module's __call__
    # ------------------------------------------------------------------ #
    def scan_step(carry, inputs):
        """Single timestep through the shared CNN+GRU module.

        Args:
            carry: GRU hidden state [B, gru_hidden_size].
            inputs: Tuple of (obs_t dict, done_t [B]).

        Returns:
            new_carry, (action_params_t, value_t).
        """
        obs_t, done_t = inputs

        # shared_module.__call__ returns (action_params, value, new_carry)
        action_params, value, new_carry = shared_module.apply(
            params.policy,
            obs=obs_t,
            carry=carry,
        )

        # Reset hidden state where episodes ended
        new_carry = reset_hidden_on_done(new_carry, done_t, "gru")

        return new_carry, (action_params, value)

    final_carry, (policy_logits, baseline) = jax.lax.scan(
        scan_step, initial_carry, (normalized_obs, done)
    )
    # policy_logits: [T, B, action_param_size]
    # baseline: [T, B]

    # ------------------------------------------------------------------ #
    # 6. Bootstrap value using final hidden state
    # ------------------------------------------------------------------ #
    last_next_obs = jax.tree_util.tree_map(
        lambda x: x[-1], data.next_observation
    )
    last_next_obs_normalized = normalize_dict_obs(last_next_obs, normalizer_params)

    _, bootstrap_value, _ = shared_module.apply(
        params.policy,
        obs=last_next_obs_normalized,
        carry=final_carry,
    )

    # ------------------------------------------------------------------ #
    # 7. Standard PPO loss computation
    # ------------------------------------------------------------------ #
    rewards = data.reward * reward_scaling
    truncation = data.extras["state_extras"]["truncation"]
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

    # Importance sampling ratio
    rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)

    # Clipped surrogate policy loss
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

    total_loss = policy_loss + v_loss + entropy_loss

    return total_loss, {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "v_loss": v_loss,
        "entropy_loss": entropy_loss,
    }
