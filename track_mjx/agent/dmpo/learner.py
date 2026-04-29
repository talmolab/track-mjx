"""DMPO learner state and constructors.

Task 9: state container + initial-state factory + optimizer factory.
Task 10: distributional Bellman target (`compute_categorical_target`).
Task 11 will add the actual SGD step that consumes this state.
"""
import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
import rlax

from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.losses import MPO, MPOParams
from track_mjx.agent.dmpo.networks import DMPONetworks


class TrainingState(NamedTuple):
    policy_params: Any
    critic_params: Any
    target_policy_params: Any
    target_critic_params: Any
    dual_params: MPOParams
    policy_opt_state: Any
    critic_opt_state: Any
    dual_opt_state: Any
    steps: jnp.ndarray
    rng: jax.Array


def make_optimizers(cfg: DMPOConfig):
    """Return (policy_opt, critic_opt, dual_opt) optax transformations.

    Policy and critic get gradient-norm clipping at cfg.grad_clip; the dual
    optimizer does not (Acme keeps it unclipped).
    """
    policy = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adam(cfg.policy_lr),
    )
    critic = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adam(cfg.critic_lr),
    )
    dual = optax.adam(cfg.dual_lr)
    return policy, critic, dual


def _build_loss(cfg: DMPOConfig) -> MPO:
    """Construct the MPO loss module from config (used both at init and step)."""
    return MPO(
        epsilon=cfg.epsilon,
        epsilon_mean=cfg.epsilon_mean,
        epsilon_stddev=cfg.epsilon_stddev,
        epsilon_penalty=cfg.epsilon_penalty,
        init_log_temperature=cfg.init_log_temperature,
        init_log_alpha_mean=cfg.init_log_alpha_mean,
        init_log_alpha_stddev=cfg.init_log_alpha_stddev,
        per_dim_constraining=cfg.per_dim_constraining,
        action_penalization=cfg.action_penalization,
    )


def init_training_state(
    rng: jax.Array,
    nets: DMPONetworks,
    env_spec: dict,
    cfg: DMPOConfig,
) -> TrainingState:
    """Initialize all params, dual variables, and optimizer states."""
    rng, k_pol, k_crit = jax.random.split(rng, 3)
    obs_dummy = jnp.zeros((env_spec["obs_size"],))
    act_dummy = jnp.zeros((env_spec["action_size"],))

    policy_params = nets.policy.init(k_pol, obs_dummy)
    critic_params = nets.critic.init(k_crit, obs_dummy, act_dummy)

    loss_fn = _build_loss(cfg)
    dual_params = loss_fn.init_params(env_spec["action_size"], jnp.float32)

    pol_opt, crit_opt, dual_opt = make_optimizers(cfg)
    return TrainingState(
        policy_params=policy_params,
        critic_params=critic_params,
        target_policy_params=policy_params,
        target_critic_params=critic_params,
        dual_params=dual_params,
        policy_opt_state=pol_opt.init(policy_params),
        critic_opt_state=crit_opt.init(critic_params),
        dual_opt_state=dual_opt.init(dual_params),
        steps=jnp.zeros((), jnp.int32),
        rng=rng,
    )


def compute_categorical_target(
    nets: DMPONetworks,
    target_critic_params: Any,
    next_obs: jnp.ndarray,
    next_action: jnp.ndarray,
    rewards: jnp.ndarray,
    discounts: jnp.ndarray,
    cfg: DMPOConfig,
) -> jnp.ndarray:
    """C51 Bellman target: project (r + γ·support) onto fixed atoms.

    Mirrors Acme's `acme/agents/jax/mpo/learning.py:482-517` (CATEGORICAL
    branch). Deviations:
      * No `tx_pair` (input/output reward transform). vnl-ray doesn't use
        it; adding it post-hoc is mechanical.
      * Single sample (no `N`-action averaging — Acme's `z_target` is a
        mean over an `N`-axis of policy-sampled actions). Caller is
        expected to pass a single `next_action` per (B, T).

    Args:
      nets: DMPO networks (we use the critic for the target distribution).
      target_critic_params: parameters of the target critic.
      next_obs: shape [B, T, obs_dim].
      next_action: shape [B, T, action_dim].
      rewards: shape [B, T].
      discounts: shape [B, T] (γ * not_done).
      cfg: DMPOConfig (vmin/vmax/num_atoms).

    Returns:
      target_probs: shape [B, T, num_atoms]. Each row sums to 1 (after
        L2 projection onto the fixed atom grid).
    """
    # Apply target critic over [B, T, ...] (vmap over both axes). Acme uses
    # a single vmap because their critic_head_apply is already over a flat
    # time axis (T-1), but here we keep the shape as [B, T, ...] coming in
    # so we vmap twice.
    apply = jax.vmap(jax.vmap(nets.critic.apply, in_axes=(None, 0, 0)),
                     in_axes=(None, 0, 0))
    target_dist = apply(target_critic_params, next_obs, next_action)
    target_logits = target_dist.logits_parameter()
    target_probs = jax.nn.softmax(target_logits, axis=-1)

    atoms = jnp.linspace(cfg.vmin, cfg.vmax, cfg.num_atoms)
    # Bellman: y = r + γ * z. Shape: [B, T, num_atoms].
    bellman_atoms = rewards[..., None] + discounts[..., None] * atoms[None, None, :]

    # rlax.categorical_l2_project(z_p, probs, z_q): project `probs` from
    # support `z_p` onto support `z_q`. We wire z_q via partial and vmap
    # the remaining (z_p, probs) over batch and time axes.
    project = functools.partial(rlax.categorical_l2_project, z_q=atoms)
    projected = jax.vmap(jax.vmap(project))(bellman_atoms, target_probs)
    return projected
