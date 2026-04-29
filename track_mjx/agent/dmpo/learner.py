"""DMPO learner state and constructors.

Task 9: state container + initial-state factory + optimizer factory.
Task 10: distributional Bellman target (`compute_categorical_target`).
Task 11: single SGD step (policy + critic + dual updates).
"""
import functools
from typing import Any, Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import optax
import rlax

from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.losses import MPO, MPOParams, clip_mpo_params
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


# ---------------------------------------------------------------------------
# Task 11: Single SGD step.
#
# Mirrors the orchestration in acme/agents/jax/mpo/learning.py (around
# lines 529-650): compute critic loss as cross-entropy against the projected
# Bellman target, compute policy + dual losses via the vendored MPO loss
# (which jointly optimizes policy log-probs and the Lagrange dual variables),
# then apply optimizer updates and (on schedule) hard-copy online -> target.
#
# Deviations from Acme worth flagging:
#   * Single-step TD target: we use only the t=0 transition of each batch
#     trajectory (Acme builds an n-step return over the [T-1] sequence axis).
#     Revisit when n-step bootstrapping is needed.
#   * Deterministic next-action for the critic target: we use the target
#     policy's mode (mean) instead of averaging Q over N samples. Acme's
#     `_compute_targets` averages over an N-axis of policy-sampled actions;
#     using the mean is lower-variance for now and avoids an extra sample-axis.
#   * Critic input action: `batch["action"]` is the raw pre-tanh Gaussian
#     sample stored by the rollout (Task 12). The critic does NOT see the
#     post-tanh bound action — the MPO loss likewise rates raw samples.
#   * Hard target updates via `jax.lax.cond` rather than `optax.periodic_update`
#     (the former is jit-friendly without bringing in optax's helper, which
#     under the hood does the same thing).
# ---------------------------------------------------------------------------


def _critic_loss_fn(
    critic_params: Any,
    nets: DMPONetworks,
    obs_t0: jnp.ndarray,
    action_t0: jnp.ndarray,
    target_probs: jnp.ndarray,
) -> jnp.ndarray:
    """Cross-entropy critic loss: -Σ target_probs * log_softmax(online_logits).

    Args:
      critic_params: online critic params.
      nets: DMPO networks.
      obs_t0: [B, obs_dim] observations at t=0.
      action_t0: [B, action_dim] raw (pre-tanh) actions stored at t=0.
      target_probs: [B, num_atoms] projected Bellman target (stop-gradient'd).
    """
    dist = jax.vmap(nets.critic.apply, in_axes=(None, 0, 0))(
        critic_params, obs_t0, action_t0
    )
    logits = dist.logits_parameter()
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    # target_probs already stop-gradient'd by the caller. Sum over atoms,
    # mean over batch.
    return -(target_probs * log_probs).sum(axis=-1).mean()


def _policy_loss_fn(
    policy_params: Any,
    dual_params: MPOParams,
    nets: DMPONetworks,
    obs_t0: jnp.ndarray,
    target_policy_params: Any,
    target_critic_params: Any,
    cfg: DMPOConfig,
    key: jax.Array,
):
    """E-step + M-step + dual losses via the vendored MPO loss.

    1. Sample `cfg.num_samples` actions from the TARGET policy.
    2. Evaluate the TARGET critic's mean Q-value on each.
    3. Compute online + target distributions for the MPO loss.
    4. Call MPO loss; it does the rest internally.

    Returns (loss_scalar, MPOStats). Loss is squeezed from shape (1,) to
    scalar for `value_and_grad`.
    """
    online_dist = jax.vmap(nets.policy.apply, in_axes=(None, 0))(
        policy_params, obs_t0
    )
    target_dist = jax.vmap(nets.policy.apply, in_axes=(None, 0))(
        target_policy_params, obs_t0
    )
    # Sample [N, B, A] from the target policy (target_dist has batch dim B).
    sampled = target_dist.sample(sample_shape=(cfg.num_samples,), seed=key)

    # Q-values via the target critic. Vmap over (N, B): for each sampled
    # action [B, A], compute the per-batch target critic distribution and
    # take its real-valued mean.
    def _q_mean_for_n(actions_n: jnp.ndarray) -> jnp.ndarray:
        # actions_n: [B, A] -> [B] real-valued mean Q.
        dist = jax.vmap(nets.critic.apply, in_axes=(None, 0, 0))(
            target_critic_params, obs_t0, actions_n
        )
        return dist.mean()

    q_values = jax.vmap(_q_mean_for_n)(sampled)  # [N, B]
    # Q-values are evaluated under target params; Acme stops gradients
    # explicitly via `compute_weights_and_temperature_loss`, but as a
    # belt-and-braces measure we wrap in stop_gradient here (q_values are
    # used both for the E-step weights and for the temperature loss).
    q_values = jax.lax.stop_gradient(q_values)

    loss_module = _build_loss(cfg)
    loss, stats = loss_module(
        params=dual_params,
        online_action_distribution=online_dist,
        target_action_distribution=target_dist,
        actions=sampled,
        q_values=q_values,
    )
    # `loss` from the MPO module is shape (1,) (sum of per-dim alpha terms
    # broadcasts to [1] via the temperature dim); squeeze for scalar grad.
    return loss.squeeze(), stats


def sgd_step(
    state: TrainingState,
    batch: Dict[str, jnp.ndarray],
    nets: DMPONetworks,
    optimizers: Tuple[optax.GradientTransformation, ...],
    cfg: DMPOConfig,
) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
    """One DMPO SGD step.

    Args:
      state: current TrainingState (Task 9).
      batch: dict with keys observation [B, T, obs], action [B, T, act],
        reward [B, T], discount [B, T], next_observation [B, T, obs].
        Only the t=0 slice is used for now (single-step TD).
      nets: DMPONetworks.
      optimizers: (policy_opt, critic_opt, dual_opt) — built ONCE outside
        and threaded through here. Re-creating optimizers inside would break
        jit caching.
      cfg: DMPOConfig.

    Returns:
      (new_state, metrics) where metrics is a flat dict of jnp scalars.
    """
    pol_opt, crit_opt, dual_opt = optimizers
    rng, k_pol = jax.random.split(state.rng)

    # First-cut: only use t=0 of each trajectory. Trajectory dimension is
    # reserved for future n-step return computation.
    obs_t0 = batch["observation"][:, 0, :]
    act_t0 = batch["action"][:, 0, :]
    rew_t0 = batch["reward"][:, 0]
    disc_t0 = batch["discount"][:, 0]
    next_obs_t0 = batch["next_observation"][:, 0, :]

    # ------------------------------------------------------------------
    # 1) Critic target: deterministic next-action via target policy mean.
    # `compute_categorical_target` expects a [B, T, ...] layout, so we
    # add a singleton T axis and squeeze it back out afterward. The whole
    # branch is wrapped in stop_gradient (no flow through target nets).
    # ------------------------------------------------------------------
    target_policy_dist_next = jax.vmap(nets.policy.apply, in_axes=(None, 0))(
        state.target_policy_params, next_obs_t0
    )
    next_action = target_policy_dist_next.mode()
    target_probs = jax.lax.stop_gradient(
        compute_categorical_target(
            nets,
            state.target_critic_params,
            next_obs_t0[:, None, :],   # [B, 1, obs]
            next_action[:, None, :],   # [B, 1, act]
            rew_t0[:, None],           # [B, 1]
            disc_t0[:, None],          # [B, 1]
            cfg,
        ).squeeze(axis=1)              # back to [B, num_atoms]
    )

    # ------------------------------------------------------------------
    # 2) Critic update.
    # ------------------------------------------------------------------
    crit_loss, crit_grads = jax.value_and_grad(_critic_loss_fn)(
        state.critic_params, nets, obs_t0, act_t0, target_probs,
    )
    crit_updates, new_crit_opt_state = crit_opt.update(
        crit_grads, state.critic_opt_state, state.critic_params,
    )
    new_critic_params = optax.apply_updates(state.critic_params, crit_updates)

    # ------------------------------------------------------------------
    # 3) Policy + dual update (combined MPO loss, gradients to both).
    # ------------------------------------------------------------------
    (pol_loss, stats), pol_dual_grads = jax.value_and_grad(
        _policy_loss_fn, argnums=(0, 1), has_aux=True,
    )(
        state.policy_params,
        state.dual_params,
        nets,
        obs_t0,
        state.target_policy_params,
        state.target_critic_params,
        cfg,
        k_pol,
    )
    pol_grads, dual_grads = pol_dual_grads

    pol_updates, new_pol_opt_state = pol_opt.update(
        pol_grads, state.policy_opt_state, state.policy_params,
    )
    dual_updates, new_dual_opt_state = dual_opt.update(
        dual_grads, state.dual_opt_state, state.dual_params,
    )
    new_pol_params = optax.apply_updates(state.policy_params, pol_updates)
    new_dual_params = optax.apply_updates(state.dual_params, dual_updates)
    # Project dual params to feasible (positive) set per Acme's
    # `_dual_clip_fn`: floor each log-variable at -18.
    new_dual_params = clip_mpo_params(
        new_dual_params, per_dim_constraining=cfg.per_dim_constraining,
    )

    # ------------------------------------------------------------------
    # 4) Hard target updates on schedule (jax.lax.cond, jit-friendly).
    # ------------------------------------------------------------------
    new_steps = state.steps + 1
    new_target_pol = jax.lax.cond(
        (new_steps % cfg.target_policy_update_period) == 0,
        lambda _: new_pol_params,
        lambda _: state.target_policy_params,
        operand=None,
    )
    new_target_crit = jax.lax.cond(
        (new_steps % cfg.target_critic_update_period) == 0,
        lambda _: new_critic_params,
        lambda _: state.target_critic_params,
        operand=None,
    )

    new_state = TrainingState(
        policy_params=new_pol_params,
        critic_params=new_critic_params,
        target_policy_params=new_target_pol,
        target_critic_params=new_target_crit,
        dual_params=new_dual_params,
        policy_opt_state=new_pol_opt_state,
        critic_opt_state=new_crit_opt_state,
        dual_opt_state=new_dual_opt_state,
        steps=new_steps,
        rng=rng,
    )

    metrics = {
        "policy_loss": pol_loss,
        "critic_loss": crit_loss,
        # MPOStats fields (see losses.MPOStats).
        "loss_policy": stats.loss_policy,
        "loss_alpha": stats.loss_alpha,
        "loss_temperature": stats.loss_temperature,
        "dual_alpha_mean": stats.dual_alpha_mean,
        "dual_alpha_stddev": stats.dual_alpha_stddev,
        "dual_temperature": stats.dual_temperature,
        "kl_q_rel": stats.kl_q_rel,
        "kl_mean_rel": jnp.mean(stats.kl_mean_rel),
        "kl_stddev_rel": jnp.mean(stats.kl_stddev_rel),
        "q_min": stats.q_min,
        "q_max": stats.q_max,
        "pi_stddev_min": stats.pi_stddev_min,
        "pi_stddev_max": stats.pi_stddev_max,
        "pi_stddev_cond": stats.pi_stddev_cond,
        # Convenience: surface log_temperature as a scalar.
        "log_temperature": new_dual_params.log_temperature.squeeze(),
    }
    return new_state, metrics
