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
from brax.training.acme import running_statistics, specs
from tensorflow_probability.substrates import jax as tfp

from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.losses import MPO, MPOParams, clip_mpo_params
from track_mjx.agent.dmpo.networks import DMPONetworks
from track_mjx.agent.observation_utils import (
    DictRunningStatisticsState,
    init_dict_normalizer,
    normalize_dict_obs,
)

tfd = tfp.distributions


class TrainingState(NamedTuple):
    policy_params: Any
    critic_params: Any
    target_policy_params: Any
    target_critic_params: Any
    dual_params: MPOParams
    policy_opt_state: Any
    critic_opt_state: Any
    dual_opt_state: Any
    normalizer_params: Any   # NEW: DictRunningStatisticsState for dict obs,
                             # flat RunningStatisticsState for flat obs.
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
    """Initialize all params, dual variables, optimizer states, and normalizer.

    ``env_spec`` may be either a flat-obs spec (with ``obs_size``) or a
    dict-obs spec (with ``obs_template`` -- a pytree of jnp arrays). The
    normalizer is initialized to match: ``DictRunningStatisticsState`` for
    dict obs, flat ``RunningStatisticsState`` for flat obs.
    """
    rng, k_pol, k_crit = jax.random.split(rng, 3)
    if "obs_template" in env_spec:
        obs_dummy = env_spec["obs_template"]
        normalizer_params = init_dict_normalizer(obs_dummy)
    else:
        obs_dummy = jnp.zeros((env_spec["obs_size"],))
        normalizer_params = running_statistics.init_state(
            specs.Array((env_spec["obs_size"],), jnp.float32)
        )
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
        normalizer_params=normalizer_params,
        steps=jnp.zeros((), jnp.int32),
        rng=rng,
    )


def _normalize_obs(obs, normalizer_params):
    """Dispatch normalization based on normalizer type.

    DictRunningStatisticsState → per-key normalize_dict_obs.
    Flat RunningStatisticsState → running_statistics.normalize.

    The branch is resolved at jit-trace time (Python type check), so this
    is JIT-safe.
    """
    if isinstance(normalizer_params, DictRunningStatisticsState):
        return normalize_dict_obs(obs, normalizer_params)
    return running_statistics.normalize(obs, normalizer_params)


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
      discounts: shape [B, T] -- the per-step not_done mask. γ is applied
        inside this function via cfg.discount; callers should pass the raw
        not_done mask (this is what rollout.py stores).
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
    bellman_atoms = (
        rewards[..., None]
        + cfg.discount * discounts[..., None] * atoms[None, None, :]
    )

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
    anchor_mu_imit: Any = None,
    anchor_log_std_imit: Any = None,
    step: jnp.ndarray = jnp.int32(0),
):
    """E-step + M-step + dual losses + optional kl-anchor loss term.

    When ``cfg.kl_anchor_alpha > 0`` AND ``anchor_mu_imit`` is provided,
    augments the loss with ``-alpha * mean(exp(-w * KL(pi_theta ‖ pi_imit)))``
    where KL is closed-form Gaussian on pre-tanh logits. The MPO loss
    class itself is untouched; the term is added here.

    The ``step`` argument is the current SGD step index used to drive the
    linear-decay schedule for ``w``: when ``cfg.kl_anchor_decay_sgd_steps > 0``,
    ``w`` decays linearly from ``cfg.kl_anchor_w`` at ``step=0`` to
    ``cfg.kl_anchor_w_floor`` at ``step >= cfg.kl_anchor_decay_sgd_steps``
    (clamped at the floor afterwards).
    """
    online_dist = jax.vmap(nets.policy.apply, in_axes=(None, 0))(
        policy_params, obs_t0
    )
    target_dist = jax.vmap(nets.policy.apply, in_axes=(None, 0))(
        target_policy_params, obs_t0
    )
    sampled = target_dist.sample(sample_shape=(cfg.num_samples,), seed=key)

    def _q_mean_for_n(actions_n: jnp.ndarray) -> jnp.ndarray:
        dist = jax.vmap(nets.critic.apply, in_axes=(None, 0, 0))(
            target_critic_params, obs_t0, actions_n
        )
        return dist.mean()

    q_values = jax.vmap(_q_mean_for_n)(sampled)
    q_values = jax.lax.stop_gradient(q_values)

    loss_module = _build_loss(cfg)
    loss, stats = loss_module(
        params=dual_params,
        online_action_distribution=online_dist,
        target_action_distribution=target_dist,
        actions=sampled,
        q_values=q_values,
    )

    # Optional kl-anchor loss term. We skip the entire branch (including
    # the KL computation, which would otherwise add wasted compute) when
    # alpha == 0 OR the anchor params are absent.
    anchor_kl_mean = jnp.float32(0.0)
    anchor_reward_mean = jnp.float32(0.0)
    anchor_loss_term = jnp.float32(0.0)
    # Default w_now to the static `kl_anchor_w` so the metric is meaningful
    # even when the anchor branch doesn't execute.
    anchor_w_now = jnp.float32(cfg.kl_anchor_w)
    if (
        cfg.kl_anchor_alpha != 0.0 or cfg.kl_anchor_beta_linear != 0.0
    ) and anchor_mu_imit is not None:
        from track_mjx.agent.dmpo.kl_anchor_utils import pretanh_gaussian_kl
        # Linear-decay schedule for w. `cfg.kl_anchor_decay_sgd_steps` is a
        # static Python int from the dataclass, so the branch is selected at
        # JIT trace time (NOT via jax.lax.cond).
        if cfg.kl_anchor_decay_sgd_steps > 0:
            progress = jnp.minimum(
                step.astype(jnp.float32) / float(cfg.kl_anchor_decay_sgd_steps),
                1.0,
            )
            w_now = (
                cfg.kl_anchor_w
                + (cfg.kl_anchor_w_floor - cfg.kl_anchor_w) * progress
            )
        else:
            w_now = jnp.float32(cfg.kl_anchor_w)
        mu_theta = online_dist.mean()
        log_std_theta = jnp.log(online_dist.stddev())
        kl = pretanh_gaussian_kl(
            mu_theta, log_std_theta, anchor_mu_imit, anchor_log_std_imit
        )
        anchor_reward = jnp.exp(-w_now * kl)
        anchor_kl_mean = jnp.mean(kl)
        anchor_reward_mean = jnp.mean(anchor_reward)
        # Saturating term (historical): gradient ~ exp(-w*kl), dies when kl is large.
        anchor_loss_term = -cfg.kl_anchor_alpha * anchor_reward_mean
        # Linear term: gradient is constant in kl, so it keeps braking at the
        # large-kl operating point where the exp form has switched itself off.
        if cfg.kl_anchor_beta_linear != 0.0:
            anchor_loss_term = (
                anchor_loss_term + cfg.kl_anchor_beta_linear * anchor_kl_mean
            )
        anchor_w_now = w_now
        loss = loss + anchor_loss_term

    stats = stats._replace(
        anchor_kl_mean=anchor_kl_mean,
        anchor_reward_mean=anchor_reward_mean,
        anchor_loss_term=anchor_loss_term,
        anchor_w_now=anchor_w_now,
    )
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
    # Recurrent (short-BPTT) learner path. The branch resolves at Python
    # trace time, so with the default rnn_bptt_length=0 the FF body below is
    # bit-identical for every existing arm (no retrace, no new ops).
    if getattr(cfg, "rnn_bptt_length", 0) > 0:
        return _sgd_step_rnn(state, batch, nets, optimizers, cfg)

    pol_opt, crit_opt, dual_opt = optimizers
    rng, k_pol = jax.random.split(state.rng)

    # First-cut: only use t=0 of each trajectory. Trajectory dimension is
    # reserved for future n-step return computation.
    # Use jax.tree.map so dict / pytree observations (vision pipeline) work
    # alongside the flat-array path; for a flat array the map is a no-op
    # except for the slice itself.
    obs_t0 = jax.tree.map(lambda x: x[:, 0], batch["observation"])
    act_t0 = batch["action"][:, 0, :]

    # ---- n-step return -------------------------------------------------------
    # `cfg.n_step = 50` was declared in DMPOConfig from the start and NEVER read:
    # this learner did single-step TD on the t=0 slice while discarding the other
    # 49 timesteps of every sampled sequence. That is the structural asymmetry
    # against the PPO reference, which gets its advantage from REALIZED returns
    # via GAE -- returns that directly encode "this action made you fall 40 steps
    # later". DMPO's Q is its ENTIRE policy-improvement signal and was a one-step
    # bootstrap through a critic that cannot observe torso height or orientation.
    #
    # With m_i = prod_{j<i} d_j (still-alive mask entering step i):
    #     R_n = sum_{i<n} gamma^i m_i r_i          bootstrap coeff = gamma^n m_n
    # `compute_categorical_target` forms `rewards + cfg.discount * discounts * z`,
    # so passing discounts = gamma^(n-1) * m_n reproduces gamma^n * m_n exactly.
    #
    # n_steps = 1 reduces to EXACTLY the previous expressions (m = [1], g = [1],
    # R_n = r_0, D = d_0), so the default is bit-identical to the old behaviour.
    _T = batch["reward"].shape[1]
    # Compressed schema (cfg.store_next_observation=False): the batch has no
    # `next_observation`; the bootstrap state s_{t+n} is observation[:, n],
    # bit-identical to next_observation[:, n-1] in flashbax trajectory storage
    # (the auto-reset wrapper swaps obs on the terminal step itself, and the
    # time axis is continuous across rollout adds). That index must exist, so
    # n is capped at T-1 -- size sequence_length = n_step + 1 to keep the full
    # horizon. Detected from the batch (not cfg) so tests and mixed callers
    # get the right behaviour per schema.
    _has_next = "next_observation" in batch
    _n_cap = _T if _has_next else _T - 1
    if _n_cap < 1:
        raise ValueError(
            f"sequence_length={_T} too short for the compressed schema: "
            "bootstrapping from observation[:, n] needs sequence_length >= 2."
        )
    n_steps = int(min(cfg.n_step, _n_cap)) if getattr(cfg, "use_n_step", False) else 1
    _d = batch["discount"][:, :n_steps]                       # [B, n]
    _r = batch["reward"][:, :n_steps]                         # [B, n]
    _alive = jnp.cumprod(_d, axis=1)                          # m_{i+1}
    _m = jnp.concatenate([jnp.ones_like(_d[:, :1]), _alive[:, :-1]], axis=1)
    _g = cfg.discount ** jnp.arange(n_steps, dtype=_r.dtype)
    rew_t0 = jnp.sum(_g[None, :] * _m * _r, axis=1)           # R_n
    disc_t0 = (cfg.discount ** (n_steps - 1)) * _alive[:, -1]  # gamma^(n-1) m_n
    if _has_next:
        next_obs_t0 = jax.tree.map(
            lambda x: x[:, n_steps - 1], batch["next_observation"]
        )
    else:
        next_obs_t0 = jax.tree.map(lambda x: x[:, n_steps], batch["observation"])

    # NEW: normalize observations using the current normalizer state.
    # Normalizer is updated only in rollout (not here) to mirror Brax PPO.
    obs_t0 = _normalize_obs(obs_t0, state.normalizer_params)
    next_obs_t0 = _normalize_obs(next_obs_t0, state.normalizer_params)

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
    # Add a singleton T axis to every leaf so [B, ...] -> [B, 1, ...]. Works
    # uniformly for flat arrays and dict / pytree observations.
    next_obs_t0_BT = jax.tree.map(lambda x: x[:, None], next_obs_t0)
    target_probs = jax.lax.stop_gradient(
        compute_categorical_target(
            nets,
            state.target_critic_params,
            next_obs_t0_BT,             # [B, 1, ...] per leaf
            next_action[:, None, :],    # [B, 1, act]
            rew_t0[:, None],            # [B, 1]
            disc_t0[:, None],           # [B, 1]
            cfg,
        ).squeeze(axis=1)               # back to [B, num_atoms]
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
    # Optional kl-anchor params from the batch (set by the kl-anchor entry's
    # rollout via extra_state_extras). Absent for non-kl-anchor entries.
    anchor_mu = batch.get("anchor_mu_imit") if isinstance(batch, dict) else None
    anchor_ls = batch.get("anchor_log_std_imit") if isinstance(batch, dict) else None
    # Fail-loud guard: if the user opted into the KL-anchor (cfg.kl_anchor_alpha
    # != 0) but the batch is missing the anchor keys, the policy would silently
    # fall back to MPO-only and the run would look like "anchor disabled" with
    # no warning. Catch this at trace time before training spends GPU on it.
    if cfg.kl_anchor_alpha != 0.0 and anchor_mu is None:
        raise ValueError(
            "cfg.kl_anchor_alpha != 0 but batch has no 'anchor_mu_imit'. "
            "Check that the kl-anchor entry's transition_template includes "
            "anchor_mu_imit / anchor_log_std_imit AND that "
            "extra_state_extras=('anchor_mu_imit','anchor_log_std_imit') "
            "is passed to run_training_loop / make_fused_train_step / "
            "collect_rollout."
        )
    if anchor_mu is not None:
        anchor_mu = anchor_mu[:, 0, :]
    if anchor_ls is not None:
        anchor_ls = anchor_ls[:, 0, :]

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
        anchor_mu,
        anchor_ls,
        state.steps,
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
        new_dual_params, getattr(cfg, "min_log_temperature", -18.0)
    )

    # ------------------------------------------------------------------
    # 3b) Critic-only warmup gate. For the first `critic_warmup_sgd_steps`
    # updates the policy and duals are held at their (warm-started) values
    # while the critic trains -- same tree_map/lax.select pattern as the
    # can_sample gate in train_dmpo_step.py. MUST run before the hard
    # target-policy copy in (4), or a warmup-period target update would
    # copy gated-off params into target_policy_params. Duals are gated too:
    # with the policy frozen, the alpha duals would otherwise decay against
    # a trivially-satisfied KL constraint and the temperature would descend
    # against a half-trained critic's Q. Default 0 = gate always open,
    # bit-identical to the pre-gate behaviour.
    # ------------------------------------------------------------------
    warmup_n = int(getattr(cfg, "critic_warmup_sgd_steps", 0) or 0)
    if warmup_n > 0:
        policy_updates_open = state.steps >= warmup_n
        (new_pol_params, new_dual_params, new_pol_opt_state, new_dual_opt_state) = (
            jax.tree_util.tree_map(
                lambda new, old: jax.lax.select(policy_updates_open, new, old),
                (new_pol_params, new_dual_params, new_pol_opt_state, new_dual_opt_state),
                (state.policy_params, state.dual_params,
                 state.policy_opt_state, state.dual_opt_state),
            )
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
        normalizer_params=state.normalizer_params,   # unchanged in SGD
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
        "mean_action_norm": stats.mean_action_norm,
        "max_action_norm": stats.max_action_norm,
        # KL-anchor stats (zero-valued if cfg.kl_anchor_alpha == 0).
        "anchor_kl_mean": stats.anchor_kl_mean,
        "anchor_reward_mean": stats.anchor_reward_mean,
        "anchor_loss_term": stats.anchor_loss_term,
        "kl_anchor/w_now": stats.anchor_w_now,
        # Convenience: surface log_temperature as a scalar.
        "log_temperature": new_dual_params.log_temperature.squeeze(),
    }
    return new_state, metrics


# ---------------------------------------------------------------------------
# Recurrent (short-BPTT) learner path (2026-08-20).
#
# R2D2-style stored-state + short BPTT, reached from `sgd_step` when
# cfg.rnn_bptt_length > 0. Each sampled sequence of T = L + n transitions
# yields L loss points: the online policy is unrolled L steps WITH gradients
# from the stored window-start hidden, the MPO loss is applied at every
# unrolled point, and each point t bootstraps with its own n-step return
# from observation[:, t + n] paired with the STORED hidden at t + n (a
# single-step, stop-gradient target apply -- staleness there only biases the
# targets, never the gradients). The critic stays feed-forward. All [B, L]
# loss points are folded into the batch axis ([B*L]) and fed through the same
# loss helpers as the FF path; the MPO loss module is batch-shape agnostic
# (every reduction is a mean over -- or linear in a mean over -- the batch
# axis; verified in test_learner_rnn), so folding L into B changes nothing
# but the sample count per update.
#
# The learner's ONLY contract with the recurrent networks is the raw-apply
# signature `policy.apply(params, obs, hidden, method="raw") ->
# (mu, scale, new_hidden)` with hidden a tuple of per-layer [B, H_l] arrays.
# It deliberately does NOT import RecurrentPolicyMeta or any networks-side
# type: the batch (schema + stored hidden) is its single source of truth.
# ---------------------------------------------------------------------------


def _reset_hidden(hidden: Any, done: jnp.ndarray) -> Any:
    """Zero every hidden leaf where ``done`` is set.

    ``hidden`` is the tuple-of-per-layer-arrays convention ([B, H_l] leaves);
    ``done`` is [B] (bool). The reshape appends singleton axes so the same
    helper broadcasts over any trailing hidden shape. Mirrors the rollout's
    post-env.step reset: auto-reset swaps obs on the terminal step itself, so
    the hidden consumed with the post-reset obs at t+1 must be zeros.
    """

    def _r(h):
        d = done.reshape(done.shape + (1,) * (h.ndim - done.ndim))
        return jnp.where(d, jnp.zeros_like(h), h)

    return jax.tree.map(_r, hidden)


def _per_point_nstep_returns(
    rewards: jnp.ndarray,
    discounts: jnp.ndarray,
    gamma: float,
    length: int,
    n: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Vectorized per-loss-point n-step returns over a [B, T] window.

    For each start t < length over the gathered window idx[t, j] = t + j
    (j < n), with m_j = prod_{i<j} d_{t+i} (still-alive mask entering t+j):

        R_n[t]  = sum_j gamma^j * m_j * r_{t+j}
        disc[t] = gamma^(n-1) * prod_{j<n} d_{t+j}

    These are EXACTLY the FF ``sgd_step`` n-step expressions applied at every
    t instead of only t=0 -- length=1 reduces to them bit-for-bit (guarded in
    test_learner_rnn). ``compute_categorical_target`` then forms
    ``r + gamma * disc * z``, reproducing the full gamma^n * m_n bootstrap
    coefficient. Windows crossing a done are handled by the alive mask: the
    reward sum stops accumulating and the bootstrap coefficient is zeroed.

    Returns:
      (R_n, disc), both [B, length].
    """
    idx = jnp.arange(length)[:, None] + jnp.arange(n)[None, :]  # [L, n]
    d_win = discounts[:, idx]                                   # [B, L, n]
    r_win = rewards[:, idx]                                     # [B, L, n]
    alive = jnp.cumprod(d_win, axis=-1)                         # m_{j+1}
    m = jnp.concatenate(
        [jnp.ones_like(d_win[..., :1]), alive[..., :-1]], axis=-1
    )
    g = gamma ** jnp.arange(n, dtype=r_win.dtype)
    returns = jnp.sum(g[None, None, :] * m * r_win, axis=-1)    # [B, L]
    disc = (gamma ** (n - 1)) * alive[..., -1]                  # [B, L]
    return returns, disc


def _unroll_policy_raw(
    policy_params: Any,
    nets: DMPONetworks,
    obs_tm: Any,
    done_tm: jnp.ndarray,
    h0: Any,
) -> Tuple[jnp.ndarray, jnp.ndarray, Any]:
    """Unroll the recurrent policy over a time-major window via ``lax.scan``.

    Args:
      policy_params: params to apply (online or target -- the FUNCTION is
        differentiable throughout; the caller decides whether gradients flow).
      nets: networks whose policy exposes ``raw(obs, hidden)`` returning
        arrays (mu, scale, new_hidden) -- scan-safe, unlike a tfd object.
      obs_tm: normalized obs pytree, leaves [L, B, ...] (time-major).
      done_tm: [L, B] bool, ``discount == 0`` per step. The hidden entering
        step t+1 is zeroed where done_tm[t] -- the exact mirror of the
        rollout-side reset, so recomputed hiddens track stored ones.
      h0: hidden entering step 0, tuple of [B, H_l]. It comes from replay
        storage and is DATA: gradients stop at it by construction.

    Returns:
      (mu, scale, h_pre): mu/scale [L, B, A]; h_pre the tuple of [L, B, H_l]
      pre-step hiddens actually consumed at each t (h_pre[0] == h0), used for
      the staleness diagnostic and the |h| metric.
    """

    def _step(h, xs):
        obs_t, done_t = xs
        # Per-env vmap over unbatched obs -- the codebase convention (the
        # vision CNN and the policy module are written for unbatched input).
        mu_t, scale_t, h_new = jax.vmap(
            lambda o, hh: nets.policy.apply(policy_params, o, hh, method="raw")
        )(obs_t, h)
        return _reset_hidden(h_new, done_t), (mu_t, scale_t, h)

    _, (mu, scale, h_pre) = jax.lax.scan(_step, h0, (obs_tm, done_tm))
    return mu, scale, h_pre


def _policy_loss_fn_rnn(
    policy_params: Any,
    dual_params: MPOParams,
    nets: DMPONetworks,
    obs_tm: Any,
    done_tm: jnp.ndarray,
    h0: Any,
    h_stored_tm: Any,
    obs_flat: Any,
    target_mu: jnp.ndarray,
    target_scale: jnp.ndarray,
    target_critic_params: Any,
    cfg: DMPOConfig,
    key: jax.Array,
    anchor_mu_imit: Any = None,
    anchor_log_std_imit: Any = None,
    step: jnp.ndarray = jnp.int32(0),
):
    """Recurrent counterpart of ``_policy_loss_fn``: BPTT unroll + MPO on [B*L].

    The online unroll lives INSIDE this function so that ``value_and_grad``
    backpropagates through all L cell applications -- that is the entire
    point of L > 1. The target dist (mu/scale, precomputed by the caller from
    its own unroll and stop-gradient'd) and the q_values are gradient-free,
    exactly as in the FF path. The [B, L] loss points are folded into the
    batch axis: dists with batch shape [B*L], samples [N, B*L, A]; the
    flattening order (b*L + t) matches ``obs_flat`` and the anchor slices.

    Aux is ``(stats, rnn_metrics)``: the staleness diagnostic must come from
    the online unroll's recomputed pre-step hiddens, which only exist here.

    Returns:
      (loss, (stats, rnn_metrics)).
    """
    length, bsz = done_tm.shape
    mu, scale, h_pre = _unroll_policy_raw(
        policy_params, nets, obs_tm, done_tm, h0
    )
    action_size = mu.shape[-1]
    # [L, B, A] -> [B, L, A] -> [B*L, A]: transpose FIRST so the flattened
    # ordering matches obs_flat / target_mu (both built from [B, L] slices).
    mu_bl = jnp.swapaxes(mu, 0, 1).reshape(bsz * length, action_size)
    scale_bl = jnp.swapaxes(scale, 0, 1).reshape(bsz * length, action_size)
    online_dist = tfd.MultivariateNormalDiag(loc=mu_bl, scale_diag=scale_bl)
    target_dist = tfd.MultivariateNormalDiag(
        loc=target_mu, scale_diag=target_scale
    )
    sampled = target_dist.sample(sample_shape=(cfg.num_samples,), seed=key)

    def _q_mean_for_n(actions_n: jnp.ndarray) -> jnp.ndarray:
        dist = jax.vmap(nets.critic.apply, in_axes=(None, 0, 0))(
            target_critic_params, obs_flat, actions_n
        )
        return dist.mean()

    q_values = jax.vmap(_q_mean_for_n)(sampled)
    q_values = jax.lax.stop_gradient(q_values)

    loss_module = _build_loss(cfg)
    loss, stats = loss_module(
        params=dual_params,
        online_action_distribution=online_dist,
        target_action_distribution=target_dist,
        actions=sampled,
        q_values=q_values,
    )

    # Optional kl-anchor loss term -- same math as the FF `_policy_loss_fn`,
    # applied to the flattened [B*L] batch (anchor inputs are the caller's
    # [:, :L] slices flattened in the same order as the dists).
    anchor_kl_mean = jnp.float32(0.0)
    anchor_reward_mean = jnp.float32(0.0)
    anchor_loss_term = jnp.float32(0.0)
    anchor_w_now = jnp.float32(cfg.kl_anchor_w)
    if (
        cfg.kl_anchor_alpha != 0.0 or cfg.kl_anchor_beta_linear != 0.0
    ) and anchor_mu_imit is not None:
        from track_mjx.agent.dmpo.kl_anchor_utils import pretanh_gaussian_kl
        if cfg.kl_anchor_decay_sgd_steps > 0:
            progress = jnp.minimum(
                step.astype(jnp.float32) / float(cfg.kl_anchor_decay_sgd_steps),
                1.0,
            )
            w_now = (
                cfg.kl_anchor_w
                + (cfg.kl_anchor_w_floor - cfg.kl_anchor_w) * progress
            )
        else:
            w_now = jnp.float32(cfg.kl_anchor_w)
        mu_theta = online_dist.mean()
        log_std_theta = jnp.log(online_dist.stddev())
        kl = pretanh_gaussian_kl(
            mu_theta, log_std_theta, anchor_mu_imit, anchor_log_std_imit
        )
        anchor_reward = jnp.exp(-w_now * kl)
        anchor_kl_mean = jnp.mean(kl)
        anchor_reward_mean = jnp.mean(anchor_reward)
        anchor_loss_term = -cfg.kl_anchor_alpha * anchor_reward_mean
        if cfg.kl_anchor_beta_linear != 0.0:
            anchor_loss_term = (
                anchor_loss_term + cfg.kl_anchor_beta_linear * anchor_kl_mean
            )
        anchor_w_now = w_now
        loss = loss + anchor_loss_term

    stats = stats._replace(
        anchor_kl_mean=anchor_kl_mean,
        anchor_reward_mean=anchor_reward_mean,
        anchor_loss_term=anchor_loss_term,
        anchor_w_now=anchor_w_now,
    )

    # RNN diagnostics (aux-only; stop_gradient for clarity). Staleness is the
    # normalized squared distance between the hidden the unroll recomputes at
    # each t and the hidden the rollout stored there -- how much the replayed
    # hidden has drifted from what the CURRENT policy would produce. t=0
    # contributes zero by construction (h_pre[0] == h0 == stored).
    staleness = jnp.mean(
        jnp.stack(
            [
                jnp.mean(
                    jnp.sum((re - st) ** 2, axis=-1)
                    / (jnp.sum(st**2, axis=-1) + 1e-6)
                )
                for re, st in zip(h_pre, h_stored_tm)
            ]
        )
    )
    hidden_abs_mean = jnp.mean(
        jnp.stack([jnp.mean(jnp.abs(h)) for h in h_pre])
    )
    rnn_metrics = {
        "rnn/hidden_staleness": jax.lax.stop_gradient(staleness),
        "rnn/hidden_abs_mean": jax.lax.stop_gradient(hidden_abs_mean),
    }
    return loss.squeeze(), (stats, rnn_metrics)


def _sgd_step_rnn(
    state: TrainingState,
    batch: Dict[str, jnp.ndarray],
    nets: DMPONetworks,
    optimizers: Tuple[optax.GradientTransformation, ...],
    cfg: DMPOConfig,
) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
    """One recurrent (short-BPTT) DMPO SGD step. See the section header above.

    Batch schema (compressed, recurrent): observation [B, T, ...] pytree,
    action [B, T, A], reward/discount [B, T], policy_hidden tuple of
    [B, T, H_l] (store_dtype, cast to f32 here), optional anchor keys
    [B, T, ...]; T == cfg.rnn_bptt_length + cfg.n_step.
    """
    L = int(cfg.rnn_bptt_length)
    n = int(cfg.n_step)

    # Fail-loud schema checks (Python/trace time, mirroring the anchor_mu
    # guard in the FF path): a silently-wrong window layout would train on
    # misaligned targets with no error, so every assumption the window math
    # relies on is asserted before any GPU time is spent.
    if "policy_hidden" not in batch:
        raise ValueError(
            "cfg.rnn_bptt_length > 0 but batch has no 'policy_hidden'. The "
            "recurrent learner needs the stored per-step hidden: add "
            "'policy_hidden' to the transition template AND pass "
            "recurrent_meta to collect_rollout so it is populated."
        )
    T = batch["reward"].shape[1]
    if T != L + n:
        raise ValueError(
            f"sequence_length={T} != rnn_bptt_length + n_step = {L} + {n} = "
            f"{L + n}. Every loss point t < L needs a full n-step reward "
            "window and a bootstrap state at observation[:, t + n]."
        )
    if not getattr(cfg, "use_n_step", False):
        raise ValueError(
            "cfg.rnn_bptt_length > 0 requires use_n_step=True: the recurrent "
            "learner's per-point returns are n-step by construction and "
            "there is no single-step fallback."
        )
    if "next_observation" in batch:
        raise ValueError(
            "cfg.rnn_bptt_length > 0 requires the compressed replay schema "
            "(store_next_observation=False): bootstrap states are read from "
            "observation[:, t + n], and a duplicate next_observation field "
            "would double the observation memory for no benefit."
        )

    pol_opt, crit_opt, dual_opt = optimizers
    rng, k_pol = jax.random.split(state.rng)
    B = batch["reward"].shape[0]

    # (1) Normalize only the two windows that are ever consumed: the unroll
    # window [:, :L] and the bootstrap window [:, n:n+L] (2L of T steps;
    # 40 of 120 for the goal arm). Slicing BEFORE the normalize matters for
    # the uint8-stored vision leaf: normalizing the full [B, T, ...] window
    # would dequantize ~T/(2L)x more f32 vision than is used if XLA fails to
    # fuse the downstream slices through normalize_dict_obs's per-key ops.
    # Normalization is elementwise over trailing feature dims, so
    # slice-then-normalize is exact. (For L > n the windows overlap and a few
    # steps normalize twice -- cheap elementwise work, still a net win.)
    obs_L = _normalize_obs(
        jax.tree.map(lambda x: x[:, :L], batch["observation"]),
        state.normalizer_params,
    )
    next_obs = _normalize_obs(
        jax.tree.map(lambda x: x[:, n : n + L], batch["observation"]),
        state.normalizer_params,
    )

    # (2) Stored hidden (store_dtype in replay, e.g. f16) -> f32 for compute.
    # h0 seeds the unroll; h_boot pairs with the bootstrap obs at t + n.
    # Both are DATA -- no gradient flows into the replay buffer.
    h_all = jax.tree.map(
        lambda h: h.astype(jnp.float32), batch["policy_hidden"]
    )
    h0 = jax.tree.map(lambda h: h[:, 0], h_all)
    h_boot = jax.tree.map(lambda h: h[:, n : n + L], h_all)      # [B, L, H]
    h_stored_tm = jax.tree.map(
        lambda h: jnp.swapaxes(h[:, :L], 0, 1), h_all            # [L, B, H]
    )

    # Time-major unroll window. done == (discount == 0) mirrors the rollout's
    # reset trigger exactly, so the recomputed hiddens track the stored ones.
    obs_tm = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), obs_L)
    done_tm = jnp.swapaxes(batch["discount"][:, :L] == 0, 0, 1)  # [L, B]

    # Flattened [B*L] views for the critic and the MPO loss. Row-major
    # reshape of [B, L, ...] gives ordering b*L + t -- every flattened
    # quantity below uses the same [B, L]-first layout.
    obs_flat = jax.tree.map(
        lambda x: x.reshape((B * L,) + x.shape[2:]), obs_L
    )
    act_flat = batch["action"][:, :L].reshape(B * L, -1)

    # (4) Target unroll: same scan with target params, gradient-free.
    tgt_mu, tgt_scale, _ = _unroll_policy_raw(
        state.target_policy_params, nets, obs_tm, done_tm, h0
    )
    action_size = tgt_mu.shape[-1]
    tgt_mu = jax.lax.stop_gradient(
        jnp.swapaxes(tgt_mu, 0, 1).reshape(B * L, action_size)
    )
    tgt_scale = jax.lax.stop_gradient(
        jnp.swapaxes(tgt_scale, 0, 1).reshape(B * L, action_size)
    )

    # (6) Per-point n-step returns, [B, L] each.
    rew_bl, disc_bl = _per_point_nstep_returns(
        batch["reward"], batch["discount"], cfg.discount, L, n
    )

    # (7) Critic target: bootstrap from observation[:, n:n+L] paired with the
    # STORED hidden at t + n -- a single-step target apply (no unroll needed:
    # the stored hidden already encodes the history up to t + n).
    # Deterministic next-action via mu (== mode for a diag Gaussian), the FF
    # path's target_policy_dist_next.mode() analog. next_obs was normalized
    # above in (1).
    next_mu, _, _ = jax.vmap(
        jax.vmap(
            lambda o, hh: nets.policy.apply(
                state.target_policy_params, o, hh, method="raw"
            )
        )
    )(next_obs, h_boot)
    next_action = jax.lax.stop_gradient(next_mu)                 # [B, L, A]
    target_probs = jax.lax.stop_gradient(
        compute_categorical_target(
            nets,
            state.target_critic_params,
            next_obs,
            next_action,
            rew_bl,
            disc_bl,
            cfg,
        )
    ).reshape(B * L, cfg.num_atoms)

    # (8) Critic update on the flattened [B*L] loss points -- the same
    # `_critic_loss_fn` body as the FF path, just a bigger batch.
    crit_loss, crit_grads = jax.value_and_grad(_critic_loss_fn)(
        state.critic_params, nets, obs_flat, act_flat, target_probs,
    )
    crit_updates, new_crit_opt_state = crit_opt.update(
        crit_grads, state.critic_opt_state, state.critic_params,
    )
    new_critic_params = optax.apply_updates(state.critic_params, crit_updates)

    # (9) Policy + dual update. Anchor terms are the [:, :L] slices flattened
    # in the same [B, L] order as the dists; guard mirrors the FF path.
    anchor_mu = batch.get("anchor_mu_imit") if isinstance(batch, dict) else None
    anchor_ls = batch.get("anchor_log_std_imit") if isinstance(batch, dict) else None
    if cfg.kl_anchor_alpha != 0.0 and anchor_mu is None:
        raise ValueError(
            "cfg.kl_anchor_alpha != 0 but batch has no 'anchor_mu_imit'. "
            "Check that the kl-anchor entry's transition_template includes "
            "anchor_mu_imit / anchor_log_std_imit AND that "
            "extra_state_extras=('anchor_mu_imit','anchor_log_std_imit') "
            "is passed to run_training_loop / make_fused_train_step / "
            "collect_rollout."
        )
    if anchor_mu is not None:
        anchor_mu = anchor_mu[:, :L].reshape(B * L, -1)
    if anchor_ls is not None:
        anchor_ls = anchor_ls[:, :L].reshape(B * L, -1)

    (pol_loss, (stats, rnn_metrics)), pol_dual_grads = jax.value_and_grad(
        _policy_loss_fn_rnn, argnums=(0, 1), has_aux=True,
    )(
        state.policy_params,
        state.dual_params,
        nets,
        obs_tm,
        done_tm,
        h0,
        h_stored_tm,
        obs_flat,
        tgt_mu,
        tgt_scale,
        state.target_critic_params,
        cfg,
        k_pol,
        anchor_mu,
        anchor_ls,
        state.steps,
    )
    pol_grads, dual_grads = pol_dual_grads

    # (10) Optimizer updates / warmup gate / target schedule: duplicated from
    # the FF `sgd_step` body (deliberately, to keep that body diff-free).
    pol_updates, new_pol_opt_state = pol_opt.update(
        pol_grads, state.policy_opt_state, state.policy_params,
    )
    dual_updates, new_dual_opt_state = dual_opt.update(
        dual_grads, state.dual_opt_state, state.dual_params,
    )
    new_pol_params = optax.apply_updates(state.policy_params, pol_updates)
    new_dual_params = optax.apply_updates(state.dual_params, dual_updates)
    new_dual_params = clip_mpo_params(
        new_dual_params, getattr(cfg, "min_log_temperature", -18.0)
    )

    warmup_n = int(getattr(cfg, "critic_warmup_sgd_steps", 0) or 0)
    if warmup_n > 0:
        policy_updates_open = state.steps >= warmup_n
        (new_pol_params, new_dual_params, new_pol_opt_state, new_dual_opt_state) = (
            jax.tree_util.tree_map(
                lambda new, old: jax.lax.select(policy_updates_open, new, old),
                (new_pol_params, new_dual_params, new_pol_opt_state, new_dual_opt_state),
                (state.policy_params, state.dual_params,
                 state.policy_opt_state, state.dual_opt_state),
            )
        )

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
        normalizer_params=state.normalizer_params,   # unchanged in SGD
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
        "mean_action_norm": stats.mean_action_norm,
        "max_action_norm": stats.max_action_norm,
        # KL-anchor stats (zero-valued if cfg.kl_anchor_alpha == 0).
        "anchor_kl_mean": stats.anchor_kl_mean,
        "anchor_reward_mean": stats.anchor_reward_mean,
        "anchor_loss_term": stats.anchor_loss_term,
        "kl_anchor/w_now": stats.anchor_w_now,
        # Convenience: surface log_temperature as a scalar.
        "log_temperature": new_dual_params.log_temperature.squeeze(),
        # Recurrent-only diagnostics from the online unroll.
        "rnn/hidden_staleness": rnn_metrics["rnn/hidden_staleness"],
        "rnn/hidden_abs_mean": rnn_metrics["rnn/hidden_abs_mean"],
    }
    return new_state, metrics
