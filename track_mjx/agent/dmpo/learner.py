"""DMPO learner state and constructors.

- state container + initial-state factory + optimizer factory;
- distributional Bellman target (`compute_categorical_target`);
- single SGD step (policy + critic + dual updates).
"""

import functools
from collections.abc import Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from brax.training.acme import running_statistics, specs
from tensorflow_probability.substrates import jax as tfp

from track_mjx.agent.dmpo import action_utils
from track_mjx.agent.dmpo.categorical import categorical_l2_project
from track_mjx.agent.dmpo.config import DMPOConfig, validate_alignment
from track_mjx.agent.dmpo.losses import MPO, MPOParams, clip_mpo_params
from track_mjx.agent.dmpo.networks import DMPONetworks
from track_mjx.agent.dmpo.obs_utils import (
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
    normalizer_params: Any  # NEW: DictRunningStatisticsState for dict obs,
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

    # A critic with a privileged input reads obs["privileged"]; it is a
    # critic-only key, so it goes into the critic's dummy obs alone (never
    # the policy's, never the normalizer's). privileged_dim 0 / absent (any
    # critic class without the field) = the historical init, unchanged.
    critic_obs_dummy = obs_dummy
    priv_dim = int(getattr(nets.critic, "privileged_dim", 0) or 0)
    if priv_dim > 0:
        if not isinstance(obs_dummy, Mapping):
            raise ValueError(
                "critic privileged_dim > 0 requires dict observations "
                "(env_spec['obs_template'])."
            )
        critic_obs_dummy = {
            **obs_dummy,
            "privileged": jnp.zeros((priv_dim,), jnp.float32),
        }

    policy_params = nets.policy.init(k_pol, obs_dummy)
    critic_params = nets.critic.init(k_crit, critic_obs_dummy, act_dummy)

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


def critic_action(a: jnp.ndarray, cfg: DMPOConfig) -> jnp.ndarray:
    """The action a critic apply consumes (``cfg.critic_action_input``).

    ``"executed"``: the action the env applies, ``action_utils.bind(a)``
    (tanh, kept inside (-1, 1)), as acme's critic sees the clipped action.
    ``"raw"``: the pre-tanh sample itself (the same object, no new op). The
    MPO loss always works on the raw samples.
    """
    if cfg.critic_action_input == "raw":
        return a
    return action_utils.bind(a)


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


# ---------------------------------------------------------------------------
# Privileged critic input (asymmetric actor-critic).
#
# The critic may see a task-state vector the policy never does. The env writes
# it to state.info[cfg.critic_privileged_key]; the rollout's
# `extra_state_extras` side channel stores it from the PRE-step state, so
# batch[key][:, t] is aligned with observation[:, t]. Every critic apply gets
# it as obs["privileged"], gathered at the time index of the obs that apply
# consumes; policy applies never see the key. All branching is Python /
# trace-time: with the default key "" the helpers return their input object
# and no op is added.
# ---------------------------------------------------------------------------


def _check_privileged_config(nets: DMPONetworks, cfg: DMPOConfig) -> str:
    """Return the active privileged key ("" = off); fail loud on a mismatch.

    Both mismatch directions are silent-or-cryptic otherwise: a key with a
    privileged_dim=0 critic would train a critic that ignores the vector,
    and a privileged critic with no key dies on a bare KeyError.
    """
    key = getattr(cfg, "critic_privileged_key", "") or ""
    dim = int(getattr(nets.critic, "privileged_dim", 0) or 0)
    if key and dim <= 0:
        raise ValueError(
            f"cfg.critic_privileged_key={key!r} but the critic has no "
            "privileged input (critic_privileged_dim=0): the vector would be "
            "silently ignored. Build the networks with critic_privileged_dim "
            "> 0, or clear cfg.critic_privileged_key."
        )
    if dim > 0 and not key:
        raise ValueError(
            f"the critic was built with critic_privileged_dim={dim} but "
            "cfg.critic_privileged_key is empty, so the learner has nothing "
            "to feed obs['privileged']. Set cfg.critic_privileged_key to the "
            "batch key that stores the vector (e.g. 'critic_priv')."
        )
    return key


def _critic_obs(obs: Any, batch: dict[str, Any], key: str, idx: Any) -> Any:
    """Attach the privileged critic input to ``obs`` for a CRITIC apply.

    Args:
      obs: (normalized) obs pytree a critic apply consumes; its leading dims
        must match the gathered vector's ([B] or [B, L]).
      batch: replay batch. ``batch[key]`` is [B, T, D], aligned with
        ``batch["observation"][:, t]``.
      key: ``cfg.critic_privileged_key``; "" = off.
      idx: time index of the state ``obs`` holds. A Python int or slice is
        shared by every row (``batch[key][:, idx]``); an integer array gives
        per-row indices, [B] -> [B, D] or [B, L] -> [B, L, D] (e.g. the
        bootstrap index of each row, or per-row tick positions).

    Returns:
      ``obs`` itself when ``key`` is "" (no new ops on the default path);
      otherwise a new dict ``{**obs, "privileged": gathered}`` (``obs`` is
      not mutated, so the policy-side pytree never carries the key).
    """
    if not key:
        return obs
    if key not in batch:
        raise ValueError(
            f"cfg.critic_privileged_key={key!r} but the batch has no {key!r}. "
            f"Add {key!r} to the transition template AND to "
            "extra_state_extras (run_training_loop / make_fused_train_step / "
            "collect_rollout) so the rollout stores state.info[key]."
        )
    if not isinstance(obs, Mapping):
        raise ValueError(
            "the privileged critic input needs dict observations; got "
            f"{type(obs).__name__}."
        )
    priv = batch[key]
    if isinstance(idx, (int, slice)) and not isinstance(idx, bool):
        gathered = priv[:, idx]
    else:
        idx = jnp.asarray(idx)
        if idx.ndim == 0:
            gathered = priv[:, idx]
        elif idx.ndim in (1, 2) and idx.shape[0] == priv.shape[0]:
            rows = idx.reshape(idx.shape[0], -1)  # [B, L']
            rows = rows.reshape(rows.shape + (1,) * (priv.ndim - 2))
            gathered = jnp.take_along_axis(priv, rows, axis=1)  # [B, L', D]
            gathered = gathered.reshape(idx.shape + priv.shape[2:])
        else:
            raise ValueError(
                "privileged gather indices must be an int, a slice, or an "
                f"integer array of shape [B] or [B, L] with B={priv.shape[0]};"
                f" got shape {idx.shape}."
            )
    return {**obs, "privileged": gathered}


# ---------------------------------------------------------------------------
# Bootstrapping through time limits.
#
# A time-out ends the episode without making its last state terminal. The
# auto-reset wrapper swaps the post-step observation for the next episode's
# first one on the done step, so the true final state s_T is never stored. A
# window that reaches the time-out step k therefore ends one step early: it
# sums the rewards before k and bootstraps from s_k (stored and valid) with
# the full coefficient gamma^k * alive. The window that starts at the time-out
# itself, (s_k -> s_T), has no valid bootstrap state and gets loss weight 0.
# The stored discount stays 1 - done, so real terminations, and every hidden
# reset keyed on discount == 0, behave as before; only the window cut reads
# the separate truncation flag.
# ---------------------------------------------------------------------------


def _require_truncation(batch: Mapping[str, Any]) -> jnp.ndarray:
    """``batch["truncation"]``; fail loud when the replay schema lacks it."""
    if "truncation" not in batch:
        raise ValueError(
            "cfg.bootstrap_on_truncation=True but the replay batch has no "
            "'truncation' field. Add 'truncation': zeros((), float32) to the "
            "transition template (the rollout stores it when the flag is on), "
            "or set bootstrap_on_truncation=False to train on a replay schema "
            "without it."
        )
    return batch["truncation"]


def _cut_windows(
    rewards: jnp.ndarray,
    discounts: jnp.ndarray,
    truncation: jnp.ndarray,
    gamma: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """n-step returns over ``[..., n]`` windows, cut at the first time-out.

    With m_i = prod_{j<i} d_j and k the first index with truncation 1 (n if
    none): R = sum_{i<k} gamma^i m_i r_i, disc = gamma^(k-1) m_k (the target
    applies one more gamma), boot = k, weight = (k > 0). Values past the cut
    or past a termination are selected away with ``where``, so inf or NaN
    stored there cannot leak into R.
    """
    n = rewards.shape[-1]
    is_cut = truncation > 0.5
    k = jnp.where(jnp.any(is_cut, axis=-1), jnp.argmax(is_cut, axis=-1), n)
    k = k.astype(jnp.int32)
    alive = jnp.cumprod(discounts, axis=-1)  # m_{i+1}
    m_full = jnp.concatenate([jnp.ones_like(alive[..., :1]), alive], axis=-1)
    m = m_full[..., :n]
    g = gamma ** jnp.arange(n, dtype=rewards.dtype)
    keep = (jnp.arange(n) < k[..., None]) & (m > 0)
    returns = jnp.sum(jnp.where(keep, g * m * rewards, 0.0), axis=-1)
    m_k = jnp.take_along_axis(m_full, k[..., None], axis=-1)[..., 0]
    disc = (gamma ** (jnp.maximum(k, 1) - 1).astype(rewards.dtype)) * m_k
    weight = (k > 0).astype(rewards.dtype)
    return returns, disc, k, weight


def window_returns(
    rewards: jnp.ndarray,
    discounts: jnp.ndarray,
    truncation: jnp.ndarray,
    gamma: float,
    n: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """n-step return of the window starting at index 0 of each row.

    Args:
      rewards, discounts, truncation: [B, T] with T >= n. ``discounts`` is the
        stored 1 - done; ``truncation`` is 1 on a time-out step.
      gamma: discount factor.
      n: window length.

    Returns:
      (R [B], disc [B], boot_idx int32 [B], weight [B]). With a time-out at
      index k < n: R sums the rewards before k, the bootstrap state is s_k
      (``boot_idx = k``) with coefficient ``gamma * disc = gamma^k * m_k``,
      and the weight is 0 when k == 0 (no valid bootstrap state). Without
      one: the plain n-step values, ``boot_idx = n``, weight 1.
    """
    return _cut_windows(rewards[:, :n], discounts[:, :n], truncation[:, :n], gamma)


def _gather_rows(x: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
    """``x[b, idx[b]]`` for a [B, T, ...] array and an int [B] index."""
    rows = idx.reshape((-1, 1) + (1,) * (x.ndim - 2))
    return jnp.take_along_axis(x, rows, axis=1)[:, 0]


def _state_at(batch: Mapping[str, Any], idx: jnp.ndarray, has_next: bool) -> Any:
    """Raw state s_idx of every row, for an int [B] index in [0, n].

    Compressed schema: ``observation[:, idx]``. With a stored
    ``next_observation`` the same state is ``next_observation[:, idx - 1]``
    (index n may lie past the stored observations), and ``observation[:, 0]``
    where idx == 0.
    """
    if not has_next:
        return jax.tree.map(lambda x: _gather_rows(x, idx), batch["observation"])
    prev = jnp.maximum(idx - 1, 0)

    def _pick(obs, next_obs):
        later = _gather_rows(next_obs, prev)
        first = (idx == 0).reshape((-1,) + (1,) * (later.ndim - 1))
        return jnp.where(first, obs[:, 0], later)

    return jax.tree.map(_pick, batch["observation"], batch["next_observation"])


def compute_categorical_target(
    nets: DMPONetworks,
    target_critic_params: Any,
    next_obs: jnp.ndarray,
    next_action: jnp.ndarray,
    rewards: jnp.ndarray,
    discounts: jnp.ndarray,
    cfg: DMPOConfig,
) -> jnp.ndarray:
    """C51 Bellman target: project (r + gamma·support) onto fixed atoms.

    Mirrors Acme's `acme/agents/jax/mpo/learning.py:482-517` (CATEGORICAL
    branch). Deviations:
      * No `tx_pair` (input/output reward transform). vnl-ray doesn't use
        it; adding it post-hoc is mechanical.
      * Single action per (B, T): this is the ``critic_target_mode=
        "mean_action"`` target. The N-sample mixture acme uses is
        ``bootstrap_samples`` + ``mixture_categorical_target``.

    Args:
      nets: DMPO networks (we use the critic for the target distribution).
      target_critic_params: parameters of the target critic.
      next_obs: shape [B, T, obs_dim].
      next_action: shape [B, T, action_dim].
      rewards: shape [B, T].
      discounts: shape [B, T] -- the per-step not_done mask. gamma is applied
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
    apply = jax.vmap(
        jax.vmap(nets.critic.apply, in_axes=(None, 0, 0)), in_axes=(None, 0, 0)
    )
    target_dist = apply(target_critic_params, next_obs, next_action)
    target_logits = target_dist.logits_parameter()
    target_probs = jax.nn.softmax(target_logits, axis=-1)

    atoms = jnp.linspace(cfg.vmin, cfg.vmax, cfg.num_atoms)
    # Bellman: y = r + gamma * z. Shape: [B, T, num_atoms].
    bellman_atoms = (
        rewards[..., None] + cfg.discount * discounts[..., None] * atoms[None, None, :]
    )

    # categorical_l2_project(z_p, probs, z_q): project `probs` from
    # support `z_p` onto support `z_q`. We wire z_q via partial and vmap
    # the remaining (z_p, probs) over batch and time axes.
    project = functools.partial(categorical_l2_project, z_q=atoms)
    projected = jax.vmap(jax.vmap(project))(bellman_atoms, target_probs)
    return projected


# ---------------------------------------------------------------------------
# N-sample mixture bootstrap target (critic_target_mode="sample_mixture").
#
# As acme's DMPO: draw num_samples actions from the target policy at the
# bootstrap state, rate each with the target critic, and average the
# categorical distributions in probability space before the Bellman shift and
# projection. The projection is linear in the probabilities, so averaging
# first equals averaging the per-sample projections.
# ---------------------------------------------------------------------------


def _has_vision(obs: Any) -> bool:
    return isinstance(obs, Mapping) and "vision" in obs


def _rate_samples(
    nets: DMPONetworks,
    target_critic_params: Any,
    crit_obs: Any,
    target_dist: Any,
    cfg: DMPOConfig,
    key: jax.Array,
    sample_eps: jnp.ndarray | None,
    use_map: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Draw [N, B, A] actions from ``target_dist`` and the target critic's
    logits [N, B, atoms] for each (rated on ``critic_action`` of the sample;
    the returned actions are the raw samples).

    ``use_map`` maps the N axis sequentially (one B-sized critic batch at a
    time) instead of vmapping it, which keeps a vision critic's activation
    peak at B images; the values are the same.
    """
    if sample_eps is None:
        actions = target_dist.sample(sample_shape=(cfg.num_samples,), seed=key)
    else:
        actions = target_dist.mean()[None] + target_dist.stddev()[None] * sample_eps

    def _logits(actions_n):
        dist = jax.vmap(nets.critic.apply, in_axes=(None, 0, 0))(
            target_critic_params, crit_obs, critic_action(actions_n, cfg)
        )
        return dist.logits_parameter()

    logits = jax.lax.map(_logits, actions) if use_map else jax.vmap(_logits)(actions)
    return actions, logits


def bootstrap_samples(
    nets: DMPONetworks,
    target_policy_params: Any,
    target_critic_params: Any,
    boot_obs: Any,
    boot_crit_obs: Any,
    cfg: DMPOConfig,
    key: jax.Array,
    sample_eps: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Target-policy samples at the bootstrap state and their critic logits.

    Args:
      boot_obs: normalized bootstrap obs, leaves [B, ...] (the policy's view).
      boot_crit_obs: the same state as the critic sees it (``boot_obs`` plus
        the privileged input gathered at the bootstrap index, when that is
        on).
      key: sampling key.
      sample_eps: optional [N, B, A] standard-normal draws; samples are then
        ``mu + sigma * eps``. A test-only injection point: production passes
        None and samples with ``key``.

    Returns:
      (actions [N, B, A], target_dist_logits [N, B, atoms]). The N axis is
      mapped sequentially when the critic reads vision.
    """
    target_dist = jax.vmap(nets.policy.apply, in_axes=(None, 0))(
        target_policy_params, boot_obs
    )
    return _rate_samples(
        nets,
        target_critic_params,
        boot_crit_obs,
        target_dist,
        cfg,
        key,
        sample_eps,
        use_map=_has_vision(boot_crit_obs),
    )


def mixture_categorical_target(
    logits: jnp.ndarray,
    rewards: jnp.ndarray,
    discounts: jnp.ndarray,
    cfg: DMPOConfig,
) -> jnp.ndarray:
    """Project r + gamma * disc * z under the mean of the N sampled
    distributions: logits [N, B, atoms], rewards/discounts [B] -> [B, atoms].
    """
    probs = jnp.mean(jax.nn.softmax(logits, axis=-1), axis=0)
    atoms = jnp.linspace(cfg.vmin, cfg.vmax, cfg.num_atoms)
    bellman_atoms = rewards[:, None] + cfg.discount * discounts[:, None] * atoms
    project = functools.partial(categorical_l2_project, z_q=atoms)
    return jax.vmap(project)(bellman_atoms, probs)


# ---------------------------------------------------------------------------
# Single SGD step.
#
# Mirrors the orchestration in acme/agents/jax/mpo/learning.py (around
# lines 529-650): compute critic loss as cross-entropy against the projected
# Bellman target, compute policy + dual losses via the vendored MPO loss
# (which jointly optimizes policy log-probs and the Lagrange dual variables),
# then apply optimizer updates and (on schedule) hard-copy online -> target.
#
# The acme-alignment switches (DMPOConfig) select between acme's choice and
# the historical one at trace time:
#   * critic_target_mode: the bootstrap distribution is the N-sample mixture
#     ("sample_mixture", acme) or one evaluation at the target policy's mean
#     action ("mean_action").
#   * bootstrap_on_truncation: n-step windows are cut at time-outs and
#     bootstrap from the last stored state (see `window_returns`).
#   * critic_action_input: every critic apply (critic loss, bootstrap, E-step
#     Q-values) sees the executed, tanh-bound action ("executed", acme's
#     clipped critic input) or the stored pre-tanh sample ("raw"). The MPO
#     loss rates the raw samples either way (see `critic_action`).
# Deviations from Acme worth flagging:
#   * n-step windows come from uniformly sampled stored sequences (the window
#     starting at t=0 of each), not from an n-step adder.
#   * The policy is a Gaussian over pre-tanh actions, squashed to the env by
#     tanh; acme's is an unsquashed Gaussian whose samples the env clips.
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
    weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Cross-entropy critic loss: -Σ target_probs * log_softmax(online_logits).

    Args:
      critic_params: online critic params.
      nets: DMPO networks.
      obs_t0: [B, obs_dim] observations at t=0.
      action_t0: [B, action_dim] actions at t=0, as the critic consumes them
        (see ``critic_action``).
      target_probs: [B, num_atoms] projected Bellman target (stop-gradient'd).
      weights: optional nonnegative per-row weights, [B]. The mean over rows
        becomes ``sum(w * x) / max(sum(w), 1)``: a zero-weight row is excluded and
        an all-zero batch gives a finite 0 loss with zero gradient. ``None``
        (default) is the plain mean, bit-identical to before.
    """
    dist = jax.vmap(nets.critic.apply, in_axes=(None, 0, 0))(
        critic_params, obs_t0, action_t0
    )
    logits = dist.logits_parameter()
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    # target_probs already stop-gradient'd by the caller. Sum over atoms,
    # mean over batch.
    per_row = (target_probs * log_probs).sum(axis=-1)
    if weights is None:
        return -per_row.mean()
    # Multiply by the reciprocal, as jnp.mean's lowering does (see
    # losses._reciprocal_mass): all-ones weights then match the plain mean bitwise.
    return -(jnp.sum(weights * per_row) * (1.0 / jnp.maximum(jnp.sum(weights), 1.0)))


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
    critic_obs_t0: Any = None,
    batch_weights: jnp.ndarray | None = None,
):
    """E-step + M-step + dual losses + optional kl-anchor loss term.

    ``critic_obs_t0`` is the obs the target critic rates the sampled actions
    on -- ``obs_t0`` plus the privileged key when that is on (see
    ``_critic_obs``). None (default) uses ``obs_t0`` itself, so the policy and
    the critic read the same pytree exactly as before.

    ``batch_weights`` ([B], optional) weights the states in every MPO batch
    mean (see ``MPO.__call__``); None is the plain mean.

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
    online_dist = jax.vmap(nets.policy.apply, in_axes=(None, 0))(policy_params, obs_t0)
    target_dist = jax.vmap(nets.policy.apply, in_axes=(None, 0))(
        target_policy_params, obs_t0
    )
    sampled = target_dist.sample(sample_shape=(cfg.num_samples,), seed=key)
    critic_obs = obs_t0 if critic_obs_t0 is None else critic_obs_t0

    def _q_mean_for_n(actions_n: jnp.ndarray) -> jnp.ndarray:
        dist = jax.vmap(nets.critic.apply, in_axes=(None, 0, 0))(
            target_critic_params, critic_obs, critic_action(actions_n, cfg)
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
        batch_weights=batch_weights,
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
                cfg.kl_anchor_w + (cfg.kl_anchor_w_floor - cfg.kl_anchor_w) * progress
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
    batch: dict[str, jnp.ndarray],
    nets: DMPONetworks,
    optimizers: tuple[optax.GradientTransformation, ...],
    cfg: DMPOConfig,
    *,
    sample_eps: jnp.ndarray | None = None,
) -> tuple[TrainingState, dict[str, jnp.ndarray]]:
    """One DMPO SGD step.

    Args:
      state: current TrainingState.
      batch: dict with keys observation [B, T, obs], action [B, T, act],
        reward [B, T], discount [B, T], next_observation [B, T, obs].
        Only the t=0 slice is used for now (single-step TD).
      nets: DMPONetworks.
      optimizers: (policy_opt, critic_opt, dual_opt) — built ONCE outside
        and threaded through here. Re-creating optimizers inside would break
        jit caching.
      cfg: DMPOConfig.
      sample_eps: test-only standard-normal draws for the bootstrap samples
        of the mixture target ([N, B, A]; [N, B*L, A] on the recurrent path).
        None (production) samples with the step's key.

    Returns:
      (new_state, metrics) where metrics is a flat dict of jnp scalars.
    """
    # Recurrent (short-BPTT) learner path. The branch resolves at Python
    # trace time, so with the default rnn_bptt_length=0 the FF body below is
    # bit-identical for every existing arm (no retrace, no new ops).
    if getattr(cfg, "rnn_bptt_length", 0) > 0:
        return _sgd_step_rnn(state, batch, nets, optimizers, cfg, sample_eps=sample_eps)

    # Check the alignment switches and resolve policy_loss_state="auto" (trace
    # time; a bad value fails before any device time is spent).
    cfg = validate_alignment(cfg, recurrent=False)

    pol_opt, crit_opt, dual_opt = optimizers
    rng, k_pol = jax.random.split(state.rng)
    use_mixture = cfg.critic_target_mode == "sample_mixture"
    if use_mixture:
        # A separate key for the bootstrap samples, derived so the policy
        # loss's own key stays what it was.
        k_tgt = jax.random.fold_in(k_pol, 1)

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
    if cfg.bootstrap_on_truncation:
        # Windows cut one step early at a time-out (see `window_returns`): the
        # bootstrap state is per row, and a window starting at the time-out
        # is excluded from both losses through `loss_w`.
        rew_t0, disc_t0, boot_idx, loss_w = window_returns(
            batch["reward"],
            batch["discount"],
            _require_truncation(batch),
            cfg.discount,
            n_steps,
        )
        next_obs_t0 = _state_at(batch, boot_idx, _has_next)
    else:
        _d = batch["discount"][:, :n_steps]  # [B, n]
        _r = batch["reward"][:, :n_steps]  # [B, n]
        _alive = jnp.cumprod(_d, axis=1)  # m_{i+1}
        _m = jnp.concatenate([jnp.ones_like(_d[:, :1]), _alive[:, :-1]], axis=1)
        _g = cfg.discount ** jnp.arange(n_steps, dtype=_r.dtype)
        rew_t0 = jnp.sum(_g[None, :] * _m * _r, axis=1)  # R_n
        disc_t0 = (cfg.discount ** (n_steps - 1)) * _alive[:, -1]  # gamma^(n-1) m_n
        if _has_next:
            next_obs_t0 = jax.tree.map(
                lambda x: x[:, n_steps - 1], batch["next_observation"]
            )
        else:
            next_obs_t0 = jax.tree.map(lambda x: x[:, n_steps], batch["observation"])
        boot_idx = n_steps
        loss_w = None

    # NEW: normalize observations using the current normalizer state.
    # Normalizer is updated only in rollout (not here) to mirror Brax PPO.
    obs_t0 = _normalize_obs(obs_t0, state.normalizer_params)
    next_obs_t0 = _normalize_obs(next_obs_t0, state.normalizer_params)

    # Critic-only views with the privileged input attached (after the
    # normalize, which would drop the key; the vector is used raw). With the
    # key off these ARE obs_t0 / next_obs_t0. batch[key][:, t] pairs with
    # observation[:, t], so the bootstrap state -- observation[:, n], or the
    # identical next_observation[:, n-1] -- pairs with index n in BOTH
    # schemas. Policy applies keep reading obs_t0 / next_obs_t0.
    priv_key = _check_privileged_config(nets, cfg)
    if priv_key and n_steps >= _T:
        raise ValueError(
            f"privileged critic input: the bootstrap state is t={n_steps} but "
            f"the batch only stores {priv_key!r} for t < {_T} (no next-step "
            "copy exists). Use sequence_length >= n_step + 1."
        )
    crit_obs_t0 = _critic_obs(obs_t0, batch, priv_key, 0)
    crit_next_obs_t0 = _critic_obs(next_obs_t0, batch, priv_key, boot_idx)

    # ------------------------------------------------------------------
    # 1) Critic target. The whole branch is wrapped in stop_gradient (no flow
    # through target nets).
    # ------------------------------------------------------------------
    if use_mixture:
        # N target-policy samples at the bootstrap state, mixed in
        # probability space.
        _, boot_logits = bootstrap_samples(
            nets,
            state.target_policy_params,
            state.target_critic_params,
            next_obs_t0,
            crit_next_obs_t0,
            cfg,
            k_tgt,
            sample_eps,
        )
        target_probs = jax.lax.stop_gradient(
            mixture_categorical_target(boot_logits, rew_t0, disc_t0, cfg)
        )
    else:
        # Deterministic next-action via target policy mean.
        # `compute_categorical_target` expects a [B, T, ...] layout, so we
        # add a singleton T axis and squeeze it back out afterward.
        target_policy_dist_next = jax.vmap(nets.policy.apply, in_axes=(None, 0))(
            state.target_policy_params, next_obs_t0
        )
        next_action = target_policy_dist_next.mode()
        # Add a singleton T axis to every leaf so [B, ...] -> [B, 1, ...].
        # Works uniformly for flat arrays and dict / pytree observations.
        next_obs_t0_BT = jax.tree.map(lambda x: x[:, None], crit_next_obs_t0)
        target_probs = jax.lax.stop_gradient(
            compute_categorical_target(
                nets,
                state.target_critic_params,
                next_obs_t0_BT,  # [B, 1, ...] per leaf
                critic_action(next_action, cfg)[:, None, :],  # [B, 1, act]
                rew_t0[:, None],  # [B, 1]
                disc_t0[:, None],  # [B, 1]
                cfg,
            ).squeeze(axis=1)  # back to [B, num_atoms]
        )

    # ------------------------------------------------------------------
    # 2) Critic update.
    # ------------------------------------------------------------------
    crit_loss, crit_grads = jax.value_and_grad(_critic_loss_fn)(
        state.critic_params,
        nets,
        crit_obs_t0,
        critic_action(act_t0, cfg),
        target_probs,
        loss_w,
    )
    crit_updates, new_crit_opt_state = crit_opt.update(
        crit_grads,
        state.critic_opt_state,
        state.critic_params,
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
        _policy_loss_fn,
        argnums=(0, 1),
        has_aux=True,
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
        # None when off: the loss reads obs_t0 for the critic, as before.
        crit_obs_t0 if priv_key else None,
        batch_weights=loss_w,
    )
    pol_grads, dual_grads = pol_dual_grads

    pol_updates, new_pol_opt_state = pol_opt.update(
        pol_grads,
        state.policy_opt_state,
        state.policy_params,
    )
    dual_updates, new_dual_opt_state = dual_opt.update(
        dual_grads,
        state.dual_opt_state,
        state.dual_params,
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
            jax.tree.map(
                lambda new, old: jax.lax.select(policy_updates_open, new, old),
                (
                    new_pol_params,
                    new_dual_params,
                    new_pol_opt_state,
                    new_dual_opt_state,
                ),
                (
                    state.policy_params,
                    state.dual_params,
                    state.policy_opt_state,
                    state.dual_opt_state,
                ),
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
        normalizer_params=state.normalizer_params,  # unchanged in SGD
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
#
# Strided ("tick") BPTT. A slow-core head (networks_kl_anchor_slow) carries a
# phase leaf at `nets.recurrent_meta.phase_leaf`: its memory core ticks where
# the stored phase is 0 -- every k = cfg.rnn_tick_stride steps, and on the
# first step after a done -- and holds in between. Then L = rnn_bptt_length
# counts ticks, T = L*k + n, and the loss points are the first L ticks of the
# window W = L*k (`tick_indices`): frames, actions, privileged inputs and
# returns are gathered there, the unroll runs one scan step per tick with the
# phase forced to 0 (so every unrolled step updates the core), the hidden is
# zeroed where a done falls between ticks, and each tick bootstraps at
# idx + n with the stored (unforced) hidden. The CNN sees 2L frames per
# sequence, as on the per-step path; the reward sums still use every step.
# Without a phase leaf the per-step path runs unchanged.
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
    truncation: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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

    With ``truncation`` ([B, T]) every window is instead cut at its first
    time-out, exactly as ``window_returns`` cuts the window starting at 0:
    point t equals ``window_returns`` applied to the window starting at t.

    Returns:
      (R_n, disc, boot_off, weight), each [B, length]. ``boot_off`` (int32) is
      the bootstrap offset from the loss point, ``weight`` the loss weight.
      Without ``truncation`` they are n and 1 everywhere.
    """
    idx = jnp.arange(length)[:, None] + jnp.arange(n)[None, :]  # [L, n]
    if truncation is not None:
        return _cut_windows(
            rewards[:, idx], discounts[:, idx], truncation[:, idx], gamma
        )
    d_win = discounts[:, idx]  # [B, L, n]
    r_win = rewards[:, idx]  # [B, L, n]
    alive = jnp.cumprod(d_win, axis=-1)  # m_{j+1}
    m = jnp.concatenate([jnp.ones_like(d_win[..., :1]), alive[..., :-1]], axis=-1)
    g = gamma ** jnp.arange(n, dtype=r_win.dtype)
    returns = jnp.sum(g[None, None, :] * m * r_win, axis=-1)  # [B, L]
    disc = (gamma ** (n - 1)) * alive[..., -1]  # [B, L]
    boot_off = jnp.full(returns.shape, n, jnp.int32)
    return returns, disc, boot_off, jnp.ones_like(returns)


def tick_indices(phase_w: jnp.ndarray, L: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-row positions of the first ``L`` slow-core ticks in a window.

    Args:
      phase_w: [B, W] stored PRE-step phase (any float dtype; the replay
        stores float16). Step t is a tick iff phase_w[:, t] < 0.5: the core
        updates where the phase is 0, and a done zeroes the hidden so the
        step after it restarts the clock with a tick.
      L: number of ticks to return (``cfg.rnn_bptt_length``).

    Returns:
      (idx, fill_frac). ``idx`` is int32 [B, L], increasing per row:
      ``jnp.nonzero(size=L, fill_value=W-1)`` vmapped over rows. A window of
      W = L*k steps always holds >= L ticks (exactly L without a reset; a
      reset only adds ticks), so ``fill_frac`` -- the fraction of the [B, L]
      slots padded with W-1 because a row had fewer than L ticks -- must be 0
      (logged as ``rnn/tick_fill_frac``). Nonzero means corrupted stored
      phases or a net/learner stride mismatch, and the padded slots would
      train on a non-tick step.
    """
    W = phase_w.shape[-1]
    is_tick = phase_w < 0.5
    idx = jax.vmap(lambda row: jnp.nonzero(row, size=L, fill_value=W - 1)[0])(is_tick)
    n_ticks = jnp.sum(is_tick, axis=-1)
    fill_frac = jnp.mean(jnp.maximum(L - n_ticks, 0).astype(jnp.float32)) / L
    return idx.astype(jnp.int32), fill_frac


def _take_steps(x: jnp.ndarray, sel: Any) -> jnp.ndarray:
    """``x[:, sel]`` over the time axis 1 of a [B, T, ...] array.

    ``sel`` is a static ``slice`` (shared by every row; the per-step path --
    the exact op it always used) or an int [B, L] array of per-row indices
    (the tick path) -> [B, L, ...].
    """
    if isinstance(sel, slice):
        return x[:, sel]
    return jax.vmap(lambda xb, ib: xb[ib])(x, sel)


def _done_between_ticks(discount: jnp.ndarray, idx: jnp.ndarray, W: int) -> jnp.ndarray:
    """[B, L] bool: any ``discount == 0`` in [idx_i, idx_{i+1}).

    The last interval runs to the window end W (its flag only resets the
    hidden after the final unrolled tick, which nothing consumes). A done at
    step j zeroes the hidden entering j+1, and j+1 is then the next tick, so
    the flag is exactly "the hidden entering tick i+1 is the reset zeros".
    Counted with a cumsum: C[j] = #dones in [0, j).
    """
    done = (discount == 0).astype(jnp.int32)
    csum = jnp.concatenate(
        [jnp.zeros_like(done[:, :1]), jnp.cumsum(done, axis=1)], axis=1
    )  # [B, T+1]
    nxt = jnp.concatenate([idx[:, 1:], jnp.full_like(idx[:, :1], W)], axis=1)  # [B, L]
    return (_take_steps(csum, nxt) - _take_steps(csum, idx)) > 0


def _force_phase(hidden: Any, leaf: int) -> Any:
    """``hidden`` (a tuple of leaves) with leaf ``leaf`` replaced by zeros."""
    return tuple(jnp.zeros_like(h) if i == leaf else h for i, h in enumerate(hidden))


def _unroll_policy_raw(
    policy_params: Any,
    nets: DMPONetworks,
    obs_tm: Any,
    done_tm: jnp.ndarray,
    h0: Any,
    force_phase_leaf: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, Any]:
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
      force_phase_leaf: tick path only (None = off, no new op). The index of
        a slow-core phase leaf, zeroed in the hidden entering EVERY step so
        each scan step is a core tick: the scan steps are the gathered tick
        positions, k env steps apart, whose stored phase is 0 by definition.

    Returns:
      (mu, scale, h_pre): mu/scale [L, B, A]; h_pre the tuple of [L, B, H_l]
      pre-step hiddens actually consumed at each t (h_pre[0] == h0, up to
      the forced phase), used for the staleness diagnostic and the |h|
      metric.
    """

    def _step(h, xs):
        obs_t, done_t = xs
        if force_phase_leaf is not None:
            h = _force_phase(h, force_phase_leaf)
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
    phase_leaf: int | None = None,
    batch_weights: jnp.ndarray | None = None,
):
    """Recurrent counterpart of ``_policy_loss_fn``: BPTT unroll + MPO on [B*L].

    The online unroll lives INSIDE this function so that ``value_and_grad``
    backpropagates through all L cell applications -- that is the entire
    point of L > 1. The target dist (mu/scale, precomputed by the caller from
    its own unroll and stop-gradient'd) and the q_values are gradient-free,
    exactly as in the FF path. The [B, L] loss points are folded into the
    batch axis: dists with batch shape [B*L], samples [N, B*L, A]; the
    flattening order (b*L + t) matches ``obs_flat`` and the anchor slices.

    ``batch_weights`` ([B*L], optional) weights the loss points in every MPO
    batch mean; None is the plain mean.

    Aux is ``(stats, rnn_metrics)``: the staleness diagnostic must come from
    the online unroll's recomputed pre-step hiddens, which only exist here.

    ``phase_leaf`` (tick path only; None = the per-step path, unchanged) is
    the slow-core phase leaf: forced to 0 in the unroll (see
    ``_unroll_policy_raw``) and left out of both rnn/ diagnostics -- a clock
    counter, not memory, it would dominate |h| (values up to k-1) and add a
    zero-staleness term to the leaf mean.

    Returns:
      (loss, (stats, rnn_metrics)).
    """
    length, bsz = done_tm.shape
    mu, scale, h_pre = _unroll_policy_raw(
        policy_params, nets, obs_tm, done_tm, h0, force_phase_leaf=phase_leaf
    )
    action_size = mu.shape[-1]
    # [L, B, A] -> [B, L, A] -> [B*L, A]: transpose FIRST so the flattened
    # ordering matches obs_flat / target_mu (both built from [B, L] slices).
    mu_bl = jnp.swapaxes(mu, 0, 1).reshape(bsz * length, action_size)
    scale_bl = jnp.swapaxes(scale, 0, 1).reshape(bsz * length, action_size)
    online_dist = tfd.MultivariateNormalDiag(loc=mu_bl, scale_diag=scale_bl)
    target_dist = tfd.MultivariateNormalDiag(loc=target_mu, scale_diag=target_scale)
    sampled = target_dist.sample(sample_shape=(cfg.num_samples,), seed=key)

    def _q_mean_for_n(actions_n: jnp.ndarray) -> jnp.ndarray:
        dist = jax.vmap(nets.critic.apply, in_axes=(None, 0, 0))(
            target_critic_params, obs_flat, critic_action(actions_n, cfg)
        )
        return dist.mean()

    # lax.map (sequential) over the N-sample axis, NOT vmap as in the FF
    # path: with the [B, L] loss points folded into the batch, a full vmap
    # materializes num_samples * B * L parallel vision-CNN activations
    # (409,600 images at the goal arm's B=1024, L=20) and OOMs the 5090
    # with a ~9 GiB conv buffer (measured, first smoke of
    # arm_r1_rnn_puresparse). Mapping N sequentially caps the peak at
    # B * L parallel forwards -- the FF path's own peak (N * B = 20 * 1024)
    # -- at identical FLOPs and identical values (the critic is per-sample;
    # map == vmap semantically).
    q_values = jax.lax.map(_q_mean_for_n, sampled)
    q_values = jax.lax.stop_gradient(q_values)

    loss_module = _build_loss(cfg)
    loss, stats = loss_module(
        params=dual_params,
        online_action_distribution=online_dist,
        target_action_distribution=target_dist,
        actions=sampled,
        q_values=q_values,
        batch_weights=batch_weights,
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
                cfg.kl_anchor_w + (cfg.kl_anchor_w_floor - cfg.kl_anchor_w) * progress
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
    # contributes zero by construction (h_pre[0] == h0 == stored). Memory
    # leaves only: the phase leaf (if any) is skipped.
    mem = [i for i in range(len(h_pre)) if i != phase_leaf]
    staleness = jnp.mean(
        jnp.stack(
            [
                jnp.mean(
                    jnp.sum((h_pre[i] - h_stored_tm[i]) ** 2, axis=-1)
                    / (jnp.sum(h_stored_tm[i] ** 2, axis=-1) + 1e-6)
                )
                for i in mem
            ]
        )
    )
    hidden_abs_mean = jnp.mean(jnp.stack([jnp.mean(jnp.abs(h_pre[i])) for i in mem]))
    rnn_metrics = {
        "rnn/hidden_staleness": jax.lax.stop_gradient(staleness),
        "rnn/hidden_abs_mean": jax.lax.stop_gradient(hidden_abs_mean),
    }
    return loss.squeeze(), (stats, rnn_metrics)


def _sgd_step_rnn(
    state: TrainingState,
    batch: dict[str, jnp.ndarray],
    nets: DMPONetworks,
    optimizers: tuple[optax.GradientTransformation, ...],
    cfg: DMPOConfig,
    *,
    sample_eps: jnp.ndarray | None = None,
) -> tuple[TrainingState, dict[str, jnp.ndarray]]:
    """One recurrent (short-BPTT) DMPO SGD step. See the section header above.

    Batch schema (compressed, recurrent): observation [B, T, ...] pytree,
    action [B, T, A], reward/discount [B, T], policy_hidden tuple of
    [B, T, H_l] (store_dtype, cast to f32 here), optional anchor keys
    [B, T, ...]; T == cfg.rnn_bptt_length * cfg.rnn_tick_stride + cfg.n_step.

    Two window layouts, chosen at trace time by the hidden's phase leaf
    (``nets.recurrent_meta.phase_leaf``):
      * none (per-step heads): loss points t in [0, L), bootstrap t + n --
        static slices, the original path op for op;
      * phase leaf (slow-core heads, any stride k incl. 1): loss points at
        the first L ticks ``idx`` of the window [0, L*k), bootstrap idx + n,
        per-row gathers (``tick_indices``; see the section header).
    """
    cfg = validate_alignment(cfg, recurrent=True)
    L = int(cfg.rnn_bptt_length)
    n = int(cfg.n_step)
    k = int(getattr(cfg, "rnn_tick_stride", 1))
    W = L * k
    meta = getattr(nets, "recurrent_meta", None)
    phase_leaf = getattr(meta, "phase_leaf", None)

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
    if k < 1:
        raise ValueError(f"rnn_tick_stride must be >= 1, got {k}.")
    if phase_leaf is None and k != 1:
        raise ValueError(
            f"cfg.rnn_tick_stride={k} but the recurrent head's hidden has no "
            "phase leaf (nets.recurrent_meta.phase_leaf), so the learner "
            "cannot locate the core's ticks. Strided BPTT is for slow-core "
            "heads (networks_kl_anchor_slow); per-step heads need "
            "rnn_tick_stride=1."
        )
    meta_k = getattr(meta, "tick_stride", None)
    if phase_leaf is not None and meta_k is not None and int(meta_k) != k:
        raise ValueError(
            f"the slow core ticks every tick_stride={int(meta_k)} steps but "
            f"cfg.rnn_tick_stride={k}: the sampled window would hold the "
            "wrong number of ticks. Set rnn_tick_stride to the net's "
            "tick_stride (and sequence_length = rnn_bptt_length * "
            "tick_stride + n_step)."
        )
    T = batch["reward"].shape[1]
    if k == 1 and L + n != T:
        raise ValueError(
            f"sequence_length={T} != rnn_bptt_length + n_step = {L} + {n} = "
            f"{L + n}. Every loss point t < L needs a full n-step reward "
            "window and a bootstrap state at observation[:, t + n]."
        )
    if W + n != T:
        raise ValueError(
            f"sequence_length={T} != rnn_bptt_length * rnn_tick_stride + "
            f"n_step = {L} * {k} + {n} = {W + n}. The L ticks are found in "
            f"the first L*k = {W} steps, and every tick (index <= {W - 1}) "
            "needs a full n-step reward window and a bootstrap state at "
            "observation[:, idx + n]."
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
    use_mixture = cfg.critic_target_mode == "sample_mixture"
    if use_mixture:
        k_tgt = jax.random.fold_in(k_pol, 1)
    B = batch["reward"].shape[0]

    # (0) Window layout. `sel_loss` / `sel_boot` pick the loss-point and
    # bootstrap steps of every row: static slices on the per-step path (the
    # exact ops it has always used), per-row [B, L] tick indices otherwise.
    # The ticks are read from the stored PRE-step phase (f16 holds every
    # phase 0..k-1 exactly); index arrays are data, not differentiated.
    if phase_leaf is None:
        idx = None
        sel_loss, sel_boot = slice(0, L), slice(n, n + L)
    else:
        phase_w = batch["policy_hidden"][phase_leaf][:, :W, 0]
        idx, tick_fill_frac = tick_indices(phase_w.astype(jnp.float32), L)
        sel_loss, sel_boot = idx, idx + n
    # Time-out cut (see `window_returns`): each loss point bootstraps at its
    # own offset, so the bootstrap steps become per-row [B, L] indices, and a
    # point starting at a time-out gets weight 0 in both losses.
    loss_w = None
    if cfg.bootstrap_on_truncation:
        trunc = _require_truncation(batch)
        if idx is None:
            rew_bl, disc_bl, boot_off, loss_w = _per_point_nstep_returns(
                batch["reward"], batch["discount"], cfg.discount, L, n, trunc
            )
            sel_boot = jnp.arange(L, dtype=jnp.int32)[None, :] + boot_off
        else:
            rew_w, disc_w, off_w, w_w = _per_point_nstep_returns(
                batch["reward"], batch["discount"], cfg.discount, W, n, trunc
            )
            rew_bl, disc_bl = _take_steps(rew_w, idx), _take_steps(disc_w, idx)
            loss_w = _take_steps(w_w, idx)
            sel_boot = idx + _take_steps(off_w, idx)
        loss_w = loss_w.reshape(B * L)

    # (1) Normalize only the two windows that are ever consumed: the unroll
    # window [:, :L] and the bootstrap window [:, n:n+L] (2L of T steps;
    # 40 of 120 for the goal arm). Slicing BEFORE the normalize matters for
    # the uint8-stored vision leaf: normalizing the full [B, T, ...] window
    # would dequantize ~T/(2L)x more f32 vision than is used if XLA fails to
    # fuse the downstream slices through normalize_dict_obs's per-key ops.
    # Normalization is elementwise over trailing feature dims, so
    # slice-then-normalize is exact. (For L > n the windows overlap and a few
    # steps normalize twice -- cheap elementwise work, still a net win.)
    # (On the tick path the same holds with "window" = the 2L gathered steps.)
    obs_L = _normalize_obs(
        jax.tree.map(lambda x: _take_steps(x, sel_loss), batch["observation"]),
        state.normalizer_params,
    )
    next_obs = _normalize_obs(
        jax.tree.map(lambda x: _take_steps(x, sel_boot), batch["observation"]),
        state.normalizer_params,
    )

    # (2) Stored hidden (store_dtype in replay, e.g. f16) -> f32 for compute.
    # h0 seeds the unroll; h_boot pairs with the bootstrap obs at t + n.
    # Both are DATA -- no gradient flows into the replay buffer.
    # On the tick path h0 is the stored hidden at the first tick, and h_boot
    # the stored hidden at idx + n -- its real phase, NOT forced: the target
    # policy must act as it would at step idx + n, tick or hold.
    h_all = jax.tree.map(lambda h: h.astype(jnp.float32), batch["policy_hidden"])
    if idx is None:
        h0 = jax.tree.map(lambda h: h[:, 0], h_all)
    else:
        h0 = jax.tree.map(lambda h: _take_steps(h, idx[:, :1])[:, 0], h_all)
    h_boot = jax.tree.map(lambda h: _take_steps(h, sel_boot), h_all)  # [B, L, H]
    h_stored_tm = jax.tree.map(
        lambda h: jnp.swapaxes(_take_steps(h, sel_loss), 0, 1),
        h_all,  # [L, B, H]
    )

    # Time-major unroll window. done == (discount == 0) mirrors the rollout's
    # reset trigger exactly, so the recomputed hiddens track the stored ones.
    # Tick path: the hidden entering tick i+1 is zeroed iff a done falls in
    # [idx_i, idx_{i+1}) -- held steps in between carry the state unchanged.
    obs_tm = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), obs_L)
    if idx is None:
        done_tm = jnp.swapaxes(batch["discount"][:, :L] == 0, 0, 1)  # [L, B]
    else:
        done_tm = jnp.swapaxes(_done_between_ticks(batch["discount"], idx, W), 0, 1)

    # Flattened [B*L] views for the critic and the MPO loss. Row-major
    # reshape of [B, L, ...] gives ordering b*L + t -- every flattened
    # quantity below uses the same [B, L]-first layout. obs_flat feeds
    # CRITIC applies only (online loss + the policy loss's target-critic Q),
    # so it carries the privileged input of loss point t when that is on
    # (attached before the flatten so the vector is gathered in the same
    # [B, L] layout; with the key off it is exactly obs_L).
    priv_key = _check_privileged_config(nets, cfg)
    obs_flat = jax.tree.map(
        lambda x: x.reshape((B * L, *x.shape[2:])),
        _critic_obs(obs_L, batch, priv_key, sel_loss),
    )
    act_flat = _take_steps(batch["action"], sel_loss).reshape(B * L, -1)

    # (4) Target unroll: same scan with target params, gradient-free.
    tgt_mu, tgt_scale, _ = _unroll_policy_raw(
        state.target_policy_params,
        nets,
        obs_tm,
        done_tm,
        h0,
        force_phase_leaf=phase_leaf,
    )
    action_size = tgt_mu.shape[-1]
    tgt_mu = jax.lax.stop_gradient(
        jnp.swapaxes(tgt_mu, 0, 1).reshape(B * L, action_size)
    )
    tgt_scale = jax.lax.stop_gradient(
        jnp.swapaxes(tgt_scale, 0, 1).reshape(B * L, action_size)
    )

    # (6) Per-point n-step returns, [B, L] each. Tick path: the returns of
    # every start in [0, W) -- each sums the n rewards after its start, tick
    # or not -- gathered at the ticks. (Already formed in (0) when windows
    # are cut at time-outs.)
    if not cfg.bootstrap_on_truncation:
        if idx is None:
            rew_bl, disc_bl, _, _ = _per_point_nstep_returns(
                batch["reward"], batch["discount"], cfg.discount, L, n
            )
        else:
            rew_w, disc_w, _, _ = _per_point_nstep_returns(
                batch["reward"], batch["discount"], cfg.discount, W, n
            )
            rew_bl, disc_bl = _take_steps(rew_w, idx), _take_steps(disc_w, idx)

    # (7) Critic target: bootstrap from observation[:, n:n+L] (tick path:
    # idx + n; with time-out cuts, each point's own bootstrap step) paired
    # with the STORED hidden there -- a single-step target apply (no unroll
    # needed: the stored hidden already encodes the history up to that step).
    # The target policy's action there is either N samples (the mixture
    # target) or its mean mu (== mode for a diag Gaussian, the FF path's
    # target_policy_dist_next.mode() analog). next_obs was normalized above
    # in (1).
    # Sequential lax.map over the L axis (time-major), vmap over B inside:
    # a double vmap would run B * L = 20,480 parallel policy-CNN forwards
    # at the goal arm -- 20x the FF path's bootstrap peak and part of the
    # first-smoke OOM. Mapping L caps the peak at B parallel forwards,
    # exactly the FF scale; values identical (per-sample network).
    next_obs_tm = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), next_obs)
    h_boot_tm = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), h_boot)

    if use_mixture:
        # N samples per bootstrap point from the target policy given the
        # stored hidden there; the critic rates them N-sequentially on the
        # flattened [B*L] points (the E-step's own memory layout).
        def _boot_dist_step(args):
            o_t, h_t = args
            mu_t, scale_t, _ = jax.vmap(
                lambda o, hh: nets.policy.apply(
                    state.target_policy_params, o, hh, method="raw"
                )
            )(o_t, h_t)
            return mu_t, scale_t

        boot_mu_tm, boot_scale_tm = jax.lax.map(
            _boot_dist_step, (next_obs_tm, h_boot_tm)
        )  # [L, B, A] each
        action_dim = boot_mu_tm.shape[-1]

        def _flat_bl(x_tm):
            return jnp.swapaxes(x_tm, 0, 1).reshape(B * L, action_dim)

        boot_dist = tfd.MultivariateNormalDiag(
            loc=_flat_bl(boot_mu_tm), scale_diag=_flat_bl(boot_scale_tm)
        )
        boot_crit_flat = jax.tree.map(
            lambda x: x.reshape((B * L, *x.shape[2:])),
            _critic_obs(next_obs, batch, priv_key, sel_boot),
        )
        _, boot_logits = _rate_samples(
            nets,
            state.target_critic_params,
            boot_crit_flat,
            boot_dist,
            cfg,
            k_tgt,
            sample_eps,
            use_map=True,
        )
        target_probs = jax.lax.stop_gradient(
            mixture_categorical_target(
                boot_logits, rew_bl.reshape(B * L), disc_bl.reshape(B * L), cfg
            )
        )
    else:

        def _boot_step(args):
            o_t, h_t = args
            mu_t, _, _ = jax.vmap(
                lambda o, hh: nets.policy.apply(
                    state.target_policy_params, o, hh, method="raw"
                )
            )(o_t, h_t)
            return mu_t

        next_mu_tm = jax.lax.map(_boot_step, (next_obs_tm, h_boot_tm))  # [L, B, A]
        next_action = jax.lax.stop_gradient(
            jnp.swapaxes(next_mu_tm, 0, 1)  # [B, L, A]
        )
        # The target critic rates the bootstrap state t + n, so its privileged
        # input is gathered at [:, n:n+L] / idx + n (the policy above read
        # next_obs bare).
        target_probs = jax.lax.stop_gradient(
            compute_categorical_target(
                nets,
                state.target_critic_params,
                _critic_obs(next_obs, batch, priv_key, sel_boot),
                critic_action(next_action, cfg),
                rew_bl,
                disc_bl,
                cfg,
            )
        ).reshape(B * L, cfg.num_atoms)

    # (8) Critic update on the flattened [B*L] loss points -- the same
    # `_critic_loss_fn` body as the FF path, just a bigger batch.
    crit_loss, crit_grads = jax.value_and_grad(_critic_loss_fn)(
        state.critic_params,
        nets,
        obs_flat,
        critic_action(act_flat, cfg),
        target_probs,
        loss_w,
    )
    crit_updates, new_crit_opt_state = crit_opt.update(
        crit_grads,
        state.critic_opt_state,
        state.critic_params,
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
        anchor_mu = _take_steps(anchor_mu, sel_loss).reshape(B * L, -1)
    if anchor_ls is not None:
        anchor_ls = _take_steps(anchor_ls, sel_loss).reshape(B * L, -1)

    (pol_loss, (stats, rnn_metrics)), pol_dual_grads = jax.value_and_grad(
        _policy_loss_fn_rnn,
        argnums=(0, 1),
        has_aux=True,
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
        phase_leaf=phase_leaf,
        batch_weights=loss_w,
    )
    pol_grads, dual_grads = pol_dual_grads

    # (10) Optimizer updates / warmup gate / target schedule: duplicated from
    # the FF `sgd_step` body (deliberately, to keep that body diff-free).
    pol_updates, new_pol_opt_state = pol_opt.update(
        pol_grads,
        state.policy_opt_state,
        state.policy_params,
    )
    dual_updates, new_dual_opt_state = dual_opt.update(
        dual_grads,
        state.dual_opt_state,
        state.dual_params,
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
            jax.tree.map(
                lambda new, old: jax.lax.select(policy_updates_open, new, old),
                (
                    new_pol_params,
                    new_dual_params,
                    new_pol_opt_state,
                    new_dual_opt_state,
                ),
                (
                    state.policy_params,
                    state.dual_params,
                    state.policy_opt_state,
                    state.dual_opt_state,
                ),
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
        normalizer_params=state.normalizer_params,  # unchanged in SGD
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
    if idx is not None:
        # Fraction of tick slots padded because a row had < L ticks in W.
        # Must be 0 (see tick_indices); tick path only.
        metrics["rnn/tick_fill_frac"] = tick_fill_frac
    return new_state, metrics
