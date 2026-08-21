"""MJX rollout for DMPO.

Stores raw (pre-tanh) actions + raw (un-normalized) observations in the
trajectory. The MPO loss sees unbounded Gaussian samples; SGD-time
normalization uses the up-to-date running statistics. Apply tanh on the
way to the env via `bind`.

Output shape conforms to flashbax's TrajectoryBuffer.add expectation:
[add_batch_size, T, ...] per leaf.
"""
import jax
import jax.numpy as jnp

from brax.training.acme import running_statistics

from track_mjx.agent.dmpo.action_utils import bind
from track_mjx.agent.observation_utils import (
    DictRunningStatisticsState,
    flatten_obs_dict,
    normalize_dict_obs,
    update_dict_normalizer,
)


def _normalize_obs(obs, normalizer_params):
    """Dispatch normalize based on normalizer type. JIT-safe (Python type check)."""
    if isinstance(normalizer_params, DictRunningStatisticsState):
        return normalize_dict_obs(obs, normalizer_params)
    return running_statistics.normalize(obs, normalizer_params)


def _update_normalizer(normalizer_params, obs):
    """Dispatch update based on normalizer type."""
    if isinstance(normalizer_params, DictRunningStatisticsState):
        return update_dict_normalizer(normalizer_params, obs)
    return running_statistics.update(normalizer_params, obs)


def _quantize_vision(v):
    """f32 [0,1] -> uint8. The renderer unpacks 8-bit channels, so the
    round-trip error through normalize_dict_obs's /255 is <= 1/510."""
    return jnp.round(jnp.clip(v, 0.0, 1.0) * 255.0).astype(jnp.uint8)


def _maybe_store_obs(obs, normalizer_params, vision_uint8):
    """Flatten for storage; optionally quantize the vision component."""
    stored = _maybe_flatten_obs(obs, normalizer_params)
    if not vision_uint8:
        return stored
    if not isinstance(stored, dict) or "vision" not in stored:
        raise ValueError(
            "vision_uint8 storage requested but the stored observation is "
            f"{type(stored).__name__} without a 'vision' key"
        )
    return {**stored, "vision": _quantize_vision(stored["vision"])}


def _reset_hidden_where_done(hidden, done):
    """Zero the per-env hidden rows where the episode just ended.

    The auto-reset wrapper swaps in the post-reset obs on the terminal step
    itself, so the hidden consumed alongside that obs at t+1 must be the
    fresh-episode zero state -- otherwise stale memory leaks across episode
    boundaries. ``done`` ([num_envs], 0/1) broadcasts over each leaf's
    trailing feature dims. Local helper on purpose: no dependency on
    RecurrentPolicyMeta or the recurrent_ppo cell types.
    """
    def _reset(h):
        mask = done.reshape(done.shape + (1,) * (h.ndim - done.ndim)) > 0
        return jnp.where(mask, jnp.zeros_like(h), h)

    return jax.tree.map(_reset, hidden)


def _maybe_flatten_obs(obs, normalizer_params):
    """Flatten nested dict obs to {'imitation_target': ..., 'proprioception': ...}.

    The vnl-playground rodent imitation env returns OrderedDict-nested obs
    (`state.task_obs.{root,quat,joint,body}`, `state.proprioception.{joint_angles,
    joint_ang_vels,...}`). The replay buffer's tree structure check requires the
    transition's obs to match the template, which is the flat-keyed shape
    (matching what the network expects). We pre-flatten here so the trajectory
    stored in replay matches the template.

    For flat-array observations (no DictRunningStatisticsState) this is a no-op
    pass-through. JIT-safe (Python isinstance check resolved at trace time).
    """
    if isinstance(normalizer_params, DictRunningStatisticsState):
        return flatten_obs_dict(obs)
    return obs


def collect_rollout(
    env,
    policy_apply,
    policy_params,
    normalizer_params,
    rng,
    num_envs: int,
    num_steps: int,
    init_state=None,
    extra_state_extras=(),
    frozen_policy_params=None,
    behavior_mix_frac=None,
    reward_remix_key=None,
    reward_remix_lambda=None,
    store_next_observation: bool = True,
    vision_uint8: bool = False,
    recurrent_meta=None,
    policy_hidden=None,
):
    """Roll out num_envs parallel envs for num_steps timesteps.

    Args:
      env: object with .reset(rng) and .step(state, action). Both must be
        jittable / vmappable. Pre-batched envs (e.g., wrap_for_brax_training
        + BinocularVisionRenderWrapper) advertise via env.pre_batched=True so
        we skip the outer vmap.
      policy_apply: stateless callable (params, obs) -> distribution.
      policy_params: pytree of policy parameters; threaded through the scan
        as data so the inner trace is reused across iterations.
      normalizer_params: running-statistics state used to normalize obs
        before policy.apply. Read-only inside the scan; the post-scan
        update returns a fresh state reflecting all observed obs.
      rng: PRNGKey for action sampling and (when init_state is None) env reset.
      num_envs: number of parallel envs.
      num_steps: trajectory length.
      init_state: optional starting env state. When provided, env.reset is
        NOT called -- the rollout resumes from the supplied state.
      extra_state_extras: Optional sequence of keys to extract from
        ``state.info`` at each step. Each key becomes a transition-dict
        entry (shape ``[num_envs, num_steps, ...]`` matching the per-env
        info value). Used by the kl-anchor pipeline to thread
        ``anchor_mu_imit`` / ``anchor_log_std_imit`` through to the
        replay buffer for the loss-side KL term. Default empty tuple
        preserves the original schema. A missing key raises ``KeyError``
        at trace time -- this is intentional, a silent default would mask
        wrapper-wiring bugs. The tuple is read at trace time; pass via
        closure or ``static_argnames`` if calling from a jitted wrapper.
      frozen_policy_params: optional second params pytree (same network) for
        BEHAVIOR MIXING: the first ``ceil(behavior_mix_frac * num_envs)``
        envs act with these params instead of ``policy_params``. Both
        policies share the per-env sample key (common random numbers). The
        stored transition action is the EXECUTED one, as MPO expects.
        ``None`` (default) keeps the single-policy path bit-identical.
      behavior_mix_frac: traced f32 scalar in [0, 1]; required when
        ``frozen_policy_params`` is given. May change between calls without
        retracing (it is data, not a static).
      reward_remix_key: optional ``new_state.metrics`` key naming the SPARSE
        reward component (e.g. ``"rewards/gap_crossing_bonus"``). When set,
        the STORED reward becomes ``sparse + lambda * (reward - sparse)``,
        i.e. the dense remainder is scaled by ``reward_remix_lambda``. The
        env's own reward/metrics are not modified. ``None`` (default) stores
        the env reward unchanged.
      reward_remix_lambda: traced f32 scalar; required with
        ``reward_remix_key``.
      store_next_observation: when False, the transition omits the
        ``next_observation`` field (it is bit-identical to the NEXT
        timestep's ``observation`` in flashbax trajectory storage -- the
        auto-reset wrapper swaps obs on the terminal step itself, and the
        time axis is continuous across rollout adds). Halves the stored
        observation memory; the learner must bootstrap from
        ``observation[:, n]`` (it detects the missing key).
      vision_uint8: when True, store ``observation["vision"]`` quantized to
        uint8 (``round(v*255)``). ``normalize_dict_obs`` dequantizes at
        sample time; the live policy in this rollout keeps consuming the
        f32 render. Requires dict observations with a "vision" key.
      recurrent_meta: optional ``RecurrentPolicyMeta``-like object exposing
        ``init_hidden(batch_size)`` and ``store_dtype``. Trace-time flag:
        ``None`` (default) keeps the feed-forward path byte-identical. When
        set, ``policy_apply`` must have the recurrent signature
        ``(params, obs, hidden) -> (distribution, new_hidden)`` with hidden
        a tuple of per-layer arrays; the transition gains a
        ``"policy_hidden"`` entry (the PRE-step hidden the policy consumed
        at this obs, cast to ``store_dtype`` -- the learner unrolls BPTT
        from it), and the return grows a 4th element ``final_hidden``.
      policy_hidden: optional starting hidden (tuple of [num_envs, H_l]
        arrays) for chaining rollouts. ``None`` zero-inits via
        ``recurrent_meta.init_hidden(num_envs)`` -- both on the reset path
        AND on resume, because hidden is transient (never checkpointed).
        Only valid together with ``recurrent_meta``.

    Returns:
      trajectory: dict with keys observation/action/reward/discount/next_observation,
        each shaped [num_envs, num_steps, ...]. Action is the raw (pre-tanh)
        Gaussian sample. Observations are RAW (not normalized).
      final_state: env state after num_steps.
      new_normalizer_params: updated running-statistics state (count+=N*T).
      final_hidden: ONLY when ``recurrent_meta`` is given (4-tuple return):
        the post-step, post-done-reset hidden after num_steps -- feed it
        back as ``policy_hidden`` together with ``init_state=final_state``
        to continue seamlessly.
    """
    pre_batched = bool(getattr(env, "pre_batched", False))
    extra_state_extras = tuple(extra_state_extras)
    recurrent = recurrent_meta is not None

    if recurrent and frozen_policy_params is not None:
        raise NotImplementedError(
            "behavior mixing with a recurrent policy is not supported (v1)"
        )
    if policy_hidden is not None and not recurrent:
        # Fail loud: silently dropping a supplied hidden would mask a
        # wiring bug in the caller (hidden threaded but meta not).
        raise ValueError(
            "policy_hidden was provided without recurrent_meta"
        )

    if init_state is None:
        rng, k_reset = jax.random.split(rng)
        reset_keys = jax.random.split(k_reset, num_envs)
        if pre_batched:
            state = env.reset(reset_keys)
        else:
            state = jax.vmap(env.reset)(reset_keys)
    else:
        state = init_state

    if recurrent and policy_hidden is None:
        # Hidden is transient (never checkpointed): zero-init on the reset
        # path AND as the fallback on resume.
        policy_hidden = recurrent_meta.init_hidden(num_envs)

    def step_fn(carry, _):
        if recurrent:
            state, rng, hidden = carry
        else:
            state, rng = carry
            hidden = None
        rng, k_act = jax.random.split(rng)
        keys = jax.random.split(k_act, num_envs)
        # Normalize the obs the policy sees; do NOT mutate the obs stored
        # in the trajectory (it stays raw so SGD can re-normalize with
        # up-to-date stats).
        norm_obs = _normalize_obs(state.obs, normalizer_params)
        if recurrent:
            # Per-env vmap over unbatched obs/hidden slices -- the module is
            # shape-agnostic, matching the FF convention below.
            def _act(o, h, k):
                dist, nh = policy_apply(policy_params, o, h)
                return dist.sample(seed=k), nh

            raw_action, new_hidden = jax.vmap(_act)(norm_obs, hidden, keys)
        else:
            new_hidden = None
            raw_action = jax.vmap(
                lambda o, k: policy_apply(policy_params, o).sample(seed=k)
            )(norm_obs, keys)
        if frozen_policy_params is not None:
            # Behavior mixing: envs [0, ceil(frac*N)) act with the frozen
            # policy. Same per-env keys as the learner branch -> common
            # random numbers; the executed action is what gets stored.
            frozen_action = jax.vmap(
                lambda o, k: policy_apply(frozen_policy_params, o).sample(seed=k)
            )(norm_obs, keys)
            n_frozen = jnp.ceil(behavior_mix_frac * num_envs)
            frozen_mask = jnp.arange(num_envs) < n_frozen
            raw_action = jnp.where(
                frozen_mask.reshape((num_envs,) + (1,) * (raw_action.ndim - 1)),
                frozen_action,
                raw_action,
            )
        bound_action = bind(raw_action)
        if pre_batched:
            new_state, reward = env.step(state, bound_action)
        else:
            new_state, reward = jax.vmap(env.step)(state, bound_action)
        if reward_remix_key is not None:
            # Store sparse + lambda * dense-remainder. The sparse component
            # is read from the env's per-term metrics on the POST-step state
            # (recomputed every step; unaffected by full_reset=False
            # auto-resets, unlike info). Per-term metrics are not
            # nan_to_num'd by the env (the summed reward is), so sanitize.
            sparse = jnp.nan_to_num(new_state.metrics[reward_remix_key])
            reward = sparse + reward_remix_lambda * (reward - sparse)
        transition = {
            # Pre-flatten dict obs to the canonical 2-key shape so the replay
            # buffer's structural check matches the network/normalizer template.
            "observation": _maybe_store_obs(
                state.obs, normalizer_params, vision_uint8
            ),
            "action": raw_action,                # RAW (pre-tanh)
            "reward": reward,
            "discount": (1.0 - new_state.done).astype(jnp.float32),
        }
        if store_next_observation:
            transition["next_observation"] = _maybe_store_obs(
                new_state.obs, normalizer_params, vision_uint8
            )
        if recurrent:
            # PRE-step hidden: the state the policy consumed at this obs,
            # which is what the learner's BPTT unroll must start from.
            # Cast to the replay storage dtype (learner upcasts on read).
            transition["policy_hidden"] = jax.tree.map(
                lambda h: h.astype(recurrent_meta.store_dtype), hidden
            )
            new_hidden = _reset_hidden_where_done(new_hidden, new_state.done)
        # Extra info keys (e.g. anchor_mu_imit) -- extracted from CURRENT
        # state.info because they describe the obs the policy conditioned on.
        for key in extra_state_extras:
            transition[key] = state.info[key]
        if recurrent:
            return (new_state, rng, new_hidden), transition
        return (new_state, rng), transition

    carry0 = (state, rng, policy_hidden) if recurrent else (state, rng)
    final_carry, traj = jax.lax.scan(
        step_fn, carry0, None, length=num_steps,
    )
    final_state = final_carry[0]
    # scan stacks along axis 0 (time); flashbax wants [B, T, ...].
    traj = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), traj)

    # Update normalizer from all observed obs (shape [N, T, ...]).
    new_normalizer_params = _update_normalizer(normalizer_params, traj["observation"])
    if recurrent:
        return traj, final_state, new_normalizer_params, final_carry[2]
    return traj, final_state, new_normalizer_params
