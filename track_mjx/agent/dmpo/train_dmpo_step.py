"""Fused DMPO training step: rollout + replay.add + scan-K SGD in one jit.

Collapses three host-side dispatches per training iteration
(``jit_collect_rollout`` -> ``rb.add`` -> ``scan_k_sgd``) into a single
jitted call. Buffers are donated for in-place reuse so XLA can fuse adjacent
kernels and reuse memory across rollout and SGD.

The step takes ``env_state`` as an explicit argument; passing ``None`` on
the first call triggers the env-reset branch in ``collect_rollout``. JAX
retraces once for that signature (None -> pytree input) and then reuses the
cached resume-path trace for every iteration thereafter.
"""
from __future__ import annotations

import functools
from typing import Any, Callable, Tuple

import jax

from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.rollout import collect_rollout
from track_mjx.agent.dmpo.schedules import (
    behavior_mix_frac,
    env_steps_estimate,
    reward_anneal_lambda,
)
from track_mjx.agent.dmpo.train_dmpo_sgd import make_scan_k_body


def make_fused_train_step(
    env: Any,
    nets: Any,
    optimizers: Any,
    rb: Any,
    cfg: DMPOConfig,
    K: int,
    extra_state_extras: tuple = (),
    frozen_behavior_params: Any = None,
) -> Callable[[Any, Any, Any, jax.Array], Tuple[Any, Any, Any, dict]]:
    """Build a single jitted ``(state, env_state, rb_state, rng) -> ...`` step.

    Args:
      env: env adapter with ``.reset(rng)`` / ``.step(state, action)``.
      nets: ``DMPONetworks`` (provides ``policy.apply`` and the SGD body).
      optimizers: ``(policy_opt, critic_opt, dual_opt)`` from
        ``learner.make_optimizers``.
      rb: flashbax trajectory buffer (uses ``add`` and ``sample``).
      cfg: ``DMPOConfig``; ``num_envs`` and ``unroll_length`` are baked in
        as Python ints.
      K: number of inner SGD updates per fused step.
      frozen_behavior_params: optional frozen policy params for behavior
        mixing (see ``DMPOConfig.behavior_mix_init``). Closed over — the
        pytree never changes, so it is safe as a capture, and this keeps the
        ``(state, env_state, rb_state, rng)`` contract intact. The mixing
        fraction and the reward-anneal lambda are DERIVED inside the step
        from ``state.steps`` (see schedules.py), so they advance per rollout
        without host threading.

    Returns:
      Jitted callable. Inputs ``env_state`` and ``rb_state`` are donated
      for in-place reuse of the large rollout / replay buffers. ``state``
      is *not* donated because the freshly initialized ``TrainingState``
      has internal buffer aliasing (``target_policy_params is
      policy_params`` until the first SGD step), which JAX's runtime
      rejects as a double-donation. Skipping state donation costs only
      a few MB of params/opt-state copy per step — the rb_state replay
      tensor is the lever that matters. ``env_state`` may be ``None`` on
      the first call (reset path); the second call retraces once for the
      resume signature.

      The signature branches at BUILD time on ``nets.recurrent_meta``:
        * ``None`` (every FF arm): today's 4-arg step, body byte-identical.
        * Set (recurrent policy head): a 5-arg step
          ``(state, env_state, policy_hidden, rb_state, rng) ->
          (state', env_state', policy_hidden', rb_state', metrics)`` with
          ``policy_hidden`` donated alongside the buffers. ``policy_hidden``
          may be ``None`` on the first call (the rollout zero-inits; hidden
          is transient and never checkpointed) — that call shares the
          env-reset retrace, so resume still costs only one extra trace.
    """
    policy_apply = nets.policy.apply
    num_envs = int(cfg.num_envs)
    unroll = int(cfg.unroll_length)
    scan_k = make_scan_k_body(rb, nets, optimizers, cfg, K)

    # Warm-start transition features (all off by default; trace-time flags).
    use_behavior_mix = (
        frozen_behavior_params is not None and float(cfg.behavior_mix_init) > 0.0
    )
    use_reward_remix = getattr(cfg, "reward_anneal_sparse_key", None) is not None
    if float(getattr(cfg, "behavior_mix_init", 0.0)) > 0.0 and frozen_behavior_params is None:
        raise ValueError(
            "cfg.behavior_mix_init > 0 but no frozen_behavior_params supplied "
            "to make_fused_train_step -- the mix would silently not happen. "
            "Pass the warm-start policy params through the training loop."
        )

    recurrent_meta = getattr(nets, "recurrent_meta", None)
    if recurrent_meta is not None:
        # Fail-loud cross-check (PLAN section 2): recurrent nets dispatch the
        # rollout here but the LEARNER dispatches on cfg.rnn_bptt_length, so
        # the default 0 would send stored-hidden sequences into the FF SGD
        # body, whose 2-arg policy.apply then dies with an opaque flax
        # missing-argument TypeError at first trace. Catch it here instead.
        if int(getattr(cfg, "rnn_bptt_length", 0)) <= 0:
            raise ValueError(
                "nets.recurrent_meta is set (recurrent policy head) but "
                f"cfg.rnn_bptt_length={getattr(cfg, 'rnn_bptt_length', 0)}. "
                "The recurrent learner needs rnn_bptt_length > 0 (and "
                "sequence_length == rnn_bptt_length + n_step); set "
                "train_config.rnn_bptt_length or build FF networks."
            )
        # Recurrent build. A separate function (rather than trace-time
        # branches inside the FF body) so the FF step below stays
        # byte-identical — the bit-identity guard for every completed arm.
        # Behavior mixing is not supported v1: collect_rollout raises
        # NotImplementedError at first trace if frozen params reach it.
        @functools.partial(jax.jit, donate_argnums=(1, 2, 3))
        def _step_rnn(state, env_state, policy_hidden, rb_state, rng):
            # Same RNG layout as the FF step: (next_rng, k_roll, k_sgd).
            # tests/agent/dmpo/test_train_dmpo_fused.py mirrors this split;
            # keep the two builds in lockstep so the FF-equivalence test
            # also pins the recurrent build's rollout/SGD key derivation.
            rng, k_roll, k_sgd = jax.random.split(rng, 3)
            t_env = env_steps_estimate(state.steps, cfg, K)
            remix_lambda = (
                reward_anneal_lambda(t_env, cfg) if use_reward_remix else None
            )
            traj, env_state, new_normalizer_params, policy_hidden = collect_rollout(
                env,
                policy_apply,
                state.policy_params,
                state.normalizer_params,
                k_roll,
                num_envs=num_envs,
                num_steps=unroll,
                init_state=env_state,
                extra_state_extras=extra_state_extras,
                frozen_policy_params=(
                    frozen_behavior_params if use_behavior_mix else None
                ),
                behavior_mix_frac=None,
                reward_remix_key=(
                    cfg.reward_anneal_sparse_key if use_reward_remix else None
                ),
                reward_remix_lambda=remix_lambda,
                store_next_observation=bool(
                    getattr(cfg, "store_next_observation", True)
                ),
                vision_uint8=bool(getattr(cfg, "vision_uint8_storage", False)),
                recurrent_meta=recurrent_meta,
                policy_hidden=policy_hidden,
            )
            state = state._replace(normalizer_params=new_normalizer_params)
            rb_state = rb.add(rb_state, traj)
            # Same warm-up gate as the FF step (see the comment there).
            sgd_state, sgd_metrics = scan_k(state, rb_state, k_sgd)
            can_sample = rb.can_sample(rb_state)
            new_state = jax.tree_util.tree_map(
                lambda new, old: jax.lax.select(can_sample, new, old),
                sgd_state, state,
            )
            metrics = sgd_metrics
            if use_reward_remix:
                metrics = dict(metrics)
                metrics["schedule/reward_anneal_lambda"] = remix_lambda
            return new_state, env_state, policy_hidden, rb_state, metrics

        return _step_rnn

    @functools.partial(jax.jit, donate_argnums=(1, 2))
    def _step(state, env_state, rb_state, rng):
        # RNG layout (part of the step's contract): split into 3 keys —
        # (next_rng, k_roll, k_sgd). The numerical-equivalence test in
        # ``tests/agent/dmpo/test_train_dmpo_fused.py`` mirrors this split
        # to derive identical k_roll / k_sgd in the unfused reference path.
        # If you reorder or grow this split, update the test in lockstep.
        rng, k_roll, k_sgd = jax.random.split(rng, 3)
        # Schedule scalars from the SGD counter (traced; advance per rollout).
        t_env = env_steps_estimate(state.steps, cfg, K)
        mix_frac = behavior_mix_frac(t_env, cfg) if use_behavior_mix else None
        remix_lambda = reward_anneal_lambda(t_env, cfg) if use_reward_remix else None
        traj, env_state, new_normalizer_params = collect_rollout(
            env,
            policy_apply,
            state.policy_params,
            state.normalizer_params,
            k_roll,
            num_envs=num_envs,
            num_steps=unroll,
            init_state=env_state,
            extra_state_extras=extra_state_extras,
            frozen_policy_params=(
                frozen_behavior_params if use_behavior_mix else None
            ),
            behavior_mix_frac=mix_frac,
            reward_remix_key=(
                cfg.reward_anneal_sparse_key if use_reward_remix else None
            ),
            reward_remix_lambda=remix_lambda,
            store_next_observation=bool(
                getattr(cfg, "store_next_observation", True)
            ),
            vision_uint8=bool(getattr(cfg, "vision_uint8_storage", False)),
        )
        state = state._replace(normalizer_params=new_normalizer_params)
        rb_state = rb.add(rb_state, traj)
        # Gate SGD on rb.can_sample. v4's loop did
        # ``if not bool(rb.can_sample(rb_state)): continue`` at the host —
        # we mirror that semantic in jit via lax.cond so warm-up is
        # data-collection-only (no policy/critic drift on partially-filled
        # replay). At min_replay_size < num_envs*unroll this is a no-op
        # because can_sample flips True at the end of the first fused
        # step, but the gate is correct for any min_replay setting.
        sgd_state, sgd_metrics = scan_k(state, rb_state, k_sgd)
        can_sample = rb.can_sample(rb_state)
        new_state = jax.tree_util.tree_map(
            lambda new, old: jax.lax.select(can_sample, new, old),
            sgd_state, state,
        )
        # Metrics carry the would-be values regardless; eval-time logging
        # uses the most recent chunk's mean which is post-warmup anyway.
        metrics = sgd_metrics
        if use_behavior_mix or use_reward_remix:
            # Surface the live schedule values. Metrics leaving the fused step
            # are SCALARS (scan_k already mean-reduces over K; the chunk then
            # means over iters), so these must be scalars too -- a [K]-shaped
            # entry survives both reductions as [K] and crashes the wandb
            # float() conversion at the first log point.
            metrics = dict(metrics)
            if use_behavior_mix:
                metrics["schedule/behavior_mix_frac"] = mix_frac
            if use_reward_remix:
                metrics["schedule/reward_anneal_lambda"] = remix_lambda
        return new_state, env_state, rb_state, metrics

    return _step
