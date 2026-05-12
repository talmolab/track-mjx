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
from track_mjx.agent.dmpo.train_dmpo_sgd import make_scan_k_body


def make_fused_train_step(
    env: Any,
    nets: Any,
    optimizers: Any,
    rb: Any,
    cfg: DMPOConfig,
    K: int,
    extra_state_extras: tuple = (),
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
    """
    policy_apply = nets.policy.apply
    num_envs = int(cfg.num_envs)
    unroll = int(cfg.unroll_length)
    scan_k = make_scan_k_body(rb, nets, optimizers, cfg, K)

    @functools.partial(jax.jit, donate_argnums=(1, 2))
    def _step(state, env_state, rb_state, rng):
        # RNG layout (part of the step's contract): split into 3 keys —
        # (next_rng, k_roll, k_sgd). The numerical-equivalence test in
        # ``tests/agent/dmpo/test_train_dmpo_fused.py`` mirrors this split
        # to derive identical k_roll / k_sgd in the unfused reference path.
        # If you reorder or grow this split, update the test in lockstep.
        rng, k_roll, k_sgd = jax.random.split(rng, 3)
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
        return new_state, env_state, rb_state, metrics

    return _step
