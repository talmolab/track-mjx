"""N-iteration scan over the fused DMPO training step.

Wraps ``fused_step`` in a ``jax.lax.scan`` of length ``n_iters`` so a single
host-side dispatch covers N training iterations. This matches FF PPO's
topology — one Python step per chunk of work — and removes per-iteration
dispatch overhead from the steady-state loop.
"""
from __future__ import annotations

import functools
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp


def make_train_chunk(
    fused_step: Callable, n_iters: int, recurrent: bool = False,
) -> Callable[[Any, Any, Any, jax.Array], Tuple[Any, Any, Any, dict]]:
    """Run ``n_iters`` of ``fused_step`` inside one jit dispatch.

    ``fused_step`` must have signature ``(state, env_state, rb_state, rng) ->
    (state, env_state, rb_state, metrics)`` and be jit-pure.
    Returned chunk donates ``env_state`` and ``rb_state`` for in-place reuse.
    Metrics are mean-aggregated over the N inner iters.

    ``recurrent=True`` (default off — the FF chunk below is untouched) pairs
    with the recurrent build of ``make_fused_train_step``: ``fused_step`` is
    then the 5-arg ``(state, env_state, policy_hidden, rb_state, rng)`` step
    and ``policy_hidden`` rides the scan carry so the GRU state is continuous
    across the N inner rollouts (a per-iter zero reset would chop memory at
    every chunk boundary). ``policy_hidden`` must be a concrete pytree here —
    the training loop zero-inits it before the first chunk (scan carries
    cannot change structure None -> tuple mid-scan).
    """
    if n_iters < 1:
        raise ValueError(f"n_iters must be >= 1, got {n_iters}")

    if recurrent:
        @functools.partial(jax.jit, donate_argnums=(1, 2, 3))
        def _chunk_rnn(state, env_state, policy_hidden, rb_state, rng):
            keys = jax.random.split(rng, n_iters)

            def body(carry, key):
                s, es, ph, rs = carry
                new_s, new_es, new_ph, new_rs, m = fused_step(s, es, ph, rs, key)
                return (new_s, new_es, new_ph, new_rs), m

            (state, env_state, policy_hidden, rb_state), metrics_seq = jax.lax.scan(
                body, (state, env_state, policy_hidden, rb_state), keys
            )
            metrics = jax.tree.map(lambda x: jnp.mean(x, axis=0), metrics_seq)
            return state, env_state, policy_hidden, rb_state, metrics

        return _chunk_rnn

    # donate_argnums=(1, 2) for the same reason as fused_step — see Task 3:
    # state (arg 0) cannot be donated on the first call after init because of
    # the target/online param aliasing. The alias is broken after one SGD step,
    # but Task 5's wiring (Step 5(c) below) keeps warm-up on fused_step and
    # only enters the chunk path after warm-up completes, so by the time the
    # chunk runs the alias is already broken. Even so, donating only env_state
    # and rb_state is the safer choice: the large buffers (vision frames,
    # replay storage) are donated, the small params are copied.
    @functools.partial(jax.jit, donate_argnums=(1, 2))
    def _chunk(state, env_state, rb_state, rng):
        keys = jax.random.split(rng, n_iters)

        def body(carry, key):
            s, es, rs = carry
            new_s, new_es, new_rs, m = fused_step(s, es, rs, key)
            return (new_s, new_es, new_rs), m

        (state, env_state, rb_state), metrics_seq = jax.lax.scan(
            body, (state, env_state, rb_state), keys
        )
        metrics = jax.tree.map(lambda x: jnp.mean(x, axis=0), metrics_seq)
        return state, env_state, rb_state, metrics

    return _chunk
