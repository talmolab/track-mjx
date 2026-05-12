"""Verify ``make_train_chunk`` numerically matches N successive ``fused_step``
calls.

The chunk path runs ``fused_step`` inside a ``jax.lax.scan`` of length N to
reduce host-side dispatch overhead. The mean-over-N metrics returned by the
chunk must equal the mean of N metrics from N separately-called fused steps
when both paths are seeded with the same starting RNG.
"""
import jax
import jax.numpy as jnp
import numpy as np

from track_mjx.agent.dmpo.train_dmpo_step import make_fused_train_step
from track_mjx.agent.dmpo.train_dmpo_chunk import make_train_chunk
from tests.agent.dmpo.test_train_dmpo_fused import _setup


def test_chunk_matches_repeated_fused_step():
    """Path A (N successive fused calls) and Path B (one chunk of length N)
    must agree on the mean-over-N metrics for the same seed RNG.

    Production wiring (train_dmpo.py) runs warm-up via ``fused_step`` so
    ``env_state`` is a concrete pytree (not ``None``) by the time the chunk
    takes over. We mirror that contract here: do one fused warm-up step on
    each path, then start the equivalence comparison from the post-warmup
    state with a fresh RNG.
    """
    s = _setup()
    cfg = s["cfg"]
    K = 2
    N = 4

    fused = make_fused_train_step(
        s["env"], s["nets"], s["optimizers"], s["rb"], cfg, K=K,
    )

    # Warm-up step (resets env_state from None -> pytree) on the shared
    # starting state, so both paths begin from the same concrete env_state.
    warmup_rng = jax.random.PRNGKey(0)
    state0, env_state0, rb_state0, _ = fused(
        s["state"], None, s["rb_state"], warmup_rng,
    )

    # Path A: N successive fused calls from the warmed-up state. We must
    # mirror the chunk's RNG split (one ``split(rng, N)`` up front, not a
    # per-iter ``split`` that derives the next-key from the prior key) —
    # otherwise the SGD samples diverge across paths and the metrics drift
    # by ~0.1 % even for the same starting state.
    chunk_rng = jax.random.PRNGKey(123)
    keys_a = jax.random.split(chunk_rng, N)
    sa, esa, rsa = state0, env_state0, rb_state0
    metrics_a = []
    for i in range(N):
        sa, esa, rsa, m = fused(sa, esa, rsa, keys_a[i])
        metrics_a.append(m)

    # Path B: one chunk of length N from the (re-derived) warmed-up state.
    # Re-run setup + warmup to avoid donated-buffer collisions with Path A.
    s2 = _setup()
    fused2 = make_fused_train_step(
        s2["env"], s2["nets"], s2["optimizers"], s2["rb"], cfg, K=K,
    )
    state0_b, env_state0_b, rb_state0_b, _ = fused2(
        s2["state"], None, s2["rb_state"], warmup_rng,
    )
    chunk = make_train_chunk(fused2, n_iters=N)
    sb, esb, rsb, mb = chunk(
        state0_b, env_state0_b, rb_state0_b, jax.random.PRNGKey(123),
    )

    # The chunk's metrics are mean-over-N. Compare to mean of metrics_a.
    assert set(metrics_a[0].keys()) == set(mb.keys()), (
        f"metric key mismatch: {metrics_a[0].keys()} vs {mb.keys()}"
    )
    for key in metrics_a[0].keys():
        ref = float(jnp.mean(jnp.stack([m[key] for m in metrics_a])))
        got = float(mb[key])
        np.testing.assert_allclose(
            got, ref, rtol=1e-4, atol=1e-5,
            err_msg=f"chunked metric '{key}' mismatch: ref={ref} got={got}",
        )


def test_chunk_n_iters_must_be_positive():
    """``n_iters < 1`` should raise immediately (factory-time check)."""
    import pytest

    # Don't need a real fused_step to test the factory's value check.
    def _noop(state, env_state, rb_state, rng):
        return state, env_state, rb_state, {}

    with pytest.raises(ValueError):
        make_train_chunk(_noop, n_iters=0)
