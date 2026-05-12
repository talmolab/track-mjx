"""Scan-K SGD for DMPO.

Collapses ``num_updates`` host-side ``rb.sample → jit_sgd_step`` calls into a
single jitted ``lax.scan`` over K updates. The replay buffer's ``sample`` is a
pure JAX function, so it can be called inside the scan body without breaking
jit purity. This collapses K host roundtrips per rollout to one and lets XLA
fuse adjacent kernel launches across SGD updates.
"""
from __future__ import annotations
from typing import Any, Callable, Tuple
import jax
import jax.numpy as jnp
from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.learner import sgd_step


def make_scan_k_body(
    rb: Any, nets: Any, optimizers: Any, cfg: DMPOConfig, K: int
) -> Callable[[Any, Any, jax.Array], Tuple[Any, dict]]:
    """Return an UN-jitted scan-K body. Caller is responsible for jitting."""
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")

    def _body(state, rb_state, rng):
        keys = jax.random.split(rng, K)

        def body(s, key):
            sample = rb.sample(rb_state, key)
            new_s, metrics = sgd_step(s, sample.experience, nets, optimizers, cfg)
            return new_s, metrics

        new_state, metrics_seq = jax.lax.scan(body, state, keys)
        metrics = jax.tree.map(lambda x: jnp.mean(x, axis=0), metrics_seq)
        return new_state, metrics

    return _body


def make_scan_k_sgd(
    rb: Any, nets: Any, optimizers: Any, cfg: DMPOConfig, K: int
) -> Callable[[Any, Any, jax.Array], Tuple[Any, dict]]:
    """Backwards-compat wrapper: jit the scan-K body once."""
    return jax.jit(make_scan_k_body(rb, nets, optimizers, cfg, K))
