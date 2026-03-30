"""Jacobian norm computation utilities.

Provides functions for computing the squared Frobenius norm of the Jacobian
of a differentiable function, useful as a regularization penalty.

Two methods are available:
- Hutchinson trace estimator (efficient, stochastic)
- Exact computation via jax.jacrev (expensive, deterministic)
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp


def hutchinson_frobenius_sq(
    fn: Callable,
    primals: Any,
    rng: jnp.ndarray,
    n_probes: int = 1,
) -> jnp.ndarray:
    """Estimate ||J||_F^2 of fn at primals using Hutchinson trace estimator.

    Uses jax.jvp with Rademacher random probes. The JVP naturally broadcasts
    over any leading batch dimensions in primals (e.g. [T, B, D]).

    ||J||_F^2 = Tr(J^T J) = E_v[||Jv||^2] where v ~ Rademacher({-1, +1}).

    Args:
        fn: Differentiable function mapping primals -> output array.
        primals: Input pytree (may have leading batch dims).
        rng: JAX random key for probe sampling.
        n_probes: Number of random probes to average over.

    Returns:
        Scalar mean estimate of ||J||_F^2 over all samples.
    """

    def single_probe(probe_key):
        tangent = jax.tree_util.tree_map(
            lambda x: 2.0 * jax.random.bernoulli(probe_key, x.shape).astype(x.dtype)
            - 1.0,
            primals,
        )
        _, Jv = jax.jvp(fn, (primals,), (tangent,))
        return jnp.mean(jnp.sum(Jv**2, axis=-1))

    keys = jax.random.split(rng, n_probes)
    return jnp.mean(jax.vmap(single_probe)(keys))


def exact_frobenius_sq(
    fn: Callable,
    primals: Any,
) -> jnp.ndarray:
    """Compute exact ||J||_F^2 of fn using jax.jacrev.

    Reshapes batched primals to [N, ...], computes per-sample Jacobian via
    reverse-mode AD, and sums the squared entries.

    Args:
        fn: Differentiable function mapping a single (unbatched) input -> output array.
        primals: Input pytree with leading batch dims [T, B, ...].

    Returns:
        Scalar mean ||J||_F^2 over all samples.
    """
    T, B = jax.tree_util.tree_leaves(primals)[0].shape[:2]
    flat_primals = jax.tree_util.tree_map(
        lambda x: x.reshape(T * B, *x.shape[2:]), primals
    )

    def single_sample_norm(single_input):
        J = jax.jacrev(fn)(single_input)
        return jax.tree_util.tree_reduce(
            lambda a, b: a + b,
            jax.tree_util.tree_map(lambda j: jnp.sum(j**2), J),
        )

    norms = jax.vmap(single_sample_norm)(flat_primals)
    return jnp.mean(norms)
