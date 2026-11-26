"""Categorical (C51) support projection.

Ported from rlax 0.1.8 `rlax._src.value_learning.categorical_l2_project`
(Copyright 2019 DeepMind Technologies Limited, Apache License 2.0) so the
DMPO package does not depend on rlax/distrax/chex. Semantics unchanged; see
`LICENSE-RLAX` for the original license text.
"""

import jax
import jax.numpy as jnp


def categorical_l2_project(
    z_p: jax.Array, probs: jax.Array, z_q: jax.Array
) -> jax.Array:
    """Project distribution (z_p, probs) onto support z_q under the Cramér distance.

    Args:
        z_p: support of the source distribution, shape [kp].
        probs: source probabilities, shape [kp].
        z_q: target support, shape [kq] (kq need not equal kp).

    Returns:
        Probabilities on z_q, shape [kq].
    """
    assert z_p.ndim == 1
    assert probs.ndim == 1
    assert z_q.ndim == 1

    kp = z_p.shape[0]
    kq = z_q.shape[0]
    d_pos = jnp.roll(z_q, shift=-1)
    d_neg = jnp.roll(z_q, shift=1)
    z_p = jnp.clip(z_p, z_q[0], z_q[-1])[None, :]
    assert z_p.shape == (1, kp)
    d_pos = (d_pos - z_q)[:, None]
    d_neg = (z_q - d_neg)[:, None]
    z_q = z_q[:, None]
    assert z_q.shape == (kq, 1)
    d_neg = jnp.where(d_neg > 0, 1.0 / d_neg, jnp.zeros_like(d_neg))
    d_pos = jnp.where(d_pos > 0, 1.0 / d_pos, jnp.zeros_like(d_pos))
    delta_qp = z_p - z_q
    d_sign = (delta_qp >= 0.0).astype(probs.dtype)
    delta_hat = (d_sign * delta_qp * d_pos) - ((1.0 - d_sign) * delta_qp * d_neg)
    probs = probs[None, :]
    return jnp.sum(jnp.clip(1.0 - delta_hat, 0.0, 1.0) * probs, axis=-1)
