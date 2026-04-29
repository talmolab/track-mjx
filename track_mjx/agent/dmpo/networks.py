"""DMPO network heads in Flax linen.

Port of acme/jax/networks/distributional.py:
- MultivariateNormalDiagHead -> GaussianPolicyHead
- DiscreteValuedTfpHead     -> CategoricalCriticHead (Task 3)
"""
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

_MIN_SCALE = 1e-6


class GaussianPolicyHead(nn.Module):
    """Linen port of acme.jax.networks.MultivariateNormalDiagHead.

    Outputs an unbounded MultivariateNormalDiag. Action squashing (tanh) is
    NOT applied here - the MPO loss requires unbounded Gaussians for its KL
    decomposition. Use action_utils.bind / unbind at the env boundary.

    Mirrors Acme's exact scale formulation:
        scale = softplus(linear(x))
        scale *= init_scale / softplus(0.)
        scale += min_scale
    so that when the linear pre-activation is ~0 (zero-init weights and bias),
    scale_diag is approximately init_scale.
    """
    action_size: int
    init_scale: float = 0.7
    min_scale: float = _MIN_SCALE
    w_init: Callable = nn.initializers.variance_scaling(
        scale=1e-4, mode="fan_in", distribution="truncated_normal"
    )
    b_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tfd.Distribution:
        loc = nn.Dense(
            self.action_size,
            kernel_init=self.w_init,
            bias_init=self.b_init,
            name="loc",
        )(x)
        scale = nn.Dense(
            self.action_size,
            kernel_init=self.w_init,
            bias_init=self.b_init,
            name="scale",
        )(x)
        scale = jax.nn.softplus(scale)
        scale = scale * (self.init_scale / jax.nn.softplus(0.0))
        scale = scale + self.min_scale
        return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
