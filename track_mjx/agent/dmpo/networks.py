"""DMPO network heads in Flax linen.

Port of acme/jax/networks/distributional.py:
- MultivariateNormalDiagHead -> GaussianPolicyHead
- DiscreteValuedTfpHead     -> CategoricalCriticHead (Task 3)
"""
from typing import Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
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


class DiscreteValuedTfpDistribution(tfd.Categorical):
    """Generalization of tfd.Categorical that knows its real-valued support.

    Port of acme.jax.networks.DiscreteValuedTfpDistribution. The support
    `values` can be any real-valued range (vs. [0, n-1] for plain Categorical),
    which lets us take a meaningful mean/variance over it. Used as the C51
    critic distribution in DMPO.
    """

    def __init__(
        self,
        values: jnp.ndarray,
        logits: Optional[jnp.ndarray] = None,
        probs: Optional[jnp.ndarray] = None,
        name: str = "DiscreteValuedDistribution",
    ):
        parameters = dict(locals())
        self._values = np.asarray(values)

        if logits is not None:
            logits = jnp.asarray(logits)
        if probs is not None:
            probs = jnp.asarray(probs)

        super().__init__(logits=logits, probs=probs, name=name)
        self._parameters = parameters

    @property
    def values(self) -> jnp.ndarray:
        return self._values

    def _sample_n(self, key, n):
        indices = super()._sample_n(key=key, n=n)
        return jnp.take_along_axis(self._values, indices, axis=-1)

    def mean(self) -> jnp.ndarray:
        """Mean using the real-valued support, not the integer indices."""
        return jnp.sum(self.probs_parameter() * self._values, axis=-1)

    def variance(self) -> jnp.ndarray:
        dist_squared = jnp.square(jnp.expand_dims(self.mean(), -1) - self._values)
        return jnp.sum(self.probs_parameter() * dist_squared, axis=-1)

    def _event_shape(self):
        return jnp.zeros((), dtype=jnp.int32)

    def _event_shape_tensor(self):
        return []


class CategoricalCriticHead(nn.Module):
    """Linen port of acme.jax.networks.DiscreteValuedTfpHead.

    Categorical critic over `num_atoms` fixed atoms uniformly spaced in
    [vmin, vmax]. This is the C51-style distributional critic head used by
    DMPO. The returned distribution exposes its support via `dist.values` and
    overrides `mean()` / `variance()` to incorporate it.

    Acme's haiku version stores the atoms as a private numpy attribute and
    exposes them via the returned distribution. The linen port additionally
    exposes them as a `@property` on the module itself, since the support is a
    deterministic function of the dataclass fields and is needed by the
    Bellman projection (Task 10) before `apply` is called.
    """

    num_atoms: int
    vmin: float
    vmax: float
    w_init: Optional[Callable] = None
    b_init: Optional[Callable] = None

    @property
    def values(self) -> jnp.ndarray:
        """Atom support, length `num_atoms`, spanning [vmin, vmax]."""
        return jnp.linspace(self.vmin, self.vmax, num=self.num_atoms)

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> tfd.Distribution:
        # Match Acme: pass init kwargs only if provided, so nn.Dense defaults
        # apply otherwise (matching Haiku Linear's default behavior with
        # w_init=None / b_init=None).
        dense_kwargs = {}
        if self.w_init is not None:
            dense_kwargs["kernel_init"] = self.w_init
        if self.b_init is not None:
            dense_kwargs["bias_init"] = self.b_init
        logits = nn.Dense(self.num_atoms, name="logits", **dense_kwargs)(inputs)
        return DiscreteValuedTfpDistribution(values=self.values, logits=logits)
