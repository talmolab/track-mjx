"""Extra parametric action distributions not in ``brax.training.distribution``.

Provides a sigmoid-bounded normal distribution and its associated bijector,
adapted from TensorFlow Probability's Sigmoid bijector for numerical stability.
"""

import jax
import jax.numpy as jnp
from brax.training import distribution

_CUTOFF_F64 = -20
_CUTOFF_F32 = -9


def _stable_sigmoid(x):
    """Numerically stable sigmoid, falling back to exp(x) for very negative inputs.

    Args:
        x: Input array.

    Returns:
        Sigmoid of x with improved numerical stability for large negative values.
    """
    x = jnp.asarray(x)
    cutoff = _CUTOFF_F64 if x.dtype == jnp.float64 else _CUTOFF_F32
    return jnp.where(x < cutoff, jnp.exp(x), jax.nn.sigmoid(x))


@jax.custom_gradient
def _stable_grad_softplus(x):
    """Numerically stable softplus with a custom gradient for large negative inputs.

    Args:
        x: Input array.

    Returns:
        Softplus of x with correct gradients for values below the stability cutoff.
    """
    x = jnp.asarray(x)
    cutoff = _CUTOFF_F64 if x.dtype == jnp.float64 else _CUTOFF_F32

    y = jnp.where(x < cutoff, jnp.log1p(jnp.exp(x)), jax.nn.softplus(x))

    def grad_fn(dy):
        return dy * jnp.where(x < cutoff, jnp.exp(x), jax.nn.sigmoid(x))

    return y, grad_fn


class SigmoidBijector:
    """Sigmoid bijector mapping reals to ``[low, high]``.

    Adapted from ``tensorflow.distributions.bijectors.Sigmoid`` with
    numerically stable forward, inverse, and log-det-jacobian computations.

    Args:
        low: Lower bound of the output range.
        high: Upper bound of the output range.
    """

    def __init__(self, low=0.0, high=1.0):
        self.low = low
        self.high = high
        self._is_standard_sigmoid = low == 0.0 and high == 1.0

    def forward(self, x):
        """Map unconstrained input to ``[low, high]``."""
        if self._is_standard_sigmoid:
            return _stable_sigmoid(x)
        lo = jnp.asarray(self.low)
        hi = jnp.asarray(self.high)
        diff = hi - lo
        left = lo + diff * _stable_sigmoid(x)
        right = hi - diff * _stable_sigmoid(-x)
        return jnp.where(x < 0, left, right)

    def inverse(self, y):
        """Map bounded value back to unconstrained space."""
        if self._is_standard_sigmoid:
            return jnp.log(y) - jnp.log1p(-y)
        return jnp.log(y - self.low) - jnp.log(self.high - y)

    def forward_log_det_jacobian(self, x):
        """Log absolute determinant of the Jacobian of ``forward``."""
        sigmoid_fldj = -_stable_grad_softplus(-x) - _stable_grad_softplus(x)
        if self._is_standard_sigmoid:
            return sigmoid_fldj
        return sigmoid_fldj + jnp.log(self.high - self.low)


class NormalSigmoidDistribution(distribution.ParametricDistribution):
    """Normal distribution followed by sigmoid."""

    def __init__(self, event_size, min_std=0.001, var_scale=1, low=0.0, high=1.0):
        """Initialize the distribution.

        Args:
          event_size: the size of events (i.e. actions).
          min_std: minimum std for the gaussian.
          var_scale: adjust the gaussian's scale parameter.
        """
        super().__init__(
            param_size=2 * event_size,
            postprocessor=SigmoidBijector(low=low, high=high),
            event_ndims=1,
            reparametrizable=True,
        )
        self._min_std = min_std
        self._var_scale = var_scale

    def create_dist(self, parameters):
        loc, scale = jnp.split(parameters, 2, axis=-1)
        scale = (jax.nn.softplus(scale) + self._min_std) * self._var_scale
        return distribution._NormalDistribution(loc=loc, scale=scale)
