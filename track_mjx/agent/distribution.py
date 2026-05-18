"""Extra parametric action distributions not implemented in brax.training.distribution"""

from brax.training import distribution
import jax
import jax.numpy as jnp


def _stable_sigmoid(x):
    """A (more) numerically stable sigmoid than `jax.nn.sigmoid`. Implemented based on `tensorflow.distributions.bijectors.Sigmoid`"""
    x = jnp.asarray(x)
    if x.dtype == jnp.float64:
        cutoff = -20
    else:
        cutoff = -9
    return jnp.where(x < cutoff, jnp.exp(x), jax.nn.sigmoid(x))


@jax.custom_gradient
def _stable_grad_softplus(x):
    """A (more) numerically stable softplus than `jax.nn.softplus`. Implemented based on `tensorflow.distributions.bijectors.Sigmoid`"""
    x = jnp.asarray(x)
    if x.dtype == jnp.float64:
        cutoff = -20
    else:
        cutoff = -9

    y = jnp.where(x < cutoff, jnp.log1p(jnp.exp(x)), jax.nn.softplus(x))

    def grad_fn(dy):
        return dy * jnp.where(x < cutoff, jnp.exp(x), jax.nn.sigmoid(x))

    return y, grad_fn


class SigmoidBijector:
    """Sigmoid Bijector. Implemented based on `tensorflow.distributions.bijectors.Sigmoid`."""

    def __init__(self, low=0.0, high=1.0):
        self.low = low
        self.high = high
        self._is_standard_sigmoid = low == 0.0 and high == 1.0

    def forward(self, x):
        if self._is_standard_sigmoid:
            return _stable_sigmoid(x)
        lo = jnp.asarray(self.low)  # Concretize only once
        hi = jnp.asarray(self.high)
        diff = hi - lo
        left = lo + diff * _stable_sigmoid(x)
        right = hi - diff * _stable_sigmoid(-x)
        return jnp.where(x < 0, left, right)

    def inverse(self, y):
        if self._is_standard_sigmoid:
            return jnp.log(y) - jnp.log1p(-y)
        return jnp.log(y - self.low) - jnp.log(self.high - y)

    def forward_log_det_jacobian(self, x):
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
