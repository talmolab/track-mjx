"""Discriminator networks for AMP training."""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from brax.training import networks
from flax import linen as nn


class AMPDiscriminator(nn.Module):
    """MLP discriminator that maps AMP observations to scalar logits."""

    hidden_layer_sizes: Sequence[int] = (1024, 1024)
    activation: networks.ActivationFn = nn.silu
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = obs
        for i, hidden_size in enumerate(self.hidden_layer_sizes):
            x = nn.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
            )(x)
            x = self.activation(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)

        logits = nn.Dense(
            1,
            name="logits",
            kernel_init=jax.nn.initializers.uniform(1.0),
            bias_init=jax.nn.initializers.zeros,
        )(x)
        return jnp.squeeze(logits, axis=-1)


def make_amp_discriminator_network(
    obs_size: int,
    hidden_layer_sizes: Sequence[int] = (1024, 1024),
    use_layer_norm: bool = False,
) -> networks.FeedForwardNetwork:
    """Create a feed-forward AMP discriminator network."""

    discriminator = AMPDiscriminator(
        hidden_layer_sizes=tuple(hidden_layer_sizes),
        use_layer_norm=use_layer_norm,
    )
    dummy_obs = jnp.zeros((1, obs_size), dtype=jnp.float32)

    return networks.FeedForwardNetwork(
        init=lambda key: discriminator.init(key, dummy_obs),
        apply=lambda params, obs: discriminator.apply(params, obs),
    )
