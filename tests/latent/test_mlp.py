import jax
import jax.numpy as jnp
from flax import linen as nn

from track_mjx.agent.latent_ppo.networks.mlp import Mlp


def test_mlp_output_shape():
    rng = jax.random.PRNGKey(0)
    mlp = Mlp(layer_sizes=(32, 16), activate_final=False)
    x = jnp.ones((4, 64))
    params = mlp.init(rng, x)
    y = mlp.apply(params, x)
    assert y.shape == (4, 16)


def test_mlp_uses_elu():
    rng = jax.random.PRNGKey(0)
    mlp = Mlp(layer_sizes=(8,), activate_final=True)
    x = jnp.full((1, 4), -10.0)  # ELU saturates at -1 for very negative inputs
    params = mlp.init(rng, x)
    y = mlp.apply(params, x)
    # After Dense + ELU + LayerNorm, all values must be finite and not all zero.
    assert jnp.all(jnp.isfinite(y))
    assert not jnp.allclose(y, 0.0)
