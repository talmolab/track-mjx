import jax
import jax.numpy as jnp

from track_mjx.agent.latent_ppo.networks.decoder import MotionDecoder


def test_decoder_output_shape():
    dec = MotionDecoder(layer_sizes=(32, 64), window_len=10, feat_dim=77)
    rng = jax.random.PRNGKey(0)
    z = jnp.ones((4, 16))
    params = dec.init(rng, z)
    out = dec.apply(params, z)
    assert out.shape == (4, 10, 77)
