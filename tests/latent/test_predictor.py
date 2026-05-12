import jax
import jax.numpy as jnp

from track_mjx.agent.latent_ppo.networks.predictor import MotionPredictor


def test_predictor_output_shape():
    pred = MotionPredictor(layer_sizes=(64, 32), horizon=5, feat_dim=77)
    rng = jax.random.PRNGKey(0)
    z = jnp.ones((4, 16))
    params = pred.init(rng, z)
    out = pred.apply(params, z)
    assert out.shape == (4, 5, 77)
