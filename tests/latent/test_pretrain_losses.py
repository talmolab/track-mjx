import jax
import jax.numpy as jnp

from track_mjx.agent.latent_ppo.losses.pretrain_losses import (
    pretrain_loss,
    reparameterize,
)
from track_mjx.agent.latent_ppo.networks.decoder import MotionDecoder
from track_mjx.agent.latent_ppo.networks.encoder import MotionEncoder
from track_mjx.agent.latent_ppo.networks.predictor import MotionPredictor


def test_loss_is_finite_scalar():
    enc = MotionEncoder(layer_sizes=(16,), latent_dim=4)
    dec = MotionDecoder(layer_sizes=(16,), window_len=3, feat_dim=5)
    pred = MotionPredictor(layer_sizes=(16,), horizon=2, feat_dim=5)
    rng = jax.random.PRNGKey(0)
    inputs = jnp.ones((2, 3, 5))
    targets = jnp.zeros((2, 2, 5))
    enc_params = enc.init(rng, inputs)
    dec_params = dec.init(rng, jnp.ones((2, 4)))
    pred_params = pred.init(rng, jnp.ones((2, 4)))
    loss, aux = pretrain_loss(
        enc, dec, pred, enc_params, dec_params, pred_params,
        inputs, targets, rng=rng, beta_kl=0.1, w_pred=1.0,
    )
    assert loss.shape == ()
    assert jnp.isfinite(loss)
    for k in ("recon", "kl", "pred"):
        assert k in aux
        assert aux[k].shape == ()


def test_kl_zero_when_mean_zero_var_one():
    """KL(N(0,I) || N(0,I)) == 0."""
    from track_mjx.agent.latent_ppo.losses.pretrain_losses import kl_to_unit_gaussian
    mean = jnp.zeros((4, 8))
    logvar = jnp.zeros((4, 8))
    kl = kl_to_unit_gaussian(mean, logvar)
    assert jnp.allclose(kl, 0.0, atol=1e-6)


def test_reparameterize_zero_logvar_recovers_mean_in_expectation():
    rng = jax.random.PRNGKey(0)
    mean = jnp.array([[1.0, -2.0, 0.5]])
    logvar = jnp.full_like(mean, -10.0)  # ~zero variance
    z = reparameterize(rng, mean, logvar)
    # With near-zero variance, sample is essentially the mean
    assert jnp.allclose(z, mean, atol=2e-2)
