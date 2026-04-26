import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import freeze, unfreeze

from track_mjx.agent.latent_ppo.losses.pretrain_losses import pretrain_loss
from track_mjx.agent.latent_ppo.networks.decoder import MotionDecoder
from track_mjx.agent.latent_ppo.networks.encoder import MotionEncoder
from track_mjx.agent.latent_ppo.networks.predictor import MotionPredictor


def test_loss_decreases_over_100_steps():
    rng = jax.random.PRNGKey(0)
    rng_data, rng_init, rng_step = jax.random.split(rng, 3)

    feat, w, n, latent = 6, 4, 2, 8
    enc = MotionEncoder(layer_sizes=(32,), latent_dim=latent)
    dec = MotionDecoder(layer_sizes=(32,), window_len=w, feat_dim=feat)
    pred = MotionPredictor(layer_sizes=(32,), horizon=n, feat_dim=feat)

    inputs = jax.random.normal(rng_data, (16, w, feat))
    # Targets: a deterministic linear function of input window mean for predictability
    targets = jnp.tile(jnp.mean(inputs, axis=1, keepdims=True), (1, n, 1))

    enc_p = enc.init(rng_init, inputs)
    dec_p = dec.init(rng_init, jnp.ones((16, latent)))
    pred_p = pred.init(rng_init, jnp.ones((16, latent)))

    params = {"enc": enc_p, "dec": dec_p, "pred": pred_p}
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    def loss_fn(params, rng):
        return pretrain_loss(
            enc, dec, pred, params["enc"], params["dec"], params["pred"],
            inputs, targets, rng=rng, beta_kl=1e-3, w_pred=1.0,
        )

    @jax.jit
    def step(params, opt_state, rng):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, rng)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    losses = []
    for i in range(100):
        rng_step, k = jax.random.split(rng_step)
        params, opt_state, loss, aux = step(params, opt_state, k)
        losses.append(float(loss))

    assert losses[-1] < 0.5 * losses[0], f"loss did not decrease: {losses[0]} -> {losses[-1]}"
