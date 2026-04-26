"""Combined autoencoder ELBO + predictor MSE loss for Phase 1 pre-training."""
from typing import Any

import jax
import jax.numpy as jnp


def kl_to_unit_gaussian(mean: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
    """Mean (over batch) KL(N(mean, exp(logvar)) || N(0, I))."""
    per_dim = -0.5 * (1.0 + logvar - jnp.square(mean) - jnp.exp(logvar))
    return jnp.mean(jnp.sum(per_dim, axis=-1))


def reparameterize(rng: jax.Array, mean: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
    eps = jax.random.normal(rng, shape=mean.shape)
    return mean + jnp.exp(0.5 * logvar) * eps


def pretrain_loss(
    encoder, decoder, predictor,
    enc_params, dec_params, pred_params,
    inputs: jnp.ndarray,         # (B, w, feat)
    targets: jnp.ndarray,        # (B, n, feat)
    rng: jax.Array,
    beta_kl: float,
    w_pred: float,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    mean, logvar = encoder.apply(enc_params, inputs)
    z = reparameterize(rng, mean, logvar)
    recon = decoder.apply(dec_params, z)
    pred = predictor.apply(pred_params, z)

    recon_loss = jnp.mean(jnp.square(recon - inputs))
    kl_loss = kl_to_unit_gaussian(mean, logvar)
    pred_loss = jnp.mean(jnp.square(pred - targets))

    total = recon_loss + beta_kl * kl_loss + w_pred * pred_loss
    return total, {"recon": recon_loss, "kl": kl_loss, "pred": pred_loss}
