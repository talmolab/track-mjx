"""Combined autoencoder ELBO + predictor MSE loss for Phase 1 pre-training."""
from typing import Any, Optional

import jax
import jax.numpy as jnp


def kl_to_unit_gaussian(mean: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
    """Mean (over batch) KL(N(mean, exp(logvar)) || N(0, I))."""
    per_dim = -0.5 * (1.0 + logvar - jnp.square(mean) - jnp.exp(logvar))
    return jnp.mean(jnp.sum(per_dim, axis=-1))


def reparameterize(rng: jax.Array, mean: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
    eps = jax.random.normal(rng, shape=mean.shape)
    return mean + jnp.exp(0.5 * logvar) * eps


def _masked_mse(pred: jnp.ndarray, target: jnp.ndarray, feat_mask: Optional[jnp.ndarray]):
    """Mean of squared error, restricted to feat_mask=1 channels along the last axis."""
    err = jnp.square(pred - target)
    if feat_mask is None:
        return jnp.mean(err)
    # err: (..., feat). Average over leading axes per dim, then mean over active dims only.
    per_dim = jnp.mean(err, axis=tuple(range(err.ndim - 1)))   # (feat,)
    return (per_dim * feat_mask).sum() / jnp.maximum(feat_mask.sum(), 1.0)


def _sigma_cap_penalty(logvar: jnp.ndarray, sigma_max) -> jnp.ndarray:
    """Hinge penalty pushing posterior std below ``sigma_max``.

    Per-dim: relu(logvar - log(sigma_max^2)). Sum over latent dims, mean over batch.
    Encourages encoder to use sharp posteriors (z ≈ mean) so reconstructions are
    not blurred by reparameterization noise during training.

    ``sigma_max`` may be either a Python float (legacy fixed cap) or a JAX
    scalar (curriculum schedule). Floor at 1e-8 to keep log finite.
    """
    sm = jnp.maximum(jnp.asarray(sigma_max, dtype=logvar.dtype), 1e-8)
    cap = 2.0 * jnp.log(sm)
    excess = jnp.maximum(logvar - cap, 0.0)   # (B, latent)
    return jnp.mean(jnp.sum(excess, axis=-1))


def pretrain_loss(
    encoder, decoder, predictor,
    enc_params, dec_params, pred_params,
    inputs: jnp.ndarray,         # (B, w, feat)
    targets: jnp.ndarray,        # (B, n, feat)
    rng: jax.Array,
    beta_kl: float,
    w_pred: float,
    feat_mask: Optional[jnp.ndarray] = None,
    deterministic: bool = False,
    sigma_max: float = 0.0,      # 0 disables the cap
    w_sigma: float = 0.0,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    mean, logvar = encoder.apply(enc_params, inputs)
    z = mean if deterministic else reparameterize(rng, mean, logvar)
    recon = decoder.apply(dec_params, z)
    pred = predictor.apply(pred_params, z)

    recon_loss = _masked_mse(recon, inputs, feat_mask)
    kl_loss = kl_to_unit_gaussian(mean, logvar)
    pred_loss = _masked_mse(pred, targets, feat_mask)
    # Always compute the sigma_pen — w_sigma=0 disables it without needing a
    # Python-level branch (which would break under JIT when sigma_max is a
    # JAX-traced scalar from a curriculum schedule).
    sigma_pen = _sigma_cap_penalty(logvar, sigma_max)

    total = recon_loss + beta_kl * kl_loss + w_pred * pred_loss + w_sigma * sigma_pen
    return total, {
        "recon": recon_loss,
        "kl": kl_loss,
        "pred": pred_loss,
        "sigma_pen": sigma_pen,
    }
