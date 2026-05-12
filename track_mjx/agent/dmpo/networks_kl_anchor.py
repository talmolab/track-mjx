"""B-aggressive DMPO networks for kl-anchor mode.

Trainable Prior + Decoder + PolicyHead, with optional warm-start of prior
and decoder weights from a frozen imitation checkpoint. Critic is the
existing CNN + task_obs + action -> CategoricalCriticHead from
networks_vision_bottleneck.py.

The Prior and Decoder modules mirror SCAMPER's
`scamper.agent.mlp_prior.prior_networks.Prior` and
`scamper.agent.imitation.intention_network.Decoder` in flax param-tree
shape (hidden_i / LayerNorm_i / fc2_mean / fc2_logvar) so that warm-start
splicing works by direct param-tree assignment. The PolicyHead's last
Dense layer is zero-initialised, which makes the residual exactly 0 at
step 0 -- combined with warm-started prior+decoder, this guarantees that
the trainable policy reproduces the frozen imitation pipeline bit-for-bit
at initialisation.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.networks import DMPONetworks
from track_mjx.agent.dmpo.networks_vision_bottleneck import (
    _ValueVisionCriticNet,
)
from track_mjx.agent.ff_ppo.binocular_vision_encoder import BinocularVisionEncoder

tfd = tfp.distributions


class _PriorModule(nn.Module):
    """Mirrors scamper.agent.mlp_prior.prior_networks.Prior in shape.

    Each block is `Dense -> silu -> LayerNorm`. The final layer has two
    parallel heads, `fc2_mean` and `fc2_logvar`. Names are explicit so
    the param tree matches SCAMPER's auto-naming exactly:
    `hidden_0`, `LayerNorm_0`, `hidden_1`, `LayerNorm_1`, `fc2_mean`,
    `fc2_logvar`.
    """

    layer_sizes: Sequence[int]
    latents: int

    @nn.compact
    def __call__(self, x):
        for i, h in enumerate(self.layer_sizes):
            x = nn.Dense(h, name=f"hidden_{i}")(x)
            x = nn.silu(x)
            x = nn.LayerNorm(name=f"LayerNorm_{i}")(x)
        mean = nn.Dense(self.latents, name="fc2_mean")(x)
        logvar = nn.Dense(self.latents, name="fc2_logvar")(x)
        return mean, logvar


class _DecoderModule(nn.Module):
    """Mirrors scamper.agent.imitation.intention_network.Decoder in shape.

    Each block is `Dense -> silu -> LayerNorm` *except* the last one, which
    is a bare Dense (no activation, no LayerNorm). `layer_sizes` already
    includes the final output dimension (typically 2 * action_size).
    """

    layer_sizes: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for i, h in enumerate(self.layer_sizes):
            x = nn.Dense(h, name=f"hidden_{i}")(x)
            if i < len(self.layer_sizes) - 1:
                x = nn.silu(x)
                x = nn.LayerNorm(name=f"LayerNorm_{i}")(x)
        return x


class _PolicyHeadModule(nn.Module):
    """CNN + task_obs + proprio -> residual (zero-init last layer).

    The output has shape `(..., latent_size)` and is added to the prior
    mean to produce the latent z fed to the decoder. The last `Dense` is
    zero-initialised in both kernel and bias so residual = 0 at step 0.
    """

    layer_sizes: Sequence[int]
    latent_size: int
    vision_shape: tuple
    cnn_feature_size: int
    cnn_channels: Sequence[int]
    mono_channels: int
    shared_weights: bool

    @nn.compact
    def __call__(self, vision, task_obs, proprio):
        vis = BinocularVisionEncoder(
            feature_size=self.cnn_feature_size,
            channels=tuple(self.cnn_channels),
            mono_channels=self.mono_channels,
            shared_weights=self.shared_weights,
        )(vision)
        h = jnp.concatenate([vis, task_obs, proprio], axis=-1)
        for i, size in enumerate(self.layer_sizes):
            h = nn.Dense(size, name=f"hidden_{i}")(h)
            h = nn.LayerNorm(name=f"LayerNorm_{i}")(h)
            h = nn.silu(h)
        residual = nn.Dense(
            self.latent_size,
            name="residual",
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(h)
        return residual


class _BAggressivePolicyNet(nn.Module):
    """Trainable Prior + PolicyHead + Decoder.

    Step-0 forward pass (with warm-started prior + decoder, zero-init
    residual):
        z = prior.mean(proprio) + 0
        (mu, log_sigma) = decoder([z, proprio]).split(2, axis=-1)
        dist = N(mu, softplus(log_sigma) + 1e-3)  # matches brax NormalTanhDistribution
    which exactly reproduces the frozen imitation pipeline.
    """

    action_size: int
    latent_size: int
    vision_shape: tuple
    prior_layer_sizes: Sequence[int]
    decoder_layer_sizes: Sequence[int]
    policy_head_layer_sizes: Sequence[int]
    cnn_feature_size: int
    cnn_channels: Sequence[int]
    mono_channels: int
    shared_weights: bool

    def setup(self):
        self.prior = _PriorModule(
            layer_sizes=tuple(self.prior_layer_sizes),
            latents=self.latent_size,
            name="prior",
        )
        self.decoder = _DecoderModule(
            layer_sizes=tuple(self.decoder_layer_sizes) + (2 * self.action_size,),
            name="decoder",
        )
        self.policy_head = _PolicyHeadModule(
            layer_sizes=tuple(self.policy_head_layer_sizes),
            latent_size=self.latent_size,
            vision_shape=self.vision_shape,
            cnn_feature_size=self.cnn_feature_size,
            cnn_channels=tuple(self.cnn_channels),
            mono_channels=self.mono_channels,
            shared_weights=self.shared_weights,
            name="policy_head",
        )

    def __call__(self, obs):
        vision = obs["vision"]
        task_obs = obs["imitation_target"]
        proprio = obs["proprioception"]

        z_prior, _ = self.prior(proprio)
        residual = self.policy_head(vision, task_obs, proprio)
        z = z_prior + residual

        decoder_input = jnp.concatenate([z, proprio], axis=-1)
        decoder_out = self.decoder(decoder_input)
        mu = decoder_out[..., : self.action_size]
        log_std = decoder_out[..., self.action_size :]
        # The imit decoder's log_std head was trained against
        # brax.training.distribution.NormalTanhDistribution, whose internal
        # create_dist uses scale = softplus(raw) + min_std (default 1e-3).
        # Use the SAME transform here so the warm-start splice is
        # bit-identical to the frozen pipeline at step 0. Using exp(log_std)
        # — the original plan's choice — would interpret the same raw value
        # as a ~44%-wider gaussian, breaking warm-start sample distributions.
        scale = jax.nn.softplus(log_std) + 1e-3
        return tfd.MultivariateNormalDiag(loc=mu, scale_diag=scale)


def _splice_warm_start(
    params: Dict, prior_params: Optional[Dict], decoder_params: Optional[Dict]
) -> Dict:
    """Replace the `prior` and/or `decoder` subtrees in `params['params']`
    with the supplied warm-start params.

    The warm-start params should be the *inner* dict (i.e. without the
    outer "params" wrapper), matching the convention in
    `scamper.agent.mlp_prior.prior_networks.load_frozen_encoder_decoder`.
    """
    if prior_params is None and decoder_params is None:
        return params
    new_inner = dict(params["params"])
    if prior_params is not None:
        new_inner["prior"] = prior_params
    if decoder_params is not None:
        new_inner["decoder"] = decoder_params
    return {"params": new_inner}


def make_dmpo_kl_anchor_networks(
    *,
    proprio_size: int,
    task_obs_size: int,
    action_size: int,
    latent_size: int,
    vision_shape: tuple,
    cfg: DMPOConfig,
    prior_layer_sizes: Sequence[int],
    decoder_layer_sizes: Sequence[int],
    policy_head_layer_sizes: Sequence[int] = (256, 256, 256),
    cnn_feature_size: int = 32,
    cnn_channels: Sequence[int] = (4, 8, 16, 32),
    mono_channels: int = 1,
    shared_weights: bool = True,
    value_hidden_layer_sizes: Sequence[int] = (512, 512, 512, 512),
    warm_start_prior_params: Optional[Dict] = None,
    warm_start_decoder_params: Optional[Dict] = None,
) -> DMPONetworks:
    """Build B-aggressive DMPO policy + critic for kl-anchor training.

    Args:
        proprio_size: Proprioceptive observation size (unused directly --
            kept for symmetry with `make_dmpo_bottleneck_vision_networks`).
        task_obs_size: Task observation size (unused directly).
        action_size: Action dimensionality.
        latent_size: Intention latent space size.
        vision_shape: Vision tensor shape, e.g. (H, W, 2*C) for binocular.
        cfg: DMPOConfig with C51 atom config.
        prior_layer_sizes: Hidden sizes for the trainable prior.
        decoder_layer_sizes: Hidden sizes for the trainable decoder
            (excludes the final 2 * action_size output layer, which is
            appended automatically).
        policy_head_layer_sizes: Hidden sizes for the residual policy
            head (post-CNN fusion MLP).
        cnn_feature_size, cnn_channels, mono_channels, shared_weights:
            BinocularVisionEncoder hyperparameters.
        value_hidden_layer_sizes: Hidden sizes for the critic MLP.
        warm_start_prior_params: Optional flax inner-params dict to splice
            into `params['params']['prior']` after init. Should match the
            shape of `_PriorModule(layer_sizes=prior_layer_sizes,
            latents=latent_size).init(...)['params']`.
        warm_start_decoder_params: Optional flax inner-params dict to
            splice into `params['params']['decoder']` after init.

    Returns:
        DMPONetworks(policy, critic).
    """
    del proprio_size, task_obs_size  # unused directly; consumed via obs dict.

    policy = _BAggressivePolicyNet(
        action_size=action_size,
        latent_size=latent_size,
        vision_shape=vision_shape,
        prior_layer_sizes=tuple(prior_layer_sizes),
        decoder_layer_sizes=tuple(decoder_layer_sizes),
        policy_head_layer_sizes=tuple(policy_head_layer_sizes),
        cnn_feature_size=cnn_feature_size,
        cnn_channels=tuple(cnn_channels),
        mono_channels=mono_channels,
        shared_weights=shared_weights,
    )

    critic = _ValueVisionCriticNet(
        layer_sizes=tuple(value_hidden_layer_sizes),
        num_atoms=cfg.num_atoms,
        vmin=cfg.vmin,
        vmax=cfg.vmax,
        vision_shape=vision_shape,
        cnn_feature_size=cnn_feature_size,
        cnn_channels=tuple(cnn_channels),
        mono_channels=mono_channels,
        shared_weights=shared_weights,
    )

    _orig_init = policy.init

    def policy_init(rng, obs):
        params = _orig_init(rng, obs)
        return _splice_warm_start(
            params, warm_start_prior_params, warm_start_decoder_params
        )

    policy.init = policy_init  # type: ignore[method-assign]

    return DMPONetworks(policy=policy, critic=critic)
