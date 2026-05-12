"""DMPO intention encoder-decoder networks for rodent imitation.

Mirrors track_mjx.agent.ff_ppo.intention_network.IntentionNetwork's topology
under DMPO's loss math. Encoder is reused verbatim from intention_network;
decoder is a fresh MLP torso terminating in a GaussianPolicyHead. The latent
is deterministic (z = mean) so the policy is a simple Gaussian and MPO's
analytical KL decomposition holds.

The critic is encoder-free and operates on concat([imit_target, proprio,
action]). Normalization is handled by the caller — both modules expect
already-normalized dict observations.

Approach (A) per spec: deterministic latent, no reparameterization sample.
Stochastic-latent variant (Approach B) is deferred — it requires designing
how the MPO dual variables handle the sample-noise floor in KL(online ||
target).
"""
import logging
from typing import Any, Callable, Mapping, Sequence

import flax.linen as nn
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.networks import (
    CategoricalCriticHead,
    DMPONetworks,
    GaussianPolicyHead,
)
from track_mjx.agent.ff_ppo.intention_network import (
    Encoder as IntentionEncoder,
    get_activation_fn,
)

tfd = tfp.distributions

log = logging.getLogger(__name__)


_VAE_KNOBS_TO_WARN = (
    "encoder_noise_std",
    "proprioception_noise_std",
    "latent_kl_weight",
    "latent_ar1_weight",
    "kl_schedule",
)


class IntentionDMPOPolicy(nn.Module):
    """Encoder-decoder policy: imitation_target -> z=mean -> decoder(z, proprio) -> Gaussian.

    Reuses ``track_mjx.agent.ff_ppo.intention_network.Encoder`` verbatim for
    the encoder MLP. Decoder is a Dense -> LayerNorm -> SiLU stack matching
    DMPO's existing _PolicyNet style. Final head is GaussianPolicyHead.
    """

    encoder_layer_sizes: Sequence[int]
    decoder_layer_sizes: Sequence[int]
    intention_size: int
    action_size: int
    activation: Callable = nn.silu

    def setup(self):
        self._encoder = IntentionEncoder(
            layer_sizes=tuple(self.encoder_layer_sizes),
            latents=self.intention_size,
            activation=self.activation,
        )

    def encode(self, obs: Mapping[str, jnp.ndarray]):
        """Return (latent_mean, latent_logvar). For Approach A the decoder
        uses only the mean, but logvar is exposed so the eval logger can
        report it (it'll be uninformative -- encoder weights still produce
        a logvar via the fc2_logvar head)."""
        return self._encoder(obs["imitation_target"])

    @nn.compact
    def __call__(self, obs: Mapping[str, jnp.ndarray]) -> tfd.Distribution:
        latent_mean, _logvar = self._encoder(obs["imitation_target"])
        z = latent_mean  # deterministic -- Approach (A); no reparam sample.
        h = jnp.concatenate([z, obs["proprioception"]], axis=-1)
        for size in self.decoder_layer_sizes:
            h = nn.Dense(size)(h)
            h = nn.LayerNorm()(h)
            h = self.activation(h)
        return GaussianPolicyHead(action_size=self.action_size)(h)


class FlatDMPOCritic(nn.Module):
    """Critic: concat([imit_target, proprio, action]) -> torso -> C51 head.

    Encoder-free by design (see spec Decision 2). Inputs are expected to
    be already normalized by the caller via running_statistics.
    """

    layer_sizes: Sequence[int]
    num_atoms: int
    vmin: float
    vmax: float
    activation: Callable = nn.silu

    @nn.compact
    def __call__(
        self, obs: Mapping[str, jnp.ndarray], action: jnp.ndarray
    ) -> tfd.Distribution:
        h = jnp.concatenate(
            [obs["imitation_target"], obs["proprioception"], action], axis=-1
        )
        for size in self.layer_sizes:
            h = nn.Dense(size)(h)
            h = nn.LayerNorm()(h)
            h = self.activation(h)
        return CategoricalCriticHead(
            num_atoms=self.num_atoms, vmin=self.vmin, vmax=self.vmax
        )(h)


def make_dmpo_intention_networks(
    *,
    obs_sizes: Mapping[str, int],
    action_size: int,
    cfg: DMPOConfig,
    network_cfg: Mapping[str, Any],
) -> DMPONetworks:
    """Factory for DMPO intention networks. Reads encoder/decoder sizes from
    ``network_cfg`` and num_atoms/vmin/vmax/critic_layer_sizes from ``cfg``.

    Logs warnings for VAE-specific knobs that are ignored under deterministic
    DMPO so users see they were noted but not honored.

    Args:
      obs_sizes: dict {"imitation_target": int, "proprioception": int}.
        Currently informational only -- networks read sizes from the actual
        obs at init/apply time.
      action_size: action dimensionality.
      cfg: DMPOConfig (used for num_atoms, vmin, vmax, critic_layer_sizes).
      network_cfg: dict-like with keys: encoder_layer_sizes,
        decoder_layer_sizes, intention_size, activation (str). Optional:
        stochastic_latent (bool, must be False -- warn if True), encoder_noise_std,
        proprioception_noise_std, latent_kl_weight, latent_ar1_weight,
        kl_schedule (all warned-as-ignored if non-default).
    """
    del obs_sizes  # see docstring; networks read sizes from obs at init.

    if bool(network_cfg.get("stochastic_latent", False)):
        log.warning(
            "network_config.stochastic_latent=True is IGNORED -- Approach A "
            "uses deterministic z=mean for MPO loss compatibility."
        )
    for k in _VAE_KNOBS_TO_WARN:
        v = network_cfg.get(k, None)
        if v is None:
            continue
        if k == "kl_schedule" and not bool(v):
            continue
        if isinstance(v, (int, float)) and float(v) == 0.0:
            continue
        log.warning(
            "network_config.%s=%s is IGNORED under DMPO+deterministic encoder.",
            k, v,
        )

    activation_name = str(network_cfg.get("activation", "silu"))
    activation = get_activation_fn(activation_name)

    return DMPONetworks(
        policy=IntentionDMPOPolicy(
            encoder_layer_sizes=tuple(network_cfg["encoder_layer_sizes"]),
            decoder_layer_sizes=tuple(network_cfg["decoder_layer_sizes"]),
            intention_size=int(network_cfg["intention_size"]),
            action_size=action_size,
            activation=activation,
        ),
        critic=FlatDMPOCritic(
            layer_sizes=tuple(cfg.critic_layer_sizes),
            num_atoms=cfg.num_atoms,
            vmin=cfg.vmin,
            vmax=cfg.vmax,
            activation=activation,
        ),
    )
