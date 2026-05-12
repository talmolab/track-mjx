"""Shared-CNN PPO networks for binocular vision + task_obs transfer learning.

Mirrors the ``shared_vision_network.py`` pattern where a single CNN is shared
between policy and value heads, but uses a ``BinocularVisionEncoder`` instead
of a monocular ``VisionEncoder``.  Both policy_loss and v_loss gradients flow
through the binocular CNN.

Architecture::

        vision (H,W,2C) ──> [shared BinocularVisionEncoder] ──> vision_features
                                         │
                     ┌───────────────────┴───────────────────┐
                     │                                       │
             Policy head                              Value head
             [vision_features, task_obs]              [vision_features, task_obs]
                 │ fusion MLP                             │ value MLP
                 │ latent z                               └──> scalar value
                 │ decoder
                 └──> action params

The ``BinocularVisionEncoder`` supports two modes:
- ``shared_weights=True`` (Siamese): A single CNN processes both eyes.
- ``shared_weights=False`` (Independent): Two separate CNNs per eye.

Both modes produce output shape [..., 2 * vision_feature_size] which is then
fed into the fusion MLP and value head.

All parameters live in a single ``BinocularSharedVisionPolicyValueModule``.
In the ``PPONetworkParams`` dataclass, ``params.policy`` holds the full module
(binocular CNN + policy head + value head) and ``params.value`` is empty.
"""

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
from brax.training import networks, types
from flax import linen as nn

from track_mjx.agent.ff_ppo.intention_network import Decoder
from track_mjx.agent.ff_ppo.binocular_vision_encoder import BinocularVisionEncoder
from track_mjx.agent.observation_utils import (
    DictRunningStatisticsState,
    normalize_dict_obs,
)


class BinocularSharedVisionPolicyValueModule(nn.Module):
    """Combined policy + value network with a single shared BinocularVisionEncoder.

    The binocular CNN is instantiated once and its features are used by both
    the policy fusion path and the value MLP.  When ``jax.grad`` is computed on
    ``policy_loss + v_loss``, the CNN receives gradient contributions from
    **both** losses.

    Attributes:
        decoder_layers: Layer sizes for the policy decoder (including action
            output dimension).
        latent_size: Dimension of the fusion MLP output / latent vector.
        vision_feature_size: Output dimension per eye of the BinocularVisionEncoder.
            Total binocular feature size is 2 * vision_feature_size.
        vision_channels: Channel sizes for each conv layer.
        fusion_layers: Hidden layer sizes for the fusion MLP.
        value_layers: Hidden layer sizes for the value head MLP.
        shared_weights: If True, use shared-weight Siamese architecture for
            the binocular encoder. If False, use independent dual-CNN.
        mono_channels: Number of channels per eye (1 for grayscale, 3 for RGB).
    """

    decoder_layers: Sequence[int]
    latent_size: int = 16
    vision_feature_size: int = 32
    vision_channels: Sequence[int] = (4, 8, 16, 32)
    fusion_layers: Sequence[int] = (256,)
    value_layers: Sequence[int] = (512, 512)
    shared_weights: bool = True
    mono_channels: int = 1
    activation: networks.ActivationFn = nn.silu

    def setup(self):
        # ── Shared Binocular CNN ─────────────────────────────────────
        self.vision_encoder = BinocularVisionEncoder(
            feature_size=self.vision_feature_size,
            channels=self.vision_channels,
            mono_channels=self.mono_channels,
            shared_weights=self.shared_weights,
        )

        # ── Policy head: fusion MLP + decoder ───────────────────────
        fusion_dense = []
        fusion_norms = []
        for h in self.fusion_layers:
            fusion_dense.append(nn.Dense(h))
            fusion_norms.append(nn.LayerNorm())
        self.fusion_dense = fusion_dense
        self.fusion_norms = fusion_norms
        self.fusion_out = nn.Dense(self.latent_size)
        self.decoder = Decoder(layer_sizes=self.decoder_layers, activation=self.activation)

        # ── Value head: MLP ─────────────────────────────────────────
        self.value_head = Decoder(
            layer_sizes=list(self.value_layers) + [1],
            activation=self.activation,
        )

    # ------------------------------------------------------------------ #
    #  Forward passes                                                      #
    # ------------------------------------------------------------------ #

    def __call__(
        self,
        obs: Mapping[str, jnp.ndarray],
        key: jax.Array,
        deterministic: bool = False,
    ):
        """Full forward: shared binocular CNN -> policy output + value output.

        Returns:
            Tuple of (action_params, latent_mean, latent_logvar, value).
        """
        task_obs = obs["imitation_target"]
        egocentric_obs = obs["proprioception"]
        vision = obs["vision"]

        # ── Shared Binocular CNN ─────────────────────────────────────
        vision_features = self.vision_encoder(vision)

        # ── Policy path ────────────────────────────────────────────
        combined = jnp.concatenate([vision_features, task_obs], axis=-1)
        z = combined
        for dense, norm in zip(self.fusion_dense, self.fusion_norms):
            z = self.activation(dense(z))
            z = norm(z)
        z = self.fusion_out(z)

        latent_mean = z
        latent_logvar = jnp.full_like(z, -20.0)  # ~deterministic

        concatenated = jnp.concatenate([z, egocentric_obs], axis=-1)
        action, _ = self.decoder(concatenated)

        # ── Value path (uses same vision_features) ──────────────────
        value_input = jnp.concatenate([vision_features, task_obs], axis=-1)
        value, _ = self.value_head(value_input)
        value = jnp.squeeze(value, axis=-1)

        return action, latent_mean, latent_logvar, value

    def policy_forward(
        self,
        obs: Mapping[str, jnp.ndarray],
        key: jax.Array,
        deterministic: bool = False,
        get_activation: bool = False,
    ):
        """Policy-only forward (for inference).

        Returns 3 values normally, or 4 with ``get_activation=True``
        (compatible with ``make_inference_fn`` in ``ppo_networks.py``).
        """
        task_obs = obs["imitation_target"]
        egocentric_obs = obs["proprioception"]
        vision = obs["vision"]

        if get_activation:
            vision_features, cnn_activations = self.vision_encoder(
                vision, get_activation=True
            )
        else:
            vision_features = self.vision_encoder(vision)

        combined = jnp.concatenate([vision_features, task_obs], axis=-1)
        z = combined
        for dense, norm in zip(self.fusion_dense, self.fusion_norms):
            z = self.activation(dense(z))
            z = norm(z)
        z = self.fusion_out(z)

        latent_mean = z
        latent_logvar = jnp.full_like(z, -20.0)

        concatenated = jnp.concatenate([z, egocentric_obs], axis=-1)

        if get_activation:
            action, decoder_activations = self.decoder(
                concatenated, get_activation=True
            )
            return (
                action,
                latent_mean,
                latent_logvar,
                {
                    "decoder": decoder_activations,
                    "vision_features": vision_features,
                    "cnn": cnn_activations,
                    "egocentric_obs": egocentric_obs,
                    "task_obs": task_obs,
                    "intention": z,
                },
            )
        else:
            action, _ = self.decoder(concatenated)
            return action, latent_mean, latent_logvar


# ====================================================================== #
#  Factory helpers                                                         #
# ====================================================================== #


def make_binocular_shared_vision_policy(
    action_param_size: int,
    latent_size: int,
    obs_sizes: Mapping[str, int],
    vision_shape: tuple[int, int, int],
    decoder_hidden_layer_sizes: Sequence[int] = (512, 512),
    value_hidden_layer_sizes: Sequence[int] = (512, 512),
    vision_feature_size: int = 32,
    vision_channels: Sequence[int] = (4, 8, 16, 32),
    fusion_hidden_layer_sizes: Sequence[int] = (256,),
    mono_channels: int = 1,
    shared_weights: bool = True,
    activation: networks.ActivationFn = nn.silu,
) -> tuple[networks.FeedForwardNetwork, "BinocularSharedVisionPolicyValueModule"]:
    """Create a shared binocular-CNN policy network and return the underlying module.

    The returned ``FeedForwardNetwork`` is inference-compatible (3 return
    values) and can be used with the standard ``make_inference_fn``.

    Returns:
        Tuple of (policy_feed_forward_network, shared_module_instance).
        The shared module is needed by the loss function.
    """
    shared_module = BinocularSharedVisionPolicyValueModule(
        decoder_layers=list(decoder_hidden_layer_sizes) + [action_param_size],
        latent_size=latent_size,
        vision_feature_size=vision_feature_size,
        vision_channels=list(vision_channels),
        fusion_layers=list(fusion_hidden_layer_sizes),
        value_layers=list(value_hidden_layer_sizes),
        shared_weights=shared_weights,
        mono_channels=mono_channels,
        activation=activation,
    )

    def apply(
        processor_params: DictRunningStatisticsState,
        policy_params,
        obs: Mapping[str, jnp.ndarray],
        key,
        deterministic: bool = False,
        get_activation: bool = False,
    ):
        """Apply policy with observation normalization (inference path)."""
        obs = normalize_dict_obs(obs, processor_params)
        return shared_module.apply(
            policy_params,
            obs=obs,
            key=key,
            deterministic=deterministic,
            get_activation=get_activation,
            method="policy_forward",
        )

    # Dummy inputs for initialization
    dummy_obs = {
        "imitation_target": jnp.zeros((1, obs_sizes.get("imitation_target", 0))),
        "proprioception": jnp.zeros((1, obs_sizes.get("proprioception", 0))),
        "vision": jnp.zeros((1,) + vision_shape),
    }
    dummy_key = jax.random.PRNGKey(0)

    policy_network = networks.FeedForwardNetwork(
        init=lambda key: shared_module.init(key, dummy_obs, dummy_key),
        apply=apply,
    )

    return policy_network, shared_module


def make_binocular_shared_vision_value_stub() -> networks.FeedForwardNetwork:
    """Create a stub value network that holds no parameters.

    In the shared-CNN architecture, the value head lives inside the
    ``BinocularSharedVisionPolicyValueModule`` (i.e. inside ``params.policy``).
    This stub exists only to satisfy the ``PPOImitationNetworks`` container
    and ``PPONetworkParams`` initialization.
    """

    def apply(processor_params, value_params, obs):
        raise RuntimeError(
            "BinocularSharedVision value stub should never be called directly. "
            "Use compute_shared_vision_ppo_loss instead."
        )

    return networks.FeedForwardNetwork(
        init=lambda key: {},
        apply=apply,
    )
