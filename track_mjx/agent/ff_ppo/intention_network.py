"""Intention network architectures for VAE-style imitation learning.

This module provides encoder-decoder neural network architectures.
The key components are:

- Encoder: Maps reference trajectory observations to a latent intention space
- Decoder: Maps latent intentions + proprioceptive state to action parameters
- IntentionNetwork: Full VAE combining encoder and decoder with reparameterization

The architecture enables learning from motion capture data by encoding trajectory
information into a compact latent space that conditions the policy.

Observations are expected as dictionaries with keys:
- "imitation_target": Reference trajectory observations (flat array)
- "proprioception": Proprioceptive state observations (flat array)
"""

from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from brax.training import networks, types
from flax import linen as nn

from track_mjx.agent.observation_utils import (
    DictRunningStatisticsState,
    normalize_dict_obs,
)


class Encoder(nn.Module):
    """VAE encoder that maps observations to latent distribution parameters.

    Processes reference trajectory observations through an MLP with LayerNorm
    to produce mean and log-variance of the latent intention distribution.

    Attributes:
        layer_sizes: Hidden layer dimensions for the MLP.
        latents: Dimension of the latent intention space.
        activation: Activation function (default: SiLU).
        kernel_init: Weight initializer (default: LeCun uniform).
        bias: Whether to use bias terms in Dense layers.
    """

    layer_sizes: Sequence[int]
    latents: int
    activation: networks.ActivationFn = nn.silu
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()
    bias: bool = True

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, get_activation: bool = False
    ) -> tuple[jnp.ndarray, jnp.ndarray] | tuple[tuple[jnp.ndarray, jnp.ndarray], dict]:
        activations = {}
        # For each layer in the sequence
        for i, hidden_size in enumerate(self.layer_sizes):
            x = nn.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(x)
            x = self.activation(x)
            x = nn.LayerNorm()(x)
            if get_activation:
                activations[f"layer_{i}"] = x

        mean_x = nn.Dense(self.latents, name="fc2_mean")(x)
        logvar_x = nn.Dense(self.latents, name="fc2_logvar")(x)

        if get_activation:
            activations["mean"] = mean_x
            activations["logvar"] = logvar_x
            return (mean_x, logvar_x), activations
        return mean_x, logvar_x


class Decoder(nn.Module):
    """VAE decoder that maps latent intentions to action distribution parameters.

    Processes concatenated latent intention and proprioceptive observations
    through an MLP to produce action distribution parameters.

    Attributes:
        layer_sizes: Layer dimensions including final output size.
        activation: Activation function (default: SiLU).
        kernel_init: Weight initializer (default: LeCun uniform).
        activate_final: Whether to apply activation after final layer.
        bias: Whether to use bias terms in Dense layers.
    """

    layer_sizes: Sequence[int]
    activation: networks.ActivationFn = nn.silu
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, get_activation: bool = False
    ) -> tuple[jnp.ndarray, dict]:
        if get_activation:
            activations = {}
        for i, hidden_size in enumerate(self.layer_sizes):
            x = nn.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(x)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                x = self.activation(x)
                x = nn.LayerNorm()(x)
                if get_activation:
                    activations[f"layer_{i}"] = x
        if get_activation:
            return x, activations
        return x, {}


def reparameterize_single(
    rng: jax.Array, mean: jnp.ndarray, logvar: jnp.ndarray
) -> jnp.ndarray:
    """Sample from Gaussian using reparameterization trick (single sample).

    Enables backpropagation through stochastic sampling by expressing the
    sample as a deterministic function of the parameters plus noise.

    Args:
        rng: JAX random key for sampling (shape [2]).
        mean: Mean of the Gaussian distribution (shape [latent_dim]).
        logvar: Log-variance of the Gaussian distribution (shape [latent_dim]).

    Returns:
        Sampled latent vector: mean + std * epsilon, where epsilon ~ N(0, I).
    """
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(rng, logvar.shape)
    return mean + eps * std


def reparameterize(
    rng: jax.Array, mean: jnp.ndarray, logvar: jnp.ndarray
) -> jnp.ndarray:
    """Sample from Gaussian using reparameterization trick (batched).

    Supports both single key (broadcasted) and per-sample keys for
    deterministic replay.

    Args:
        rng: JAX random key(s) for sampling. Either shape [2] (single key
            for all samples) or [batch_size, 2] (per-sample keys).
        mean: Mean of the Gaussian distribution, shape [latent_dim] (unbatched)
            or [batch_size, latent_dim] (batched).
        logvar: Log-variance of the Gaussian distribution, same shape as mean.

    Returns:
        Sampled latent vectors with same shape as mean.
    """
    if rng.ndim == 1:
        # Single key - use original behavior
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(rng, logvar.shape)
        return mean + eps * std
    elif mean.ndim == 1:
        # Per-sample keys but unbatched mean/logvar - use first key
        # This is a fallback safety for shape mismatches
        return reparameterize_single(rng[0], mean, logvar)
    else:
        # Per-sample keys with batched mean/logvar - vmap over batch dimension
        return jax.vmap(reparameterize_single)(rng, mean, logvar)


class IntentionNetwork(nn.Module):
    """Full VAE model combining encoder and decoder for intention-based policy.

    The network receives observations as a dictionary with keys:
    - "imitation_target": Reference trajectory observations (encoder input)
    - "proprioception": Proprioceptive state observations (decoder input with latent)

    The encoder processes trajectory observations to produce latent intentions,
    which are then concatenated with proprioceptive state and decoded into
    action distribution parameters.

    Attributes:
        encoder_layers: Hidden layer sizes for the encoder MLP.
        decoder_layers: Layer sizes for decoder (excluding action output).
        latents: Dimension of the latent intention space.
    """

    encoder_layers: Sequence[int]
    decoder_layers: Sequence[int]
    latents: int = 60

    def setup(self):
        """Initialize encoder and decoder submodules."""
        self.encoder = Encoder(layer_sizes=self.encoder_layers, latents=self.latents)
        self.decoder = Decoder(layer_sizes=self.decoder_layers)

    def __call__(
        self,
        obs: Mapping[str, jnp.ndarray],
        key: jax.Array,
        deterministic: bool = False,
        get_activation: bool = False,
    ):
        # Access observations by name
        traj = obs["imitation_target"]
        egocentric_obs = obs["proprioception"]

        # Check if observations are actually batched (based on normalized obs shape)
        obs_is_batched = traj.ndim >= 2

        # Handle key splitting based on both key shape AND observation shape
        if key.ndim == 1:
            # Single key - split for encoder
            _, encoder_rng = jax.random.split(key)
        elif not obs_is_batched:
            # Per-sample keys but unbatched observation - use first key
            # This can happen when key batching was determined from nested obs structure
            # before normalization flattened it to unbatched
            _, encoder_rng = jax.random.split(key[0])
        else:
            # Per-sample keys [batch_size, 2] - vmap split over batch
            _, encoder_rng = jax.vmap(jax.random.split)(key).swapaxes(0, 1)

        if get_activation:
            (latent_mean, latent_logvar), encoder_activations = self.encoder(
                traj, get_activation=True
            )
            # Uses mean in the case of deterministic evaluation
            if deterministic:
                z = latent_mean
            else:
                z = reparameterize(encoder_rng, latent_mean, latent_logvar)
            concatenated = jnp.concatenate([z, egocentric_obs], axis=-1)
            action, decoder_activations = self.decoder(
                concatenated, get_activation=True
            )
            return (
                action,
                latent_mean,
                latent_logvar,
                {
                    "encoder": encoder_activations,
                    "decoder": decoder_activations,
                    "egocentric_obs": egocentric_obs,
                    "traj_obs": traj,
                    "intention": z,
                },
            )
        else:
            latent_mean, latent_logvar = self.encoder(traj, get_activation=False)
            # Uses mean in the case of deterministic evaluation
            if deterministic:
                z = latent_mean
            else:
                z = reparameterize(encoder_rng, latent_mean, latent_logvar)
            action, _ = self.decoder(jnp.concatenate([z, egocentric_obs], axis=-1))
            return action, latent_mean, latent_logvar


def make_intention_policy(
    action_param_size: int,
    latent_size: int,
    obs_sizes: Mapping[str, int],
    encoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    decoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
) -> networks.FeedForwardNetwork:
    """Create an intention-based policy network.

    Constructs an encoder-decoder VAE policy where the encoder processes
    reference trajectory observations and the decoder generates action
    parameters conditioned on latent intentions and proprioceptive state.

    Args:
        action_param_size: Output dimension (typically 2x action_size for
            Gaussian mean and variance).
        latent_size: Dimension of the latent intention space.
        obs_sizes: Dict mapping observation keys to their sizes, e.g.
            {"imitation_target": 3716, "proprioception": 226}.
        encoder_hidden_layer_sizes: Hidden layer sizes for encoder MLP.
        decoder_hidden_layer_sizes: Hidden layer sizes for decoder MLP.

    Returns:
        FeedForwardNetwork with init and apply methods. The apply function
        returns (action_params, latent_mean, latent_logvar).
    """

    policy_module = IntentionNetwork(
        encoder_layers=list(encoder_hidden_layer_sizes),
        decoder_layers=list(decoder_hidden_layer_sizes)
        + [action_param_size],  # add action size to the last layer
        latents=latent_size,
    )

    def apply(
        processor_params: DictRunningStatisticsState,
        policy_params,
        obs: Mapping[str, jnp.ndarray],
        key,
        deterministic: bool = False,
        get_activation: bool = False,
    ):
        """Apply policy with observation normalization."""
        obs = normalize_dict_obs(obs, processor_params)
        return policy_module.apply(
            policy_params,
            obs=obs,
            key=key,
            deterministic=deterministic,
            get_activation=get_activation,
        )

    # Create dummy dict observation for initialization
    dummy_obs = {
        "imitation_target": jnp.zeros((1, obs_sizes["imitation_target"])),
        "proprioception": jnp.zeros((1, obs_sizes["proprioception"])),
    }
    dummy_key = jax.random.PRNGKey(0)

    return networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_obs, dummy_key),
        apply=apply,
    )


def make_decoder_policy(
    param_size: int,
    decoder_obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    decoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
) -> networks.FeedForwardNetwork:
    """Create a decoder-only policy network for downstream tasks.

    Creates a standalone decoder policy that can be used with externally
    provided latent intentions. Useful for transfer learning or hierarchical
    control where intentions come from a separate module.

    The normalizer params only apply to the proprioceptive portion of the
    observation (latent intentions are not normalized).

    Args:
        param_size: Output dimension for action distribution parameters.
        decoder_obs_size: Input dimension (latent_size + proprioceptive_size).
        preprocess_observations_fn: Normalization function for proprioceptive obs.
        decoder_hidden_layer_sizes: Hidden layer sizes for decoder MLP.

    Returns:
        FeedForwardNetwork with init and apply methods.
    """
    policy_module = Decoder(
        layer_sizes=list(decoder_hidden_layer_sizes) + [param_size],
    )

    def apply(processor_params, policy_params, obs):
        """Apply decoder with selective normalization of proprioceptive obs."""
        # Split obs into latent (unnormalized) and proprioceptive (normalized)
        latent_obs = obs[..., : -processor_params.mean.shape[-1]]
        proprio_obs = preprocess_observations_fn(
            obs[..., -processor_params.mean.shape[-1] :], processor_params
        )
        obs = jnp.concatenate([latent_obs, proprio_obs], axis=-1)
        return policy_module.apply(policy_params, x=obs, get_activation=False)

    dummy_obs = jnp.zeros((1, decoder_obs_size))

    return networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_obs),
        apply=apply,
    )


class VisionIntentionNetwork(nn.Module):
    """Intention network with vision encoder branch.

    Extends the IntentionNetwork to incorporate visual observations from an
    egocentric camera. The encoder receives concatenated trajectory features
    and vision features, while the decoder receives latent intentions
    concatenated with proprioceptive state.

    Observations dict must include:
    - "imitation_target": Reference trajectory observations
    - "proprioception": Proprioceptive state observations
    - "vision": Egocentric camera image (H, W, 3) normalized to [0, 1]

    Attributes:
        encoder_layers: Hidden layer sizes for the encoder MLP.
        decoder_layers: Layer sizes for decoder (including action output).
        latents: Dimension of the latent intention space.
        vision_feature_size: Output dimension of the vision encoder CNN.
        vision_channels: Channel sizes for each conv layer in the vision encoder.
    """

    encoder_layers: Sequence[int]
    decoder_layers: Sequence[int]
    latents: int = 60
    vision_feature_size: int = 128
    vision_channels: Sequence[int] = (32, 64, 64)

    def setup(self):
        """Initialize vision encoder, encoder, and decoder submodules."""
        from track_mjx.agent.ff_ppo.vision_encoder import VisionEncoder

        self.vision_encoder = VisionEncoder(
            feature_size=self.vision_feature_size,
            channels=self.vision_channels,
        )
        self.encoder = Encoder(layer_sizes=self.encoder_layers, latents=self.latents)
        self.decoder = Decoder(layer_sizes=self.decoder_layers)

    def __call__(
        self,
        obs: Mapping[str, jnp.ndarray],
        key: jax.Array,
        deterministic: bool = False,
        get_activation: bool = False,
    ):
        traj = obs["imitation_target"]
        egocentric_obs = obs["proprioception"]
        vision = obs["vision"]

        # Encode vision to feature vector
        vision_features = self.vision_encoder(vision)

        # Concat trajectory + vision features for encoder
        encoder_input = jnp.concatenate([traj, vision_features], axis=-1)

        # Key handling (same as IntentionNetwork)
        obs_is_batched = traj.ndim >= 2
        if key.ndim == 1:
            _, encoder_rng = jax.random.split(key)
        elif not obs_is_batched:
            _, encoder_rng = jax.random.split(key[0])
        else:
            _, encoder_rng = jax.vmap(jax.random.split)(key).swapaxes(0, 1)

        if get_activation:
            (latent_mean, latent_logvar), encoder_activations = self.encoder(
                encoder_input, get_activation=True
            )
            if deterministic:
                z = latent_mean
            else:
                z = reparameterize(encoder_rng, latent_mean, latent_logvar)
            concatenated = jnp.concatenate([z, egocentric_obs], axis=-1)
            action, decoder_activations = self.decoder(
                concatenated, get_activation=True
            )
            return (
                action,
                latent_mean,
                latent_logvar,
                {
                    "encoder": encoder_activations,
                    "decoder": decoder_activations,
                    "vision_features": vision_features,
                    "egocentric_obs": egocentric_obs,
                    "traj_obs": traj,
                    "intention": z,
                },
            )
        else:
            latent_mean, latent_logvar = self.encoder(
                encoder_input, get_activation=False
            )
            if deterministic:
                z = latent_mean
            else:
                z = reparameterize(encoder_rng, latent_mean, latent_logvar)
            action, _ = self.decoder(jnp.concatenate([z, egocentric_obs], axis=-1))
            return action, latent_mean, latent_logvar


def make_vision_intention_policy(
    action_param_size: int,
    latent_size: int,
    obs_sizes: Mapping[str, int],
    vision_shape: tuple[int, int, int],
    encoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    decoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    vision_feature_size: int = 128,
    vision_channels: Sequence[int] = (32, 64, 64),
) -> networks.FeedForwardNetwork:
    """Create a vision-enabled intention-based policy network.

    Constructs an encoder-decoder VAE policy where the encoder processes
    concatenated reference trajectory and vision features, and the decoder
    generates action parameters conditioned on latent intentions and
    proprioceptive state.

    Args:
        action_param_size: Output dimension (typically 2x action_size for
            Gaussian mean and variance).
        latent_size: Dimension of the latent intention space.
        obs_sizes: Dict mapping observation keys to their sizes, e.g.
            {"imitation_target": 100, "proprioception": 60}.
        vision_shape: Shape of the vision input (H, W, C), e.g. (64, 64, 3).
        encoder_hidden_layer_sizes: Hidden layer sizes for encoder MLP.
        decoder_hidden_layer_sizes: Hidden layer sizes for decoder MLP.
        vision_feature_size: Output dimension of the vision encoder CNN.
        vision_channels: Channel sizes for each conv layer in the vision encoder.

    Returns:
        FeedForwardNetwork with init and apply methods. The apply function
        returns (action_params, latent_mean, latent_logvar).
    """

    policy_module = VisionIntentionNetwork(
        encoder_layers=list(encoder_hidden_layer_sizes),
        decoder_layers=list(decoder_hidden_layer_sizes) + [action_param_size],
        latents=latent_size,
        vision_feature_size=vision_feature_size,
        vision_channels=vision_channels,
    )

    def apply(
        processor_params: DictRunningStatisticsState,
        policy_params,
        obs: Mapping[str, jnp.ndarray],
        key,
        deterministic: bool = False,
        get_activation: bool = False,
    ):
        """Apply policy with observation normalization."""
        obs = normalize_dict_obs(obs, processor_params)
        return policy_module.apply(
            policy_params,
            obs=obs,
            key=key,
            deterministic=deterministic,
            get_activation=get_activation,
        )

    # Create dummy dict observation for initialization
    dummy_obs = {
        "imitation_target": jnp.zeros((1, obs_sizes["imitation_target"])),
        "proprioception": jnp.zeros((1, obs_sizes["proprioception"])),
        "vision": jnp.zeros((1,) + vision_shape),
    }
    dummy_key = jax.random.PRNGKey(0)

    return networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_obs, dummy_key),
        apply=apply,
    )
