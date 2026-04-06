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

_ACTIVATION_MAP: dict[str, networks.ActivationFn] = {
    "silu": nn.silu,
    "relu": nn.relu,
    "tanh": nn.tanh,
    "gelu": nn.gelu,
    "elu": nn.elu,
}


def get_activation_fn(name: str) -> networks.ActivationFn:
    """Resolve an activation function name to a callable.

    Args:
        name: Activation function name (case-insensitive). Supported:
            "silu", "relu", "tanh", "gelu", "elu".

    Returns:
        The corresponding Flax/JAX activation function.

    Raises:
        ValueError: If name is not recognized.
    """
    fn = _ACTIVATION_MAP.get(name.lower())
    if fn is None:
        raise ValueError(
            f"Unknown activation {name!r}. "
            f"Supported: {sorted(_ACTIVATION_MAP.keys())}"
        )
    return fn


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
        encoder_noise_std: Stddev for additive Gaussian noise on the encoder's
            imitation_target input during stochastic training passes. Forces
            the encoder to be more robust, producing a more expressive latent
            space. Defaults to 0.0 (no noise).
        proprioception_noise_std: Stddev for Gaussian noise on decoder
            proprioception input during stochastic training passes.
        proprioception_noise_mode: How noise is applied. "multiplicative" scales
            each dimension by (1 + std * N(0,1)); "additive" adds std * N(0,1)
            uniformly in normalized space.
    """

    encoder_layers: Sequence[int]
    decoder_layers: Sequence[int]
    latents: int = 60
    encoder_noise_std: float = 0.0
    proprioception_noise_std: float = 0.0
    proprioception_noise_mode: str = "multiplicative"
    activation: networks.ActivationFn = nn.silu

    def setup(self):
        """Initialize encoder and decoder submodules."""
        self.encoder = Encoder(layer_sizes=self.encoder_layers, latents=self.latents, activation=self.activation)
        self.decoder = Decoder(layer_sizes=self.decoder_layers, activation=self.activation)

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
            # Single key - split for encoder, encoder noise, and proprio noise
            encoder_rng, enc_noise_rng, noise_rng = jax.random.split(key, 3)
        elif not obs_is_batched:
            # Per-sample keys but unbatched observation - use first key
            # This can happen when key batching was determined from nested obs structure
            # before normalization flattened it to unbatched
            encoder_rng, enc_noise_rng, noise_rng = jax.random.split(key[0], 3)
        else:
            # Per-sample keys [batch_size, 2] - vmap split over batch
            split_keys = jax.vmap(lambda k: jax.random.split(k, 3))(key)
            encoder_rng = split_keys[:, 0]
            enc_noise_rng = split_keys[:, 1]
            noise_rng = split_keys[:, 2]

        if not deterministic and self.encoder_noise_std > 0.0:
            if enc_noise_rng.ndim == 1:
                enc_noise = jax.random.normal(enc_noise_rng, traj.shape)
            elif not obs_is_batched:
                enc_noise = jax.random.normal(enc_noise_rng[0], traj.shape)
            else:
                enc_noise = jax.vmap(
                    lambda rng_key, obs_i: jax.random.normal(rng_key, obs_i.shape)
                )(enc_noise_rng, traj)
            traj = traj + self.encoder_noise_std * enc_noise

        if not deterministic and self.proprioception_noise_std > 0.0:
            if noise_rng.ndim == 1:
                noise = jax.random.normal(noise_rng, egocentric_obs.shape)
            elif not obs_is_batched:
                noise = jax.random.normal(noise_rng[0], egocentric_obs.shape)
            else:
                noise = jax.vmap(
                    lambda rng_key, obs_i: jax.random.normal(rng_key, obs_i.shape)
                )(noise_rng, egocentric_obs)
            if self.proprioception_noise_mode == "additive":
                egocentric_obs = egocentric_obs + self.proprioception_noise_std * noise
            else:
                egocentric_obs = egocentric_obs * (
                    1.0 + self.proprioception_noise_std * noise
                )

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
    encoder_noise_std: float = 0.0,
    proprioception_noise_std: float = 0.0,
    proprioception_noise_mode: str = "multiplicative",
    activation: networks.ActivationFn = nn.silu,
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
        encoder_noise_std: Stddev for additive Gaussian noise on the
            encoder's imitation_target input during stochastic passes.
        proprioception_noise_std: Stddev for Gaussian noise on decoder
            proprioception input during stochastic training passes.
        proprioception_noise_mode: "multiplicative" or "additive".

    Returns:
        FeedForwardNetwork with init and apply methods. The apply function
        returns (action_params, latent_mean, latent_logvar).
    """

    policy_module = IntentionNetwork(
        encoder_layers=list(encoder_hidden_layer_sizes),
        decoder_layers=list(decoder_hidden_layer_sizes)
        + [action_param_size],  # add action size to the last layer
        latents=latent_size,
        encoder_noise_std=encoder_noise_std,
        proprioception_noise_std=proprioception_noise_std,
        proprioception_noise_mode=proprioception_noise_mode,
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
        encoder_noise_std: Stddev for additive Gaussian noise on the encoder's
            imitation_target input (before vision concatenation) during
            stochastic training passes. Defaults to 0.0 (no noise).
        proprioception_noise_std: Stddev for Gaussian noise on decoder
            proprioception input during stochastic training passes.
        proprioception_noise_mode: How noise is applied. "multiplicative" scales
            each dimension by (1 + std * N(0,1)); "additive" adds std * N(0,1)
            uniformly in normalized space.
    """

    encoder_layers: Sequence[int]
    decoder_layers: Sequence[int]
    latents: int = 60
    vision_feature_size: int = 8
    vision_channels: Sequence[int] = (2, 4, 8, 16)
    encoder_noise_std: float = 0.0
    proprioception_noise_std: float = 0.0
    proprioception_noise_mode: str = "multiplicative"
    activation: networks.ActivationFn = nn.silu

    def setup(self):
        """Initialize vision encoder, encoder, and decoder submodules."""
        from track_mjx.agent.ff_ppo.vision_encoder import VisionEncoder

        self.vision_encoder = VisionEncoder(
            feature_size=self.vision_feature_size,
            channels=self.vision_channels,
        )
        self.encoder = Encoder(layer_sizes=self.encoder_layers, latents=self.latents, activation=self.activation)
        self.decoder = Decoder(layer_sizes=self.decoder_layers, activation=self.activation)

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

        # Key handling (same as IntentionNetwork, 3-way split)
        obs_is_batched = traj.ndim >= 2
        if key.ndim == 1:
            encoder_rng, enc_noise_rng, noise_rng = jax.random.split(key, 3)
        elif not obs_is_batched:
            encoder_rng, enc_noise_rng, noise_rng = jax.random.split(key[0], 3)
        else:
            split_keys = jax.vmap(lambda k: jax.random.split(k, 3))(key)
            encoder_rng = split_keys[:, 0]
            enc_noise_rng = split_keys[:, 1]
            noise_rng = split_keys[:, 2]

        # Apply encoder noise to traj BEFORE vision concatenation
        if not deterministic and self.encoder_noise_std > 0.0:
            if enc_noise_rng.ndim == 1:
                enc_noise = jax.random.normal(enc_noise_rng, traj.shape)
            elif not obs_is_batched:
                enc_noise = jax.random.normal(enc_noise_rng[0], traj.shape)
            else:
                enc_noise = jax.vmap(
                    lambda rng_key, obs_i: jax.random.normal(rng_key, obs_i.shape)
                )(enc_noise_rng, traj)
            traj = traj + self.encoder_noise_std * enc_noise

        # Concat trajectory + vision features for encoder
        encoder_input = jnp.concatenate([traj, vision_features], axis=-1)

        if not deterministic and self.proprioception_noise_std > 0.0:
            if noise_rng.ndim == 1:
                noise = jax.random.normal(noise_rng, egocentric_obs.shape)
            elif not obs_is_batched:
                noise = jax.random.normal(noise_rng[0], egocentric_obs.shape)
            else:
                noise = jax.vmap(
                    lambda rng_key, obs_i: jax.random.normal(rng_key, obs_i.shape)
                )(noise_rng, egocentric_obs)
            if self.proprioception_noise_mode == "additive":
                egocentric_obs = egocentric_obs + self.proprioception_noise_std * noise
            else:
                egocentric_obs = egocentric_obs * (
                    1.0 + self.proprioception_noise_std * noise
                )

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


class VisionOnlyNetwork(nn.Module):
    """Vision-based policy network without imitation target.

    The encoder is a CNN that maps raw pixels to a fixed-size latent feature
    vector. This latent is concatenated with proprioception and decoded into
    action distribution parameters.

    Observations dict must include:
    - "proprioception": Proprioceptive state observations (flat array)
    - "vision": Egocentric camera image (H, W, 3) normalized to [0, 1]

    Attributes:
        decoder_layers: Layer sizes for decoder (including action output).
        latent_size: Dimension of the CNN output / latent feature vector.
        vision_channels: Channel sizes for each conv layer in the vision encoder.
    """

    decoder_layers: Sequence[int]
    latent_size: int = 8
    vision_channels: Sequence[int] = (2, 4, 8, 16)
    activation: networks.ActivationFn = nn.silu

    def setup(self):
        """Initialize vision encoder and decoder submodules."""
        from track_mjx.agent.ff_ppo.vision_encoder import VisionEncoder

        self.vision_encoder = VisionEncoder(
            feature_size=self.latent_size,
            channels=self.vision_channels,
        )
        self.decoder = Decoder(layer_sizes=self.decoder_layers, activation=self.activation)

    def __call__(
        self,
        obs: Mapping[str, jnp.ndarray],
        key: jax.Array,
        deterministic: bool = False,
        get_activation: bool = False,
    ):
        egocentric_obs = obs["proprioception"]
        vision = obs["vision"]

        # CNN encode vision → latent features (deterministic encoding)
        z = self.vision_encoder(vision)
        latent_mean = z
        latent_logvar = jnp.full_like(z, -20.0)  # ~deterministic

        # Decode: [latent, proprioception] → action
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
                    "vision_features": z,
                    "egocentric_obs": egocentric_obs,
                    "intention": z,
                },
            )
        else:
            action, _ = self.decoder(concatenated)
            return action, latent_mean, latent_logvar


def make_vision_only_policy(
    action_param_size: int,
    latent_size: int,
    obs_sizes: Mapping[str, int],
    vision_shape: tuple[int, int, int],
    decoder_hidden_layer_sizes: Sequence[int] = (512, 512),
    vision_channels: Sequence[int] = (2, 4, 8, 16),
    activation: networks.ActivationFn = nn.silu,
) -> networks.FeedForwardNetwork:
    """Create a vision-only policy network (CNN encoder + MLP decoder).

    The CNN encodes raw pixels to a fixed-size latent, which is concatenated
    with proprioception and decoded into action parameters. No imitation
    target / VAE encoder is used.

    Args:
        action_param_size: Output dimension (typically 2x action_size).
        latent_size: Dimension of the CNN output feature vector.
        obs_sizes: Dict mapping observation keys to their sizes.
        vision_shape: Shape of the vision input (H, W, C).
        decoder_hidden_layer_sizes: Hidden layer sizes for decoder MLP.
        vision_channels: Channel sizes for each conv layer.

    Returns:
        FeedForwardNetwork with init and apply methods.
    """
    policy_module = VisionOnlyNetwork(
        decoder_layers=list(decoder_hidden_layer_sizes) + [action_param_size],
        latent_size=latent_size,
        vision_channels=vision_channels,
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
        """Apply policy with observation normalization."""
        obs = normalize_dict_obs(obs, processor_params)
        return policy_module.apply(
            policy_params,
            obs=obs,
            key=key,
            deterministic=deterministic,
            get_activation=get_activation,
        )

    dummy_obs = {
        "imitation_target": jnp.zeros((1, obs_sizes.get("imitation_target", 0))),
        "proprioception": jnp.zeros((1, obs_sizes["proprioception"])),
        "vision": jnp.zeros((1,) + vision_shape),
    }
    dummy_key = jax.random.PRNGKey(0)

    return networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_obs, dummy_key),
        apply=apply,
    )


def make_vision_intention_policy(
    action_param_size: int,
    latent_size: int,
    obs_sizes: Mapping[str, int],
    vision_shape: tuple[int, int, int],
    encoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    decoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    vision_feature_size: int = 8,
    vision_channels: Sequence[int] = (2, 4, 8, 16),
    encoder_noise_std: float = 0.0,
    proprioception_noise_std: float = 0.0,
    proprioception_noise_mode: str = "multiplicative",
    activation: networks.ActivationFn = nn.silu,
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
        encoder_noise_std: Stddev for additive Gaussian noise on the
            encoder's imitation_target input (before vision concatenation)
            during stochastic passes.
        proprioception_noise_std: Stddev for Gaussian noise on decoder
            proprioception input during stochastic training passes.
        proprioception_noise_mode: "multiplicative" or "additive".

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
        encoder_noise_std=encoder_noise_std,
        proprioception_noise_std=proprioception_noise_std,
        proprioception_noise_mode=proprioception_noise_mode,
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


class VisionTaskObsNetwork(nn.Module):
    """Vision + task observation network for transfer learning.

    Combines CNN-encoded vision features with task-relevant body signals
    (prev_action, kinematic sensors, touch, origin) through a fusion MLP
    to produce a deterministic latent intention.

    Observations dict must include:
    - "imitation_target": Task observations (flat array) — mapped from "task_obs"
      by observation_utils
    - "proprioception": Proprioceptive state (flat array, can be empty for transfer)
    - "vision": Egocentric camera image (H, W, C) normalized to [0, 1]

    Attributes:
        decoder_layers: Layer sizes for decoder (including action output).
        latent_size: Dimension of the fusion MLP output / latent vector.
        vision_feature_size: Output dimension of the vision encoder CNN.
        vision_channels: Channel sizes for each conv layer.
        fusion_layers: Hidden layer sizes for the fusion MLP.
    """

    decoder_layers: Sequence[int]
    latent_size: int = 16
    vision_feature_size: int = 8
    vision_channels: Sequence[int] = (2, 4, 8, 16)
    fusion_layers: Sequence[int] = (256,)
    activation: networks.ActivationFn = nn.silu

    def setup(self):
        """Initialize vision encoder, fusion MLP, and decoder submodules."""
        from track_mjx.agent.ff_ppo.vision_encoder import VisionEncoder

        self.vision_encoder = VisionEncoder(
            feature_size=self.vision_feature_size,
            channels=self.vision_channels,
        )
        # Fusion MLP: hidden layers with LayerNorm + output projection
        fusion_dense = []
        fusion_norms = []
        for h in self.fusion_layers:
            fusion_dense.append(nn.Dense(h))
            fusion_norms.append(nn.LayerNorm())
        self.fusion_dense = fusion_dense
        self.fusion_norms = fusion_norms
        self.fusion_out = nn.Dense(self.latent_size)
        self.decoder = Decoder(layer_sizes=self.decoder_layers, activation=self.activation)

    def __call__(
        self,
        obs: Mapping[str, jnp.ndarray],
        key: jax.Array,
        deterministic: bool = False,
        get_activation: bool = False,
    ):
        task_obs = obs["imitation_target"]
        egocentric_obs = obs["proprioception"]
        vision = obs["vision"]

        # CNN encode vision
        vision_features = self.vision_encoder(vision)

        # Fuse: [vision_features, task_obs] → MLP → latent
        combined = jnp.concatenate([vision_features, task_obs], axis=-1)
        z = combined
        for dense, norm in zip(self.fusion_dense, self.fusion_norms):
            z = self.activation(dense(z))
            z = norm(z)
        z = self.fusion_out(z)

        latent_mean = z
        latent_logvar = jnp.full_like(z, -20.0)  # ~deterministic

        # Decode: [latent, proprioception] → action params
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
                    "egocentric_obs": egocentric_obs,
                    "task_obs": task_obs,
                    "intention": z,
                },
            )
        else:
            action, _ = self.decoder(concatenated)
            return action, latent_mean, latent_logvar


def make_vision_task_obs_policy(
    action_param_size: int,
    latent_size: int,
    obs_sizes: Mapping[str, int],
    vision_shape: tuple[int, int, int],
    decoder_hidden_layer_sizes: Sequence[int] = (512, 512),
    vision_feature_size: int = 8,
    vision_channels: Sequence[int] = (2, 4, 8, 16),
    fusion_hidden_layer_sizes: Sequence[int] = (256,),
    activation: networks.ActivationFn = nn.silu,
) -> networks.FeedForwardNetwork:
    """Create a vision + task observation policy network.

    Constructs a deterministic fusion policy where the CNN encodes raw pixels,
    which are concatenated with task-relevant body signals and passed through
    a fusion MLP to produce a latent intention. The decoder generates action
    parameters conditioned on the latent intention and proprioceptive state.

    Args:
        action_param_size: Output dimension (typically 2x action_size for
            Gaussian mean and variance).
        latent_size: Dimension of the fusion MLP output / latent vector.
        obs_sizes: Dict mapping observation keys to their sizes, e.g.
            {"imitation_target": 100, "proprioception": 60}.
        vision_shape: Shape of the vision input (H, W, C), e.g. (64, 64, 3).
        decoder_hidden_layer_sizes: Hidden layer sizes for decoder MLP.
        vision_feature_size: Output dimension of the vision encoder CNN.
        vision_channels: Channel sizes for each conv layer in the vision encoder.
        fusion_hidden_layer_sizes: Hidden layer sizes for the fusion MLP.

    Returns:
        FeedForwardNetwork with init and apply methods. The apply function
        returns (action_params, latent_mean, latent_logvar).
    """

    policy_module = VisionTaskObsNetwork(
        decoder_layers=list(decoder_hidden_layer_sizes) + [action_param_size],
        latent_size=latent_size,
        vision_feature_size=vision_feature_size,
        vision_channels=vision_channels,
        fusion_layers=list(fusion_hidden_layer_sizes),
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
        "imitation_target": jnp.zeros((1, obs_sizes.get("imitation_target", 0))),
        "proprioception": jnp.zeros((1, obs_sizes.get("proprioception", 0))),
        "vision": jnp.zeros((1,) + vision_shape),
    }
    dummy_key = jax.random.PRNGKey(0)

    return networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_obs, dummy_key),
        apply=apply,
    )
