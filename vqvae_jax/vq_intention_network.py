"""VQ-VAE intention network architectures for discrete latent imitation learning.

This module provides VQ-VAE (Vector Quantized Variational Autoencoder) based
encoder-decoder architectures that replace the continuous Gaussian latent space
of the standard VAE with a discrete codebook of learned embeddings.

Key components:
- VQEncoder: Maps reference trajectory observations to a continuous embedding
- VectorQuantizer: Quantizes embeddings to nearest codebook entries
- Decoder: Maps quantized latents + proprioceptive state to action parameters
- VQIntentionNetwork: Full VQ-VAE combining encoder, quantizer, and decoder

The discrete latent space enables learning interpretable "motor primitives" that
can be analyzed and potentially reused across tasks.

Reference: van den Oord et al., "Neural Discrete Representation Learning", 2017
https://arxiv.org/abs/1711.00937
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from brax.training import networks, types
from flax import linen as nn


class VQEncoder(nn.Module):
    """VQ-VAE encoder that maps observations to continuous embeddings.

    Unlike the VAE encoder which outputs (mean, logvar) for sampling,
    the VQ encoder outputs a single continuous embedding z_e that will
    be quantized to the nearest codebook entry.

    Attributes:
        layer_sizes: Hidden layer dimensions for the MLP.
        latent_dim: Dimension of the output embedding (must match codebook).
        activation: Activation function (default: SiLU).
        kernel_init: Weight initializer (default: LeCun uniform).
        bias: Whether to use bias terms in Dense layers.
    """

    layer_sizes: Sequence[int]
    latent_dim: int
    activation: networks.ActivationFn = nn.silu
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()
    bias: bool = True

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, get_activation: bool = False
    ) -> jnp.ndarray | tuple[jnp.ndarray, dict]:
        """Encode observations to continuous embedding.

        Args:
            x: Input observations, shape [..., input_dim].
            get_activation: If True, return intermediate activations.

        Returns:
            z_e: Continuous embedding, shape [..., latent_dim].
            If get_activation=True, also returns dict of activations.
        """
        activations = {}

        # Process through hidden layers with LayerNorm
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

        # Project to latent dimension (no activation on final layer)
        z_e = nn.Dense(self.latent_dim, name="latent_projection")(x)

        if get_activation:
            activations["z_e"] = z_e
            return z_e, activations
        return z_e


class VectorQuantizer(nn.Module):
    """Vector quantization layer with learnable codebook.

    Maps continuous encoder outputs to discrete codebook entries using
    nearest-neighbor lookup. Uses the straight-through estimator to
    enable gradient flow through the non-differentiable quantization.

    The codebook is updated via standard gradient descent (not EMA),
    which simplifies integration with the existing training loop.

    Attributes:
        num_codes: Number of codebook entries (vocabulary size).
        latent_dim: Dimension of each codebook entry.
        commitment_cost: Weight for commitment loss (beta in paper).
        codebook_init_scale: Scale for codebook initialization.
    """

    num_codes: int = 512
    latent_dim: int = 60
    commitment_cost: float = 0.25
    codebook_init_scale: float = 1.0

    def setup(self):
        """Initialize the codebook as a learnable parameter."""
        self.codebook = self.param(
            "embeddings",
            nn.initializers.uniform(scale=self.codebook_init_scale),
            (self.num_codes, self.latent_dim),
        )

    def __call__(
        self, z_e: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Quantize encoder output to nearest codebook entry.

        Args:
            z_e: Continuous encoder output, shape [..., latent_dim].

        Returns:
            z_q_st: Quantized output with straight-through gradient,
                shape [..., latent_dim]. Use this for decoder.
            indices: Codebook indices, shape [...]. Use for analysis.
            z_q: Quantized output (no gradient), shape [..., latent_dim].
                Use this for loss computation with proper stop_gradient.
        """
        # Store original shape for reshape at end
        input_shape = z_e.shape
        flat_z_e = z_e.reshape(-1, self.latent_dim)  # [N, D]

        # Compute squared distances using expansion:
        # ||z_e - e_k||^2 = ||z_e||^2 + ||e_k||^2 - 2 * z_e @ e_k.T
        z_e_sq = jnp.sum(flat_z_e**2, axis=-1, keepdims=True)  # [N, 1]
        codebook_sq = jnp.sum(self.codebook**2, axis=-1)  # [K]
        cross = jnp.matmul(flat_z_e, self.codebook.T)  # [N, K]

        distances = z_e_sq + codebook_sq - 2 * cross  # [N, K]

        # Find nearest codebook entry (non-differentiable)
        flat_indices = jnp.argmin(distances, axis=-1)  # [N]

        # Look up quantized vectors
        flat_z_q = self.codebook[flat_indices]  # [N, D]

        # Reshape back to original batch shape
        indices = flat_indices.reshape(input_shape[:-1])  # [...]
        z_q = flat_z_q.reshape(input_shape)  # [..., D]

        # Straight-through estimator using Sterbenz pattern
        # Forward: z_q_st = z_q
        # Backward: d(z_q_st)/d(z_e) = 1
        z_q_st = z_e - jax.lax.stop_gradient(z_e) + jax.lax.stop_gradient(z_q)

        return z_q_st, indices, z_q


class Decoder(nn.Module):
    """Decoder that maps quantized latents to action distribution parameters.

    Identical to the VAE decoder - processes concatenated quantized latent
    and proprioceptive observations through an MLP.

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
        """Decode latent + proprioceptive to action parameters.

        Args:
            x: Concatenated [z_q, proprio_obs], shape [..., latent_dim + proprio_dim].
            get_activation: If True, return intermediate activations.

        Returns:
            action_params: Action distribution parameters, shape [..., action_param_size].
            activations: Dict of intermediate activations (empty if get_activation=False).
        """
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


class VQIntentionNetwork(nn.Module):
    """Full VQ-VAE model combining encoder, quantizer, and decoder.

    The network splits observations into reference trajectory and proprioceptive
    components. The encoder processes trajectory observations to produce continuous
    embeddings, which are quantized via the codebook, then concatenated with
    proprioceptive state and decoded into action distribution parameters.

    Attributes:
        encoder_layers: Hidden layer sizes for the encoder MLP.
        decoder_layers: Layer sizes for decoder (including action output).
        reference_obs_size: Dimension of reference trajectory observations.
        latent_dim: Dimension of the latent/codebook embedding space.
        num_codes: Number of codebook entries.
        commitment_cost: Weight for commitment loss (beta).
        codebook_init_scale: Scale for codebook initialization.
    """

    encoder_layers: Sequence[int]
    decoder_layers: Sequence[int]
    reference_obs_size: int
    latent_dim: int = 60
    num_codes: int = 512
    commitment_cost: float = 0.25
    codebook_init_scale: float = 1.0

    def setup(self):
        """Initialize encoder, quantizer, and decoder submodules."""
        self.encoder = VQEncoder(
            layer_sizes=self.encoder_layers,
            latent_dim=self.latent_dim,
        )
        self.quantizer = VectorQuantizer(
            num_codes=self.num_codes,
            latent_dim=self.latent_dim,
            commitment_cost=self.commitment_cost,
            codebook_init_scale=self.codebook_init_scale,
        )
        self.decoder = Decoder(layer_sizes=self.decoder_layers)

    def __call__(
        self,
        obs: jnp.ndarray,
        key: jax.Array,
        deterministic: bool = False,
        get_activation: bool = False,
    ):
        """Forward pass through VQ-VAE intention network.

        Note: The `key` and `deterministic` arguments are kept for API
        compatibility with VAE IntentionNetwork but are not used since
        VQ quantization is deterministic (nearest neighbor lookup).

        Args:
            obs: Full observation, shape [..., obs_dim].
            key: JAX random key (unused, for API compatibility).
            deterministic: Unused, VQ is always deterministic.
            get_activation: If True, return intermediate activations.

        Returns:
            action_params: Action distribution parameters, shape [..., action_size*2].
            z_e: Continuous encoder output, shape [..., latent_dim].
                Used for commitment loss computation.
            indices: Codebook indices, shape [...].
                Used for logging and analysis.
            (optional) extras: Dict of activations if get_activation=True.
        """
        # Split observations
        traj = obs[..., : self.reference_obs_size]
        egocentric_obs = obs[..., self.reference_obs_size :]

        if get_activation:
            # Get encoder activations
            z_e, encoder_activations = self.encoder(traj, get_activation=True)

            # Quantize
            z_q_st, indices, z_q = self.quantizer(z_e)

            # Decode
            concatenated = jnp.concatenate([z_q_st, egocentric_obs], axis=-1)
            action, decoder_activations = self.decoder(concatenated, get_activation=True)

            return (
                action,
                z_e,
                indices,
                {
                    "encoder": encoder_activations,
                    "decoder": decoder_activations,
                    "egocentric_obs": egocentric_obs,
                    "traj_obs": traj,
                    "z_e": z_e,
                    "z_q": z_q,
                    "z_q_st": z_q_st,
                    "indices": indices,
                },
            )
        else:
            # Standard forward pass
            z_e = self.encoder(traj)
            z_q_st, indices, _ = self.quantizer(z_e)
            concatenated = jnp.concatenate([z_q_st, egocentric_obs], axis=-1)
            action, _ = self.decoder(concatenated)

            return action, z_e, indices


def make_vq_intention_policy(
    action_param_size: int,
    latent_dim: int,
    total_obs_size: int,
    reference_obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    encoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    decoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    num_codes: int = 512,
    commitment_cost: float = 0.25,
    codebook_init_scale: float = 1.0,
) -> networks.FeedForwardNetwork:
    """Create a VQ-VAE intention-based policy network.

    Constructs an encoder-quantizer-decoder VQ-VAE policy where the encoder
    processes reference trajectory observations, the quantizer maps to discrete
    codebook entries, and the decoder generates action parameters conditioned
    on quantized intentions and proprioceptive state.

    Args:
        action_param_size: Output dimension (typically 2x action_size for
            Gaussian mean and variance).
        latent_dim: Dimension of the latent/codebook embedding space.
        total_obs_size: Total observation dimension.
        reference_obs_size: Dimension of reference trajectory portion of obs.
        preprocess_observations_fn: Observation normalization function.
        encoder_hidden_layer_sizes: Hidden layer sizes for encoder MLP.
        decoder_hidden_layer_sizes: Hidden layer sizes for decoder MLP.
        num_codes: Number of codebook entries.
        commitment_cost: Weight for commitment loss (beta).
        codebook_init_scale: Scale for codebook initialization.

    Returns:
        FeedForwardNetwork with init and apply methods. The apply function
        returns (action_params, z_e, indices) or with activations.
    """
    policy_module = VQIntentionNetwork(
        encoder_layers=list(encoder_hidden_layer_sizes),
        decoder_layers=list(decoder_hidden_layer_sizes) + [action_param_size],
        reference_obs_size=reference_obs_size,
        latent_dim=latent_dim,
        num_codes=num_codes,
        commitment_cost=commitment_cost,
        codebook_init_scale=codebook_init_scale,
    )

    def apply(
        processor_params,
        policy_params,
        obs,
        key,
        deterministic: bool = False,
        get_activation: bool = False,
    ):
        """Apply VQ policy with observation normalization.

        Args:
            processor_params: Normalization statistics.
            policy_params: Network weights.
            obs: Observations, shape [..., total_obs_size].
            key: JAX random key (unused, for API compatibility).
            deterministic: Unused, VQ is always deterministic.
            get_activation: If True, return intermediate activations.

        Returns:
            action_params: Action distribution parameters.
            z_e: Continuous encoder output (for loss computation).
            indices: Codebook indices (for logging).
            (optional) extras: Dict of activations if get_activation=True.
        """
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(
            policy_params,
            obs=obs,
            key=key,
            deterministic=deterministic,
            get_activation=get_activation,
        )

    dummy_total_obs = jnp.zeros((1, total_obs_size))
    dummy_key = jax.random.PRNGKey(0)

    return networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_total_obs, dummy_key),
        apply=apply,
    )
