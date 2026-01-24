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

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
from brax.training import networks, types
from flax import linen as nn

from track_mjx.agent.observation_utils import (
    DictRunningStatisticsState,
    normalize_dict_obs,
)


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


# =============================================================================
# TEMPORAL DOWNSAMPLING ENCODER COMPONENTS
# =============================================================================


class TemporalConvBlock(nn.Module):
    """Single temporal convolution block with downsampling.

    Applies Conv1D with optional striding for temporal downsampling,
    followed by SiLU activation and LayerNorm.

    Attributes:
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Stride for temporal downsampling (default: 2).
        activation: Activation function (default: SiLU).
    """

    out_channels: int
    kernel_size: int = 3
    stride: int = 2
    activation: networks.ActivationFn = nn.silu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply temporal convolution with downsampling.

        Args:
            x: Input tensor, shape [B, T, C].

        Returns:
            Output tensor, shape [B, T//stride, out_channels].
        """
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(self.kernel_size,),
            strides=(self.stride,),
            padding="SAME",
        )(x)
        x = self.activation(x)
        x = nn.LayerNorm()(x)
        return x


class VQTemporalEncoder(nn.Module):
    """VQ-VAE encoder with temporal downsampling using Conv1D.

    Downsamples the temporal dimension so that each code represents a chunk
    of frames (e.g., 4 frames per code), forcing codes to learn semantic
    behavioral primitives rather than instantaneous pose phases.

    Attributes:
        hidden_channels: Sequence of hidden channel dimensions for conv layers.
        latent_dim: Dimension of the output embedding (must match codebook).
        temporal_stride: Total temporal downsampling factor (default: 4).
        kernel_size: Convolution kernel size (default: 3).
        activation: Activation function (default: SiLU).
    """

    hidden_channels: Sequence[int] = (256, 256)
    latent_dim: int = 16
    temporal_stride: int = 4
    kernel_size: int = 3
    activation: networks.ActivationFn = nn.silu

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, get_activation: bool = False
    ) -> jnp.ndarray | tuple[jnp.ndarray, dict]:
        """Encode observations with temporal downsampling.

        Handles multiple input shapes:
        - [T, B, D]: Batched training input with time dimension
        - [B, D]: Batched single-step inference (no time dimension)
        - [D]: Single observation (no batch, no time - used in logging rollouts)

        For single-step inference, we add a temporal dimension of 1 and run
        through the same Conv1D path. With padding='SAME', Conv1D preserves
        T=1 even with stride>1.

        Args:
            x: Input observations, shape [T, B, D], [B, D], or [D].
            get_activation: If True, return intermediate activations.

        Returns:
            z_e: Continuous embedding with matching batch structure.
            If get_activation=True, also returns dict of activations.
        """
        activations = {}

        # Track original shape to restore at the end
        original_ndim = x.ndim

        # Handle different input shapes
        if x.ndim == 1:
            # Single observation [D] -> [1, 1, D]
            x = x[None, None, :]
        elif x.ndim == 2:
            # Single-step batched [B, D] -> [1, B, D]
            x = x[None, :, :]

        # Now x has shape [T, B, D]
        # Swap axes: [T, B, D] -> [B, T, D] for Conv1D
        x = jnp.swapaxes(x, 0, 1)  # [B, T, D]

        if get_activation:
            activations["input"] = x

        # Calculate per-layer stride to achieve total temporal_stride
        # For temporal_stride=4 with 2 conv layers, each layer uses stride=2
        num_conv_layers = len(self.hidden_channels)
        per_layer_stride = int(round(self.temporal_stride ** (1.0 / num_conv_layers)))

        # Apply temporal convolution layers
        # Note: With padding='SAME', T=1 is preserved even with stride>1
        # since ceil(1/stride) = 1 for any stride
        for i, channels in enumerate(self.hidden_channels):
            x = TemporalConvBlock(
                out_channels=channels,
                kernel_size=self.kernel_size,
                stride=per_layer_stride,
                activation=self.activation,
            )(x)
            if get_activation:
                activations[f"conv_{i}"] = x

        # Project to latent dimension
        z_e = nn.Dense(self.latent_dim, name="latent_projection")(x)

        # Swap back: [B, T', D] -> [T', B, D]
        z_e = jnp.swapaxes(z_e, 0, 1)

        # Restore original shape structure
        if original_ndim == 1:
            # [1, 1, D] -> [D]
            z_e = z_e[0, 0]
        elif original_ndim == 2:
            # [1, B, D] -> [B, D]
            z_e = z_e[0]

        if get_activation:
            activations["z_e"] = z_e
            return z_e, activations
        return z_e


def upsample_temporal(z_q: jnp.ndarray, target_length: int) -> jnp.ndarray:
    """Upsample z_q by repeating codes to match original temporal resolution.

    Uses simple repeat-based upsampling (not learned) for simplicity and
    interpretability. Each downsampled code is repeated to fill its
    corresponding chunk of the original sequence.

    Args:
        z_q: Quantized latents, shape [T_down, B, D].
        target_length: Target temporal length (original T before downsampling).

    Returns:
        Upsampled latents, shape [target_length, B, D].
    """
    t_down = z_q.shape[0]

    if t_down == target_length:
        return z_q

    if t_down == 0:
        raise ValueError("Cannot upsample empty sequence (t_down=0)")

    stride = target_length // t_down

    # [T_down, B, D] -> [T_down, 1, B, D] -> [T_down, stride, B, D]
    z_q_expanded = jnp.repeat(z_q[:, None, :, :], stride, axis=1)

    # Reshape to [T_down * stride, B, D]
    z_q_upsampled = z_q_expanded.reshape(t_down * stride, *z_q.shape[1:])

    # Handle non-divisible lengths by padding/truncating
    current_length = z_q_upsampled.shape[0]
    if current_length < target_length:
        # Pad by repeating the last code
        padding = target_length - current_length
        last_code = z_q_upsampled[-1:]
        z_q_upsampled = jnp.concatenate(
            [z_q_upsampled, jnp.repeat(last_code, padding, axis=0)], axis=0
        )
    elif current_length > target_length:
        # Truncate
        z_q_upsampled = z_q_upsampled[:target_length]

    return z_q_upsampled


class VQIntentionNetwork(nn.Module):
    """Full VQ-VAE model combining encoder, quantizer, and decoder.

    The network splits observations into reference trajectory and proprioceptive
    components. The encoder processes trajectory observations to produce continuous
    embeddings, which are quantized via the codebook, then concatenated with
    proprioceptive state and decoded into action distribution parameters.

    Attributes:
        encoder_layers: Hidden layer sizes for the encoder MLP.
        decoder_layers: Layer sizes for decoder (including action output).
        latent_dim: Dimension of the latent/codebook embedding space.
        num_codes: Number of codebook entries.
        commitment_cost: Weight for commitment loss (beta).
        codebook_init_scale: Scale for codebook initialization.
    """

    encoder_layers: Sequence[int]
    decoder_layers: Sequence[int]
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
        obs: Mapping[str, jnp.ndarray],
        key: jax.Array,
        deterministic: bool = False,
        get_activation: bool = False,
    ):
        """Forward pass through VQ-VAE intention network.

        Note: The `key` and `deterministic` arguments are kept for API
        compatibility with VAE IntentionNetwork but are not used since
        VQ quantization is deterministic (nearest neighbor lookup).

        Args:
            obs: Dictionary observation with keys:
                - "imitation_target": Reference trajectory observations.
                - "proprioception": Proprioceptive state observations.
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
        # Access observations by key
        traj = obs["imitation_target"]
        egocentric_obs = obs["proprioception"]

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


class VQTemporalIntentionNetwork(nn.Module):
    """VQ-VAE with temporal downsampling for semantic codes.

    Uses temporal convolutions to downsample the input sequence before
    quantization, forcing each code to represent a chunk of frames.
    The quantized codes are then upsampled back to the original temporal
    resolution before being passed to the decoder.

    Attributes:
        encoder_hidden_channels: Channel sizes for temporal conv layers.
        decoder_layers: Layer sizes for decoder (including action output).
        latent_dim: Dimension of the latent/codebook embedding space.
        temporal_stride: Temporal downsampling factor (e.g., 4 means T -> T//4).
        encoder_kernel_size: Kernel size for temporal convolutions.
        num_codes: Number of codebook entries.
        commitment_cost: Weight for commitment loss (beta).
        codebook_init_scale: Scale for codebook initialization.
    """

    encoder_hidden_channels: Sequence[int]
    decoder_layers: Sequence[int]
    latent_dim: int = 16
    temporal_stride: int = 4
    encoder_kernel_size: int = 3
    num_codes: int = 512
    commitment_cost: float = 0.25
    codebook_init_scale: float = 1.0

    def setup(self):
        """Initialize temporal encoder, quantizer, and decoder submodules."""
        self.encoder = VQTemporalEncoder(
            hidden_channels=self.encoder_hidden_channels,
            latent_dim=self.latent_dim,
            temporal_stride=self.temporal_stride,
            kernel_size=self.encoder_kernel_size,
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
        obs: Mapping[str, jnp.ndarray],
        key: jax.Array,
        deterministic: bool = False,
        get_activation: bool = False,
    ):
        """Forward pass through temporal VQ-VAE intention network.

        Handles multiple input shapes:
        - [T, B, D]: Batched training input with time dimension
        - [B, D]: Batched single-step inference (no time dimension)
        - [D]: Single observation (no batch, no time - used in logging rollouts)

        Args:
            obs: Dictionary observation with keys:
                - "imitation_target": Reference trajectory observations.
                - "proprioception": Proprioceptive state observations.
            key: JAX random key (unused, for API compatibility).
            deterministic: Unused, VQ is always deterministic.
            get_activation: If True, return intermediate activations.

        Returns:
            action_params: Action distribution parameters with matching shape.
            z_e: Continuous encoder output.
            indices: Codebook indices.
            (optional) extras: Dict of activations if get_activation=True.
        """
        traj = obs["imitation_target"]
        egocentric_obs = obs["proprioception"]

        # Detect single-step inference (1D or 2D input without time dimension)
        is_single_step = traj.ndim <= 2

        if is_single_step:
            # Single-step inference: no temporal operations needed
            if get_activation:
                z_e, encoder_activations = self.encoder(traj, get_activation=True)
                z_q_st, indices, z_q = self.quantizer(z_e)
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
                        "is_single_step": True,
                    },
                )
            else:
                z_e = self.encoder(traj)  # [B, latent_dim]
                z_q_st, indices, _ = self.quantizer(z_e)
                concatenated = jnp.concatenate([z_q_st, egocentric_obs], axis=-1)
                action, _ = self.decoder(concatenated)
                return action, z_e, indices

        # Batched training: apply temporal downsampling and upsampling
        # Store original temporal length for upsampling
        original_t = traj.shape[0]

        if get_activation:
            # Get encoder activations
            z_e, encoder_activations = self.encoder(traj, get_activation=True)

            # Quantize (z_e is now [T//stride, B, latent_dim])
            z_q_st, indices, z_q = self.quantizer(z_e)

            # Upsample quantized codes to match original temporal resolution
            z_q_upsampled = upsample_temporal(z_q_st, original_t)

            # Decode: concatenate upsampled z_q with proprioception
            concatenated = jnp.concatenate([z_q_upsampled, egocentric_obs], axis=-1)
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
                    "z_q_upsampled": z_q_upsampled,
                    "indices": indices,
                    "temporal_stride": self.temporal_stride,
                },
            )
        else:
            # Standard forward pass
            z_e = self.encoder(traj)  # [T//stride, B, latent_dim]
            z_q_st, indices, _ = self.quantizer(z_e)

            # Upsample for decoder
            z_q_upsampled = upsample_temporal(z_q_st, original_t)

            # Decode
            concatenated = jnp.concatenate([z_q_upsampled, egocentric_obs], axis=-1)
            action, _ = self.decoder(concatenated)

            return action, z_e, indices


def make_vq_intention_policy(
    action_param_size: int,
    latent_dim: int,
    obs_sizes: Mapping[str, int],
    encoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    decoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    num_codes: int = 512,
    commitment_cost: float = 0.25,
    codebook_init_scale: float = 1.0,
    encoder_type: str = "mlp",
    temporal_stride: int = 1,
    encoder_hidden_channels: Sequence[int] | None = None,
    encoder_kernel_size: int = 3,
) -> networks.FeedForwardNetwork:
    """Create a VQ-VAE intention-based policy network.

    Constructs an encoder-quantizer-decoder VQ-VAE policy where the encoder
    processes reference trajectory observations, the quantizer maps to discrete
    codebook entries, and the decoder generates action parameters conditioned
    on quantized intentions and proprioceptive state.

    Supports two encoder types:
    - "mlp": Standard MLP encoder (original behavior, no temporal downsampling)
    - "temporal_conv": Conv1D encoder with temporal downsampling

    Args:
        action_param_size: Output dimension (typically 2x action_size for
            Gaussian mean and variance).
        latent_dim: Dimension of the latent/codebook embedding space.
        obs_sizes: Dict mapping observation keys to their sizes, e.g.
            {"imitation_target": 3716, "proprioception": 226}.
        encoder_hidden_layer_sizes: Hidden layer sizes for MLP encoder.
        decoder_hidden_layer_sizes: Hidden layer sizes for decoder MLP.
        num_codes: Number of codebook entries.
        commitment_cost: Weight for commitment loss (beta).
        codebook_init_scale: Scale for codebook initialization.
        encoder_type: Type of encoder ("mlp" or "temporal_conv").
        temporal_stride: Temporal downsampling factor for temporal_conv encoder.
        encoder_hidden_channels: Channel sizes for temporal conv encoder.
            Defaults to (256, 256) if not provided.
        encoder_kernel_size: Kernel size for temporal convolutions.

    Returns:
        FeedForwardNetwork with init and apply methods. The apply function
        returns (action_params, z_e, indices) or with activations.
    """
    if encoder_hidden_channels is None:
        encoder_hidden_channels = (256, 256)

    if encoder_type == "temporal_conv":
        policy_module = VQTemporalIntentionNetwork(
            encoder_hidden_channels=list(encoder_hidden_channels),
            decoder_layers=list(decoder_hidden_layer_sizes) + [action_param_size],
            latent_dim=latent_dim,
            temporal_stride=temporal_stride,
            encoder_kernel_size=encoder_kernel_size,
            num_codes=num_codes,
            commitment_cost=commitment_cost,
            codebook_init_scale=codebook_init_scale,
        )
        # For temporal encoder, we need a temporal dimension in dummy input
        # Use temporal_stride as minimum to ensure at least 1 downsampled timestep
        dummy_t = max(temporal_stride, 2)
        dummy_obs = {
            "imitation_target": jnp.zeros(
                (dummy_t, 1, obs_sizes["imitation_target"])
            ),
            "proprioception": jnp.zeros((dummy_t, 1, obs_sizes["proprioception"])),
        }
    else:
        # Default MLP encoder
        policy_module = VQIntentionNetwork(
            encoder_layers=list(encoder_hidden_layer_sizes),
            decoder_layers=list(decoder_hidden_layer_sizes) + [action_param_size],
            latent_dim=latent_dim,
            num_codes=num_codes,
            commitment_cost=commitment_cost,
            codebook_init_scale=codebook_init_scale,
        )
        dummy_obs = {
            "imitation_target": jnp.zeros((1, obs_sizes["imitation_target"])),
            "proprioception": jnp.zeros((1, obs_sizes["proprioception"])),
        }

    def apply(
        processor_params: DictRunningStatisticsState,
        policy_params,
        obs: Mapping[str, jnp.ndarray],
        key,
        deterministic: bool = False,
        get_activation: bool = False,
    ):
        """Apply VQ policy with observation normalization.

        Args:
            processor_params: Dict normalizer with per-key statistics.
            policy_params: Network weights.
            obs: Dict observation with "imitation_target" and "proprioception".
            key: JAX random key (unused, for API compatibility).
            deterministic: Unused, VQ is always deterministic.
            get_activation: If True, return intermediate activations.

        Returns:
            action_params: Action distribution parameters.
            z_e: Continuous encoder output (for loss computation).
            indices: Codebook indices (for logging).
            (optional) extras: Dict of activations if get_activation=True.
        """
        obs = normalize_dict_obs(obs, processor_params)
        return policy_module.apply(
            policy_params,
            obs=obs,
            key=key,
            deterministic=deterministic,
            get_activation=get_activation,
        )

    dummy_key = jax.random.PRNGKey(0)

    return networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_obs, dummy_key),
        apply=apply,
    )
