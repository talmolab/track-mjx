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

    Optionally supports a stickiness bias that favors selecting the
    previous code, creating hysteresis in code transitions. This is
    controlled by passing prev_indices and setting stickiness_bias > 0.

    Attributes:
        num_codes: Number of codebook entries (vocabulary size).
        latent_dim: Dimension of each codebook entry.
        commitment_cost: Weight for commitment loss (beta in paper).
        codebook_init_scale: Scale for codebook initialization.
        stickiness_bias: Bias subtracted from distance to previous code.
            When > 0, makes the previous code appear closer, creating
            temporal persistence. Default 0.0 (no bias).
    """

    num_codes: int = 512
    latent_dim: int = 60
    commitment_cost: float = 0.25
    codebook_init_scale: float = 1.0
    stickiness_bias: float = 0.0

    def setup(self):
        """Initialize the codebook as a learnable parameter."""
        self.codebook = self.param(
            "embeddings",
            nn.initializers.uniform(scale=self.codebook_init_scale),
            (self.num_codes, self.latent_dim),
        )

    def __call__(
        self,
        z_e: jnp.ndarray,
        prev_indices: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Quantize encoder output to nearest codebook entry.

        Args:
            z_e: Continuous encoder output, shape [..., latent_dim].
            prev_indices: Optional indices from previous timestep, shape [...].
                When provided with stickiness_bias > 0, the distance to the
                previous code is reduced by the bias, making it "sticky".

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

        # Apply stickiness bias if previous indices provided
        # This makes the previous code appear closer, creating hysteresis
        if prev_indices is not None and self.stickiness_bias > 0:
            flat_prev_indices = prev_indices.reshape(-1)  # [N]
            prev_one_hot = jax.nn.one_hot(
                flat_prev_indices, self.num_codes
            )  # [N, K]
            # Subtract bias from distance to previous code
            # Lower distance = more likely to be selected
            distances = distances - self.stickiness_bias * prev_one_hot

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

    Supports optional stickiness bias for temporal code persistence. When
    stickiness_bias > 0, the quantizer favors selecting the previous timestep's
    code, creating hysteresis that reduces rapid code switching.

    Attributes:
        encoder_layers: Hidden layer sizes for the encoder MLP.
        decoder_layers: Layer sizes for decoder (including action output).
        latent_dim: Dimension of the latent/codebook embedding space.
        num_codes: Number of codebook entries.
        commitment_cost: Weight for commitment loss (beta).
        codebook_init_scale: Scale for codebook initialization.
        stickiness_bias: Bias for temporal code persistence (default 0.0).
    """

    encoder_layers: Sequence[int]
    decoder_layers: Sequence[int]
    latent_dim: int = 60
    num_codes: int = 512
    commitment_cost: float = 0.25
    codebook_init_scale: float = 1.0
    stickiness_bias: float = 0.0

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
            stickiness_bias=self.stickiness_bias,
        )
        self.decoder = Decoder(layer_sizes=self.decoder_layers)

    def __call__(
        self,
        obs: Mapping[str, jnp.ndarray],
        key: jax.Array,
        deterministic: bool = False,
        get_activation: bool = False,
        prev_indices: jnp.ndarray | None = None,
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
            prev_indices: Optional indices from previous timestep for
                stickiness bias. Shape should match batch dimensions.

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

            # Quantize (with optional stickiness bias via prev_indices)
            z_q_st, indices, z_q = self.quantizer(z_e, prev_indices=prev_indices)

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
            z_q_st, indices, _ = self.quantizer(z_e, prev_indices=prev_indices)
            concatenated = jnp.concatenate([z_q_st, egocentric_obs], axis=-1)
            action, _ = self.decoder(concatenated)

            return action, z_e, indices

    def forward_temporal(
        self,
        obs: Mapping[str, jnp.ndarray],
        episode_mask: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward pass with temporal stickiness bias using sequential processing.

        Processes a sequence of observations where each timestep's code selection
        is biased toward the previous timestep's code. Uses jax.lax.scan for
        efficient sequential processing.

        The encoder runs in parallel over all timesteps, but quantization runs
        sequentially to enable the stickiness bias mechanism.

        Args:
            obs: Dictionary observation with keys:
                - "imitation_target": Shape [T, B, traj_dim].
                - "proprioception": Shape [T, B, proprio_dim].
            episode_mask: Optional mask indicating episode boundaries.
                Shape [T, B]. Value of 0 indicates episode start (reset prev_indices).
                If None, assumes continuous episode (no resets).

        Returns:
            action_params: Action distribution parameters, shape [T, B, action_size*2].
            z_e: Continuous encoder output, shape [T, B, latent_dim].
            indices: Codebook indices, shape [T, B].
        """
        traj = obs["imitation_target"]  # [T, B, traj_dim]
        egocentric_obs = obs["proprioception"]  # [T, B, proprio_dim]

        # Encode all timesteps in parallel (encoder has no temporal dependency)
        z_e = self.encoder(traj)  # [T, B, latent_dim]

        # If no bias, process in parallel (faster)
        if self.stickiness_bias <= 0:
            z_q_st, indices, _ = self.quantizer(z_e, prev_indices=None)
            concatenated = jnp.concatenate([z_q_st, egocentric_obs], axis=-1)
            action, _ = self.decoder(concatenated)
            return action, z_e, indices

        # Sequential quantization with stickiness bias using scan
        T, B = z_e.shape[0], z_e.shape[1]

        # Define scan step function
        def quantize_step(prev_indices, inputs):
            """Process one timestep with bias toward previous code."""
            z_e_t, proprio_t, mask_t = inputs

            # If mask_t is 0 (episode boundary), don't use prev_indices
            # Use where to handle episode boundaries: reset to no-bias quantization
            if episode_mask is not None:
                # Expand mask for broadcasting: [B] -> [B, 1] for proper shapes
                use_prev = mask_t > 0
                # When mask=0 (episode start), pass None-equivalent behavior
                # We achieve this by passing indices that won't affect selection
                # when bias is applied (use dummy indices and zero out the bias effect)
                effective_prev = jnp.where(
                    use_prev[:, None],
                    jax.nn.one_hot(prev_indices, self.num_codes),
                    jnp.zeros((B, self.num_codes)),
                )
                # Manually compute distances with conditional bias
                flat_z_e = z_e_t.reshape(-1, self.latent_dim)
                codebook = self.quantizer.codebook
                z_e_sq = jnp.sum(flat_z_e**2, axis=-1, keepdims=True)
                codebook_sq = jnp.sum(codebook**2, axis=-1)
                cross = jnp.matmul(flat_z_e, codebook.T)
                distances = z_e_sq + codebook_sq - 2 * cross
                # Apply bias only where mask is valid
                distances = distances - self.stickiness_bias * effective_prev
                curr_indices = jnp.argmin(distances, axis=-1)
                z_q_t = codebook[curr_indices]
                z_q_st_t = z_e_t - jax.lax.stop_gradient(z_e_t) + jax.lax.stop_gradient(z_q_t)
            else:
                # No episode mask - always use prev_indices
                z_q_st_t, curr_indices, _ = self.quantizer(z_e_t, prev_indices=prev_indices)

            return curr_indices, (z_q_st_t, curr_indices)

        # Prepare inputs for scan: [T, B, ...] -> scan over T
        if episode_mask is not None:
            scan_inputs = (z_e, egocentric_obs, episode_mask)
        else:
            # Create dummy mask of ones (all valid)
            dummy_mask = jnp.ones((T, B))
            scan_inputs = (z_e, egocentric_obs, dummy_mask)

        # Initial indices for first timestep (no previous code)
        # First timestep uses unbiased quantization
        z_q_st_0, indices_0, _ = self.quantizer(z_e[0], prev_indices=None)

        # Scan over remaining timesteps
        if T > 1:
            _, (z_q_st_rest, indices_rest) = jax.lax.scan(
                quantize_step,
                indices_0,  # Initial carry: indices from first timestep
                (z_e[1:], egocentric_obs[1:], scan_inputs[2][1:]),  # Inputs for t=1 to T-1
            )
            # Combine first timestep with rest
            z_q_st = jnp.concatenate([z_q_st_0[None], z_q_st_rest], axis=0)
            indices = jnp.concatenate([indices_0[None], indices_rest], axis=0)
        else:
            z_q_st = z_q_st_0[None]
            indices = indices_0[None]

        # Decode all timesteps in parallel
        concatenated = jnp.concatenate([z_q_st, egocentric_obs], axis=-1)
        action, _ = self.decoder(concatenated)

        return action, z_e, indices


class VQPolicyNetwork:
    """VQ-VAE policy network with both standard and temporal apply methods.

    This class wraps a VQIntentionNetwork and provides two apply methods:
    - apply: Standard forward pass (parallel processing, optional prev_indices)
    - apply_temporal: Sequential processing with stickiness bias via scan

    Attributes:
        init: Initialization function.
        apply: Standard apply function.
        apply_temporal: Temporal apply function with sequential quantization.
        stickiness_bias: The stickiness bias value.
        num_codes: Number of codebook entries.
        latent_dim: Dimension of latent space.
    """

    def __init__(
        self,
        init,
        apply,
        apply_temporal,
        stickiness_bias: float,
        num_codes: int,
        latent_dim: int,
    ):
        """Initialize VQPolicyNetwork.

        Args:
            init: Initialization function.
            apply: Standard apply function.
            apply_temporal: Temporal apply function.
            stickiness_bias: Stickiness bias value.
            num_codes: Number of codebook entries.
            latent_dim: Dimension of latent space.
        """
        self.init = init
        self.apply = apply
        self.apply_temporal = apply_temporal
        self.stickiness_bias = stickiness_bias
        self.num_codes = num_codes
        self.latent_dim = latent_dim


def make_vq_intention_policy(
    action_param_size: int,
    latent_dim: int,
    obs_sizes: Mapping[str, int],
    encoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    decoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    num_codes: int = 512,
    commitment_cost: float = 0.25,
    codebook_init_scale: float = 1.0,
    stickiness_bias: float = 0.0,
) -> VQPolicyNetwork:
    """Create a VQ-VAE intention-based policy network.

    Constructs an encoder-quantizer-decoder VQ-VAE policy where the encoder
    processes reference trajectory observations, the quantizer maps to discrete
    codebook entries, and the decoder generates action parameters conditioned
    on quantized intentions and proprioceptive state.

    Args:
        action_param_size: Output dimension (typically 2x action_size for
            Gaussian mean and variance).
        latent_dim: Dimension of the latent/codebook embedding space.
        obs_sizes: Dict mapping observation keys to their sizes, e.g.
            {"imitation_target": 3716, "proprioception": 226}.
        encoder_hidden_layer_sizes: Hidden layer sizes for encoder MLP.
        decoder_hidden_layer_sizes: Hidden layer sizes for decoder MLP.
        num_codes: Number of codebook entries.
        commitment_cost: Weight for commitment loss (beta).
        codebook_init_scale: Scale for codebook initialization.
        stickiness_bias: Bias for temporal code persistence. When > 0,
            the quantizer favors selecting the previous timestep's code.

    Returns:
        VQPolicyNetwork with init, apply, and apply_temporal methods.
        The apply function returns (action_params, z_e, indices).
        The apply_temporal function processes sequences with stickiness bias.
    """
    policy_module = VQIntentionNetwork(
        encoder_layers=list(encoder_hidden_layer_sizes),
        decoder_layers=list(decoder_hidden_layer_sizes) + [action_param_size],
        latent_dim=latent_dim,
        num_codes=num_codes,
        commitment_cost=commitment_cost,
        codebook_init_scale=codebook_init_scale,
        stickiness_bias=stickiness_bias,
    )

    def apply(
        processor_params: DictRunningStatisticsState,
        policy_params,
        obs: Mapping[str, jnp.ndarray],
        key,
        deterministic: bool = False,
        get_activation: bool = False,
        prev_indices: jnp.ndarray | None = None,
    ):
        """Apply VQ policy with observation normalization.

        Args:
            processor_params: Dict normalizer with per-key statistics.
            policy_params: Network weights.
            obs: Dict observation with "imitation_target" and "proprioception".
            key: JAX random key (unused, for API compatibility).
            deterministic: Unused, VQ is always deterministic.
            get_activation: If True, return intermediate activations.
            prev_indices: Optional previous timestep indices for stickiness bias.

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
            prev_indices=prev_indices,
        )

    def apply_temporal(
        processor_params: DictRunningStatisticsState,
        policy_params,
        obs: Mapping[str, jnp.ndarray],
        episode_mask: jnp.ndarray | None = None,
    ):
        """Apply VQ policy with temporal stickiness bias.

        Processes sequences with sequential quantization where each timestep's
        code selection is biased toward the previous code. Uses jax.lax.scan
        for efficient processing.

        Args:
            processor_params: Dict normalizer with per-key statistics.
            policy_params: Network weights.
            obs: Dict observation with shape [T, B, ...] for each key.
            episode_mask: Optional mask for episode boundaries, shape [T, B].
                Value of 0 indicates episode start (resets prev_indices).

        Returns:
            action_params: Shape [T, B, action_size*2].
            z_e: Shape [T, B, latent_dim].
            indices: Shape [T, B].
        """
        obs = normalize_dict_obs(obs, processor_params)
        return policy_module.apply(
            policy_params,
            obs=obs,
            episode_mask=episode_mask,
            method=policy_module.forward_temporal,
        )

    # Create dummy dict observation for initialization
    dummy_obs = {
        "imitation_target": jnp.zeros((1, obs_sizes["imitation_target"])),
        "proprioception": jnp.zeros((1, obs_sizes["proprioception"])),
    }
    dummy_key = jax.random.PRNGKey(0)

    return VQPolicyNetwork(
        init=lambda key: policy_module.init(key, dummy_obs, dummy_key),
        apply=apply,
        apply_temporal=apply_temporal,
        stickiness_bias=stickiness_bias,
        num_codes=num_codes,
        latent_dim=latent_dim,
    )
