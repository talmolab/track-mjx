"""VQ-VAE intention network architectures for discrete latent imitation learning.

This module provides VQ-VAE (Vector Quantized Variational Autoencoder) based
encoder-decoder architectures that replace the continuous Gaussian latent space
of the standard VAE with a discrete codebook of learned embeddings.

Key components:
- VQEncoder: Maps reference trajectory observations to a continuous embedding
- ResidualVectorQuantizer: Multi-level residual VQ with per-level codebooks
- Decoder: Maps quantized latents + proprioceptive state to action parameters
- VQIntentionNetwork: Full VQ-VAE combining encoder, quantizer, and decoder

The discrete latent space enables learning interpretable "motor primitives" that
can be analyzed and potentially reused across tasks. Residual VQ (depth > 1)
naturally separates coarse and fine representations across levels.

References:
- van den Oord et al., "Neural Discrete Representation Learning", 2017
  https://arxiv.org/abs/1711.00937
- STAR: arXiv:2506.03863 (Residual VQ formulation)
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


class _CodebookLevel(nn.Module):
    """Single codebook level for use within ResidualVectorQuantizer.

    Attributes:
        num_codes: Number of codebook entries.
        latent_dim: Dimension of each codebook entry.
        codebook_init_scale: Scale for codebook initialization.
    """

    num_codes: int = 512
    latent_dim: int = 60
    codebook_init_scale: float = 1.0

    def setup(self):
        self.embeddings = self.param(
            "embeddings",
            nn.initializers.uniform(scale=self.codebook_init_scale),
            (self.num_codes, self.latent_dim),
        )


def _quantize_single_level(
    z_e: jnp.ndarray,
    codebook: jnp.ndarray,
    num_codes: int,
    latent_dim: int,
    stickiness_bias: float,
    prev_indices: jnp.ndarray | None,
    use_rotation: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Quantize a single input to the nearest codebook entry.

    Args:
        z_e: Input to quantize, shape [..., latent_dim].
        codebook: Codebook embeddings, shape [K, D].
        num_codes: Number of codebook entries K.
        latent_dim: Dimension D.
        stickiness_bias: Bias for previous code. 0.0 = no bias.
        prev_indices: Previous timestep indices, shape [...] or None.
        use_rotation: If True, use Householder rotation-augmented STE.

    Returns:
        z_q: Quantized vectors (no gradient), shape [..., D].
        indices: Selected codebook indices, shape [...].
        z_q_st: Quantized vectors with STE gradient, shape [..., D].
    """
    input_shape = z_e.shape
    flat_z_e = z_e.reshape(-1, latent_dim)  # [N, D]

    # Compute squared distances: ||z_e - e_k||^2
    z_e_sq = jnp.sum(flat_z_e**2, axis=-1, keepdims=True)  # [N, 1]
    codebook_sq = jnp.sum(codebook**2, axis=-1)  # [K]
    cross = jnp.matmul(flat_z_e, codebook.T)  # [N, K]
    distances = z_e_sq + codebook_sq - 2 * cross  # [N, K]

    # Apply stickiness bias
    if prev_indices is not None and stickiness_bias > 0:
        flat_prev_indices = prev_indices.reshape(-1)
        prev_one_hot = jax.nn.one_hot(flat_prev_indices, num_codes)
        distances = distances - stickiness_bias * prev_one_hot

    flat_indices = jnp.argmin(distances, axis=-1)  # [N]
    flat_z_q = codebook[flat_indices]  # [N, D]

    indices = flat_indices.reshape(input_shape[:-1])
    z_q = flat_z_q.reshape(input_shape)

    # Straight-through estimator
    if use_rotation:
        z_q_st = _rotation_quantize(z_e, jax.lax.stop_gradient(z_q))
    else:
        # Vanilla STE (Sterbenz pattern)
        z_q_st = z_e - jax.lax.stop_gradient(z_e) + jax.lax.stop_gradient(z_q)

    return z_q, indices, z_q_st


def _rotation_quantize(
    r: jnp.ndarray,
    z_q: jnp.ndarray,
) -> jnp.ndarray:
    """Rotation-augmented STE (STAR Eq. 7-9, Fifty et al. 2024).

    Forward: returns z_q (identical to vanilla STE).
    Backward: gradient is sg[scale * R] applied to dr, where R is a
    rank-1 corrected Householder rotation mapping r_hat -> q_hat.

    Args:
        r: Residual input (live, receives gradients), shape [..., D].
        z_q: Nearest codebook vector (stop-gradiented), shape [..., D].

    Returns:
        q_tilde: shape [..., D]. Forward = z_q, Backward = scale * R * grad.
    """
    eps = 1e-8

    # Compute scale and rotation from sg(r) so they're constants
    r_sg = jax.lax.stop_gradient(r)

    r_norm = jnp.linalg.norm(r_sg, axis=-1, keepdims=True) + eps
    z_q_norm = jnp.linalg.norm(z_q, axis=-1, keepdims=True) + eps

    r_hat = r_sg / r_norm  # [..., D]  (constant)
    q_hat = z_q / z_q_norm  # [..., D]  (constant, z_q already sg)

    scale = z_q_norm / r_norm  # [..., 1]  (constant)

    # Rank-1 corrected Householder rotation (Fifty et al. Eq. 9, STAR Eq. 9):
    # R = I - 2*m*m^T + 2*q_hat*r_hat^T  where m = normalize(r_hat + q_hat)
    #
    # Applied to live r:
    # R @ r = r - 2*(m^T r)*m + 2*(r_hat^T r)*q_hat
    m = r_hat + q_hat  # [..., D]
    m_norm = jnp.linalg.norm(m, axis=-1, keepdims=True) + eps
    m_hat = m / m_norm  # [..., D]

    m_dot_r = jnp.sum(m_hat * r, axis=-1, keepdims=True)  # [..., 1]
    rhat_dot_r = jnp.sum(r_hat * r, axis=-1, keepdims=True)  # [..., 1]

    R_r = r - 2 * m_dot_r * m_hat + 2 * rhat_dot_r * q_hat  # [..., D]

    scaled_R_r = scale * R_r  # [..., D], linear in live r

    # Degenerate case: r_hat + q_hat ~ 0 (anti-parallel), m_norm ~ 0
    is_degenerate = m_norm.squeeze(-1) < 1e-6

    # Vanilla STE fallback: forward = z_q, backward = identity
    vanilla_ste = r - jax.lax.stop_gradient(r) + jax.lax.stop_gradient(z_q)

    # Rotation STE: forward = z_q, backward = scale * R
    rotation_ste = scaled_R_r + jax.lax.stop_gradient(z_q - scaled_R_r)

    q_tilde = jnp.where(is_degenerate[..., None], vanilla_ste, rotation_ste)
    return q_tilde


class ResidualVectorQuantizer(nn.Module):
    """Residual vector quantization with learnable codebooks.

    When depth=1, behaves identically to a flat VectorQuantizer.
    When depth>1, each level quantizes the residual from the previous level,
    naturally separating coarse and fine representations.

    Based on STAR (arXiv:2506.03863) residual VQ formulation.

    Attributes:
        num_codes: Number of codebook entries per level.
        latent_dim: Dimension of each codebook entry.
        depth: Number of RVQ levels. 1 = vanilla VQ.
        commitment_cost: Weight for commitment loss (beta).
        codebook_init_scale: Scale for codebook initialization.
        stickiness_bias: Per-level stickiness bias. A single float applies
            to all levels. A tuple specifies per-level values.
    """

    num_codes: int = 512
    latent_dim: int = 60
    depth: int = 1
    commitment_cost: float = 0.25
    codebook_init_scale: float = 1.0
    stickiness_bias: float | tuple[float, ...] = 0.0
    use_rotation: bool = False
    coupled_residual_grad: bool = False

    def setup(self):
        """Initialize per-level codebooks."""
        # Normalize stickiness_bias to tuple of length depth
        if isinstance(self.stickiness_bias, (int, float)):
            self._bias_per_level = tuple(
                float(self.stickiness_bias) for _ in range(self.depth)
            )
        else:
            bias = tuple(float(b) for b in self.stickiness_bias)
            if len(bias) < self.depth:
                # Pad with 0.0 for missing levels
                bias = bias + (0.0,) * (self.depth - len(bias))
            self._bias_per_level = bias[: self.depth]

        # Create one codebook module per depth level
        self.codebooks = [
            _CodebookLevel(
                num_codes=self.num_codes,
                latent_dim=self.latent_dim,
                codebook_init_scale=self.codebook_init_scale,
                name=f"codebooks_{d}",
            )
            for d in range(self.depth)
        ]

    def __call__(
        self,
        z_e: jnp.ndarray,
        prev_indices: tuple[jnp.ndarray, ...] | jnp.ndarray | None = None,
    ) -> tuple[
        jnp.ndarray,
        tuple[jnp.ndarray, ...],
        tuple[jnp.ndarray, ...],
        tuple[jnp.ndarray, ...],
    ]:
        """Quantize encoder output through residual codebook levels.

        Args:
            z_e: Continuous encoder output, shape [..., latent_dim].
            prev_indices: Optional previous timestep indices.
                - None: no stickiness at any level.
                - Single array: backward-compat for depth=1, shape [...].
                - Tuple of D arrays: per-level previous indices.

        Returns:
            z_hat_st: Sum of STE-quantized vectors across levels. Decoder input.
                Shape [..., latent_dim].
            all_indices: Tuple of D index arrays, each shape [...].
            all_z_q: Tuple of D quantized vectors (no STE), each [..., latent_dim].
                For loss computation.
            all_residuals: Tuple of D+1 residual vectors.
                all_residuals[0] = z_e (input), all_residuals[d] = residual after
                level d. For diagnostics.
        """
        # Normalize prev_indices to tuple
        if prev_indices is None:
            prev_per_level = tuple(None for _ in range(self.depth))
        elif isinstance(prev_indices, tuple):
            prev_per_level = prev_indices
        else:
            # Single array (backward compat for depth=1)
            prev_per_level = (prev_indices,) + tuple(
                None for _ in range(self.depth - 1)
            )

        residual = z_e
        all_indices = []
        all_z_q = []
        all_residuals = [z_e]
        z_hat_st_parts = []

        for d in range(self.depth):
            codebook = self.codebooks[d].embeddings
            bias = self._bias_per_level[d]
            prev_idx = prev_per_level[d] if d < len(prev_per_level) else None

            z_q_d, indices_d, z_q_st_d = _quantize_single_level(
                z_e=residual,
                codebook=codebook,
                num_codes=self.num_codes,
                latent_dim=self.latent_dim,
                stickiness_bias=bias,
                prev_indices=prev_idx,
                use_rotation=self.use_rotation,
            )

            all_indices.append(indices_d)
            all_z_q.append(z_q_d)
            z_hat_st_parts.append(z_q_st_d)

            # Compute residual for next level
            if self.coupled_residual_grad and self.use_rotation:
                # STAR-style: subtract q_tilde (no stop_gradient), coupling
                # depth gradients through the rotation transform.
                residual = residual - z_q_st_d
            else:
                # Standard: r_{d+1} = r_d - sg(z_q_d)
                residual = residual - jax.lax.stop_gradient(z_q_d)
            all_residuals.append(residual)

        # Decoder input = sum of STE-quantized parts
        z_hat_st = z_hat_st_parts[0]
        for part in z_hat_st_parts[1:]:
            z_hat_st = z_hat_st + part

        return z_hat_st, tuple(all_indices), tuple(all_z_q), tuple(all_residuals)


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
    """Full VQ-VAE model combining encoder, residual quantizer, and decoder.

    The network splits observations into reference trajectory and proprioceptive
    components. The encoder processes trajectory observations to produce continuous
    embeddings, which are quantized via the residual codebook levels, then
    concatenated with proprioceptive state and decoded into action parameters.

    Supports optional stickiness bias for temporal code persistence, with
    per-level configuration for multi-depth RVQ.

    Attributes:
        encoder_layers: Hidden layer sizes for the encoder MLP.
        decoder_layers: Layer sizes for decoder (including action output).
        latent_dim: Dimension of the latent/codebook embedding space.
        num_codes: Number of codebook entries per level.
        commitment_cost: Weight for commitment loss (beta).
        codebook_init_scale: Scale for codebook initialization.
        stickiness_bias: Per-level stickiness bias. Float or tuple of floats.
        rvq_depth: Number of RVQ levels. 1 = vanilla VQ.
    """

    encoder_layers: Sequence[int]
    decoder_layers: Sequence[int]
    latent_dim: int = 60
    num_codes: int = 512
    commitment_cost: float = 0.25
    codebook_init_scale: float = 1.0
    stickiness_bias: float | tuple[float, ...] = 0.0
    rvq_depth: int = 1
    use_rotation: bool = False
    coupled_residual_grad: bool = False
    proprio_noise_scale: float = 0.0

    def setup(self):
        """Initialize encoder, quantizer, and decoder submodules."""
        self.encoder = VQEncoder(
            layer_sizes=self.encoder_layers,
            latent_dim=self.latent_dim,
        )
        self.quantizer = ResidualVectorQuantizer(
            num_codes=self.num_codes,
            latent_dim=self.latent_dim,
            depth=self.rvq_depth,
            commitment_cost=self.commitment_cost,
            codebook_init_scale=self.codebook_init_scale,
            stickiness_bias=self.stickiness_bias,
            use_rotation=self.use_rotation,
            coupled_residual_grad=self.coupled_residual_grad,
        )
        self.decoder = Decoder(layer_sizes=self.decoder_layers)

    @property
    def _has_stickiness(self) -> bool:
        """Check if any level has nonzero stickiness bias."""
        bias = self.stickiness_bias
        if isinstance(bias, (int, float)):
            return bias > 0
        return any(b > 0 for b in bias)

    def __call__(
        self,
        obs: Mapping[str, jnp.ndarray],
        key: jax.Array,
        deterministic: bool = False,
        get_activation: bool = False,
        prev_indices: tuple[jnp.ndarray, ...] | jnp.ndarray | None = None,
    ):
        """Forward pass through VQ-VAE intention network.

        Args:
            obs: Dictionary observation with keys:
                - "imitation_target": Reference trajectory observations.
                - "proprioception": Proprioceptive state observations.
            key: JAX random key (unused, for API compatibility).
            deterministic: Unused, VQ is always deterministic.
            get_activation: If True, return intermediate activations.
            prev_indices: Optional indices from previous timestep for
                stickiness bias. Single array (depth=1 compat) or tuple.

        Returns:
            action_params: Shape [..., action_size*2].
            z_e: Continuous encoder output, shape [..., latent_dim].
            all_indices: Tuple of D index arrays, each shape [...].
            (optional) extras: Dict of activations if get_activation=True.
        """
        traj = obs["imitation_target"]
        egocentric_obs = obs["proprioception"]

        if self.proprio_noise_scale > 0.0 and not deterministic:
            # key may be batched [B, 2] during rollout or single [2] during loss
            noise_key = key[0] if key.ndim > 1 else key
            noise = (
                jax.random.normal(noise_key, egocentric_obs.shape)
                * self.proprio_noise_scale
            )
            egocentric_obs = egocentric_obs + noise

        if get_activation:
            z_e, encoder_activations = self.encoder(traj, get_activation=True)
            z_hat_st, all_indices, all_z_q, all_residuals = self.quantizer(
                z_e, prev_indices=prev_indices
            )
            concatenated = jnp.concatenate([z_hat_st, egocentric_obs], axis=-1)
            action, decoder_activations = self.decoder(
                concatenated, get_activation=True
            )
            return (
                action,
                z_e,
                all_indices,
                {
                    "encoder": encoder_activations,
                    "decoder": decoder_activations,
                    "egocentric_obs": egocentric_obs,
                    "traj_obs": traj,
                    "z_e": z_e,
                    "all_z_q": all_z_q,
                    "all_residuals": all_residuals,
                    "z_hat_st": z_hat_st,
                    "all_indices": all_indices,
                },
            )
        else:
            z_e = self.encoder(traj)
            z_hat_st, all_indices, _, _ = self.quantizer(z_e, prev_indices=prev_indices)
            concatenated = jnp.concatenate([z_hat_st, egocentric_obs], axis=-1)
            action, _ = self.decoder(concatenated)
            return action, z_e, all_indices

    def forward_temporal(
        self,
        obs: Mapping[str, jnp.ndarray],
        episode_mask: jnp.ndarray | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, tuple[jnp.ndarray, ...]]:
        """Forward pass with temporal stickiness bias using sequential processing.

        Processes a sequence of observations where each timestep's code selection
        is biased toward the previous timestep's code. Uses jax.lax.scan for
        efficient sequential processing.

        Args:
            obs: Dictionary observation with keys:
                - "imitation_target": Shape [T, B, traj_dim].
                - "proprioception": Shape [T, B, proprio_dim].
            episode_mask: Optional mask indicating episode boundaries.
                Shape [T, B]. Value of 0 indicates episode start.

        Returns:
            action_params: Shape [T, B, action_size*2].
            z_e: Shape [T, B, latent_dim].
            all_indices: Tuple of D arrays, each shape [T, B].
        """
        traj = obs["imitation_target"]  # [T, B, traj_dim]
        egocentric_obs = obs["proprioception"]  # [T, B, proprio_dim]

        if self.proprio_noise_scale > 0.0 and key is not None:
            noise = (
                jax.random.normal(key, egocentric_obs.shape) * self.proprio_noise_scale
            )
            egocentric_obs = egocentric_obs + noise

        # Encode all timesteps in parallel
        z_e = self.encoder(traj)  # [T, B, latent_dim]

        D = self.rvq_depth

        # If no bias, process in parallel (faster)
        if not self._has_stickiness:
            z_hat_st, all_indices, _, _ = self.quantizer(z_e, prev_indices=None)
            concatenated = jnp.concatenate([z_hat_st, egocentric_obs], axis=-1)
            action, _ = self.decoder(concatenated)
            return action, z_e, all_indices

        # Sequential quantization with stickiness bias using scan
        T, B = z_e.shape[0], z_e.shape[1]

        # Normalize bias to tuple for per-level handling
        if isinstance(self.stickiness_bias, (int, float)):
            bias_per_level = tuple(float(self.stickiness_bias) for _ in range(D))
        else:
            bias = tuple(float(b) for b in self.stickiness_bias)
            if len(bias) < D:
                bias = bias + (0.0,) * (D - len(bias))
            bias_per_level = bias[:D]

        def quantize_step(carry, inputs):
            """Process one timestep with per-level bias toward previous codes."""
            # carry: tuple of D prev_indices arrays, each [B]
            prev_indices_per_level = carry
            z_e_t, mask_t = inputs

            residual = z_e_t
            new_indices = []
            z_hat_st_parts = []

            for d in range(D):
                codebook = self.quantizer.codebooks[d].embeddings
                bias = bias_per_level[d]
                prev_idx = prev_indices_per_level[d]

                if episode_mask is not None and bias > 0:
                    use_prev = mask_t > 0  # [B]
                    effective_prev = jnp.where(
                        use_prev[:, None],
                        jax.nn.one_hot(prev_idx, self.num_codes),
                        jnp.zeros((B, self.num_codes)),
                    )
                    flat_z = residual.reshape(-1, self.latent_dim)
                    z_sq = jnp.sum(flat_z**2, axis=-1, keepdims=True)
                    cb_sq = jnp.sum(codebook**2, axis=-1)
                    cross = jnp.matmul(flat_z, codebook.T)
                    distances = z_sq + cb_sq - 2 * cross
                    distances = distances - bias * effective_prev
                    curr_idx = jnp.argmin(distances, axis=-1)
                    z_q_d = codebook[curr_idx]
                    if self.quantizer.use_rotation:
                        z_q_st_d = _rotation_quantize(
                            residual, jax.lax.stop_gradient(z_q_d)
                        )
                    else:
                        z_q_st_d = (
                            residual
                            - jax.lax.stop_gradient(residual)
                            + jax.lax.stop_gradient(z_q_d)
                        )
                else:
                    z_q_d, curr_idx, z_q_st_d = _quantize_single_level(
                        z_e=residual,
                        codebook=codebook,
                        num_codes=self.num_codes,
                        latent_dim=self.latent_dim,
                        stickiness_bias=bias,
                        prev_indices=prev_idx,
                        use_rotation=self.quantizer.use_rotation,
                    )

                new_indices.append(curr_idx)
                z_hat_st_parts.append(z_q_st_d)
                if self.quantizer.coupled_residual_grad and self.quantizer.use_rotation:
                    residual = residual - z_q_st_d
                else:
                    residual = residual - jax.lax.stop_gradient(z_q_d)

            z_hat_st_t = z_hat_st_parts[0]
            for part in z_hat_st_parts[1:]:
                z_hat_st_t = z_hat_st_t + part

            new_carry = tuple(new_indices)
            return new_carry, (z_hat_st_t, tuple(new_indices))

        # Prepare scan mask
        if episode_mask is not None:
            scan_mask = episode_mask
        else:
            scan_mask = jnp.ones((T, B))

        # First timestep: unbiased quantization
        z_hat_st_0, indices_0, _, _ = self.quantizer(z_e[0], prev_indices=None)

        if T > 1:
            init_carry = indices_0  # tuple of D arrays
            _, (z_hat_st_rest, indices_rest) = jax.lax.scan(
                quantize_step,
                init_carry,
                (z_e[1:], scan_mask[1:]),
            )
            # indices_rest is tuple of D arrays, each [T-1, B]
            # Combine with first timestep
            z_hat_st = jnp.concatenate([z_hat_st_0[None], z_hat_st_rest], axis=0)
            all_indices = tuple(
                jnp.concatenate([indices_0[d][None], indices_rest[d]], axis=0)
                for d in range(D)
            )
        else:
            z_hat_st = z_hat_st_0[None]
            all_indices = tuple(idx[None] for idx in indices_0)

        # Decode all timesteps in parallel
        concatenated = jnp.concatenate([z_hat_st, egocentric_obs], axis=-1)
        action, _ = self.decoder(concatenated)

        return action, z_e, all_indices


class VQPolicyNetwork:
    """VQ-VAE policy network with both standard and temporal apply methods.

    This class wraps a VQIntentionNetwork and provides two apply methods:
    - apply: Standard forward pass (parallel processing, optional prev_indices)
    - apply_temporal: Sequential processing with stickiness bias via scan

    Attributes:
        init: Initialization function.
        apply: Standard apply function.
        apply_temporal: Temporal apply function with sequential quantization.
        stickiness_bias: The stickiness bias value (float or tuple).
        num_codes: Number of codebook entries per level.
        latent_dim: Dimension of latent space.
        rvq_depth: Number of RVQ levels.
    """

    def __init__(
        self,
        init,
        apply,
        apply_temporal,
        stickiness_bias: float | tuple[float, ...],
        num_codes: int,
        latent_dim: int,
        rvq_depth: int = 1,
    ):
        self.init = init
        self.apply = apply
        self.apply_temporal = apply_temporal
        self.stickiness_bias = stickiness_bias
        self.num_codes = num_codes
        self.latent_dim = latent_dim
        self.rvq_depth = rvq_depth


def make_vq_intention_policy(
    action_param_size: int,
    latent_dim: int,
    obs_sizes: Mapping[str, int],
    encoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    decoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    num_codes: int = 512,
    commitment_cost: float = 0.25,
    codebook_init_scale: float = 1.0,
    stickiness_bias: float | tuple[float, ...] = 0.0,
    rvq_depth: int = 1,
    use_rotation: bool = False,
    coupled_residual_grad: bool = False,
    proprio_noise_scale: float = 0.0,
) -> VQPolicyNetwork:
    """Create a VQ-VAE intention-based policy network.

    Args:
        action_param_size: Output dimension (typically 2x action_size).
        latent_dim: Dimension of the latent/codebook embedding space.
        obs_sizes: Dict mapping observation keys to their sizes.
        encoder_hidden_layer_sizes: Hidden layer sizes for encoder MLP.
        decoder_hidden_layer_sizes: Hidden layer sizes for decoder MLP.
        num_codes: Number of codebook entries per level.
        commitment_cost: Weight for commitment loss (beta).
        codebook_init_scale: Scale for codebook initialization.
        stickiness_bias: Per-level stickiness bias. Float or tuple.
        rvq_depth: Number of RVQ levels. 1 = vanilla VQ.
        use_rotation: If True, use Householder rotation-augmented STE.
        coupled_residual_grad: If True and use_rotation, couple depth
            gradients through the rotation transform (STAR-style).

    Returns:
        VQPolicyNetwork with init, apply, and apply_temporal methods.
        apply returns (action_params, z_e, all_indices) where all_indices
        is a tuple of D arrays.
    """
    policy_module = VQIntentionNetwork(
        encoder_layers=list(encoder_hidden_layer_sizes),
        decoder_layers=list(decoder_hidden_layer_sizes) + [action_param_size],
        latent_dim=latent_dim,
        num_codes=num_codes,
        commitment_cost=commitment_cost,
        codebook_init_scale=codebook_init_scale,
        stickiness_bias=stickiness_bias,
        rvq_depth=rvq_depth,
        use_rotation=use_rotation,
        coupled_residual_grad=coupled_residual_grad,
        proprio_noise_scale=proprio_noise_scale,
    )

    def apply(
        processor_params: DictRunningStatisticsState,
        policy_params,
        obs: Mapping[str, jnp.ndarray],
        key,
        deterministic: bool = False,
        get_activation: bool = False,
        prev_indices: tuple[jnp.ndarray, ...] | jnp.ndarray | None = None,
    ):
        """Apply VQ policy with observation normalization.

        Returns:
            action_params: Action distribution parameters.
            z_e: Continuous encoder output (for loss computation).
            all_indices: Tuple of D codebook index arrays.
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
        proprio_noise_key: jax.Array | None = None,
    ):
        """Apply VQ policy with temporal stickiness bias.

        Returns:
            action_params: Shape [T, B, action_size*2].
            z_e: Shape [T, B, latent_dim].
            all_indices: Tuple of D arrays, each shape [T, B].
        """
        obs = normalize_dict_obs(obs, processor_params)
        return policy_module.apply(
            policy_params,
            obs=obs,
            episode_mask=episode_mask,
            key=proprio_noise_key,
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
        rvq_depth=rvq_depth,
    )
